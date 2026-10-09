"""op_parity lane TR1 (plans/2026-10-10-tr1-prereg.md, results/tr1_speed_prior.md): the training-side removal of the adapter's ego-only speed
prior (decision 218). The arms are options of pp_train.py (its TR1 block); this file holds what is not training:

  tok     (op-train env, one small GPU job) the lead-token MLP of arm P2L pre-trained with its own head (decision 204: a new input path is
          warm-started, never random): [3 lead probabilities, lead distance / speed / acceleration, fed speed] of the frozen base model
          (teacher outputs) -> the residual of the logged arc length on the base plan's arc length at 0.5 .. 4 s. Controls trained the same
          way: speed only (lead numbers zeroed) and no input. -> $DATA_DIR/runs/op_parity/tr1/tok/<name>.pt {lead, report}
  probe   (op-train env, one GPU job) DIAG1's open-loop probes on navtest for a list of checkpoints: plans as served, with the fed ax
          +- 2 m/s^2 (DIAG1's acceleration probe), with ax +- 2 m/s^2 and the history poses made consistent with it (the ego was slower /
          faster before: the state a closed loop feeds after its own plan accelerated), and the base model (P0) with its lead outputs
          -> $DATA_DIR/runs/op_parity/tr1/probe-<name>.npz
  wod-report  (jevdrive env, CPU) prereg amendment 1: WOD-recipe arms against WLG on WOD val -> results/tr1/<name>_{arms,contrasts}.{csv,md}
  report  (.venv, CPU) the probe table with diag1.py's definitions (closing-lead frames, "slows", standstill launch), the continuation
          slopes, and the registered stage gates against the base model and the reference arm -> results/tr1/probe_<name>.{csv,md}
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

import diag1 as D  # noqa: E402

DATA = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
WD = DATA / "runs/op_parity/tr1"
OUT = _R / "experiments/op_parity/results/tr1"
DAX = 2.0                                                   # m/s^2 injected (DIAG1's accp2 / accm2)
VARS = ("main", "axp", "axm", "jp", "jm")
GATE_ONSET, GATE_SLOPE, GATE_NAV = 0.9, 0.3, 0.5            # the registered stage-0 gates


# ---------------------------------------------------------------- tok
def cmd_tok(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import parity_adapter as PA
    import pp_train as T
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", f"tr1/tok-{a.name}", seed=a.seed, config=vars(a)) as run:
        S = T.Store(a.data, dev, need_side=False, frames="warp", host=True)
        tr, dv, sp = T.split_rows(dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d, is_wod=S.is_wod), a.split)
        for x in sp:
            run.use_split(x)
        X = T.teacher_lead(S, A.load("cinque", torch.float16).slices)
        W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        arc = lambda x, y: torch.cat([torch.hypot(x[:, :1], y[:, :1]), torch.hypot(x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1])], 1).cumsum(1)  # noqa: E731
        tx, ty, _ = T.rear(S.t_plan, S.cam_x, W)
        Y = (arc(S.fut[..., 0], S.fut[..., 1]) - arc(tx, ty)) / torch.as_tensor(T.SIG_X, dtype=torch.float32, device=dev)
        ok = S.has_fut.cpu().numpy()
        tr, dv = tr[ok[tr]], dv[ok[dv]]
        tr_t, dv_t = torch.as_tensor(tr, device=dev), torch.as_tensor(dv, device=dev)
        rep = {"n_train": len(tr), "n_dev": len(dv), "dev_huber_zero": float(F.huber_loss(Y[dv_t], torch.zeros_like(Y[dv_t])))}
        keep = {"lead": slice(0, 7), "speed": slice(6, 7), "none": slice(0, 0)}
        for kind, cols in keep.items():
            torch.manual_seed(a.seed)
            g = torch.Generator(device=dev).manual_seed(a.seed)
            m = torch.zeros(7, device=dev)
            m[cols] = 1
            lead = nn.Sequential(nn.Linear(PA.LEAD_DIM, 256), nn.GELU(), nn.Linear(256, 256)).to(dev)
            head = nn.Sequential(nn.GELU(), nn.Linear(256, 8)).to(dev)
            opt = torch.optim.AdamW(list(lead.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01)
            for step in range(a.steps):
                r = tr_t[torch.randint(len(tr_t), (a.batch,), device=dev, generator=g)]
                for gq in opt.param_groups:
                    gq["lr"] = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / a.steps))
                loss = F.huber_loss(head(lead(X[r] * m)), Y[r])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                p = head(lead(X[dv_t] * m))
                rep[f"dev_huber_{kind}"] = float(F.huber_loss(p, Y[dv_t]))
                rep[f"dev_r2_4s_{kind}"] = float(1 - ((p[:, -1] - Y[dv_t][:, -1]) ** 2).mean() / Y[dv_t][:, -1].var())
            if kind == "lead":
                st = {k: v.detach().cpu() for k, v in lead.state_dict().items()}
            run.info("%s: %s", kind, {k: round(v, 4) for k, v in rep.items() if k.endswith(kind)})
        (WD / "tok").mkdir(parents=True, exist_ok=True)
        torch.save({"lead": st, "report": rep, "cfg": vars(a)}, WD / "tok" / f"{a.name}.pt")
        (WD / "tok" / f"{a.name}.json").write_text(json.dumps(rep, indent=1))
        run.summary.update(rep)


# ---------------------------------------------------------------- probe
def cmd_probe(a):
    import torch
    from jevdrive import op_interp as I
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", f"tr1/probe-{a.name}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        S = T.Store(["lb_navtest"], dev, need_side=False, frames="warp")
        tb, n = S.tb, (min(S.n, a.limit) if a.limit else S.n)
        th = torch.from_numpy(D.T_HIST).to(S.ego)

        def ed(dax, joint):
            e = S.ego.clone()
            e[:, 6] += dax / 3.0
            if joint:                                                   # x(t) of a past with the speed v0 + dax * t (t < 0): the ego was slower before
                e[:, D.PX] += 0.5 * dax * th[None] ** 2 / 10.0
            return e
        V = {"main": S.ego, "axp": ed(DAX, False), "axm": ed(-DAX, False), "jp": ed(DAX, True), "jm": ed(-DAX, True)}

        def fwd(model, ego):
            sl = model.net.slices
            pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
            mu, lead, lp = np.zeros((n, 33, 15), np.float32), np.zeros((n, 144), np.float32), np.zeros((n, 3), np.float32)
            with torch.no_grad():
                for i in range(0, n, 128):
                    r = torch.arange(i, min(i + 128, n), device=dev)
                    o = model(S.front[r], ego[r], S.tc[r], None, None).float().cpu().numpy()
                    mu[i:i + len(r)], lead[i:i + len(r)], lp[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15), o[:, sl["lead"]], o[:, sl["lead_prob"]]
            return np.stack([I.to_rear(mu[i, :, 0:3], mu[i, :, 11], I.T_IDXS, tb["cam"][i, :2], D.T8, "lever") for i in range(n)]), lead, lp
        R = dict(tokens=tb["names"][:n], log=tb["log"][:n], speed=tb["speed"][:n], fut=tb["fut"][:n], cam=tb["cam"][:n], ego=tb["ego"][:n], tags=np.array(a.tags))
        R["pose_P0"], R["lead_P0"], R["lp_P0"] = fwd(T.load_pmodel("P0", dev), S.ego)
        for tag in a.tags:
            model = N._load_ckpt(T, resolve(f"{tag}@warp", check=True).ckpt, dev)
            for var, ego in V.items():
                R[f"pose_{tag}_{var}"], lead, lp = fwd(model, ego)
                if var == "main":
                    R[f"lead_{tag}"], R[f"lp_{tag}"] = lead, lp
            run.info("%s done (tr1 %s, arm %s)", tag, model.tr1, model.arm)
            del model
            torch.cuda.empty_cache()
        WD.mkdir(parents=True, exist_ok=True)
        np.savez(WD / f"probe-{a.name}.npz", **R)
        run.summary.update(n=int(n), tags=list(a.tags))


def probe_rows(z, tags, nb=2000):
    """One row per tag (and the base model) from a probe npz: diag1.py's quantities."""
    from jevdrive import stats
    v0, u = z["speed"].astype(np.float64), z["log"].astype(str)
    ax = z["ego"][:, 6].astype(np.float64) * 3.0
    arc = lambda k: D.arc_at(z[k][..., :2].astype(np.float64), 0.5)  # noqa: E731
    A0, la = arc("pose_P0"), D.arc_at(z["fut"][..., :2].astype(np.float64), 0.5)
    ld = z["lead_P0"][:, :72].reshape(-1, 3, 6, 4)
    lp, d, lv = D.sigm(z["lp_P0"][:, 0]), ld[:, 0, 0, 0] - (D.NAV_FRONT - z["cam"][:, 0]), ld[:, 0, 0, 2].astype(np.float64)
    c = v0 - lv
    cl = (lp > 0.5) & (v0 >= 2) & (c >= 1) & (d / np.maximum(c, 0.1) <= 8)          # diag1's registered closing-lead frames
    nbk, brk = cl & (ax >= -0.3), cl & (ax < -0.3)
    ss = v0 < 0.5
    stl, nol = ss & (lp > 0.5) & (d < 15) & (lv < 1), ss & (lp <= 0.5)
    m212, m25, m512 = (v0 >= 2) & (v0 < 12), (v0 >= 2) & (v0 < 5), (v0 >= 5) & (v0 < 12)
    slow = lambda A: D.q_acc(A, v0) <= -0.3  # noqa: E731
    sS = slow(A0)

    def row(name, A, Ap=None, Am=None, Jp=None, Jm=None):
        sA = slow(A)
        b = stats.bootstrap(sA[nbk].astype(float), groups=u[nbk], n_boot=nb)
        r = {"arm": name, "closing-lead, ego not braking: n": int(nbk.sum()), "starts to slow": b["mean"], "lo": b["lo"], "hi": b["hi"],
             "/ base model": b["mean"] / sS[nbk].mean(), "mean plan a (m/s^2)": float(D.q_acc(A, v0)[nbk].mean()),
             "closing-lead, ego braking: slows": float(sA[brk].mean()), "base slows, arm not (all closing)": float((sS & ~sA)[cl].mean()),
             "standstill, stopped lead: n": int(stl.sum()), "launches (2 s arc > 0.5 m)": float((A[stl, 3] > 0.5).mean()), "2 s arc (m)": float(A[stl, 3].mean()),
             "standstill, no lead: launches": float((A[nol, 3] > 0.5).mean()),
             "first-seg speed / v0 - 1 (2-5)": float(D.q_r0(A, v0)[m25].mean()), "(5-12)": float(D.q_r0(A, v0)[m512].mean()),
             "4 s arc / log (v0 >= 2)": float(A[v0 >= 2, 4].sum() / la[v0 >= 2, 4].sum())}
        if Ap is not None:
            f = lambda P, M: (P[:, 0] - M[:, 0]) / 0.5 / (2 * DAX)  # noqa: E731  m/s of first-segment speed per m/s^2
            r |= {"slope ax (m/s per m/s^2)": float(f(Ap, Am)[m212].mean()), "slope ax + history": float(f(Jp, Jm)[m212].mean()),
                  "4 s arc, ax +2 (2-5)": float(Ap[m25, 4].sum() / A[m25, 4].sum() - 1), "4 s arc, ax + history +2 (2-5)": float(Jp[m25, 4].sum() / A[m25, 4].sum() - 1),
                  "observed slope on fed ax": float(np.polyfit(ax[m212], ((A[:, 0] - A0[:, 0]) / 0.5)[m212], 1)[0])}
        return r
    rows = [row("base model (P0)", A0), row("log", la)]
    for t in tags:
        rows.append(row(t, *(arc(f"pose_{t}_{v}") for v in VARS)))
    return rows


def cmd_report(a):
    from jevdrive import stats
    z = np.load(WD / f"probe-{a.name}.npz")
    tags = [t for t in (a.tags or z["tags"].astype(str).tolist())]
    rows = probe_rows(z, tags)
    by = {r["arm"]: r for r in rows}
    ref, base = by[a.ref], by["base model (P0)"]
    nav = {}
    for t in tags:
        f = DATA / "runs/bench/navtest" / f"{t}@warp" / "summary.json"
        nav[t] = json.loads(f.read_text()).get("EPDMS", np.nan) if f.exists() else np.nan
    for r in rows[2:]:
        t = r["arm"]
        s = max(abs(r["slope ax (m/s per m/s^2)"]), abs(r["slope ax + history"]))
        s_ref = max(abs(ref["slope ax (m/s per m/s^2)"]), abs(ref["slope ax + history"]))
        r["navtest EPDMS"] = nav[t]
        r["navtest - ref"] = nav[t] - nav[a.ref]
        r["gate onset (>= 0.9 x base)"] = bool(r["starts to slow"] >= GATE_ONSET * base["starts to slow"])
        r["gate slope (<= 0.3 x ref)"] = bool(s <= GATE_SLOPE * s_ref)
        r["gate navtest (>= ref - 0.5)"] = bool(np.isfinite(nav[t]) and nav[t] >= nav[a.ref] - GATE_NAV)
        r["stage gate"] = "ref" if t == a.ref else ("pass" if (r["gate onset (>= 0.9 x base)"] and r["gate slope (<= 0.3 x ref)"] and r["gate navtest (>= ref - 0.5)"]) else "fail")
    OUT.mkdir(parents=True, exist_ok=True)
    stats.write_table(rows, OUT / f"probe_{a.name}")
    print((OUT / f"probe_{a.name}.md").read_text())


def cmd_wod_report(a):
    """Prereg amendment 1: WOD-recipe arms against WLG on WOD val. Recipe = the 2-seed trajectory mean (decision 180's submission form) and
    the mean of the seeds' per-frame scores; paired bootstrap over sequences (diag1.wod_ctx: B 4 000); strata; the held-out part f3 + f4
    of decision 171's sequence folds; ADE@3s on all frames."""
    from jevdrive import stats
    from wod_pref import folds_of
    C = D.wod_ctx()
    n = C.n
    grp = lambda g: (g.split("=")[0], g.split("=")[1].split("+"))  # noqa: E731
    G = dict([grp(a.ref)] + [grp(g) for g in a.arms])
    ref = grp(a.ref)[0]
    P = {k: [C.preds(t, True) for t in v] for k, v in G.items()}
    traj = {k: C.rfs(np.mean(v, 0)) for k, v in P.items()}                     # RFS of the seed-mean trajectory
    seed = {k: [C.rfs(p) for p in v] for k, v in P.items()}
    err = lambda p: np.linalg.norm(p - C.fut, axis=-1)[:, :12].mean(1)  # noqa: E731
    ade = {k: err(np.mean(v, 0)) for k, v in P.items()}
    fold, _ = folds_of(C.seq[:n])
    v0, lead = C.vfed[:n], C.lp[:n, 0] > 0.5
    st = {"all": np.ones(n, bool), "held-out part (folds f3 + f4)": fold >= 3, "other part (folds f0 - f2)": fold < 3, "standstill (v0 < 0.5)": v0 < 0.5,
          "moving": v0 >= 0.5, "lead (p > 0.5)": lead, "no lead": ~lead, "standstill, lead": (v0 < 0.5) & lead, "moving, lead": (v0 >= 0.5) & lead}
    rows, arms = [], []
    for k in G:
        arms.append({"arm": k, "RFS (2-seed trajectory mean)": C.cm(traj[k]), "RFS (mean of seed scores)": C.cm(np.mean(seed[k], 0)),
                     "seeds": " / ".join(f"{C.cm(x):.3f}" for x in seed[k]), "ADE@3s (m, trajectory mean, 1 437 frames)": float(ade[k].mean()),
                     "RFS standstill": C.cm(traj[k], st["standstill (v0 < 0.5)"]), "RFS moving": C.cm(traj[k], st["moving"])})
        if k == ref:
            continue
        for nm, m in st.items():
            p, lo, hi = C.ci(traj[k] - traj[ref], m)
            q, qlo, qhi = C.ci(np.mean(seed[k], 0) - np.mean(seed[ref], 0), m)
            r = {"contrast": f"{k} - {ref}", "stratum": nm, "n": int(m.sum()), "dRFS (trajectory mean)": p, "lo": lo, "hi": hi, "dRFS (seed scores)": q, "s lo": qlo, "s hi": qhi,
                 "per seed": " / ".join(f"{C.cm(x - y, m):+.3f}" for x, y in zip(seed[k], seed[ref]))}
            if nm == "all":
                dd = ade[k] - ade[ref]
                bb = (C.Ka @ dd) / C.Ka.sum(1)
                ho = C.cm(traj[k] - traj[ref], st["held-out part (folds f3 + f4)"])
                r |= {"d ADE@3s (m)": float(dd.mean()), "a lo": float(np.percentile(bb, 2.5)), "a hi": float(np.percentile(bb, 97.5)),
                      "label": "beats WLG" if (p >= 0.05 and lo > 0 and ho > 0) else "worse" if hi < 0 else "no measurable gain"}
            rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    stats.write_table(arms, OUT / f"{a.name}_arms")
    stats.write_table(rows, OUT / f"{a.name}_contrasts")
    print((OUT / f"{a.name}_arms.md").read_text(), (OUT / f"{a.name}_contrasts.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("tok")
    p.add_argument("--name", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--split", default="navsim/op-parity-s234")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("probe")
    p.add_argument("--name", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("report")
    p.add_argument("--name", required=True)
    p.add_argument("--ref", required=True, help="the reference arm (P2H10 recipe at the same scale)")
    p.add_argument("--tags", nargs="*", default=[])
    p = sp.add_parser("wod-report")
    p.add_argument("--name", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    {"tok": cmd_tok, "probe": cmd_probe, "report": cmd_report, "wod-report": cmd_wod_report}[a.cmd](a)


if __name__ == "__main__":
    main()
