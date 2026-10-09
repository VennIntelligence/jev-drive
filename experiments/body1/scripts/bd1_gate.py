#!/usr/bin/env python3
"""BODY1 arm S0 gate reads (plans/2026-10-10-body1-prereg.md section 3, Amendment 1): G1 on body1-hold-logs, G2 on decision 220's decisions.

  std      (GPU) baseline (b): lateral std of the student's own plan at 2 s / 4 s (P2H10-F-s0 / -s1 plan head, decision 220's `plan_std2`) on
           the hold states -> s0/std/<cache dir>.npz
  predict  (GPU) --runs NAME=RUN_DIR ...: every checkpoint on the hold states (all 24 queries) and on the G2 decisions (the served plan)
           -> s0/pred/<NAME>.npz; each call appends one line per checkpoint to s0/reads.jsonl (every hold / G2 read is listed in the results doc)
  report   (CPU) --final A B --blind C [--lc 0.25=D 0.5=E] [--extra N ...]: tables -> s0/report/*.csv (copied to experiments/body1/results/s0/)
  figs     (CPU) ROC per class, learning curve, BEV panels of hold-log cases -> s0/report/figs/ (copied to experiments/body1/figs/s0/)
The G2 decisions are navtest logs: scored only, nothing is fitted or chosen on them. The stop threshold comes from the hold logs.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402
import sweep as SW  # noqa: E402

FAMS = ("log", "ot1", "yr1")
G2_SETS = ("P2H10-F-s0", "P2H10-F-s1", "ctrl")
QFAM = ("own0", "own1", "ship", "log", "lat", "head", "gain", "arc", "stop")
QF = np.r_[0, 1, 2, 3, np.repeat(np.arange(4, 9), 4)]
LINE = 0.80


def hold_rows(steps=False):
    R = C.load_rows(FAMS, range(B.NSH), lambda log, hold: hold, steps=steps)
    from jevdrive.data import splits
    ho, tr = splits.load(B.HOLD), splits.load(B.TRAIN)
    assert ho.mask(R["log"]).all() and not tr.mask(R["log"]).any()
    return R, (ho, tr)


# ---------------------------------------------------------------- baseline (b): the plan's own lateral std
def cmd_std(a):
    import torch
    import pp_train as T
    from jevdrive.run import Run
    T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
    dev = torch.device("cuda")
    with Run("body1", "s0/std", config=vars(a)) as run:
        models = [T.load_pmodel(t, dev) for t in ("P2H10-F-s0", "P2H10-F-s1")]
        out = C.sroot() / "std"
        out.mkdir(parents=True, exist_ok=True)
        for f in FAMS:
            for k in run.tqdm(range(B.NSH), desc=f):
                d = B.cdir(f, k)
                if (out / f"{d}.npz").exists() and not a.force:
                    continue
                hold = np.flatnonzero(np.load(B.root() / "rows" / f"{d}.npz")["hold"])
                S = T.Store([d], dev, need_side=False, frames="warp")
                pi = torch.as_tensor(S.pi, device=dev)
                sd = np.full((2, S.n, 2), np.nan, np.float32)
                with torch.no_grad():
                    for m, model in enumerate(models):
                        sl = model.net.slices["plan"]
                        for i in range(0, len(hold), 128):
                            r = torch.as_tensor(hold[i:i + 128], device=dev)
                            o = model(S.front[r], S.ego[r], S.tc[r], None, None).float()
                            raw = o[:, sl]
                            assert raw.shape[1] == 990 and torch.equal(raw[:, :495], o[:, pi])
                            y = torch.exp(raw[:, 495:].view(-1, 33, 15)[:, :, 1].clamp(max=11)).cpu().numpy()
                            sd[m, hold[i:i + 128]] = np.stack([[np.interp(t, T_IDXS, row) for t in (2.0, 4.0)] for row in y])
                np.savez(out / f"{d}.npz", std=sd)
                del S
                torch.cuda.empty_cache()


# ---------------------------------------------------------------- predictions
def cmd_predict(a):
    import torch
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("body1", "s0/predict", config=vars(a)) as run:
        R, sp = hold_rows()
        for s in sp:
            run.use_split(s)
        n = len(R["gi"])
        V = C.load_tokens(R["src"], n, dev)
        Qp = torch.as_tensor(R["q"]).to(dev)
        G2 = {}
        for g in G2_SETS:
            z = np.load(B.root() / "g2" / "dump" / f"{g}.npz")
            G2[g] = dict(scene=z["scene"], k=z["k"], V=torch.as_tensor(z["tokens"]).to(dev), valid=torch.as_tensor(z["valid"]).to(dev), ego=z["ego"].astype(np.float32),
                         q=torch.as_tensor(z["poses"].astype(np.float32)[:, None]).to(dev))
        run.info("hold: %d states, %d logs; G2: %s decisions", n, len(set(R["log"].tolist())), {g: len(v["k"]) for g, v in G2.items()})
        (C.sroot() / "pred").mkdir(parents=True, exist_ok=True)
        for spec in a.runs:
            name, d = spec.split("=")
            net, ck = C.load_ckpt(_pl.Path(d) / "ckpt.pt", dev)
            st = lambda e: torch.as_tensor((e - ck["emu"]) / ck["esd"]).to(dev)  # noqa: E731
            vis = net.vision
            out, step = C.predict(net, V if vis else None, st(R["ego"]), Qp)
            arr = dict(hold_out=out.astype(np.float32), gi=R["gi"], fam=R["fam"])
            if step is not None:
                arr["hold_step"] = step.astype(np.float16)
            for k in (1, 2, 4) if (vis and a.slots) else ():
                arr[f"hold_out_slots{k}"] = C.predict(net, V, st(R["ego"]), Qp, slots=k)[0][:, list(C.OWN)].astype(np.float32)
            for g, z in G2.items():
                o, _ = C.predict(net, z["V"] if vis else None, st(z["ego"]), z["q"], valid=z["valid"])
                arr[f"g2_{g}"] = o[:, 0].astype(np.float32)
            np.savez(C.sroot() / "pred" / f"{name}.npz", **arr)
            with open(C.sroot() / "reads.jsonl", "a") as f:
                f.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), name=name, ckpt=str(d), read="hold rows (G1) + G2 decisions", config=ck["config"], val=ck["val"])) + "\n")
            run.info("%s: predicted (%s, val own agent %.3f boundary %.3f)", name, ck["config"]["arch"], ck["val"]["own_a"], ck["val"]["own_b"])
            del net
        run.summary.update(runs=a.runs)


# ---------------------------------------------------------------- report
def _own(R, T):
    """Own-plan decision table of the hold states: one row per (state, own slot)."""
    import pandas as pd
    cls, gt45 = C.classes(T)
    Y = C.targets(R)
    n = len(R["gi"])
    speed = np.concatenate([np.load(B.cache_root() / d / "tab.npz")["speed"][m] for d, m, _ in R["src"]])
    std = np.concatenate([np.load(C.sroot() / "std" / f"{d}.npz")["std"][:, m] for d, m, _ in R["src"]], 1)          # (2 models, n, 2)
    d = SW.dense(R["q"][:, list(C.OWN)])
    arc = SW.arc(d)[..., -1]
    rep = lambda x: np.repeat(x, 2)  # noqa: E731
    D = pd.DataFrame(dict(i=rep(np.arange(n)), slot=np.tile(C.OWN, n), log=rep(R["log"]), fam=rep(np.array(R["fams"])[R["fam"]]), cls=rep(cls[R["gi"]]), gt45=rep(gt45[R["gi"]]),
                          a=Y["a"][:, :2].ravel(), a_ok=Y["a_ok"][:, :2].ravel(), b=Y["b"][:, :2].ravel(), b_ok=Y["b_ok"][:, :2].ravel(), rear_only=(R["a_rear"] & ~R["a_hit"])[:, :2].ravel(),
                          a_s=R["a_s"][:, :2].ravel(), a_t=R["a_t"][:, :2].ravel(), a_t0=R["a_t0"][:, :2].ravel(), speed=rep(speed), arc4=arc.ravel(),
                          std2=np.stack([std[0, :, 0], std[1, :, 0]], 1).ravel(), std4=np.stack([std[0, :, 1], std[1, :, 1]], 1).ravel(),
                          lat4=np.abs(R["q"][:, :2, -1, 1]).ravel(), dyaw4=np.abs(R["q"][:, :2, -1, 2]).ravel()))
    return D, Y


def _subsets(D):
    yield "pooled", np.ones(len(D), bool)
    for c in (1, 2, 3, 0):
        yield (f"class {c}" if c else "other"), (D.cls == c).to_numpy()
    yield "> 45 deg", D.gt45.to_numpy()
    for f in FAMS:
        yield f"states {f}", (D.fam == f).to_numpy()


def _g1(D, score, name):
    rows = []
    for kind, y, ok, col in (("agent", "a", "a_ok", 0), ("boundary", "b", "b_ok", 1)):
        s = score[col] if isinstance(score, tuple) else score
        for sub, m in _subsets(D):
            m = m & D[ok].to_numpy() & np.isfinite(s)
            r = C.auc_boot(D[y].to_numpy()[m], s[m], D.log.to_numpy()[m])
            rows.append(dict(model=name, kind=kind, subset=sub, **r, line=LINE if sub in ("pooled", "class 1", "class 2", "class 3") else np.nan,
                             gated=bool(sub in ("pooled", "class 1", "class 2", "class 3") and r["pos"] >= 30)))
    return rows


def cmd_report(a):
    import pandas as pd
    import g2 as G
    from jevdrive import stats
    from jevdrive.run import Run
    with Run("body1", "s0/report", config=vars(a)) as run:
        R, sp = hold_rows()
        for s in sp:
            run.use_split(s)
        T = dict(np.load(B.root() / "taxonomy" / "tax.npz"))
        D, Y = _own(R, T)
        out = C.sroot() / "report"
        out.mkdir(parents=True, exist_ok=True)
        P = lambda nm: np.load(C.sroot() / "pred" / f"{nm}.npz")  # noqa: E731
        lc = dict(x.split("=") for x in a.lc)
        names = list(dict.fromkeys(a.final + [a.blind] + list(lc.values()) + a.extra))
        pr = {nm: P(nm) for nm in names}
        for z in pr.values():
            assert (z["gi"] == R["gi"]).all() and (z["fam"] == R["fam"]).all()
        own = lambda z: (z["hold_out"][:, :2, 0].ravel().astype(float), z["hold_out"][:, :2, 1].ravel().astype(float))  # noqa: E731
        ens_out = np.mean([pr[nm]["hold_out"] for nm in a.final], 0)
        sc = {"S0 (mean of seeds)": (ens_out[:, :2, 0].ravel().astype(float), ens_out[:, :2, 1].ravel().astype(float))} | {nm: own(pr[nm]) for nm in names}
        # ---- G1
        g1 = [r for nm, s in sc.items() for r in _g1(D, s, nm)]
        for nm, s in (("plan lateral std 2 s", D.std2.to_numpy()), ("plan lateral std 4 s", D.std4.to_numpy()), ("ego speed", D.speed.to_numpy()), ("- ego speed", -D.speed.to_numpy()),
                      ("plan 4 s arc length", D.arc4.to_numpy()), ("plan |lateral| at 4 s", D.lat4.to_numpy()), ("plan |heading change| at 4 s", D.dyaw4.to_numpy())):
            g1 += _g1(D, s.astype(float), nm)
        g1 = pd.DataFrame(g1)
        g1.to_csv(out / "g1.csv", index=False)
        run.info("G1\n%s", g1[g1.subset.isin(["pooled", "class 1", "class 2", "class 3", "> 45 deg"])][["model", "kind", "subset", "auc", "lo", "hi", "n", "pos", "logs_pos"]].to_string(index=False))
        # ---- slot count (reported, no line): the final seeds with only the k newest slots
        rows = []
        for k in (1, 2, 4):
            if all(f"hold_out_slots{k}" in pr[nm].files for nm in a.final):
                e = np.mean([pr[nm][f"hold_out_slots{k}"] for nm in a.final], 0)
                rows += [dict(r, slots=k) for r in _g1(D, (e[..., 0].ravel().astype(float), e[..., 1].ravel().astype(float)), "S0 (mean of seeds)") if r["subset"] == "pooled"]
        pd.DataFrame(rows).to_csv(out / "g1_slots.csv", index=False)
        # ---- perturbed queries (no line)
        rows = []
        for qf, nm in enumerate(QFAM):
            qs = np.flatnonzero(QF == qf)
            for kind, y, ok, col in (("agent", "a", "a_ok", 0), ("boundary", "b", "b_ok", 1)):
                m = Y[ok][:, qs]
                lg = np.broadcast_to(R["log"][:, None], m.shape)
                rows.append(dict(query=nm, kind=kind, **C.auc_boot(Y[y][:, qs][m], ens_out[:, qs, col][m], lg[m], n_boot=2000)))
        pd.DataFrame(rows).to_csv(out / "g1_queries.csv", index=False)
        # ---- regressions on all hold rows (ensemble): clearance / margin MAE inside the clips
        clr, mg = np.clip(R["a_clr"], *C.A_CLIP), R["b_margin"]
        okc, okm = R["qok"] & (R["a_clr"] < 90), R["qok"] & (mg < 90)
        reg = dict(clr_mae=float(np.abs(ens_out[..., 5] - clr)[okc].mean()), clr_mae_own=float(np.abs(ens_out[:, :2, 5] - clr[:, :2])[okc[:, :2]].mean()),
                   margin_mae=float(np.abs(np.clip(ens_out[..., 6], *C.M_CLIP) - np.clip(mg, *C.M_CLIP))[okm].mean()),
                   margin_mae_own=float(np.abs(np.clip(ens_out[:, :2, 6], *C.M_CLIP) - np.clip(mg[:, :2], *C.M_CLIP))[okm[:, :2]].mean()),
                   margin_r_own=float(np.corrcoef(np.clip(ens_out[:, :2, 6], *C.M_CLIP)[okm[:, :2]], np.clip(mg[:, :2], *C.M_CLIP)[okm[:, :2]])[0, 1]))
        (out / "regress.json").write_text(json.dumps(reg, indent=1))
        # ---- G2
        table = G.load_table()
        pos, neg = G.split(table)
        key = lambda d: (d["set"], d["scene"], int(d["k"]))  # noqa: E731
        dump = {g: np.load(B.root() / "g2" / "dump" / f"{g}.npz") for g in G2_SETS}
        ids = [(g, str(s), int(k)) for g in G2_SETS for s, k in zip(dump[g]["scene"], dump[g]["k"])]
        g2s = {nm: np.concatenate([pr[nm][f"g2_{g}"] for g in G2_SETS]).astype(float) for nm in names}
        g2s = {"S0 (mean of seeds)": np.mean([g2s[nm] for nm in a.final], 0)} | g2s
        rows = []
        for nm, s in g2s.items():
            r = G.score(pd.DataFrame(dict(set=[i[0] for i in ids], scene=[i[1] for i in ids], k=[i[2] for i in ids], score=s)), table, strict=False)
            rows.append(dict(model=nm, **r, passes=bool(r["auc"] >= 0.80 and r["lo"] > 0.70)))
        g2 = pd.DataFrame(rows)
        g2.to_csv(out / "g2.csv", index=False)
        run.info("G2\n%s", g2.to_string(index=False))
        # ---- operating point for the stop arm: <= 2 % flags on clean own-plan hold decisions
        s = sc["S0 (mean of seeds)"][0]
        ok = D.a_ok.to_numpy()
        yy, lg = D.a.to_numpy(), D.log.to_numpy()
        clean = ok & ~yy & ~D.rear_only.to_numpy()
        thr = float(np.quantile(s[clean], 0.98))
        tp = ok & yy & (s >= thr)
        ps = ens_out[:, :2, 2].ravel() * C.S_SCALE
        pt = ens_out[:, :2, 3].ravel() * C.T_SCALE
        es, et = np.abs(ps - D.a_s.to_numpy())[tp], np.abs(pt - D.a_t.to_numpy())[tp]
        idx = {i: j for j, i in enumerate(ids)}
        gp, gn = np.array([idx[key(d)] for d in pos]), np.array([idx[key(d)] for d in neg])
        s2 = g2s["S0 (mean of seeds)"]
        nctrl = len(dump["ctrl"]["k"])
        stop = dict(thr_logit=thr, fp_clean_hold=float((s[clean] >= thr).mean()), n_clean=int(clean.sum()),
                    recall=stats.bootstrap((s[ok & yy] >= thr).astype(float), groups=lg[ok & yy]), n_pos=int((ok & yy).sum()), n_tp=int(tp.sum()),
                    recall_after_t0=float((s[ok & yy & ~D.a_t0.to_numpy()] >= thr).mean()),
                    recall_by_class={str(c): dict(recall=float((s[ok & yy & (D.cls.to_numpy() == c)] >= thr).mean()), n=int((ok & yy & (D.cls.to_numpy() == c)).sum())) for c in (1, 2, 3, 0)},
                    arc_err_median=float(np.median(es)), arc_err_p90=float(np.quantile(es, 0.9)), arc_bias=float(np.median((ps - D.a_s.to_numpy())[tp])),
                    time_err_median=float(np.median(et)), time_err_p90=float(np.quantile(et, 0.9)), true_arc_median=float(np.median(D.a_s.to_numpy()[tp])),
                    g2_flag_rate_clean=float((s2[gn] >= thr).mean()), g2_n_clean=int(len(gn)), g2_flag_rate_ctrl_all=float((s2[-nctrl:] >= thr).mean()), g2_n_ctrl=nctrl,
                    g2_recall=float((s2[gp] >= thr).mean()), g2_n_pos=int(len(gp)),
                    g2_rollouts_flagged=int(len({(d["set"], d["scene"]) for d, j in zip(pos, gp) if s2[j] >= thr})), g2_rollouts=int(len({(d["set"], d["scene"]) for d in pos})))
        (out / "stop.json").write_text(json.dumps(stop, indent=1))
        run.info("stop operating point: %s", json.dumps(stop))
        # ---- learning curve (hold G1 pooled, one seed per point; validation reads from the checkpoints)
        rows = []
        for fr, nm in sorted([(float(k), v) for k, v in lc.items()] + [(1.0, a.final[0])]):
            for r in g1[(g1.model == nm) & (g1.subset == "pooled")].to_dict("records"):
                rows.append(dict(frac=fr, **r))
        pd.DataFrame(rows).to_csv(out / "lc.csv", index=False)
        D.assign(s_agent=sc["S0 (mean of seeds)"][0], s_boundary=sc["S0 (mean of seeds)"][1], blind_agent=sc[a.blind][0], blind_boundary=sc[a.blind][1], pred_s=ps, pred_t=pt).to_parquet(out / "own.parquet")
        np.savez(out / "g2_scores.npz", ids=np.array(ids, dtype=object), **{f"s{j}": v for j, v in enumerate(g2s.values())}, names=np.array(list(g2s)))
        run.summary.update(g1_pooled={r["kind"]: [r["auc"], r["lo"], r["hi"]] for r in g1[(g1.model == "S0 (mean of seeds)") & (g1.subset == "pooled")].to_dict("records")},
                           g2=g2.iloc[0].to_dict(), stop=stop, reg=reg)


# ---------------------------------------------------------------- figures
def _roc(y, s):
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    return np.r_[0, np.cumsum(~y) / max((~y).sum(), 1)], np.r_[0, np.cumsum(y) / max(y.sum(), 1)]


def cmd_figs(a):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    _sys.path.insert(0, str(B.REPO / "research"))
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
    import plot_style as P
    import bd1_fig as BF
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    P.apply()
    rep = C.sroot() / "report"
    out = rep / "figs"
    out.mkdir(parents=True, exist_ok=True)
    D, g1, lc = pd.read_parquet(rep / "own.parquet"), pd.read_csv(rep / "g1.csv"), pd.read_csv(rep / "lc.csv")
    stop = json.loads((rep / "stop.json").read_text())
    # ---- ROC per class
    fig, axs = plt.subplots(1, 2, figsize=(P.DOUBLE_COLUMN_IN, 2.9))
    cols = {"pooled": "#222222", "class 1": P.PALETTE["blue"], "class 2": P.PALETTE["orange"], "class 3": P.PALETTE["green"], "> 45 deg": P.PALETTE["purple"]}
    for ax, kind, y, ok, sc, bl in ((axs[0], "agent", "a", "a_ok", "s_agent", "blind_agent"), (axs[1], "boundary", "b", "b_ok", "s_boundary", "blind_boundary")):
        for sub, col in cols.items():
            m = D[ok].to_numpy() & {"pooled": np.ones(len(D), bool), "> 45 deg": D.gt45.to_numpy()}.get(sub, (D.cls == int(sub[-1])).to_numpy() if sub.startswith("class") else None)
            r = g1[(g1.model == "S0 (mean of seeds)") & (g1.kind == kind) & (g1.subset == sub)].iloc[0]
            ax.plot(*_roc(D[y].to_numpy()[m], D[sc].to_numpy()[m]), color=col, lw=1.3 if sub == "pooled" else 0.9, label=f"{sub}: {r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}], {int(r.pos)} pos")
        m = D[ok].to_numpy()
        r = g1[(g1.model == a.blind) & (g1.kind == kind) & (g1.subset == "pooled")].iloc[0]
        ax.plot(*_roc(D[y].to_numpy()[m], D[bl].to_numpy()[m]), color=P.BASELINE, ls="--", label=f"no vision, pooled: {r.auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.5, zorder=0)
        if kind == "agent":
            ax.axvline(0.02, color=P.PALETTE["vermillion"], lw=0.6, ls=":")
        ax.set_xlabel("false positive rate"), ax.set_ylabel("true positive rate")
        ax.set_title(f"{kind} contact of the student's own plan, hold logs", loc="left")
        ax.legend(loc="lower right", fontsize=6.3)
    fig.tight_layout(pad=0.4)
    P.save(fig, out / "roc")
    plt.close(fig)
    # ---- learning curve
    fig, ax = plt.subplots(figsize=(P.SINGLE_COLUMN_IN, 2.3))
    for kind, col in (("agent", P.PALETTE["blue"]), ("boundary", P.PALETTE["green"])):
        x = lc[lc.kind == kind].sort_values("frac")
        ax.errorbar(100 * x.frac, x.auc, yerr=[x.auc - x.lo, x.hi - x.auc], color=col, marker="o", ms=3, capsize=2, label=kind)
        b = g1[(g1.model == a.blind) & (g1.kind == kind) & (g1.subset == "pooled")].iloc[0]
        ax.axhline(b.auc, color=col, ls="--", lw=0.6)
    ax.axhline(LINE, color=P.PALETTE["vermillion"], lw=0.6, ls=":")
    ax.set_xscale("log"), ax.set_xticks([25, 50, 100], ["25", "50", "100"])
    ax.set_xlabel("training logs used (%)"), ax.set_ylabel("G1 pooled AUC (hold logs)")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    P.save(fig, out / "lc")
    plt.close(fig)
    # ---- BEV panels: own0, at the stop threshold; 3 true positives (one per class), 3 misses, 2 false alarms, each from a different log
    R, _ = hold_rows()
    T, L = dict(np.load(B.root() / "taxonomy" / "tax.npz")), B.labels()
    thr = stop["thr_logit"]
    d0 = D[(D.slot == 0) & D.a_ok & ~D.a_t0].copy()
    picks, used = [], set()

    def take(df, kind, k):
        for r in df.itertuples():
            if r.log not in used and len([p for p in picks if p[0] == kind]) < k:
                used.add(r.log), picks.append((kind, r))
    for c in (1, 2, 3):
        take(d0[d0.a & (d0.s_agent >= thr) & (d0.cls == c)].sort_values("s_agent", ascending=False), f"hit{c}", 1)
    take(d0[d0.a & (d0.s_agent < thr)].sample(frac=1.0, random_state=0), "miss", 3)
    take(d0[~d0.a & ~d0.rear_only & (d0.s_agent >= thr)].sample(frac=1.0, random_state=0), "false alarm", 2)
    fig, axs = plt.subplots(2, 4, figsize=(P.DOUBLE_COLUMN_IN, 3.3))
    rows = []
    for ax, (kind, r) in zip(axs.ravel(), picks):
        i, gi = r.i, R["gi"][r.i]
        sdf = np.asarray(L["sdf"][gi]).astype(np.float32)
        xs, ys = SW.X0 + (np.arange(SW.NH) + 0.5) * SW.RES, SW.Y0 + (np.arange(SW.NW) + 0.5) * SW.RES
        ax.contourf(xs, ys, sdf.T, levels=[-1e3, 0], colors=["#D4D4D4"], zorder=0)
        box, valid, cls = (np.asarray(L[x][gi]) for x in ("box", "valid", "cls"))
        for k in np.flatnonzero(cls >= 0):
            v = np.flatnonzero(valid[:, k])
            if not len(v):
                continue
            col = P.PALETTE[BF.OBJ_COL[cls[k]]]
            ax.plot(box[v, k, 0], box[v, k, 1], color=col, lw=0.5, zorder=2)
            BF.poly(ax, *box[v[0], k, :3], box[v[0], k, 3] / 2, box[v[0], k, 4] / 2, fc=col, ec="none", alpha=0.75, zorder=3)
            if np.hypot(*(box[v[-1], k, :2] - box[v[0], k, :2])) > 1.0:
                BF.poly(ax, *box[v[-1], k, :3], box[v[-1], k, 3] / 2, box[v[-1], k, 4] / 2, fc="none", ec=col, lw=0.5, ls=(0, (2, 1)), zorder=3)
        d = SW.dense(R["q"][i, 0], R["off"][i])
        ex, ey = SW.ego_centre(d)
        s_arc = SW.arc(d)
        BF.poly(ax, ex[0], ey[0], d[0, 2], SW.HALF_L, SW.HALF_W, fc="none", ec="#222222", lw=0.45, zorder=4)
        ax.plot(d[:, 0], d[:, 1], color=P.PREDICTION, lw=0.9, zorder=6)
        if r.a:
            f = int(round(float(r.a_t) / SW.DT))
            BF.poly(ax, ex[f], ey[f], d[f, 2], SW.HALF_L, SW.HALF_W, fc="none", ec=P.PALETTE["vermillion"], lw=0.6, zorder=7)
            ax.plot(d[f, 0], d[f, 1], marker="x", ms=3.5, mew=0.8, color=P.PALETTE["vermillion"], zorder=8)
        if r.s_agent >= thr:
            f = int(np.argmin(np.abs(s_arc - np.clip(r.pred_s, 0, s_arc[-1]))))
            ax.plot(d[f, 0], d[f, 1], marker="o", ms=3.2, mfc="none", mew=0.8, color=P.PALETTE["green"], zorder=9)
        ax.set_xlim(-8, 44), ax.set_ylim(-19.5, 19.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False), ax.set_xticks([]), ax.set_yticks([])
        t = f"true {r.a_s:.1f} m / {r.a_t:.1f} s" if r.a else "no contact"
        ax.set_title(f"{kind} | class {r.cls or 'other'} | {r.fam} | {r.speed:.0f} m/s\np {1 / (1 + np.exp(-r.s_agent)):.2f} (thr {1 / (1 + np.exp(-thr)):.2f}), pred {r.pred_s:.1f} m; {t}", fontsize=5.4, pad=1.5, loc="left")
        rows.append(dict(kind=kind, token=str(T["token"][gi]), log=r.log, fam=r.fam, cls=int(r.cls), speed=float(r.speed), logit=float(r.s_agent), pred_s=float(r.pred_s), true_s=float(r.a_s) if r.a else np.nan,
                         true_t=float(r.a_t) if r.a else np.nan))
    for ax in axs.ravel()[len(picks):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.03, hspace=0.22)
    print(P.save(fig, out / "bev_cases"))
    pd.DataFrame(rows).to_csv(out / "bev_cases.csv", index=False)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("std")
    cli_args(p)
    p = sp.add_parser("predict")
    p.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR")
    p.add_argument("--slots", action="store_true", help="also predict the own plan with the 1 / 2 / 4 newest slots")
    cli_args(p)
    for nm in ("report", "figs"):
        p = sp.add_parser(nm)
        p.add_argument("--final", nargs="+", required=True)
        p.add_argument("--blind", required=True)
        p.add_argument("--lc", nargs="*", default=[], help="FRAC=NAME")
        p.add_argument("--extra", nargs="*", default=[])
        cli_args(p)
    a = ap.parse_args()
    {"std": cmd_std, "predict": cmd_predict, "report": cmd_report, "figs": cmd_figs}[a.cmd](a)
