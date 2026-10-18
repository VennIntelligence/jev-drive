#!/usr/bin/env python3
"""SELF-FWD offline gate (plans/2026-10-19-fwd-gate-prereg.md): a small forward model (served plan + ego state at a decision) -> the 4 s the car
actually executed in the rollout (controller + the next seven replans), and whether offroad / contact judged on its predicted sweep beats the
same judgement on the plan. Box, envs/op-train; nothing here changes a model or a rollout.

  build   CPU: drive.jsonl of every source run -> $DATA_DIR/runs/self_model/data/<src>.npz (names, logu, scene, k, plan, ego, exe, row)
          exe (n, 8, 3) = the driver anchors of decisions k+1..k+8 of the same session in decision k's anchor frame (0.5 s grid = the plan's)
  gate    one card: fit the MLP (+ ridge), gate (a) on T1 / T2, gate (b) on T1 (+ the train-pool validation logs) -> experiments/self_model/results/

    $DATA_DIR/envs/op-train/bin/python experiments/self_model/scripts/sm_gate.py build
    $DATA_DIR/envs/op-train/bin/python experiments/self_model/scripts/sm_gate.py gate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/self_model/data"
RES = REPO / "experiments/self_model/results"
ROWS, CACHE = D / "runs/onpol_nuplan/rows", D / "runs/op_parity/cache"
# src: (kind, pool name or read glob, role); train = on1_JT's four S sources, T1 / T2 = held (onpol-held scenes)
SRC = {"rb0": ("pool", "L18RB0", "train"), "rtb1": ("pool", "L18RTB1", "train"), "ys": ("pool", "L18YS", "train"), "y": ("pool", "L18Y", "train"),
       "rsgh": ("pool", "L18RSGallheld", "T1"), "jtnh": ("read", str(D / "runs/route_mem/read/JTN-L18-s0/c*/run"), "T2")}
H, STEP_US, KMAX = 8, 500_000, 9                  # 8 poses at 0.5 s; decisions 0..17, the full 4 s exists for k <= 9


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def into(a0, a):
    """(3,) pose a0, (m, 3) poses a (same frame) -> a in a0's frame."""
    c, s = np.cos(a0[2]), np.sin(a0[2])
    dx, dy = a[:, 0] - a0[0], a[:, 1] - a0[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(a[:, 2] - a0[2])], -1)


def logu_of(scene):
    return scene.rsplit("-", 1)[0][:26]


def sessions(run):
    """drive.jsonl -> {session: dict(scene, k (m,), now, anchor (m, 3), poses (m, 8, 3), ego (m, 20))} sorted by k."""
    S = {}
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "drive":
            continue
        S.setdefault(r["session"], []).append(r)
    out = {}
    for s, rr in S.items():
        rr = sorted(rr, key=lambda r: r["k"])
        ks = [r["k"] for r in rr]
        assert ks == list(range(len(ks))), f"{run} {s}: decisions {ks}"
        out[s] = dict(scene=rr[0]["scene"], k=np.array(ks), now=np.array([r["now"] for r in rr], np.int64),
                      anchor=np.array([r["anchor"] for r in rr], np.float64), poses=np.array([r["poses"] for r in rr], np.float64),
                      ego=np.array([r["ego"] for r in rr], np.float64))
    return out


def exe_of(a, k):
    """Session dict, decision k -> (8, 3) executed poses in decision k's anchor frame (NaN beyond the session's last decision)."""
    e = np.full((H, 3), np.nan)
    m = min(H, len(a["k"]) - 1 - k)
    if m > 0:
        assert np.all(np.diff(a["now"][k:k + m + 1]) == STEP_US), "decisions not on the 0.5 s grid"
        e[:m] = into(a["anchor"][k], a["anchor"][k + 1:k + 1 + m])
    return e


def work(job):
    """(run, rows) -> list of (row index or -1, name, scene, k, plan, ego, exe). rows None = every decision k <= KMAX of the run (read)."""
    run, rows = job
    S = sessions(run)
    out = []
    if rows is None:
        for s, a in S.items():
            for k in range(min(KMAX + 1, len(a["k"]))):
                out.append((-1, f"rd:{Path(run).parent.name}:{s}:{k}", a["scene"], k, a["poses"][k], a["ego"][k], exe_of(a, k)))
        return out
    by = {}
    for s, a in S.items():
        by.setdefault(a["scene"], []).append(a)
    for i, name, scene, k, anc in rows:
        hit = [a for a in by.get(scene, []) if k < len(a["k"]) and np.abs(a["anchor"][k] - anc).max() < 1e-6]
        assert len(hit) == 1, f"{run}: {name} matches {len(hit)} sessions"
        a = hit[0]
        out.append((i, name, scene, k, a["poses"][k], a["ego"][k], exe_of(a, k)))
    return out


def cmd_build(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("self_model", "build", config=vars(a)) as run:
        OUT.mkdir(parents=True, exist_ok=True)
        for src, (kind, ref, role) in SRC.items():
            if kind == "pool":
                M = np.load(ROWS / ref / "meta.npz")
                keep = np.flatnonzero(M["k"] <= KMAX)
                jobs = {}
                for i in keep:
                    jobs.setdefault(str(M["run"][i]), []).append((int(i), str(M["names"][i]), str(M["scene"][i]), int(M["k"][i]), M["anchor"][i]))
                jl = list(jobs.items())
            else:
                jl = [(r, None) for r in sorted(glob.glob(ref))]
            res = par.pmap(work, jl, run=run, workers=min(len(jl), a.workers), desc=src)
            res.raise_if_failed()
            R = [r for part in res.values for r in part]
            if kind == "pool":
                R.sort(key=lambda r: r[0])
            names = np.array([r[1] for r in R])
            scene = np.array([r[2] for r in R])
            exe = np.stack([r[6] for r in R]).astype(np.float32)
            plan = np.stack([r[4] for r in R]).astype(np.float32)
            d1 = np.hypot(*(exe[:, 0, :2] - plan[:, 0, :2]).T)
            np.savez(OUT / f"{src}.npz", names=names, scene=scene, logu=np.array([logu_of(s) for s in scene]), k=np.array([r[3] for r in R]),
                     plan=plan, ego=np.stack([r[5] for r in R]).astype(np.float32), exe=exe, row=np.array([r[0] for r in R]), role=role, kind=kind, ref=ref)
            full = np.isfinite(exe).all((1, 2))
            run.summary[src] = dict(n=len(R), full=int(full.sum()), runs=len(jl), d05_median=float(np.nanmedian(d1)), d05_p90=float(np.nanpercentile(d1, 90)))
            run.info(f"{src}: {len(R)} decisions ({full.sum()} with 4 s), {len(jl)} runs; exe(0.5 s) - plan(0.5 s) median {np.nanmedian(d1):.3f} m")


# ---------------------------------------------------------------- gate
def load(src):
    z = np.load(OUT / f"{src}.npz")
    ok = np.isfinite(z["exe"]).all((1, 2))
    return {k: z[k][ok] for k in ("names", "scene", "logu", "k", "plan", "ego", "exe", "row")}


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def feats(d):
    return np.concatenate([d["plan"].reshape(len(d["plan"]), -1), d["ego"]], 1).astype(np.float32)


def resid(d):
    r = d["exe"] - d["plan"]
    r[..., 2] = wrap(r[..., 2])
    return r.reshape(len(r), -1).astype(np.float32)


def end_err(pred, d):
    return np.hypot(*(pred[:, -1, :2] - d["exe"][:, -1, :2]).T)


def fit_mlp(tr, va, dev, run, seed=0, epochs=60):
    import torch
    torch.manual_seed(seed)
    X, Y = feats(tr), resid(tr)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    Xv = torch.as_tensor((feats(va) - mu) / sd, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, Y.shape[1])).to(dev)
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    bs, n = 512, len(Xt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((n + bs - 1) // bs))
    w = torch.tensor([1.0, 1.0, 2.0], device=dev).repeat(H)
    hub = torch.nn.functional.huber_loss
    best, state, hist = 1e9, None, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = (hub(net(Xt[b]), Yt[b], reduction="none", delta=0.5) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = va["plan"] + net(Xv).cpu().numpy().reshape(-1, H, 3)
        m = float(np.median(end_err(pv, va)))
        hist.append(m)
        run.scalar("val_end_median", m, ep)
        if m < best:
            best, state = m, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    run.info(f"MLP: best val 4 s end-point median {best:.3f} m at epoch {int(np.argmin(hist))}")

    def predict(d):
        with torch.no_grad():
            r = net(torch.as_tensor((feats(d) - mu) / sd, device=dev)).cpu().numpy().reshape(-1, H, 3)
        p = d["plan"] + r
        p[..., 2] = wrap(p[..., 2])
        return p
    return predict, dict(best_val=best, best_epoch=int(np.argmin(hist)), curve=hist), (net, mu, sd)


def fit_ridge(tr):
    from sklearn.linear_model import Ridge
    X = feats(tr)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = Ridge(alpha=1.0).fit((X - mu) / sd, resid(tr))

    def predict(d):
        p = d["plan"] + m.predict((feats(d) - mu) / sd).reshape(-1, H, 3)
        p[..., 2] = wrap(p[..., 2])
        return p
    return predict


def strata(d):
    yaw8 = np.degrees(np.abs(d["plan"][:, -1, 2]))
    v = np.hypot(d["ego"][:, 4], d["ego"][:, 5]) * 10.0
    turn = np.where(yaw8 >= 15, "turn", np.where(yaw8 < 5, "straight", "mild"))
    sp = np.select([v < 1, v < 5, v < 10], ["v<1", "v1-5", "v5-10"], "v>=10")
    return {"all": np.ones(len(v), bool), **{t: turn == t for t in ("straight", "mild", "turn")}, **{s: sp == s for s in ("v<1", "v1-5", "v5-10", "v>=10")}}


def lonlat(pred, d):
    """(n, 8) longitudinal / lateral error (m) of pred against exe, along the executed heading at that time."""
    e = pred[..., :2] - d["exe"][..., :2]
    c, s = np.cos(d["exe"][..., 2]), np.sin(d["exe"][..., 2])
    return c * e[..., 0] + s * e[..., 1], -s * e[..., 0] + c * e[..., 1]


def gate_a(d, P, label):
    """P: {arm: (n, 8, 3)} -> per-arm horizon errors, strata, the registered gate (a) on 'mlp' against 'plan'."""
    from jevdrive import stats
    out = dict(set=label, n=len(d["k"]), logs=int(len(np.unique(d["logu"]))), arms={})
    for arm, p in P.items():
        err = np.hypot(*(p[..., :2] - d["exe"][..., :2]).transpose(2, 0, 1))
        lo, la = lonlat(p, d)
        out["arms"][arm] = dict(median_by_t=[float(x) for x in np.median(err, 0)], mean_by_t=[float(x) for x in err.mean(0)],
                                p90_end=float(np.percentile(err[:, -1], 90)), lon_abs_median_end=float(np.median(np.abs(lo[:, -1]))),
                                lat_abs_median_end=float(np.median(np.abs(la[:, -1]))), lon_mean_end=float(lo[:, -1].mean()),
                                yaw_abs_median_end_deg=float(np.degrees(np.median(np.abs(wrap(p[:, -1, 2] - d["exe"][:, -1, 2]))))),
                                strata={k: dict(n=int(m.sum()), median_end=float(np.median(err[m, -1])) if m.any() else None) for k, m in strata(d).items()})
    em, ep = end_err(P["mlp"], d), end_err(P["plan"], d)
    r = stats.paired(ep, em, groups=d["logu"])
    out["gate"] = dict(median_mlp=float(np.median(em)), median_plan=float(np.median(ep)), ratio=float(np.median(em) / np.median(ep)),
                       diff_mean=r["mean"], diff_lo=r["lo"], diff_hi=r["hi"], units=r["units"],
                       passed=bool(np.median(em) < 0.20 and np.median(em) <= 0.5 * np.median(ep) and r["lo"] > 0))
    return out


def clearances(d, src, poses_by_arm, dev):
    """Min road-corner SDF and min counted agent distance over steps 1..40 (the S hinges' footprint code) for each arm's (n, 8, 3) poses."""
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    pool = "on1_" + SRC[src][1]
    names = d["names"]
    Rh = Hinge(CACHE / pool / "road.npz", names, dev, margin=0.0)
    Ah = AgentHinge(CACHE / pool / "agent.npz", names, dev, margin=0.0)
    act = np.load(ROWS / SRC[src][1] / "active.npz")
    pos = {t: i for i, t in enumerate(act["names"].tolist())}
    off0 = act["road_off_t0"][[pos[t] for t in names.tolist()]]
    out = {}
    for arm, p in poses_by_arm.items():
        road, ag = [], []
        with torch.no_grad():
            for i in range(0, len(names), 2048):
                rows = torch.arange(i, min(i + 2048, len(names)), device=dev)
                P = torch.as_tensor(p[i:i + 2048], dtype=torch.float32, device=dev)
                m = Rh.margins(P[..., 0], P[..., 1], P[..., 2], rows).reshape(len(rows), 41, 4)[:, 1:].amin((1, 2))
                dist, cnt, _ = Ah.distances(P[..., 0], P[..., 1], P[..., 2], rows)
                a = dist.masked_fill(~cnt, 1e3)[:, 1:].amin((1, 2))
                road.append(m.cpu().numpy())
                ag.append(a.cpu().numpy())
        out[arm] = (np.concatenate(road), np.concatenate(ag))
    ok_road = Rh.ok.cpu().numpy() & ~off0
    ok_ag = Ah.ok.cpu().numpy()
    return out, ok_road, ok_ag


def auc_boot(y, s1, s0, g, B=10_000, seed=0):
    """AUC(s1) - AUC(s0) for labels y with a cluster bootstrap over g (resample clusters, rows weighted by draw count)."""
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    a1, a0 = roc_auc_score(y, s1), roc_auc_score(y, s0)
    codes, uniq = pd.factorize(g)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(uniq), size=(B, len(uniq)))
    diffs = []
    for b in range(B):
        w = np.bincount(idx[b], minlength=len(uniq))[codes].astype(float)
        yw = y[w > 0]
        if yw.min() == yw.max():
            continue
        m = w > 0
        diffs.append(roc_auc_score(y[m], s1[m], sample_weight=w[m]) - roc_auc_score(y[m], s0[m], sample_weight=w[m]))
    diffs = np.array(diffs)
    return dict(auc_pred=float(a1), auc_plan=float(a0), gain=float(a1 - a0), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                n_boot=len(diffs), units=len(uniq))


def gate_b(d, src, P, dev, label, boot=True):
    from sklearn.metrics import roc_auc_score
    C, ok_road, ok_ag = clearances(d, src, {"exe": d["exe"], **P}, dev)
    ev_road = C["exe"][0] < 0
    ev_ag = C["exe"][1] < 0
    ok = ok_road & ok_ag
    out = dict(set=label, n=int(ok.sum()), logs=int(len(np.unique(d["logu"][ok]))))
    S = strata(d)
    for ev_name, y, score_of, m in (("any", (ev_road | ev_ag), lambda a: -np.minimum(C[a][0], C[a][1]), ok),
                                    ("offroad", ev_road, lambda a: -C[a][0], ok_road), ("contact", ev_ag, lambda a: -C[a][1], ok_ag)):
        y_m = y[m].astype(int)
        r = dict(n=int(m.sum()), events=int(y_m.sum()), rate=float(y_m.mean()),
                 plan_flags=float(((score_of("plan") > 0)[m]).mean()), mlp_flags=float(((score_of("mlp") > 0)[m]).mean()))
        if 0 < y_m.sum() < len(y_m):
            if boot:
                r.update(auc_boot(y_m, score_of("mlp")[m], score_of("plan")[m], d["logu"][m]))
            else:
                r.update(auc_pred=float(roc_auc_score(y_m, score_of("mlp")[m])), auc_plan=float(roc_auc_score(y_m, score_of("plan")[m])))
            r["auc_ridge"] = float(roc_auc_score(y_m, score_of("ridge")[m]))
            # event judged at 0 on each sweep: hit / false alarm rates
            for arm in ("plan", "mlp"):
                f = score_of(arm)[m] > 0
                r[f"{arm}_tpr"], r[f"{arm}_fpr"] = float(f[y_m == 1].mean()), float(f[y_m == 0].mean())
            r["strata"] = {}
            for k, sm in S.items():
                mm = sm[m]
                yy = y_m[mm]
                if mm.sum() and 0 < yy.sum() < len(yy):
                    r["strata"][k] = dict(n=int(mm.sum()), events=int(yy.sum()), auc_plan=float(roc_auc_score(yy, score_of("plan")[m][mm])),
                                          auc_pred=float(roc_auc_score(yy, score_of("mlp")[m][mm])))
        out[ev_name] = r
    out["gate"] = dict(passed=bool(out["any"].get("gain", -1) >= 0.03 and out["any"].get("lo", -1) > 0)) if boot else None
    return out


def cmd_gate(a):
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Run("self_model", "gate", seed=0, config=vars(a)) as run:
        sp_tr, sp_he = splits.load("alpasim_nuplan/onpol-train"), splits.load("alpasim_nuplan/onpol-held")
        run.use_split(sp_tr)
        run.use_split(sp_he)
        D_ = {s: load(s) for s in SRC}
        for s, (_, _, role) in SRC.items():
            sp = sp_tr if role == "train" else sp_he
            assert sp.mask(D_[s]["scene"]).all(), f"{s}: scenes outside {sp.id}"
        tr_all = cat([D_[s] for s in SRC if SRC[s][2] == "train"])
        isval = np.array([zlib.crc32(u.encode()) % 7 == 0 for u in tr_all["logu"]])
        tr = {k: v[~isval] for k, v in tr_all.items()}
        va = {k: v[isval] for k, v in tr_all.items()}
        run.info(f"train {len(tr['k'])} rows / {len(np.unique(tr['logu']))} logs, val {len(va['k'])} / {len(np.unique(va['logu']))} logs")
        mlp, info, _ = fit_mlp(tr, va, dev, run, epochs=a.epochs)
        ridge = fit_ridge(tr)
        res = dict(prereg="experiments/self_model/plans/2026-10-19-fwd-gate-prereg.md", train=dict(rows=len(tr["k"]), logs=int(len(np.unique(tr["logu"])))),
                   val=dict(rows=len(va["k"]), logs=int(len(np.unique(va["logu"])))), mlp=info, a={}, b={})
        sets = {"val": (va, None), "T1": (D_["rsgh"], "rsgh"), "T2": (D_["jtnh"], None)}
        for lab, (d, _) in sets.items():
            P = {"plan": d["plan"], "ridge": ridge(d), "mlp": mlp(d)}
            res["a"][lab] = gate_a(d, P, lab)
            run.info(f"(a) {lab}: mlp {res['a'][lab]['gate']['median_mlp']:.3f} m vs plan {res['a'][lab]['gate']['median_plan']:.3f} m, "
                     f"pass {res['a'][lab]['gate']['passed']}")
        d = D_["rsgh"]
        res["b"]["T1"] = gate_b(d, "rsgh", {"plan": d["plan"], "ridge": ridge(d), "mlp": mlp(d)}, dev, "T1")
        # val logs: per source pool (each has its own rasters); no bootstrap, a secondary read
        vb = {}
        for s in [s for s in SRC if SRC[s][2] == "train"]:
            dv = D_[s]
            m = np.array([zlib.crc32(u.encode()) % 7 == 0 for u in dv["logu"]])
            dv = {k: v[m] for k, v in dv.items()}
            vb[s] = gate_b(dv, s, {"plan": dv["plan"], "ridge": ridge(dv), "mlp": mlp(dv)}, dev, f"val-{s}", boot=False)
        res["b"]["val"] = vb
        g = res["b"]["T1"]["any"]
        run.info(f"(b) T1 any: AUC pred {g.get('auc_pred')} plan {g.get('auc_plan')} gain {g.get('gain')} [{g.get('lo')}, {g.get('hi')}]")
        res["verdict"] = dict(a=bool(res["a"]["T1"]["gate"]["passed"] and res["a"]["T2"]["gate"]["passed"]), b=bool(res["b"]["T1"]["gate"]["passed"]))
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "gate.json").write_text(json.dumps(res, indent=1))
        (run.path("gate.json")).write_text(json.dumps(res, indent=1))
        run.summary.update(verdict=res["verdict"], a_T1=res["a"]["T1"]["gate"], a_T2=res["a"]["T2"]["gate"],
                           b_T1={k: v for k, v in g.items() if k != "strata"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("gate")
    p.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    {"build": cmd_build, "gate": cmd_gate}[a.cmd](a)
