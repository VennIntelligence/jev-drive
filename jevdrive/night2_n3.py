"""Night queue 2, N3: leaderboard head x reaction (todos/2026-09-26-night-queue-2.md, N3 and the [B] entries, each
written before the numbers it affects).

  prep     Hydra seed s > 0: the CPU k-means vocabulary with seed s over E6's navtrain rows; the scored subset (E6's
           20 000 tokens) and its metric cache stay fixed, so only anchors.npz is new
  fit      elicit_e6's fit (seed s: vocabulary s, held-out logs s + 1), with P5 v1 BA and I3 riding along as extra
           evaluation matrices (navtrain statistics): Hydra and (a') cls_late selections, sigma(DAC) of the selection
  navridge NAVSIM `ridge_late` (navsim_heads.fit_ridge, deterministic) applied to P5 / I3
  p5cls    P5-trained `cls_late` (route folds, K = 1024, late fusion on `cls ego`'s OOF logits), I3 = 5-fold mean
  gates    g2 (G1's main arm, the navtrain probe; the WOD probe for description) on P5 (history-arc embedding
           recomputed from E5's detections) and I3 (G0's embedding)
  exam     compatibility check, P5 / I3 flips (p5_exam.exam unchanged), NAVSIM devkit job list

    python -m jevdrive.night2_n3 prep --seed 1
    python -m jevdrive.night2_n3 fit --seed 1 --model cinque
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import elicit_e6 as E6, navsim_heads as H, traj
from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
SEEDS = (0, 1, 2)
E6_PREP = "runs/elicitation/e6-prep/20260926-003758"
E6_FIT = "runs/elicitation/e6-fit/20260926-025946"
E1_NAV = "runs/elicitation/e1-navsim/20260926-020742"
G2_RUN = "runs/real-data-transfer/g2/20260926-090503"
MC_RUNS = {0: "runs/reactivity/mc-carla_p5v1_ba/20260925-233126", 1: "runs/reactivity/mc-carla_p5v1_ba-seed1/20260926-012944",
           2: "runs/reactivity/mc-carla_p5v1_ba-seed2/20260926-013348"}
I3_EXAM = "runs/elicitation/i3-exam/20260926-012841"
NAV_CMD = {0: 3, 1: 1, 2: 0, 3: 2}      # waymo.INTENTS (UNKNOWN, GO_STRAIGHT, GO_LEFT, GO_RIGHT) -> [left, straight, right, unknown]
HIST = (9, 11, 13, 15)                  # past steps at -1.5, -1.0, -0.5, 0 s (0.25 s grid, t0 last)


def out(*p) -> Path:
    d = data_dir() / "processed" / "night2" / "n3"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


# ---------------------------------------------------------------- inputs

def nav_ego(past: np.ndarray, intent: np.ndarray) -> np.ndarray:
    """(n, 32) NAVSIM ego input from a WOD-shaped past ([B] 09:58 (2)): poses at -1.5 ... 0 s with the heading of the
    step's velocity (carried from the next later step below 0.5 m/s; 0 at t0), velocity and acceleration in each step's
    body frame, the command one-hot."""
    P = past[:, HIST].astype(np.float64)
    v, a = P[..., 2:4], P[..., 4:6] / 0.25
    sp, hv = np.linalg.norm(v, axis=-1), np.arctan2(v[..., 1], v[..., 0])
    h = np.zeros(sp.shape)
    for k in (2, 1, 0):
        h[:, k] = np.where(sp[:, k] >= 0.5, hv[:, k], h[:, k + 1])
    c, s = np.cos(h), np.sin(h)
    body = lambda u: np.stack([c * u[..., 0] + s * u[..., 1], -s * u[..., 0] + c * u[..., 1]], -1)  # noqa: E731
    pose = np.concatenate([P[..., :2], h[..., None]], -1)
    cmd = np.eye(4)[[NAV_CMD[int(i)] for i in intent]]
    return np.concatenate([pose.reshape(len(P), -1), body(v).reshape(len(P), -1), body(a).reshape(len(P), -1), cmd],
                          1).astype(np.float32)


def p5_data() -> dict:
    """P5 v1 BA: index, past / future, pair tables, op `temporal` (op_streams_vis, M-C's), obs rows."""
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        op = p5_openpilot.load(t, MODELS, sub="op_streams_vis")
    return {"t": t, "past": past, "fut": fut, "obs": obs, "null": null, "pairs": pairs, "op": op,
            "rows": np.flatnonzero(t.role.to_numpy() == "obs")}


def i3_data() -> dict:
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot
    with I.p5_set(I.I3):
        t3a, past3a, _, obs, null, pairs = E.load()
        keep = t3a.frame_name.isin(set(I.needed())).to_numpy()
        t, past = t3a[keep].reset_index(drop=True), past3a[keep]
        op = p5_openpilot.load(t, MODELS, sub="op_streams")
    return {"t": t, "past": past, "obs": obs, "null": null, "pairs": pairs, "op": op}


def extra_sets(model: str, p5: dict | None = None, i3: dict | None = None) -> dict:
    """{name: (ego (n, 32), temporal (n, 512), frame names)} for the zero-shot evaluation of NAVSIM heads."""
    p5, i3 = p5 or p5_data(), i3 or i3_data()
    r = p5["rows"]
    return {"p5": (nav_ego(p5["past"][r], p5["t"].intent.to_numpy()[r]), p5["op"][f"op-{model} temporal"][r],
                   p5["t"].frame_name.to_numpy()[r]),
            "i3": (nav_ego(i3["past"], i3["t"].intent.to_numpy()), i3["op"][f"op-{model} temporal"], i3["t"].frame_name.to_numpy())}


# ---------------------------------------------------------------- Hydra seeds

def prep(rl, seed: int):
    """E6.prep's vocabulary step with k-means seed `seed` (CPU, deterministic); tokens and cache are E6's."""
    tr = E6._navtrain(False)
    fut = tr["fut"]
    F = torch.as_tensor(fut[..., :2].reshape(len(fut), -1))
    A = traj.kmeans(F, H.K, seed=seed)
    ids = traj.nearest(F, A, 1)[0][:, 0]
    s_, c_ = np.zeros((H.K, 8)), np.zeros((H.K, 8))
    np.add.at(s_, ids, np.sin(fut[..., 2]))
    np.add.at(c_, ids, np.cos(fut[..., 2]))
    anchors = np.concatenate([A.reshape(H.K, 8, 2).numpy(), np.arctan2(s_, c_)[..., None]], -1).astype(np.float32)
    np.savez(rl.dir / "anchors.npz", anchors=anchors, ids=ids)
    src = data_dir() / E6_PREP
    (rl.dir / "tokens.txt").write_text((src / "tokens.txt").read_text())
    (rl.dir / "seed.json").write_text(json.dumps({"kmeans_seed": seed, "holdout_seed": seed + 1, "subset": str(src)}))
    rl.log.info(f"seed {seed}: vocabulary K={H.K}, oracle (x, y) ADE "
                f"{np.linalg.norm(A.numpy()[ids].reshape(-1, 8, 2) - fut[..., :2], axis=-1).mean():.3f} m -> {rl.dir}")


def prep_dir(seed: int) -> Path:
    if seed == 0:
        return data_dir() / E6_PREP
    return sorted((data_dir() / "runs/night2" / f"n3-prep-s{seed}").glob("*/score.done"))[-1].parent


def fit(rl, model: str, seed: int):
    """elicit_e6.fit with the held-out split seed s + 1 and P5 / I3 as extra evaluation matrices."""
    P = prep_dir(seed)
    g3 = json.loads((data_dir() / E6.G3 / "stats.json").read_text())
    lams = (g3["cls ego K1024"]["lam"], g3[f"cls_late {model} temporal"]["lam"])
    tr = E6._navtrain(True)
    ev = {"navtest": H.load("navtest", True)}
    ex = extra_sets(model)
    an = np.load(P / "anchors.npz")
    anchors, ids = an["anchors"], an["ids"]
    toks, sub, pdms = [], [], []
    for p in sorted((P / "score").glob("chunk_*.npz")):
        z = np.load(p)
        toks.append(z["tokens"]), sub.append(z["sub"]), pdms.append(z["pdms"])
    toks, sub, pdms = np.concatenate(toks), np.concatenate(sub), np.concatenate(pdms)
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    rows = np.array([pos[t] for t in toks])
    rl.log.info(f"{model} seed {seed}: {len(rows)} scored tokens from {P}; vocabulary oracle PDMS {pdms.max(1).mean():.4f}")
    sets = ["navtest", *ex]
    Xe, *Xe_ev = H._std(tr["ego"], ev["navtest"]["ego"], *(v[0] for v in ex.values()))
    Xf, *Xf_ev = H._std(tr[model], ev["navtest"][model], *(v[1] for v in ex.values()))
    Xe_ev, Xf_ev = dict(zip(sets, Xe_ev)), dict(zip(sets, Xf_ev))
    X = torch.cat([Xe[rows], Xf[rows]], 1)
    Xev = {s: torch.cat([Xe_ev[s], Xf_ev[s]], 1) for s in sets}
    T = {m: torch.as_tensor(sub[:, :, i], device=H.DEV) for i, m in enumerate(E6.SUBS)}
    hold = H._group_folds(tr["log"][rows], 5, seed=seed + 1) == 0
    fit_r, hold_r = np.flatnonzero(~hold), np.flatnonzero(hold)
    lam_sel, heads_hold, heads_all = {}, {}, {}
    for m in E6.SUBS:
        sc = []
        for lam in E6.LAMS:
            W, b = E6._bce_fit(X, T[m], fit_r, lam)
            sc.append(E6._bce(X, T[m], hold_r, W, b))
        k = int(np.argmin(sc))
        lam_sel[m] = E6.LAMS[k]
        W, b = E6._bce_fit(X, T[m], fit_r, E6.LAMS[k])
        heads_hold[m] = X[hold_r] @ W + b
        W, b = E6._bce_fit(X, T[m], np.arange(len(rows)), E6.LAMS[k])
        heads_all[m] = {s: Xev[s] @ W + b for s in sets}
        rl.event("n3_head", model=model, seed=seed, sub=m, lam=E6.LAMS[k], bce_hold=sc, edge=k in (0, len(E6.LAMS) - 1))
    hold_logs = set(tr["log"][rows[hold_r]])
    fit_all = np.flatnonzero(~np.isin(tr["log"], list(hold_logs)))
    im_hold = E6._cls_logits(Xe, Xf, ids, tr["log"], fit_all, {"h": Xe[rows[hold_r]]}, {"h": Xf[rows[hold_r]]}, lams)["h"]
    P_hold = torch.as_tensor(pdms[hold_r], device=H.DEV)
    ar = torch.arange(len(hold_r), device=H.DEV)
    best = None
    for w in itertools.product(E6.W_IM, E6.W_MUL, E6.W_ONE, E6.W_ONE, E6.W_ONE):
        v = float(P_hold[ar, E6._select(im_hold, heads_hold, w)].mean())
        if best is None or v > best[1]:
            best = (w, v)
    v_im = float(P_hold[ar, im_hold.argmax(1)].mean())
    rl.log.info(f"held-out PDMS: oracle {pdms[hold_r].max(1).mean():.4f}, imitation {v_im:.4f}, weights {best[0]} -> {best[1]:.4f}")
    im_ev = E6._cls_logits(Xe, Xf, ids, tr["log"], np.arange(len(tr["tokens"])), Xe_ev, Xf_ev, lams)
    res = {}
    for s in sets:
        im_sel = im_ev[s].argmax(1).cpu().numpy()
        hs = {m: heads_all[m][s] for m in E6.SUBS}
        sel = E6._select(im_ev[s], hs, best[0]).cpu().numpy()
        dac = torch.sigmoid(hs["DAC"])
        res[s] = {"hydra": sel, "clsref": im_sel, "dac_sel": dac[torch.arange(len(sel)), torch.as_tensor(sel)].cpu().numpy(),
                  "dac_mean": dac.mean(1).cpu().numpy(), "keys": ev["navtest"]["tokens"] if s == "navtest" else ex[s][2]}
        if s == "navtest":
            for arm in ("hydra", "clsref"):
                np.savez(rl.dir / f"navtest_{arm}_{model}_s{seed}.npz", tokens=res[s]["keys"], poses=anchors[res[s][arm]])
    np.savez_compressed(rl.dir / f"sel_{model}_s{seed}.npz", anchors=anchors,
                        **{f"{s}_{k}": v for s, d in res.items() for k, v in d.items()})
    stats = {"model": model, "seed": seed, "prep": str(P), "lam": lam_sel, "weights": best[0], "pdms_hold": best[1],
             "pdms_hold_imitation": v_im, "pdms_hold_oracle": float(pdms[hold_r].max(1).mean()), "n_scored": len(rows)}
    if seed == 0:                          # reproduction of E6's stored navtest selection (the [B] 09:58 (1) check)
        ref = np.load(data_dir() / E6_FIT / f"navtest_hydra_{model}_temporal.npz") if model == "cinque" else None
        if ref is not None:
            assert (ref["tokens"] == res["navtest"]["keys"]).all()
            same = np.abs(ref["poses"] - anchors[res["navtest"]["hydra"]]).max((1, 2)) < 1e-4
            stats["e6_same_anchor"] = float(same.mean())
            rl.log.info(f"seed 0 vs E6's stored navtest selection: same anchor on {same.mean():.4f} of tokens")
    (rl.dir / f"stats_{model}_s{seed}.json").write_text(json.dumps(stats, indent=1, default=float))
    rl.event("n3_fit", **stats)
    rl.log.info(json.dumps(stats, default=float))


# ---------------------------------------------------------------- NAVSIM ridge_late, P5-trained cls_late, gates

def navridge(rl):
    """navsim_heads' `ridge ego` -> `ridge_late` (navtrain, deterministic) on navtest (checked against G3) and P5 / I3."""
    tr = E6._navtrain(True)
    te = H.load("navtest", True)
    p5, i3 = p5_data(), i3_data()
    folds = H._group_folds(tr["log"], H.FOLDS, seed=0)
    Y = torch.as_tensor(tr["fut"].reshape(len(tr["fut"]), -1), device=H.DEV)
    res = {}
    for m in MODELS:
        ex = extra_sets(m, p5, i3)
        names = ["navtest", *ex]
        Xe, *Xe_ev = H._std(tr["ego"], te["ego"], *(v[0] for v in ex.values()))
        base, oof_e, st_e = H.fit_ridge(Xe, Y, dict(zip(names, Xe_ev)), folds)
        Xf, *Xf_ev = H._std(tr[m], te[m], *(v[1] for v in ex.values()))
        r, _, st_l = H.fit_ridge(Xf, Y - oof_e, dict(zip(names, Xf_ev)), folds)
        pr = {s: (base[s] + r[s]).reshape(-1, 8, 3).cpu().numpy() for s in names}
        g3 = np.load(data_dir() / E6.G3 / f"navtest_ridge_late_{m}_temporal.npz")
        d = float(np.abs(g3["poses"] - pr["navtest"]).max())
        rl.log.info(f"{m}: ridge ego lam {st_e['lam']:g}, ridge_late lam {st_l['lam']:g}; navtest vs G3 max |diff| {d:.2e}")
        rl.event("n3_navridge", model=m, lam_ego=st_e["lam"], lam_late=st_l["lam"], navtest_max_diff=d)
        for s in ex:
            res[f"{s}_{m}"], res[f"{s}_keys"] = pr[s], ex[s][2]
        del Xf, Xf_ev
        torch.cuda.empty_cache()
    np.savez_compressed(out("navridge.npz"), **res)


def _pick_cls(X, tgt, fit_r, sel_r, Axy, fut, offset=None) -> float:
    from . import planner
    W, _ = planner.ce_solve(X, tgt, fit_r, planner.LAM_CLS, H.K, offset=offset)
    top = planner.cls_topk(W, X, sel_r, 1, offset=offset)
    err = [np.linalg.norm(Axy[top[:, i, 0]] - fut[sel_r], axis=-1).mean() for i in range(len(planner.LAM_CLS))]
    del W
    return float(planner.LAM_CLS[planner._pick(err, planner.LAM_CLS, "cls")])


def p5cls(rl, seed: int):
    """P5-trained `cls_late` ([B] 09:58 (4)): per route fold (seed s) a K = 1024 vocabulary (seed s) over the training
    futures, `cls ego` (lambda by top-1 ADE on a 20% base-route inner split, seed s), its 5-fold OOF logits as the
    offset of `cls_late` on the op `temporal`; obs rows get their fold's top-1 anchor, I3 the mean over the 5 folds."""
    from sklearn.model_selection import GroupShuffleSplit
    from . import p5_exam as E, planner
    p5, i3 = p5_data(), i3_data()
    t, n, n3 = p5["t"], len(p5["t"]), len(i3["t"])
    fold = E.folds(t, p5["pairs"], seed)
    role = np.r_[t.role.to_numpy(), np.full(n3, "i3")]
    seq = np.r_[t.base_id.to_numpy().astype(str), ("i3:" + i3["t"].base_id.astype(str)).to_numpy()]
    fut = np.r_[p5["fut"], np.zeros((n3, 20, 2), np.float32)]
    F = torch.as_tensor(fut.reshape(n + n3, -1), device=H.DEV)
    Ego = torch.as_tensor(np.r_[E.ego_input(t, p5["past"]), E.ego_input(i3["t"], i3["past"])], device=H.DEV)
    Xop = {m: torch.as_tensor(np.r_[p5["op"][f"op-{m} temporal"], i3["op"][f"op-{m} temporal"]], device=H.DEV) for m in MODELS}
    obs, I = p5["rows"], np.arange(n, n + n3)
    preds = {m: np.full((n, 20, 2), np.nan, np.float32) for m in MODELS}
    i3p = {m: np.zeros((n3, 20, 2), np.float64) for m in MODELS}
    for f in range(E.K_FOLDS):
        tr = np.flatnonzero((role == "train") & (np.r_[fold, np.full(n3, -2)] != f))
        ev = obs[fold[obs] == f]
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr, groups=seq[tr]))
        fit_r, sel_r = tr[a], tr[b]
        A = traj.kmeans(F[tr], H.K, seed=seed)
        ids = traj.nearest(F, A, 1)[0][:, 0]
        Axy = A.reshape(H.K, 20, 2).cpu().numpy()
        tgt = (ids[:, None], np.ones((len(ids), 1), np.float32))
        Xe = planner.standardize(Ego, tr)
        lam_e = _pick_cls(Xe, tgt, fit_r, sel_r, Axy, fut)
        We, _ = planner.ce_solve(Xe, tgt, tr, [lam_e], H.K)
        off = planner.linear_apply(We, Xe, np.arange(n + n3))[0]
        inner = H._group_folds(seq[tr], 5, seed=seed)
        for k in range(5):
            Wk, _ = planner.ce_solve(Xe, tgt, tr[inner != k], [lam_e], H.K)
            off[tr[inner == k]] = planner.linear_apply(Wk, Xe, tr[inner == k])[0]
        for m in MODELS:
            Xf = planner.standardize(Xop[m], tr)
            lam_l = _pick_cls(Xf, tgt, fit_r, sel_r, Axy, fut, off)
            Wl, _ = planner.ce_solve(Xf, tgt, tr, [lam_l], H.K, offset=off)
            top = planner.cls_topk(Wl, Xf, np.r_[ev, I], 1, offset=off)[:, 0, 0]
            preds[m][ev] = Axy[top[:len(ev)]]
            i3p[m] += Axy[top[len(ev):]] / E.K_FOLDS
            rl.event("n3_p5cls", seed=seed, fold=f, model=m, lam_ego=lam_e, lam_late=lam_l, n_train=len(tr), n_obs=len(ev))
            rl.log.info(f"seed {seed} fold {f} {m}: cls ego lam {lam_e:g}, cls_late lam {lam_l:g}")
            del Xf, Wl
        del Xe, We, off
        torch.cuda.empty_cache()
    np.savez_compressed(out(f"p5cls_s{seed}.npz"), p5_rows=obs, p5_keys=t.frame_name.to_numpy()[obs], i3_keys=i3["t"].frame_name.to_numpy(),
                        **{f"p5_{m}": preds[m][obs] for m in MODELS}, **{f"i3_{m}": i3p[m].astype(np.float32) for m in MODELS})


def p5_arc_embed() -> tuple[np.ndarray, np.ndarray]:
    """G0-format (history-arc corridor) embedding of every P5 v1 BA row from E5's detections: real_g0.geom_check (1)."""
    from multiprocessing import Pool
    from . import elicit_i3 as I, fusion_q4 as Q, p5_exam as E, real_g0 as G0
    from .common import n_cpus
    with I.p5_set(I.BA):
        t, past, *_ = E.load()
    sub = sorted(p for p in (data_dir() / "processed/elicit_e5/dets").iterdir() if p.is_dir())
    d = pd.concat([Q.load_dets(x, G0.SCORE) for x in sub], ignore_index=True)
    d = d[d.prompt.isin(G0.CLS3)]
    d = Q.lift_dets(d, d.key.str.split("|").str[1].to_numpy(), Q.p5_calib())
    d = d[d.lift_ok]
    Hh = pd.read_parquet(sorted(sub[0].glob("part-*.parquet"))[0], columns=["H"]).H.iloc[0]
    arr = np.c_[d.prompt.map({c: i for i, c in enumerate(G0.CLS3)}).to_numpy(), np.zeros((len(d), 2)),
                d.gx.to_numpy() + Q.REAR_AXLE_X, d.gy.to_numpy(), ((d.y1 - d.y0) / Hh).to_numpy(), d.score.to_numpy()]
    by = pd.Series(np.arange(len(d))).groupby(d.key.str.split("|").str[0].to_numpy()).indices
    p1 = past[:, -int(G0.ARC_T / 0.25) - 1, :2].astype(np.float64)
    jobs = [[(i, G0.arc_path(p1[i], Q.REAR_AXLE_X), arr[by[fn]] if fn in by else np.zeros((0, 7)),
              by.get(fn, np.zeros(0, np.int64))) for i, fn in enumerate(t.frame_name[c0:c0 + 512], c0)]
            for c0 in range(0, len(t), 512)]
    Ea = np.zeros((len(t), 64), np.float32)
    with Pool(min(32, n_cpus())) as pool:
        for part in pool.imap_unordered(G0._embed_frames, jobs):
            for i, e, *_ in part:
                Ea[i] = e
    return t.frame_name.to_numpy(), Ea


def gates(rl):
    """g2 (G1's main arm) on P5 obs rows and I3 frames: the navtrain probe (the package's gate) and the WOD probe."""
    import pickle
    from . import real_g0 as G0
    names, Ea = p5_arc_embed()
    np.savez_compressed(out("p5_arc_embed.npz"), frame_name=names, embed=Ea)
    i3 = i3_data()
    fr, X3 = G0.load_embed("i3")
    at = pd.Series(np.arange(len(fr)), index=fr.frame_id.to_numpy()).reindex(i3["t"].frame_name).astype(int).to_numpy()
    res = {"p5_keys": names, "i3_keys": i3["t"].frame_name.to_numpy()}
    for ds in ("nav", "wod"):
        with open(data_dir() / G2_RUN / f"g2_{ds}.pkl", "rb") as f:
            pr = pickle.load(f)
        res[f"p5_g2{ds}"], res[f"i3_g2{ds}"] = pr(Ea), pr(X3[at])
        rl.log.info(f"g2 {ds} probe: P5 mean {res[f'p5_g2{ds}'].mean():.3f} (> 0.5: {(res[f'p5_g2{ds}'] > 0.5).mean():.3f}), "
                    f"I3 mean {res[f'i3_g2{ds}'].mean():.3f}")
    np.savez_compressed(out("gates.npz"), **res)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "fit", "navridge", "p5cls", "gates"))
    ap.add_argument("--seed", default="0", help="comma list for fit / p5cls")
    ap.add_argument("--model", default="cinque,lebowski", help="comma list for fit")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seed.split(",")]
    rl = RunLog("night2", f"n3-{a.step}" + (f"-s{a.seed}" if a.step == "prep" else ""))
    if a.step == "prep":
        prep(rl, seeds[0])
    elif a.step == "fit":
        for sd in seeds:
            for m in a.model.split(","):
                fit(rl, m, sd)
                torch.cuda.empty_cache()
    elif a.step == "p5cls":
        for sd in seeds:
            p5cls(rl, sd)
    else:
        {"navridge": navridge, "gates": gates}[a.step](rl)
    rl.close()


if __name__ == "__main__":
    main()
