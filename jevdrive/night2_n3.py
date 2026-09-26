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
    from .real_g1 import Probe

    class _U(pickle.Unpickler):             # G1 pickled the probe from `python -m jevdrive.real_g1`, i.e. as __main__.Probe
        def find_class(self, mod, name):
            return Probe if name == "Probe" else super().find_class(mod, name)
    for ds in ("nav", "wod"):
        with open(data_dir() / G2_RUN / f"g2_{ds}.pkl", "rb") as f:
            pr = _U(f).load()
        res[f"p5_g2{ds}"], res[f"i3_g2{ds}"] = pr(Ea), pr(X3[at])
        rl.log.info(f"g2 {ds} probe: P5 mean {res[f'p5_g2{ds}'].mean():.3f} (> 0.5: {(res[f'p5_g2{ds}'] > 0.5).mean():.3f}), "
                    f"I3 mean {res[f'i3_g2{ds}'].mean():.3f}")
    np.savez_compressed(out("gates.npz"), **res)


# ---------------------------------------------------------------- compatibility check, exams, NAVSIM jobs

def _sel_files() -> dict:
    """{(model, seed): latest sel_<model>_s<seed>.npz} over the n3-fit runs."""
    out_ = {}
    for f in sorted((data_dir() / "runs/night2/n3-fit").glob("*/sel_*_s*.npz")):
        m, sd = f.stem.split("_")[1], int(f.stem.split("_s")[-1])
        out_[(m, sd)] = f
    return out_


def _null_frames(obs_null: pd.DataFrame) -> np.ndarray:
    return pd.unique(np.r_[obs_null.fn_plus.to_numpy(), obs_null.fn_null.to_numpy()])


def compat(rl) -> pd.DataFrame:
    """[B] 09:58 (3): top-10 overlap of Hydra's selected anchors (P5 / I3 null frames vs navtest) and the sigma(DAC)
    difference, per model x seed; seed 0 decides."""
    p5n = _null_frames(pd.read_parquet(data_dir() / "processed/carla_p5v1_ba/null.parquet"))
    i3n = _null_frames(pd.read_parquet(data_dir() / "processed/hugsim_pairs/null.parquet"))
    rows = []
    for (m, sd), f in sorted(_sel_files().items()):
        z = np.load(f, allow_pickle=True)
        top = lambda sel: set(pd.Series(sel).value_counts().index[:10])  # noqa: E731
        nav_top = top(z["navtest_hydra"])
        for s, fr in (("p5", p5n), ("i3", i3n)):
            keys = pd.Series(np.arange(len(z[f"{s}_keys"])), index=z[f"{s}_keys"].astype(str))
            r = keys.reindex(fr.astype(str)).dropna().astype(int).to_numpy()
            ov = len(top(z[f"{s}_hydra"][r]) & nav_top) / 10
            ov_cls = len(top(z[f"{s}_clsref"][r]) & top(z["navtest_clsref"])) / 10    # [B] 10:11, description only
            rows.append({"model": m, "seed": sd, "set": s, "null_frames": len(r), "top10_overlap": ov,
                         "cls_late_top10_overlap (post hoc)": ov_cls,
                         "distinct_anchors_null": int(len(np.unique(z[f"{s}_hydra"][r]))),
                         "distinct_anchors_navtest": int(len(np.unique(z["navtest_hydra"]))),
                         "dac_sel_null": float(z[f"{s}_dac_sel"][r].mean()), "dac_sel_navtest": float(z["navtest_dac_sel"].mean()),
                         "dac_sel_diff": float(z[f"{s}_dac_sel"][r].mean() - z["navtest_dac_sel"].mean()),
                         "dac_all_diff": float(z[f"{s}_dac_mean"][r].mean() - z["navtest_dac_mean"].mean()),
                         "comparable": ov >= 0.3})
    t = pd.DataFrame(rows)
    t.to_csv(rl.dir / "compat.csv", index=False)
    rl.log.info("compatibility\n%s", t.to_markdown(index=False, floatfmt=".3f"))
    return t


def _grid(p8: np.ndarray) -> np.ndarray:
    from .elicit_e1 import _grid20
    return _grid20(p8)


def _rows_of(keys_all: np.ndarray, keys: np.ndarray) -> np.ndarray:
    return pd.Series(np.arange(len(keys_all)), index=keys_all.astype(str)).reindex(keys.astype(str)).astype(int).to_numpy()


def exam(rl):
    """P5 v1 BA and I3 exams of every N3 examinee (p5_exam.exam unchanged; I3 without the TFv6 columns), the pedestrian /
    cut-in criteria (reactivity_mc.criteria) against each seed's P5 prior, and the NAVSIM devkit job list."""
    from . import p5_exam as E, reactivity_mc as MC
    sels = _sel_files()
    nr = np.load(out("navridge.npz"), allow_pickle=True)
    gt = np.load(out("gates.npz"), allow_pickle=True)
    p5, i3 = p5_data(), i3_data()
    t, n, rows = p5["t"], len(p5["t"]), p5["rows"]
    t3, n3 = i3["t"], len(i3["t"])
    g5 = {k: gt[f"p5_{k}"][_rows_of(gt["p5_keys"], t.frame_name.to_numpy()[rows])] for k in ("g2nav", "g2wod")}
    g3i = {k: gt[f"i3_{k}"][_rows_of(gt["i3_keys"], t3.frame_name.to_numpy())] for k in ("g2nav", "g2wod")}
    z3 = np.load(data_dir() / I3_EXAM / "preds_i3.npz", allow_pickle=True)
    assert (z3["frame_name"].astype(str) == t3.frame_name.to_numpy()).all()
    P5, P3 = {}, {}

    def put5(name, v):                     # v: (len(rows), 20, 2)
        a = np.full((n, 20, 2), np.nan, np.float32)
        a[rows] = v
        P5[name] = a
    mc = {sd: np.load(data_dir() / MC_RUNS[sd] / "preds_obs.npz") for sd in SEEDS}
    for sd in SEEDS:
        assert (mc[sd]["rows"] == rows).all()
    for m in MODELS:
        dP5 = mc[0][f"M-C pair [{m}]"] - mc[0][f"prior [{m}]"]
        dI3 = z3[f"M-C pair [{m}]"] - z3[f"ridge_late op-{m} temporal"]
        put5(f"M-C pair [{m}]", mc[0][f"M-C pair [{m}]"])
        for sd in SEEDS:
            put5(f"prior s{sd} [{m}]", mc[sd][f"prior [{m}]"])
        P3[f"ridge_late op-{m} temporal"], P3[f"M-C pair [{m}]"] = z3[f"ridge_late op-{m} temporal"], z3[f"M-C pair [{m}]"]
        put5(f"ridge_late NAV [{m}]", _grid(nr[f"p5_{m}"][_rows_of(nr["p5_keys"], t.frame_name.to_numpy()[rows])]))
        P3[f"ridge_late NAV [{m}]"] = _grid(nr[f"i3_{m}"][_rows_of(nr["i3_keys"], t3.frame_name.to_numpy())])
        for sd in SEEDS:
            c = np.load(out(f"p5cls_s{sd}.npz"), allow_pickle=True)
            put5(f"cls_late P5 s{sd} [{m}]", c[f"p5_{m}"][_rows_of(c["p5_keys"], t.frame_name.to_numpy()[rows])])
            P3[f"cls_late P5 s{sd} [{m}]"] = c[f"i3_{m}"][_rows_of(c["i3_keys"], t3.frame_name.to_numpy())]
            if (m, sd) not in sels:
                log.warning("no Hydra fit for %s seed %d", m, sd)
                continue
            z = np.load(sels[(m, sd)], allow_pickle=True)
            A = z["anchors"]
            r5, r3 = _rows_of(z["p5_keys"], t.frame_name.to_numpy()[rows]), _rows_of(z["i3_keys"], t3.frame_name.to_numpy())
            hy5, hy3 = _grid(A[z["p5_hydra"][r5]]), _grid(A[z["i3_hydra"][r3]])
            put5(f"cls_late NAV s{sd} [{m}]", _grid(A[z["p5_clsref"][r5]]))
            P3[f"cls_late NAV s{sd} [{m}]"] = _grid(A[z["i3_clsref"][r3]])
            put5(f"Hydra s{sd} [{m}]", hy5)
            P3[f"Hydra s{sd} [{m}]"] = hy3
            for tag, g5v, g3v in (("", np.ones(len(rows)), np.ones(n3)), (" x g2", g5["g2nav"], g3i["g2nav"]),
                                  (" x g2wod", g5["g2wod"], g3i["g2wod"])):
                put5(f"Hydra s{sd} + Delta{tag} [{m}]", hy5 + g5v[:, None, None] * dP5)
                P3[f"Hydra s{sd} + Delta{tag} [{m}]"] = hy3 + g3v[:, None, None] * dI3
    np.savez_compressed(rl.dir / "preds_p5_obs.npz", rows=rows, **{k: v[rows] for k, v in P5.items()})
    np.savez_compressed(rl.dir / "preds_i3.npz", frame_name=t3.frame_name.to_numpy(), **P3)
    # P5
    oo, nn = E.deltas(p5["obs"], p5["null"], t, P5)
    res = E.exam(oo, nn, p5["pairs"], list(P5))
    res["flips"].to_csv(rl.dir / "p5_flip_rates.csv", index=False)
    crit = []
    for m in MODELS:
        for sd in SEEDS:
            arms = [k for k in P5 if k.endswith(f"[{m}]") and (f"s{sd} " in k or (sd == 0 and " s" not in k))]
            crit.append(MC.criteria(res, arms, f"prior s{sd} [{m}]").assign(model=m, seed=sd))
    crit = pd.concat(crit)
    crit.to_csv(rl.dir / "p5_criteria.csv", index=False)
    # I3
    tfv6, E.TFV6 = E.TFV6, {}
    o3, n3_ = E.deltas(i3["obs"], i3["null"], t3, P3)
    r3 = E.exam(o3, n3_, i3["pairs"], list(P3))
    E.TFV6 = tfv6
    r3["flips"].to_csv(rl.dir / "i3_flip_rates.csv", index=False)
    rl.log.info("P5 criteria\n%s", crit.to_markdown(index=False, floatfmt=".3f"))
    fl = r3["flips"]
    rl.log.info("I3 pooled\n%s", fl[fl.scope == "pooled"][["examinee", "tau_model", "flip_rate", "flip_lo", "flip_hi",
                                                            "false_flip_null_oos"]].to_markdown(index=False, floatfmt=".3f"))


def navjobs(rl):
    """navtest predictions for the devkit: Hydra (every model x seed), Hydra + g2 * Delta (E1's M-C correction on
    0.5 ... 4.0 s, heading kept) and Hydra + Delta ungated (seed 0); the activation of Delta on navtest (description)."""
    from . import elicit_e1 as E1, p5_pairs as P
    from .real_g1 import mc_taus
    sc = pd.read_csv(data_dir() / E1_NAV / "navtest_scopes.csv")
    g2 = np.load(data_dir() / G2_RUN / "g2_nav_eval.npz", allow_pickle=True)
    taus, jobs, acts = mc_taus(), [], []
    for (m, sd), f in sorted(_sel_files().items()):
        fitdir = f.parent
        hy = np.load(fitdir / f"navtest_hydra_{m}_s{sd}.npz")
        tok = hy["tokens"]
        d = np.load(data_dir() / E1_NAV / f"navtest_delta_{m}.npz")
        assert (d["tokens"] == tok).all() and (sc.token.to_numpy() == tok).all()
        g = pd.Series(g2["gate"], index=g2["frame_id"].astype(str)).reindex(tok.astype(str)).to_numpy().astype(np.float32)
        assert not np.isnan(g).any()
        arms = {f"n3_hydra_{m}_s{sd}": hy["poses"]}
        for tag, gv in (("g2mc", g), ("mc", np.ones(len(tok), np.float32))):
            if tag == "mc" and sd != 0:
                continue
            a = hy["poses"].copy()
            a[..., :2] += gv[:, None, None] * d["delta"][:, 1:16:2]
            arms[f"n3_hydra_{tag}_{m}_s{sd}"] = a
            act = (np.abs(P.v2(E1._grid20(a)) - P.v2(E1._grid20(hy["poses"]))) >= taus[m]).astype(float)
            for k, msk in (("all", np.ones(len(tok), bool)), ("straight", sc.straight.to_numpy()), ("ped_cyc_corridor", sc.ped_cyc_corridor.to_numpy())):
                acts.append({"model": m, "seed": sd, "arm": tag, "scope": k, "n": int(msk.sum()), "activation": float(act[msk].mean()),
                             "gate_mean": float(gv[msk].mean())})
        for name, poses in arms.items():
            path = rl.dir / f"navtest_{name}.npz"
            np.savez(path, tokens=tok, poses=poses.astype(np.float32))
            jobs += [f"{v} navtest {name} {path}" for v in ("v1", "v2")]
    (rl.dir / "score_jobs.txt").write_text("\n".join(jobs) + "\n")
    pd.DataFrame(acts).to_csv(rl.dir / "navsim_activation.csv", index=False)
    rl.log.info("%d devkit jobs -> %s\n%s", len(jobs), rl.dir / "score_jobs.txt", pd.DataFrame(acts).to_markdown(index=False, floatfmt=".3f"))


def navtable(rl):
    """Official navtest scores with token-bootstrap CIs, and the paired deltas the N3 table and criterion read."""
    from .openloop_standing import _latest
    f = lambda df: df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)].set_index("token")["score"].astype(float)  # noqa: E731
    sc = pd.read_csv(data_dir() / E1_NAV / "navtest_scopes.csv").set_index("token")
    names = {}
    for m in MODELS:
        names[f"ridge_late [{m}]"] = f"heads_ridge_late_{m}_temporal"
        names[f"cls_late G3 [{m}]"] = f"heads_cls_late_{m}_temporal"
        names[f"ridge_late + Delta [{m}]"] = f"ridge_late_{m}_plus_mc"
        for sd in SEEDS:
            names[f"Hydra s{sd} [{m}]"] = f"n3_hydra_{m}_s{sd}"
            names[f"Hydra s{sd} + g2 Delta [{m}]"] = f"n3_hydra_g2mc_{m}_s{sd}"
        names[f"Hydra s0 + Delta [{m}]"] = f"n3_hydra_mc_{m}_s0"
    sco, rows, pairs = {}, [], []
    rng = np.random.default_rng(0)
    for ver, metric in (("v1", "PDMS"), ("v2", "EPDMS")):
        for label, name in names.items():
            df = _latest(ver, "navtest", name)
            if df is None:
                log.warning("%s %s missing (%s)", metric, label, name)
                continue
            v = f(df)
            sco[(metric, label)] = v
            bs = v.to_numpy()[rng.integers(0, len(v), (2000, len(v)))].mean(1)
            rows.append({"metric": metric, "row": label, "n": len(v), "score": 100 * v.mean(), "lo": 100 * np.percentile(bs, 2.5),
                         "hi": 100 * np.percentile(bs, 97.5)})
        for m in MODELS:
            cmp = [(f"Hydra s{sd} [{m}]", f"ridge_late [{m}]") for sd in SEEDS] + \
                  [(f"Hydra s{sd} + g2 Delta [{m}]", f"Hydra s{sd} [{m}]") for sd in SEEDS] + \
                  [(f"Hydra s0 + Delta [{m}]", "Hydra s0 [%s]" % m), (f"ridge_late + Delta [{m}]", f"ridge_late [{m}]")]
            for a, b in cmp:
                if (metric, a) not in sco or (metric, b) not in sco:
                    continue
                x, y = sco[(metric, a)].align(sco[(metric, b)], join="inner")
                for grp, msk in (("all", None), ("ped_cyc_corridor", sc.ped_cyc_corridor), ("straight", sc.straight)):
                    keep = np.ones(len(x), bool) if msk is None else msk.reindex(x.index).fillna(False).to_numpy(bool)
                    dd = (x - y).to_numpy()[keep]
                    bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
                    pairs.append({"metric": metric, "a": a, "b": b, "group": grp, "n": len(dd), "delta": 100 * dd.mean(),
                                  "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
    pd.DataFrame(rows).to_csv(rl.dir / "navsim_scores.csv", index=False)
    pd.DataFrame(pairs).to_csv(rl.dir / "navsim_paired.csv", index=False)
    rl.log.info("scores\n%s\npaired\n%s", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"),
                pd.DataFrame(pairs).query("group == 'all'").to_markdown(index=False, floatfmt=".2f"))


# ---------------------------------------------------------------- the two side cells

NUSC_RUN = "runs/nusc_backbones/ladder/20260925-144240"
E5_FIT = "runs/elicitation/e5-fit/20260926-021421"


def nusc(rl):
    """[B] 09:58 (7): collision of the frozen nuScenes heads (decision 40 (4)'s ladder predictions) under decision 39's
    exam code: rear-axle 0.25 s points -> LIDAR_TOP point at the GT keyframe times, VAD and BEV-Planner collision."""
    from . import nuscenes_zs as Z
    z = np.load(data_dir() / NUSC_RUN / "nusc_preds.npz", allow_pickle=True)
    idx = Z.load_index("val")
    by = {e["token"]: e for e in idx["samples"] if e.get("valid") and "boxes" in e}
    keep = np.array([t in by for t in z["token"]])
    rl.log.info(f"{keep.sum()} / {len(keep)} ladder val samples have decision 39's GT boxes")
    samples = [by[t] for t in z["token"][keep]]
    lid = [idx["scenes"][e["scene"]]["lidar_xyz"] for e in samples]
    t_src = np.arange(1, 13) * 0.25
    arms = {k[5:]: v[keep] for k, v in z.items() if k.startswith("pred_")}
    arms["log future (sanity)"] = z["fut"][keep]
    v0 = np.linalg.norm(np.stack([e["gt_rear"][0] for e in samples]), axis=1) / np.array([e["fut_t"][0] for e in samples])
    arms["CV (speed of the first 0.5 s)"] = (v0[:, None] * t_src[None])[..., None] * np.array([1.0, 0.0])
    scene = z["scene"][keep]
    per, rows = {}, []
    for a, P in arms.items():
        pl = np.stack([Z.to_lidar_point(t_src, P[i], Z.traj_yaw(P[i].astype(np.float64)), lid[i], samples[i]["fut_t"])[0]
                       for i in range(len(P))])
        per[a] = Z.horizons(Z.per_sample(pl, samples))
    u, inv = np.unique(scene, return_inverse=True)
    draws = np.random.default_rng(0).integers(len(u), size=(2000, len(u)))
    M = np.stack([np.bincount(d, minlength=len(u)) for d in draws]).astype(float)
    cnt = np.bincount(inv, minlength=len(u)).astype(float)

    def boot(v):
        s_ = np.bincount(inv, v, minlength=len(u))
        b = (M @ s_) / (M @ cnt)
        return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
    base = "ridge ego"
    for a, h in per.items():
        r = {"arm": a, "n": len(scene)}
        for k in ("l2", "col_vad", "col_bevp"):
            mean_ = (h[f"{k}_1s"] + h[f"{k}_2s"] + h[f"{k}_3s"]) / 3
            r[f"{k}_avg"], r[f"{k}_lo"], r[f"{k}_hi"] = boot(mean_)
            if a != base:
                d, lo, hi = boot(mean_ - (per[base][f"{k}_1s"] + per[base][f"{k}_2s"] + per[base][f"{k}_3s"]) / 3)
                r[f"d_{k}"], r[f"d_{k}_lo"], r[f"d_{k}_hi"] = d, lo, hi
            for t_ in (1, 2, 3):
                r[f"{k}_{t_}s"] = float(h[f"{k}_{t_}s"].mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(rl.dir / "nusc_collision.csv", index=False)
    rl.log.info("\n%s", t[["arm", "n", "l2_avg", "col_vad_avg", "col_vad_lo", "col_vad_hi", "d_col_vad", "d_col_vad_lo", "d_col_vad_hi",
                           "col_bevp_avg", "d_col_bevp", "d_col_bevp_lo", "d_col_bevp_hi"]].to_markdown(index=False, floatfmt=".3f"))


def e4c_students(rl):
    """[B] 09:58 (7): elicit_e4c's curves on the BA set with E5's students added (L = 0.1 s), from their stored preds."""
    from . import elicit_e4 as E4, elicit_e4c as C, elicit_i3 as I, p5_exam as E
    obs, null, taus, pooled, _ = E4.load("ba")
    with I.p5_set(I.BA):
        t, _, _, o0, n0, _ = E.load()
    z = np.load(data_dir() / E5_FIT / "preds_obs.npz")
    names = [k for k in z.keys() if k.startswith("E5 ")]
    preds = {}
    for k in names:
        a = np.full((len(t), 20, 2), np.nan, np.float32)
        a[z["rows"]] = z[k]
        preds[k] = a
    tfv6, E.TFV6 = E.TFV6, {}
    oo, nn = E.deltas(o0, n0, t, preds)
    E.TFV6 = tfv6
    key, nkey = ["base_id", "seed", "k", "fn_plus"], ["base_id", "seed", "k", "fn_plus", "fn_null"]
    for df in (oo, nn):
        df["base_id"] = df.base_id.astype(str)
    obs = obs.merge(oo[key + names], on=key, how="left", validate="1:1")
    null = null.merge(nn[nkey + names], on=nkey, how="left", validate="1:1")
    assert obs[names].notna().all().all() and null[names].notna().all().all()
    fl = pd.read_csv(data_dir() / E5_FIT / "flip_rates.csv")
    taus |= fl[fl.scope == "pooled"].set_index("examinee").tau_model.loc[names].to_dict()
    lat = E4.latency
    E4.latency = lambda ex: 0.1 if ex.startswith("E5 ") else lat(ex)
    ex = names + ["prior [cinque]", "M-C pair [cinque]", "M-C pair op [cinque]", "prior [lebowski]", "M-C pair [lebowski]"]
    c, a, f, _ = C.curves(obs, null, taus, pooled, ex, rl, "ba")
    E4.latency = lat
    c.to_csv(rl.dir / "curves.csv", index=False)
    a.to_csv(rl.dir / "areas.csv", index=False)
    f.to_csv(rl.dir / "first_flips.csv", index=False)
    rl.log.info("\n%s", a[a.scope != "pooled"][["examinee", "scope", "L_s", "pairs", "area_L_10", "area_minus_null", "diff_lo", "diff_hi",
                                              "A3", "A3_minus_null", "A3_diff_lo", "A3_diff_hi"]].to_markdown(index=False, floatfmt=".3f"))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "fit", "navridge", "p5cls", "gates", "compat", "exam", "navjobs", "navtable", "nusc", "e4c"))
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
        {"navridge": navridge, "gates": gates, "compat": compat, "exam": exam, "navjobs": navjobs, "navtable": navtable,
         "nusc": nusc, "e4c": e4c_students}[a.step](rl)
    rl.close()


if __name__ == "__main__":
    main()
