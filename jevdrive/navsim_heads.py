"""Thin heads on NAVSIM (todos/2026-09-25-openpilot-openloop-comparison.md, G2 / G3): fit on navtrain, predict navtest and
navhard_two_stage, score with the official devkits through the replay agent (scripts/navsim_zs_score.sh).

The recipe is the project's (decisions 9 / 10, P0), carried to NAVSIM's 8 poses @2 Hz:

  ridge ego          ridge from the 32 ego inputs a NAVSIM agent sees (4 history poses, velocities, accelerations,
                     current driving command) to the 8 future (x, y, yaw); lambda by 5-fold CV grouped by log
  ridge_late <tap>   ridge from the standardised feature to the residual of `ridge ego`'s out-of-fold prediction
  cls ego K1024      k-means vocabulary (K = 1024) over navtrain futures' (x, y), linear softmax (planner.ce_solve),
                     lambda by top-1 ADE on a log-grouped inner split; heading = circular mean of the anchor's members
  cls_late <tap>     the feature's linear softmax on top of `cls ego`'s frozen out-of-fold logits (late fusion)

    python -m jevdrive.navsim_heads fit                       # writes runs/navsim_zs/heads/<ts>/<split>_<arm>.npz
"""
import json
import time

import numpy as np
import torch

from . import navsim_zs as Z
from . import planner, traj
from .common import get_logger

log = get_logger(__name__)
DEV = "cuda"
K, FOLDS, TAPS = 1024, 5, ("cinque", "lebowski")
EVAL = ("navtest", "navhard_two_stage")


def ego_features(idx: list) -> np.ndarray:
    """(n, 32): pose (4 x 3), velocity (4 x 2), acceleration (4 x 2) in the t0 rear-axle frame, current command one-hot."""
    return np.stack([np.r_[e["pose"].ravel(), e["vel"].ravel(), e["acc"].ravel(), e["cmd"][-1]] for e in idx]).astype(np.float32)


def load(split: str, feats: bool) -> dict:
    idx = Z.load_index(split, slim=True)
    tok = np.array([e["token"] for e in idx])
    d = {"tokens": tok, "log": np.array([e["log_name"] for e in idx]), "ego": ego_features(idx),
         "stage": np.array([e["stage"] for e in idx])}
    fp = Z.root("index") / f"{split}_future.npz"
    if fp.exists():
        with np.load(fp) as f:                       # read each array once: f[key] re-reads it from the zip
            ft, fp_ = f["tokens"], f["poses"]
        pos = dict(zip(ft.tolist(), range(len(ft))))
        nan = np.full((8, 3), np.nan, np.float32)
        d["fut"] = np.stack([fp_[pos[t]] if t in pos else nan for t in tok]).astype(np.float32)
    if feats:
        for m in TAPS:
            with np.load(Z.root("openpilot", split) / f"{m}_temporal.npz") as z:
                zt, zf, zp = z["tokens"], z["temporal"], z["poses"]
            at = dict(zip(zt.tolist(), range(len(zt))))
            sel = np.array([at[t] for t in tok])
            d[m], d[f"native {m}"] = zf[sel].astype(np.float32), zp[sel]
    return d


def _std(Xtr, *Xs):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-6, sd, 1)
    return [torch.as_tensor((X - mu) / sd, device=DEV) for X in (Xtr, *Xs)]


def _group_folds(groups: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    u = np.unique(groups)
    fold_of = dict(zip(np.random.default_rng(seed).permutation(u), np.arange(len(u)) % k))
    return np.array([fold_of[g] for g in groups])


def ridge_cv(X: torch.Tensor, Y: torch.Tensor, folds: np.ndarray) -> tuple[float, torch.Tensor]:
    """lambda by grouped K-fold (mean ADE over the (x, y) waypoints), and the out-of-fold predictions at it."""
    lams, T = planner.LAM_RIDGE, Y.shape[1] // 3
    err, oof = np.zeros(len(lams)), torch.zeros(len(lams), *Y.shape, device=DEV)
    for f in range(folds.max() + 1):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        W = planner.ridge_solve(X, Y, tr, lams)
        oof[:, te] = planner.linear_apply(W, X, te)
    xy = lambda P: P.reshape(*P.shape[:-1], T, 3)[..., :2]  # noqa: E731
    err = (xy(oof) - xy(Y)).norm(dim=-1).mean((1, 2)).cpu().numpy()
    b = planner._pick(err, lams, "ridge")
    return float(lams[b]), oof[b]


def fit_ridge(Xtr, Ytr, Xev: dict, folds) -> tuple[dict, torch.Tensor, dict]:
    lam, oof = ridge_cv(Xtr, Ytr, folds)
    W = planner.ridge_solve(Xtr, Ytr, np.arange(len(Xtr)), [lam])
    return {s: planner.linear_apply(W, X, np.arange(len(X)))[0] for s, X in Xev.items()}, oof, {"lam": lam}


def vocabulary(fut: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """K-means anchors over (x, y), the navtrain rows' nearest anchor, and each anchor's circular-mean heading."""
    F = torch.as_tensor(fut[..., :2].reshape(len(fut), -1), device=DEV)
    A = traj.kmeans(F, K, seed=0)
    ids = traj.nearest(F, A, 1)[0][:, 0]
    s, c = np.zeros((K, 8)), np.zeros((K, 8))
    np.add.at(s, ids, np.sin(fut[..., 2]))
    np.add.at(c, ids, np.cos(fut[..., 2]))
    yaw = np.arctan2(s, c).astype(np.float32)
    return A, ids, yaw


def anchor_poses(A: torch.Tensor, yaw: np.ndarray, top: np.ndarray) -> np.ndarray:
    xy = A.reshape(K, 8, 2).cpu().numpy()
    return np.concatenate([xy[top], yaw[top][..., None]], -1)


def fit_cls(Xtr, ids, A, yaw, fut_xy, Xev: dict, inner: np.ndarray, off_tr=None, off_ev=None) -> tuple[dict, dict]:
    """planner.Heads.cls on NAVSIM rows: lambda by top-1 (x, y) ADE on the inner split, refit on all of navtrain."""
    tgt = (ids[:, None], np.ones((len(ids), 1), np.float32))
    fit_r, sel_r, all_r = np.flatnonzero(~inner), np.flatnonzero(inner), np.arange(len(Xtr))
    Axy = A.reshape(K, 8, 2).cpu().numpy()
    W, _ = planner.ce_solve(Xtr, tgt, fit_r, planner.LAM_CLS, K, offset=off_tr)
    top = planner.cls_topk(W, Xtr, sel_r, 1, offset=off_tr)[:, :, 0]
    err = [np.linalg.norm(Axy[top[:, i]] - fut_xy[sel_r], axis=-1).mean() for i in range(len(planner.LAM_CLS))]
    del W
    b = planner._pick(err, planner.LAM_CLS, "cls")
    W, _ = planner.ce_solve(Xtr, tgt, all_r, [planner.LAM_CLS[b]], K, offset=off_tr)
    out = {s: anchor_poses(A, yaw, planner.cls_topk(W, X, np.arange(len(X)), 1,
                                                     offset=None if off_ev is None else off_ev[s])[:, 0, 0])
           for s, X in Xev.items()}
    return out, {"lam": float(planner.LAM_CLS[b]), "sel_ade": float(err[b]), "W": W}


def oof_logits(Xtr, ids, lam, folds) -> torch.Tensor:
    """`cls ego`'s out-of-fold logits on navtrain (grouped folds at its chosen lambda): the offset cls_late corrects."""
    tgt = (ids[:, None], np.ones((len(ids), 1), np.float32))
    out = torch.zeros(len(Xtr), K, device=DEV)
    for f in range(folds.max() + 1):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        W, _ = planner.ce_solve(Xtr, tgt, tr, [lam], K)
        out[te] = planner.linear_apply(W, Xtr, te)[0]
    return out


def fit(feats: bool = True) -> dict:
    from .runlog import RunLog
    rl = RunLog("navsim_zs", "heads")
    t0 = time.time()
    tr = load("navtrain", feats)
    keep = (tr["stage"] == "one") & ~np.isnan(tr["fut"]).any((1, 2))
    tr = {k: v[keep] for k, v in tr.items()}
    ev = {s: load(s, feats) for s in EVAL}
    fut = tr["fut"]
    fut_xy = fut[..., :2]
    folds = _group_folds(tr["log"], FOLDS)
    inner = _group_folds(tr["log"], 5, seed=1) == 0            # 20% of the logs: the cls lambda's inner split
    rl.log.info(f"navtrain rows {len(fut)} over {len(np.unique(tr['log']))} logs; eval {[len(v['tokens']) for v in ev.values()]}")
    Y = torch.as_tensor(fut.reshape(len(fut), -1), device=DEV)
    Xe, *Xe_ev = _std(tr["ego"], *(ev[s]["ego"] for s in EVAL))
    Xe_ev = dict(zip(EVAL, Xe_ev))
    preds, stats = {}, {}
    p, oof_e, st = fit_ridge(Xe, Y, Xe_ev, folds)
    preds["ridge ego"], stats["ridge ego"] = {s: v.reshape(-1, 8, 3).cpu().numpy() for s, v in p.items()}, st
    base_ev = p
    A, ids, yaw = vocabulary(fut)
    rl.log.info(f"vocabulary K={K}: oracle (x, y) ADE {np.linalg.norm(A.reshape(K, 8, 2).cpu().numpy()[ids] - fut_xy, axis=-1).mean():.3f} m")
    preds["cls ego K1024"], st = fit_cls(Xe, ids, A, yaw, fut_xy, Xe_ev, inner)
    We = st.pop("W")
    stats["cls ego K1024"] = st
    off_tr = oof_logits(Xe, ids, st["lam"], folds)
    off_ev = {s: planner.linear_apply(We, X, np.arange(len(X)))[0] for s, X in Xe_ev.items()}
    for m in (TAPS if feats else ()):
        Xf, *Xf_ev = _std(tr[m], *(ev[s][m] for s in EVAL))
        Xf_ev = dict(zip(EVAL, Xf_ev))
        r, _, st = fit_ridge(Xf, Y - oof_e, Xf_ev, folds)
        preds[f"ridge_late {m} temporal"] = {s: (base_ev[s] + r[s]).reshape(-1, 8, 3).cpu().numpy() for s in EVAL}
        stats[f"ridge_late {m} temporal"] = st
        preds[f"cls_late {m} temporal"], st = fit_cls(Xf, ids, A, yaw, fut_xy, Xf_ev, inner, off_tr, off_ev)
        st.pop("W")
        stats[f"cls_late {m} temporal"] = st
        for s in EVAL:
            preds.setdefault(f"native {m}", {})[s] = ev[s][f"native {m}"]
        del Xf, Xf_ev
        torch.cuda.empty_cache()
    for arm, ps in preds.items():
        for s, P in ps.items():
            np.savez(rl.dir / f"{s}_{arm.replace(' ', '_')}.npz", tokens=ev[s]["tokens"], poses=P.astype(np.float32))
    for s in EVAL:   # (x, y) ADE vs log on stage-one tokens, a sanity readout only (the devkit scores are the result)
        f = ev[s].get("fut")
        m = ~np.isnan(f).any((1, 2))
        stats[f"ade_{s}"] = {arm: float(np.linalg.norm(ps[s][m, :, :2] - f[m, :, :2], axis=-1).mean()) for arm, ps in preds.items()}
    (rl.dir / "stats.json").write_text(json.dumps(stats, indent=1))
    rl.log.info(json.dumps(stats, indent=1) + f"\n{time.time() - t0:.0f} s -> {rl.dir}")
    rl.close()
    return stats


def ctrv(split: str) -> np.ndarray:
    """(n, 8, 3) constant turn rate and velocity from the AgentInput: speed |v(t0)|, yaw rate from the last two
    history poses (0.5 s apart), an arc from the t0 rear axle (I2 of the comparison todo)."""
    idx = Z.load_index(split, slim=True)
    v = np.array([np.linalg.norm(e["vel"][-1]) for e in idx])
    dyaw = np.array([e["pose"][-1, 2] - e["pose"][-2, 2] for e in idx])
    w = np.arctan2(np.sin(dyaw), np.cos(dyaw)) / 0.5
    t = Z.T_OUT[None]
    wt = w[:, None] * t
    small = np.abs(w[:, None]) < 1e-4
    ws = np.where(small, 1.0, w[:, None])
    x = np.where(small, v[:, None] * t, v[:, None] * np.sin(wt) / ws)
    y = np.where(small, 0.5 * v[:, None] * w[:, None] * t ** 2, v[:, None] * (1 - np.cos(wt)) / ws)
    return np.stack([x, y, wt], -1).astype(np.float32), np.array([e["token"] for e in idx])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fit", "ctrv"))
    ap.add_argument("--no-feats", action="store_true", help="ego heads only (before the temporal extraction exists)")
    a = ap.parse_args()
    if a.step == "ctrv":
        for sp in EVAL:
            P, tok = ctrv(sp)
            np.savez(Z.root("heads", "kinematic") / f"{sp}_ctrv.npz", tokens=tok, poses=P)
            print(sp, len(tok), "mean 4 s x", float(P[:, -1, 0].mean()))
    else:
        fit(not a.no_feats)
