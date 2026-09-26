"""Night queue 3, lane C, Q2: eliciting bypass on frozen openpilot features, on P6 (todos/2026-09-26-night-queue-3.md,
Q2 and the [C] entries, written before any number).

Arms (every arm shares the prior, the folds and the judge jevdrive.nq3_p6):
  A0  `ridge_late` on the P6 expert futures: ridge on the ego input, ridge on openpilot `temporal` for the residual
      (reactivity_mc.fit_fold's prior, lambda by grouped CV) -- uniform imitation, lateral and longitudinal 0.25-5 s
  A1  A0 + lateral pair-Delta: W z(x) added to y only, fitted on (x10, x00) same-tick expert lateral differences minus
      the prior's, mu |Z W|^2 zero constraint on the training rows, lambda by 3-fold route-grouped inner CV
      (reactivity_mc._solve_pair, M-C's `pair` arm with the lateral target)
  A2  mode head: multinomial logistic regression over keep / stop / bypass_L / bypass_R / wait (labels from the
      expert's own 5 s future with decision 52's world rule, in route coordinates) on [ego input | temporal];
      output = prior + the predicted mode's lateral template (mean training residual y - prior_y of that mode)
  A3  A2 + paired consistency: Delta on the logits, W z(x), fitted like A1 on (x10, x00) pairs with target
      c (e_mode(x10) - e_mode(x00)) - (logit_A2(x10) - logit_A2(x00)), c = log 16
  A4  `cls_late` (night2_n3.p5cls's recipe) over a vocabulary of K-means 1024 on the training futures plus 64 K-means
      anchors of the training x10 bypass-shaped expert futures plus 64 of WOD train's bypass-shaped futures
  A5  (v1 only) A1 / A3 with a second pair term (x11 - x10)
Splits: `loco` = leave one obstacle class out (9 folds: the 8 main classes + VehicleOpensDoorTwoWays; InvadingTurn and
YieldToEmergencyVehicle rows are training-only); `route` = 5 route folds (seeded permutation of the base routes).
Training rows = worlds x10, x00, x11, x01 of the training folds; wnull / shoulder / mirror are exam-only.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import nq3_p6 as J
from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "nq3" / "q2"
CLASSES9 = J.MAIN + ("VehicleOpensDoorTwoWays",)
TRAIN_WORLDS = ("x10", "x00", "x11", "x01")
MODES = ("keep", "stop", "bypass_L", "bypass_R", "wait")
C_LOGIT = float(np.log(16.0))
ARMS = ("A0", "A1", "A2", "A3", "A4")
MODELS = ("cinque", "lebowski")
SEEDS = (0, 1, 2)
DEV = "cuda"
WINDOW = 100                    # 5 s of 20 Hz ticks for the mode labels


def proc(set_: str, *p) -> Path:
    return data_dir() / "processed" / set_ / Path(*p)


# ---------------------------------------------------------------- mode labels (decision 52's rule on the 5 s future)

def _lat_world(g: Path, rid: str):
    from . import p6
    x = p6._load(g, rid, {})
    if x is None:
        return None
    lat = x[1]
    return pd.DataFrame({"rid": rid, "k": lat.index.to_numpy(), "d": lat.d.to_numpy(np.float32),
                         "v": lat.v.to_numpy(np.float32)})


def lateral_cache(set_: str = "carla_p6", gen: Path | None = None, workers: int = 16) -> pd.DataFrame:
    """Every world's route offset d (left +) and speed v per 20 Hz tick -> processed/<set>/nq3_lateral.parquet."""
    from joblib import Parallel, delayed
    from . import p6
    f = proc(set_, "nq3_lateral.parquet")
    if f.exists():
        return pd.read_parquet(f)
    g = gen or p6.root("gen")
    rids = sorted(pd.read_parquet(proc(set_, "index.parquet"), columns=["route_id"]).route_id.unique())
    lat = pd.concat([x for x in Parallel(workers)(delayed(_lat_world)(g, r) for r in rids) if x is not None])
    lat.to_parquet(f, index=False)
    return lat


def mode_labels(t: pd.DataFrame, lat: pd.DataFrame) -> np.ndarray:
    """Per frame: decision 52's world rule on ticks k .. k + 5 s of the expert's own recording: bypass = max|d| >= 1 m
    (side by its sign), wait = a stop (v < 0.5 m/s for 1 s after having driven > 3 m/s) before the lateral start
    (|d| >= 0.5 m for 0.5 s), stop, keep. Codes index MODES (wait = wait-then-bypass, either side)."""
    from . import p6
    out = np.full(len(t), -1, np.int64)
    for rid, g in t.groupby("route_id"):
        L = lat[lat.rid == rid].set_index("k").sort_index()
        d, v = L.d.to_numpy(), L.v.to_numpy()
        ks = L.index.to_numpy()
        moved = np.maximum.accumulate(v > p6.MOVING_V)
        at = pd.Series(np.arange(len(ks)), index=ks)
        for i, k in zip(g.index, g.k):
            j = at.get(k)
            if j is None:
                continue
            w = slice(j, j + WINDOW + 1)
            dw, vw = d[w], v[w]
            st = p6._runs(np.abs(dw) >= p6.LAT_ON, p6.LAT_HOLD)
            t_lat = st[0] if len(st) else None
            ip = int(np.argmax(np.abs(dw)))
            stops = p6._runs((vw < p6.STOP_V) & moved[w], p6.STOP_HOLD)
            if abs(dw[ip]) >= p6.BYPASS_M:
                out[i] = 4 if t_lat is not None and any(s < t_lat for s in stops) else (2 if dw[ip] > 0 else 3)
            else:
                out[i] = 1 if len(stops) else 0
    return out


# ---------------------------------------------------------------- data

def load(set_: str = "carla_p6", models=MODELS, streams=()) -> dict:
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot
    t = pd.read_parquet(proc(set_, "index.parquet"))
    past, fut = np.load(proc(set_, "past.npy")), np.load(proc(set_, "future.npy"))
    with I.p5_set(set_):
        op = p5_openpilot.load(t, models, sub="op_streams_vis")
    labf = proc(set_, "nq3_modes.npy")
    if labf.exists():
        lab = np.load(labf)
    else:
        lab = mode_labels(t, lateral_cache(set_))
        np.save(labf, lab)
    p = J.pairs(set_)
    D = {"set": set_, "t": t, "past": past, "fut": fut, "lab": lab, "pairs": p,
         "F": torch.as_tensor(fut.reshape(len(t), -1), device=DEV),
         "Ego": torch.as_tensor(E.ego_input(t, past), device=DEV),
         "op": {m: torch.as_tensor(op[f"op-{m} temporal"], device=DEV) for m in models}}
    for s in streams:                                          # backbone controls: rows without features stay NaN
        X, has = backbone(set_, t, s)
        D.setdefault("bb", {})[s] = (torch.as_tensor(X, device=DEV), has)
    log.info("%s: %d rows, mode labels %s", set_, len(t), dict(zip(MODES, np.bincount(lab[lab >= 0], minlength=5))))
    return D


def backbone(set_: str, t: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    from . import elicit_i3 as I, p5_pairs as P
    if name == "qwen L18_last":
        d = proc(set_, "features")
        names = pd.concat([pd.read_parquet(c / "index.parquet") for c in sorted(d.glob("c[0-9]*"))
                           if (c / "meta.json").exists()]).frame_name
        has = t.frame_name.isin(set(names)).to_numpy()
        X = np.zeros((len(t), 2560), np.float32)
        with I.p5_set(set_):
            X[has] = P.load_features(t[has], ("L18_last",))["L18_last"]
        return X, has
    d = proc(set_, "bb_vjepa2")
    names = pd.read_parquet(d / "index.parquet").frame_name
    a = np.load(d / "mean.npy", mmap_mode="r")
    at = t.frame_name.map(pd.Series(np.arange(len(names)), index=names))
    has = at.notna().to_numpy()
    X = np.zeros((len(t), a.shape[1]), np.float32)
    X[has] = a[at[has].astype(int).to_numpy()]
    return X, has


# ---------------------------------------------------------------- folds

def fold_ids(D: dict, split: str, seed: int) -> np.ndarray:
    t = D["t"]
    if split == "loco":
        m = {c: i for i, c in enumerate(CLASSES9)}
        return t.scenario.map(m).fillna(-1).astype(int).to_numpy()
    bases = np.array(sorted(t.base_id.unique()))
    f = dict(zip(np.random.default_rng(seed).permutation(bases), np.arange(len(bases)) % 5))
    return t.base_id.map(f).astype(int).to_numpy()


def n_folds(split: str) -> int:
    return len(CLASSES9) if split == "loco" else 5


# ---------------------------------------------------------------- arms

def _stats(X: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """planner.standardize's statistics."""
    mu, sd = X[rows].double().mean(0), X[rows].double().std(0, correction=0)
    return mu.float(), torch.where(sd > 1e-6, sd, 1).float()


def _inner(groups: np.ndarray, seed: int, k: int = 3):
    u = np.unique(groups)
    g = dict(zip(np.random.default_rng(seed).permutation(u), np.arange(len(u)) % k))
    f = np.array([g[x] for x in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def _pair_delta(Z, tr, ip, im, R, groups, seed, head=None, tag=""):
    """M-C's pair solve: W for min |D W - R|^2 + mu |Zc W|^2 + lam |W|^2, lam by route-grouped inner CV."""
    from . import reactivity_mc as MC
    zbar = Z[tr].mean(0)
    Zc = Z[tr] - zbar
    mu = len(ip) / len(tr)
    Dm = Z[ip] - Z[im]
    score = np.zeros(len(MC.LAMS))
    for a, b in _inner(groups, seed):
        Ws = MC._solve_pair(Dm[a], R[a], Zc, mu, MC.LAMS)
        score += [float(((Dm[b] @ W - R[b]) ** 2).sum()) for W in Ws]
    best = int(np.argmin(score))
    W = MC._solve_pair(Dm, R, Zc, mu, [MC.LAMS[best]])[0]
    if head is not None:
        head.update({f"{tag}W": W.cpu().numpy(), f"{tag}zbar": zbar.cpu().numpy(), f"{tag}lam": float(MC.LAMS[best])})
    return (Z - zbar) @ W, float(MC.LAMS[best]), best in (0, len(MC.LAMS) - 1)


def _mlr(X: torch.Tensor, y: np.ndarray, rows: np.ndarray, lam: float, K: int, W0=None) -> torch.Tensor:
    """Multinomial logistic regression, mean cross-entropy + lam / 2 |W|^2 (bias unpenalised, planner.ce_solve's
    objective), (d + 1, K), full-batch L-BFGS on the CPU in float64 (a 5-class problem is too small for a time-sliced
    GPU)."""
    Xr = torch.cat([X[rows].double(), torch.ones(len(rows), 1, dtype=torch.float64)], 1)
    yr = torch.as_tensor(y[rows])
    W = torch.zeros(X.shape[1] + 1, K, dtype=torch.float64) if W0 is None else W0.clone()
    W.requires_grad_(True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xr @ W, yr) + lam / 2 * (W[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach()


def _mode_head(X: torch.Tensor, lab: np.ndarray, tr: np.ndarray, groups: np.ndarray, seed: int, K: int):
    """lambda from planner.LAM_CLS by inner-validation cross-entropy on a 20 % route-grouped split, then the fit on tr."""
    from sklearn.model_selection import GroupShuffleSplit
    from . import planner
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr, groups=groups[tr]))
    Xb = torch.cat([X[tr[b]].double(), torch.ones(len(b), 1, dtype=torch.float64)], 1)
    ll, W = [], None
    for lam in planner.LAM_CLS[::-1]:                        # strong -> weak, warm started
        W = _mlr(X, lab, tr[a], float(lam), K, W)
        ll.append(float(torch.nn.functional.cross_entropy(Xb @ W, torch.as_tensor(lab[tr[b]]))))
    lams = planner.LAM_CLS[::-1]
    lam = float(lams[planner._pick(ll, lams, "mode head")])
    return _mlr(X, lab, tr, lam, K), lam


_WOD_ANCHORS = {}


def wod_bypass_anchors(seed: int, k: int = 64) -> torch.Tensor:
    from . import p6, traj, waymo_heads as H
    if seed not in _WOD_ANCHORS:
        f = data_dir() / "processed" / "carla_p6" / f"nq3_wod_bypass_anchors_s{seed}.npy"
        if not f.exists():
            fut = H.train_futures()
            b = fut[p6.bypass_shape(fut)]
            A = traj.kmeans(torch.as_tensor(b.reshape(len(b), -1), device=DEV), k, seed=seed)
            np.save(f, A.cpu().numpy().astype(np.float32))
        _WOD_ANCHORS[seed] = torch.as_tensor(np.load(f), device=DEV)
    return _WOD_ANCHORS[seed]


def fit_fold(D: dict, fold: np.ndarray, f: int, m: str, seed: int, arms=ARMS, stream: str | None = None,
             head: dict | None = None, rl=None, te_rows: np.ndarray | None = None) -> dict:
    """Predictions (len(te), 20, 2) of every arm for the rows of fold f, fitted on the other folds' training worlds.
    `stream` swaps the Delta stream of A1 / A3 for a backbone (the prior stays on openpilot `temporal`).
    `head` (dict) receives every fitted parameter (the closed-loop export)."""
    from sklearn.model_selection import GroupShuffleSplit
    from . import navsim_heads as NH, night2_n3 as N3, p6, planner, traj, waymo_stage_a as sa
    t, F, Ego, Xop, lab, p = D["t"], D["F"], D["Ego"], D["op"][m], D["lab"], D["pairs"]
    n = len(t)
    world = t.world.to_numpy()
    tr = np.flatnonzero((fold != f) & np.isin(world, TRAIN_WORLDS) & (lab >= 0))
    te = np.flatnonzero(fold == f) if te_rows is None else te_rows
    seq = t.base_id.to_numpy().astype(str)
    fut = D["fut"]
    sp = SimpleNamespace(train=tr, val=te, seq=seq)
    out, info = {}, {}
    mu_e, sd_e = _stats(Ego, tr)
    Xe = (Ego - mu_e) / sd_e
    _, st_e, We = sa.ridge_cv(Xe, F, sp, fut)
    base = planner.linear_apply(We, Xe, np.arange(n))[0]
    mu_o, sd_o = _stats(Xop, tr)
    Xi = (Xop - mu_o) / sd_o
    R0 = F - base
    _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
    prior = base + planner.linear_apply(Wp, Xi, np.arange(n))[0]
    info.update(lam_ego=st_e["lam"], lam_prior=st_p["lam"], n_train=len(tr), n_test=len(te))
    if head is not None:
        head.update(ego_mu=mu_e.cpu().numpy(), ego_sd=sd_e.cpu().numpy(), We=We[0].cpu().numpy(),
                    op_mu=mu_o.cpu().numpy(), op_sd=sd_o.cpu().numpy(), Wp=Wp[0].cpu().numpy())
    P3 = prior.reshape(n, 20, 2)
    if "A0" in arms:
        out["A0"] = P3[te].cpu().numpy()
    # training pairs: (x10, x00) bypass windows of the training folds' classes
    pb = p[(p.reading == "bypass")]
    pb = pb[fold[pb.ia.to_numpy()] != f]
    pb = pb[np.isin(pb.ia.to_numpy(), tr) & np.isin(pb.ib.to_numpy(), tr)]
    ip, im, grp = pb.ia.to_numpy(), pb.ib.to_numpy(), pb.base_id.to_numpy().astype(str)
    info["n_pairs"] = len(ip)
    if stream is None:
        Zsrc, trz, ok = Xop, tr, np.ones(n, bool)
    else:
        Zsrc, ok = D["bb"][stream]
        trz = tr[ok[tr]]
        keep = ok[ip] & ok[im]
        ip, im, grp = ip[keep], im[keep], grp[keep]
    mu_z, sd_z = _stats(Zsrc, trz)
    Z = (Zsrc - mu_z) / sd_z / np.sqrt(Zsrc.shape[1])
    if head is not None:
        head.update(z_mu=mu_z.cpu().numpy(), z_sd=sd_z.cpu().numpy(), z_dim=int(Zsrc.shape[1]))
    Fy, Py = F.reshape(n, 20, 2)[..., 1], P3[..., 1]
    if "A1" in arms:
        R = (Fy[ip] - Fy[im]) - (Py[ip] - Py[im])
        dlt, lam, edge = _pair_delta(Z, trz, ip, im, R, grp, seed, head, "A1_")
        a1 = P3.clone()
        a1[..., 1] += dlt
        v = a1[te].cpu().numpy()
        v[~ok[te]] = np.nan
        out["A1"] = v
        info.update(lam_A1=lam, edge_A1=edge)
    if {"A2", "A3"} & set(arms):
        X2 = torch.cat([Xe, Xi], 1).cpu()
        W2, lam2 = _mode_head(X2, lab.clip(0), tr, seq, seed, len(MODES))
        logit = (X2.double() @ W2[:-1] + W2[-1]).float().to(DEV)              # (n, 5)
        W2 = W2.float()[None]
        res_y = (Fy - Py).cpu().numpy()
        T = np.zeros((len(MODES), 20), np.float32)
        for c in range(len(MODES)):
            r = tr[lab[tr] == c]
            if len(r):
                T[c] = res_y[r].mean(0)
        Tt = torch.as_tensor(T, device=DEV)
        info.update(lam_A2=lam2, mode_counts_train=np.bincount(lab[tr], minlength=5).tolist())
        if head is not None:
            head.update(W2=W2[0].cpu().numpy(), templates=T)
        if "A2" in arms:
            a2 = P3.clone()
            a2[..., 1] += Tt[logit.argmax(1)]
            out["A2"] = a2[te].cpu().numpy()
            out["A2_mode"] = logit[te].argmax(1).cpu().numpy()
        if "A3" in arms:
            E1 = torch.eye(len(MODES), device=DEV)
            L = torch.as_tensor(lab, device=DEV)
            R = C_LOGIT * (E1[L[ip]] - E1[L[im]]) - (logit[ip] - logit[im])
            dl, lam3, edge3 = _pair_delta(Z, trz, ip, im, R, grp, seed, head, "A3_")
            lg3 = logit + dl
            a3 = P3.clone()
            a3[..., 1] += Tt[lg3.argmax(1)]
            v = a3[te].cpu().numpy()
            v[~ok[te]] = np.nan
            out["A3"] = v
            out["A3_mode"] = lg3[te].argmax(1).cpu().numpy()
            info.update(lam_A3=lam3, edge_A3=edge3)
    if "A4" in arms:
        A = traj.kmeans(F[tr], NH.K, seed=seed)
        xb = tr[(world[tr] == "x10") & p6.bypass_shape(fut[tr])]
        Ab = traj.kmeans(F[xb], 64, seed=seed) if len(xb) >= 64 else F[xb]
        A = torch.cat([A, Ab, wod_bypass_anchors(seed)], 0)
        K = len(A)
        ids = traj.nearest(F, A, 1)[0][:, 0]
        Axy = A.reshape(K, 20, 2).cpu().numpy()
        tgt = (ids[:, None], np.ones((len(ids), 1), np.float32))
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr, groups=seq[tr]))
        fit_r, sel_r = tr[a], tr[b]
        lam_e = N3._pick_cls(Xe, tgt, fit_r, sel_r, Axy, fut)
        Wce, _ = planner.ce_solve(Xe, tgt, tr, [lam_e], K)
        off = planner.linear_apply(Wce, Xe, np.arange(n))[0]
        inner = NH._group_folds(seq[tr], 5, seed=seed)
        for k in range(5):
            Wk, _ = planner.ce_solve(Xe, tgt, tr[inner != k], [lam_e], K)
            off[tr[inner == k]] = planner.linear_apply(Wk, Xe, tr[inner == k])[0]
        lam_l = N3._pick_cls(Xi, tgt, fit_r, sel_r, Axy, fut, off)
        Wl, _ = planner.ce_solve(Xi, tgt, tr, [lam_l], K, offset=off)
        top = planner.cls_topk(Wl, Xi, te, 1, offset=off)[:, 0, 0]
        out["A4"] = Axy[top]
        info.update(lam_A4_ego=lam_e, lam_A4_late=lam_l, K_A4=K, n_bypass_anchor_src=len(xb),
                    A4_bypass_anchor_pick=float(np.isin(top, np.arange(NH.K, K)).mean()))
        del Wce, off, Wl
    if rl is not None:
        rl.event("q2_fold", fold=int(f), model=m, seed=seed, stream=stream or "op", **{k: v for k, v in info.items()})
    torch.cuda.empty_cache()
    return {"te": te, "pred": out, "info": info}


# ---------------------------------------------------------------- pilot

def run(rl, set_: str = "carla_p6", split: str = "loco", seeds=SEEDS, models=MODELS, arms=ARMS, streams=(),
        tag: str = "") -> pd.DataFrame:
    """Fit every arm per (seed, model, fold), judge per arm, paired differences; predictions and tables to rl.dir."""
    D = load(set_, models, streams)
    n = len(D["t"])
    rows, diffs, infos = [], [], []
    runs = [(m, None) for m in models] + [("cinque", s) for s in streams]
    for seed in seeds:
        fold = fold_ids(D, split, seed)
        for m, stream in runs:
            a_run = arms if stream is None else tuple(a for a in arms if a in ("A1", "A3"))
            preds = {a: np.full((n, 20, 2), np.nan, np.float32) for a in a_run}
            for f in range(n_folds(split)):
                r = fit_fold(D, fold, f, m, seed, a_run, stream, rl=rl)
                for a in a_run:
                    preds[a][r["te"]] = r["pred"][a]
                infos.append({"seed": seed, "model": m, "stream": stream or "op", "fold": f, **r["info"]})
                rl.log.info("%s seed %d %s %s fold %d (%s): %s", split, seed, m, stream or "op", f,
                            CLASSES9[f] if split == "loco" else "route", json.dumps(r["info"], default=str)[:300])
            key = f"{split}_s{seed}_{m}" + (f"_{stream.split()[0]}" if stream else "")
            np.savez_compressed(rl.dir / f"preds_{key}.npz", **preds)
            scored = {}
            for a in a_run:
                row, per, s = J.judge_one(a, D["pairs"], preds[a], scopes=False)
                scored[a] = s
                rows.append({"split": split, "seed": seed, "model": m, "stream": stream or "op", **row})
                rl.log.info("  %s: flip %.3f [%.3f, %.3f] ff %.3f shoulder %.3f ref %.3f stop %.3f mirror %.3f -> %s", a,
                            row["bypass_flip"], row["lo"], row["hi"], row["null_ff_oos"], row["shoulder_flip"],
                            row["shoulder_ref"], row["stop_sub"], row["mirror_borrow"], row["verdict"])
            for a, b in (("A1", "A0"), ("A3", "A2"), ("A1", "A2"), ("A3", "A0"), ("A4", "A0")):
                if a in scored and b in scored:
                    d, lo, hi = J.paired_diff(scored[a], scored[b])
                    diffs.append({"split": split, "seed": seed, "model": m, "stream": stream or "op", "arm": a, "vs": b,
                                  "delta": d, "lo": lo, "hi": hi})
    tab, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    tab.to_csv(rl.dir / f"summary_{split}{tag}.csv", index=False)
    dif.to_csv(rl.dir / f"paired_{split}{tag}.csv", index=False)
    pd.DataFrame(infos).to_csv(rl.dir / f"folds_{split}{tag}.csv", index=False)
    return tab


def criteria(tab: pd.DataFrame, dif: pd.DataFrame) -> pd.DataFrame:
    """Q2's pre-registered readings per (split, model, stream) over the seeds."""
    out = []
    for (sp, m, st), g in tab.groupby(["split", "model", "stream"]):
        d = dif[(dif.split == sp) & (dif.model == m) & (dif.stream == st)]
        has = g.groupby("examinee").has_bypass.all()
        seeds_ok = lambda a, b: bool(len(d[(d.arm == a) & (d.vs == b)]) and (d[(d.arm == a) & (d.vs == b)].lo > 0).all())  # noqa: E731
        elicited = {a: bool(has.get(a, False)) and seeds_ok(a, "A0") and seeds_ok(a, "A2") for a in ("A1", "A3")}
        out.append({"split": sp, "model": m, "stream": st,
                    **{f"{a} passes gate (all seeds)": bool(has.get(a, False)) for a in ARMS if a in has},
                    "bypass elicited (A1 or A3)": any(elicited.values()), "via": ",".join(k for k, v in elicited.items() if v),
                    "vocabulary only (A4 passes, A0 / A2 not)": bool(has.get("A4", False)) and not has.get("A0", False)
                    and not has.get("A2", False),
                    "mirror > 50% (any arm, mean over seeds)": ",".join(a for a, x in g.groupby("examinee").mirror_borrow.mean().items()
                                                                     if x > 0.5)})
    return pd.DataFrame(out)


def choose(tab: pd.DataFrame) -> tuple[str, str]:
    """The closed-loop head ([C] entry): the passing trajectory arm (all seeds, LOCO, Cinque) with the highest mean
    bypass flip; A1 if none passes. The mode head for CL5d: A3 if it passes, else A2."""
    g = tab[(tab.split == "loco") & (tab.model == "cinque") & (tab.stream == "op")]
    agg = g.groupby("examinee").agg(ok=("has_bypass", "all"), flip=("bypass_flip", "mean"))
    ok = agg[agg.ok & agg.index.isin(["A0", "A1", "A2", "A3", "A4"])]
    arm = ok.flip.idxmax() if len(ok) else "A1"
    mode = "A3" if bool(agg.ok.get("A3", False)) else "A2"
    return arm, mode


# ---------------------------------------------------------------- closed-loop export

def export(rl, arm: str, mode_arm: str, set_: str = "carla_p6", model: str = "cinque", seed: int = 0,
           out_dir: Path | None = None) -> Path:
    """Refit on every v0 training world (all classes) and write the head: npz of parameters + manifest.json;
    inference is jevdrive.nq3_head.Head (numpy only)."""
    out_dir = out_dir or data_dir() / "runs" / "nq3" / "q2" / "closed_loop_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    D = load(set_, (model,))
    n = len(D["t"])
    fold = np.zeros(n, int)                                      # one fold: everything trains, nothing is held out
    head = {}
    if arm == "A4":
        raise NotImplementedError("A4 export (anchor vocabulary) is not wired; say so in the [C] entry and export A1")
    arms = tuple(sorted({"A0", arm, mode_arm}))
    chk = np.random.default_rng(0).choice(n, 1024, replace=False)
    r = fit_fold(D, fold, -1, model, seed, arms, None, head, rl, te_rows=chk)
    np.savez(out_dir / "head.npz", **{k: np.asarray(v) for k, v in head.items() if not isinstance(v, (float, int))})
    man = {"trajectory_arm": arm, "mode_arm": mode_arm, "model": model, "feature": f"openpilot {model} temporal (512, float32)",
           "fit": {"set": set_, "rows": int(r["info"]["n_train"]), "pairs": int(r["info"]["n_pairs"]), "seed": seed,
                   **{k: v for k, v in r["info"].items() if k.startswith("lam")}},
           "scalars": {k: v for k, v in head.items() if isinstance(v, (float, int))},
           "modes": list(MODES), "c_logit": C_LOGIT,
           "inputs": {"temporal": "openpilot `temporal` tap of the model's own 5 Hz stream from a zero state "
                                  "(scripts/p5_openpilot.py run_stream; Cinque: each frame held 4 steps of its 20 Hz clock)",
                      "ego": "p5_exam.ego_input: waymo.ego_state(past) (16 x (x, y, vx, vy, dvx, dvy), rear axle, current "
                             "frame, 0.25 s steps, p4_carla.route_rows convention) | one-hot intent (waymo.INTENTS)"},
           "output": {"trajectory": "(20, 2) future positions at t = 0.25 .. 5.0 s, rear-axle ego frame at the current "
                                    "tick, x forward, y left, metres (P5 / P6 future convention)",
                      "mode": "argmax over modes (keep, stop, bypass_L, bypass_R, wait) of the mode head",
                      "rate": "one plan per 5 Hz camera frame"},
           "loader": "from jevdrive.nq3_head import Head; h = Head(dir); traj, mode = h(temporal, ego_input)",
           "written": pd.Timestamp.now().isoformat()}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    # rule 8: the numpy loader against the in-process fit on 1024 rows
    from .nq3_head import Head
    h = Head(out_dir)
    tr_, md_ = h(D["op"][model][chk].cpu().numpy(), D["Ego"][chk].cpu().numpy())
    dt = float(np.abs(tr_ - r["pred"][arm]).max())
    dm = float((md_ != r["pred"][f"{mode_arm}_mode"]).mean())
    man["check"] = {"rows": len(chk), "traj_max_abs_diff_m": dt, "mode_mismatch_share": dm}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    rl.log.info("export check: trajectory max |diff| %.2e m, mode mismatch %.4f", dt, dm)
    assert dt <= 1e-3 and dm <= 1e-3, "exported head does not reproduce the fit"
    return out_dir


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("labels", "pilot", "controls", "export"))
    ap.add_argument("--set", default="carla_p6")
    ap.add_argument("--split", default="loco")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--arm", default="")
    ap.add_argument("--mode-arm", default="")
    ap.add_argument("--run", default="", help="pilot run dir (export: choose the arm from its summary)")
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    if a.cmd == "labels":
        D = load(a.set, ("cinque",))
        print(np.bincount(D["lab"][D["lab"] >= 0], minlength=5))
        return
    rl = RunLog("nq3_c", f"q2-{a.cmd}-{a.set}-{a.split}")
    if a.cmd == "pilot":
        run(rl, a.set, a.split, seeds, MODELS, tuple(a.arms.split(",")))
    elif a.cmd == "controls":
        run(rl, a.set, a.split, seeds, (), ("A1", "A3"), ("qwen L18_last", "vjepa2 mean"), tag="_controls")
    else:
        arm, mode = a.arm, a.mode_arm
        if not arm:
            arm, mode = choose(pd.read_csv(Path(a.run) / f"summary_loco.csv"))
        d = export(rl, arm, mode, a.set)
        rl.log.info("exported %s (trajectory %s, mode %s)", d, arm, mode)
    rl.close()


if __name__ == "__main__":
    main()
