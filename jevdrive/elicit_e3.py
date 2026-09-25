"""E3 feasibility: twin frames mined from logs (todos/2026-09-26-elicitation-program.md, E3, and the [E3] entry of the
deviation log, written before any mining). No training.

  twin pair   two frames of different logs / sequences (and, on navtrain, not the same road section: same map and
              global distance < 30 m) whose standardised ego-history vectors are nearest neighbours inside the same
              intent / command stratum, at distance <= m * tau_ego (m in {0.5, 1, 2}); tau_ego = p95 of the distance
              between temporally adjacent frames of one log / sequence
  divergent   |dv_T| >= th_v or |dy_T| >= th_y (terminal speed over the last 1 s, terminal lateral offset in the own
              t0 frame), grid {(1, .5), (2, 1), (3, 2)}; twin null: |dv_T| <= 0.5 and |dy_T| <= 0.3
  x+ side     the side whose future departs more (ADE) from its own constant-velocity continuation
  cause       navtrain: a GT agent in the corridor (own logged path extended to 30 m, +-1.5 m, 0 < s <= 30 m) with
              closing speed < -0.5 m/s; WOD (val 20 237-frame SAM subset only): a SAM detection in the same corridor

    python -m jevdrive.elicit_e3 extract navtrain     # per-token GT agents + global pose from the logs (cached)
    python -m jevdrive.elicit_e3 run                  # mining, statistics, tables
    python -m jevdrive.elicit_e3 scorer-prep <run>    # PDM scorer subsample: token list and the two proposals
    python -m jevdrive.elicit_e3 scorer-read <run>    # read the devkit csvs back
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger, n_cpus

log = get_logger(__name__)
HALF_W, REACH, CLOSE = 1.5, 30.0, -0.5
TAU_MULT = (0.5, 1.0, 2.0)
DIV_GRID = ((1.0, 0.5), (2.0, 1.0), (3.0, 2.0))
MAIN = (1.0, (2.0, 1.0))
NULL_V, NULL_Y = 0.5, 0.3
SECTION_M = 30.0
NB = 2000
CLASSES = ("vehicle", "pedestrian", "bicycle", "other")
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "elicitation" / "e3"
CACHE = data_dir() / "runs" / "elicitation" / "e3-cache"
WORKERS = min(20, n_cpus())


# ---------------------------------------------------------------- corridor (shared with E1's NAVSIM grouping)

def extend(path: np.ndarray, reach: float = REACH) -> np.ndarray:
    """Polyline continued along its last heading until its arc length reaches `reach` (a standing ego: +x)."""
    seg = np.diff(path, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    good = ln > 0.05
    h = seg[good][-1] / ln[good][-1] if good.any() else np.array([1.0, 0.0])
    left = reach - ln.sum()
    return np.r_[path, [path[-1] + h * left]] if left > 0 else path


def corridor_objects(fut_xy: np.ndarray, boxes: np.ndarray, vel: np.ndarray, v_ego: np.ndarray,
                     half_w: float = HALF_W, reach: float = REACH):
    """Per object (n,): in the corridor of the logged path (origin + future points, extended to `reach` m), and the
    closing speed (unit position vector . (v_obj - v_ego)). boxes (n, >=5): x, y, z, l, w[, h, yaw] ego frame."""
    from .fusion_diag import project
    if not len(boxes):
        return np.zeros(0, bool), np.zeros(0)
    path = extend(np.r_[[[0.0, 0.0]], fut_xy[:, :2]], reach)
    x, y, l, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6] if boxes.shape[1] > 6 else np.zeros(len(boxes))
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [np.c_[x, y]]
    for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        pts.append(np.c_[x + a * l / 2 * c - b * w / 2 * s, y + a * l / 2 * s + b * w / 2 * c])
    P = np.concatenate(pts)
    sa, d, _, _ = project(path, P)
    ok = ((np.abs(d) <= half_w) & (sa > 0) & (sa <= reach)).reshape(5, -1).any(0)
    r = np.c_[x, y]
    rn = r / np.linalg.norm(r, axis=1, keepdims=True).clip(1e-6)
    closing = (rn * (vel[:, :2] - v_ego[None, :2])).sum(1)
    return ok, closing


# ---------------------------------------------------------------- NAVSIM log extraction

def _cls(names):
    return np.array([CLASSES.index(n) if n in CLASSES[:3] else 3 for n in names], np.int8)


def _log_worker(args):
    path, toks = args
    with open(path, "rb") as f:
        L = pickle.load(f)
    out = {}
    for fr in L:
        if fr["token"] not in toks:
            continue
        a = fr["anns"]
        out[fr["token"]] = dict(t=fr["timestamp"] / 1e6, gxy=np.asarray(fr["ego2global_translation"][:2], np.float64),
                                map=fr["map_location"], v_ego=np.asarray(fr["ego_dynamic_state"][:2], np.float32),
                                boxes=np.asarray(a["gt_boxes"], np.float32), vel=np.asarray(a["gt_velocity_3d"], np.float32)[:, :2],
                                cls=_cls(a["gt_names"]), cone=np.asarray([n == "traffic_cone" for n in a["gt_names"]]))
    return out


def extract(split: str) -> dict:
    """token -> GT agents at t0, global xy, timestamp (cached per split)."""
    fp = CACHE / f"{split}_agents.pkl"
    if fp.exists():
        with open(fp, "rb") as f:
            return pickle.load(f)
    from . import navsim_zs as Z
    idx = Z.load_index(split, slim=True)
    by = {}
    for e in idx:
        by.setdefault(e["log_name"], set()).add(e["token"])
    sub = "test" if split in ("navtest",) else "trainval"
    base = data_dir() / "datasets" / "navsim" / "navsim_logs" / sub
    jobs = [(base / f"{ln}.pkl", toks) for ln, toks in by.items()]
    out = {}
    with Pool(WORKERS) as p:
        for i, d in enumerate(p.imap_unordered(_log_worker, jobs, chunksize=4)):
            out.update(d)
            if i % 100 == 0:
                log.info("extract %s: %d / %d logs", split, i, len(jobs))
    miss = len(idx) - len(out)
    assert miss == 0, f"{miss} tokens not found in the logs"
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(fp, "wb") as f:
        pickle.dump(out, f, protocol=5)
    return out


# ---------------------------------------------------------------- mining

def nearest(X: np.ndarray, strat: np.ndarray, group: np.ndarray, gxy=None, gmap=None, chunk: int = 4096):
    """For every row the nearest row in the same stratum, excluding the same group and (if gxy) the same road
    section (same map and global distance < SECTION_M). Returns (index, distance), -1 / inf when none."""
    torch.set_num_threads(WORKERS)
    n = len(X)
    nn, dist = np.full(n, -1), np.full(n, np.inf)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for s in np.unique(strat):
        rows = np.flatnonzero(strat == s)
        R = Xt[rows]
        r2 = (R * R).sum(1)
        g = torch.as_tensor(group[rows])
        if gxy is not None:
            G = torch.as_tensor(gxy[rows], dtype=torch.float64)
            M = torch.as_tensor(gmap[rows])
        for a in range(0, len(rows), chunk):
            q = slice(a, a + chunk)
            D = (r2[q, None] + r2[None] - 2 * R[q] @ R.T).clamp_min(0)
            bad = g[q, None] == g[None]
            if gxy is not None:
                bad |= (M[q, None] == M[None]) & (torch.cdist(G[q], G) < SECTION_M)
            D[bad] = float("inf")
            d, j = D.min(1)
            ok = torch.isfinite(d)
            nn[rows[a:a + chunk][ok.numpy()]] = rows[j[ok].numpy()]
            dist[rows[a:a + chunk]] = d.sqrt().numpy()
    return nn, dist


def standardise(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 1e-6, sd, 1.0)


def tau_ego(Xs: np.ndarray, group: np.ndarray, t: np.ndarray, gap: float, tol: float) -> float:
    """p95 of the ego-vector distance between frames of the same group `gap` apart (+- tol)."""
    o = np.lexsort((t, group))
    g, tt = group[o], t[o]
    d = []
    for k in (1, 2, 3, 4, 5):
        same = g[k:] == g[:-k]
        dt = tt[k:] - tt[:-k]
        m = same & (np.abs(dt - gap) <= tol)
        if m.any():
            d.append(np.linalg.norm(Xs[o[k:][m]] - Xs[o[:-k][m]], axis=1))
    return float(np.quantile(np.concatenate(d), 0.95))


def future_stats(fut: np.ndarray, dt: float, v0: np.ndarray):
    """Terminal speed (mean over the last 1 s), terminal lateral offset, ADE to the constant-velocity continuation."""
    k = int(round(1.0 / dt))
    vT = np.linalg.norm(fut[:, -1, :2] - fut[:, -1 - k, :2], axis=1)
    yT = fut[:, -1, 1]
    tt = dt * np.arange(1, fut.shape[1] + 1)
    cv = np.stack([v0[:, None] * tt, np.zeros_like(v0[:, None] * tt)], -1)
    dep = np.linalg.norm(fut[..., :2] - cv, axis=-1).mean(1)
    return vT, yT, dep


def pairs_table(nn, dist, tau, vT, yT, dep, group):
    """Unique unordered pairs (i < j) of mutual-or-one-way nearest neighbours, with the side assignment."""
    i = np.flatnonzero(nn >= 0)
    j = nn[i]
    a, b = np.minimum(i, j), np.maximum(i, j)
    P = pd.DataFrame({"a": a, "b": b, "d": dist[i]}).drop_duplicates(["a", "b"]).reset_index(drop=True)
    a, b = P.a.to_numpy(), P.b.to_numpy()
    P["d_tau"] = P.d / tau
    P["dv"], P["dy"] = np.abs(vT[a] - vT[b]), np.abs(yT[a] - yT[b])
    plus_a = dep[a] >= dep[b]
    P["xp"], P["xm"] = np.where(plus_a, a, b), np.where(plus_a, b, a)
    P["grp"], P["grp_m"] = group[P.xp.to_numpy()], group[P.xm.to_numpy()]
    P["null"] = (P.dv <= NULL_V) & (P.dy <= NULL_Y)
    return P


def div(P, th):
    return (P.dv >= th[0]) | (P.dy >= th[1])


def boot_diff(v: np.ndarray, grp: np.ndarray, hi: np.ndarray, lo: np.ndarray, b: int = NB, seed: int = 0):
    """mean(v[hi]) - mean(v[lo]) with a joint cluster bootstrap over `grp` (percentile 95 % CI)."""
    codes, u = pd.factorize(grp)
    k = len(u)
    sh, nh = np.bincount(codes[hi], v[hi], k), np.bincount(codes[hi], minlength=k).astype(float)
    sl, nl = np.bincount(codes[lo], v[lo], k), np.bincount(codes[lo], minlength=k).astype(float)
    W = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=b).astype(float)
    m = (W @ sh) / (W @ nh).clip(1e-9) - (W @ sl) / (W @ nl).clip(1e-9)
    d = v[hi].mean() - v[lo].mean()
    return float(d), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_mean(v, grp, b=NB, seed=0):
    codes, u = pd.factorize(grp)
    k = len(u)
    s, n = np.bincount(codes, v, k), np.bincount(codes, minlength=k).astype(float)
    W = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=b).astype(float)
    m = (W @ s) / (W @ n).clip(1e-9)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def grid(P, n_pool) -> pd.DataFrame:
    rows = []
    for m in TAU_MULT:
        Q = P[P.d_tau <= m]
        for th in DIV_GRID:
            rows.append({"tau_mult": m, "th_v": th[0], "th_y": th[1], "pool": n_pool, "pairs": len(Q),
                         "divergent": int(div(Q, th).sum()), "twin_null": int(Q.null.sum()),
                         "divergent_group_pairs": int(Q[div(Q, th)].pipe(lambda x: x.grp.astype(str) + "|" + x.grp_m.astype(str)).nunique())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- navtrain

def navtrain(rl):
    from . import navsim_heads as H, navsim_zs as Z
    idx = Z.load_index("navtrain", slim=True)
    tok = np.array([e["token"] for e in idx])
    lg = np.array([e["log_name"] for e in idx])
    ego = np.stack([np.r_[e["pose"].ravel(), e["vel"].ravel(), e["acc"].ravel()] for e in idx]).astype(np.float32)
    cmd = np.array([int(np.argmax(e["cmd"][-1])) for e in idx])
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        pos = dict(zip(f["tokens"].tolist(), range(len(f["tokens"]))))
        fut = f["poses"][[pos[t] for t in tok]].astype(np.float32)
    ag = extract("navtrain")
    t = np.array([ag[k]["t"] for k in tok])
    gxy = np.stack([ag[k]["gxy"] for k in tok])
    gmap = pd.factorize(np.array([ag[k]["map"] for k in tok]))[0]
    # velocity sanity: are gt velocities absolute (static cones ~0 while ego moves)?
    cone_v = np.concatenate([np.linalg.norm(ag[k]["vel"][ag[k]["cone"]], axis=1) for k in tok
                             if np.linalg.norm(ag[k]["v_ego"]) > 5 and ag[k]["cone"].any()])
    rl.event("velocity_check", cone_speed_median=float(np.median(cone_v)), n=len(cone_v))
    assert np.median(cone_v) < 0.3, f"gt velocities look relative: cone median {np.median(cone_v):.2f}"
    Xs = standardise(ego)
    grp = pd.factorize(lg)[0]
    tau = tau_ego(Xs, grp, t, 0.5, 0.05)
    v0 = np.linalg.norm(np.stack([e["vel"][-1] for e in idx]), axis=1)
    vT, yT, dep = future_stats(fut, 0.5, v0)
    nn, dist = nearest(Xs, cmd, grp, gxy, gmap)
    P = pairs_table(nn, dist, tau, vT, yT, dep, lg)
    nn0, dist0 = nearest(Xs, cmd, grp)                     # descriptive: road-section exclusion off
    P0 = pairs_table(nn0, dist0, tau, vT, yT, dep, lg)
    rl.log.info("navtrain: tau_ego %.3f, %d pairs within 2 tau (%d without the section rule)", tau,
                (P.d_tau <= 2).sum(), (P0.d_tau <= 2).sum())
    # cause objects on every token
    flags = cause_flags_nav(tok, fut, ag)
    return dict(name="navtrain", P=P, P0=P0, tau=tau, n_pool=len(tok), flags=flags, tok=tok, cmd=cmd, v0=v0,
                vT=vT, dep=dep, fut=fut)


def _flag_worker(args):
    fut, boxes, vel, cls, v_ego = args
    ok, cl = corridor_objects(fut, boxes, vel, v_ego)
    cause = ok & (cl < CLOSE)
    r = {"in_any": ok.any(), "cause": cause.any()}
    for c, name in enumerate(CLASSES):
        r[f"cause_{name}"] = (cause & (cls == c)).any()
        r[f"in_{name}"] = (ok & (cls == c)).any()
    return r


def cause_flags_nav(tok, fut, ag) -> pd.DataFrame:
    jobs = [(fut[i], ag[k]["boxes"], ag[k]["vel"], ag[k]["cls"], ag[k]["v_ego"]) for i, k in enumerate(tok)]
    with Pool(WORKERS) as p:
        rows = p.map(_flag_worker, jobs, chunksize=256)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- WOD

def wod(rl):
    from . import waymo as W
    df = W.load_index()
    past, future = W.load_ego()
    names = W.frame_names(df)
    sub = W.subsets(df, past, future)
    Xall = past.reshape(len(past), -1).astype(np.float32)
    v0 = np.linalg.norm(past[:, -1, 2:4], axis=1)
    vT, yT, dep = future_stats(future, 0.25, v0)
    seq = df.sequence.to_numpy()
    fr = df.frame.to_numpy()
    tr = (df.split == "train").to_numpy() & df.has_future.to_numpy()
    out = {"name": "wod"}
    # train pool at 2 Hz (main) and at 10 Hz (descriptive)
    for tag, m in (("2hz", tr & (fr % 5 == 0)), ("10hz", tr)):
        rows = np.flatnonzero(m)
        Xs = standardise(Xall[rows])
        g = pd.factorize(seq[rows])[0]
        tau = tau_ego(Xs, g, fr[rows] / 10.0, 0.3, 0.01) if tag == "10hz" else None
        out[f"rows_{tag}"] = rows
        out[f"Xs_{tag}"] = Xs
        out[f"g_{tag}"] = g
        out[f"tau_{tag}"] = tau
    # tau_ego is defined on consecutive frames 0.3 s apart, so it comes from the full-rate pool; the 2 Hz pool is
    # standardised with its own statistics, so recompute on the 2 Hz rows' standardisation: frames 3 apart of the
    # same sequence, standardised with the 2 Hz pool's mean / sd
    r2, r10 = out["rows_2hz"], out["rows_10hz"]
    X2 = Xall[r2]
    mu, sd = X2.mean(0), np.where(X2.std(0) > 1e-6, X2.std(0), 1.0)
    Xs10_in2 = (Xall[r10] - mu) / sd
    tau2 = tau_ego(Xs10_in2, out["g_10hz"], fr[r10] / 10.0, 0.3, 0.01)
    out["tau_2hz"] = tau2
    intent = df.intent.to_numpy()
    for tag in ("2hz", "10hz"):
        rows = out[f"rows_{tag}"]
        nn, dist = nearest(out[f"Xs_{tag}"], intent[rows], out[f"g_{tag}"])
        P = pairs_table(nn, dist, out[f"tau_{tag}"], vT[rows], yT[rows], dep[rows], seq[rows])
        for c in ("a", "b", "xp", "xm"):
            P[c] = rows[P[c].to_numpy()]
        out[f"P_{tag}"] = P
        rl.log.info("wod %s: tau %.3f, %d rows, %d pairs within 2 tau", tag, out[f"tau_{tag}"], len(rows),
                    (P.d_tau <= 2).sum())
    # val SAM subset: its own mining pass (pool = the 20 237 Q2b frames), cause = SAM detection in the corridor
    from . import fusion_q4 as Q
    lst = pd.read_parquet(Q.root("lists") / "wod.parquet")
    at = pd.Series(np.arange(len(df)), index=names)
    vr = at[lst.key].to_numpy()
    vr = vr[df.has_future.to_numpy()[vr]]
    Xs = standardise(Xall[vr])
    gv = pd.factorize(seq[vr])[0]
    tauv = tau_ego((Xall[out["rows_10hz"]] - Xall[vr].mean(0)) / np.where(Xall[vr].std(0) > 1e-6, Xall[vr].std(0), 1),
                   out["g_10hz"], fr[out["rows_10hz"]] / 10.0, 0.3, 0.01)
    nn, dist = nearest(Xs, intent[vr], gv)
    Pv = pairs_table(nn, dist, tauv, vT[vr], yT[vr], dep[vr], seq[vr])
    for c in ("a", "b", "xp", "xm"):
        Pv[c] = vr[Pv[c].to_numpy()]
    out.update(P_val=Pv, tau_val=tauv, n_val=len(vr), sam=sam_flags(df, names, future, lst), sub=sub,
               cluster=df.cluster.to_numpy(), v0=v0, names=names)
    return out


def sam_flags(df, names, future, lst) -> pd.DataFrame:
    """Per frame of the Q2b list: a SAM detection (score > 0.5, flat-ground lift) of each class group in the
    corridor (logged 5 s path extended to 30 m, +-1.5 m, 0 < s <= 30 m)."""
    from . import fusion_q4 as Q
    from .fusion_diag import project
    from .fusion_q2b import calibs
    cal = calibs()
    d = Q.load_dets(Q.root("sam", "wod"))
    d = Q.lift_dets(d, d.key.map(lst.set_index("key").sequence).to_numpy(), cal)
    d = d[d.lift_ok]
    at = pd.Series(np.arange(len(df)), index=names)
    groups = {"pedestrian": ("pedestrian",), "cyclist": ("cyclist",), "vehicle": ("vehicle",),
              "cone|debris": ("cone", "debris"), "emergency vehicle": ("emergency vehicle",)}
    by = d.groupby("key")
    rows = []
    for key in lst.key:
        r = {"key": key, **{f"in_{g}": False for g in groups}}
        if key in by.groups:
            g = by.get_group(key)
            path = extend(np.r_[[[0.0, 0.0]], future[at[key], :, :2]])
            pts = g[["gx", "gy"]].to_numpy(float)
            s, dl, _, _ = project(path, pts)
            near = (np.abs(dl) <= HALF_W) & (s > 0) & (s <= REACH)
            hit = set(g.prompt.to_numpy()[near])
            for grp, mem in groups.items():
                r[f"in_{grp}"] = bool(hit & set(mem))
        rows.append(r)
    f = pd.DataFrame(rows)
    f["in_any"] = f[[c for c in f if c.startswith("in_")]].any(axis=1)
    f["row"] = at[f.key].to_numpy()
    return f


# ---------------------------------------------------------------- statistics

def cause_contrast(P, flag: np.ndarray, grp_col="grp", th=MAIN[1], m=MAIN[0]) -> dict:
    """Share of x+ frames with a cause object: divergent vs twin-null pairs (main cell), and x+ vs x- inside the
    divergent pairs."""
    Q = P[P.d_tau <= m]
    dv, nl = div(Q, th).to_numpy(), Q.null.to_numpy()
    fp, fm = flag[Q.xp.to_numpy()].astype(float), flag[Q.xm.to_numpy()].astype(float)
    g = Q[grp_col].astype(str).to_numpy()
    r = {"n_div": int(dv.sum()), "n_null": int(nl.sum())}
    r["div_xplus"], r["null_xplus"] = float(fp[dv].mean()) if dv.any() else np.nan, float(fp[nl].mean()) if nl.any() else np.nan
    r["div_xminus"] = float(fm[dv].mean()) if dv.any() else np.nan
    if dv.any() and nl.any():
        r["diff"], r["diff_lo"], r["diff_hi"] = boot_diff(fp, g, dv, nl)
    if dv.any():
        r["paired"], r["paired_lo"], r["paired_hi"] = boot_mean((fp - fm)[dv], g[dv])
    return r


def main_nav(rl, N):
    P, fl = N["P"], N["flags"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    g = grid(P, N["n_pool"])
    g0 = grid(N["P0"], N["n_pool"]).rename(columns=lambda c: c + "_nosection" if c in ("pairs", "divergent", "twin_null") else c)
    g = g.merge(g0[["tau_mult", "th_v", "th_y", "pairs_nosection", "divergent_nosection", "twin_null_nosection"]])
    g.insert(0, "tau_ego", N["tau"])
    g.to_csv(RESULTS / "navtrain_grid.csv", index=False)
    rows = []
    for col in ["cause", "in_any"] + [f"cause_{c}" for c in CLASSES] + [f"in_{c}" for c in CLASSES]:
        for m in TAU_MULT:
            r = cause_contrast(P, fl[col].to_numpy(), m=m)
            rows.append({"flag": col, "tau_mult": m, **r})
    C = pd.DataFrame(rows)
    C.to_csv(RESULTS / "navtrain_cause.csv", index=False)
    # overlaps (main cell, divergent pairs, x+ side)
    Q = P[(P.d_tau <= MAIN[0]) & div(P, MAIN[1])]
    xp = Q.xp.to_numpy()
    from . import navsim_zs as Z
    hard = {e["token"] for e in Z.load_index("navhard_two_stage", slim=True)}
    ov = {"divergent": len(Q), "ped_or_bicycle_in_corridor": fl.in_pedestrian.to_numpy()[xp].mean() + 0,
          "ped_or_bicycle_in_corridor_any": (fl.in_pedestrian | fl.in_bicycle).to_numpy()[xp].mean(),
          "stationary_start_xplus": (N["v0"][xp] < 0.5).mean(),
          "xplus_slower": (N["vT"][xp] < N["vT"][Q.xm.to_numpy()]).mean(),
          "lateral_only": ((Q.dv < MAIN[1][0]) & (Q.dy >= MAIN[1][1])).mean(),
          "navhard_token_overlap": int(np.isin(N["tok"][np.r_[xp, Q.xm.to_numpy()]], list(hard)).sum())}
    for c, name in enumerate(("left", "straight", "right", "unknown")):
        ov[f"cmd_{c}"] = (N["cmd"][xp] == c).mean()
    O = pd.DataFrame([ov])
    O.to_csv(RESULTS / "navtrain_overlap.csv", index=False)
    Q.assign(tok_xp=N["tok"][xp], tok_xm=N["tok"][Q.xm.to_numpy()]).to_parquet(rl.dir / "navtrain_divergent_main.parquet")
    P.to_parquet(rl.dir / "navtrain_pairs.parquet")
    fl.assign(token=N["tok"]).to_parquet(rl.dir / "navtrain_flags.parquet")
    rl.log.info("navtrain grid\n%s", g.to_markdown(index=False))
    rl.log.info("navtrain cause\n%s", C[C.tau_mult == 1].to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("navtrain overlap\n%s", O.T.to_markdown(floatfmt=".3f"))
    return g, C, O


def main_wod(rl, Wd):
    RESULTS.mkdir(parents=True, exist_ok=True)
    g = pd.concat([grid(Wd["P_2hz"], len(Wd["rows_2hz"])).assign(pool="train 2 Hz", tau_ego=Wd["tau_2hz"]),
                   grid(Wd["P_10hz"], len(Wd["rows_10hz"])).assign(pool="train 10 Hz", tau_ego=Wd["tau_10hz"]),
                   grid(Wd["P_val"], Wd["n_val"]).assign(pool="val SAM subset", tau_ego=Wd["tau_val"])])
    g.to_csv(RESULTS / "wod_grid.csv", index=False)
    sam = Wd["sam"]
    n = len(Wd["names"])
    rows = []
    for col in [c for c in sam if c.startswith("in_")]:
        flag = np.zeros(n, bool)
        flag[sam.row.to_numpy()] = sam[col].to_numpy()
        for m in TAU_MULT:
            rows.append({"flag": col, "tau_mult": m, **cause_contrast(Wd["P_val"], flag, m=m)})
    C = pd.DataFrame(rows)
    C.to_csv(RESULTS / "wod_cause_val.csv", index=False)
    sub, cl = Wd["sub"], Wd["cluster"]
    ov = []
    for tag in ("P_2hz", "P_val"):
        P = Wd[tag]
        Q = P[(P.d_tau <= MAIN[0]) & div(P, MAIN[1])]
        xp = Q.xp.to_numpy()
        r = {"pool": tag, "divergent": len(Q)}
        for k in ("pre_onset", "straight_yaw", "turn_yaw"):
            r[k] = sub[k][xp].mean()
        r["stationary_start_xplus"] = (Wd["v0"][xp] < 0.5).mean()
        if tag == "P_val":
            for c in ("Pedestrian", "Cyclist", "Cut_ins", "Foreign Object Debris", "Interections"):
                r[f"cluster_{c}"] = (cl[xp] == c).mean()
        ov.append(r)
    O = pd.DataFrame(ov)
    O.to_csv(RESULTS / "wod_overlap.csv", index=False)
    for tag in ("P_2hz", "P_val"):
        Wd[tag].to_parquet(rl.dir / f"wod_{tag[2:]}_pairs.parquet")
    rl.log.info("wod grid\n%s", g.to_markdown(index=False))
    rl.log.info("wod cause (val)\n%s", C[C.tau_mult == 1].to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("wod overlap\n%s", O.T.to_markdown(floatfmt=".3f"))
    return g, C, O


def features(run_dir: Path) -> pd.DataFrame:
    """Divergent main-cell pairs with both sides' features on disk now, and the Qwen extraction still needed."""
    import glob
    from . import navsim_zs as Z, waymo as W
    rows = []
    with np.load(Z.root("openpilot", "navtrain") / "cinque_temporal.npz") as z:
        op = pd.Index(z["tokens"].astype(str))
    Q = pd.read_parquet(run_dir / "navtrain_divergent_main.parquet")
    both_op = op.get_indexer(Q.tok_xp) >= 0
    both_op &= op.get_indexer(Q.tok_xm) >= 0
    uniq = len(set(Q.tok_xp) | set(Q.tok_xm))
    rows.append({"dataset": "navtrain", "divergent": len(Q), "both_op_temporal": int(both_op.sum()), "both_qwen": 0,
                 "frames_needing_qwen": uniq, "gpu_h_lo": uniq * 0.33 / 3600, "gpu_h_hi": uniq * 0.40 / 3600})
    D = data_dir() / "processed" / "waymo_e2e" / "features"
    names = pd.Series(W.frame_names(W.load_index()))
    qn = pd.concat([pd.read_parquet(f, columns=["frame_name"]) for f in glob.glob(str(D / "qwenvid_train_t4" / "*" / "index.parquet"))]).frame_name
    on = pd.read_parquet(D / "op_cinque_p3_trainval" / "index.parquet", columns=["frame_name"]).frame_name
    hq, ho = names.isin(qn).to_numpy(), names.isin(on).to_numpy()
    P = pd.read_parquet(run_dir / "wod_2hz_pairs.parquet")
    Q = P[(P.d_tau <= MAIN[0]) & div(P, MAIN[1])]
    a, b = Q.xp.to_numpy(), Q.xm.to_numpy()
    need = len(set(a[~hq[a]]) | set(b[~hq[b]]))
    rows.append({"dataset": "wod train 2 Hz", "divergent": len(Q), "both_op_temporal": int((ho[a] & ho[b]).sum()),
                 "both_qwen": int((hq[a] & hq[b]).sum()), "frames_needing_qwen": need,
                 "gpu_h_lo": need * 0.33 / 3600, "gpu_h_hi": need * 0.40 / 3600})
    F = pd.DataFrame(rows)
    F.to_csv(RESULTS / "features.csv", index=False)
    print(F.to_markdown(index=False, floatfmt=".2f"))
    return F


# ---------------------------------------------------------------- post-hoc descriptive (not part of the criterion)

def posthoc(run_dir: Path):
    """Longitudinal brake-vs-continue pairs on navtrain: |dv_T| >= 2, |dy_T| < 1, both t0 speeds >= 2 m/s (main tau).
    Cause-object share on the slower (braking) side vs the faster side (paired), and vs the twin nulls restricted to
    the same speed condition. Written after the pre-registered navtrain numbers were seen; descriptive only."""
    from . import navsim_zs as Z
    idx = Z.load_index("navtrain", slim=True)
    tok = np.array([e["token"] for e in idx])
    v0 = np.linalg.norm(np.stack([e["vel"][-1] for e in idx]), axis=1)
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        pos = dict(zip(f["tokens"].tolist(), range(len(f["tokens"]))))
        fut = f["poses"][[pos[t] for t in tok]].astype(np.float32)
    vT, _, _ = future_stats(fut, 0.5, v0)
    P = pd.read_parquet(run_dir / "navtrain_pairs.parquet")
    fl = pd.read_parquet(run_dir / "navtrain_flags.parquet")
    P = P[P.d_tau <= MAIN[0]]
    a, b = P.a.to_numpy(), P.b.to_numpy()
    moving = (v0[a] >= 2) & (v0[b] >= 2)
    lon = moving & (P.dv.to_numpy() >= 2) & (P.dy.to_numpy() < 1)
    nul = moving & P.null.to_numpy()
    slow = np.where(vT[a] <= vT[b], a, b)
    fast = np.where(vT[a] <= vT[b], b, a)
    rows = []
    for col in ("cause", "cause_vehicle", "cause_pedestrian", "in_pedestrian", "in_any"):
        f = fl[col].to_numpy().astype(float)
        g = P.grp.astype(str).to_numpy()
        fs, ff = f[slow], f[fast]
        pd_, plo, phi = boot_mean((fs - ff)[lon], g[lon])
        both = np.r_[fs[lon], fs[nul]]
        gg = np.r_[g[lon], g[nul]]
        dd, dlo, dhi = boot_diff(both, gg, np.r_[np.ones(lon.sum(), bool), np.zeros(nul.sum(), bool)],
                                 np.r_[np.zeros(lon.sum(), bool), np.ones(nul.sum(), bool)])
        rows.append({"flag": col, "n_long": int(lon.sum()), "n_null_moving": int(nul.sum()), "brake_side": fs[lon].mean(),
                     "continue_side": ff[lon].mean(), "null_moving": fs[nul].mean(), "brake_minus_continue": pd_,
                     "bmc_lo": plo, "bmc_hi": phi, "brake_minus_null": dd, "bmn_lo": dlo, "bmn_hi": dhi})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "navtrain_posthoc_longitudinal.csv", index=False)
    print(R.to_markdown(index=False, floatfmt=".3f"))


# ---------------------------------------------------------------- PDM scorer subsample

N_SCORER, DECEL, SEED = 300, 3.0, 0


def _retime(path_xy: np.ndarray, s_t: np.ndarray) -> np.ndarray:
    """Poses (x, y, yaw) at arc lengths s_t along `path_xy` (extended far along its last heading)."""
    p = extend(path_xy, max(200.0, float(s_t.max()) + 10))
    seg = np.diff(p, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    keep = np.r_[True, ln > 1e-4]
    p = p[keep]
    seg = np.diff(p, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(ln)]
    x, y = np.interp(s_t, cum, p[:, 0]), np.interp(s_t, cum, p[:, 1])
    hd = np.arctan2(seg[:, 1], seg[:, 0])
    j = np.clip(np.searchsorted(cum, s_t, side="right") - 1, 0, len(hd) - 1)
    yaw = np.where(s_t < 1e-3, 0.0, hd[j])
    return np.c_[x, y, yaw].astype(np.float32)


def scorer_prep(run_dir: Path):
    Q = pd.read_parquet(run_dir / "navtrain_divergent_main.parquet")
    from . import navsim_zs as Z
    idx = {e["token"]: e for e in Z.load_index("navtrain", slim=True)}
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        fut = dict(zip(f["tokens"].tolist(), f["poses"]))
    v0 = lambda t: float(np.linalg.norm(idx[t]["vel"][-1]))  # noqa: E731
    ok = np.array([v0(a) >= 2 and v0(b) >= 2 for a, b in zip(Q.tok_xp, Q.tok_xm)])
    S = Q[ok].sample(n=min(N_SCORER, int(ok.sum())), random_state=SEED)
    toks = sorted(set(S.tok_xp) | set(S.tok_xm))
    tt = 0.5 * np.arange(1, 9)
    cont, brake = [], []
    for t in toks:
        path = np.r_[[[0.0, 0.0]], fut[t][:, :2]]
        v = v0(t)
        cont.append(_retime(path, v * tt))
        ts = np.minimum(tt, v / DECEL)
        brake.append(_retime(path, v * ts - 0.5 * DECEL * ts ** 2))
    d = run_dir / "scorer"
    d.mkdir(exist_ok=True)
    S.to_parquet(d / "pairs.parquet")
    (d / "tokens.txt").write_text("\n".join(toks) + "\n")
    np.savez(d / "continue.npz", tokens=np.array(toks), poses=np.stack(cont))
    np.savez(d / "brake.npz", tokens=np.array(toks), poses=np.stack(brake))
    log.info("scorer subsample: %d pairs, %d tokens (of %d eligible pairs)", len(S), len(toks), int(ok.sum()))


def scorer_read(run_dir: Path, cont_csv: str, brake_csv: str):
    from . import navsim_zs as Z
    d = run_dir / "scorer"
    S = pd.read_parquet(d / "pairs.parquet")
    sc = {k: pd.read_csv(p).set_index("token").score for k, p in (("c", cont_csv), ("b", brake_csv))}
    idx = {e["token"]: e for e in Z.load_index("navtrain", slim=True)}
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        fut = dict(zip(f["tokens"].tolist(), f["poses"]))

    def human_slow(t):   # constant-speed arc length minus the logged arc length at 4 s (> 0: the human slowed)
        p = np.r_[[[0.0, 0.0]], fut[t][:, :2]]
        return 4.0 * float(np.linalg.norm(idx[t]["vel"][-1])) - float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    toks = [t for t in np.loadtxt(d / "tokens.txt", dtype=str) if t in sc["c"].index and t in sc["b"].index]
    T = pd.DataFrame({"token": toks, "diff": [sc["b"][t] - sc["c"][t] for t in toks], "slow": [human_slow(t) for t in toks]})
    T["human"] = np.where(T.slow > 2, 1, np.where(T.slow < -2, -1, 0))
    m = (T.human != 0) & (T["diff"].abs() >= 0.05)
    agree_tok = (np.sign(T["diff"][m]) == T.human[m]).to_numpy().astype(float)
    Ti = T.set_index("token")
    S = S[S.tok_xp.isin(Ti.index) & S.tok_xm.isin(Ti.index)]
    dd = Ti["diff"][S.tok_xp].to_numpy() - Ti["diff"][S.tok_xm].to_numpy()
    hs = Ti.slow[S.tok_xp].to_numpy() - Ti.slow[S.tok_xm].to_numpy()
    mp = (np.abs(dd) >= 0.05) & (np.abs(hs) > 2)
    agree_pair = (np.sign(dd[mp]) == np.sign(hs[mp])).astype(float)
    rng = np.random.default_rng(0)

    def ci(v):
        if not len(v):
            return np.nan, np.nan, np.nan
        bs = rng.choice(v, size=(NB, len(v))).mean(1)
        return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    R = pd.DataFrame([{"level": "token", "n_scored": len(T), "n_decisive": int(m.sum()), **dict(zip(("agree", "lo", "hi"), ci(agree_tok)))},
                      {"level": "pair", "n_scored": len(S), "n_decisive": int(mp.sum()), **dict(zip(("agree", "lo", "hi"), ci(agree_pair)))}])
    R["diff_median"] = float(T["diff"].median())
    R["diff_p10"], R["diff_p90"] = float(T["diff"].quantile(.1)), float(T["diff"].quantile(.9))
    R["share_prefers_brake"] = float((T["diff"] > 0).mean())
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / "navtrain_scorer.csv", index=False)
    T.to_csv(d / "per_token.csv", index=False)
    print(R.to_markdown(index=False, floatfmt=".3f"))


def main():
    from .runlog import RunLog
    cmd = sys.argv[1]
    if cmd == "extract":
        extract(sys.argv[2])
    elif cmd == "run":
        rl = RunLog("elicitation", "e3-feas")
        N = navtrain(rl)
        main_nav(rl, N)
        Wd = wod(rl)
        main_wod(rl, Wd)
        features(rl.dir)
        rl.close()
    elif cmd == "features":
        features(Path(sys.argv[2]))
    elif cmd == "posthoc":
        posthoc(Path(sys.argv[2]))
    elif cmd == "scorer-prep":
        scorer_prep(Path(sys.argv[2]))
    elif cmd == "scorer-read":
        scorer_read(Path(sys.argv[2]), sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
