"""Independent recheck of decisions 20's claim about the top s_ego decile of the Waymo E2E rater frames.

The claim under audit: in the top s_ego decile of the 479 rater-scored val frames the *logged* future scores
RFS 5.92, is floored 41.7 % of the time and sits 7.82 m (ADE) from the top-rated rater trajectory, therefore
"human raters do not endorse the driver's own choice there".

Nothing here calls jevdrive.waymo_l0.rater_bins or jevdrive.waymo.rater_feedback_score for the numbers it
reports. The Rater Feedback Score is re-implemented from the published specification in `rfs()` below and is
cross-checked against the repository's port on every rater frame plus randomly perturbed candidates; the
surprise s_ego is re-fitted here with its own ridge rather than reused. Data loading (the index, the ego
arrays, the rater parquet, the sequence halves) is shared, because that is the raw data, not the claim.

Everything runs on CPU.

Steps:
  bins      the 10-row decile table, recomputed, plus per-decile kinematics and trust-region geometry
  ab        why the logged future scores low: outside every trust region (multimodal) vs inside a
            low-rated one (no good option available)
  confound  the same table binned by the CTRV residual and by speed, neither of which is fitted
  frames    per-frame detail and the contact sheets for 20 top-decile and 10 control frames
"""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive import plots, waymo
from jevdrive import waymo_stage_a as sa
from jevdrive.common import get_logger

log = get_logger(__name__)

HORIZ_S = (3.0, 5.0)          # s, where the trust region is checked
FREQ = 4                      # Hz, the future grid
BASE_LAT = (1.0, 1.8)         # m, lateral half-width at 3 s / 5 s before the speed scale
LON_MULT = 4.0                # longitudinal half-width is this many times the lateral one
DECAY, FLOOR = 0.1, 4.0
V_LO, V_HI = 1.4, 11.0        # m/s, the speed range the scale interpolates over
LAMS = np.logspace(-6, 4, 21)


# ---------------------------------------------------------------- Rater Feedback Score, re-implemented

def rater_axes(traj):
    """(longitudinal, lateral) unit vectors at every waypoint of every rater trajectory, (n, P, T, 2) each.

    The longitudinal direction at waypoint t is the step from t-1 to t (the origin precedes waypoint 0); a
    zero step repeats the last non-zero one, and a trajectory that never moves points along +x. Lateral is
    that turned 90 degrees counter-clockwise.
    """
    t = np.asarray(traj, np.float64)
    prev = np.concatenate([np.zeros_like(t[:, :, :1]), t[:, :, :-1]], axis=2)
    step = t - prev
    lng = np.empty_like(step)
    for k in range(step.shape[2]):                      # carry the last non-zero step forward, waypoint by
        cur = step[:, :, k]                             # waypoint: 20 iterations, clearer than an argmax scan
        prev_dir = lng[:, :, k - 1] if k else np.broadcast_to(np.array([1.0, 0.0]), cur.shape)
        moved = np.linalg.norm(cur, axis=-1) > 0
        lng[:, :, k] = np.where(moved[..., None], cur, prev_dir)
    lng = lng / np.linalg.norm(lng, axis=-1, keepdims=True)
    lat = np.stack([-lng[..., 1], lng[..., 0]], axis=-1)
    return lng, lat


def rfs(pred, traj, scores, speed):
    """Per-frame Rater Feedback Score of one candidate, re-implemented from the published definition.

    `pred` (n, T, 2), `traj` (n, P, T, 2), `scores` (n, P), `speed` (n,). Returns a dict with
      score    (n,)      the frame's RFS
      inside   (n,)      the candidate is within some single rater's trust region at *both* horizons
      norm     (n, P, H) the overshoot ratio per rater and horizon (<= 1 is inside)
      per      (n, P, H) that rater's decayed contribution
      lat_thr, lng_thr   (n, H) the trust-region half-widths actually used
    """
    pred = np.asarray(pred, np.float64)
    traj = np.asarray(traj, np.float64)
    scores = np.asarray(scores, np.float64)
    lng, lat = rater_axes(traj)
    k = np.array([int(round(h * FREQ)) - 1 for h in HORIZ_S])
    d = pred[:, None] - traj                                      # (n, P, T, 2)
    d_lng = np.abs((d * lng).sum(-1))[:, :, k]                    # (n, P, H)
    d_lat = np.abs((d * lat).sum(-1))[:, :, k]
    scale = np.clip(0.5 + 0.5 * (np.asarray(speed, np.float64) - V_LO) / (V_HI - V_LO), 0.5, 1.0)[:, None]
    base = np.array(BASE_LAT)
    lat_thr = scale * (base * 1.0)                                # (n, H); this multiplication order is the
    lng_thr = scale * (base * LON_MULT)                           # official one and float maths is not assoc.
    norm = np.maximum(d_lng / lng_thr[:, None], d_lat / lat_thr[:, None])
    inside = (norm <= 1.0).all(-1).any(-1)                        # (n,)
    per = scores[..., None] * DECAY ** np.maximum(norm - 1.0, 0.0)
    score = per.max(1).mean(-1)                                   # best rater per horizon, then mean over both
    score = np.where(inside, score, np.maximum(score, FLOOR))
    return {"score": score, "inside": inside, "norm": norm, "per": per,
            "lat_thr": lat_thr, "lng_thr": lng_thr}


def official_rfs(path, pred, traj, scores, speed):
    """Waymo's own `get_rater_feedback_score`, loaded from a copy of the upstream file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rater_feedback_utils_official", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.get_rater_feedback_score(
        inference_trajectories=np.asarray(pred, np.float64)[:, None],
        inference_probs=np.ones((len(pred), 1), np.float64),
        rater_specified_trajectories=[[np.asarray(x, np.float64) for x in b] for b in traj],
        rater_feedback_labels=[np.asarray(s, np.float64) for s in scores],
        init_speed=np.asarray(speed, np.float64))
    return out


def check_against_port(traj, scores, speed, fut, seed=0, official=None):
    """The re-implementation against jevdrive.waymo's port and, where available, against Waymo's own code."""
    rng = np.random.default_rng(seed)
    worst, worst_off = 0.0, None
    cands = (("logged", fut), ("rater0", traj[:, 0]),
             ("jitter", traj[:, 0] + rng.normal(0, 3.0, traj[:, 0].shape)),
             ("far", traj[:, 0] + 1e3))
    for name, p in cands:
        mine = rfs(p, traj, scores, speed)
        theirs, inside = waymo.rater_feedback_score(p, traj, scores, speed, details=True)
        e = float(np.abs(mine["score"] - theirs).max())
        assert (mine["inside"] == inside[:, 0]).all(), name
        worst = max(worst, e)
        log.info("rfs cross-check [%s]: max |mine - port| = %.3g, inside masks identical", name, e)
        if official is not None and Path(official).exists():
            o = official_rfs(official, p, traj, scores, speed)
            key = next(k for k in o if "score" in k and "rater" in k) if "rater_feedback_score" not in o \
                else "rater_feedback_score"
            eo = float(np.abs(np.asarray(o[key]).reshape(-1) - theirs).max())
            worst_off = eo if worst_off is None else max(worst_off, eo)
            log.info("rfs cross-check [%s]: max |port - Waymo upstream| = %.3g", name, eo)
    assert worst < 1e-9, worst
    if worst_off is not None:
        assert worst_off < 1e-9, worst_off
    return worst, worst_off


# ---------------------------------------------------------------- data and the surprise, re-fitted

def ridge(X, Y, rows, lams):
    """Closed-form ridge with an intercept, one weight matrix per lambda: (L, d + 1, k)."""
    Xa, Ya = X[rows].astype(np.float64), Y[rows].astype(np.float64)
    mu, ym = Xa.mean(0), Ya.mean(0)
    Xc = Xa - mu
    G = Xc.T @ Xc
    B = Xc.T @ (Ya - ym)
    out = np.empty((len(lams), X.shape[1] + 1, Y.shape[1]))
    for i, lam in enumerate(lams):
        W = np.linalg.solve(G + lam * np.eye(len(G)), B)
        out[i, :-1] = W
        out[i, -1] = ym - mu @ W
    return out


def apply_w(X, W, T):
    return (X @ W[:-1] + W[-1]).reshape(len(X), T, 2)


def ade(p, g):
    return np.linalg.norm(np.asarray(p, np.float64) - np.asarray(g, np.float64), axis=-1).mean(-1)


def surprise(ego, fut, train, val, seq, folds=4):
    """s_ego for every frame: out-of-fold inside the fit half, out-of-sample on the evaluation half."""
    from sklearn.model_selection import GroupKFold
    T = fut.shape[1]
    Y = fut.reshape(len(fut), -1)
    mu, sd = ego[train].mean(0), ego[train].std(0)
    X = (ego - mu) / np.where(sd > 1e-6, sd, 1.0)
    oof = np.zeros((len(LAMS), len(fut), T, 2))
    sc = np.zeros(len(LAMS))
    for a, b in GroupKFold(folds).split(train, groups=seq[train]):
        W = ridge(X, Y, train[a], LAMS)
        te = train[b]
        for i in range(len(LAMS)):
            p = apply_w(X[te], W[i], T)
            oof[i, te] = p
            sc[i] += ade(p, fut[te]).mean()
    best = int(np.argmin(sc))
    W = ridge(X, Y, train, [LAMS[best]])[0]
    oof[best, val] = apply_w(X[val], W, T)
    log.info("s_ego: lambda %.3g, held-out ADE %.3f (%d fit / %d eval frames)",
             LAMS[best], sc[best] / folds, len(train), len(val))
    return ade(oof[best], fut)


def load(set_name="qwen_front3", seed=0):
    """Raw val data restricted the way stage A restricts it, plus the rater frames and the two halves."""
    df = waymo.load_index()
    past, future = waymo.load_ego()
    feat_idx = pd.concat([pd.read_parquet(d / "index.parquet", columns=["frame_name"])
                          for d in sorted(waymo.out_dir("features", set_name).iterdir())
                          if d.is_dir() and (d / "meta.json").exists()], ignore_index=True)
    name = waymo.frame_names(df)
    # a hashed membership test: np.isin on object arrays sorts them and costs minutes at this size
    has_feature = pd.Index(name).isin(pd.Index(feat_idx.frame_name.to_numpy()))
    keep = (df.split == "val").to_numpy() & df.has_future.to_numpy() & has_feature
    rows = np.flatnonzero(keep)
    ego = np.concatenate([waymo.ego_state(past[rows]), waymo.intent_onehot(df.iloc[rows])], 1).astype(np.float64)
    fut = waymo.future_xy(future[rows]).astype(np.float64)
    seq = df.sequence.to_numpy()[rows]
    halves = sa.val_halves(df, seed)
    half = df.sequence.map(halves).to_numpy()[rows]
    pos, rtraj, scores = sa.rfs_rows(df, rows)
    log.info("loaded %d val frames with a feature, %d of them rater-scored", len(rows), len(pos))
    return dict(df=df, past=past, future=future, rows=rows, ego=ego, fut=fut, seq=seq, half=half,
                pos=pos, rtraj=rtraj.astype(np.float64), scores=scores.astype(np.float64), name=name)


def pooled_surprise(d, folds=4):
    """Every frame's s_ego taken from the direction in which it belongs to the *evaluation* half, i.e. from
    the fit that never saw its sequence. Both directions are returned so the choice can be inspected."""
    s = np.full((2, len(d["rows"])), np.nan)
    for direction in (0, 1):
        train = np.flatnonzero(d["half"] == direction)
        val = np.flatnonzero(d["half"] == 1 - direction)
        s[direction] = surprise(d["ego"], d["fut"], train, val, d["seq"], folds)
    # direction `dd` fits on half dd and evaluates half 1-dd, so a frame in half h is evaluated by dd = 1-h
    oos = np.where(d["half"] == 0, s[1], s[0])
    ins = np.where(d["half"] == 0, s[0], s[1])
    return oos, ins, s


# ---------------------------------------------------------------- the tables

def bin_of(x, q):
    edge = np.quantile(x, np.linspace(0, 1, q + 1))
    return np.clip(np.searchsorted(edge[1:-1], x, "right"), 0, q - 1)


def frame_table(d, s_pool):
    """One row per rater frame: everything the audit needs, before any binning."""
    pos, rtraj, scores = d["pos"], d["rtraj"], d["scores"]
    rows = d["rows"][pos]
    past = d["past"][rows]
    speed = waymo.init_speed(past)
    kin = waymo.past_kinematics(past)
    log_xy = d["fut"][pos]
    best = rtraj[np.arange(len(pos)), scores.argmax(1)]
    r = rfs(log_xy, rtraj, scores, speed)
    base = waymo.baselines(past)
    # which rater trajectory the log is closest to in trust-region units, and whether it is inside it
    worst_norm = r["norm"].max(-1)                                   # (n, P) overshoot over both horizons
    nearest = worst_norm.argmin(1)
    t = pd.DataFrame({
        "frame_name": d["name"][rows], "sequence": d["df"].sequence.to_numpy()[rows],
        "frame": d["df"].frame.to_numpy()[rows], "cluster": d["df"].cluster.to_numpy()[rows],
        "intent": d["df"].intent.to_numpy()[rows], "half": d["half"][pos],
        "s_ego": s_pool[pos], "speed": speed, "a0": kin["a"], "yaw_rate": kin["w"],
        "log_path_len": np.linalg.norm(np.diff(np.concatenate(
            [np.zeros((len(log_xy), 1, 2)), log_xy], 1), axis=1), axis=-1).sum(1),
        "log_rfs": r["score"], "log_inside": r["inside"],
        "log_floored": r["score"] <= FLOOR + 1e-9,
        "log_ade_best": ade(log_xy, best),
        "nearest_rater": nearest,
        "nearest_norm": worst_norm[np.arange(len(pos)), nearest],
        "nearest_score": scores[np.arange(len(pos)), nearest],
        "rater_max": scores.max(1), "rater_min": scores.min(1), "rater_mean": scores.mean(1),
        "lat_thr3": r["lat_thr"][:, 0], "lat_thr5": r["lat_thr"][:, 1],
        "lng_thr3": r["lng_thr"][:, 0], "lng_thr5": r["lng_thr"][:, 1],
        "ade_best_rater_spread": np.array([ade(rtraj[i], best[i]).mean() for i in range(len(pos))]),
        "s_ctrv": ade(base["ctrv"], log_xy), "s_cv": ade(base["cv"], log_xy),
    })
    # (a) the log is outside every trust region -> a different choice from all three proposals
    # (b) the log is inside one -> the score it gets is that rater's own label
    t["case"] = np.where(t.log_inside, "inside", "outside")
    t["best_rater_score"] = scores.max(1)
    return t


def decile_table(t, col="s_ego", q=10, label="s_ego"):
    b = bin_of(t[col].to_numpy(), q)
    rows = []
    for i in range(q):
        m = b == i
        g = t[m]
        rows.append({
            "binned_by": label, "bin": i + 1, "n": int(m.sum()),
            "lo": g[col].min(), "hi": g[col].max(),
            "log_rfs": g.log_rfs.mean(), "log_floored": g.log_floored.mean(),
            "log_ade_best": g.log_ade_best.mean(),
            "outside_all": (~g.log_inside).mean(),
            "inside_and_low": (g.log_inside & g.log_floored).mean(),
            "rfs_given_inside": g.log_rfs[g.log_inside].mean(),
            "rfs_given_outside": g.log_rfs[~g.log_inside].mean(),
            "rater_max": g.rater_max.mean(), "rater_min": g.rater_min.mean(),
            "speed": g.speed.mean(), "abs_a0": g.a0.abs().mean(),
            "path_len": g.log_path_len.mean(),
            "ade_over_path": (g.log_ade_best / g.log_path_len).mean(),
            "rater_spread": g.ade_best_rater_spread.mean(),
            "lat_thr5": g.lat_thr5.mean(), "lng_thr5": g.lng_thr5.mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- padding and the log's membership

def raw_waypoint_counts():
    """Per rater-scored frame: how many waypoints each rated trajectory really has before the metric pads it.

    WOD issue #985: rated trajectories shorter than 20 waypoints are padded by repeating the last one, which
    parks them at a standstill for the rest of the horizon and makes the 5 s check reward slow candidates.
    """
    t = pd.read_parquet(waymo.out_dir() / "rater.parquet")
    out = {}
    for fn, g in t.groupby("frame_name", sort=False):
        out[fn] = [len(p) for p in g.pos_x]
    return out


def rfs_no_padded_horizon(pred, traj, scores, speed, raw_len):
    """RFS with every (rater, horizon) pair whose waypoint is padding left out of the comparison.

    `raw_len` is (n, P). A horizon at index k is real for a rater only when that rater's trajectory actually
    reaches it (raw_len > k). Raters with no real horizon left do not compete; frames with no rater left at
    all fall back to the ordinary score.
    """
    r = rfs(pred, traj, scores, speed)
    k = np.array([int(round(h * FREQ)) - 1 for h in HORIZ_S])
    valid = np.asarray(raw_len)[:, :, None] > k[None, None, :]              # (n, P, H)
    norm = np.where(valid, r["norm"], np.nan)
    with np.errstate(invalid="ignore"):
        ok = (np.nan_to_num(norm, nan=0.0) <= 1.0) | ~valid
        inside = (ok.all(-1) & valid.any(-1)).any(-1)
        per = np.where(valid, r["per"], -np.inf)
        best = per.max(1)                                                   # (n, H), -inf where no rater
        keep = np.isfinite(best)
        score = np.where(keep.any(-1), np.nansum(np.where(keep, best, 0), -1) / np.maximum(keep.sum(-1), 1),
                         r["score"])
    score = np.where(inside, score, np.maximum(score, FLOOR))
    return {"score": score, "inside": inside}


# ---------------------------------------------------------------- contact sheets

def read_jpeg(d, row, cam="front"):
    from PIL import Image
    shard = waymo.shard_dir() / str(d["df"].shard.to_numpy()[row])
    off = int(d["df"][f"{cam}_off"].to_numpy()[row])
    n = int(d["df"][f"{cam}_len"].to_numpy()[row])
    fd = os.open(shard, os.O_RDONLY)
    try:
        return Image.open(io.BytesIO(os.pread(fd, n, off))).convert("RGB")
    finally:
        os.close(fd)


def contact_sheet(d, t, sel, out, name, ncol=4):
    """One cell per frame: the front camera above, a bird's-eye view of log vs raters below."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    nrow = int(np.ceil(len(sel) / ncol))
    with mpl.rc_context(plots.STYLE):
        fig = plt.figure(figsize=(3.4 * ncol, 3.5 * nrow))
        outer = fig.add_gridspec(nrow, ncol, hspace=0.30, wspace=0.18)
        for j, i in enumerate(sel):
            r = t.iloc[i]
            row = d["rows"][d["pos"][i]]
            inner = outer[j // ncol, j % ncol].subgridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.05)
            ax = fig.add_subplot(inner[0])
            ax.imshow(read_jpeg(d, row))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
            ax.set_title(f"[{j + 1}] {r.sequence[:10]} f{r.frame}  v={r.speed:.1f} m/s  "
                         f"{r.cluster}", fontsize=7, pad=2)

            bev = fig.add_subplot(inner[1])
            log_xy = d["fut"][d["pos"][i]]
            traj, sc = d["rtraj"][i], d["scores"][i]
            k5 = int(round(HORIZ_S[1] * FREQ)) - 1
            for p in range(traj.shape[0]):
                c = plots.OKABE_ITO[[1, 2, 3][p]]
                bev.plot(-traj[p, :, 1], traj[p, :, 0], "-", color=c, lw=1.0,
                         label=f"rater {p} ({sc[p]:.0f})")
                for h, kk in enumerate([int(round(HORIZ_S[0] * FREQ)) - 1, k5]):
                    lat = r["lat_thr3"] if h == 0 else r["lat_thr5"]
                    lng = r["lng_thr3"] if h == 0 else r["lng_thr5"]
                    step = traj[p, kk] - (traj[p, kk - 1] if kk else np.zeros(2))
                    ang = np.degrees(np.arctan2(step[1], step[0])) if np.any(step) else 0.0
                    # the trust region is a box, 2*lng long and 2*lat wide, in the rater trajectory's own
                    # frame. The axes plot (-y, x), which turns a world heading of `ang` into `ang + 90`.
                    box = Rectangle((-lng, -lat), 2 * lng, 2 * lat, facecolor="none", edgecolor=c,
                                    lw=0.5, ls=":")
                    box.set_transform(Affine2D().rotate_deg(90 + ang)
                                      .translate(-traj[p, kk, 1], traj[p, kk, 0]) + bev.transData)
                    bev.add_patch(box)
            bev.plot(-log_xy[:, 1], log_xy[:, 0], "-", color="#000000", lw=1.6, label="logged future")
            bev.plot([0], [0], "o", color="#000000", ms=3)
            bev.set_aspect("equal")
            bev.grid(True, ls=":")
            bev.set_xlabel("lateral (m)")
            bev.set_ylabel("longitudinal (m)")
            bev.set_title(f"log RFS {r.log_rfs:.1f}{' (floored)' if r.log_floored else ''}, "
                          f"ADE {r.log_ade_best:.1f} m, {'inside' if r.log_inside else 'outside'}",
                          fontsize=7, pad=2)
            bev.legend(fontsize=5.5, loc="upper left", handlelength=1.2, labelspacing=0.2)
        plots.save(fig, out, name)
    log.info("wrote %s", out / f"{name}.png")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="qwen_front3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-top", type=int, default=20)
    ap.add_argument("--n-control", type=int, default=10)
    ap.add_argument("--out", default=None, help="output directory (default: a runlog dir)")
    ap.add_argument("--official", default=None,
                    help="path to a copy of waymo_open_dataset/metrics/python/rater_feedback_utils.py; "
                         "when given, the port is checked against it directly")
    a = ap.parse_args()

    from jevdrive.runlog import RunLog
    rl = RunLog("waymo_l0", "top_decile_audit") if a.out is None else None
    out = Path(a.out) if a.out else rl.dir
    out.mkdir(parents=True, exist_ok=True)

    d = load(a.set, a.seed)
    err, err_off = check_against_port(d["rtraj"], d["scores"],
                                      waymo.init_speed(d["past"][d["rows"][d["pos"]]]),
                                      d["fut"][d["pos"]], official=a.official)
    s_oos, s_ins, _ = pooled_surprise(d)
    t = frame_table(d, s_oos)
    t_ins = frame_table(d, s_ins)
    t.to_csv(out / "rater_frames.csv", index=False)

    tab = pd.concat([decile_table(t, "s_ego", 10, "s_ego (out-of-fold)"),
                     decile_table(t_ins, "s_ego", 10, "s_ego (in-fold, sensitivity)"),
                     decile_table(t, "s_ctrv", 10, "ctrv residual"),
                     decile_table(t, "s_cv", 10, "cv residual"),
                     decile_table(t, "speed", 10, "speed")], ignore_index=True)
    tab.to_csv(out / "bins.csv", index=False)
    log.info("decile table:\n%s", tab[tab.binned_by.str.startswith("s_ego (out")].round(3).to_string(index=False))

    # (a) vs (b) in the top decile against the rest
    b = bin_of(t.s_ego.to_numpy(), 10)
    ab = []
    for name, m in (("decile 10", b == 9), ("deciles 1-9", b < 9), ("all", np.ones(len(t), bool))):
        g = t[m]
        ab.append({"group": name, "n": int(m.sum()),
                   "outside_all_trust_regions": (~g.log_inside).mean(),
                   "inside_some": g.log_inside.mean(),
                   "floored_and_outside": (g.log_floored & ~g.log_inside).mean(),
                   "floored_and_inside": (g.log_floored & g.log_inside).mean(),
                   "rfs_given_outside": g.log_rfs[~g.log_inside].mean(),
                   "rfs_given_inside": g.log_rfs[g.log_inside].mean(),
                   "mean_best_rater_label": g.rater_max.mean(),
                   "mean_worst_rater_label": g.rater_min.mean(),
                   "nearest_norm_median": g.nearest_norm.median()})
    ab = pd.DataFrame(ab)
    ab.to_csv(out / "ab_split.csv", index=False)
    log.info("outside vs inside:\n%s", ab.round(3).to_string(index=False))

    # --- padding (WOD issue #985) and whether the logged future is one of the rated trajectories
    counts = raw_waypoint_counts()
    raw = np.array([(lambda c: (c + c[-1:] * 3)[:3])(counts[f]) for f in t.frame_name], float)
    t["n_rater_raw"] = [len(counts[f]) for f in t.frame_name]
    t["n_padded"] = (raw < 20).sum(1)
    t["any_padded"] = t.n_padded > 0
    log_xy = d["fut"][d["pos"]]
    # if the log were one of the rated proposals it would coincide with one of them waypoint for waypoint
    t["min_max_dev_to_rated"] = np.linalg.norm(log_xy[:, None] - d["rtraj"], axis=-1).max(-1).min(-1)
    r_nop = rfs_no_padded_horizon(log_xy, d["rtraj"], d["scores"],
                                  waymo.init_speed(d["past"][d["rows"][d["pos"]]]), raw)
    t["log_rfs_no_padded_horizon"] = r_nop["score"]
    t["log_floored_no_padded_horizon"] = r_nop["score"] <= FLOOR + 1e-9
    pad = []
    for name, m in (("decile 10", b == 9), ("deciles 1-9", b < 9), ("all", np.ones(len(t), bool))):
        g = t[m]
        clean = g[~g.any_padded]
        pad.append({"group": name, "n": int(m.sum()),
                    "traj_shorter_than_20": float((raw[m] < 20).mean()),
                    "frames_with_any_padded": float(g.any_padded.mean()),
                    "n_clean": len(clean),
                    "log_rfs": g.log_rfs.mean(), "log_floored": g.log_floored.mean(),
                    "log_rfs_clean": clean.log_rfs.mean(), "log_floored_clean": clean.log_floored.mean(),
                    "log_ade_best_clean": clean.log_ade_best.mean(),
                    "log_rfs_no_padded_horizon": g.log_rfs_no_padded_horizon.mean(),
                    "log_floored_no_padded_horizon": g.log_floored_no_padded_horizon.mean(),
                    "log_equals_a_rated_traj": float((g.min_max_dev_to_rated < 0.1).mean()),
                    "min_max_dev_to_rated_p01": float(g.min_max_dev_to_rated.quantile(0.01)),
                    "min_max_dev_to_rated_median": float(g.min_max_dev_to_rated.median())})
    pad = pd.DataFrame(pad)
    pad.to_csv(out / "padding.csv", index=False)
    log.info("padding and log membership:\n%s", pad.round(3).to_string(index=False))
    t.to_csv(out / "rater_frames.csv", index=False)

    rng = np.random.default_rng(a.seed)
    top = np.flatnonzero(b == 9)
    order = np.argsort(t.sequence.to_numpy()[top])          # spread over sequences, then thin evenly
    sel_top = top[order][np.linspace(0, len(top) - 1, a.n_top).round().astype(int)]
    ctrl = np.flatnonzero(b < 9)
    sel_ctrl = rng.choice(ctrl, a.n_control, replace=False)
    t.iloc[sel_top].assign(group="top_decile").to_csv(out / "sheet_top.csv", index=False)
    t.iloc[sel_ctrl].assign(group="control").to_csv(out / "sheet_control.csv", index=False)

    contact_sheet(d, t, sel_top, out, "top-decile-rater-frames", ncol=4)
    contact_sheet(d, t, sel_ctrl, out, "mid-decile-rater-frames", ncol=4)

    (out / "audit.json").write_text(json.dumps({
        "n_rater_frames": int(len(t)), "rfs_port_max_abs_diff": err,
        "rfs_vs_waymo_upstream_max_abs_diff": err_off,
        "top_decile": {"n": int((b == 9).sum()), "rfs": float(t.log_rfs[b == 9].mean()),
                       "floored": float(t.log_floored[b == 9].mean()),
                       "ade_best": float(t.log_ade_best[b == 9].mean()),
                       "s_lo": float(t.s_ego[b == 9].min()), "s_hi": float(t.s_ego[b == 9].max())},
    }, indent=2))
    log.info("done -> %s", out)
    if rl:
        rl.event("end")
        rl.close()


if __name__ == "__main__":
    main()
