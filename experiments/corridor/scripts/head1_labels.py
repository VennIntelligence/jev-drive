"""HEAD1 turn labels (experiments/corridor, plans/2026-10-10-head1-prereg.md). CPU only; envs/navsim2 (nuPlan map layers).
A speed-independent description of the road ahead: the heading (relative to the ego heading at t0) as a function of ARC LENGTH, on GRID.
Two label sources, both TRAINING LABELS / ANALYSIS ONLY, never an inference input:

  L  the logged path: heading of the logged rear-axle pose at arc length s along the logged path (log frames after t0, up to MAXF frames
     or until MAX_S m are covered). Leaks at training time: the path the driver took (shape, in-lane placement, lane changes). The time
     axis is removed: a stop or a slow car ahead does not change the label. nan beyond the logged coverage.
  M  the map: tangent heading of the centreline of the lane sequence the log drove (CORR0's Viterbi match on the lane graph, decision 240,
     amendment B: the lane run the log is in at 4 s), at arc length s along that centreline from the ego's projection. Leaks: the map,
     the localisation and lane at t0, the exit the log chose at every fork inside the coverage. Not the lateral placement or the path
     shape. nan beyond the last matched logged pose of the run (after it the centreline is only the straightest successor).

  $DATA_DIR/envs/navsim2/bin/python experiments/corridor/scripts/head1_labels.py build --split navtrain|navtest [--tokens f.txt] [--tag T]
        -> $DATA_DIR/runs/corridor/head1/labels/<split><tag>.npz, rows in token-cache order (navtrain: the 12 navtrain_full shards
           concatenated; navtest: lb_navtest): names, log, L, Lxy, M, Mxy, s4, dyaw4, s_log, s_cov, status, lane_change, d0, fut_err
  ... head1_labels.py bev --split S --tag T [--tokens t1 t2 ..]     check sheet: lane graph, driven sequence, logged path, the path
        re-integrated from L, the centreline, and both profiles against arc length -> $DATA_DIR/runs/corridor/head1/labels/bev_<S><T>.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "research"), str(Path(__file__).parent)]
import corr_geom as C  # noqa: E402  (sets the map env, imports lanegraph)

LG = C.LG
D = C.D
OUT = D / "runs/corridor/head1/labels"
CR = D / "runs/op_parity/cache"
NSH = 12
GRID = np.r_[np.arange(0.0, 40.001, 2.5), 45.0, 50.0, 60.0, 70.0, 80.0]      # arc lengths (m) of the profile, 22 points
MAX_S, MAXF, EPS_V = 82.0, 60, 0.05                                          # logged window: 82 m or 30 s; steps under 5 cm are standstill
STATUS = ("ok", "no_candidate_t0", "no_candidate_4s", "no_connected_sequence", "no_frames")
_J = {}


def tab_of(split):
    """Token-cache rows of a split -> (names, logs, fut (n, 8, 3))."""
    dirs = [f"navtrain_full.s{k}of{NSH}" for k in range(NSH)] if split == "navtrain" else ["lb_navtest"]
    tabs = [np.load(CR / d / "tab.npz") for d in dirs]
    return tuple(np.concatenate([t[k] for t in tabs]) for k in ("names", "log", "fut"))


def window(xy, i):
    """Frames i .. j of the logged window and the arc length of each (standstill steps count as zero)."""
    n = len(xy)
    j = min(n - 1, i + MAXF)
    st = np.hypot(*np.diff(xy[i:j + 1], axis=0).T)
    cum = np.r_[0.0, np.cumsum(np.where(st < EPS_V, 0.0, st))]
    over = np.flatnonzero(cum >= MAX_S)
    if len(over):
        cum = cum[:over[0] + 1]
    return i + len(cum) - 1, cum


def profile_log(xy, yaw, i):
    """L: (heading (G,), xy (G, 2), s4, dyaw4, covered arc length) in the ego frame of frame i."""
    j, cum = window(xy, i)
    P = C.to_ego(xy[i:j + 1], xy[i], yaw[i])
    psi = np.unwrap(yaw[i:j + 1]) - yaw[i]
    keep = np.r_[True, np.diff(cum) > 0]
    s, P2, h = cum[keep], P[keep], psi[keep]
    ok = GRID <= s[-1] + 1e-9
    Lh = np.where(ok, np.interp(GRID, s, h), np.nan)
    Lxy = np.stack([np.where(ok, np.interp(GRID, s, P2[:, k]), np.nan) for k in (0, 1)], 1)
    s4, d4 = (cum[8], psi[8]) if len(cum) > 8 else (np.nan, np.nan)
    return Lh, Lxy, s4, d4, float(s[-1]), j


def driven(M, i, j):
    """CORR0's Matcher.driven with a longer first window (frames i .. j, the logged window of L), then its 10 s and 4 s windows."""
    n = len(M.xy)
    if not M.cand(i):
        return 1, None
    near = [k for k in range(i, min(n - 1, i + 8) + 1) if M.cand(k)]
    if len(near) < min(5, n - i):
        return 2, None
    for m in sorted({j - i, C.NF, 8}, reverse=True):
        if m > j - i and m > 8:
            continue
        fs = [k for k in range(i, min(n - 1, i + m) + 1) if M.cand(k)]
        p = M.g.viterbi([M.cand(k) for k in fs])
        if p is not None:
            return 0, [(k - i, nd) for k, nd in zip(fs, p)]
    return 3, None


def profile_map(g, M, xy, yaw, i, j):
    """M: (status, heading (G,), xy (G, 2), covered arc length, lane change in 0-4 s, d0, node ids of the run)."""
    nanv = np.full(len(GRID), np.nan)
    st, fpath = driven(M, i, j)
    if fpath is None:
        return st, nanv, np.full((len(GRID), 2), np.nan), np.nan, False, np.nan, []
    runs = g.runs([nd for _, nd in fpath])
    ri = max(k for k, (f, _) in enumerate(runs) if fpath[f][0] <= 8)          # decision 240 amendment B: the run the log is in at 4 s
    last = fpath[(runs[ri + 1][0] if ri + 1 < len(runs) else len(fpath)) - 1][0]  # frame offset of the last pose matched inside this run
    Rxy, used, _ = g.centreline(runs[ri][1], xy[i], 90.0)
    R = LG.Line(C.to_ego(Rxy, xy[i], yaw[i]))
    cov = float(R.frenet(C.to_ego(xy[i + last][None], xy[i], yaw[i]))[0][0])
    d0 = float(R.frenet(np.zeros((1, 2)))[1][0])
    ok = GRID <= min(cov, R.L) + 1e-9
    q = R.at(GRID)
    return 0, np.where(ok, q[:, 2], np.nan), np.where(ok[:, None], q[:, :2], np.nan), cov, ri > 0, d0, [g.ids[k] for k in runs[ri][1]]


def label_log(log):
    split, toks = _J["split"], _J["logs"][log]
    fr, xy, yaw = C.log_frames("trainval" if split == "navtrain" else "test", log)
    g = LG.get(fr[0]["map_location"])
    M = C.Matcher(g, xy, yaw)
    pos = {f["token"]: k for k, f in enumerate(fr)}
    out = []
    for row, tok in toks:
        i = pos.get(tok)
        if i is None:
            out.append(dict(row=row, status=4))
            continue
        Lh, Lxy, s4, d4, s_log, j = profile_log(xy, yaw, i)
        st, Mh, Mxy, cov, lc, d0, seq = profile_map(g, M, xy, yaw, i, j)
        r = dict(row=row, status=st, L=Lh, Lxy=Lxy, M=Mh, Mxy=Mxy, s4=s4, dyaw4=d4, s_log=s_log, s_cov=cov, lane_change=lc, d0=d0,
                 fut8=C.to_ego(xy[i + 1:i + 9], xy[i], yaw[i]) if i + 8 < len(xy) else None)
        if _J.get("keep"):
            r.update(o=np.r_[xy[i], yaw[i]], loc=g.loc, seq=seq, path=C.to_ego(xy[i:j + 1], xy[i], yaw[i]))
        out.append(r)
    return out


def build(split, want=None, keep=False, run=None):
    from jevdrive import par
    names, logs, fut = tab_of(split)
    n = len(names)
    by = {}
    for k in range(n):
        if want is None or names[k] in want:
            by.setdefault(str(logs[k]), []).append((k, str(names[k])))
    _J.update(split=split, logs=by, keep=keep)
    res = par.pmap(label_log, sorted(by), run=run, desc=f"{split} logs")
    res.raise_if_failed()
    G = len(GRID)
    Z = dict(names=names, log=logs, grid=GRID, L=np.full((n, G), np.nan, np.float32), Lxy=np.full((n, G, 2), np.nan, np.float32),
             M=np.full((n, G), np.nan, np.float32), Mxy=np.full((n, G, 2), np.nan, np.float32), status=np.full(n, -1, np.int8),
             lane_change=np.zeros(n, bool), **{k: np.full(n, np.nan, np.float32) for k in ("s4", "dyaw4", "s_log", "s_cov", "d0", "fut_err")})
    extra = {}
    for rows in res.values:
        for r in rows:
            k = r["row"]
            Z["status"][k] = r["status"]
            if r["status"] == 4:
                continue
            for key in ("L", "Lxy", "M", "Mxy", "s4", "dyaw4", "s_log", "s_cov", "lane_change", "d0"):
                Z[key][k] = r[key]
            if r["fut8"] is not None and np.isfinite(fut[k]).all():           # the label frame is the token cache's frame
                Z["fut_err"][k] = np.abs(r["fut8"] - fut[k, :, :2]).max()
            if keep:
                extra[str(names[k])] = {q: r[q] for q in ("o", "loc", "seq", "path")}
    return Z, extra


def cmd_build(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", f"head1/labels-{a.split}{a.tag}", config=vars(a)) as run:
        sp = splits.load(f"navsim/{a.split}")
        run.use_split(sp)
        want = set(open(a.tokens).read().split()) if a.tokens else None
        Z, extra = build(a.split, want, keep=bool(a.tokens), run=run)
        assert want is not None or sp.mask(Z["names"]).all(), "token cache rows outside the registered split"
        OUT.mkdir(parents=True, exist_ok=True)
        dst = OUT / f"{a.split}{a.tag}.npz"
        np.savez(dst.with_suffix(".tmp.npz"), **Z)
        dst.with_suffix(".tmp.npz").rename(dst)
        if extra:
            import pickle
            pickle.dump(extra, open(OUT / f"{a.split}{a.tag}.extra.pkl", "wb"))
        done = Z["status"] >= 0
        turn = done & (np.abs(np.degrees(Z["dyaw4"])) > 45)
        fe = Z["fut_err"][done & np.isfinite(Z["fut_err"])]
        summ = dict(tokens=int(done.sum()), status={STATUS[s]: int((Z["status"][done] == s).sum()) for s in range(5)},
                    status_turn45={STATUS[s]: int((Z["status"][turn] == s).sum()) for s in range(5)},
                    fut_err_max_m=float(fe.max()) if len(fe) else None, fut_err_over_5cm=int((fe > 0.05).sum()),
                    L_cover=[float(np.isfinite(Z["L"][done][:, k]).mean()) for k in (8, 16, 21)],
                    M_cover=[float(np.isfinite(Z["M"][done][:, k]).mean()) for k in (8, 16, 21)],
                    L_cover_turn45=[float(np.isfinite(Z["L"][turn][:, k]).mean()) for k in (4, 8, 16)],
                    M_cover_turn45=[float(np.isfinite(Z["M"][turn][:, k]).mean()) for k in (4, 8, 16)], out=str(dst))
        run.summary.update(summ)
        run.info(str(summ))


def cmd_bev(a):
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PSY
    from shapely.geometry import Point
    PSY.apply()
    Z = np.load(OUT / f"{a.split}{a.tag}.npz")
    X = pickle.load(open(OUT / f"{a.split}{a.tag}.extra.pkl", "rb"))
    pos = {t: k for k, t in enumerate(Z["names"].astype(str))}
    toks = a.tokens or sorted(X)
    nc = len(toks)
    fig, axs = plt.subplots(2, nc, figsize=(1.75 * nc, 4.3), squeeze=False, gridspec_kw=dict(height_ratios=[1.6, 1]))
    col = PSY.PALETTE
    for n, t in enumerate(toks):
        k, x = pos[t], X[t]
        g = LG.get(x["loc"])
        o, yaw = x["o"][:2], x["o"][2]
        ax = axs[0, n]
        for q in g.tree.query(Point(*o).buffer(60), predicate="intersects"):
            e = C.to_ego(np.asarray(g.poly[q].exterior.coords)[:, :2], o, yaw)
            on = g.ids[q] in set(x["seq"])
            ax.fill(-e[:, 1], e[:, 0], fc=col["sky_blue"] if on else "#DDDDDD", ec=col["blue"] if on else "#888888", lw=0.25, alpha=0.45 if on else 0.35, zorder=1 + on)
        ax.plot(-x["path"][:, 1], x["path"][:, 0], "k.-", ms=2.2, lw=0.6, zorder=6, label="log frames")
        Lh, okL = Z["L"][k], np.isfinite(Z["L"][k])
        if okL.sum() > 1:                                                     # the path re-integrated from the heading profile alone
            s = GRID[okL]
            hm = 0.5 * (Lh[okL][1:] + Lh[okL][:-1])
            xi, yi = np.r_[0, np.cumsum(np.diff(s) * np.cos(hm))], np.r_[0, np.cumsum(np.diff(s) * np.sin(hm))]
            ax.plot(-yi, xi, "-", color=col["vermillion"], lw=0.9, zorder=5, label="integrated L")
        ax.plot(-Z["Mxy"][k][:, 1], Z["Mxy"][k][:, 0], ".-", color=col["blue"], ms=2, lw=0.8, zorder=4, label="centreline (M)")
        m = max(14.0, float(np.nanmax(np.abs(Z["Lxy"][k]))) * 0.6 + 6) if okL.any() else 20.0
        ax.set_xlim(-m, m); ax.set_ylim(-6, 2 * m - 6); ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.98, f"{t[:6]} {STATUS[Z['status'][k]]}\ndyaw4 {np.degrees(Z['dyaw4'][k]):+.0f} s4 {Z['s4'][k]:.1f} m\ncov L {Z['s_log'][k]:.0f} M {Z['s_cov'][k]:.0f} m"
                + (" LC" if Z["lane_change"][k] else ""), transform=ax.transAxes, va="top", fontsize=5)
        ax = axs[1, n]
        ax.plot(GRID, np.degrees(Z["L"][k]), ".-", color="k", ms=3, label="L (log)")
        ax.plot(GRID, np.degrees(Z["M"][k]), ".-", color=col["blue"], ms=3, label="M (map)")
        ax.axvline(Z["s4"][k], color="#999999", lw=0.5)
        ax.set_xlim(0, 45); ax.set_xlabel("arc length (m)")
        if n == 0:
            ax.set_ylabel("heading (deg)"); ax.legend(fontsize=5.5); axs[0, 0].legend(fontsize=5, loc="lower left")
    fig.tight_layout(pad=0.4)
    dst = OUT / f"bev_{a.split}{a.tag}"
    PSY.save(fig, dst)
    print(dst.with_suffix(".png"))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--split", required=True, choices=("navtrain", "navtest")); b.add_argument("--tokens", default=""); b.add_argument("--tag", default="")
    cli_args(b)
    v = sub.add_parser("bev"); v.add_argument("--split", required=True); v.add_argument("--tag", default=""); v.add_argument("--tokens", nargs="*")
    a = ap.parse_args()
    {"build": cmd_build, "bev": cmd_bev}[a.cmd](a)
