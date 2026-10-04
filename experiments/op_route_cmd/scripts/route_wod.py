"""Route polylines for WOD-E2E (train and val sequences; test has no future and is never read).

  envs/jevdrive on the box (CPU only):  python experiments/op_route_cmd/scripts/route_wod.py [--workers N] [--limit-seq N]

WOD-E2E logs 5 s of future per frame (20 x 4 Hz, ego frame) but a 150 m route needs more. The frames of a 20 s sequence are dense (10 Hz), so the
path is chained: frame f -> f+25 (2.5 s) -> f+50 ..., where every link is a rigid 2D fit (Kabsch, rotation + translation) between the 20 positions
that the two frames both log at the same absolute 4 Hz times (f.future[0:20] against g.past[6:16] + g.future[0:10]; stationary spread < 0.5 m ->
pure translation). A link whose fit residual (rms) is > 0.3 m or whose partner frame is missing ends the chain, and the polyline is masked beyond it.
Path = f.future[0:10], then g1.future[0:10] moved into f's frame, ... (+ the last frame's remaining 2.5 s), prefixed by the origin.
Map-free: `jct_s`, `jct_dist`, `n_exit` do not exist here (nan / -1) and `turn_junction` is False (unknown): a turn >= 25 deg is geometric (a bend counts too).
Key `id` = "<sequence>-<frame:03d>" (the `name` of op_adapt_L / op_adapt_H tables). Output $DATA_DIR/processed/op_route_cmd/wod/route.npz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib")]
import route_poly as RP  # noqa: E402

LINK = 25                     # frames per link (2.5 s: the 4 Hz lattices of both frames coincide)
MAX_LINKS = 8
RMS_MAX, SPREAD_MIN = 0.3, 0.5
PATH_M = 160.0


def fit_links(past, fut, has_next, nxt):
    """Rigid fit frame(nxt[r]) -> frame(r) for every row r with has_next[r]. Returns theta (n,), t (n, 2), ok (n,) bool, rms (n,)."""
    n = len(past)
    theta, t, ok, rms = np.zeros(n), np.zeros((n, 2)), np.zeros(n, bool), np.full(n, np.nan)
    r = np.flatnonzero(has_next)
    g = nxt[r]
    p = fut[r, 0:20, :2].astype(float)
    q = np.concatenate([past[g, 6:16, :2], fut[g, 0:10, :2]], axis=1).astype(float)
    pc, qc = p.mean(1, keepdims=True), q.mean(1, keepdims=True)
    pt, qt = p - pc, q - qc
    cross = (qt[..., 0] * pt[..., 1] - qt[..., 1] * pt[..., 0]).sum(1)
    dot = (qt * pt).sum((1, 2))
    th = np.arctan2(cross, dot)
    spread = np.sqrt((qt ** 2).sum(2).mean(1))
    th = np.where(spread < SPREAD_MIN, 0.0, th)
    c, s = np.cos(th), np.sin(th)
    rq = np.stack([c[:, None] * qt[..., 0] - s[:, None] * qt[..., 1], s[:, None] * qt[..., 0] + c[:, None] * qt[..., 1]], -1)
    res = np.sqrt(((rq - pt) ** 2).sum(2).mean(1))
    tr = pc[:, 0] - np.stack([c * qc[:, 0, 0] - s * qc[:, 0, 1], s * qc[:, 0, 0] + c * qc[:, 0, 1]], -1)
    theta[r], t[r], rms[r] = th, tr, res
    ok[r] = res <= RMS_MAX
    return theta, t, ok, rms


def chain_paths(fut, nxt, link_ok, theta, t, rows):
    """Chained future paths for `rows` (indices into the frame arrays): list of (m_i, 2) arrays in each row's own ego frame, first row (0, 0),
    plus the usable duration (s)."""
    out = []
    for r in rows:
        pts, T_c, T_s, T_t, cur, links = [np.zeros((1, 2))], 1.0, 0.0, np.zeros(2), r, 0
        while True:
            seg_n = 10 if (links < MAX_LINKS and nxt[cur] >= 0 and link_ok[cur]) else 20
            seg = fut[cur, :seg_n, :2].astype(float)
            pts.append(np.stack([T_c * seg[:, 0] - T_s * seg[:, 1], T_s * seg[:, 0] + T_c * seg[:, 1]], -1) + T_t)
            if seg_n == 20:
                break
            # compose: frame(g) -> frame(cur) -> frame(f)
            th, tt = theta[cur], t[cur]
            c, s = np.cos(th), np.sin(th)
            T_t = np.array([T_c * tt[0] - T_s * tt[1], T_s * tt[0] + T_c * tt[1]]) + T_t
            T_c, T_s = T_c * c - T_s * s, T_s * c + T_c * s
            cur, links = nxt[cur], links + 1
            P = np.concatenate(pts)
            if np.hypot(*np.diff(P, axis=0).T).sum() >= PATH_M:
                break
        out.append((np.concatenate(pts), 2.5 * links + (5.0 if seg_n == 20 else 0.0)))
    return out


_G = {}


def _init(past, fut, nxt, link_ok, theta, t):  # noqa: ARG001
    _G.update(fut=fut, nxt=nxt, link_ok=link_ok, theta=theta, t=t)


def work(rows):
    G = _G
    res = []
    for (path, dur), r in zip(chain_paths(G["fut"], G["nxt"], G["link_ok"], G["theta"], G["t"], rows), rows):
        h = RP.hindsight(path)
        h["dur"] = dur
        res.append(h)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit-seq", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from jevdrive import par, waymo as W
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = Path(a.out) if a.out else data_dir() / "processed" / "op_route_cmd" / "wod"
    with Run("op_route_cmd", "wod" + (f"-s{a.limit_seq}" if a.limit_seq else "")) as run:
        tr, va, te = splits.load("wod/train"), splits.load("wod/val"), splits.load("wod/test")
        run.use_split(tr)
        run.use_split(va)
        splits.check_disjoint(tr, va, te)
        df = W.load_index().reset_index(drop=True)
        past, fut = W.load_ego()
        sel = (df.split.isin(["train", "val"]) & df.has_future).to_numpy()
        assert not df.sequence[sel].isin(set(te)).any() and tr.mask(df.sequence[sel & (df.split == "train").to_numpy()]).all()
        if a.limit_seq:
            keep = sorted(set(df.sequence[sel]))[: a.limit_seq]
            sel &= df.sequence.isin(keep).to_numpy()
        key = {(s, f): i for i, (s, f) in enumerate(zip(df.sequence, df.frame))}
        nxt = np.array([key.get((s, f + LINK), -1) for s, f in zip(df.sequence, df.frame)])
        nxt[~sel | ((nxt >= 0) & ~np.where(nxt >= 0, sel[np.maximum(nxt, 0)], False))] = -1
        theta, tt, ok, rms = fit_links(past, fut, nxt >= 0, nxt)
        run.info(f"links: {int((nxt >= 0).sum())} candidate, {int(ok.sum())} ok (rms <= {RMS_MAX} m), rms quantiles "
                 f"{np.nanpercentile(rms, [50, 90, 99, 99.9]).round(3).tolist()}")
        rows = np.flatnonzero(sel)
        chunks = [rows[i:i + 500] for i in range(0, len(rows), 500)]
        t0 = time.time()
        _init(past, fut, nxt, ok, theta, tt)              # module globals, inherited by the forked workers
        res = par.pmap(work, chunks, run=run, workers=a.workers or None)
        res.raise_if_failed()
        H = [h for part in res.values for h in part]
        run.info(f"labelled {len(H)} frames in {time.time() - t0:.0f} s")
        d = df.iloc[rows]
        f32 = lambda k: np.array([h[k] for h in H], np.float32)  # noqa: E731
        v0 = np.hypot(past[rows, -1, 2], past[rows, -1, 3])
        sidecar = dict(
            id=np.array([f"{s}-{f:03d}" for s, f in zip(d.sequence, d.frame)]), split=d.split.to_numpy().astype(str),
            cluster=d.sequence.to_numpy().astype(str), scene=d.sequence.to_numpy().astype(str), v0=v0.astype(np.float32),
            cmd=np.array(W.INTENTS)[d.intent.to_numpy()].astype(str),
            poly=np.stack([h["poly"] for h in H]), pmask=np.stack([h["pmask"] for h in H]), plen=f32("plen"), dur=f32("dur"),
            turn_deg=f32("turn_deg"), turn_s=f32("turn_s"), turn_end_s=f32("turn_end_s"), turn_rmin=f32("turn_rmin"),
            n_turn=np.array([h["n_turn"] for h in H], np.int16), in_turn=np.array([h["in_turn"] for h in H], bool), max_turn_deg=f32("max_turn_deg"),
            jct_s=np.full(len(H), np.nan, np.float32), turn_junction=np.zeros(len(H), bool), jct_dist=np.full(len(H), np.nan, np.float32),
            n_exit=np.full(len(H), -1, np.int16), taken_cls=np.full(len(H), "", "U1"), status=np.full(len(H), "", "U1"))
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "route.npz", **sidecar)
        run.summary.update(n=len(H), out=str(out / "route.npz"))


if __name__ == "__main__":
    main()
