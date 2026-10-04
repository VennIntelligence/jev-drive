"""Route polylines for NAVSIM navtrain (hindsight label from the logged future of the OpenScene log, whole log at 2 Hz).

  envs/navsim2 on the box (nuPlan map API, CPU only):
  NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim \
    python experiments/op_route_cmd/scripts/route_nav.py [--limit-logs N] [--workers N]

One row per navtrain token (split navsim/navtrain, the frames of the decision-93 pair inventory), keyed by token = `id` of op_adapt_H's nav
sample table. Path: the ego poses of the same log from the token onwards, in the ego frame (rear axle, x forward, y left), until 160 m of
path or the end of the log (cap 180 frames = 90 s); `lib/route_poly.hindsight` makes the 10 m / 150 m polyline and the turn statistics.
Map-derived extras (nuPlan lane connectors / intersections, never part of the model input): `jct_s` arc length to the first connector or
intersection on the driven path, `turn_junction` = the first turn >= 25 deg lies mostly (>= 30% of its points) inside connectors / intersections
(else it is a road bend), `jct_dist` / `n_exit` / `taken_cls` from the pair inventory (decision 93, branching lane within 30 m, any exit class).
Output $DATA_DIR/processed/op_route_cmd/navtrain/route.npz (+ summary.json by route_report.py).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib")]
import route_poly as RP  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

MAX_FRAMES, PATH_M = 180, 160.0
CMD = ["left", "straight", "right", "unknown"]
OUT = data_dir() / "processed" / "op_route_cmd" / "navtrain"
INV = data_dir() / "processed" / "op_common_cause" / "pair_inventory" / "navtrain_frames.parquet"

_maps = {}


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _conn_fn(loc):
    """(x, y) -> bool: inside a lane connector or intersection polygon of the nuPlan map `loc`."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    if loc not in _maps:
        from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
        db = get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))
        _maps[loc] = NuPlanMapFactory(db).build_map_from_name(loc)
    m, cache = _maps[loc], {}

    def f(x, y):
        k = (int(x // 1.0), int(y // 1.0))
        if k not in cache:
            p = Point2D(float(x), float(y))
            cache[k] = bool(m.get_all_map_objects(p, L.LANE_CONNECTOR) or m.get_all_map_objects(p, L.INTERSECTION))
        return cache[k]
    return f


def label_frames(xs, idx, conn=None):
    """xs (n, 3) global (x, y, yaw) of the consecutive 2 Hz frames of one log; idx = frame indices to label.
    Returns a list of dicts: hindsight fields + jct_s, turn_junction."""
    n, out = len(xs), []
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xs[:, :2], axis=0).T))]
    for i in idx:
        j1 = int(min(n, i + MAX_FRAMES + 1, np.searchsorted(cum, cum[i] + PATH_M) + 2))
        d = xs[i:j1, :2] - xs[i, :2]
        c, s = np.cos(xs[i, 2]), np.sin(xs[i, 2])
        loc = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)     # ego frame of frame i
        h = RP.hindsight(loc, want_path=True)
        P = h.pop("path", None)
        h["jct_s"], h["turn_junction"] = np.nan, False
        if conn is not None and P is not None:
            g = np.r_[xs[i, 0] + c * P[:, 0] - s * P[:, 1], xs[i, 1] + s * P[:, 0] + c * P[:, 1]].reshape(2, -1)
            inn = np.array([conn(x, y) for x, y in g[:, ::4].T])                        # every 2 m of the 0.5 m grid
            lim = int(h["plen"] / 2.0) + 1
            if inn[:lim].any():
                h["jct_s"] = 2.0 * float(np.argmax(inn[:lim]))
            if not np.isnan(h["turn_s"]):
                a, b = int(h["turn_s"] / 2.0), int(h["turn_end_s"] / 2.0) + 1
                h["turn_junction"] = bool(inn[a:b].mean() >= 0.3)
        out.append(h)
    return out


def work(job):
    log, rows = job
    fr = pickle.load(open(os.path.join(os.environ["OPENSCENE_DATA_ROOT"], "navsim_logs", "trainval", log + ".pkl"), "rb"))
    xs = np.array([(f["ego2global_translation"][0], f["ego2global_translation"][1], _quat_yaw(f["ego2global_rotation"])) for f in fr])
    ts = np.array([f["timestamp"] for f in fr]) / 1e6
    assert (np.abs(np.diff(ts) - 0.5) < 0.05).all(), f"{log}: not a contiguous 2 Hz log"
    tok = [f["token"] for f in fr]
    idx = np.array([r[1] for r in rows])
    assert [tok[i] for i in idx] == [r[0] for r in rows]
    res = label_frames(xs, idx, _conn_fn(fr[0]["map_location"]))
    return [(r[0], h) for r, h in zip(rows, res)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_route_cmd", "navtrain" + (f"-l{a.limit_logs}" if a.limit_logs else "")) as run:
        sp = splits.load("navsim/navtrain")
        run.use_split(sp)
        for other in ("navsim/navtest", "navsim/navhard_two_stage"):
            splits.check_disjoint(sp, splits.load(other))
        df = pd.read_parquet(INV)
        assert sp.mask(df.token).all(), "inventory token outside navsim/navtrain"
        df = df.sort_values(["log", "frame_idx"])
        logs = sorted(df.log.unique())[: a.limit_logs or None]
        df = df[df.log.isin(logs)]
        jobs = [(l, list(zip(g.token, g.frame_idx))) for l, g in df.groupby("log")]
        run.info(f"{len(df)} tokens in {len(jobs)} logs")
        t0 = time.time()
        res = par.pmap(work, jobs, run=run, workers=a.workers or None)
        res.raise_if_failed()
        H = {t: h for part in res.values for t, h in part}
        run.info(f"labelled {len(H)} tokens in {time.time() - t0:.0f} s")
        d = df.set_index("token").loc[list(H)]
        ids = np.array(list(H))
        f32 = lambda k: np.array([H[t][k] for t in ids], np.float32)  # noqa: E731
        sidecar = dict(
            id=ids, split=np.full(len(ids), "navtrain"), cluster=d.log.to_numpy().astype(str), scene=d.scene.to_numpy().astype(str),
            v0=d.v.to_numpy().astype(np.float32), cmd=d.cmd.to_numpy().astype(str),
            poly=np.stack([H[t]["poly"] for t in ids]), pmask=np.stack([H[t]["pmask"] for t in ids]),
            plen=f32("plen"), turn_deg=f32("turn_deg"), turn_s=f32("turn_s"), turn_end_s=f32("turn_end_s"), turn_rmin=f32("turn_rmin"),
            n_turn=np.array([H[t]["n_turn"] for t in ids], np.int16), in_turn=np.array([H[t]["in_turn"] for t in ids], bool),
            max_turn_deg=f32("max_turn_deg"), jct_s=f32("jct_s"), turn_junction=np.array([H[t]["turn_junction"] for t in ids], bool),
            jct_dist=np.where(d.status.to_numpy() == "branch", d.dist.to_numpy(), np.nan).astype(np.float32),
            n_exit=np.where(d.status.to_numpy() == "branch", d.n_branch.to_numpy(), -1).astype(np.int16),
            taken_cls=d.taken.to_numpy().astype(str), status=d.status.to_numpy().astype(str))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "route.npz", **sidecar)
        run.summary.update(n=len(ids), out=str(out / "route.npz"))


if __name__ == "__main__":
    main()
