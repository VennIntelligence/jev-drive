"""Drivable-area signed distance field per op_adapt_H navtrain sample (off-road hinge penalty for the op_route_ft fine-tune).

  envs/navsim2 on the box (nuPlan map API, CPU only):
  NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim \
    python experiments/op_route_ft/scripts/drivable_sdf.py [--workers 40] [--limit N]

Drivable surface = union of nuPlan map polygons of LANE, LANE_CONNECTOR, INTERSECTION, ROADBLOCK, ROADBLOCK_CONNECTOR, CARPARK_AREA (exactly
lib/route_neg._DRIVABLE; the generic raster-only drivable areas and the NAVSIM scorer polygons are NOT used).
Ego pose = ego2global of the OpenScene log frame of the token (nuPlan ego pose = rear axle). Grid is in the rear-axle frame at t0:
sdf[i, j] is the cell centre x = x0 + (i + .5) res (forward, axis 1 = H), y = y0 + (j + .5) res (left, axis 2 = W). Positive inside, metres.
Computed on a 0.25 m raster (polygons filled, EDT both sides, signed = d_in - d_out), 2x2-averaged to the 0.5 m grid.
Output $DATA_DIR/runs/op_route_ft/drivable/nav.npz.
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
from jevdrive.common import data_dir  # noqa: E402

X0, Y0, RES, H, W = -10.0, -40.0, 0.5, 160, 160
FINE = 0.25
PAD = 10.0                                   # canvas margin (m) so the EDT is right at the grid edge and fut points beyond it still read
CX0, CY0 = X0 - PAD, Y0 - PAD
NX, NY = int((H * RES + 2 * PAD) / FINE), int((W * RES + 2 * PAD) / FINE)
LAYERS = ("LANE", "LANE_CONNECTOR", "INTERSECTION", "ROADBLOCK", "ROADBLOCK_CONNECTOR", "CARPARK_AREA")
RADIUS = 110.0                               # canvas half-diagonal ~ 85 m from the pose; polygons are fetched by centre distance
INSIDE_TOL = 0.4
OUT = data_dir() / "runs" / "op_route_ft" / "drivable"
INV = data_dir() / "processed" / "op_common_cause" / "pair_inventory" / "navtrain_frames.parquet"
_maps = {}


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _map(loc):
    if loc not in _maps:
        from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
        db = get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))
        _maps[loc] = NuPlanMapFactory(db).build_map_from_name(loc)
    return _maps[loc]


def _ring(xy, pose):
    """global ring (k, 2) -> canvas fixed-point (col, row) = (y, x) pixel coords (centre = integer), 4 fractional bits."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(xy) - pose[:2]
    x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]            # ego frame
    return np.round(np.stack([(y - CY0) / FINE - 0.5, (x - CX0) / FINE - 0.5], -1) * 16).astype(np.int32)


def sdf_fine(m, pose):
    """(NX, NY) float32 signed distance (m, + inside) on the 0.25 m canvas; pixel (i, j) centre = (CX0 + (i+.5) FINE, CY0 + (j+.5) FINE)."""
    import cv2
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from scipy.ndimage import distance_transform_edt as edt
    from shapely.ops import unary_union
    objs = m.get_proximal_map_objects(Point2D(float(pose[0]), float(pose[1])), RADIUS, [getattr(L, n) for n in LAYERS])
    polys = [o.polygon for v in objs.values() for o in v]
    img = np.zeros((NX, NY), np.uint8)
    if polys:
        u = unary_union(polys)
        for g in (u.geoms if hasattr(u, "geoms") else [u]):
            cv2.fillPoly(img, [_ring(g.exterior.coords, pose)], 1, shift=4)
            for h in g.interiors:
                cv2.fillPoly(img, [_ring(h.coords, pose)], 0, shift=4)
    ins = img > 0
    if not ins.any():
        return np.full((NX, NY), -10.0, np.float32)
    d = edt(ins) - edt(~ins)
    d = np.where(ins, d - 0.5, d + 0.5)                                    # the boundary lies half a pixel from the nearest centres
    return (d * FINE).astype(np.float32)


def work(job):
    log, rows = job                                                      # rows: (token, frame_idx, fut20 (20, 2))
    fr = pickle.load(open(os.path.join(os.environ["OPENSCENE_DATA_ROOT"], "navsim_logs", "trainval", log + ".pkl"), "rb"))
    m = _map(fr[0]["map_location"])
    from scipy.ndimage import map_coordinates
    out = []
    for tok, i, fut in rows:
        f = fr[i]
        assert f["token"] == tok
        pose = np.array([f["ego2global_translation"][0], f["ego2global_translation"][1], _quat_yaw(f["ego2global_rotation"])])
        try:
            d = sdf_fine(m, pose)
            sub = d[int(PAD / FINE):NX - int(PAD / FINE), int(PAD / FINE):NY - int(PAD / FINE)]
            g = sub.reshape(H, 2, W, 2).mean((1, 3))                     # fine 0.25 m cells -> 0.5 m cell centres
            ij = np.stack([(fut[:, 0] - CX0) / FINE - 0.5, (fut[:, 1] - CY0) / FINE - 0.5])
            sl = map_coordinates(d, ij, order=1, mode="nearest")
            out.append((tok, True, g.astype(np.float16), float(sl.min()), pose.tolist(), fr[0]["map_location"]))
        except Exception as e:  # noqa: BLE001
            out.append((tok, False, None, np.nan, pose.tolist(), f"{type(e).__name__}: {e}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "nav.npz"))
    a = ap.parse_args()
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_route_ft", "drivable_sdf") as run:
        with np.load(data_dir() / "runs" / "op_adapt_H" / "samples" / "nav" / "tab.npz", allow_pickle=True) as z:   # = Samples("nav").t (that import needs onnx, absent in navsim2)
            S = {k: z[k] for k in ("id", "fut20")}
        for sp in ("navsim/op-adapt-h-nav-train", "navsim/op-adapt-h-nav-dev"):
            run.use_split(splits.load(sp))
        ids = S["id"].astype(str)
        n = a.limit or len(ids)
        df = pd.read_parquet(INV).set_index("token")
        fi = df.loc[ids[:n]]
        jobs = {}
        for k in range(n):
            jobs.setdefault(fi.log.iloc[k], []).append((ids[k], int(fi.frame_idx.iloc[k]), S["fut20"][k].astype(np.float64)))
        run.info(f"{n} samples in {len(jobs)} logs")
        t0 = time.time()
        res = par.pmap(work, list(jobs.items()), run=run, workers=a.workers)
        res.raise_if_failed()
        R = {r[0]: r for part in res.values for r in part}
        wall = time.time() - t0
        sdf = np.full((n, H, W), 10.0, np.float16)
        ok = np.zeros(n, bool)
        mn = np.full(n, np.nan, np.float32)
        for k in range(n):
            r = R[ids[k]]
            if r[1]:
                sdf[k], ok[k], mn[k] = r[2], True, r[3]
        bad = [R[ids[k]][5] for k in range(n) if not ok[k]][:5]
        inside = ok & (mn >= -INSIDE_TOL)
        pose = np.array([R[t][4] for t in ids[:n]])
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, id=ids[:n], sdf=sdf, x0=X0, y0=Y0, res=RES, ok=ok, logged_inside=inside, min_sdf_logged=mn, pose=pose,
                 layers=np.array(LAYERS), axes="sdf[i, j]: i = x forward (H), j = y left (W); cell centre x0+(i+.5)res")
        run.summary.update(n=n, ok=int(ok.sum()), logged_inside=float(inside.sum() / max(ok.sum(), 1)), compute_s=wall, fail_examples=bad, out=str(out))
        run.info(f"ok {ok.sum()}/{n}, logged_inside {inside.sum() / max(ok.sum(), 1):.4f}, {wall:.0f} s, fails {bad}")


if __name__ == "__main__":
    main()
