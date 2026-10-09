"""op_probe labels: drivable signed-distance raster per NAVSIM token, from the nuPlan map API with the scorer's own layers
(plans/2026-10-06-dac-localize-prereg.md section 2). envs/navsim2 on the box, CPU only:

  $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_labels.py build --split navtest [--limit N] [--workers 64]
  $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_labels.py build --split navtrain --shards 2 3 4 5 6
  $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_labels.py check [--n 300]    # vs the v2 navtest metric cache

Drivable surface = ROADBLOCK | INTERSECTION | CARPARK_AREA: the polygon types of PDMScorer's NON_DRIVABLE_AREA check (PDMDrivableMap holds
roadblock connectors only as LANE_CONNECTOR polygons, which that check ignores; DRIVABLE_AREA is never cached). Fetched within 80 m (the
scorer's map uses 50 m; 80 m covers the raster). Frame: rear axle at t0 (nuPlan ego pose), x forward, y left. SDF (m, + inside) on a 0.25 m
canvas (EDT both sides), stored at 0.5 m: sdf[i, j] at cell centre x = X0 + (i + .5) RES, y = Y0 + (j + .5) RES, float16, clipped to +-20 m.
Out: $DATA_DIR/runs/op_probe/labels/<split>.npz (tokens, log, pose_global, sdf, ok).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(D / "datasets/navsim"))
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

X0, Y0, RES, NH, NW = -8.0, -24.0, 0.5, 128, 96          # stored raster: x in [-8, 56), y in [-24, 24)
FINE, PAD = 0.25, 8.0
CX0, CY0 = X0 - PAD, Y0 - PAD
NX, NY = int((NH * RES + 2 * PAD) / FINE), int((NW * RES + 2 * PAD) / FINE)
LAYERS = ("ROADBLOCK", "INTERSECTION", "CARPARK_AREA")      # the scorer's drivable_area_idcs (roadblock connectors only enter as lane connectors, not drivable)
RADIUS = 80.0
OUT = D / "runs" / "op_probe" / "labels"
if os.environ.get("OPB_LAYERS"):                            # another layer list into another directory (experiments/body1, Amendment 4 item C); read
    LAYERS = tuple(os.environ["OPB_LAYERS"].split(","))     # from the environment so that worker processes of any start method see it
    OUT = Path(os.environ["OPB_OUT"])
LOGDIR = {"navtest": "test", "navtrain": "trainval"}
_maps = {}


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _map(loc):
    if loc not in _maps:
        from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
        db = get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"])
        _maps[loc] = NuPlanMapFactory(db).build_map_from_name(loc)
    return _maps[loc]


def _ring(xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(xy)[:, :2] - pose[:2]
    x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return np.round(np.stack([(y - CY0) / FINE - 0.5, (x - CX0) / FINE - 0.5], -1) * 16).astype(np.int32)


def sdf_raster(m, pose):
    """(NH, NW) float32 SDF at the stored 0.5 m cell centres (2 x 2 mean of the 0.25 m canvas)."""
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
        return np.full((NH, NW), -20.0, np.float32)
    d = edt(ins) - edt(~ins)
    d = np.where(ins, d - 0.5, d + 0.5) * FINE
    p = int(PAD / FINE)
    return d[p:NX - p, p:NY - p].reshape(NH, 2, NW, 2).mean((1, 3)).astype(np.float32)


def work(job):
    split, log, toks = job
    fr = pickle.load(open(Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / LOGDIR[split] / f"{log}.pkl", "rb"))
    at = {f["token"]: f for f in fr}
    m = _map(fr[0]["map_location"])
    out = []
    for t in toks:
        f = at[t]
        pose = np.array([f["ego2global_translation"][0], f["ego2global_translation"][1], _quat_yaw(f["ego2global_rotation"])])
        try:
            out.append((t, pose, np.clip(sdf_raster(m, pose), -20, 20).astype(np.float16), ""))
        except Exception as e:  # noqa: BLE001
            out.append((t, pose, None, f"{type(e).__name__}: {e}"))
    return out


def tokens_of(split, shards):
    """(tokens, logs) in the order of the op_parity caches: navtest = lb_navtest/tab.npz, navtrain = navtrain_full.s<k>of12 shards."""
    cr = D / "runs" / "op_parity" / "cache"
    dirs = ["lb_navtest"] if split == "navtest" else [f"navtrain_full.s{k}of12" for k in shards]
    tabs = [np.load(cr / d / "tab.npz") for d in dirs]
    return np.concatenate([t["names"] for t in tabs]), np.concatenate([t["log"] for t in tabs])


def cmd_build(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    tag = a.split if a.split == "navtest" else f"navtrain_s{''.join(map(str, a.shards))}"
    with Run("op_probe", f"labels-{tag}", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{a.split}"))
        toks, logs = tokens_of(a.split, a.shards)
        if a.limit:
            toks, logs = toks[:a.limit], logs[:a.limit]
        jobs = {}
        for t, lg in zip(toks, logs):
            jobs.setdefault(lg, []).append(t)
        run.info(f"{tag}: {len(toks)} tokens in {len(jobs)} logs, {a.workers} workers")
        t0 = time.time()
        res = par.pmap(work, [(a.split, lg, ts) for lg, ts in jobs.items()], run=run, workers=a.workers)
        res.raise_if_failed()
        R = {r[0]: r for part in res.values for r in part}
        sdf = np.full((len(toks), NH, NW), -20, np.float16)
        ok = np.zeros(len(toks), bool)
        pose = np.zeros((len(toks), 3))
        for k, t in enumerate(toks):
            r = R[t]
            pose[k] = r[1]
            if r[2] is not None:
                sdf[k], ok[k] = r[2], True
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / f"{tag}{'-lim' if a.limit else ''}.npz"
        np.savez(f, tokens=toks, log=logs, pose_global=pose, sdf=sdf, ok=ok, x0=X0, y0=Y0, res=RES, layers=np.array(LAYERS))
        bad = [R[t][3] for t in toks if R[t][2] is None][:5]
        run.summary.update(n=len(toks), ok=int(ok.sum()), compute_s=time.time() - t0, out=str(f), fails=bad)
        run.info(f"ok {ok.sum()}/{len(toks)} in {time.time() - t0:.0f} s -> {f}; fails {bad}")


def _check_one(args):
    import glob
    import lzma
    tok, sdf, pose = args
    fs = glob.glob(str(D / "runs/navsim/metric_cache/v2_navtest/*/unknown" / tok / "metric_cache.pkl"))
    if not fs:
        return tok, np.nan, 0, np.nan
    with lzma.open(fs[0], "rb") as f:
        mc = pickle.load(f)
    am = mc.drivable_area_map
    xs, ys = X0 + (np.arange(NH) + 0.5) * RES, Y0 + (np.arange(NW) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    sel = (np.hypot(gx, gy) < 45) & (np.abs(sdf) > 0.3)
    px, py = gx[sel], gy[sel]
    o = np.array(mc.ego_state.rear_axle.serialize())
    assert np.hypot(*(o[:2] - pose[:2])) < 0.05, "pose mismatch"
    c, s = np.cos(o[2]), np.sin(o[2])
    G = np.stack([o[0] + c * px - s * py, o[1] + s * px + c * py], -1)
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    idc = am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    ins = am.points_in_polygons(G[None])                       # (n_poly, 1, n)
    inside = ins[idc].any(0)[0]
    agree = float(np.mean(inside == (sdf[sel] > 0)))
    # the logged future path inside the label surface
    return tok, agree, int(sel.sum()), float(np.mean(inside))


def cmd_check(a):
    from multiprocessing import Pool
    from jevdrive.run import Run
    z = dict(np.load(OUT / "navtest.npz"))                      # materialised: NpzFile re-reads the whole array on every z[key]
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(z["ok"]), a.n, replace=False)
    with Run("op_probe", "labels-check", config=vars(a)) as run:
        with Pool(a.workers) as p:
            res = p.map(_check_one, [(z["tokens"][i], z["sdf"][i].astype(np.float32), z["pose_global"][i]) for i in pick], chunksize=1)
        ag = np.array([r[1] for r in res])
        tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
        assert (tab["names"] == z["tokens"]).all()
        fut = tab["fut"]
        inside = []
        from scipy.ndimage import map_coordinates
        for k in range(len(z["tokens"])):
            f = fut[k]
            if not z["ok"][k] or np.isnan(f[0, 0]):
                continue
            ij = np.stack([(f[:, 0] - X0) / RES - 0.5, (f[:, 1] - Y0) / RES - 0.5])
            inside.append(map_coordinates(z["sdf"][k].astype(np.float32), ij, order=1, mode="nearest").min() >= -0.4)
        run.summary.update(n=len(res), agree_mean=float(np.nanmean(ag)), agree_p05=float(np.nanpercentile(ag, 5)),
                           frac_tokens_agree98=float(np.nanmean(ag >= 0.98)), log_future_inside=float(np.mean(inside)), n_future=len(inside))
        run.info(str(run.summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--split", choices=["navtest", "navtrain"], required=True)
    p.add_argument("--shards", type=int, nargs="*", default=[2, 3, 4, 5, 6])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=64)
    p = sp.add_parser("check")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
