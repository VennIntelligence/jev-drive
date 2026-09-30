"""op-adapt round 2, package S: maps, slot contexts, registered checks and the score tables of the S_jev scorer
(jevdrive/op_adapt_score.py; todos/2026-09-29-op-adapt-r2-prereg.md §2.1, §5 item 1, §4.4; interface: section S of
tmp/2026-09-30-op-adapt-r2-build.md).

  points      route points per CARLA town (Cosmos pass-1 runs, P5 v1 BA runs) -> R/maps/carla/points/<town>.npy
  maps-carla  OpenDRIVE lane samples (scripts/op_adapt_carla_lanes.py, CARLA env) -> R/maps/carla/<town>.npz raster
  nus-scenes  nuScenes per-scene arrays (ego poses, keyframes, boxes, CAM_FRONT visibility) -> R/scene/nus/<scene>.npz
  maps-nus    nuScenes map expansion (drivable_area, road_segment intersections, lanes + connectors with arcline
              direction) around every scene's ego path -> R/maps/nus/<location>.npz raster
  v1          NC vs WL cg on WL-1's 2 814 branch runs            -> R/checks/V1.json
  v2          nuScenes val log trajectories, S_jev >= 0.8         -> R/checks/V2.json (+ V5 part b: log DDC = 1)
  v345        V3 / V4 (sim pairs) and V6 (offset dev) from the score tables -> R/checks/V{3,4,6}.json
  score       R/score/<domain>.npz for simC simK nus off p5       (needs C's index + teacher)
  sanity      §4.4 scorer sanity                                   -> R/score/sanity.json
CARLA geometry is mirrored to a right-handed frame here (y -> -y, yaw -> -yaw); nuScenes / NAVSIM are right-handed.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import op_adapt_score as S
from .common import data_dir, get_logger

log = get_logger(__name__)
REAR_TF = 1.388633220                        # CARLA hero transform ahead of its rear axle (meta.json rear_axle_x)
CAM_X = {"sim": 1.519, "p5": 1.519}          # openpilot origin (the camera) ahead of the rear axle (cosmos_pair_agent.CAM)
HFOV = {"sim": 64.0, "p5": 52.08, "nav": 63.7}          # Cosmos camera; P5 Waymo front (meta.json fov); CAM_F0
VIS_PX, VIS_RANGE = 68.0, 60.0
N_CTX = 9
CARLA_KIND = (("walker.", S.PED), ("static.", S.STATIC))
BIKES = ("bh.crossbike", "diamondback", "gazelle", "harley", "kawasaki", "yamaha", "vespa", "bicycle", "century")


def R(*p) -> Path:
    d = data_dir() / "runs" / "op_adapt_r2"
    return d.joinpath(*p)


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def cores() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4


# ================================================================ rasterisation

def rasterize(polys, keep: set | None = None) -> S.TileMap:
    """polys: iterable of (polygon (m, 2) world, heading rad or None, lane id (0 none), flags). Lane polygons carry one
    travel direction each (a lane is a chain of short quads); a pixel keeps up to two lane ids / headings. keep: tile
    (i, j) set to rasterise (None: every tile a polygon touches)."""
    import cv2
    tiles: dict = {}

    def tile(i, j):
        t = tiles.get((i, j))
        if t is None:
            t = tiles[i, j] = (np.zeros((S.TILE, S.TILE), np.uint8), np.zeros((2, S.TILE, S.TILE), np.int32),
                               np.full((2, S.TILE, S.TILE), 255, np.uint8))
        return t
    for poly, head, lid, fl in polys:
        poly = np.asarray(poly, float)
        lo, hi = poly.min(0), poly.max(0)
        for i in range(int(math.floor(lo[0] / S.TM)), int(math.floor(hi[0] / S.TM)) + 1):
            for j in range(int(math.floor(lo[1] / S.TM)), int(math.floor(hi[1] / S.TM)) + 1):
                if keep is not None and (i, j) not in keep:
                    continue
                q = (poly - (i * S.TM, j * S.TM)) / S.RES - 0.5          # pixel-centre coordinates
                x0, y0 = max(int(math.floor(q[:, 0].min())), 0), max(int(math.floor(q[:, 1].min())), 0)
                x1, y1 = min(int(math.ceil(q[:, 0].max())) + 1, S.TILE), min(int(math.ceil(q[:, 1].max())) + 1, S.TILE)
                if x1 <= x0 or y1 <= y0:
                    continue
                m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(m, [np.round((q - (x0, y0)) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
                m = m.astype(bool)
                if not m.any():
                    continue
                F, L, Hd = tile(i, j)
                F[y0:y1, x0:x1][m] |= fl
                if lid:
                    hq = 255 if head is None else int(round((head % (2 * math.pi)) / (2 * math.pi) * 255)) % 255
                    l0, l1 = L[0, y0:y1, x0:x1], L[1, y0:y1, x0:x1]
                    h0, h1 = Hd[0, y0:y1, x0:x1], Hd[1, y0:y1, x0:x1]
                    a = m & ((l0 == 0) | (l0 == lid))
                    l0[a], h0[a] = lid, hq
                    b = m & ~a & ((l1 == 0) | (l1 == lid))
                    l1[b], h1[b] = lid, hq
    keys = sorted(tiles)
    if not keys:
        return S.TileMap(np.zeros((0, 2), np.int64), np.zeros((0, S.TILE, S.TILE), np.uint8),
                         np.zeros((0, 2, S.TILE, S.TILE), np.int32), np.zeros((0, 2, S.TILE, S.TILE), np.uint8))
    return S.TileMap(np.array(keys), np.stack([tiles[k][0] for k in keys]), np.stack([tiles[k][1] for k in keys]),
                     np.stack([tiles[k][2] for k in keys]))


def save_map(tm: S.TileMap, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ij=tm.ij, flags=tm.flags, lane=tm.lane, head=tm.head)
    tmp.replace(path)


def keep_tiles(xy: np.ndarray, radius: float = 40.0) -> set:
    """Tiles within `radius` of any point (right-handed world)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    r = int(math.ceil(radius / S.TM))
    base = {(int(i), int(j)) for i, j in np.floor(xy / S.TM).astype(int)}
    return {(i + a, j + b) for i, j in base for a in range(-r, r + 1) for b in range(-r, r + 1)}


# ================================================================ CARLA maps

def _town(meta_town: str) -> str:
    return meta_town.rstrip("/").split("/")[-1]


def _expert_one(adir: str):
    a = Path(adir)
    try:
        town = _town(json.loads((a / "meta.json").read_text())["town"])
        p = pd.read_json(a / "pose.jsonl", lines=True)[["x", "y", "z", "yaw"]].to_numpy(float)[::5]
        return town, p
    except Exception:  # noqa: BLE001
        return None


def expert_poses(workers: int = 24) -> dict:
    """Pass-1 driven poses (every 5th tick) per town: R/maps/carla/expert/<town>.npy (x, y, z, yaw_deg; CARLA frame)."""
    from multiprocessing import Pool
    dirs = [str(p) for g in ("runs/cosmos_full/gen/attempts", "runs/p5v1/gen-ba/attempts")
            for p in (data_dir() / g).glob("*/*") if (p / "meta.json").exists()]
    out: dict = {}
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_expert_one, dirs, chunksize=8):
            if r is not None:
                out.setdefault(r[0], []).append(r[1])
    R("maps", "carla", "expert").mkdir(parents=True, exist_ok=True)
    for town, parts in out.items():
        np.save(R("maps", "carla", "expert", f"{town}.npy"), np.vstack(parts).astype(np.float32))
    return {t: int(sum(len(p) for p in v)) for t, v in out.items()}


def _route_points_one(adir: str):
    a = Path(adir)
    try:
        town = _town(json.loads((a / "meta.json").read_text())["town"])
        r = pd.read_json(a / "route.json")[["x", "y"]].to_numpy(float)
        p = pd.read_json(a / "pose.jsonl", lines=True)[["x", "y"]].to_numpy(float)
        return town, np.unique(np.round(np.vstack([r, p]) / 2.0), axis=0) * 2.0
    except Exception:  # noqa: BLE001 (an unfinished attempt)
        return None


def route_points(workers: int = 24) -> dict:
    """Every Cosmos full pass-1 attempt and every P5 v1 BA attempt: route + driven path points per town (CARLA frame)."""
    from multiprocessing import Pool
    dirs = [str(p) for g in ("runs/cosmos_full/gen/attempts", "runs/p5v1/gen-ba/attempts")
            for p in (data_dir() / g).glob("*/*") if (p / "meta.json").exists()]
    out: dict = {}
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_route_points_one, dirs, chunksize=8):
            if r is not None:
                out.setdefault(r[0], []).append(r[1])
    res = {}
    for town, parts in out.items():
        pts = np.unique(np.vstack(parts), axis=0)
        f = R("maps", "carla", "points", f"{town}.npy")
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, pts.astype(np.float32))
        res[town] = len(pts)
    log.info("route points: %s (%d attempts)", res, len(dirs))
    return res


def xodr_path(town: str) -> Path:
    base = data_dir() / "third_party/carla/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps"
    for p in (base / "OpenDrive" / f"{town}.xodr", base / town / "OpenDrive" / f"{town}.xodr"):
        if p.exists():
            return p
    raise FileNotFoundError(town)


def carla_polys(z) -> list:
    """Lane samples -> mirrored quads (lane id per (road, section, lane)), junction convex hulls, parking."""
    from scipy.spatial import ConvexHull
    x, y, yaw = z["x"], -z["y"], -np.radians(z["yaw"])
    w, typ, junc = z["width"], z["type"], z["junction"]
    half = float(z["step"]) * 0.6
    key = pd.Series(list(zip(z["road"], z["section"], z["lane"])))
    lid = key.map({k: i + 1 for i, k in enumerate(sorted(set(key)))}).to_numpy()
    u = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    v = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    c = np.stack([x, y], -1)
    corners = c[:, None] + np.array([half, -half, -half, half])[None, :, None] * u[:, None] + \
        (np.array([1, 1, -1, -1])[None, :, None] * (w / 2)[:, None, None]) * v[:, None]
    polys = []
    for q, t, hd, li, jn in zip(corners, typ, yaw, lid, junc):
        fl = S.F_DRIVE | (S.F_JUNC if jn >= 0 else 0)
        if t == 3:                                              # parking: drivable, not a lane (navsim: carpark)
            polys.append((q, None, 0, fl))
        else:
            polys.append((q, hd, int(li), fl | S.F_LANE | (S.F_BIDIR if t == 2 else 0)))
    for jn in np.unique(junc[junc >= 0]):                       # the junction interior between its connecting lanes
        pts = corners[(junc == jn) & (typ != 3)].reshape(-1, 2)
        if len(pts) >= 3:
            polys.append((pts[ConvexHull(pts).vertices], None, 0, S.F_DRIVE | S.F_JUNC))
    return polys


def maps_carla(towns=None) -> dict:
    """R/maps/carla/<town>.npz for every town with route points (lane samples via the CARLA env script)."""
    import subprocess
    out = {}
    for f in sorted(R("maps", "carla", "points").glob("*.npy")):
        town = f.stem
        if towns and town not in towns:
            continue
        lanes = R("maps", "carla", "lanes", f"{town}.npz")
        lanes.parent.mkdir(parents=True, exist_ok=True)
        if not lanes.exists():
            t0 = time.time()
            subprocess.run([str(data_dir() / "envs/carla/bin/python"), str(Path(__file__).resolve().parents[1] / "scripts/op_adapt_carla_lanes.py"),
                            town, str(xodr_path(town)), str(f), str(lanes)], check=True)
            log.info("%s lanes %.0f s", town, time.time() - t0)
        z = np.load(lanes)
        pts = np.load(f).astype(float)
        tm = rasterize(carla_polys(z), keep_tiles(np.stack([pts[:, 0], -pts[:, 1]], -1), 40.0))
        save_map(tm, R("maps", "carla", f"{town}.npz"))
        out[town] = {"samples": int(len(z["x"])), "tiles": int(len(tm.ij)), "drivable_km2": float((tm.flags & S.F_DRIVE > 0).sum() * S.RES ** 2 / 1e6)}
        log.info("%s %s", town, out[town])
    _save_json(R("maps", "carla", "summary.json"), out)
    return out


_MAPS: dict = {}


def carla_map(town: str) -> S.TileMap:
    if town not in _MAPS:
        _MAPS[town] = S.TileMap.load(R("maps", "carla", f"{town}.npz"))
    return _MAPS[town]


# ================================================================ CARLA slots (Cosmos pairs, P5 pass-1 worlds)

# ---------------------------------------------------------------- CARLA static scene geometry in NC (prereg v4)
NC_LABELS = (3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 26, 28)       # buildings ... guardrail (carla.CityObjectLabel);
#   Dynamic (21) left out: movable dressing the recorded experts drive through (Town10HD: 106 of 329 sampled poses)
VEH_LABELS = (14, 15, 16, 18, 19)                                           # map-baked parked vehicles
OBJ_MAX_HALF, OBJ_ROAD_M2, BODY_Z = 25.0, 2.0, (0.0, 1.5)
_OBJ: dict = {}


def objects_nc(towns=None) -> dict:
    """R/maps/carla/objects_nc/<town>.npz: the NC set of each town's scene objects (scripts/op_adapt_carla_objects.py),
    mirrored to the right-handed frame. Kept: the NC labels; no half extent above 25 m (landscape foliage, water planes,
    whole-block meshes are containers, not obstacles); non-vehicle boxes covering <= 2 m2 of the drivable raster (a box
    lying over the road surface is an aggregate or an overhead structure, handled by the height band otherwise)."""
    out = {}
    for f in sorted(R("maps", "carla", "objects").glob("*.npz")):
        town = f.stem
        if towns and town not in towns:
            continue
        z = np.load(f)
        lab, ext, cen, rot = z["label"], z["extent"], z["centre"], z["rot"]
        keep = np.isin(lab, NC_LABELS) & (ext[:, :2].max(1) <= OBJ_MAX_HALF) & (ext[:, :2].min(1) > 0)
        c = np.stack([cen[:, 0], -cen[:, 1]], -1)
        h = -np.radians(rot[:, 0])
        m = carla_map(town)
        road = np.zeros(len(c))
        for i in np.flatnonzero(keep & ~np.isin(lab, VEH_LABELS)):
            gx, gy = np.meshgrid(np.arange(-ext[i, 0], ext[i, 0] + 1e-6, 0.4), np.arange(-ext[i, 1], ext[i, 1] + 1e-6, 0.4))
            u, v = np.array([math.cos(h[i]), math.sin(h[i])]), np.array([-math.sin(h[i]), math.cos(h[i])])
            q = c[i] + gx.reshape(-1, 1) * u + gy.reshape(-1, 1) * v
            road[i] = float(((m.lookup(q)[0] & S.F_DRIVE) > 0).sum()) * 0.16
        keep &= np.isin(lab, VEH_LABELS) | (road <= OBJ_ROAD_M2)
        # boxes a recorded expert drove through (pass-1 Cosmos / P5 runs, no collision needed: the geometry is not at
        # body height there, e.g. a lamp arm or a canopy inside a pole's or a tree's box) are not obstacles
        ep = R("maps", "carla", "expert", f"{town}.npy")
        through = np.zeros(len(c), bool)
        if ep.exists() and keep.any():
            from scipy.spatial import cKDTree
            E = S.EGO["carla"]
            P = np.load(ep).astype(float)                     # (n, 4) CARLA x, y, z, yaw_deg of the vehicle transform
            yaw = -np.radians(P[:, 3])
            ec = np.stack([P[:, 0], -P[:, 1]], -1) + (E.rc - REAR_TF) * np.stack([np.cos(yaw), np.sin(yaw)], -1)
            tree = cKDTree(ec)
            for i in np.flatnonzero(keep & ~np.isin(lab, VEH_LABELS)):
                j = np.array(tree.query_ball_point(c[i], float(np.hypot(ext[i, 0], ext[i, 1])) + 3.0), int)
                if len(j):
                    band = (cen[i, 2] + ext[i, 2] >= P[j, 2]) & (cen[i, 2] - ext[i, 2] <= P[j, 2] + BODY_Z[1])
                    through[i] = bool((S.obb_overlap(ec[j], yaw[j], E.hl, E.hw, c[i], h[i], ext[i, 0], ext[i, 1]) & band).any())
        keep &= ~through
        o = R("maps", "carla", "objects_nc", f"{town}.npz")
        o.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(o, c=c[keep], h=h[keep], hl=ext[keep, 0], hw=ext[keep, 1], zlo=cen[keep, 2] - ext[keep, 2],
                            zhi=cen[keep, 2] + ext[keep, 2], label=lab[keep])
        out[town] = {"objects": int(len(lab)), "nc": int(keep.sum()), "driven_through": int(through.sum()),
                     "by_label": {int(k): int(n) for k, n in zip(*np.unique(lab[keep], return_counts=True))}}
        log.info("objects_nc %s %s", town, out[town])
    _save_json(R("maps", "carla", "objects_nc", "summary.json"), out)
    return out


def static_objects(town: str):
    if town not in _OBJ:
        from scipy.spatial import cKDTree
        f = R("maps", "carla", "objects_nc", f"{town}.npz")
        z = {k: v for k, v in np.load(f).items()} if f.exists() else None
        _OBJ[town] = (z, cKDTree(z["c"]) if z is not None and len(z["c"]) else None)
    return _OBJ[town]


def add_static(cols: dict, town: str, pose, ground_z: float, fov: float | None, cam_x: float, radius: float = 90.0, nt: int = S.NT):
    """Append the town's NC scene objects within `radius` of the ego (height band 0-1.5 m above the ego's ground) as
    standing STATIC actors; they enter the box overlap only (ref = NaN keeps them out of the _gap_front gap term).
    A(s): map-baked vehicles by the background-vehicle rule (front camera field of view, <= 60 m); other geometry
    is map knowledge and always scored."""
    z, tree = static_objects(town)
    if tree is None:
        return
    idx = tree.query_ball_point([pose[0], pose[1]], radius)
    if not idx:
        return
    idx = np.array(idx)
    band = (z["zhi"][idx] >= ground_z + BODY_Z[0]) & (z["zlo"][idx] <= ground_z + BODY_Z[1])
    for i in idx[band]:
        ce = S.to_ego(z["c"][i][None], pose)[0]
        veh = int(z["label"][i]) in VEH_LABELS
        if veh and fov is not None:
            d = ce - (cam_x, 0.0)
            vis = bool(d[0] > 0 and np.hypot(*d) <= VIS_RANGE and abs(math.atan2(d[1], d[0])) <= math.radians(fov / 2))
        else:
            vis = True
        cols["c"].append(np.tile(ce, (nt, 1)))
        cols["h"].append(np.full(nt, z["h"][i] - pose[2]))
        cols["hl"].append(float(z["hl"][i]))
        cols["hw"].append(float(z["hw"][i]))
        cols["ref"].append(np.full((nt, 2), np.nan))
        cols["valid"].append(np.ones(nt, bool))
        cols["speed"].append(np.zeros(nt))
        cols["kind"].append(S.STATIC)
        cols["vis"].append(vis)


def _carla_bb(tid: str, bb) -> list:
    """Actor bbox [loc x, y, z, ext x, y, z]; B2D's parked-vehicle meshes (static.prop.mesh) report the box turned by
    90 deg against the actor yaw (the parking-slot yaw is the lane's, the long side must lie along it): swap x / y."""
    bb = list(bb)
    if tid == "static.prop.mesh":
        bb[0], bb[1], bb[3], bb[4] = bb[1], bb[0], bb[4], bb[3]
    return bb


def _carla_kind(type_id: str) -> int:
    if type_id.startswith("vehicle.") and any(b in type_id for b in BIKES):
        return S.CYC
    for p, k in CARLA_KIND:
        if type_id.startswith(p):
            return k
    return S.VEH


def carla_slot(world: dict, k: int, town: str, hazards, hz_px=None, fov=64.0, cam_x=1.519, freeze=None) -> S.Slot:
    """Slot at pass-1 tick k of one CARLA world. world: pose (n, 7) [k, x, y, z, yaw_deg, vx, vy]; act_k / act_id /
    act_xyz / act_yaw / act_v (5 Hz actor records); kinds {id: [type, role, bbox]}; route (m, >= 2); k_trig.
    hazards: scenario hazard ids; hz_px: (ticks,) hazard px_eq per window tick indexed by k (dict or callable) for A(s).
    freeze: hazards frozen at k (pre-trigger), default k < k_trig. Right-handed (mirrored) output."""
    P = world["pose"]
    pk = P[:, 0].astype(int)
    row = np.searchsorted(pk, k)
    e = P[row]
    yaw_c = math.radians(e[4])
    tf = np.array([e[1], e[2]])
    rear_c = tf - REAR_TF * np.array([math.cos(yaw_c), math.sin(yaw_c)])
    pose = (rear_c[0], -rear_c[1], -yaw_c)
    kk = k + 2 * np.arange(S.NT)
    n = int(np.searchsorted(kk, pk[-1], side="right"))
    ak, aid = np.asarray(world["act_k"]), np.asarray(world["act_id"])
    kinds = world["kinds"]
    freeze = (world["k_trig"] < 0 or k < world["k_trig"]) if freeze is None else freeze
    hz = set(int(h) for h in hazards)
    cols = {f: [] for f in ("c", "h", "hl", "hw", "ref", "valid", "speed", "kind", "vis")}
    ctx = k - 4 * np.arange(N_CTX)
    for i in np.unique(aid):
        m = aid == i
        kr = ak[m]
        if len(kr) == 0 or str(int(i)) not in kinds:
            continue
        o = np.argsort(kr)
        kr, xyz, yw, vv = kr[o].astype(float), world["act_xyz"][m][o].astype(float), world["act_yaw"][m][o].astype(float), world["act_v"][m][o].astype(float)
        if np.hypot(*(xyz[np.argmin(np.abs(kr - k)), :2] - tf)) < 0.5 and abs(kr[np.argmin(np.abs(kr - k))] - k) <= 4:
            continue                                              # the hero itself
        tid, _, bb = kinds[str(int(i))]
        if not tid.startswith(("vehicle.", "walker.", "static.prop.")):
            continue
        bb = _carla_bb(tid, bb)
        is_hz = int(i) in hz
        q = kk if not (is_hz and freeze) else np.full(S.NT, float(k))
        far = (q < kr[0] - 4) | (q > kr[-1] + 4)
        x = np.interp(q, kr, xyz[:, 0])
        y = np.interp(q, kr, xyz[:, 1])
        z = np.interp(q, kr, xyz[:, 2])
        ya = np.radians(np.interp(q, kr, np.unwrap(yw, period=360)))
        sp = np.hypot(np.interp(q, kr, vv[:, 0]), np.interp(q, kr, vv[:, 1])) * (0 if (is_hz and freeze) else 1)
        valid = ~far & (np.abs(z - e[3]) <= 5.0)
        if not valid.any():
            continue
        if is_hz and not freeze and valid[0]:                    # destroyed scenario actor: truncate the horizon
            gone = np.flatnonzero(~valid)
            if len(gone):
                n = min(n, int(gone[0]))
        cx = x + bb[0] * np.cos(ya) - bb[1] * np.sin(ya)
        cy = y + bb[0] * np.sin(ya) + bb[1] * np.cos(ya)
        cw = to_ego_m(np.stack([cx, -cy], -1), pose)
        rw = to_ego_m(np.stack([x, -y], -1), pose)
        kind = _carla_kind(tid)
        if is_hz and kind == S.PED:
            vis = hz_px is not None and max((hz_px(t) for t in ctx), default=0.0) >= VIS_PX
        else:                                                    # in the front camera's field of view, <= 60 m, any context slot
            xc = np.interp(ctx, kr, xyz[:, 0])
            yc = np.interp(ctx, kr, xyz[:, 1])
            okc = ~((ctx < kr[0] - 4) | (ctx > kr[-1] + 4))
            vis = False
            for t, a_, b_, ok_ in zip(ctx, xc, yc, okc):
                j = min(np.searchsorted(pk, t), len(pk) - 1)
                if not ok_ or pk[j] != t:
                    continue
                pe = P[j]
                yc_ = math.radians(pe[4])
                camp = np.array([pe[1], pe[2]]) + (cam_x - REAR_TF) * np.array([math.cos(yc_), math.sin(yc_)])
                d = to_ego_m(np.array([[a_, -b_]]), (camp[0], -camp[1], -yc_))[0]
                if d[0] > 0 and np.hypot(*d) <= VIS_RANGE and abs(math.atan2(d[1], d[0])) <= math.radians(fov / 2):
                    vis = True
                    break
        cols["c"].append(cw)
        cols["h"].append(-ya - pose[2])
        cols["hl"].append(bb[3])
        cols["hw"].append(bb[4])
        cols["ref"].append(rw)
        cols["valid"].append(valid)
        cols["speed"].append(sp)
        cols["kind"].append(kind)
        cols["vis"].append(vis)
    add_static(cols, town, pose, float(e[3]), fov, cam_x)
    A = _stack_actors(cols)
    rt = np.asarray(world["route"])[:, :2].astype(float)
    route = to_ego_m(np.stack([rt[:, 0], -rt[:, 1]], -1), pose)
    route = _ahead(route)
    v0 = float(np.hypot(e[5], e[6]))
    return S.Slot(S.EGO["carla"], pose, v0, n, A, route, carla_map(town), route=route, meta={"cam_x": cam_x, "tick": k})


def to_ego_m(xy, pose):
    return S.to_ego(xy, pose)


def _ahead(route: np.ndarray, back: float = 5.0, fwd: float = 250.0) -> np.ndarray:
    """The route from `back` m behind its point nearest to the ego to `fwd` m ahead of it."""
    d = np.hypot(route[:, 0], route[:, 1])
    i = int(np.argmin(d))
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    keep = (s >= s[i] - back) & (s <= s[i] + fwd)
    r = route[keep]
    return r if len(r) >= 2 else np.array([[0.0, 0.0], [fwd, 0.0]])


def _stack_actors(cols: dict) -> S.Actors:
    if not cols["c"]:
        return S.Actors.empty()
    return S.Actors(np.stack(cols["c"], 1), np.stack(cols["h"], 1), np.array(cols["hl"], float), np.array(cols["hw"], float),
                    np.stack(cols["ref"], 1), np.stack(cols["valid"], 1), np.stack(cols["speed"], 1),
                    np.array(cols["kind"], int), np.array(cols["vis"], bool))


def sim_world(pair: str, sign: int) -> tuple[dict, list, object]:
    """(world dict, hazard ids, px_eq by pass-1 tick) of one Cosmos pair world from C's R/sim/world/<pair>.npz."""
    z = np.load(R("sim", "world", f"{pair}.npz"), allow_pickle=False)
    m = "plus" if sign > 0 else "minus"
    w = {"pose": z[f"{m}_pose"], "act_k": z[f"{m}_act_k"], "act_id": z[f"{m}_act_id"], "act_xyz": z[f"{m}_act_xyz"],
         "act_yaw": z[f"{m}_act_yaw"], "act_v": z[f"{m}_act_v"], "kinds": json.loads(str(z["kinds"])), "route": z["route"],
         "k_trig": int(z[f"{m}_k_trig"])}
    k0, px = int(z["k0"]), np.asarray(z["px"], float)
    hz = list(z["hz_ids"]) if sign > 0 else list(z[f"{m}_hazards"]) + list(z["hz_ids"])
    return w, hz, (lambda t: float(px[t - k0]) if 0 <= t - k0 < len(px) and sign > 0 else 0.0)


_P5: dict = {}
P5_PX_EQ = 3.38                               # P5 segmentation-view pixels -> px_eq (M0)


def p5_world(rid: str) -> tuple:
    """(world dict, hazard ids, px_eq by tick, town, frame -> tick) of one P5 v1 BA pass-1 run (world 1 x+, 2 x-, 3 null)."""
    if rid not in _P5:
        from . import p5_pairs as PP
        if len(_P5) > 16:
            _P5.clear()
        gen = data_dir() / "runs" / "p5v1" / "gen-ba"
        a = PP.attempt(gen, rid)
        W = PP.load_world(a)
        p = W["pose"]
        summ = json.loads((a / "p5_summary.json").read_text())
        tt = summ.get("t_trigger")
        w = {"pose": np.c_[p.index.to_numpy(), p[["x", "y", "z", "yaw", "vx", "vy"]].to_numpy()].astype(np.float64),
             "act_k": np.asarray(W["act"]["k"], np.int64), "act_id": W["act"]["id"], "act_xyz": W["act"]["xyz"], "act_yaw": W["act"]["yaw"],
             "act_v": W["act"]["v"], "kinds": W["kinds"], "route": pd.read_json(a / "route.json")[["x", "y", "z"]].to_numpy(np.float32),
             "k_trig": int(round(tt / 0.05)) if tt is not None else -1}
        hz = [int(h) for h in W["hazards"]]
        px = {int(k): P5_PX_EQ * max([float(v) for kk, v in (r.px or {}).items() if int(kk) in hz] + [0.0])
              for k, r in W["frames"].iterrows() if isinstance(r.px, dict)}
        f2k = dict(zip(W["frames"].frame.astype(int), W["frames"].index.astype(int)))
        _P5[rid] = (w, hz, lambda t, px=px: px.get(int(t), 0.0), _town(json.loads((a / "meta.json").read_text())["town"]), f2k)
    return _P5[rid]


# ================================================================ nuScenes

def nus_scenes(split: str = "all", workers: int = 24) -> dict:
    """R/scene/nus/<scene>.npz: CAM_FRONT sample-data ego poses (t, x, y, yaw), keyframes (token, t, cam_x, pose), and
    every annotation (keyframe index, instance index, x, y, yaw, half l / w, kind, CAM_FRONT-visible)."""
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    nusc = NuScenes("v1.0-trainval", dataroot=str(data_dir() / "datasets" / "nuscenes"), verbose=False)
    out = R("scene", "nus")
    out.mkdir(parents=True, exist_ok=True)
    yaw_of = lambda q: Quaternion(q).yaw_pitch_roll[0]  # noqa: E731
    n = 0
    for sc in nusc.scene:
        f = out / f"{sc['name']}.npz"
        if f.exists():
            continue
        loc = nusc.get("log", sc["log_token"])["location"]
        sd = nusc.get("sample_data", nusc.get("sample", sc["first_sample_token"])["data"]["CAM_FRONT"])
        ts, poses = [], []
        while True:
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            ts.append(sd["timestamp"] * 1e-6)
            poses.append((ep["translation"][0], ep["translation"][1], yaw_of(ep["rotation"])))
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
        tok, kt, kpose, kcam = [], [], [], []
        ann = []
        inst = {}
        s = nusc.get("sample", sc["first_sample_token"])
        while True:
            cs = nusc.get("sample_data", s["data"]["CAM_FRONT"])
            cal = nusc.get("calibrated_sensor", cs["calibrated_sensor_token"])
            ep = nusc.get("ego_pose", cs["ego_pose_token"])
            K = np.array(cal["camera_intrinsic"])
            q_e, t_e = Quaternion(ep["rotation"]), np.array(ep["translation"])
            q_c, t_c = Quaternion(cal["rotation"]), np.array(cal["translation"])
            ki = len(tok)
            tok.append(s["token"])
            kt.append(cs["timestamp"] * 1e-6)
            kpose.append((ep["translation"][0], ep["translation"][1], yaw_of(ep["rotation"])))
            kcam.append(cal["translation"][0])
            for at in s["anns"]:
                a = nusc.get("sample_annotation", at)
                p = np.array(a["translation"])
                pc = q_c.inverse.rotate(q_e.inverse.rotate(p - t_e) - t_c)
                uvw = K @ pc
                inimg = pc[2] > 0.5 and 0 <= uvw[0] / uvw[2] < 1600 and 0 <= uvw[1] / uvw[2] < 900
                vis = int(a["visibility_token"] or 0) >= 2
                cat = a["category_name"]
                kind = (S.CYC if cat.startswith(("vehicle.bicycle", "vehicle.motorcycle")) else S.VEH if cat.startswith("vehicle.")
                        else S.PED if cat.startswith(("human.", "animal")) else S.STATIC)
                ii = inst.setdefault(a["instance_token"], len(inst))
                ann.append((ki, ii, p[0], p[1], yaw_of(a["rotation"]), a["size"][1] / 2, a["size"][0] / 2, kind, inimg and vis))
            if not s["next"]:
                break
            s = nusc.get("sample", s["next"])
        A = np.array(ann, float).reshape(-1, 9)
        np.savez_compressed(f, location=loc, t=np.array(ts), pose=np.array(poses), tok=np.array(tok), kt=np.array(kt),
                            kpose=np.array(kpose), kcam=np.array(kcam), ann=A)
        n += 1
    log.info("nuScenes scenes written: %d", n)
    return {"written": n}


def maps_nus(workers: int = 4) -> dict:
    """R/maps/nus/<location>.npz from the map expansion, only tiles within 40 m of some scene's ego path."""
    from nuscenes.map_expansion import arcline_path_utils as AP
    from nuscenes.map_expansion.map_api import NuScenesMap
    from scipy.spatial import cKDTree
    scenes = sorted(R("scene", "nus").glob("*.npz"))
    pts: dict = {}
    for f in scenes:
        z = np.load(f)
        pts.setdefault(str(z["location"]), []).append(z["pose"][:, :2])
    out = {}
    for loc, pp in pts.items():
        keep = keep_tiles(np.vstack(pp), 40.0)
        M = NuScenesMap(dataroot=str(data_dir() / "datasets" / "nuscenes"), map_name=loc)
        polys = []

        def exterior(tok):
            p = M.extract_polygon(tok)
            return np.array(p.exterior.coords)
        for rec in M.drivable_area:
            for pt in rec["polygon_tokens"]:
                polys.append((exterior(pt), None, 0, S.F_DRIVE))
        for rec in M.road_segment:
            if rec["is_intersection"]:
                polys.append((exterior(rec["polygon_token"]), None, 0, S.F_DRIVE | S.F_JUNC))
        lanes = [(r, S.F_LANE) for r in M.lane] + [(r, S.F_LANE) for r in M.lane_connector]
        lane_px = []
        for li, (rec, fl) in enumerate(lanes, 1):
            poly = exterior(rec["polygon_token"])
            path = AP.discretize_lane(M.arcline_path_3[rec["token"]], resolution_meters=0.5)
            lane_px.append((poly, np.array(path), li))
        # lane polygons carry a varying direction: rasterise each lane at its polygon, heading = nearest arcline pose
        base = rasterize(polys, keep)
        tiles = {tuple(k): i for i, k in enumerate(base.ij)}
        import cv2
        for poly, path, li in lane_px:
            if len(path) < 2:
                continue
            tree = cKDTree(path[:, :2])
            lo, hi = poly.min(0), poly.max(0)
            for i in range(int(math.floor(lo[0] / S.TM)), int(math.floor(hi[0] / S.TM)) + 1):
                for j in range(int(math.floor(lo[1] / S.TM)), int(math.floor(hi[1] / S.TM)) + 1):
                    t = tiles.get((i, j))
                    if t is None:
                        continue
                    q = (poly - (i * S.TM, j * S.TM)) / S.RES - 0.5          # pixel-centre coordinates
                    x0, y0 = max(int(math.floor(q[:, 0].min())), 0), max(int(math.floor(q[:, 1].min())), 0)
                    x1, y1 = min(int(math.ceil(q[:, 0].max())) + 1, S.TILE), min(int(math.ceil(q[:, 1].max())) + 1, S.TILE)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    cv2.fillPoly(m, [np.round((q - (x0, y0)) * 16).astype(np.int32)], 1, shift=4)
                    yy, xx = np.nonzero(m)
                    if not len(yy):
                        continue
                    wx, wy = i * S.TM + (xx + x0 + 0.5) * S.RES, j * S.TM + (yy + y0 + 0.5) * S.RES
                    _, nn = tree.query(np.stack([wx, wy], -1))
                    hq = (np.round((path[nn, 2] % (2 * np.pi)) / (2 * np.pi) * 255) % 255).astype(np.uint8)
                    py, px = yy + y0, xx + x0
                    base.flags[t][py, px] |= S.F_LANE | S.F_DRIVE
                    l0, l1 = base.lane[t, 0], base.lane[t, 1]
                    a = (l0[py, px] == 0) | (l0[py, px] == li)
                    l0[py[a], px[a]] = li
                    base.head[t, 0][py[a], px[a]] = hq[a]
                    b = ~a & ((l1[py, px] == 0) | (l1[py, px] == li))
                    l1[py[b], px[b]] = li
                    base.head[t, 1][py[b], px[b]] = hq[b]
        save_map(base, R("maps", "nus", f"{loc}.npz"))
        out[loc] = {"tiles": int(len(base.ij)), "lanes": len(lane_px), "polys": len(polys)}
        log.info("nus map %s %s", loc, out[loc])
    _save_json(R("maps", "nus", "summary.json"), out)
    return out


def nus_map(loc: str) -> S.TileMap:
    if ("nus", loc) not in _MAPS:
        _MAPS["nus", loc] = S.TileMap.load(R("maps", "nus", f"{loc}.npz"))
    return _MAPS["nus", loc]


_NUS: dict = {}


def nus_scene(scene: str) -> dict:
    if scene not in _NUS:
        if len(_NUS) > 64:
            _NUS.clear()
        z = np.load(R("scene", "nus", f"{scene}.npz"))
        _NUS[scene] = {k: z[k] for k in z.files}
    return _NUS[scene]


def nus_slot(scene: str, token: str, ref: np.ndarray | None = None) -> S.Slot:
    """Keyframe slot: actors = every annotated instance, boxes linearly interpolated between its keyframes to the 10 Hz
    grid (valid only between its first and last annotation); A(s) = CAM_FRONT visible (visibility >= 40 %, centre in
    the image) at a keyframe inside the 1.6 s context. Horizon: to the scene's last keyframe."""
    z = nus_scene(scene)
    ki = int(np.flatnonzero(z["tok"] == token)[0])
    t0 = float(z["kt"][ki])
    pose = tuple(z["kpose"][ki])
    t_end = float(z["kt"][-1])
    n = int(min(S.NT, math.floor((t_end - t0) / S.DT + 1e-6) + 1))
    tq = t0 + S.TS
    A = z["ann"]
    kt = z["kt"]
    cols = {f: [] for f in ("c", "h", "hl", "hw", "ref", "valid", "speed", "kind", "vis")}
    ctx_k = np.flatnonzero((kt >= t0 - 1.6 - 1e-3) & (kt <= t0 + 1e-3))
    for ii in np.unique(A[:, 1]).astype(int):
        a = A[A[:, 1] == ii]
        a = a[np.argsort(a[:, 0])]
        ta = kt[a[:, 0].astype(int)]
        valid = (tq >= ta[0] - 1e-3) & (tq <= ta[-1] + 1e-3)
        if not valid.any():
            continue
        x, y = np.interp(tq, ta, a[:, 2]), np.interp(tq, ta, a[:, 3])
        yw = np.interp(tq, ta, np.unwrap(a[:, 4]))
        c = S.to_ego(np.stack([x, y], -1), pose)
        sp = np.hypot(np.gradient(x, S.DT), np.gradient(y, S.DT)) if len(ta) > 1 else np.zeros(S.NT)
        cols["c"].append(c)
        cols["h"].append(yw - pose[2])
        cols["hl"].append(float(a[0, 5]))
        cols["hw"].append(float(a[0, 6]))
        cols["ref"].append(c)
        cols["valid"].append(valid)
        cols["speed"].append(sp)
        cols["kind"].append(int(a[0, 7]))
        cols["vis"].append(bool(a[np.isin(a[:, 0].astype(int), ctx_k), 8].astype(bool).any()))
    v0 = _nus_speed(z, t0)
    log_path = nus_log_path(z, t0, pose)
    return S.Slot(S.EGO["nus"], pose, v0, n, _stack_actors(cols), log_path if ref is None else ref, nus_map(str(z["location"])),
                  meta={"cam_x": float(z["kcam"][ki]), "t0": t0, "log": log_path})


def _nus_speed(z, t0) -> float:
    t, p = z["t"], z["pose"]
    j = int(np.clip(np.searchsorted(t, t0), 1, len(t) - 1))
    return float(np.hypot(*(p[j, :2] - p[j - 1, :2])) / max(t[j] - t[j - 1], 1e-3))


def nus_log_path(z, t0, pose) -> np.ndarray:
    """The logged ego (rear axle) on the 10 Hz grid from t0, ego frame, (NT, 2) (held at the scene end)."""
    t, p = z["t"], z["pose"]
    tq = np.minimum(t0 + S.TS, t[-1])
    return S.to_ego(np.stack([np.interp(tq, t, p[:, 0]), np.interp(tq, t, p[:, 1])], -1), pose)


# ================================================================ NAVSIM (maps / agents extracted in the navsim env)

def nav_slot(token_npz: Path, e: float = 0.0, psi: float = 0.0, cam_x: float = 1.5) -> S.Slot:
    """Slot of one NAVSIM token (R/maps/nav/<cache>/<token>.npz from scripts/op_adapt_nav.py), optionally from an
    offset start: the log rear-axle pose moved by (0, e) in its own frame and turned by psi. A(s): actors whose box
    centre is in CAM_F0's field of view and <= 60 m at t = 0 (the cache has no past tracks)."""
    import shapely
    z = np.load(token_npz, allow_pickle=False)
    x0, y0, yaw0 = z["pose"]
    off = S.to_world(np.array([0.0, e]), (x0, y0, yaw0))
    pose = (float(off[0]), float(off[1]), float(yaw0 + psi))
    def geo(k):
        b, o = z[f"{k}_wkb"].tobytes(), z[f"{k}_off"]
        return list(shapely.from_wkb([b[o[i]:o[i + 1]] for i in range(len(o) - 1)])) if len(o) > 1 else []
    mq = PolyMap_cached(str(token_npz), geo("drivable"), geo("lanes"), z["lane_route"].astype(bool), geo("intersection"))
    c = S.to_ego(z["agent_c"], pose)                              # (NT, N, 2)
    N = c.shape[1]
    d = S.to_ego(z["agent_c"][0], (pose[0] + cam_x * math.cos(pose[2]), pose[1] + cam_x * math.sin(pose[2]), pose[2]))
    vis = (d[:, 0] > 0) & (np.hypot(d[:, 0], d[:, 1]) <= VIS_RANGE) & (np.abs(np.arctan2(d[:, 1], d[:, 0])) <= math.radians(HFOV["nav"] / 2)) \
        & z["agent_valid"][0]
    A = S.Actors(c, z["agent_h"] - pose[2], z["agent_hl"], z["agent_hw"], c.copy(), z["agent_valid"].astype(bool),
                 z["agent_speed"], z["agent_kind"].astype(int), vis) if N else S.Actors.empty()
    centre = S.to_ego(z["centerline"], pose)
    return S.Slot(S.EGO["nav"], pose, float(z["v0"]), S.NT, A, _ahead(centre, 10.0, 300.0), mq,
                  meta={"cam_x": float(cam_x), "centre": _ahead(centre, 10.0, 300.0)})


_PM: dict = {}


def PolyMap_cached(key, *a) -> S.PolyMap:
    if key not in _PM:
        if len(_PM) > 32:
            _PM.clear()
        _PM[key] = S.PolyMap(*a)
    return _PM[key]


# ================================================================ V1: NC vs cg on WL-1 branch runs

def _v1_one(args):
    from . import wl as WL
    rid, adir, k = args
    a = Path(adir)
    try:
        o = WL.outcome(a, int(k))
        meta = json.loads((a / "meta.json").read_text())
        kinds = json.loads((a / "actor_kinds.json").read_text())
        p = WL._pose(a)
        t1 = min(int(k) + 60, int(p.index.max()))
        ticks = np.arange(int(k), t1 + 1)
        e = p.loc[ticks]
        yaw = np.radians(e.yaw.to_numpy())
        tf = e[["x", "y"]].to_numpy()
        rear = tf + meta.get("rear_axle_x", -REAR_TF) * np.stack([np.cos(yaw), np.sin(yaw)], -1)
        pr = np.stack([rear[:, 0], -rear[:, 1]], -1)
        h = -yaw
        g = {"p": pr[None], "h": h[None], "v": np.hypot(e.vx, e.vy).to_numpy()[None], "u": np.stack([np.cos(h), np.sin(h)], -1)[None]}
        z = np.load(a / "actors.npz")
        fr0 = int(WL._frames(a).frame.iloc[0])
        tk = z["frame"] - fr0 + 1
        hero = meta.get("hero_id")
        cols = {f: [] for f in ("c", "h", "hl", "hw", "ref", "valid", "speed", "kind", "vis")}
        for i in np.unique(z["id"]):
            if int(i) == hero or str(int(i)) not in kinds:
                continue
            tid, _, bb = kinds[str(int(i))]
            if not tid.startswith(("walker.", "vehicle.", "static.prop.")):
                continue
            bb = _carla_bb(tid, bb)
            m = z["id"] == i
            kr = tk[m].astype(float)
            o_ = np.argsort(kr)
            kr, xyz, yw, vv = kr[o_], z["xyz"][m][o_].astype(float), z["yaw"][m][o_].astype(float), z["v"][m][o_].astype(float)
            valid = (ticks >= kr[0] - 4) & (ticks <= kr[-1] + 4)
            if not valid.any():
                continue
            x, y, zz = (np.interp(ticks, kr, xyz[:, j]) for j in range(3))
            valid &= np.abs(zz - e.z.to_numpy()) <= 5.0
            ya = np.radians(np.interp(ticks, kr, np.unwrap(yw, period=360)))
            cx, cy = x + bb[0] * np.cos(ya) - bb[1] * np.sin(ya), y + bb[0] * np.sin(ya) + bb[1] * np.cos(ya)
            cols["c"].append(np.stack([cx, -cy], -1))
            cols["h"].append(-ya)
            cols["hl"].append(bb[3])
            cols["hw"].append(bb[4])
            cols["ref"].append(np.stack([x, -y], -1))
            cols["valid"].append(valid)
            cols["speed"].append(np.hypot(np.interp(ticks, kr, vv[:, 0]), np.interp(ticks, kr, vv[:, 1])))
            cols["kind"].append(_carla_kind(tid))
            cols["vis"].append(True)
        e0 = e.iloc[0]
        y0 = math.radians(float(e0.yaw))
        r0 = np.array([e0.x, e0.y]) + meta.get("rear_axle_x", -REAR_TF) * np.array([math.cos(y0), math.sin(y0)])
        pose0 = (float(r0[0]), float(-r0[1]), -y0)
        st = {f: [] for f in cols}
        add_static(st, _town(meta["town"]), pose0, float(e0.z), None, 0.0, radius=120.0, nt=len(ticks))
        for f in cols:                                           # back to the (mirrored) world frame of this check
            cols[f] += [S.to_world(x, pose0) if f == "c" else (x + pose0[2] if f == "h" else x) for x in st[f]]
        A = _stack_actors(cols) if cols["c"] else None
        ncv, fail, _ = S.nc(g, S.EGO["carla"], A, len(ticks)) if A is not None else (np.ones(1, bool), None, None)
        # decomposition: the same rule without the at-fault speed / front-half conditions (overlap of the full box)
        return {"route_id": rid, "cg": bool(o["unsafe_cg"]), "nc_fail": not bool(ncv[0]), "collision": o["collision"],
                "collision_road": o["collision_road"], "gap_min_m": o["gap_min_m"], "types": json.dumps(o["collision_types"]),
                "v_min": float(g["v"].min()), "n": len(ticks)}
    except Exception as ex:  # noqa: BLE001
        return {"route_id": rid, "error": repr(ex)[:300]}


def check_v1(workers: int | None = None, tag: str = "_v4") -> dict:
    """V1 (§5 item 1): S_jev's NC on WL-1's branch runs (actual trajectory, recorded actors, 3 s after the fork) against
    WL's cg label (collision or in-lane gap < 2 m); registered line: agreement >= 95 %."""
    from multiprocessing import Pool
    os.environ["WL_NAME"] = "wl"
    f = pd.read_parquet(data_dir() / "runs" / "wl" / "forks.parquet")
    rows = []
    for s in ("ba", "p6"):
        d = data_dir() / "runs" / "wl" / "gen" / s
        for rec in sorted((d / "done").glob("*.json")):
            rid = rec.stem
            rows.append((rid, str(d / "attempts" / rid / str(json.loads(rec.read_text())["attempt"]))))
    runs = pd.DataFrame(rows, columns=["route_id", "adir"]).merge(f[["route_id", "fork_tick"]].astype({"route_id": str}), on="route_id")
    with Pool(workers or cores()) as p:
        t = pd.DataFrame(p.map(_v1_one, list(zip(runs.route_id, runs.adir, runs.fork_tick)), chunksize=8))
    R("checks").mkdir(parents=True, exist_ok=True)
    t.to_parquet(R("checks", f"V1{tag}_runs.parquet"), index=False)
    ok = t[t.get("error").isna()] if "error" in t else t
    agree = float((ok.cg == ok.nc_fail).mean())
    res = {"check": "V1", "line": ">= 0.95 agreement of NC with WL cg", "runs": int(len(t)), "scored": int(len(ok)),
           "errors": int(len(t) - len(ok)), "agreement": agree, "pass": agree >= 0.95,
           "confusion": {"cg1_nc0": int((ok.cg & ok.nc_fail).sum()), "cg1_nc1": int((ok.cg & ~ok.nc_fail).sum()),
                         "cg0_nc0": int((~ok.cg & ok.nc_fail).sum()), "cg0_nc1": int((~ok.cg & ~ok.nc_fail).sum())},
           "cg_rate": float(ok.cg.mean()), "nc_fail_rate": float(ok.nc_fail.mean())}
    dis = ok[ok.cg != ok.nc_fail]
    res["disagreements"] = {"cg_only_collision_nonroad": int((dis.cg & dis.collision & ~dis.collision_road & (dis.gap_min_m >= 2)).sum()),
                            "cg_only_standing_ego": int((dis.cg & (dis.v_min < S.V_FAULT)).sum()),
                            "nc_only": int((~dis.cg).sum())}
    res["definition"] = "v4: NC boxes = actors (static.prop.mesh true orientation) + CARLA static scene geometry" if tag else "v3"
    _save_json(R("checks", f"V1{tag}.json"), res)
    log.info("V1 %s", res)
    return res


# ================================================================ V2 / V5b: nuScenes val log trajectories

def _v2_one(args):
    scene, tokens = args
    rows = []
    for tok in tokens:
        try:
            s = nus_slot(scene, tok)
            if s.n - 1 < round(S.H_MIN / S.DT):
                continue
            p = s.meta["log"][None]
            r = S.score_slot(s, p, ["log"], all_actors=False)
            rows.append({"scene": scene, "token": tok, "S": float(r["S"][0]), "NC": bool(r["NC"][0]), "DAC": bool(r["DAC"][0]),
                         "DDC": float(r["DDC"][0]), "TTC": bool(r["TTC"][0]), "C": bool(r["C"][0]), "D_onc": float(r["D_onc"][0]),
                         "n_act": int(len(s.actors.vis)), "n_vis": int(s.actors.vis.sum()), "v0": s.v0})
        except Exception as ex:  # noqa: BLE001
            rows.append({"scene": scene, "token": tok, "error": repr(ex)[:300]})
    return rows


def check_v2(workers: int | None = None) -> dict:
    """V2: every nuScenes val keyframe with >= 3 s of logged future, the log trajectory scored alone (C(s) = {log}, so
    P = 1 when it is safe; the check is about map alignment and footprints): S_jev >= 0.8 on >= 90 % of frames.
    V5 part b on the same rollouts: DDC = 1 on >= 99 %."""
    from multiprocessing import Pool
    from . import op_adapt_r2_data as C
    val = [s for s, v in C.splits("nus_scenes").items() if v == "val"]
    jobs = [(s, list(nus_scene(s)["tok"])) for s in val if R("scene", "nus", f"{s}.npz").exists()]
    with Pool(workers or cores()) as p:
        t = pd.DataFrame([r for part in p.imap_unordered(_v2_one, jobs) for r in part])
    R("checks").mkdir(parents=True, exist_ok=True)
    t.to_parquet(R("checks", "V2_frames.parquet"), index=False)
    ok = t[t["error"].isna()] if "error" in t else t
    f08 = float((ok.S >= 0.8).mean())
    ddc1 = float((ok.DDC == 1.0).mean())
    res2 = {"check": "V2", "line": "S_jev(log) >= 0.8 on >= 90 % of nuScenes val frames", "scenes": len(jobs), "frames": int(len(ok)),
            "errors": int(len(t) - len(ok)), "frac_S_ge_0.8": f08, "pass": f08 >= 0.90,
            "fail_rates": {k: float(1 - ok[k].astype(float).mean()) if k != "DDC" else float((ok.DDC < 1).mean())
                           for k in ("NC", "DAC", "DDC", "TTC", "C")}, "S_median": float(ok.S.median())}
    res5 = {"check": "V5b", "line": "DDC(log) = 1 on >= 99 % of nuScenes val frames", "frames": int(len(ok)), "frac_DDC_1": ddc1,
            "pass": ddc1 >= 0.99}
    _save_json(R("checks", "V2.json"), res2)
    _save_json(R("checks", "V5b.json"), res5)
    log.info("V2 %s\nV5b %s", res2, res5)
    return {"V2": res2, "V5b": res5}


# ================================================================ score tables

class SlotContext:
    """uid -> Slot (and its candidates) for one domain, from C's index / teacher and S's maps and scene files."""

    def __init__(self, domain: str, root=None):
        from . import op_adapt_r2_data as C
        _MAPS.clear()
        self.domain = domain
        self.idx = C.load_index(domain).set_index("uid", drop=False)
        self._teacher = None
        self._off = None
        if domain == "off":
            self._off = pd.read_parquet(R("offset", "table.parquet")).set_index("uid")

    def cam_x(self, token: str) -> float:
        """CAM_F0's x ahead of the rear axle (navsim_zs slim index, R/maps/nav/cam_x.parquet)."""
        if not hasattr(self, "_cam"):
            self._cam = pd.read_parquet(R("maps", "nav", "cam_x.parquet")).set_index("token").cam_x
        return float(self._cam.get(token, 1.5))

    def scores(self, cols):
        z = S.load_scores(self.domain)
        return tuple(z[c] for c in cols)

    @property
    def teacher(self):
        if self._teacher is None:
            from . import op_adapt_r2_data as C
            t = C.load_teacher(self.domain)
            self._teacher = (pd.Series(np.arange(len(t["uid"])), index=t["uid"]), t["plan_mu"])
        return self._teacher

    def plans(self, uid: int) -> dict:
        pos, mu = self.teacher
        m = mu[int(pos[uid])]
        return {"op": m[0], "op_L": m[1], "op_R": m[2]}

    def slot(self, uid: int) -> S.Slot:
        r = self.idx.loc[uid]
        if self.domain in ("simC", "simK"):
            w, hz, px = sim_world(r.key, int(r.sign))
            return carla_slot(w, int(r.tick), "Town12", hz, px, HFOV["sim"], CAM_X["sim"])
        if self.domain == "nus":
            return nus_slot(r.key, r.token)
        if self.domain == "p5":
            rid, frame = str(r["name"]).rsplit("-", 1)
            w, hz, px, town, f2k = p5_world(rid)
            return carla_slot(w, f2k[int(frame)], town, hz, px, HFOV["p5"], CAM_X["p5"])
        if self.domain == "off":
            o = self._off.loc[uid]
            return nav_slot(R("maps", "nav", "v1_navtrain", f"{o.token}.npz"), float(o.e), float(o.psi), self.cam_x(o.token))
        raise KeyError(self.domain)

    def candidates(self, uid: int, s: S.Slot):
        names = S.CANDS_OFF if self.domain == "off" else S.CANDS
        if self.domain == "nus":
            s.ref = S.plan_at(self.plans(uid)["op"], S.TS, s.meta["cam_x"])          # real side: op's path
        return names, S.candidates(self.plans(uid), s.v0, s.meta["cam_x"], route=s.route,
                                   centre=s.meta.get("centre"), names=names)


_CTX: dict = {}


def _score_chunk(args):
    domain, uids = args
    ctx = _CTX.get(domain) or _CTX.setdefault(domain, SlotContext(domain))
    out = []
    for uid in uids:
        try:
            s = ctx.slot(int(uid))
            names, (P, trj) = ctx.candidates(int(uid), s)
            r = S.score_slot(s, P, names, all_actors=domain != "off")
            A = s.actors
            d0 = np.hypot(A.c[0, :, 0], A.c[0, :, 1]) if len(A.vis) else np.zeros(0)
            r["vru30"] = bool((A.vis & np.isin(A.kind, (S.PED, S.CYC)) & A.valid[0] & (d0 <= 30.0) & (A.c[0, :, 0] > 0)).any()) if len(A.vis) else False
            out.append((int(uid), trj, r, None))
        except Exception as ex:  # noqa: BLE001
            out.append((int(uid), None, None, repr(ex)[:300]))
    return out


def score_domain(domain: str, uids=None, workers: int | None = None, chunk: int = 64) -> dict:
    """R/score/<domain>.npz (fields in the build doc's S section)."""
    from multiprocessing import Pool
    from . import op_adapt_r2_data as C
    t0 = time.time()
    idx = C.load_index(domain)
    if uids is None:
        uids = idx.uid.to_numpy()
        if domain == "nus":
            uids = idx.uid[idx.labeled].to_numpy()
    tu = set(np.asarray(C.load_teacher(domain)["uid"]).tolist())
    uids = np.array([u for u in uids if int(u) in tu], np.int64)
    jobs = [(domain, uids[i:i + chunk]) for i in range(0, len(uids), chunk)]
    rows, errs = [], []
    with Pool(workers or cores()) as p:
        for part in p.imap_unordered(_score_chunk, jobs):
            for uid, trj, r, err in part:
                (errs.append((uid, err)) if err else rows.append((uid, trj, r)))
    rows.sort(key=lambda x: x[0])
    names = S.CANDS_OFF if domain == "off" else S.CANDS
    K = len(names)
    out = {"uid": np.array([u for u, _, _ in rows], np.int64), "cands": np.array(names),
           "traj": np.stack([t for _, t, _ in rows]).astype(np.float32) if rows else np.zeros((0, K, S.N21, 4), np.float32)}
    for k, dt in (("S", np.float32), ("P", np.float32), ("prog", np.float32), ("NC", np.uint8), ("DAC", np.uint8), ("DDC", np.float32),
                  ("TTC", np.uint8), ("C", np.uint8), ("top", bool), ("D_onc", np.float32), ("lat", np.float32)):
        out[k] = np.stack([r[k] for _, _, r in rows]).astype(dt) if rows else np.zeros((0, K), dt)
    if domain != "off":
        for k, dt in (("S_all", np.float32), ("top_all", bool), ("NC_all", np.uint8)):
            out[k] = np.stack([r[k] for _, _, r in rows]).astype(dt) if rows else np.zeros((0, K), dt)
    for k, dt in (("valid", bool), ("h", np.float32), ("prog_norm", np.float32), ("exempt", bool), ("vru30", bool)):
        out[k] = np.array([r[k] for _, _, r in rows], dt)
    f = R("score", f"{domain}.npz")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    res = {"domain": domain, "slots": len(rows), "errors": len(errs), "err_examples": errs[:5], "valid": float(out["valid"].mean()) if rows else 0,
           "op_in_top": float(out["top"][:, 0].mean()) if rows else 0, "wall_s": time.time() - t0}
    _save_json(R("score", f"{domain}.json"), res)
    log.info("score %s", res)
    return res


# ================================================================ V3 / V4 / V6 and sanity (from the score tables)

def check_v346() -> dict:
    """V3: x- slots (sim pairs): op in Top >= 70 %. V4: x+ slots with the pedestrian visible >= 500 px_eq and in the
    corridor: Top holds a slowing or lateral candidate >= 90 %, and hold's NC failure rate clearly above x-'s (reported
    with both rates). V6: offset dev: rej NC * DAC * DDC = 1 >= 90 %."""
    from . import op_adapt_r2_data as C
    res = {}
    slow_lat = [S.CANDS.index(k) for k in ("op_slow", "op_stop", "brake_hard", "brake_mild", "shift_L", "shift_R",
                                             "shift_L_slow", "shift_R_slow", "nudge_L", "op_L", "op_R")]
    hold = S.CANDS.index("hold")
    for dom in ("simC", "simK"):
        f = R("score", f"{dom}.npz")
        if not f.exists():
            continue
        z = S.load_scores(dom)
        idx = C.load_index(dom).set_index("uid").loc[z["uid"]]
        minus, plus = (idx.sign < 0).to_numpy(), (idx.sign > 0).to_numpy()
        big = plus & (idx.px_eq >= 500).to_numpy() & idx.ped_corr.to_numpy(bool)
        v = z["valid"]
        v3 = float(z["top"][minus & v, 0].mean())
        v4a = float(z["top"][big & v][:, slow_lat].any(1).mean())
        h_p, h_m = float(1 - z["NC"][big, hold].mean()), float(1 - z["NC"][minus, hold].mean())
        res[dom] = {"V3": {"x_minus_valid": int((minus & v).sum()), "op_in_top": v3, "pass": v3 >= 0.70},
                    "V4": {"x_plus_big_corr_valid": int((big & v).sum()), "top_has_slow_or_lateral": v4a,
                           "hold_nc_fail_plus": h_p, "hold_nc_fail_minus": h_m, "pass": v4a >= 0.90 and h_p > h_m + 0.10}}
    if R("score", "off.npz").exists():
        z = S.load_scores("off")
        t = pd.read_parquet(R("offset", "table.parquet")).set_index("uid").loc[z["uid"]]
        dev = (t.split == "dev").to_numpy()
        j = list(z["cands"]).index("rej")
        ok = (z["NC"][:, j] == 1) & (z["DAC"][:, j] == 1) & (z["DDC"][:, j] == 1)
        by = {int(c): float(ok[dev & (t.corner == c).to_numpy()].mean()) for c in range(4)}
        res["off"] = {"V6": {"dev": int(dev.sum()), "rej_ok": float(ok[dev].mean()), "by_corner": by, "pass": float(ok[dev].mean()) >= 0.90},
                      "op_dac_ddc_fail_train": float(1 - ((z["DAC"][~dev, 0] == 1) & (z["DDC"][~dev, 0] == 1)).mean())}
    _save_json(R("checks", "V346.json"), res)
    log.info("V3/V4/V6 %s", res)
    return res


def sanity() -> dict:
    """§4.4 scorer sanity (descriptive): share of slots whose op is not in Top, per set; visible- vs all-actor Top
    disagreement."""
    from . import op_adapt_r2_data as C
    out = {}
    for dom in ("simC", "simK", "nus", "p5", "off"):
        if not R("score", f"{dom}.npz").exists():
            continue
        z = S.load_scores(dom)
        idx = C.load_index(dom).set_index("uid").loc[z["uid"]]
        sets = {"all": np.ones(len(z["uid"]), bool)}
        if dom.startswith("sim"):
            sets = {"x_plus": (idx.sign > 0).to_numpy(), "x_minus": (idx.sign < 0).to_numpy()}
        elif dom == "nus":
            vru = idx.vru_wide.to_numpy(bool) & z["vru30"]
            sets = {f"{sp}_{k}": (idx.split == sp).to_numpy() & m for sp in ("train", "val") for k, m in (("vru", vru), ("normal", ~vru))}
        row = {}
        for k, m in sets.items():
            v = m & z["valid"]
            row[k] = {"n": int(v.sum()), "op_not_top": float(1 - z["top"][v, 0].mean()) if v.any() else None}
            if "top_all" in z:
                row[k]["top_vis_vs_all_differ"] = float((z["top"][v] != z["top_all"][v]).any(1).mean()) if v.any() else None
        out[dom] = row
    _save_json(R("score", "sanity.json"), out)
    return out


def nav_cam_x() -> dict:
    """token -> CAM_F0 x (m ahead of the rear axle) for navtrain / navtest from navsim_zs's slim index."""
    from . import navsim_zs as Z
    rows = []
    for split in ("navtrain", "navtest"):
        for e in Z.load_index(split, slim=True):
            rows.append({"token": e["token"], "split": split, "cam_x": float(np.asarray(e["cams"][-1]["CAM_F0"]["t"])[0])})
    t = pd.DataFrame(rows).drop_duplicates("token")
    R("maps", "nav").mkdir(parents=True, exist_ok=True)
    t.to_parquet(R("maps", "nav", "cam_x.parquet"), index=False)
    return {"tokens": len(t), "cam_x": t.cam_x.describe().to_dict()}


def score_all(workers: int | None = None) -> dict:
    """Every domain whose inputs exist (C's teacher; off: extracted NAVSIM geometry), then V3 / V4 / V6 and the sanity."""
    out = {}
    for dom in ("simC", "simK", "nus", "off", "p5"):
        if not R("teacher", f"{dom}.npz").exists():
            out[dom] = "no teacher"
            continue
        if dom == "off" and not any(R("maps", "nav", "v1_navtrain").glob("*.npz")):
            out[dom] = "no NAVSIM geometry"
            continue
        out[dom] = score_domain(dom, workers=workers)
    out["checks"] = check_v346()
    out["sanity"] = sanity()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--towns", nargs="*", default=None)
    a = ap.parse_args()
    fn = {"points": lambda: route_points(a.workers or cores()), "maps-carla": lambda: maps_carla(a.towns),
          "nus-scenes": lambda: nus_scenes(), "maps-nus": lambda: maps_nus(), "v1": lambda: check_v1(a.workers),
          "v2": lambda: check_v2(a.workers), "v346": check_v346, "sanity": sanity, "nav-cam": nav_cam_x,
          "objects-nc": lambda: objects_nc(a.towns), "expert": lambda: expert_poses(a.workers or cores()), "score": lambda: score_domain(a.domain, workers=a.workers), "score-all": lambda: score_all(a.workers)}[a.step]
    print(json.dumps(fn(), indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
