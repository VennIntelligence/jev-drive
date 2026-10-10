"""FOV check (experiments/corridor): on navtest tokens with a logged 4 s heading change > 45 deg, was the road the driver needed inside the
camera view the model receives? CPU only, envs/navsim2 (nuPlan map layers via the v2 metric cache). The map, the logged future and the
departure points are privileged: analysis only.

  $DATA_DIR/envs/navsim2/bin/python experiments/corridor/scripts/fov_build.py build [--procs N] [--limit K]
      -> $DATA_DIR/runs/corridor/fov/tok.parquet   one row per token: bearings and visibility of the logged path, the exit-lane centreline
                                                   and the inside / outside road edges, from the t0 camera and from the history cameras
      -> $DATA_DIR/runs/corridor/fov/unit.parquet  one row per DAC-failing (token, member) of SH30-F-s0 / s1 and WA-JEPA: the first
                                                   footprint departure of the LQR replay (the scorer's own, decision 153's replay) and the
                                                   visibility of that point, of the logged-path point at the same arc length, and of the
                                                   road edge within EDGE_R of the departure

Views (all from code): the model receives the openpilot wide frame (focal 455 px in a 512 x 256 frame, principal point row 151.8 = 58.7 deg
horizontal) and the road frame (focal 910, row 47.6 = 31.4 deg), both rendered from NAVSIM CAM_F0 (jevdrive.navsim_zs.OpenpilotMaps) with
the optical axis along the ego x axis, the viewpoint at the CAM_F0 position (rays only, no parallax) and calibration level / straight.
Ground points sit at z = -0.35 m (NAVSIM's ego origin is the rear axle at wheel-centre height) and the camera at its sensor2lidar z; "in
image" is the pinhole test u in [0, W), v in [0, H) of that frame; "in FOV" is the horizontal test only. WA-JEPA reads CAM_L0 / F0 / R0 /
B0 natively (pinhole test on the native 1920 x 1080 image; lens distortion ignored, so the edge of the field is approximate).
History cameras: the 4 keyframes the model reads at t0 - 1.5, -1.0, -0.5, 0 s (2 Hz logs).
"""
import argparse
import glob
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/corridor/lib")]
import fd_navsim as FD  # noqa: E402  (sets the nuPlan / navsim env vars)
import lanegraph as LG  # noqa: E402

OUT = D / "runs/corridor/fov"
LOGS = D / "datasets/navsim/navsim_logs/test"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
PT = D / "runs/op_parity/pt_swap"
GEOM = D / "runs/corridor/geom.pkl"
UNITS = {"sh0": "sh0_pp", "sh1": "sh1_pp", "wa": "wa_pp"}
GROUND_Z = -0.35
NEAR = 8.2              # m, ground nearer than this ahead of the camera is below the bottom row of the wide / road frames (camera 1.87 m high)
EDGE_R = 10.0            # m, road edge within this radius of the departure point is "the edge the plan crossed"
LAT_MAX = 12.0           # m, lateral search for the road edge beside the logged path
HIST = (0, -1, -2, -3)   # keyframe offsets (frames at 2 Hz); 0 = t0
WIDE = dict(W=512, H=256, f=455.0, cx=256.0, cy=151.8)
ROAD = dict(W=512, H=256, f=910.0, cx=256.0, cy=47.6)
R_VIRT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])      # camera (right, down, fwd) columns in ego (x fwd, y left, z up)
_G = {}


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def project(P, pos, R, K):
    """Ego-frame points (N, 3) -> pixels and depth for a pinhole camera at `pos` with camera-to-ego rotation R and K = (f, cx, cy) or a 3x3."""
    c = (np.asarray(P, float) - pos) @ R
    z = c[:, 2]
    zs = np.where(z > 1e-3, z, np.inf)
    if np.ndim(K) == 2:
        return K[0, 0] * c[:, 0] / zs + K[0, 2], K[1, 1] * c[:, 1] / zs + K[1, 2], z
    f, cx, cy = K
    return f * c[:, 0] / zs + cx, f * c[:, 1] / zs + cy, z


def virt_vis(xy, base, view):
    """Visibility of ground points xy (N, 2), t0 ego frame, for the model's virtual camera at the pose base = (x, y, yaw, cam_dx, cam_dy, cam_z)
    -> (in horizontal FOV, in image)."""
    x, y, yaw, dx, dy, cz = base
    pos = np.array([x, y, 0.0]) + rz(yaw)[:, :2] @ np.array([dx, dy]) + np.array([0, 0, cz])
    P = np.c_[xy, np.full(len(xy), GROUND_Z)]
    u, v, z = project(P, pos, rz(yaw) @ R_VIRT, (view["f"], view["cx"], view["cy"]))
    fov = (z > 1e-3) & (u >= 0) & (u < view["W"])
    return fov, fov & (v >= 0) & (v < view["H"])


def wa_vis(xy, base, cams):
    """WA-JEPA: visible in any of its four native cameras (pinhole test; distortion ignored)."""
    x, y, yaw = base[:3]
    P = np.c_[xy, np.full(len(xy), GROUND_Z)]
    fov = np.zeros(len(xy), bool)
    img = np.zeros(len(xy), bool)
    for c in cams.values():
        pos = np.array([x, y, 0.0]) + rz(yaw)[:, :2] @ c["t"][:2] + np.array([0, 0, c["t"][2]])
        u, v, z = project(P, pos, rz(yaw) @ c["R"], c["K"])
        h = (z > 1e-3) & (u >= 0) & (u < 1920)
        fov |= h
        img |= h & (v >= 0) & (v < 1080)
    return fov, img


def frames_of(log):
    if log not in _G:
        fr = pickle.load(open(LOGS / f"{log}.pkl", "rb"))
        xy = np.array([f["ego2global_translation"][:2] for f in fr], np.float64)
        yaw = np.array([LG.yaw_of(f["ego2global_rotation"]) for f in fr])
        _G[log] = (fr, xy, yaw, {f["token"]: i for i, f in enumerate(fr)})
    return _G[log]


def to_ego(pts, o, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(pts, float) - o
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def resample(P, step):
    """Polyline (N, 2) -> points at every `step` m of arc length (first point included), and the arc lengths."""
    seg = np.hypot(*np.diff(P, axis=0).T)
    S = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, S[-1] + 1e-9, step)
    return np.stack([np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])], 1), s


def edge_points(path, hd, area_ego, sgn):
    """Beside each path sample, the nearest drivable-area boundary hit on the inside (turn side) and outside; NaN when none within LAT_MAX."""
    import shapely
    bnd = area_ego.boundary
    n = np.stack([-np.sin(hd), np.cos(hd)], 1)
    out = {}
    for name, sd in (("in", sgn), ("out", -sgn)):
        a = path + 1e-3 * sd * n
        b = path + LAT_MAX * sd * n
        segs = shapely.linestrings(np.stack([a, b], 1))
        hit = shapely.intersection(segs, bnd)
        pts = np.full((len(path), 2), np.nan)
        for k, g in enumerate(hit):
            if g.is_empty:
                continue
            c = shapely.get_coordinates(g)
            pts[k] = c[np.argmin(np.hypot(*(c - path[k]).T))]
        out[name] = pts
    return out


def vis_set(pts, bases, views, wa_cams):
    """pts (N, 2) -> dict of boolean arrays: view -> (fov, img) at t0 and 'any' over the history keyframes."""
    out = {}
    ok = np.isfinite(pts).all(1)
    p = np.where(ok[:, None], pts, 0.0)
    for nm, f in (("w", lambda b: virt_vis(p, b, WIDE)), ("r", lambda b: virt_vis(p, b, ROAD)), ("a", lambda b: wa_vis(p, b, wa_cams))):
        r = [f(b) for b in bases]
        for j, kind in enumerate(("fov", "img")):
            out[f"{nm}_{kind}_t0"] = r[0][j] & ok
            out[f"{nm}_{kind}_any"] = np.any([x[j] for x in r], 0) & ok
    return out


def bearing(pts, base):
    """Bearing (deg, left +) of points from the t0 camera relative to the ego heading."""
    x, y, yaw, dx, dy, _ = base
    cam = np.array([x, y]) + rz(yaw)[:2, :2] @ np.array([dx, dy])
    d = pts - cam
    c, s = np.cos(yaw), np.sin(yaw)
    return np.degrees(np.arctan2(-s * d[:, 0] + c * d[:, 1], c * d[:, 0] + s * d[:, 1]))


def pick_s(P, s, targets):
    """Points of the resampled polyline P (arc s) at arc lengths `targets`; NaN beyond the end."""
    out = np.full((len(targets), 2), np.nan)
    for k, t in enumerate(targets):
        if t <= s[-1]:
            out[k] = [np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])]
    return out


def work(token):
    tab, geom, mcp, units = _G["tab"], _G["geom"], _G["mc"], _G["units"]
    i = _G["pos"][token]
    log = str(tab["log"][i])
    fr, gxy, gyaw, fpos = frames_of(log)
    j = fpos[token]
    o, yaw0 = gxy[j], gyaw[j]
    cams = {c: fr[j]["cams"][c] for c in ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")}
    f0 = cams["CAM_F0"]
    cdx, cdy, cz = np.asarray(f0["sensor2lidar_translation"], float)
    wa_cams = {c: dict(t=np.asarray(v["sensor2lidar_translation"], float), R=np.asarray(v["sensor2lidar_rotation"], float),
                       K=np.asarray(v["cam_intrinsic"], float)) for c, v in cams.items()}
    bases = []                                       # (x, y, yaw, cam dx, dy, z) of the keyframes in the t0 ego frame
    for h in HIST:
        k = max(j + h, 0)
        p = to_ego(gxy[k][None], o, yaw0)[0]
        bases.append((p[0], p[1], float(LG.wrap(gyaw[k] - yaw0)), cdx, cdy, cz))
    fut = tab["fut"][i].astype(float)
    sgn = float(np.sign(fut[-1, 2]))
    t = np.arange(0, 41) * 0.1
    P8 = np.vstack([[0, 0, 0], fut])
    dense = np.stack([np.interp(t, np.arange(0, 9) * 0.5, P8[:, 0]), np.interp(t, np.arange(0, 9) * 0.5, P8[:, 1])], 1)
    path, s = resample(dense, 0.25)
    hd = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0]))
    with lzma.open(mcp[token], "rb") as f:
        mc = pickle.load(f)
    import shapely
    from shapely.affinity import affine_transform
    from shapely.ops import unary_union
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    am = mc.drivable_area_map
    area = unary_union([am._geometries[k] for k in am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])])
    om = np.array(mc.ego_state.rear_axle.serialize())
    c, sn = np.cos(om[2]), np.sin(om[2])
    area_e = affine_transform(area, [c, sn, -sn, c, -(c * om[0] + sn * om[1]), -(-sn * om[0] + c * om[1])])
    row = dict(token=token, log=log, sgn=sgn, dpsi=float(np.degrees(abs(fut[-1, 2]))), path_len=float(s[-1]), v0=float(mc.ego_state.dynamic_car_state.speed))
    # --- bearings of the logged path
    base0 = bases[0]
    for nm, tt in (("1s", fut[1, :2]), ("2s", fut[3, :2]), ("3s", fut[5, :2]), ("4s", fut[7, :2])):
        row[f"brg_path_{nm}"] = float(bearing(tt[None], base0)[0])
        row[f"dist_path_{nm}"] = float(np.hypot(tt[0] - base0[3], tt[1] - base0[4]))     # from the camera, m (points under ~8 m are below the image)
    for a in (5, 10, 15, 20, 30):
        p = pick_s(path, s, [a])
        row[f"brg_path_s{a}"] = float(bearing(p, base0)[0]) if np.isfinite(p).all() else np.nan
    # --- logged path visibility (fraction of arc length)
    cam0 = np.array([cdx, cdy])
    farm = np.hypot(*(path - cam0).T) >= NEAR        # points nearer than NEAR m ahead of the camera are below the wide frame at any heading
    row["path_far_frac"] = float(farm.mean())
    pv = vis_set(path[farm], bases, None, wa_cams) if farm.any() else {}
    for k, v in pv.items():
        row[f"path_{k}"] = float(v.mean())
    # --- road edges beside the logged path (inside / outside)
    ed = edge_points(path[::4], hd[::4], area_e, sgn)                     # every 1 m
    for nm, pts in ed.items():
        okp = np.isfinite(pts).all(1)
        pts = np.where(okp[:, None], pts, np.nan)
        okp &= np.hypot(*(np.nan_to_num(pts) - cam0).T) >= NEAR
        pts = np.where(okp[:, None], pts, np.nan)
        row[f"edge_{nm}_n"] = int(okp.sum())
        row[f"edge_{nm}_dist_med"] = float(np.nanmedian(np.hypot(*(pts - path[::4]).T))) if okp.any() else np.nan
        for a in (10, 20, 30):
            ia = np.argmin(np.abs(s[::4] - a))
            row[f"brg_edge_{nm}_s{a}"] = float(bearing(pts[ia][None], base0)[0]) if s[::4][ia] <= s[-1] and okp[ia] and abs(s[::4][ia] - a) < 1.01 else np.nan
        for k, v in vis_set(pts, bases, None, wa_cams).items():
            row[f"edge_{nm}_{k}"] = float(v[okp].sum() / max(okp.sum(), 1)) if okp.any() else np.nan
    # --- exit-lane centreline (corridor R, ego frame, starts abeam): bearings at fixed arc lengths and visibility of its first 40 m
    g = geom.get(token)
    if g is not None and g.get("R") is not None:
        Rp, sR = resample(np.asarray(g["R"], float), 0.5)
        for a in (10, 20, 30, 40):
            p = pick_s(Rp, sR, [a])
            row[f"brg_cl_s{a}"] = float(bearing(p, base0)[0]) if np.isfinite(p).all() else np.nan
        m40 = (sR <= 40) & (np.hypot(*(Rp - cam0).T) >= NEAR)
        for k, v in vis_set(Rp[m40], bases, None, wa_cams).items():
            row[f"cl_{k}"] = float(v.mean())
    # --- departures of the failing members
    urows = []
    for key in units.get(token, []):
        r = _G["dep"].get((key, token))
        if r is None:
            continue
        u = dict(token=token, log=log, unit=key, sgn=sgn, dpsi=row["dpsi"])
        u.update({k: r[k] for k in ("lqr_out", "lqr_t", "lqr_x", "lqr_y", "lqr_side", "lqr_corner", "lqr_depth", "raw_out", "raw_t", "raw_x", "raw_y") if k in r})
        for pre in ("lqr", "raw"):
            if not r.get(f"{pre}_out"):
                continue
            q = np.array([[r[f"{pre}_x"], r[f"{pre}_y"]]])
            u[f"{pre}_brg"] = float(bearing(q, base0)[0])
            for k, v in vis_set(q, bases, None, wa_cams).items():
                u[f"{pre}_pt_{k}"] = bool(v[0])
            # the logged-path point at the plan's arc length up to the departure (the road the plan needed to follow there)
            plan = np.vstack([[0, 0], np.asarray(_G["plans"][key][token], float)[:, :2]])
            pdn = np.stack([np.interp(t, np.arange(0, 9) * 0.5, plan[:, 0]), np.interp(t, np.arange(0, 9) * 0.5, plan[:, 1])], 1)
            sdep = float(np.r_[0, np.cumsum(np.hypot(*np.diff(pdn, axis=0).T))][int(round(r[f"{pre}_t"] * 10))])
            lp = pick_s(path, s, [sdep])
            u[f"{pre}_sdep"] = sdep
            if np.isfinite(lp).all():
                u[f"{pre}_need_brg"] = float(bearing(lp, base0)[0])
                for k, v in vis_set(lp, bases, None, wa_cams).items():
                    u[f"{pre}_need_{k}"] = bool(v[0])
            # road edge within EDGE_R of the departure point
            cut = shapely.intersection(area_e.boundary, shapely.buffer(shapely.points(q[0]), EDGE_R))
            if not cut.is_empty:
                cut = shapely.segmentize(cut, 0.5)
                ep = shapely.get_coordinates(cut)
                u[f"{pre}_edge_n"] = len(ep)
                for k, v in vis_set(ep, bases, None, wa_cams).items():
                    u[f"{pre}_edge_{k}"] = float(v.mean())
        urows.append(u)
    return row, urows


def init(tokens_file):
    _G.update(pickle.load(open(tokens_file, "rb")))


def cmd_build(a):
    import pandas as pd
    import multiprocessing as mp
    from jevdrive.common import n_cpus
    OUT.mkdir(parents=True, exist_ok=True)
    tab = {k: v for k, v in np.load(TAB).items()}
    names = tab["names"].astype(str)
    fut = tab["fut"].astype(float)
    dpsi = np.degrees(np.abs(fut[:, -1, 2]))
    sel = names[dpsi > 45]
    if a.limit:
        sel = sel[:: max(1, len(sel) // a.limit)][: a.limit]
    sc = pd.read_csv(PT / "score_all.csv", usecols=["key", "token", "drivable_area_compliance"])
    sc = sc[sc.key.isin(UNITS.values()) & sc.token.isin(set(sel)) & (sc.drivable_area_compliance < 1)]
    Z = np.load(PT / "poses.npz")
    posz = {t: i for i, t in enumerate(Z["tokens"].astype(str))}
    plans = {k: {t: np.asarray(Z[k][posz[t]], np.float64) for t in g.token} for k, g in sc.groupby("key")}
    want = {}
    for k, g in sc.groupby("key"):
        for t in g.token:
            want.setdefault(t, []).append(k)
    # the scorer's replay (decision 153): runs pdm_score on the failing members and returns the first footprint departure
    kf = OUT / "dep_keys.pkl"
    pickle.dump({"plans": plans, "want": want}, open(kf, "wb"))
    procs = min(a.procs, max(4, n_cpus() // 6))
    dep = {}
    with mp.get_context("fork").Pool(procs, initializer=FD._init, initargs=("navtest", kf)) as pool:
        for r in pool.imap_unordered(FD.work, sorted(want), chunksize=2):
            for x in r["rows"]:
                dep[(x["key"], r["token"])] = {k: v for k, v in x.items() if k != "_states"}
    print(f"replayed {len(dep)} failing members of {len(want)} tokens", flush=True)
    geom = {r["token"]: r for r in pickle.load(open(GEOM, "rb"))}
    mcp = {Path(p).parent.name: p for p in glob.glob(str(FD.MC["navtest"] / "*/*/*/metric_cache.pkl"))}
    pk = OUT / "work_ctx.pkl"
    pickle.dump(dict(tab=tab, geom=geom, mc=mcp, units=want, pos={t: i for i, t in enumerate(names)}, dep=dep, plans=plans), open(pk, "wb"), protocol=4)
    rows, urows = [], []
    with mp.get_context("fork").Pool(procs, initializer=init, initargs=(pk,)) as pool:
        for r, u in pool.imap_unordered(work, [str(t) for t in sel], chunksize=4):
            rows.append(r)
            urows.extend(u)
    pd.DataFrame(rows).to_parquet(OUT / "tok.parquet")
    pd.DataFrame(urows).to_parquet(OUT / "unit.parquet")
    pk.unlink()
    print(f"tokens {len(rows)} units {len(urows)} -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--procs", type=int, default=32)
    b.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cmd_build(a)
