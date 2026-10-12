#!/usr/bin/env python3
"""ONPOL-LONG Amendment 15 (arm B'): the oncoming region of the on-policy rows by zeros_mech's lane rule (zm_read.Lanes, unedited), as a raster.
Box, AlpaSim's venv (the run's own scene loader, trajdata vector map, the evaluator's lane polygon and centre-line yaw).

  build --pool on1_L18YS --rows L18YS --out DIR [--workers N] [--armb DIR]
      per row, every cell centre of the (128, 96) 0.5 m row-frame raster classified with Lanes.cls's rule at the row's t0 heading (vectorised:
      same shapely operations as the evaluator's per-point functions); oncoming cells -> signed distance (+ outside) -> DIR/onc.npz (road.npz layout).
      Gates (exit 2): the log's own t0 rear axle in an oncoming lane by Lanes.cls itself <= 0.01 of rows; served-plan oncoming hinge <= 0.10; road
      hinge 0.122; no errors. Also DIR/road_dir.npz = min(pool road SDF, oncoming-free SDF) for arm B redone (its gate: road hinge <= 0.16, stats only here). --armb: arm B's road.npz -> the share of its removed cells that this rule calls oncoming (note to decision 274).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/zeros_mech/scripts"), str(R / "experiments/anchor_offset/scripts"), str(R / "experiments/body1/lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
X0, Y0, RES, NH, NW = -8.0, -24.0, 0.5, 128, 96
XC, YC = np.meshgrid(X0 + RES * (np.arange(NH) + 0.5), Y0 + RES * (np.arange(NW) + 0.5), indexing="ij")
EPS = 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def classify(lanes, P, z, psi, r):
    """Points P (n, 2) local frame, heading psi -> class (n,) int8 by Lanes.cls's rule (0 none, 1 same, 2 onc, 3 cross), lanes within r of their centre."""
    import shapely
    import zm_read as Z
    c = P.mean(0)
    same, onc, any_ = np.zeros(len(P), bool), np.zeros(len(P), bool), np.zeros(len(P), bool)
    for ln in lanes.vm.get_lanes_within(np.array([c[0], c[1], z], float), r):
        poly, cl = lanes.lane(ln)
        inn = shapely.contains_xy(poly, P[:, 0], P[:, 1])
        if not inn.any():
            continue
        pts = shapely.points(P[inn])
        dd = shapely.line_locate_point(cl, pts)
        a = shapely.get_coordinates(shapely.line_interpolate_point(cl, np.minimum(dd + EPS, cl.length)))
        b = shapely.get_coordinates(shapely.line_interpolate_point(cl, np.maximum(dd - EPS, 0)))
        d = np.abs(wrap(np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0]) - psi))
        i = np.flatnonzero(inn)
        any_[i] = True
        same[i] |= d < Z.SAME
        onc[i] |= d > Z.ONC
    return np.where(~any_, 0, np.where(same, 1, np.where(onc, 2, 3))).astype(np.int8)


def sdf_of(mask):
    from scipy.ndimage import distance_transform_edt as edt
    if not mask.any():
        return np.full(mask.shape, 20.0, np.float32)
    return (np.where(mask, -(edt(mask) - 0.5), edt(~mask) - 0.5) * RES).astype(np.float32)


def work(job):
    import zm_read as Z
    from ao_e1 import loader
    cfg, scene, idx, anchor, t_us, logfut, ab = job
    out = []
    try:
        ds = loader(cfg).get_data_source(scene)
        tr = ds.rig.trajectory
        Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
        lanes = Z.Lanes(ds.map)
    except Exception as e:  # noqa: BLE001
        return [(int(i), None, None, None, None, f"loader: {type(e).__name__}: {e}") for i in idx]
    for j, i in enumerate(idx):
        try:
            a = anchor[j]
            z = float(np.interp(t_us[j], Tg, Pg[:, 2]))
            P = Z.to_local(a, np.c_[XC.ravel(), YC.ravel()])
            cls = classify(lanes, P, z, a[2], 60.0).reshape(NH, NW)
            m = cls == 2
            lf = logfut[j][0]
            if np.isfinite(lf).all():
                q = Z.to_local(a, lf[None, :2])[0]
                lc = int(lanes.cls(q[0], q[1], z, a[2] + lf[2])[0])
            else:
                lc = -1
            rem = None
            if ab is not None:
                rm = ab[j]
                rem = (int(rm.sum()), int((rm & m).sum()))
            out.append((int(i), sdf_of(m).astype(np.float16), lc, rem, int(m.sum()), ""))
        except Exception as e:  # noqa: BLE001
            out.append((int(i), None, None, None, None, f"{type(e).__name__}: {e}"))
    return out


def cmd_build(a):
    import sweep as SW
    from ol_dirraster import act
    C = DATA / "runs/op_parity/cache" / a.pool
    z = dict(np.load(C / "road.npz"))
    M = np.load(DATA / "runs/onpol_nuplan/rows" / a.rows / "meta.npz")
    assert np.array_equal(M["names"], z["tokens"])
    n = len(z["tokens"])
    rem = None
    if a.armb:
        b = np.load(Path(a.armb) / "road.npz")["sdf"]
        rem = (z["sdf"] > 0) & (b <= 0)
        del b
    keys = sorted(set(zip(M["run"].tolist(), M["scene"].tolist())))
    jobs = []
    for run, sc in keys:
        ii = np.flatnonzero((M["run"] == run) & (M["scene"] == sc))
        jobs.append((str(Path(run) / "generated-user-config-0.yaml"), sc, ii, M["anchor"][ii], M["t_us"][ii], M["logfut"][ii], None if rem is None else rem[ii]))
    sdf, lc, err, rs, ncell = np.full_like(z["sdf"], 20.0), np.full(n, -2, np.int8), [], [], np.zeros(n, int)
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work, jobs, chunksize=1):
            for i, s, c, r_, nc, e in res:
                if e:
                    err.append((i, e))
                    continue
                sdf[i], lc[i], ncell[i] = s, c, nc
                if r_ is not None:
                    rs.append(r_)
    d = SW.dense(M["poses"][:, None].astype(np.float32), np.zeros((n, 1, 2), np.float32))
    m_, ins = SW.corner_margins(d, sdf.astype(np.float32))
    a_onc = (np.where(ins.all(-1), m_.min(-1), np.inf) < 0).any(-1)[:, 0]
    a_road = act(M["poses"], z["sdf"])
    lf = M["logfut"][:, 0]
    okf = np.isfinite(lf).all(1)
    Q = np.zeros((n, 1, 8, 3), np.float32)
    m0, ins0 = SW.corner_margins(SW.dense(Q, np.zeros((n, 1, 2), np.float32)), sdf.astype(np.float32))
    ego_in = (m0[:, 0, 0].min(-1) < 0)
    dpsi = np.degrees(np.abs(wrap(M["logfut"][:, -1, 2] - M["logfut"][:, 0, 2])))
    bins = {f"{lo}-{hi} deg": (dpsi >= lo) & (dpsi < hi) for lo, hi in ((0, 15), (15, 45), (45, 360))}
    inl = okf & (lc > 0)
    st = dict(rows=n, errors=len(err), err_sample=err[:5], log_onc_all=float((lc[okf] == 2).mean()), log_onc_in_lane=float((lc[inl] == 2).mean()),
              log_same_in_lane=float((lc[inl] == 1).mean()), log_rows=int(okf.sum()), log_rows_in_lane=int(inl.sum()),
              onc_cells_rows_nonempty=float((ncell > 0).mean()), onc_active=float(a_onc.mean()), road_active=float(a_road.mean()),
              ego_t0_footprint_in_onc=float(ego_in.mean()), onc_active_by_log_turn={k: float(a_onc[b].mean()) for k, b in bins.items()})
    if rs:
        rs = np.array(rs)
        st["armb_removed_cells"] = int(rs[:, 0].sum())
        st["armb_removed_cells_zm_onc_share"] = float(rs[:, 1].sum() / max(rs[:, 0].sum(), 1))
        sdfb = np.load(Path(a.armb) / "road.npz")["sdf"]
        m1, ins1 = SW.corner_margins(SW.dense(Q, np.zeros((n, 1, 2), np.float32)), sdfb.astype(np.float32))
        st["armb_ego_t0_footprint_off_road"] = float((m1[:, 0, 0].min(-1) < 0).mean())
    road_dir = np.minimum(z["sdf"].astype(np.float32), sdf.astype(np.float32)).astype(np.float16)     # arm B redone: road minus oncoming cells
    a_rd = act(M["poses"], road_dir)
    st |= dict(roaddir_active=float(a_rd.mean()), gate_b=bool(a_rd.mean() <= 0.16),
               roaddir_t0_footprint_off=float((SW.corner_margins(SW.dense(Q, np.zeros((n, 1, 2), np.float32)), road_dir.astype(np.float32))[0][:, 0, 0].min(-1) < 0).mean()))
    st["gate_human"] = bool(st["log_onc_all"] <= 0.01)
    st["gate_onc"] = bool(st["onc_active"] <= 0.10)
    st["gate_road"] = bool(round(st["road_active"], 3) == 0.122)
    st["gate_errors"] = not err
    st["pass"] = st["gate_human"] and st["gate_onc"] and st["gate_road"] and st["gate_errors"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "onc.npz", **(z | {"sdf": sdf, "ok": np.ones(n, bool), "layers": np.array(("oncoming: zeros_mech Lanes.cls rule, per 0.5 m cell",))}))
    np.savez(out / "road_dir.npz", **(z | {"sdf": road_dir, "layers": np.array(tuple(z["layers"]) + ("minus oncoming cells, zeros_mech rule",))}))
    (out / "onc_stats.json").write_text(json.dumps(st, indent=1) + "\n")
    print(json.dumps(st, indent=1), flush=True)
    if not st["pass"]:
        print("label gate failed: no training", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pool", default="on1_L18YS"), p.add_argument("--rows", default="L18YS"), p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=64), p.add_argument("--armb", default="")
    a = ap.parse_args()
    {"build": cmd_build}[a.cmd](a)
