#!/usr/bin/env python3
"""ONPOL-LONG Amendment 17 (label study): the rear-axle oncoming check (variant a) with the region judged per plan point at the plan's local heading,
against the t0 heading for the whole 4 s. Exact zeros_mech rule per point (no raster). Box, AlpaSim's venv, CPU.

  ol_onclab2.py --rows L18YS --events CSV --held <held run dir> --out F.json [--workers N]
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments/onpol_long/scripts"), str(R / "experiments/body1/lib"), str(R / "experiments/zeros_mech/scripts"), str(R / "experiments/anchor_offset/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def onc_points(lanes, P, z, psi):
    """Points P (n, 2) local, headings psi (n,) -> oncoming (n,) by Lanes.cls's rule (same < 60 deg wins, else any > 120 deg)."""
    import shapely
    import zm_read as Z
    c = P.mean(0)
    r = float(np.hypot(*(P - c).T).max()) + 15.0
    same, onc = np.zeros(len(P), bool), np.zeros(len(P), bool)
    for ln in lanes.vm.get_lanes_within(np.array([c[0], c[1], z], float), r):
        poly, cl = lanes.lane(ln)
        inn = shapely.contains_xy(poly, P[:, 0], P[:, 1])
        if not inn.any():
            continue
        pts = shapely.points(P[inn])
        dd = shapely.line_locate_point(cl, pts)
        a = shapely.get_coordinates(shapely.line_interpolate_point(cl, np.minimum(dd + 0.01, cl.length)))
        b = shapely.get_coordinates(shapely.line_interpolate_point(cl, np.maximum(dd - 0.01, 0)))
        d = np.abs(wrap(np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0]) - psi[inn]))
        i = np.flatnonzero(inn)
        same[i] |= d < Z.SAME
        onc[i] |= d > Z.ONC
    return onc & ~same


def dense41(P9):
    t, T = np.arange(41) * 0.1, np.arange(len(P9)) * (4.0 / (len(P9) - 1))
    return np.stack([np.interp(t, T, P9[:, c]) for c in range(3)], -1)


def tracks(served8, logfut9):
    s = dense41(np.vstack([np.zeros((1, 3)), served8]))
    h = dense41(np.nan_to_num(logfut9))
    return s, h


def judge(lanes, a, z, D):
    """D (41, 3) row-frame poses -> (active with t0 heading, active with local heading)."""
    import zm_read as Z
    P = Z.to_local(a, D[:, :2])
    return bool(onc_points(lanes, P, z, np.full(len(P), a[2])).any()), bool(onc_points(lanes, P, z, a[2] + D[:, 2]).any())


def work(job):
    import zm_read as Z
    from ao_e1 import loader
    cfg, scene, idx, anchor, t_us, served, logfut = job
    ds = loader(cfg).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
    lanes = Z.Lanes(ds.map)
    out = []
    for j, i in enumerate(idx):
        z = float(np.interp(t_us[j], Tg, Pg[:, 2]))
        s, h = tracks(served[j], logfut[j])
        out.append((int(i), *judge(lanes, anchor[j], z, s), *judge(lanes, anchor[j], z, h)))
    return out


def main(a):
    M = np.load(DATA / "runs/onpol_nuplan/rows" / a.rows / "meta.npz")
    n = len(M["names"])
    keys = sorted(set(zip(M["run"].tolist(), M["scene"].tolist())))
    jobs = []
    for run, sc in keys:
        ii = np.flatnonzero((M["run"] == run) & (M["scene"] == sc))
        jobs.append((str(Path(run) / "generated-user-config-0.yaml"), sc, ii, M["anchor"][ii], M["t_us"][ii], M["poses"][ii], M["logfut"][ii]))
    A = np.zeros((n, 4), bool)
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work, jobs, chunksize=1):
            for i, *v in res:
                A[i] = v
    lf = M["logfut"]
    dpsi = np.degrees(np.abs(wrap(lf[:, -1, 2] - lf[:, 0, 2])))
    st, tu = dpsi < 15, dpsi >= 45
    cols = ("served_t0", "served_local", "human_t0", "human_local")
    res = dict(rows=n, table={c: dict(all=float(A[:, k].mean()), straight=float(A[st, k].mean()), turn=float(A[tu, k].mean())) for k, c in enumerate(cols)})
    import zm_read as Z
    from ao_e1 import drive_records, loader
    ev = [r for r in csv.DictReader(open(a.events)) if r["cls"] == "early-inside" and np.sign(float(r["dpsi_log"])) * float(r["s_onc"]) > 0]
    recs, cfg, E = drive_records(a.held), str(Path(a.held) / "generated-user-config-0.yaml"), []
    for r in ev:
        k = max(0, int(float(r["k_ev"])) - 2)
        rr = {x["k"]: x for x in recs.get(r["scene"], [])}.get(k)
        if rr is None:
            continue
        ds = loader(cfg).get_data_source(r["scene"])
        tr = ds.rig.trajectory
        z = float(np.interp(rr["t0"], np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, 2]))
        s = dense41(np.vstack([np.zeros((1, 3)), np.asarray(rr["poses"], float)]))
        t0, loc = judge(Z.Lanes(ds.map), np.asarray(rr["anchor"], float), z, s)
        E.append(dict(scene=r["scene"][-16:-8], type=r["type"], k=k, t0=t0, local=loc))
    res["events"] = E
    res["events_active"] = dict(t0=sum(e["t0"] for e in E), local=sum(e["local"] for e in E), n=len(E))
    res["use_local"] = bool(res["table"]["human_local"]["turn"] < 0.05 and res["events_active"]["local"] >= 8)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="L18YS"), ap.add_argument("--events", required=True), ap.add_argument("--held", required=True)
    ap.add_argument("--out", required=True), ap.add_argument("--workers", type=int, default=64)
    main(ap.parse_args())
