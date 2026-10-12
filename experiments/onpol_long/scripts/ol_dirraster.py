#!/usr/bin/env python3
"""ONPOL-LONG Amendment 13 (arm B): the S road-hinge raster of the on-policy rows with direction. Box, envs/navsim2 (nuPlan map API), CPU.

  build --pool on1_L18YS --rows L18YS --out DIR [--workers N]
      per row: the road-and-lane raster of on_rows.py (ROAD_LAYERS through opb_labels' rasteriser, same radius / grid) without the LANE polygons whose
      baseline path, at its point nearest to the row origin, points more than 90 deg away from the row heading, and without the ROADBLOCKs all of whose
      lanes are dropped; INTERSECTION / LANE_CONNECTOR / ROADBLOCK_CONNECTOR kept. Rows with nothing dropped keep the pool's raster bit for bit.
      -> DIR/road.npz (the pool's road.npz layout, same tokens), DIR/dir_stats.json / .md (gate (i), changed rows, hinge activation before / after)
  fig   --events CSV --run <pseudo run> --tag T --out F.pkl     old and new raster at chosen held decisions (the early-inside across-traffic turns)
  onc   --pool on1_L18YS --rows L18YS --out DIR [--workers N]   Amendment 14 (arm B'): the oncoming region per row = arm B's removed lanes / roadblocks
        minus intersections and connectors; raster = signed distance to it (+ outside) -> DIR/onc.npz (road.npz layout), DIR/onc_stats.json; label gate:
        oncoming hinge active on the served plan <= 0.10 of rows and the pool road hinge activity 0.122 (exit 2 otherwise)
  lines --reads <zm_read report dir> --collect <zm_read collect dir> --out F.json   (AlpaSim venv) Amendment 13's mechanism lines per label group from
        zm_read's per_rollout_reads.csv.gz (its own classifications, unedited): early-inside zeros on across-traffic turns, late-outside zeros on
        kerb-side turns, share of corridor + collision zeros with the rear axle in an oncoming lane 2 s before, straight corridor zeros past the log end
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/op_probe/scripts"), str(R / "experiments/body1/lib"), str(R / "experiments/onpol_nuplan/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
LAYERS = ("ROADBLOCK", "ROADBLOCK_CONNECTOR", "INTERSECTION", "LANE", "LANE_CONNECTOR")
MAXD = np.radians(90.0)
_LOC = {}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def loc_of(log):
    if log not in _LOC:
        fr = pickle.load(open(DATA / "datasets/navsim/navsim_logs/test" / f"{log}.pkl", "rb"))
        _LOC[log] = fr[0]["map_location"]
    return _LOC[log]


def raster(polys, pose):
    """opb_labels.sdf_raster after its map query, on a given polygon list (same canvas, same SDF)."""
    import cv2
    import opb_labels as OL
    from scipy.ndimage import distance_transform_edt as edt
    from shapely.ops import unary_union
    img = np.zeros((OL.NX, OL.NY), np.uint8)
    if polys:
        u = unary_union(polys)
        for g in (u.geoms if hasattr(u, "geoms") else [u]):
            cv2.fillPoly(img, [OL._ring(g.exterior.coords, pose)], 1, shift=4)
            for h in g.interiors:
                cv2.fillPoly(img, [OL._ring(h.coords, pose)], 0, shift=4)
    ins = img > 0
    if not ins.any():
        return np.full((OL.NH, OL.NW), -20.0, np.float32)
    d = edt(ins) - edt(~ins)
    d = np.where(ins, d - 0.5, d + 0.5) * OL.FINE
    p = int(OL.PAD / OL.FINE)
    return d[p:OL.NX - p, p:OL.NY - p].reshape(OL.NH, 2, OL.NW, 2).mean((1, 3)).astype(np.float32)


def lane_dir(lane, pose):
    P = np.array([[s.x, s.y, s.heading] for s in lane.baseline_path.discrete_path])
    i = int(np.hypot(P[:, 0] - pose[0], P[:, 1] - pose[1]).argmin())
    return float(abs(wrap(P[i, 2] - pose[2])))


def directional(m, pose, drop=True):
    """-> (sdf or None when nothing is dropped, number of lanes dropped, number of roadblocks dropped); drop False: every polygon (gate (i))."""
    import opb_labels as OL
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    objs = m.get_proximal_map_objects(Point2D(float(pose[0]), float(pose[1])), OL.RADIUS, [getattr(L, n) for n in LAYERS])
    if not drop:
        return raster([o.polygon for v in objs.values() for o in v], pose), 0, 0
    bad = {o.id for o in objs[L.LANE] if lane_dir(o, pose) > MAXD}
    rb_bad = set()
    for o in objs[L.ROADBLOCK]:
        lanes = list(o.interior_edges)
        if lanes and all((x.id in bad) or lane_dir(x, pose) > MAXD for x in lanes):
            rb_bad.add(o.id)
    if not bad and not rb_bad:
        return None, 0, 0
    polys = [o.polygon for k, v in objs.items() for o in v
             if not ((k == L.LANE and o.id in bad) or (k == L.ROADBLOCK and o.id in rb_bad))]
    return raster(polys, pose), len(bad), len(rb_bad)


def oncoming(m, pose):
    """-> signed distance (+ outside) to the oncoming region (arm B's dropped set minus intersections and connectors), region non-empty."""
    import opb_labels as OL
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.ops import unary_union
    objs = m.get_proximal_map_objects(Point2D(float(pose[0]), float(pose[1])), OL.RADIUS, [getattr(L, n) for n in LAYERS])
    bad = {o.id for o in objs[L.LANE] if lane_dir(o, pose) > MAXD}
    reg = [o.polygon for o in objs[L.LANE] if o.id in bad]
    for o in objs[L.ROADBLOCK]:
        lanes = list(o.interior_edges)
        if lanes and all((x.id in bad) or lane_dir(x, pose) > MAXD for x in lanes):
            reg.append(o.polygon)
    if not reg:
        return np.full((OL.NH, OL.NW), 20.0, np.float32), False
    keep = [o.polygon for k in (L.INTERSECTION, L.LANE_CONNECTOR, L.ROADBLOCK_CONNECTOR) for o in objs[k]]
    r = unary_union(reg).difference(unary_union(keep)) if keep else unary_union(reg)
    if r.is_empty:
        return np.full((OL.NH, OL.NW), 20.0, np.float32), False
    return -raster([r], pose), True


def work_onc(job):
    import opb_labels as OL
    loc, idx, poses = job
    m = OL._map(loc)
    out = []
    for i, p in zip(idx, poses):
        try:
            s, ne = oncoming(m, p)
            out.append((i, np.clip(s, -20, 20).astype(np.float16), ne, ""))
        except Exception as e:  # noqa: BLE001
            out.append((i, None, False, f"{type(e).__name__}: {e}"))
    return out


def cmd_onc(a):
    import opb_labels  # noqa: F401
    C = DATA / "runs/op_parity/cache" / a.pool
    z = dict(np.load(C / "road.npz"))
    M = np.load(DATA / "runs/onpol_nuplan/rows" / a.rows / "meta.npz")
    assert np.array_equal(M["names"], z["tokens"])
    n = len(z["tokens"])
    locs = np.array([loc_of(lg) for lg in M["log"]])
    jobs = [(lc, q, z["pose_global"][q]) for lc in sorted(set(locs)) for q in np.array_split(np.flatnonzero(locs == lc), max(1, int((locs == lc).sum()) // 200))]
    sdf, ne, err = np.full_like(z["sdf"], 20.0), np.zeros(n, bool), []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work_onc, jobs):
            for i, s_, e_, msg in res:
                if msg:
                    err.append((int(i), msg))
                else:
                    sdf[i], ne[i] = s_, e_
    import sweep as SW
    d = SW.dense(M["poses"][:, None].astype(np.float32), np.zeros((n, 1, 2), np.float32))
    m_, ins = SW.corner_margins(d, sdf.astype(np.float32))
    a_onc = (np.where(ins.all(-1), m_.min(-1), np.inf) < 0).any(-1)[:, 0]
    a_road = act(M["poses"], z["sdf"])
    dpsi = np.degrees(np.abs(wrap(M["logfut"][:, -1, 2] - M["logfut"][:, 0, 2])))
    bins = {f"{lo}-{hi} deg": (dpsi >= lo) & (dpsi < hi) for lo, hi in ((0, 15), (15, 45), (45, 360))}
    st = dict(rows=n, errors=len(err), err_sample=err[:5], region_nonempty=float(ne.mean()), onc_active=float(a_onc.mean()),
              road_active=float(a_road.mean()), onc_active_by_log_turn={k: float(a_onc[b].mean()) for k, b in bins.items()},
              gate_onc=bool(a_onc.mean() <= 0.10), gate_road=bool(round(float(a_road.mean()), 3) == 0.122), gate_errors=not err)
    st["pass"] = st["gate_onc"] and st["gate_road"] and st["gate_errors"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "onc.npz", **(z | {"sdf": sdf, "ok": np.ones(n, bool), "layers": np.array(("oncoming: LANE / ROADBLOCK > 90 deg minus INTERSECTION / connectors",))}))
    (out / "onc_stats.json").write_text(json.dumps(st, indent=1) + "\n")
    print(json.dumps(st, indent=1), flush=True)
    if not st["pass"]:
        print("label gate failed: no training", flush=True)
        raise SystemExit(2)


def work(job):
    import opb_labels as OL
    loc, idx, poses, chk = job
    m = OL._map(loc)
    out = []
    for i, p, c in zip(idx, poses, chk):
        try:
            s, nl, nr = directional(m, p)
            g = directional(m, p, drop=False)[0] if c else None
            out.append((i, None if s is None else np.clip(s, -20, 20).astype(np.float16), nl, nr,
                        None if g is None else np.clip(g, -20, 20).astype(np.float16), ""))
        except Exception as e:  # noqa: BLE001
            out.append((i, None, 0, 0, None, f"{type(e).__name__}: {e}"))
    return out


def act(poses, sdf, margin=0.3):
    """served plans (n, 8, 3), rasters (n, 128, 96) -> road hinge active (any covered footprint corner margin < 0.3 m over the dense plan)."""
    import sweep as SW
    d = SW.dense(poses[:, None].astype(np.float32), np.zeros((len(poses), 1, 2), np.float32))
    m, ins = SW.corner_margins(d, sdf.astype(np.float32))
    return (np.where(ins.all(-1), m.min(-1), np.inf) < margin).any(-1)[:, 0]


def cmd_build(a):
    import opb_labels  # noqa: F401  (its environment defaults before the workers fork)
    C = DATA / "runs/op_parity/cache" / a.pool
    z = dict(np.load(C / "road.npz"))
    M = np.load(DATA / "runs/onpol_nuplan/rows" / a.rows / "meta.npz")
    assert np.array_equal(M["names"], z["tokens"]) and np.abs(M["pose_map"] - z["pose_global"]).max() < 1e-6, "meta and road.npz rows differ"
    n = len(z["tokens"])
    chk = np.zeros(n, bool)
    chk[np.random.default_rng(0).choice(n, 200, replace=False)] = True
    locs = np.array([loc_of(lg) for lg in M["log"]])
    jobs = []
    for lc in sorted(set(locs)):
        ii = np.flatnonzero(locs == lc)
        for q in np.array_split(ii, max(1, len(ii) // 200)):
            jobs.append((lc, q, z["pose_global"][q], chk[q]))
    sdf, nl, nr, err, gate = z["sdf"].copy(), np.zeros(n, int), np.zeros(n, int), [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work, jobs):
            for i, s, l_, r_, g, e in res:
                if e:
                    err.append((int(i), e))
                    continue
                if s is not None:
                    sdf[i] = s
                nl[i], nr[i] = l_, r_
                if g is not None:
                    gate.append(float(np.abs(g.astype(np.float32) - z["sdf"][i].astype(np.float32)).max()))
    ch = (nl > 0) | (nr > 0)
    flip = ((sdf[ch] > 0) != (z["sdf"][ch] > 0)).mean() if ch.any() else 0.0
    a0, a1 = act(M["poses"], z["sdf"]), act(M["poses"], sdf)
    dpsi = np.degrees(np.abs(wrap(M["logfut"][:, -1, 2] - M["logfut"][:, 0, 2]))) if "logfut" in M.files else np.zeros(n)
    bins = {f"{lo}-{hi} deg": (dpsi >= lo) & (dpsi < hi) for lo, hi in ((0, 15), (15, 45), (45, 360))}
    st = dict(rows=n, errors=len(err), err_sample=err[:5], gate_i_max_diff=max(gate) if gate else None, gate_i_rows=len(gate),
              gate_i_pass=bool(gate and max(gate) <= 0.01 and not err), changed_rows=int(ch.sum()), changed_share=float(ch.mean()),
              lanes_dropped_mean=float(nl[ch].mean()) if ch.any() else 0.0, roadblocks_dropped_rows=int((nr > 0).sum()),
              sign_flip_pixel_share_of_changed=float(flip), hinge_active_before=float(a0.mean()), hinge_active_after=float(a1.mean()),
              hinge_newly_active=int((a1 & ~a0).sum()), changed_share_by_log_turn={k: float(ch[b].mean()) if b.any() else None for k, b in bins.items()},
              rows_by_log_turn={k: int(b.sum()) for k, b in bins.items()})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "road.npz", **(z | {"sdf": sdf, "layers": np.array(LAYERS + ("directional>90deg",))}))
    np.savez_compressed(out / "dir_rows.npz", changed=ch, lanes=nl, roadblocks=nr, act_before=a0, act_after=a1, dpsi=dpsi)
    (out / "dir_stats.json").write_text(json.dumps(st, indent=1) + "\n")
    print(json.dumps(st, indent=1), flush=True)
    if not st["gate_i_pass"]:
        raise SystemExit("gate (i) failed: the rebuilt undirected raster differs from the pool's (or errors); the arm stops here")


def cmd_fig(a):
    import csv
    import on_rows as OR
    import opb_labels as OL
    ev = [r for r in csv.DictReader(open(a.events)) if r["cls"] == "early-inside"][:a.n]
    out = {}
    for r in ev:
        k = max(0, int(float(r["k_ev"])) - 2)
        g = {x["k"]: x for x in OR.geo_one((a.tag, a.run, r["scene"]))["rows"]}.get(k)
        if g is None or "pose_map" not in g:
            continue
        m = OL._map(loc_of(r["log"]))
        new = directional(m, g["pose_map"])[0]
        out[r["scene"]] = dict(k=k, type=r["type"], old=g.get("sdf_road"), new=new, poses=g.get("poses"), box=g.get("box"), bval=g.get("bval"),
                               gpath=g.get("gpath"))
        print(r["scene"], k, "changed" if new is not None else "unchanged", flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)


GROUPS = {"YSB": ("YSB-L18-s0", "YSB-L18-s1"), "YSO25": ("YSO25-L18-s0", "YSO25-L18-s1"), "YSO50": ("YSO50-L18-s0", "YSO50-L18-s1"), "YS2G": ("YS2G-L18-s0", "YS2G-L18-s1"), "YS": ("YS-L18-s0", "YS-L18-s1"), "base": ("YR10m10-F-s0",)}


def cmd_lines(a):
    import pandas as pd
    RR = pd.read_csv(Path(a.reads) / "per_rollout_reads.csv.gz")
    D = pd.read_parquet(Path(a.collect) / "per_decision.parquet")
    L17 = D.groupby("scene").s_log_t.max()
    Dk = {k: x for k, x in D.groupby(["label", "scene"])}
    ZE = RR[RR.zero.astype(bool) & RR.type.isin(("corridor", "offroad", "collision"))].copy()
    ZE["dirn"] = np.where(np.sign(ZE.dpsi_log) * ZE.s_onc > 0, "across", "kerb")
    ZE["past_end"] = [bool(Dk[(r.label, r.scene)].pipe(lambda x: x[x.k <= r.k_ev]).s_e.max() > L17[r.scene]) for r in ZE.itertuples()]
    out = {}
    for g, labs in GROUPS.items():
        if not set(labs) <= set(RR.label):
            continue
        z = ZE[ZE.label.isin(labs)]
        n = len(labs)
        cc = z[z.type.isin(("corridor", "collision"))]
        st = z[(z.scls == "straight") & (z.type == "corridor")]
        out[g] = dict(seeds=n, zeros=len(z) / n,
                      early_inside_across=int(((z.scls == "turn") & (z.dirn == "across") & (z.cls == "early-inside")).sum()) / n,
                      late_outside_kerb=int(((z.scls == "turn") & (z.dirn == "kerb") & (z.cls == "late-outside")).sum()) / n,
                      onc_pre_share=float(cc.onc_pre.astype(bool).mean()) if len(cc) else None, onc_pre_n=[int(cc.onc_pre.astype(bool).sum()), int(len(cc))],
                      straight_corridor_past_end=int(st.past_end.sum()) / n, straight_corridor=len(st) / n)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pool", default="on1_L18YS"), p.add_argument("--rows", default="L18YS"), p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=64)
    p = sub.add_parser("fig")
    p.add_argument("--events", required=True), p.add_argument("--run", required=True), p.add_argument("--tag", default="YS-L18-s1")
    p.add_argument("--out", required=True), p.add_argument("--n", type=int, default=3)
    p = sub.add_parser("onc")
    p.add_argument("--pool", default="on1_L18YS"), p.add_argument("--rows", default="L18YS"), p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=64)
    p = sub.add_parser("lines")
    p.add_argument("--reads", required=True), p.add_argument("--collect", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "fig": cmd_fig, "onc": cmd_onc, "lines": cmd_lines}[a.cmd](a)
