"""stoplbl: per-frame situation labels from the nuPlan map + the log's own traffic-light states, for NAVSIM logs (CPU only, read-only).

  $DATA_DIR/envs/navsim2/bin/python experiments/lowboard_diag/scripts/stoplbl_label.py run --split navtest|navtrain [--nlogs N] [--workers W]

Per frame t (2 Hz keyframes of the log) the ego's *hindsight route* is the driven path of the next <= 60 frames (30 s), extended 15 m straight
along the final heading (so a car that stops before a line still "has" the line ahead). Targets are the map's stop polygons crossed by that path
(type 0 pedestrian crossing, 1 stop sign, 2 traffic-light stop line, 3 turn stop = unprotected-turn yield, 4 yield) with the arc distance to the
first crossing. The light state of a traffic-light line is the log state (lane_connector id, is_red) of the lane connector the route takes
after the line. The log driver's behaviour is read from the frame speeds along the same path. Turns come from the path heading (frame yaw).
Output $DATA_DIR/runs/lowboard_diag/stoplbl/<split>.parquet (one row per frame; filter to the split's tokens afterwards).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
D = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(k, "1")
LOGDIR = {"navtest": "test", "navtrain": "trainval"}
OUT = D / "runs/lowboard_diag/stoplbl"
HOR, EXT, FMAX = 60, 15.0, 60            # frames of hindsight, straight extension (m)
A_LAT = 2.0                              # m/s^2 comfortable lateral acceleration for the required-speed reading
_M = {}


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def get_map(loc):
    if loc in _M:
        return _M[loc]
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
    from shapely.strtree import STRtree
    m = NuPlanMapFactory(get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"])).build_map_from_name(loc)
    sp = m._get_vector_map_layer(L.STOP_LINE)
    lc = m._get_vector_map_layer(L.LANE_CONNECTOR)
    cw = m._get_vector_map_layer(L.CROSSWALK)
    spg = list(sp.geometry)
    ty = sp.stop_polygon_type_fid.astype(int).values
    # stop polygon -> lane connectors that reference it as their traffic-light stop line
    sp_conn = {}
    for cid, s in zip(lc.index, lc.traffic_light_stop_line_fids):
        for f in str(s or "").split(","):
            if f.strip():
                sp_conn.setdefault(f.strip(), []).append(cid)
    d = dict(sp_ids=list(sp.index), sp=spg, ty=ty, tree=STRtree(spg), sp_conn=sp_conn, lcg=dict(zip(lc.index, lc.geometry)),
             cw=list(cw.geometry), cwtree=STRtree(list(cw.geometry)))
    _M[loc] = d
    return d


from shapely.geometry import Point, MultiPoint


def first_cross(line, poly):
    """arc distance along `line` of the first point inside/on `poly` (None if disjoint)."""
    x = line.intersection(poly)
    if x.is_empty:
        return None
    import shapely
    ds = [line.project(Point(c)) for c in shapely.get_coordinates(x)]
    return float(min(ds)) if ds else None


def label_log(job):
    from shapely.geometry import LineString, Point
    path_pkl, = job
    fr = pickle.load(open(path_pkl, "rb"))
    mp = get_map(fr[0]["map_location"])
    n = len(fr)
    xy = np.array([f["ego2global_translation"][:2] for f in fr], float)
    yaw = np.unwrap(np.array([yaw_of(f["ego2global_rotation"]) for f in fr]))
    sp = np.array([np.hypot(*f["ego_dynamic_state"][:2]) for f in fr], float)
    rows = []
    for i in range(n):
        j = min(n - 1, i + FMAX)
        p = xy[i:j + 1]
        seg = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
        s_k = np.concatenate([[0], np.cumsum(seg)]) + np.arange(len(p)) * 1e-6
        h_end = yaw[j]
        ext = p[-1] + np.array([[np.cos(h_end), np.sin(h_end)]]) * EXT
        pe = np.vstack([p, ext])
        line = LineString(pe) if len(pe) > 1 else None
        r = dict(token=fr[i]["token"], log=fr[i]["log_name"], idx=i, v0=sp[i], s_end=float(s_k[-1]), full=bool(j - i >= FMAX or s_k[-1] >= 100))
        tl = {str(a): bool(b) for a, b in (fr[i].get("traffic_lights") or [])}
        r["n_tl"] = len(tl)
        r["n_red_conn"] = sum(tl.values())
        # speeds along the path
        v_k = sp[i:j + 1]

        def beh(d):
            vl = float(np.interp(d, s_k, v_k))
            m = (s_k >= d - 12) & (s_k <= d + 1)
            vm = float(min(v_k[m].min() if m.any() else 99, vl))
            return vl, vm
        if line is not None and line.length > 1:
            cand = mp["tree"].query(line.buffer(3.0))
            hits = []
            for c in cand:
                d = first_cross(line, mp["sp"][c])
                if d is not None:
                    hits.append((d, int(mp["ty"][c]), c))
            hits.sort()
            for t, nm in ((0, "pc"), (1, "ss"), (3, "ts"), (4, "yl"), (2, "tl")):
                hh = [h for h in hits if h[1] == t and h[0] > 0.5]
                r[nm + "_d"] = hh[0][0] if hh else np.nan
                if hh:
                    r[nm + "_vline"], r[nm + "_vmin"] = beh(hh[0][0])
            # traffic-light lines: state of the connector taken after the line, for every TL line within 80 m
            tls = []
            for d, t, c in hits:
                if t != 2 or d <= 0.5 or d > 80:
                    continue
                conns = mp["sp_conn"].get(mp["sp_ids"][c], [])
                best, bd = None, 1e9
                for cid in conns:
                    g = mp["lcg"][cid]
                    qs = [line.interpolate(min(d + a, line.length)) for a in (4, 8, 12)]
                    dd = np.mean([g.distance(q) for q in qs])
                    if dd < bd:
                        best, bd = cid, dd
                st = tl.get(str(best), None) if best is not None and bd < 3.0 else None
                tls.append((d, st, any(tl.get(str(x), False) for x in conns), best is not None and bd < 3.0))
            red = [x for x in tls if x[1] is True]
            r["tl_line_d"] = tls[0][0] if tls else np.nan
            r["tl_line_state"] = {True: "red", False: "notred", None: "unk"}[tls[0][1]] if tls else ""
            r["tl_red_d"] = red[0][0] if red else np.nan
            r["tl_redany_d"] = next((x[0] for x in tls if x[2]), np.nan)
            if red:
                r["tl_red_vline"], r["tl_red_vmin"] = beh(red[0][0])
            # crosswalk with a pedestrian near it
            a = fr[i]["anns"]
            peds = np.zeros((0, 2))
            if len(a["gt_boxes"]):
                m_ = np.asarray(a["gt_names"]) == "pedestrian"
                b = a["gt_boxes"][m_][:, :2]
                c_, s_ = np.cos(yaw[i]), np.sin(yaw[i])
                peds = xy[i] + np.stack([c_ * b[:, 0] - s_ * b[:, 1], s_ * b[:, 0] + c_ * b[:, 1]], 1)
            cwd = np.nan
            for c in mp["cwtree"].query(line.buffer(1.0)):
                d = first_cross(line, mp["cw"][c])
                if d is not None and d > 0.5 and d <= 80 and len(peds) and mp["cw"][c].buffer(3.0).intersects(
                        MultiPoint([tuple(q) for q in peds])):
                    cwd = d if np.isnan(cwd) else min(cwd, d)
            r["cwped_d"] = cwd
            if not np.isnan(cwd):
                r["cwped_vline"], r["cwped_vmin"] = beh(cwd)
        # turn: heading along the path (frame yaw interpolated on arc length)
        if s_k[-1] > 10:
            g = np.arange(0, s_k[-1], 1.0)
            th = np.interp(g, s_k, yaw[i:j + 1])
            k = np.gradient(np.convolve(th, np.ones(3) / 3, "same"), 1.0)
            k = np.convolve(k, np.ones(5) / 5, "same")
            on = np.abs(k) > 0.025
            runs, a0 = [], None
            for q, o in enumerate(on):
                if o and a0 is None:
                    a0 = q
                if (not o) and a0 is not None:
                    runs.append([a0, q]); a0 = None
            if a0 is not None:
                runs.append([a0, len(on)])
            mg = []
            for ru in runs:
                if mg and ru[0] - mg[-1][1] <= 5:
                    mg[-1][1] = ru[1]
                else:
                    mg.append(ru)
            best = None
            for a_, b_ in mg:
                dth = abs(th[min(b_, len(th) - 1)] - th[a_])
                if np.degrees(dth) >= 45 and (best is None or a_ < best[0]):
                    best = (a_, b_, dth)
            if best is not None:
                a_, b_, dth = best
                r["turn_s"] = float(g[a_]); r["turn_deg"] = float(np.degrees(dth))
                r["turn_kmax"] = float(np.abs(k[a_:b_]).max())
                r["turn_vc"] = float(np.sqrt(A_LAT / max(r["turn_kmax"], 1e-3)))
                r["turn_vin"] = float(np.interp(g[a_], s_k, v_k))
                r["turn_lenb"] = float(b_ - a_)
        rows.append(r)
    return rows


def cmd_run(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    from jevdrive.common import n_cpus
    with Run("lowboard_diag", f"stoplbl-{a.split}", config=vars(a)) as run:
        sp = splits.load(f"navsim/{a.split}")
        run.use_split(sp)
        d = D / "datasets/navsim/navsim_logs" / LOGDIR[a.split]
        logs = sorted(p.stem for p in d.glob("*.pkl"))
        toks_logs = None
        if a.nlogs:
            rng = np.random.default_rng(0)
            logs = list(rng.choice(logs, min(a.nlogs, len(logs)), replace=False))
        res = par.pmap(label_log, [((d / f"{l}.pkl"),) for l in logs], run=run, workers=min(a.workers or n_cpus(), 32))
        res.raise_if_failed()
        df = pd.DataFrame([r for part in res.values for r in part])
        df = df[sp.mask(df.token)] if False else df
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT / f"{a.split}{'_s%d' % a.nlogs if a.nlogs else ''}.parquet")
        run.summary.update(frames=len(df), logs=len(logs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--nlogs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    cmd_run(a)
