"""Lane EDGE M1 (plan: experiments/skill_pack/plans/2026-10-04-roadedge-diagnosis-plan.md): openpilot lane lines / road
edges vs the NAVSIM scorer map and the nuPlan layers the scorer leaves out, on lateral cross-sections ahead of the ego.

navsim2 env, CPU. Per token and ego x in XS (rear-axle frame, x fwd, y left), the section is the line y in [-40, 40]:
  model: lane lines (4) and road edges (2), MDN mean, moved into the ego frame with the CAM_F0 position (openpilot calib
         frame = NAVSIM ego axes at the camera, y right, z down); lane-line probs; ground z = cam_z - z_model;
  scorer: intervals of the cached PDMDrivableMap DAC layers (ROADBLOCK, INTERSECTION, CARPARK_AREA; DRIVABLE_AREA is never
          cached); lanes: the cached LANE / LANE_CONNECTOR polygons, one interval each;
  nuPlan raw layers (map gpkg): generic_drivable_areas (GDA), walkways, carpark_areas;
  human: lateral of the logged 4 s future at x (if it reaches x); intersection: an INTERSECTION polygon within |y| < 15.
Output: <out>/sections_<board>.pkl, a DataFrame with one row per (token, x).
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
XS = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
BOARDS = {"navtest": dict(idx=L.D / "runs/navsim_zs/index/navtest_slim.pkl", mc=L.D / "runs/navsim/metric_cache/v2_navtest",
                          plans=L.D / "runs/op_lb/lb_navtest/plans/gimm@cinque.npz"),
          "navhard": dict(idx=L.IDX, mc=L.MCACHE, plans=L.NATIVE_PLANS)}
OUT = L.D / "runs/skill_pack/edge_diag"
_G = {}


def _init(board):
    import glob
    import pickle
    b = BOARDS[board]
    z = np.load(b["plans"])
    sl = json.loads(str(z["info"]))["heads_slices"]
    _G.update(names=z["names"].tolist(), heads=z["heads"], plan_pos=z["plan_pos"], plan_yaw=z["plan_yaw"], sl=sl,
              idx={e["token"]: e for e in pickle.load(open(b["idx"], "rb"))},
              cp={Path(p).parent.name: p for p in glob.glob(str(b["mc"] / "*/*/*/metric_cache.pkl"))}, maps={})
    _G["row"] = {t: i for i, t in enumerate(_G["names"])}


def _raw_layers(name):
    """generic_drivable_areas, walkways, carpark_areas of one map, as shapely STRtrees (cached per process)."""
    if name not in _G["maps"]:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        from shapely.strtree import STRtree
        a = get_maps_api(str(L.D / "datasets/navsim/maps"), "nuplan-maps-v1.0", name)
        lay = {}
        for k, ln in (("gda", "generic_drivable_areas"), ("walk", "walkways"), ("park", "carpark_areas")):
            g = [x for x in a._load_vector_map_layer(ln).geometry.values if x is not None and not x.is_empty]
            lay[k] = (STRtree(g), g)
        _G["maps"][name] = lay
    return _G["maps"][name]


def _intervals(geoms, line, to_y):
    """Lateral intervals (sorted, unmerged) where `line` crosses each geometry."""
    out = []
    for g in geoms:
        q = g.intersection(line)
        if q.is_empty:
            continue
        for s in (getattr(q, "geoms", None) or [q]):
            if s.geom_type != "LineString":
                continue
            y = [to_y(*p) for p in s.coords]
            out.append((min(y), max(y)))
    return sorted(out)


def _merge(iv, gap=0.05):
    m = []
    for a, b in iv:
        if m and a <= m[-1][1] + gap:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return m


def _around(m, y):
    """Merged interval containing y, else nan pair."""
    for a, b in m:
        if a - 1e-6 <= y <= b + 1e-6:
            return a, b
    return np.nan, np.nan


def _mdn(h, base, shape):
    n = int(np.prod(shape))
    return h[base:base + n].reshape(shape), np.exp(np.minimum(h[base + n:base + 2 * n], 11)).reshape(shape)


def work(t):
    from shapely.geometry import LineString
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    i = _G["row"][t]
    e = _G["idx"][t]
    mc = L.load_cache(_G["cp"][t])
    cam = np.asarray(e["cams"][-1]["CAM_F0"]["t"], float)
    h, sl = _G["heads"][i], _G["sl"]
    ll, _ = _mdn(h, sl["lane_lines"], (4, 33, 2))
    llp = 1 / (1 + np.exp(-h[sl["lane_lines_prob"]:sl["lane_lines_prob"] + 8][1::2]))
    re, res = _mdn(h, sl["road_edges"], (2, 33, 2))
    ex = X_IDXS + cam[0]
    m_ll = np.stack([np.interp(XS, ex, cam[1] - ll[k, :, 0]) for k in range(4)], 1)       # (6, 4), left +
    m_llz = np.stack([np.interp(XS, ex, cam[2] - ll[k, :, 1]) for k in range(4)], 1)      # ground height in ego frame
    m_re = np.stack([np.interp(XS, ex, cam[1] - re[k, :, 0]) for k in range(2)], 1)       # (6, 2) [left, right]
    m_res = np.stack([np.interp(XS, ex, res[k, :, 0]) for k in range(2)], 1)
    plan = I.to_rear(_G["plan_pos"][i], _G["plan_yaw"][i], I.T_IDXS, cam[:2], I.T_IDXS, "lever", "linear")
    px = np.maximum.accumulate(plan[:, 0])
    plan_y = np.where(XS <= px[-1], np.interp(XS, px, plan[:, 1]), np.nan)

    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    g = lambda x, y: (o[0] + c * x - s * y, o[1] + s * x + c * y)  # noqa: E731
    to_y = lambda px_, py_: -s * (px_ - o[0]) + c * (py_ - o[1])  # noqa: E731
    ht = mc.human_trajectory                                          # logged 4 s future, ego frame (None on navhard stage 2)
    hum = np.vstack([[0, 0, 0], ht.poses]) if ht is not None else np.zeros((1, 3))
    hx = np.maximum.accumulate(hum[:, 0])
    hum_y = np.where(XS <= hx[-1], np.interp(XS, hx, hum[:, 1]), np.nan)

    am = mc.drivable_area_map
    G = am._geometries
    dac = [G[k] for k in am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA])]
    lanes = [G[k] for k in am.get_indices_of_map_type([S.LANE, S.LANE_CONNECTOR])]
    inter = [G[k] for k in am.get_indices_of_map_type([S.INTERSECTION])]
    raw = _raw_layers(mc.map_parameters.map_name)
    rows = []
    for k, x in enumerate(XS):
        line = LineString([g(x, -40), g(x, 40)])
        near = lambda key: [raw[key][1][j] for j in raw[key][0].query(line)]  # noqa: E731
        sc = _merge(_intervals(dac, line, to_y))
        gda_iv = _intervals(near("gda"), line, to_y)
        ext = _merge(sorted(_intervals(dac, line, to_y) + gda_iv))
        walk = _merge(_intervals(near("walk"), line, to_y))
        ln = _intervals(lanes, line, to_y)
        it = _intervals(inter, line, to_y)
        sL, sR = _around(sc, 0.0)[::-1]                    # left (max y), right (min y)
        xL, xR = _around(ext, 0.0)[::-1]
        y0 = hum_y[k] if np.isfinite(hum_y[k]) else 0.0
        lane = [v for v in ln if v[0] - 1e-6 <= y0 <= v[1] + 1e-6]
        lane = min(lane, key=lambda v: v[1] - v[0]) if lane else (np.nan, np.nan)
        lane0 = [v for v in ln if v[0] - 1e-6 <= 0 <= v[1] + 1e-6]
        lane0 = min(lane0, key=lambda v: v[1] - v[0]) if lane0 else (np.nan, np.nan)
        lanes_m = _merge(ln)
        lL, lR = _around(lanes_m, 0.0)[::-1]               # outer lane boundaries of the contiguous lane set around 0
        wl = [a for a, b in walk if np.isfinite(sL) and a >= sL - 0.5]      # nearest walkway start beyond the left scorer edge
        wr = [b for a, b in walk if np.isfinite(sR) and b <= sR + 0.5]
        rows.append(dict(token=t, stage=e["stage"], map=e["map"], x=x, cmd=int(np.argmax(e["cmd"][-1])),
                         inter=any(a < 15 and b > -15 for a, b in it), hum_y=hum_y[k], plan_y=plan_y[k],
                         ll0=m_ll[k, 0], ll1=m_ll[k, 1], ll2=m_ll[k, 2], ll3=m_ll[k, 3],
                         p0=llp[0], p1=llp[1], p2=llp[2], p3=llp[3], z1=m_llz[k, 1], z2=m_llz[k, 2],
                         reL=m_re[k, 0], reR=m_re[k, 1], sdL=m_res[k, 0], sdR=m_res[k, 1],
                         scL=sL, scR=sR, extL=xL, extR=xR, laneL=lL, laneR=lR,
                         egoL=lane[1], egoR=lane[0], ego0L=lane0[1], ego0R=lane0[0],
                         gda_any=bool(gda_iv), walkL=min(wl) if wl else np.nan, walkR=max(wr) if wr else np.nan,
                         n_lanes=len(lanes_m and [v for v in ln if np.isfinite(lL) and v[0] >= lR - 1e-3 and v[1] <= lL + 1e-3])))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=list(BOARDS), required=True)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    _init(a.board)
    toks = list(_G["names"])
    if a.limit:
        toks = toks[:: max(1, len(toks) // a.limit)][:a.limit]
    with mp.get_context("fork").Pool(a.procs) as pool:
        rows = [r for rs in pool.imap(work, toks, chunksize=16) for r in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_pickle(OUT / f"sections_{a.board}{'_dbg' if a.limit else ''}.pkl")
    print(len(toks), len(df))
