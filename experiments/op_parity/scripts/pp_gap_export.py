"""op_parity gap page, case export: pick representative navtest tokens where WA-JEPA passes a gap sub-metric and P2 fails it, and dump
their scene (map polygons, logged agents per 0.1 s from the v2 metric cache), the plans (P2 seeds, P0, WA-JEPA), the logged future and the
sub-scores to cases.json. No model runs. Needs the navsim2 env (nuplan + navsim):

  $DATA_DIR/envs/navsim2/bin/python experiments/op_parity/scripts/pp_gap_export.py [--n-metrics 4] [--per 2]

Selection rule (printed into cases.json, shown on the page): for each of the top-N sub-metrics by P2's Shapley loss on navtest (gap_tables.json),
candidates are tokens where (a) both P2 seeds score that sub-metric below WA-JEPA's by more than 0.25 for the continuous EP, or fail it (< 1) for
the others, while WA-JEPA passes (>= 1; EP: not applicable), (b) the term's Shapley share of the token's seed-mean gap is >= 0.25 (EP: 0.08, its gap is spread thinly), (c) WA-JEPA's
token score is >= 0.8, (d) ego speed > 2 m/s, (e) the metric cache loads. Candidates are sorted by that Shapley share and the tokens at the 40th
and 60th percentile positions are taken (next candidate when the log is already used): typical, not the worst.
"""
import argparse
import glob
import json
import lzma
import os
import pickle
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(D / "datasets/navsim"))
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L  # noqa: E402

TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
GAP = D / "runs/op_parity/gap"
PRED = D / "runs/op_lb/lb_navtest/preds"
WA_CSV_DIR = D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl"


def load_plans():
    P = {}
    for k, f in (("P2s0", "warp-cinque_PPP2-F-s0__base.npz"), ("P2s1", "warp-cinque_PPP2-F-s1__base.npz"), ("P0", "warp-cinque_PPP0__base.npz")):
        z = np.load(PRED / f)
        P[k] = dict(zip(z["tokens"].tolist(), z["poses"]))
    P["WA"] = pickle.load(open(WA_CSV_DIR, "rb"))["trajectories"]
    return P


def select(n_metrics, per, mcache, speed):
    tb = json.loads((GAP / "gap_tables.json").read_text())["navtest"]["arms"]["P2"]
    order = sorted(TERMS, key=lambda t: -tb[t]["loss"])[:n_metrics]
    z = np.load(GAP / "gap_navtest_shap.npz")
    toks, log, phi, Xw, X2 = z["tokens"], z["log"], z["shap_P2"], z["Xw"], z["X2"]
    from math import isfinite  # noqa: F401
    pick, used = [], set()
    for t in order:
        j = TERMS.index(t)
        fail = (X2[:, :, j] < Xw[None, :, j] - 0.25).all(0) if t == "EP" else ((X2[:, :, j] < 1 - 1e-9).all(0) & (Xw[:, j] >= 1 - 1e-9))
        wa_score = np.array([_score(x) for x in Xw])
        ok = fail & (phi[:, j] >= (0.08 if t == "EP" else 0.25)) & (wa_score >= 0.8) & np.array([speed.get(x, 0) > 2 for x in toks]) & np.array([x in mcache for x in toks])
        cand = np.flatnonzero(ok)
        cand = cand[np.argsort(-phi[cand, j], kind="stable")]
        n = len(cand)
        got = 0
        for q in (0.4, 0.6, 0.5, 0.3, 0.7)[:per + 3]:
            for off in range(0, n):
                i = cand[min(n - 1, int(q * n) + off)] if int(q * n) + off < n else None
                if i is None or log[i] in used:
                    continue
                used.add(log[i])
                pick.append((t, toks[i], int(i), n, float(phi[i, j])))
                got += 1
                break
            if got >= per:
                break
        print(t, "candidates", n, "picked", [p[1] for p in pick if p[0] == t], flush=True)
    return order, pick


def _score(x):
    ec = x[8]
    return float(np.prod(x[:4]) * (5 * x[4] + 5 * x[5] + 2 * x[6] + 2 * x[7] + 2 * np.nan_to_num(ec)) / (14 + 2 * np.isfinite(ec)))


def export_case(t, mc, e, plans, fut, i_row, z):
    ego = mc.ego_state
    origin = np.array(ego.rear_axle.serialize())
    c, s = np.cos(origin[2]), np.sin(origin[2])
    R = np.array([[c, -s], [s, c]])
    local = lambda p: (np.asarray(p)[..., :2] - origin[:2]) @ R  # noqa: E731
    amap = mc.drivable_area_map
    area_ids = amap.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    lane_ids = amap.get_indices_of_map_type([L.LANE, L.LANE_CONNECTOR])
    polys = []
    for k in area_ids + lane_ids:
        g = amap._geometries[k]
        if g.distance(ego.car_footprint.oriented_box.geometry) > 80:
            continue
        for p in (list(g.geoms) if hasattr(g, "geoms") else [g]):
            polys.append({"kind": "area" if k in area_ids else "route" if amap.tokens[k] in mc.route_lane_ids else "lane",
                          "exterior": local(p.exterior.coords).round(2).tolist(), "holes": [local(h.coords).round(2).tolist() for h in p.interiors]})
    obs, agents = mc.observation, {}
    n_t = 41
    for k in range(n_t):
        om = obs[k]
        for tok, g in zip(om.tokens, om._geometries):
            kind = "red_light" if obs.red_light_token in tok else obs.unique_objects[tok].tracked_object_type.name.lower()
            xy = local(g.exterior.coords if hasattr(g, "exterior") else g.coords)
            if np.abs(xy).max() > 85:
                continue
            agents.setdefault(tok, {"kind": kind, "polys": [None] * n_t})["polys"][k] = xy.round(2).tolist()
    vp = ego.car_footprint.vehicle_parameters
    return {"token": t, "log": e["log_name"], "map": e["map"], "speed": float(np.linalg.norm(e["vel"][-1])),
            "cmd": ["left", "straight", "right", "unknown"][int(np.argmax(e["cmd"][-1]))],
            "dims": {"front": float(vp.front_length), "rear": float(vp.rear_length), "width": float(vp.width)},
            "polygons": polys, "agents": agents, "plans": {k: np.asarray(v[t], float).round(3).tolist() for k, v in plans.items()},
            "gt": np.asarray(fut[t], float).round(3).tolist(),
            "sub": {"P2s0": z["X2"][0, i_row].tolist(), "P2s1": z["X2"][1, i_row].tolist(), "P0": z["X0"][i_row].tolist(), "WA": z["Xw"][i_row].tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-metrics", type=int, default=4)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    idx = {e["token"]: e for e in pickle.load(open(D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
    speed = {t: float(np.linalg.norm(e["vel"][-1])) for t, e in idx.items()}
    mcache = {Path(p).parent.name: p for p in glob.glob(str(D / "runs/navsim/metric_cache/v2_navtest/*/*/*/metric_cache.pkl"))}
    order, pick = select(a.n_metrics, a.per, mcache, speed)
    plans = load_plans()
    fz = np.load(D / "runs/navsim_zs/index/navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    z = np.load(GAP / "gap_navtest_shap.npz")
    cases = []
    for t, tok, i, n_cand, share in pick:
        with lzma.open(mcache[tok], "rb") as f:
            mc = pickle.load(f)
        c = export_case(tok, mc, idx[tok], plans, fut, i, z)
        c.update(metric=t, n_candidates=n_cand, shapley_share=share, id=f"{t.lower()}-{tok[:6]}")
        cases.append(c)
        print("exported", c["id"], flush=True)
    sel = ("Per sub-metric (top by P2 Shapley loss): candidates are tokens where WA-JEPA passes the sub-metric and both P2 seeds fail it (EP: both seeds "
           "more than 0.25 below WA-JEPA), the term's Shapley share of the seed-mean token gap is >= 0.25 (EP: 0.08), WA-JEPA's token score >= 0.8, ego speed > 2 m/s. "
           "Sorted by Shapley share; the tokens at the 40th and 60th percentile positions are shown (distinct logs): typical, not the worst.")
    (GAP / "cases.json").write_text(json.dumps({"selection": sel, "order": order, "cases": cases}))
    print("DONE", len(cases))


if __name__ == "__main__":
    main()
