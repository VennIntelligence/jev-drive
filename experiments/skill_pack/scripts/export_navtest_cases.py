"""Export existing NAVSIM v1 navtest inputs, cached plans and official PDM scoring rollouts for the visual review. No model inference.

Reads (all already on the box): the per-token PDMS CSVs of native Cinque (GIMM input, 84.18) and N3 (91.59), their cached
8-pose plans, the v1 navtest metric cache, the slim navtest index and the CAM_F0 history frames. Re-scores only the selected
tokens with the devkit's own simulator + scorer (CPU, LQR + bicycle model) and asserts every sub-score equals the CSV row.
Writes cases.json, decomp.json and resized history frames to runs/openloop_visual_review_20261002/navtest/.
Run in the navsim1 env:  CUDA_VISIBLE_DEVICES="" envs/navsim1/bin/python experiments/skill_pack/scripts/export_navtest_cases.py
Mirrors tmp/openloop-review/export_cached_cases.py (navhard stage 2)."""
import glob, json, lzma, os, pickle
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(D / "datasets/navsim"))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(D / "runs/navsim/eval"))
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")  # the box's OpenBLAS kernel bug (decision 73)
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "4")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from PIL import Image
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from shapely import creation
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex, EgoAreaIndex, MultiMetricIndex, StateIndex, WeightedMetricIndex)

O = D / "runs/openloop_visual_review_20261002/navtest"
O.mkdir(parents=True, exist_ok=True)
EVAL = D / "runs/navsim/eval"
RUNS = {"native": ("v1_navtest_opi_lb_navtest_gimm-cinque__base", D / "runs/op_lb/lb_navtest/preds/gimm-cinque__base.npz"),
        "n3": ("v1_navtest_sp_n3_navtest", D / "runs/skill_pack/raise/n3/navtest_n3.npz")}
TERMS = dict(no_at_fault_collisions="NC", drivable_area_compliance="DAC", ego_progress="EP",
             time_to_collision_within_bound="TTC", comfort="C", driving_direction_compliance="DDC", score="PDMS")
CMDS = ["left", "straight", "right", "unknown"]


def load_csv(name):
    p = sorted(glob.glob(str(EVAL / name / "*/*.csv")))[-1]
    t = pd.read_csv(p)
    avg = t[t.token == "average"].iloc[0]
    t = t[t.token.str.len() == 16].set_index("token").rename(columns=TERMS)
    assert t.valid.all()
    return t[list(TERMS.values())].astype(float), {v: float(avg[k]) for k, v in TERMS.items()}, p


def decomp(d, avg):
    """Additive split of the per-token shortfall 1 - PDMS; PDMS = NC * DAC * (5 EP + 5 TTC + 2 C) / 12 (DDC weight is 0 in v1)."""
    M = d.NC * d.DAC
    rec = M * (5 * d.EP + 5 * d.TTC + 2 * d.C) / 12
    gate = 1 - M
    den = ((1 - d.NC) + (1 - d.DAC)).replace(0, 1)
    out = {"n": len(d), "pdms_recomputed": 100 * rec.mean(), "pdms_token_mean": 100 * d.PDMS.mean(), "pdms_stored_average_row": 100 * avg["PDMS"],
           "max_abs_token_err": float((rec - d.PDMS).abs().max()), "lost": 100 * (1 - rec.mean()),
           "loss_NC": 100 * (gate * (1 - d.NC) / den).mean(), "loss_DAC": 100 * (gate * (1 - d.DAC) / den).mean(),
           "loss_EP": 100 * (M * 5 * (1 - d.EP) / 12).mean(), "loss_TTC": 100 * (M * 5 * (1 - d.TTC) / 12).mean(),
           "loss_C": 100 * (M * 2 * (1 - d.C) / 12).mean(),
           "zero_share": 100 * (d.PDMS == 0).mean(), "ep_mean_gates_passed": 100 * d.EP[M == 1].mean(), "n_gates_passed": int((M == 1).sum())}
    for k in ("NC", "DAC", "EP", "TTC", "C", "DDC"):
        out["mean_" + k] = 100 * d[k].mean()
        out["fail_share_" + k] = 100 * (d[k] < 1).mean()
        out["if_perfect_" + k] = 100 * (d.assign(**{k: 1.0}).pipe(lambda e: e.NC * e.DAC * (5 * e.EP + 5 * e.TTC + 2 * e.C) / 12).mean() - rec.mean())
    return out


tabs, avgs, paths = {}, {}, {}
for n, (run, _) in RUNS.items():
    tabs[n], avgs[n], paths[n] = load_csv(run)
A, B = tabs["native"], tabs["n3"].loc[tabs["native"].index]
dec = {"native": decomp(A, avgs["native"]), "n3": decomp(B, avgs["n3"])}
trans = {k: {"native_fail_n3_pass": int(((A[k] < 1) & (B[k] == 1)).sum()), "native_pass_n3_fail": int(((A[k] == 1) & (B[k] < 1)).sum()),
             "both_fail": int(((A[k] < 1) & (B[k] < 1)).sum()), "both_pass": int(((A[k] == 1) & (B[k] == 1)).sum())}
         for k in ("NC", "DAC", "TTC", "C", "DDC")}
trans["PDMS"] = {"n3_higher": int((B.PDMS > A.PDMS + 1e-9).sum()), "n3_lower": int((B.PDMS < A.PDMS - 1e-9).sum()),
                 "equal": int(((B.PDMS - A.PDMS).abs() <= 1e-9).sum()),
                 "native_zero_n3_ge80": int(((A.PDMS == 0) & (B.PDMS >= .8)).sum()), "native_ge80_n3_zero": int(((A.PDMS >= .8) & (B.PDMS == 0)).sum())}
print(json.dumps(dec, indent=1), json.dumps(trans), flush=True)

idx = {e["token"]: e for e in pickle.load(open(D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
plans = {}
for n, (_, p) in RUNS.items():
    z = np.load(p)
    plans[n] = dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float64)))
mcache = {Path(p).parent.name: p for p in glob.glob(str(D / "runs/navsim/metric_cache/v1_navtest/*/*/*/metric_cache.pkl"))}
with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base=None):
    cfg = compose("default_run_pdm_score", overrides=["experiment_name=navtest_review_export"])
sim, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
sampling = sim.proposal_sampling
speed = lambda t: float(np.linalg.norm(idx[t]["vel"][-1]))
gates = lambda d: (d.NC == 1) & (d.DAC == 1)
fine = lambda d: gates(d) & (d.TTC == 1) & (d.DDC == 1)
moving = pd.Series({t: speed(t) > 3 for t in A.index})
# One group per PDMS sub-score, in both directions (N3 worse / N3 better), plus a control. First eligible token in sorted order.
GROUPS = [
    ("nc_regression", "NC", (A.PDMS >= .8) & fine(A) & (B.NC == 0) & (B.DAC == 1) & moving),
    ("nc_recovery", "NC", (A.NC == 0) & (A.DAC == 1) & (B.PDMS >= .8) & fine(B) & moving),
    ("dac_regression", "DAC", (A.PDMS >= .8) & fine(A) & (B.DAC == 0) & (B.NC == 1) & moving),
    ("dac_recovery", "DAC", (A.DAC == 0) & (A.NC == 1) & (B.PDMS >= .8) & fine(B) & moving),
    ("ttc_regression", "TTC", (A.PDMS >= .8) & fine(A) & gates(B) & (B.TTC == 0) & moving),
    ("ep_recovery", "EP", fine(A) & (A.EP < .5) & fine(B) & (B.EP - A.EP > .4) & moving),
    ("ddc_regression", "DDC", fine(A) & gates(B) & (B.TTC == 1) & (B.DDC == 0) & moving),
    ("control", "none", (A.PDMS > .95) & (B.PDMS > .95) & fine(A) & fine(B) & moving),
]


def score_one(mc, poses):
    """Official pdm_score body for [PDM reference, plan]; returns states and the scorer's intermediate arrays for the plan."""
    ego = mc.ego_state
    tr = np.stack([get_trajectory_as_array(mc.trajectory, sampling, ego.time_point),
                   get_trajectory_as_array(transform_trajectory(Trajectory(poses), ego), sampling, ego.time_point)])
    st = sim.simulate_proposals(tr, ego)
    sc = scorer.score_proposals(st, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
    W, Mm = scorer._weighted_metrics, scorer._multi_metrics
    res = {"NC": Mm[MultiMetricIndex.NO_COLLISION, 1], "DAC": Mm[MultiMetricIndex.DRIVABLE_AREA, 1], "EP": W[WeightedMetricIndex.PROGRESS, 1],
           "TTC": W[WeightedMetricIndex.TTC, 1], "C": W[WeightedMetricIndex.COMFORTABLE, 1], "DDC": W[WeightedMetricIndex.DRIVING_DIRECTION, 1], "PDMS": sc[1]}
    obs, red = mc.observation, mc.observation.red_light_token
    tc, tt = scorer._collision_time_idcs[1], scorer._ttc_time_idcs[1]
    hit = set()
    if np.isfinite(tc):  # objects overlapping the ego footprint at the first at-fault collision step
        k = int(tc)
        hit |= {obs[k].tokens[i] for i in obs[k].query(scorer._ego_polygons[1, k], predicate="intersects")}
    near = set()
    if np.isfinite(tt):  # objects met by the scorer's constant-velocity projections (0, 0.3, 0.6, 0.9 s) at the first TTC step
        k = int(tt)
        co = scorer._ego_coords[1, k].copy()
        co[BBCoordsIndex.CENTER] = co[BBCoordsIndex.FRONT_LEFT]
        v = np.hypot(st[1, k, StateIndex.VELOCITY_X], st[1, k, StateIndex.VELOCITY_Y])
        d = np.array([np.cos(st[1, k, StateIndex.HEADING]), np.sin(st[1, k, StateIndex.HEADING])]) * v
        for f in (0, 3, 6, 9):
            near |= {obs[k + f].tokens[i] for i in obs[k + f].query(creation.polygons(co + d * f * .1), predicate="intersects")}
    clean = lambda s: sorted(t for t in s if red not in t and t not in obs.collided_track_ids)
    det = {"collision_s": None if not np.isfinite(tc) else round(tc * .1, 1), "ttc_s": None if not np.isfinite(tt) else round(tt * .1, 1),
           "hit": clean(hit), "ttc_objects": clean(near), "progress_raw_m": float(scorer._progress_raw[1]), "pdm_progress_raw_m": float(scorer._progress_raw[0]),
           "oncoming": scorer._ego_areas[1, :, EgoAreaIndex.ONCOMING_TRAFFIC].tolist(),
           "offroad": scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA].tolist(),
           "speed": np.hypot(st[1, :, StateIndex.VELOCITY_X], st[1, :, StateIndex.VELOCITY_Y]).round(3).tolist()}
    return st, {k: float(v) for k, v in res.items()}, det


seen, cases, counts = set(), [], {}
for group, term, mask in GROUPS:
    eligible = sorted(mask.index[mask.values])
    counts[group] = len(eligible)
    for t in eligible:
        e = idx[t]
        if e["log_name"] in seen or t not in mcache:
            continue
        with lzma.open(mcache[t], "rb") as f:
            mc = pickle.load(f)
        ego = mc.ego_state
        origin = np.array(ego.rear_axle.serialize())
        c, s = np.cos(origin[2]), np.sin(origin[2])
        R = np.array([[c, -s], [s, c]])
        local = lambda p: (np.asarray(p)[..., :2] - origin[:2]) @ R
        states, scores, details, ok = [], {}, {}, True
        for n, tab in (("native", A), ("n3", B)):
            st, res, det = score_one(mc, plans[n][t])
            err = max(abs(res[k] - tab.loc[t, k]) for k in res)
            if err > 1e-9:  # every case must reproduce its CSV row
                print("SCORE MISMATCH", group, t, n, res, tab.loc[t].to_dict(), flush=True)
                ok = False
            states += [st[0], st[1]] if n == "native" else [st[1]]
            scores[n], details[n] = res, det
        if not ok:
            continue
        states = np.stack(states)  # PDM reference, native, N3
        corners = state_array_to_coords_array(states, ego.car_footprint.vehicle_parameters)
        amap = mc.drivable_area_map
        area_ids = amap.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
        lane_ids = amap.get_indices_of_map_type([L.LANE, L.LANE_CONNECTOR])
        inside = amap.points_in_polygons(corners[..., :-1, :])[area_ids].any(axis=0).all(axis=-1)
        if not inside[:, 0].all():  # show a departure, not an initial-state penalty
            continue
        if [int(inside[1].all()), int(inside[2].all())] != [int(scores["native"]["DAC"]), int(scores["n3"]["DAC"])]:
            print("DAC MISMATCH", t, flush=True)
            continue
        bad = "n3" if "regression" in group else "native"
        if term == "NC" and (details[bad]["collision_s"] < .5 or not details[bad]["hit"]):
            continue  # skip overlaps that exist at t=0
        if term == "TTC" and (details[bad]["ttc_s"] < .3 or not details[bad]["ttc_objects"]):
            continue
        polys = []
        for k in area_ids + lane_ids:
            g = amap._geometries[k]
            if g.distance(ego.car_footprint.oriented_box.geometry) > 70:
                continue
            for p in (list(g.geoms) if hasattr(g, "geoms") else [g]):
                polys.append({"kind": "area" if k in area_ids else "route" if amap.tokens[k] in mc.route_lane_ids else "lane",
                              "exterior": local(p.exterior.coords).round(3).tolist(), "holes": [local(h.coords).round(3).tolist() for h in p.interiors]})
        obs, agents = mc.observation, {}
        for k in range(sampling.num_poses + 1):  # logged (non-reactive) object boxes from the metric cache, one set per 0.1 s
            om = obs[k]
            for tok, g in zip(om.tokens, om._geometries):
                if obs.red_light_token in tok:
                    kind = "red_light"
                else:
                    kind = obs.unique_objects[tok].tracked_object_type.name.lower()
                xy = local(g.exterior.coords if hasattr(g, "exterior") else g.coords)
                if np.abs(xy).max() > 75:
                    continue
                agents.setdefault(tok, {"kind": kind, "pre_collided": tok in obs.collided_track_ids, "polys": [None] * (sampling.num_poses + 1)})["polys"][k] = xy.round(2).tolist()
        case_id = f"nt-{len(cases) + 1:02d}-{group}"
        imgs = []
        for k, cam in enumerate(e["cams"]):
            im = Image.open(cam["CAM_F0"]["path"]).convert("RGB")
            im.thumbnail((960, 540))
            fn = f"{case_id}-history-{k}.jpg"
            im.save(O / fn, quality=88)
            imgs.append(fn)
        cl = mc.centerline
        cases.append({"id": case_id, "group": group, "term": term, "token": t, "log": e["log_name"], "map": e["map"],
                      "command": CMDS[int(np.argmax(e["cmd"][-1]))], "speed": speed(t), "images": imgs, "polygons": polys, "agents": agents,
                      "scores": scores, "csv_scores": {"native": A.loc[t].to_dict(), "n3": B.loc[t].to_dict()}, "details": details,
                      "plans_ego": {n: plans[n][t].round(4).tolist() for n in RUNS},
                      "states_xy": local(states[..., :2]).round(4).tolist(), "corners_xy": local(corners).round(4).tolist(), "inside": inside.tolist(),
                      "first_departure_s": [None if a.all() else round(float(np.where(~a)[0][0]) * .1, 1) for a in inside],
                      "eligible_in_group": len(eligible), "source_metric_cache": mcache[t], "source_camera_paths": [x["CAM_F0"]["path"] for x in e["cams"]]})
        seen.add(e["log_name"])
        print(case_id, t, cases[-1]["command"], scores, {n: (d["collision_s"], d["ttc_s"]) for n, d in details.items()}, flush=True)
        break
(O / "decomp.json").write_text(json.dumps({"formula": "PDMS = NC * DAC * (5 EP + 5 TTC + 2 C) / 12; DDC stored, weight 0", "csv": paths, "runs": dec,
                                           "transitions": trans, "eligible_per_group": counts}, indent=1))
(O / "cases.json").write_text(json.dumps({"selection": "First eligible token in sorted token order per sub-score group; distinct logs; speed > 3 m/s; "
                                          "all footprints start inside the drivable area; collisions at t >= 0.5 s. Not representative sampling.",
                                          "csv": paths, "plans": {n: str(p) for n, (_, p) in RUNS.items()}, "sampling_dt": .1, "cases": cases}, ensure_ascii=False))
print("DONE", len(cases), counts, flush=True)
