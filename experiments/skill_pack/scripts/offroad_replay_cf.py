"""navhard off-road diagnosis, steps A + B (plan: experiments/skill_pack/plans/2026-10-03-navhard-offroad-diagnosis-plan.md).

CPU only, navsim2 env. For every navhard token (5 912, both stages) and both models (native, N4):
  A. official LQR + bicycle replay of the cached pose file -> features (first departure, outside distance, lateral offset
     to the PDM reference, start offset to the route centreline, junction flag ...) -> features_<model>.pkl
  B. counterfactual stage-2 plans (lag / scale / pdm-1s), each scored with the devkit's `pdm_score` and aggregated with
     the devkit's two-stage code -> counterfactuals.json (+ per-token CSV).
The baseline is checked against the cached official CSVs (per-token score and DAC, combined EPDMS) before anything else.

  DATA_DIR=... PYTHONPATH=$DATA_DIR/third_party/navsim:$DATA_DIR/third_party/nuplan-devkit \
    $DATA_DIR/envs/navsim2/bin/python experiments/skill_pack/scripts/offroad_replay_cf.py --procs 48 [--limit 200]
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offroad_lib as L  # noqa: E402

VARIANTS = ["lag0.3", "lag0.5", "lag1.0", "scale1.25", "scale1.5", "pdm1s"]
MODELS = ["native", "n4"]


# ---------------------------------------------------------------- counterfactual plan edits (dense 0.1 s, ego frame)

def v_lag(d, ref, dt):
    ts = np.minimum(L.T_DENSE + dt, 4.0)
    o = d.copy()
    o[:, 1] = np.interp(ts, L.T_DENSE, d[:, 1])
    o[:, 2] = np.interp(ts, L.T_DENSE, d[:, 2])
    return o


def v_scale(d, ref, k):
    yr = ref[-1, 1]
    s = np.sign(yr) if abs(yr) > 0.5 else 0.0
    o = d.copy()
    if s == 0:
        return o
    y = d[:, 1]
    y2 = np.where(y * s > 0, y * k, y)
    o[:, 1] = y2
    dx, dy, dy2 = np.gradient(d[:, 0], 0.1), np.gradient(y, 0.1), np.gradient(y2, 0.1)
    o[:, 2] = d[:, 2] + np.where(np.hypot(dx, dy) > 1.0, np.arctan2(dy2, dx) - np.arctan2(dy, dx), 0.0)
    return o


def v_pdm1s(d, ref):
    o = d.copy()
    r = ref.copy()
    r[:, 2] = np.unwrap(r[:, 2])
    o[:11] = r[:11]
    dpsi = r[10, 2] - d[10, 2]
    c, s = np.cos(dpsi), np.sin(dpsi)
    rel = d[11:, :2] - d[10, :2]
    o[11:, 0] = r[10, 0] + c * rel[:, 0] - s * rel[:, 1]
    o[11:, 1] = r[10, 1] + s * rel[:, 0] + c * rel[:, 1]
    o[11:, 2] = d[11:, 2] + dpsi
    return o


def make_variant(name, p8, ref):
    d = L.dense_from_poses(p8)
    if name.startswith("lag"):
        o = v_lag(d, ref, float(name[3:]))
    elif name.startswith("scale"):
        o = v_scale(d, ref, float(name[5:]))
    elif name == "pdm1s":
        o = v_pdm1s(d, ref)
    else:
        raise ValueError(name)
    return L.poses_from_dense(o)


# ---------------------------------------------------------------- per-token features of a simulated trajectory

_W = {}


def geoms(mc):
    """Per-token cached geometry helpers (global frame)."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    from shapely.ops import unary_union
    am = mc.drivable_area_map
    area = am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA])
    inter = am.get_indices_of_map_type([S.INTERSECTION])
    conn = am.get_indices_of_map_type([S.LANE_CONNECTOR])
    u = lambda ids: unary_union([am._geometries[k] for k in ids]) if ids else None  # noqa: E731
    return dict(am=am, area=area, area_u=u(area), inter_u=u(inter), conn_u=u(conn))


def corners_of(mc, states):
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    return state_array_to_coords_array(states[None], mc.ego_state.car_footprint.vehicle_parameters)[0]  # (41, 5, 2)


def features(mc, g, states, p8, ref):
    """Departure + lateral features of one simulated trajectory (states (41, 11) global)."""
    import shapely
    cor = corners_of(mc, states)
    inside = g["am"].points_in_polygons(cor[None, :, :-1, :])[g["area"]].any(axis=0)[0]   # (41, 4) corner in some area polygon
    out_state = ~inside.all(axis=1)
    f = dict(dac=int(not out_state.any()), start_outside=bool(out_state[0]), n_out=int(out_state.sum()))
    ego = L.to_ego(mc, states[:, :3])
    f["sim_ego"] = ego
    f["raw_dense"] = L.dense_from_poses(p8)
    if out_state.any():
        dist = np.zeros(inside.shape)
        idx = np.argwhere(~inside)
        pts = shapely.points(cor[idx[:, 0], idx[:, 1]])
        dist[idx[:, 0], idx[:, 1]] = shapely.distance(g["area_u"], pts)
        worst_state = int(dist.max(axis=1).argmax())
        first = int(np.flatnonzero(out_state)[0])
        # first departure after t = 0 when the start is already outside: first state outside after being inside, else None
        f.update(first=first, d_first=float(dist[first].max()), d_worst=float(dist.max()), worst_state=worst_state)
        c = int(dist[first].argmax())
        h = states[first, 2]
        rel = cor[first, c] - states[first, :2]
        f["side_first"] = float(-np.sin(h) * rel[0] + np.cos(h) * rel[1])      # + = left corner of the car
        cen = shapely.points(cor[first, 4])
        ptc = shapely.points(cor[first, c])
        near = lambda u: (u is not None) and (shapely.distance(u, cen) <= 3 or shapely.distance(u, ptc) <= 3)  # noqa: E731
        f["junction"] = bool(near(g["inter_u"]) or (g["conn_u"] is not None and (shapely.contains(g["conn_u"], cen) or shapely.distance(g["conn_u"], ptc) < 0.01)))
        f["dep_xy_ego"] = ego[first, :2]
    else:
        f.update(first=None, d_first=0.0, d_worst=0.0, worst_state=None, side_first=0.0, junction=False, dep_xy_ego=None)
    # lateral offset to the PDM reference at 0 / 0.5 / 1 / 2 / 4 s, normal of the reference at that time (the stage-2 reference
    # starts at the undisplaced log pose, so t = 0 is generally not 0)
    lat = {}
    for t in (0.0, 0.5, 1.0, 2.0, 4.0):
        i = int(round(t / 0.1))
        n = np.array([-np.sin(ref[i, 2]), np.cos(ref[i, 2])])
        lat[t] = float((ego[i, :2] - ref[i, :2]) @ n)
    f["lat_ref"] = lat
    return f


def start_features(mc):
    """Signed lateral offset (left +) and heading error of the t0 pose w.r.t. the route centreline."""
    from shapely.geometry import Point
    o = mc.ego_state.rear_axle
    s = float(mc.centerline.project([Point(o.x, o.y)])[0])
    c = mc.centerline.interpolate([s], as_array=True)[0]
    n = np.array([-np.sin(c[2]), np.cos(c[2])])
    off = float((np.array([o.x, o.y]) - c[:2]) @ n)
    he = float((o.heading - c[2] + np.pi) % (2 * np.pi) - np.pi)
    return dict(center_off=off, center_head_err=he, center_dist_to_path=float(np.hypot(*(np.array([o.x, o.y]) - c[:2]))))


# ---------------------------------------------------------------- worker

def _init(args):
    from navsim.evaluate.pdm_score import pdm_score  # noqa: F401
    sim, scorer, policy, mapping, samp = L.setup_scoring()
    _W.update(sim=sim, scorer=scorer, policy=policy, samp=samp, cp=L.cache_paths(),
              P={"native": L.poses_by_token(L.NATIVE_POSES), "n4": L.poses_by_token(L.N4_POSES)},
              stage={e["token"]: e["stage"] for e in L.index()}, variants=args.variants)


def _score(mc, p8, token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    W = _W
    row, st = pdm_score(metric_cache=mc, model_trajectory=Trajectory(np.asarray(p8, np.float64)), future_sampling=W["samp"],
                        simulator=W["sim"], scorer=W["scorer"], traffic_agents_policy=W["policy"])
    row["valid"] = True
    row["log_name"] = mc.log_name
    row["frame_type"] = mc.scene_type
    row["start_time"] = mc.timepoint.time_s
    e = relative_to_absolute_poses(mc.ego_state.rear_axle, [StateSE2(*p8[-1])])[0]
    row["endpoint_x"], row["endpoint_y"] = e.x, e.y
    row["start_point_x"], row["start_point_y"] = mc.ego_state.rear_axle.x, mc.ego_state.rear_axle.y
    row["ego_simulated_states"] = [st]
    row["token"] = token
    return row, st


def work(token):
    W = _W
    mc = L.load_cache(W["cp"][token])
    g = geoms(mc)
    ref = L.pdm_ref_ego(mc, W["samp"])
    out = {"token": token, "stage": W["stage"][token], "start": start_features(mc), "ref": ref, "rows": {}, "feat": {}}
    for m in MODELS:
        p8 = W["P"][m][token]
        row, st = _score(mc, p8, token)
        out["rows"][(m, "base")] = row
        out["feat"][m] = features(mc, g, st, p8, ref)
        if out["stage"] == "two":
            for v in W["variants"]:
                row, _ = _score(mc, make_variant(v, p8, ref), token)
                out["rows"][(m, v)] = row
    # the PDM reference itself as a plan (features only, no scoring)
    p_ref = L.poses_from_dense(ref)
    out["feat"]["pdm"] = features(mc, g, L.simulate(W["sim"], mc, p_ref), p_ref, ref)
    return out


# ---------------------------------------------------------------- aggregation with the devkit's two-stage code

def aggregate(rows, mapping, samp):
    """rows: list of single-row DataFrames (one per token). Returns (combined, stage1, stage2 Series), per-token df."""
    from dataclasses import fields
    from navsim.common.dataclasses import PDMResults
    from navsim.planning.script.run_pdm_score import (calculate_individual_mapping_scores, compute_final_scores,
                                                        create_scene_aggregators)
    df = pd.concat(rows)
    df = create_scene_aggregators(mapping, df, samp)
    df = compute_final_scores(df)
    cols = [c for c in df.columns if ((any(s.name in c for s in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
                                      and c != "pdm_score")]
    comb, s1, s2 = calculate_individual_mapping_scores(df[cols + ["token", "weight"]], mapping)
    return comb, s1, s2, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N tokens of every 20th stride (aggregation skipped)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--out", default=str(L.OUT))
    ap.add_argument("--reuse", action="store_true", help="skip scoring, load raw.pkl of --out (aggregation only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = L.index()
    tokens = [e["token"] for e in idx]
    if a.limit:
        tokens = tokens[::max(1, len(tokens) // a.limit)][:a.limit]
    t0 = time.time()
    res = {}
    if a.reuse:
        res = pickle.load(open(out / "raw.pkl", "rb"))
        for t, r in res.items():                       # raw.pkl of the first run has no token column in the score rows
            for row in r["rows"].values():
                row["token"] = t
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(a.procs, initializer=_init, initargs=(a,)) as pool:
            for i, r in enumerate(pool.imap_unordered(work, tokens, chunksize=4)):
                res[r["token"]] = r
                if i % 500 == 0:
                    print(f"{i}/{len(tokens)} {time.time() - t0:.0f}s", flush=True)
        print(f"scored {len(res)} tokens in {time.time() - t0:.0f}s", flush=True)
        if not a.limit:
            pickle.dump(res, open(out / "raw.pkl", "wb"), protocol=4)
    pickle.dump({t: {"feat": r["feat"], "start": r["start"], "ref": r["ref"], "stage": r["stage"]} for t, r in res.items()},
                open(out / ("features_debug.pkl" if a.limit else "features.pkl"), "wb"))
    if a.limit:
        return
    _, _, _, mapping, samp = L.setup_scoring()
    summary = {}
    ref_csv = {m: L.eval_csv(c)[0].set_index("token") for m, c in (("native", L.NATIVE_CSV), ("n4", L.N4_CSV))}
    for m in MODELS:
        base_rows = [res[t]["rows"][(m, "base")] for t in tokens]
        comb, s1, s2, df = aggregate(base_rows, mapping, samp)
        off = ref_csv[m]
        d = df.set_index("token")
        tk = [t for t in tokens]
        err_score = float(np.abs(d.loc[tk, "score"].to_numpy() - np.where(
            pd.isna(off.loc[tk, "score"]), np.nan, off.loc[tk, "score"].to_numpy())).max())
        offc = off.loc["extended_pdm_score_combined"]
        summary[f"{m}/base"] = dict(combined=float(comb["score"]), s1=float(s1["score"]), s2=float(s2["score"]),
                                    official_combined=float(offc["score"]), max_token_score_err=err_score,
                                    dac_s2_weighted=float(s2["drivable_area_compliance"]))
        print(m, "baseline replay vs official CSV:", summary[f"{m}/base"], flush=True)
        assert abs(comb["score"] - offc["score"]) < 1e-4 and err_score < 1e-6, f"baseline of {m} does not reproduce the official score"
        failing = {t for t in tokens if res[t]["stage"] == "two" and res[t]["rows"][(m, "base")]["drivable_area_compliance"].iloc[0] == 0}
        summary[f"{m}/n_stage2_dac_fail"] = len(failing)
        d.reset_index().drop(columns=["ego_simulated_states"], errors="ignore").to_csv(out / f"tokens_{m}_base.csv", index=False)
        for v in a.variants:
            for gate in ("all", "failures"):
                rows = []
                for t in tokens:
                    r = res[t]["rows"]
                    use = (m, v) in r and (gate == "all" or t in failing)
                    rows.append(r[(m, v)] if use else r[(m, "base")])
                comb, s1, s2, df = aggregate(rows, mapping, samp)
                d = df.set_index("token")
                s2tok = [t for t in tokens if res[t]["stage"] == "two"]
                dac_u = float(d.loc[s2tok, "drivable_area_compliance"].mean())
                base = pd.concat([res[t]["rows"][(m, "base")] for t in s2tok]).set_index("token")["drivable_area_compliance"]
                new = d.loc[s2tok, "drivable_area_compliance"]
                key = f"{m}/{v}/{gate}"
                summary[key] = dict(combined=float(comb["score"]), s1=float(s1["score"]), s2=float(s2["score"]),
                                    dac_s2_weighted=float(s2["drivable_area_compliance"]), dac_s2_uniform=dac_u,
                                    fixed=int(((base == 0) & (new == 1)).sum()), broken=int(((base == 1) & (new == 0)).sum()),
                                    n_fail_after=int((new == 0).sum()))
                print(key, summary[key], flush=True)
                d.reset_index().to_csv(out / f"tokens_{m}_{v}_{gate}.csv", index=False)
    (out / "counterfactuals.json").write_text(json.dumps(summary, indent=1))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
