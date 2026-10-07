"""op_parity four directions, NAVSIM part: sharp turns, wide turns (road-edge grazes) and collisions of P2H (P2 + drivable hinge, lambda 10)
on navtest and navhard two-stage, measured as buckets (size by replacement oracle + Shapley, mechanism from stored outputs). No model runs.

  replay  (envs/navsim2, CPU)  instrumented devkit re-score (the devkit's pdm_score with a scorer hook) of P2H s0 / s1, WA-JEPA and the
                               reference on every token where one of them fails a gate: sub-scores, first footprint departure of the raw plan
                               and of the LQR replay (time, corner, side, depth), first at-fault collision / TTC event (object type, nuPlan
                               collision type, relative pose and speed), LQR states. navhard also stores the PDM-Closed reference path of every
                               token (turn geometry) and the two-stage mapping.  -> $OUT/replay_<bench>.parquet, _states.npz
  analyze (.venv, CPU)         buckets, oracles (a) WA-JEPA / (b) reference with clip / (c) bucket Shapley, mechanism shares, bootstraps
                               -> experiments/op_parity/results/four_dirs/nav_*.csv, nav_summary.json
  cases   (envs/navsim2, CPU)  BEV scene dump of the typical cases picked by analyze -> $OUT/cases_<bench>.pkl
  figs    (.venv)              figures -> results/four_dirs/figs/nav_*.png

  $DATA_DIR/envs/navsim2/bin/python experiments/op_parity/scripts/fd_navsim.py replay --bench navtest --procs 32
  .venv/bin/python experiments/op_parity/scripts/fd_navsim.py analyze

Bucket rules (fixed before any score was read; written at the top of results/four_dirs/navsim.md):
  path geometry from the logged future (navtest) or the PDM-Closed reference path (navhard, both stages: stage 2 has no logged future),
  8 poses + origin. dpsi = heading change at 4 s. R_min = min over 1 s windows (3 consecutive poses, arc >= 2 m) of arc / |dheading|.
  turning = |dpsi| >= 8 deg and path length >= 3 m. sharp = turning and R_min < 15 m. wide = turning and R_min >= 15 m
  (sub-split curve 8-20 deg / turn >= 20 deg). Alternative heading rule: sharp' = |dpsi| > 45 deg; standard bins <5 / 5-20 / 20-45 / >45.
  D1 sharp-turn DAC failure, D2 wide-turn / curve DAC failure: a token in the geometry set whose arm DAC < 1 (per seed).
  DAC side: side (left / right corner of the ego box) of the first footprint corner outside the scorer's drivable polygons along the LQR replay,
  inside = same side as the turn direction (sign of dpsi), outside = opposite. raw_out = the raw plan's footprint (0.1 s linear
  interpolation) also leaves; LQR-only = only the replay leaves. graze = max depth < 0.3 m.
  D3 collision: arm NC < 1 or TTC < 1. Type from the first at-fault NC event (else the first TTC event): static object (non-agent),
  VRU (pedestrian / bicycle), stopped vehicle ahead (nuPlan STOPPED_TRACK, vehicle), lead vehicle (ACTIVE_FRONT, |rel heading| < 30 deg,
  object starts within 1.5 m laterally of the ego path), cut-in (ACTIVE_FRONT, |rel heading| < 45 deg, object starts > 1.5 m lateral),
  crossing / turn conflict (ACTIVE_FRONT, 30-150 deg), oncoming (> 150 deg), sideswipe (ACTIVE_LATERAL with ego in two lanes or
  off-road), rear / other. "during a turn" = token in the turning set.
"""
import argparse
import glob
import json
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
for k, v in dict(NUPLAN_MAP_VERSION="nuplan-maps-v1.0", NUPLAN_MAPS_ROOT=str(D / "datasets/navsim/maps"),
                 OPENSCENE_DATA_ROOT=str(D / "datasets/navsim"), NAVSIM_EXP_ROOT=str(D / "runs/navsim/eval"),
                 NAVSIM_DEVKIT_ROOT=str(D / "third_party/navsim"), OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="1").items():
    os.environ.setdefault(k, v)
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).parent)]
OUT = D / "runs/op_parity/four_dirs"
RES = REPO / "experiments/op_parity/results/four_dirs"
MC = {"navtest": D / "runs/navsim/metric_cache/v2_navtest", "navhard": D / "runs/navsim/metric_cache/v2_navhard_two_stage"}
SPLIT = {"navtest": "navtest", "navhard": "navhard_two_stage"}
PRED = {"navtest": D / "runs/op_lb/lb_navtest/preds/warp-cinque_PPP2H10-F-s{}__base.npz",
        "navhard": D / "runs/op_lb/lb_navhard/preds/gimm-cinque_PPP2H10-F-s{}__base.npz"}
WA_NAVTEST_TRAJ = D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl"
WA_NAVTEST_CSV = D / "runs/top10_t2/navsim/wajepa/20260926-122804/v2/2026.09.26.16.32.53.csv"
WA_NAVHARD = D / "runs/op_parity/navhard/wajepa/navhard_preds.npz"
NH_HARNESS = {"P2Hs0": D / "runs/op_parity/hinge/harness/P2H10-F-s0", "P2Hs1": D / "runs/op_parity/hinge/harness/P2H10-F-s1",
              "WA": D / "runs/op_parity/navhard/harness/wajepa"}
NT_CSV = {"P2Hs0": "v2_navtest_opi_lb_navtest_warp-cinque_PPP2H10-F-s0__base", "P2Hs1": "v2_navtest_opi_lb_navtest_warp-cinque_PPP2H10-F-s1__base"}
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
COLS = SUBS + ["two_frame_extended_comfort"]
TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
T_POSE, T_DENSE = np.arange(1, 9) * 0.5, np.arange(0, 41) * 0.1
_W = {}


# ---------------------------------------------------------------- replay (navsim2)
def _hook(scorer):
    """Wrap scorer.score_proposals: after the first 2-proposal call of a pdm_score (PDM reference + plan; the human-penalty call that
    follows has 1 proposal), read the scorer's state for proposal 1 into _W['cap']."""
    orig = scorer.score_proposals

    def wrapped(states, *a, **kw):
        r = orig(states, *a, **kw)
        if states.shape[0] == 2 and not _W["cap"]:
            _W["cap"].update(_extract(scorer, 1))
        return r
    scorer.score_proposals = wrapped


def _obj_state(obs, token, t):
    """Centroid (x, y) of a track at step t (None if absent) and its speed from the centroid track over the last 0.5 s."""
    def c(k):
        try:
            p = obs[k][token]
        except (KeyError, IndexError):
            return None
        return None if p is None else np.array([p.centroid.x, p.centroid.y])
    now = c(t)
    if now is None:
        return None, np.nan
    for dk in (5, 3, 1):
        if t - dk >= 0 and (prev := c(t - dk)) is not None:
            return now, float(np.linalg.norm(now - prev) / (0.1 * dk))
        if t + dk <= 40 and (nxt := c(t + dk)) is not None:
            return now, float(np.linalg.norm(nxt - now) / (0.1 * dk))
    return now, np.nan


def _rel(state, xy):
    """Global xy -> ego frame of `state` (x, y, heading ...)."""
    c, s = np.cos(state[2]), np.sin(state[2])
    d = xy - state[:2]
    return float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1])


def _event(sc, p, t, token, prefix, ctype=None):
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex as EA
    obs, st = sc._observation, sc._states[p]
    obj = obs.unique_objects[token]
    xy, v = _obj_state(obs, token, t)
    xy0, _ = _obj_state(obs, token, 0)
    e = {f"{prefix}_t": 0.1 * t, f"{prefix}_type": str(obj.tracked_object_type.name), f"{prefix}_agent": obj.tracked_object_type in AGENT_TYPES,
         f"{prefix}_ctype": None if ctype is None else str(ctype.name), f"{prefix}_ego_v": float(np.hypot(st[t, 3], st[t, 4])),
         f"{prefix}_obj_v": v, f"{prefix}_dh": float(np.degrees(np.angle(np.exp(1j * (obj.box.center.heading - st[t, 2]))))),
         f"{prefix}_multi": bool(sc._ego_areas[p, t, EA.MULTIPLE_LANES] or sc._ego_areas[p, t, EA.NON_DRIVABLE_AREA])}
    if xy is not None:
        e[f"{prefix}_dx"], e[f"{prefix}_dy"] = _rel(st[t], xy)
    if xy0 is not None:
        e[f"{prefix}_dx0"], e[f"{prefix}_dy0"] = _rel(st[0], xy0)                  # object at t0 in the ego t0 frame
    return e


def _extract(sc, p):
    """First at-fault collision (the scorer's own rule, re-run for proposal p), number of non-at-fault contacts, first TTC event."""
    import copy
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.metrics.utils.collision_utils import CollisionType as CT
    from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex as BB, EgoAreaIndex as EA, StateIndex as SI
    from shapely import creation
    obs = sc._observation
    out = {"n_nonfault": 0}
    seen = copy.deepcopy(obs.collided_track_ids)
    for t in range(sc.proposal_sampling.num_poses + 1):
        hit = obs[t].query(sc._ego_polygons[p, t], predicate="intersects")
        for g in np.atleast_1d(hit):
            tok = obs[t].tokens[g]
            if obs.red_light_token in tok or tok in seen:
                continue
            ct = get_collision_type(sc._states[p, t], sc._ego_polygons[p, t], obs.unique_objects[tok], obs[t][tok])
            multi = sc._ego_areas[p, t, EA.MULTIPLE_LANES] or sc._ego_areas[p, t, EA.NON_DRIVABLE_AREA]
            if ct in (CT.ACTIVE_FRONT_COLLISION, CT.STOPPED_TRACK_COLLISION) or (multi and ct == CT.ACTIVE_LATERAL_COLLISION):
                if "col_t" not in out:
                    out.update(_event(sc, p, t, tok, "col", ct))
            else:
                seen.append(tok)
                out["n_nonfault"] += 1
                out.setdefault("nf_ctype", str(ct.name))
    # TTC: same loop as PDMScorer._calculate_ttc for proposal p; first infraction
    seen = copy.deepcopy(obs.collided_track_ids)
    fut = np.arange(0, int(sc._config.future_collision_horizon_window * 10), 3)
    co = sc._ego_coords[p].copy()
    co[:, BB.CENTER] = co[:, BB.FRONT_LEFT]
    S = sc._states[p]
    v = np.hypot(S[:, SI.VELOCITY_X], S[:, SI.VELOCITY_Y])
    dxy = np.stack([np.cos(S[:, SI.HEADING]) * v, np.sin(S[:, SI.HEADING]) * v], -1)
    for t in range(sc.proposal_sampling.num_poses - int(fut.max()) + 1):
        for ft in fut:
            poly = creation.polygons(co[t] + dxy[t] * ft * sc.proposal_sampling.interval_length)
            hit = obs[t + ft].query(poly, predicate="intersects")
            for g in np.atleast_1d(hit):
                tok = obs[t + ft].tokens[g]
                if obs.red_light_token in tok or tok in seen or v[t] < sc._config.stopped_speed_threshold:
                    continue
                multi = sc._ego_areas[p, t, EA.MULTIPLE_LANES] or sc._ego_areas[p, t, EA.NON_DRIVABLE_AREA]
                ego = StateSE2(*S[t, SI.STATE_SE2])
                cen = obs[t + ft][tok].centroid
                trk = StateSE2(cen.x, cen.y, obs.unique_objects[tok].box.center.heading)
                if is_agent_ahead(ego, trk) or ((multi or sc._drivable_area_map.is_in_layer(ego.point, layer=SemanticMapLayer.INTERSECTION))
                                                 and not is_agent_behind(ego, trk)):
                    if "ttc_t" not in out:
                        out.update(_event(sc, p, t + ft, tok, "ttc"))
                        out["ttc_t_ego"] = 0.1 * t
                else:
                    seen.append(tok)
        if "ttc_t" in out:
            break
    out["nc_raw"] = float(sc._multi_metrics[0, p])
    return out


def _init(bench, keys_file):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    GlobalHydra.instance().clear()
    with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_pdm_score", overrides=[f"train_test_split={SPLIT[bench]}", f"metric_cache_path={MC[bench]}",
                                                        "experiment_name=four_dirs"])
    sim, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    _hook(scorer)
    P = pickle.load(open(keys_file, "rb"))
    # navtest: the official run's traffic (non_reactive, as the stored CSVs); navhard: the harness that produced the stored navhard results (reactive)
    pol = cfg.traffic_agents_policy.non_reactive if bench == "navtest" else cfg.traffic_agents_policy.reactive
    _W.update(cap={}, sim=sim, scorer=scorer, policy=instantiate(pol, sim.proposal_sampling), samp=sim.proposal_sampling,
              P=P["plans"], want=P["want"], bench=bench, cp={Path(p).parent.name: p for p in glob.glob(str(MC[bench] / "*/*/*/metric_cache.pkl"))})


def _dense(p8):
    P = np.vstack([[0, 0, 0], p8])
    t = np.r_[0, T_POSE]
    return np.stack([np.interp(T_DENSE, t, P[:, 0]), np.interp(T_DENSE, t, P[:, 1]), np.interp(T_DENSE, t, np.unwrap(P[:, 2]))], -1)


def _departure(mc, states, idc, area_u, o):
    """First footprint departure along global states (41, >=3): time, corner (0 FL, 1 RL, 2 RR, 3 FR), side (+1 left / -1 right),
    max depth (m), first departing point in the t0 ego frame, number of steps off."""
    import shapely
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    cor = state_array_to_coords_array(np.asarray(states)[None], mc.ego_state.car_footprint.vehicle_parameters)[0]   # (41, 5, 2)
    ins = mc.drivable_area_map.points_in_polygons(cor[None, :, :-1, :])[idc].any(0)[0]          # (41, 4)
    if ins.all():
        return dict(out=False, depth=0.0)
    t = int(np.argmax(~ins.all(1)))
    out_c = np.where(~ins[t])[0]
    dist = shapely.distance(area_u, shapely.points(cor[t, out_c]))
    k = int(out_c[np.argmax(dist)])
    allp = shapely.points(cor[:, :-1][~ins])
    c, s = np.cos(o[2]), np.sin(o[2])
    d = cor[t, k] - o[:2]
    return dict(out=True, t=0.1 * t, corner=k, side=1 if k in (0, 1) else -1, depth=float(shapely.distance(area_u, allp).max()),
                x=float(c * d[0] + s * d[1]), y=float(-s * d[0] + c * d[1]), n_off=int((~ins.all(1)).sum()))


def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.ops import unary_union
    W = _W
    with lzma.open(W["cp"][token], "rb") as f:
        mc = pickle.load(f)
    o = np.array(mc.ego_state.rear_axle.serialize())
    res = {"token": token, "rows": []}
    if W["bench"] == "navhard":
        from navsim.evaluate.pdm_score import get_trajectory_as_array
        st = get_trajectory_as_array(mc.trajectory, W["samp"], mc.ego_state.time_point)[:, :3]
        c, s = np.cos(o[2]), np.sin(o[2])
        d = st[:, :2] - o[:2]
        res["ref_path"] = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.unwrap(st[:, 2] - o[2])], -1).astype(np.float32)
        res["v0"] = float(mc.ego_state.dynamic_car_state.speed)
    keys = W["want"].get(token, [])
    if not keys:
        return res
    am = mc.drivable_area_map
    idc = am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    area_u = unary_union([am._geometries[k] for k in idc])
    c, s = np.cos(o[2]), np.sin(o[2])
    for k in keys:
        p8 = np.asarray(W["P"][k][token], np.float64)
        W["cap"] = {}
        row, st = pdm_score(metric_cache=mc, model_trajectory=Trajectory(p8), future_sampling=W["samp"], simulator=W["sim"],
                            scorer=W["scorer"], traffic_agents_policy=W["policy"])
        r = row.iloc[0]
        out = {"key": k, "token": token, **{m: float(r[m]) for m in SUBS}, **W["cap"]}
        X = np.array([out[m] for m in SUBS])
        out["score_noec"] = float(np.prod(X[:4]) * (5 * X[4] + 5 * X[5] + 2 * X[6] + 2 * X[7]) / 14)
        d = _dense(p8)
        g = np.zeros((41, st.shape[-1]))
        g[:, 0], g[:, 1], g[:, 2] = o[0] + c * d[:, 0] - s * d[:, 1], o[1] + s * d[:, 0] + c * d[:, 1], o[2] + d[:, 2]
        for nm, S in (("raw", g), ("lqr", np.asarray(st))):
            for kk, vv in _departure(mc, S, idc, area_u, o).items():
                out[f"{nm}_{kk}"] = vv
        S = np.asarray(st)
        dd = S[:, :2] - o[:2]
        out["_states"] = np.stack([c * dd[:, 0] + s * dd[:, 1], -s * dd[:, 0] + c * dd[:, 1], np.unwrap(S[:, 2] - o[2]),
                                   np.hypot(S[:, 3], S[:, 4])], -1).astype(np.float32)
        res["rows"].append(out)
    return res


def gate_fail_tokens(bench):
    """Tokens where P2H s0 / s1 or WA-JEPA fails any gate (NC, DAC, DDC, TLC < 1 or TTC < 1), from the stored official results."""
    import pandas as pd
    tabs = official(bench)
    bad = set()
    for t in tabs.values():
        m = (t[["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
                "time_to_collision_within_bound"]] < 1).any(axis=1)
        bad |= set(t.index[m])
    return sorted(bad), tabs


def official(bench):
    """Stored per-token results (index token, COLS + score [+ stage, group, weight]) of P2Hs0, P2Hs1, WA."""
    import pandas as pd
    if bench == "navtest":
        def rd(p):
            d = pd.read_csv(p)
            d = d[~d.token.astype(str).str.startswith(("average", "extended"))]
            return d.set_index("token")
        out = {k: rd(sorted((D / "runs/navsim/eval" / v).glob("*/*.csv"))[-1]) for k, v in NT_CSV.items()}
        out["WA"] = rd(WA_NAVTEST_CSV)
        return out
    return {k: pd.read_csv(v / "harness_tokens.csv").set_index("token") for k, v in NH_HARNESS.items()}


def plans(bench):
    P = {}
    for s in (0, 1):
        z = np.load(str(PRED[bench]).format(s))
        P[f"P2Hs{s}"] = dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float64)))
    if bench == "navtest":
        P["WA"] = {t: np.asarray(v, np.float64) for t, v in pickle.load(open(WA_NAVTEST_TRAJ, "rb"))["trajectories"].items()}
        z = np.load(D / "runs/navsim_zs/index/navtest_future.npz")
        P["HUM"] = dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float64)))
    else:
        z = np.load(WA_NAVHARD)
        P["WA"] = dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float64)))
    return P


def cmd_replay(a):
    import multiprocessing as mp
    import pandas as pd
    from jevdrive.run import Run
    OUT.mkdir(parents=True, exist_ok=True)
    bad, tabs = gate_fail_tokens(a.bench)
    P = plans(a.bench)
    all_tok = sorted(tabs["WA"].index)
    if a.limit:
        bad = bad[:: max(1, len(bad) // a.limit)][: a.limit]
        all_tok = bad if a.bench == "navtest" else all_tok[:: max(1, len(all_tok) // a.limit)][: a.limit] + bad
    want = {t: [k for k in P if t in P[k]] for t in bad}
    kf = OUT / f"keys_{a.bench}.pkl"
    pickle.dump({"plans": P, "want": want}, open(kf, "wb"))
    todo = sorted(set(all_tok if a.bench == "navhard" else bad))
    tag = f"{a.bench}{'-lim' if a.limit else ''}"
    with Run("op_parity", f"four-dirs-replay-{tag}", config=vars(a) | {"n_tokens": len(todo), "n_bad": len(bad)}) as run:
        t0 = time.time()
        rows, ref = [], {}
        with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.bench, kf)) as pool:
            for i, r in enumerate(pool.imap_unordered(work, todo, chunksize=2)):
                rows.extend(r["rows"])
                if "ref_path" in r:
                    ref[r["token"]] = (r["ref_path"], r["v0"])
                if (i + 1) % 250 == 0:
                    run.status(f"{i + 1}/{len(todo)} tokens, {time.time() - t0:.0f} s")
        S = np.stack([r.pop("_states") for r in rows]) if rows else np.zeros((0, 41, 4))
        df = pd.DataFrame(rows)
        sfx = "_lim" if a.limit else ""
        df.to_parquet(OUT / f"replay_{a.bench}{sfx}.parquet")
        np.savez_compressed(OUT / f"replay_{a.bench}{sfx}_states.npz", key=df.key.to_numpy(), token=df.token.to_numpy(), states=S)
        if ref:
            tk = sorted(ref)
            np.savez_compressed(OUT / f"refpath_navhard{sfx}.npz", tokens=np.array(tk), path=np.stack([ref[t][0] for t in tk]),
                                v0=np.array([ref[t][1] for t in tk]))
            from omegaconf import OmegaConf  # noqa: F401
            sys.path.insert(0, str(REPO / "experiments/skill_pack/scripts"))
            import offroad_lib as OL
            _, _, _, mapping, _ = OL.setup_scoring()
            json.dump([[list(k), [list(p) for p in v]] for k, v in mapping.items()], open(OUT / "mapping_navhard.json", "w"))
        # check against the official per-token results (scores are the devkit's own; EC is not recomputed)
        chk = {}
        for k in ("P2Hs0", "P2Hs1", "WA"):
            q = df[df.key == k].set_index("token")
            if len(q):
                o = tabs[k].loc[q.index]
                chk[k] = {m: float(np.abs(q[m] - o[m]).max()) for m in SUBS}
        run.summary.update(n=len(todo), rows=len(df), wall_s_scoring=time.time() - t0, max_abs_diff_vs_official=chk)
        run.info(json.dumps(run.summary, indent=1))


# ---------------------------------------------------------------- analyze (.venv)
def path_geom(P):
    """P (n, 8, 3) poses at 0.5 .. 4 s in the t0 frame -> dpsi (deg, 4 s), R_min (m, min over 1 s windows with arc >= 2 m), path length (m)."""
    n = len(P)
    Q = np.concatenate([np.zeros((n, 1, 3)), P], 1)
    h = np.unwrap(Q[:, :, 2], axis=1)
    seg = np.linalg.norm(np.diff(Q[:, :, :2], axis=1), axis=2)              # (n, 8)
    arc = seg[:, :-1] + seg[:, 1:]                                           # 1 s windows
    dh = np.abs(h[:, 2:] - h[:, :-2])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(arc >= 2.0, arc / np.maximum(dh, 1e-6), np.inf)
    return dict(dpsi=np.degrees(h[:, -1]), R_min=R.min(1), path_len=seg.sum(1))


def buckets(g):
    """g: DataFrame with dpsi, R_min, path_len -> geometry sets (rules in the module docstring)."""
    import pandas as pd
    a = g.dpsi.abs()
    turn = (a >= 8) & (g.path_len >= 3)
    out = pd.DataFrame(index=g.index)
    out["turning"] = turn
    out["sharp"] = turn & (g.R_min < 15)
    out["wide"] = turn & (g.R_min >= 15)
    out["wide_curve"] = out.wide & (a < 20)
    out["wide_turn"] = out.wide & (a >= 20)
    out["sharp45"] = a > 45
    out["bin"] = pd.cut(a, [-1, 5, 20, 45, 1e9], labels=["<5", "5-20", "20-45", ">45"]).astype(str)
    out["rbin"] = pd.cut(g.R_min.where(turn, np.inf), [0, 8, 15, 30, 60, np.inf], labels=["<8", "8-15", "15-30", "30-60", "not turning or >60"],
                         right=False).astype(str)
    return out


def score9(X):
    ec = X[..., 8]
    num = 5 * X[..., 4] + 5 * X[..., 5] + 2 * X[..., 6] + 2 * X[..., 7] + 2 * np.nan_to_num(ec)
    return np.prod(X[..., :4], -1) * num / (14 + 2 * np.isfinite(ec))


class NavhardAgg:
    """The devkit's calculate_individual_mapping_scores in numpy: per scene-mapping group, (stage-1 weighted mean x stage-2 weighted mean)
    averaged over the group's two sub-groups; weights are the arm's own (stage-2 weights come from the arm's stage-1 replay)."""

    def __init__(self, tokens, tok_log):
        mp = json.load(open(OUT / "mapping_navhard.json"))
        pos = {t: i for i, t in enumerate(tokens)}
        self.sub = []
        for (orig, prev), pairs in mp:
            self.sub.append(([pos[orig]], [pos[p[0]] for p in pairs if len(p) > 0 and p[0] in pos]))
            self.sub.append(([pos[prev]], [pos[p[1]] for p in pairs if len(p) > 1 and p[1] in pos]))
        self.log = np.array([tok_log[orig] for (orig, prev), _ in mp])     # cluster of a group = log of its stage-1 token

    def groups(self, s, w):
        """token scores s (n,), weights w (n,) -> per-group combined (ng,), per-subgroup stage 1 / stage 2 (2 ng,)."""
        def wavg(ix):
            ww = w[ix]
            return float((s[ix] * ww).sum() / ww.sum()) if len(ix) and ww.sum() > 0 else np.nan
        s1 = np.array([wavg(a) for a, _ in self.sub])
        s2 = np.array([wavg(b) for _, b in self.sub])
        return (s1 * s2).reshape(-1, 2).mean(1), s1, s2


def _load(bench, pd):
    tabs = official(bench)
    rep = pd.read_parquet(OUT / f"replay_{bench}.parquet")
    toks = sorted(tabs["WA"].index)
    if bench == "navtest":
        tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
        ix = {t: i for i, t in enumerate(tab["names"].tolist())}
        sel = np.array([ix[t] for t in toks])
        fut = tab["fut"][sel].astype(np.float64)
        geo = pd.DataFrame(path_geom(fut), index=toks)
        geo["v0"], geo["log"] = tab["speed"][sel], tab["log"][sel]
    else:
        z = np.load(OUT / "refpath_navhard.npz")
        ix = {t: i for i, t in enumerate(z["tokens"].tolist())}
        sel = np.array([ix[t] for t in toks])
        fut = z["path"][sel].astype(np.float64)[:, 5::5]                     # PDM-Closed reference at 0.5 .. 4 s
        geo = pd.DataFrame(path_geom(fut), index=toks)
        geo["v0"] = z["v0"][sel]
        tab = np.load(D / "runs/op_parity/cache/lb_navhard/tab.npz")
        li = {t: i for i, t in enumerate(tab["names"].tolist())}
        geo["log"] = [tab["log"][li[t]] for t in toks]
    for k in tabs:
        tabs[k] = tabs[k].loc[toks]
    geo = pd.concat([geo, buckets(geo)], axis=1)
    return tabs, rep, toks, geo, fut


BUCKETS = {
    "D1 sharp (R<15 m) DAC": lambda f, g: f.dac & g.sharp,
    "D1' sharp (|dpsi|>45) DAC": lambda f, g: f.dac & g.sharp45,
    "D2 wide (R>=15 m) DAC": lambda f, g: f.dac & g.wide,
    "D2a wide curve 8-20 deg DAC": lambda f, g: f.dac & g.wide_curve,
    "D2b wide turn >=20 deg DAC": lambda f, g: f.dac & g.wide_turn,
    "context: straight DAC": lambda f, g: f.dac & ~g.turning,
    "context: all DAC": lambda f, g: f.dac,
    "D3 collision (NC or TTC)": lambda f, g: f.col,
    "D3 on turning tokens": lambda f, g: f.col & g.turning,
    "D1 + D2 + D3 union": lambda f, g: (f.dac & g.turning) | f.col,
}


def analyze_bench(bench, pd, stats, shapley):
    tabs, rep, toks, geo, fut = _load(bench, pd)
    n = len(toks)
    arms = ["P2Hs0", "P2Hs1"]
    X = {k: tabs[k][COLS].to_numpy(float) for k in tabs}
    rp = {k: rep[rep.key == k].set_index("token") for k in rep.key.unique()}
    if bench == "navtest":                       # reference: the logged human trajectory, replayed in-process (no official v2 human run)
        R = pd.DataFrame(np.nan, index=toks, columns=SUBS)
        R.loc[rp["HUM"].index, SUBS] = rp["HUM"][SUBS].to_numpy()
        grp = geo.log.to_numpy()
        stage = np.ones(n, int)
    else:                                        # reference: PDM-Closed, the loss-budget prep's in-process rows
        prep = pickle.load(open(D / "runs/leaderboard_audit/loss_budget/prep_navhard.pkl", "rb"))
        R = pd.DataFrame([{m: float(prep[t]["rows"]["ref"][m].iloc[0]) for m in SUBS} for t in toks], index=toks)
        stage = tabs["WA"].stage.to_numpy()
        agg = NavhardAgg(toks, dict(zip(toks, geo.log)))
        grp = agg.log[tabs["WA"].group.to_numpy()]                          # token -> stage-1 log of its group
    Rx = R[SUBS[:7]].to_numpy(float)
    F = {}
    for k, t in tabs.items():
        F[k] = pd.DataFrame(dict(dac=t.drivable_area_compliance < 1, nc=t.no_at_fault_collisions < 1,
                                 col=(t.no_at_fault_collisions < 1) | (t.time_to_collision_within_bound < 1)), index=toks)
    W = {k: (tabs[k].weight.to_numpy(float) if bench == "navhard" else None) for k in tabs}

    def board(Xa, w):
        s = score9(Xa)
        return (s, None, None) if bench == "navtest" else agg.groups(s, w)

    def substitute(Xa, m, how):
        Y = Xa.copy()
        if how == "WA":
            Y[m] = X["WA"][m]
        else:                                    # reference trajectory terms (NC .. LK), comfort stays the arm's own; never-worse clip
            C = Y.copy()
            C[m, :7] = Rx[m]
            ok = m & np.isfinite(Rx).all(1) & (score9(C) > score9(Y))
            Y[ok] = C[ok]
        return Y

    base = {k: board(X[k], W[k]) for k in arms + ["WA"]}
    out = dict(n=n, base={k: 100 * float(np.nanmean(v[0])) for k, v in base.items()})
    if bench == "navhard":
        out["base_s12"] = {k: [100 * float(np.nanmean(v[1])), 100 * float(np.nanmean(v[2]))] for k, v in base.items()}
        out["harness_check"] = {k: json.load(open(NH_HARNESS[k] / "harness_summary.json"))["combined"] for k in NH_HARNESS}
    else:
        out["csv_check"] = {k: float(np.abs(score9(X[k]) - tabs[k].score.to_numpy()).max()) for k in tabs}
    phi = {k: shapley(X[k], X["WA"]) for k in arms}                        # gap WA - arm per token, additive over terms
    rows, shap_rows = [], []
    ug = grp if bench == "navtest" else agg.log
    for name, fn in BUCKETS.items():
        for how in ("WA", "ref"):
            du, d1, d2, cnt = [], [], [], []
            for k in arms:
                m = fn(F[k], geo).to_numpy()
                cnt.append(int(m.sum()))
                b = board(substitute(X[k], m, how), W[k])
                du.append(b[0] - base[k][0])
                if bench == "navhard":
                    d1.append(b[1] - base[k][1])
                    d2.append(b[2] - base[k][2])
            r = stats.bootstrap(np.mean(du, 0), ug)
            row = dict(bench=bench, bucket=name, oracle=how, n_s0=cnt[0], n_s1=cnt[1], gain=100 * r["mean"], lo=100 * r["lo"], hi=100 * r["hi"])
            if bench == "navhard":
                for st, dd in (("stage1", d1), ("stage2", d2)):
                    q = stats.bootstrap(np.mean(dd, 0), np.repeat(agg.log, 2))
                    row.update({st: 100 * q["mean"], f"{st}_lo": 100 * q["lo"], f"{st}_hi": 100 * q["hi"]})
            rows.append(row)
        # (c) Shapley split of the WA - P2H gap on the bucket's tokens that fail for P2H (that seed) or for WA; points of the board mean
        ph = np.mean([np.where((fn(F[k], geo) | fn(F["WA"], geo)).to_numpy()[:, None], phi[k], 0.0) for k in arms], 0)
        for st in ((0,) if bench == "navtest" else (1, 2)):
            ms = np.ones(n, bool) if st == 0 else stage == st
            row = dict(bench=bench, bucket=name, stage=st or "all", n_tokens_s0=int(((fn(F["P2Hs0"], geo) | fn(F["WA"], geo)).to_numpy() & ms).sum()))
            q = stats.bootstrap(ph[ms].sum(1), grp[ms])
            row.update(total=100 * q["mean"], total_lo=100 * q["lo"], total_hi=100 * q["hi"])
            for j, t in enumerate(TERMS):
                q = stats.bootstrap(ph[ms, j], grp[ms])
                row.update({t: 100 * q["mean"], f"{t}_lo": 100 * q["lo"], f"{t}_hi": 100 * q["hi"]})
            shap_rows.append(row)
    pd.DataFrame(rows).to_csv(RES / f"nav_oracle_{bench}.csv", index=False, float_format="%.4f")
    pd.DataFrame(shap_rows).to_csv(RES / f"nav_shapley_{bench}.csv", index=False, float_format="%.4f")
    if bench == "navhard":                       # official gap per stage for context (unweighted token Shapley sums to the unweighted gap)
        out["gap_unweighted_stage"] = {st: 100 * float((score9(X["WA"]) - np.mean([score9(X[k]) for k in arms], 0))[stage == st].mean()) for st in (1, 2)}
    else:
        out["gap"] = 100 * float((score9(X["WA"]) - np.mean([score9(X[k]) for k in arms], 0)).mean())
    sets = {d: [BUCKETS[b](F[k], geo) for k in arms] for d, b in (("D1", "D1 sharp (R<15 m) DAC"), ("D2", "D2 wide (R>=15 m) DAC"),
                                                                 ("D3", "D3 collision (NC or TTC)"))}
    out["overlap"] = {f"{x}&{y}": float(np.mean([(sets[x][i] & sets[y][i]).sum() for i in (0, 1)])) for x, y in (("D1", "D2"), ("D1", "D3"), ("D2", "D3"))}
    out["geometry_counts"] = {c: int(geo[c].sum()) for c in ("turning", "sharp", "wide", "wide_curve", "wide_turn", "sharp45")}
    out["bucket_sizes"] = {d: float(np.mean([s.sum() for s in v])) for d, v in sets.items()}
    rate_rows = []
    for col in ("bin", "rbin"):
        for b in sorted(geo[col].unique()):
            m = (geo[col] == b).to_numpy()
            for metric in ("dac", "col"):
                for arm, ks in (("P2H", arms), ("WA", ["WA"])):
                    q = stats.bootstrap(np.mean([F[k][metric].to_numpy(float) for k in ks], 0)[m], grp[m])
                    rate_rows.append(dict(bench=bench, by=col, bin=b, metric=metric, arm=arm, n=int(m.sum()), rate=100 * q["mean"], lo=100 * q["lo"], hi=100 * q["hi"]))
    pd.DataFrame(rate_rows).to_csv(RES / f"nav_rates_{bench}.csv", index=False, float_format="%.4f")
    out["mechanism"] = mechanism(bench, pd, stats, tabs, rp, geo, fut, F, toks, grp)
    return out


def _share(stats, x, g):
    x = np.asarray(x, float)
    q = stats.bootstrap(x, np.asarray(g))
    return dict(share=100 * q["mean"], lo=100 * q["lo"], hi=100 * q["hi"], n=q["n"])


def _arc(P):
    n = len(P)
    return np.concatenate([np.zeros((n, 1)), np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros((n, 1, 2)), P[:, :, :2]], 1), axis=1), axis=2), 1)], 1)


def plan_kin(P, fut):
    """Plan vs logged / reference path: heading gain at 4 s (plan dpsi / path dpsi), lateral offset at 1 / 2 / 4 s along the path's normal
    (toward the inside of the turn = +), speed ratio over 4 s (plan arc / path arc)."""
    sgn = np.sign(fut[:, -1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.unwrap(np.concatenate([np.zeros((len(P), 1)), P[:, :, 2]], 1), axis=1)[:, -1] / np.where(np.abs(fut[:, -1, 2]) > 0.05, fut[:, -1, 2], np.nan)
    out = dict(gain=gain)
    for t, i in (("1", 1), ("2", 3), ("4", 7)):
        h = fut[:, i, 2]
        out[f"lat{t}"] = sgn * ((P[:, i, :2] - fut[:, i, :2]) * np.stack([-np.sin(h), np.cos(h)], 1)).sum(1)
    out["speed_ratio"] = _arc(P)[:, -1] / np.maximum(_arc(fut)[:, -1], 0.5)
    return out


def ctype_class(r):
    """Collision type of one failing token from the instrumented events (rules in the module docstring)."""
    f = lambda k: r.get(k) if r.get(k) is not None and not (isinstance(r.get(k), float) and np.isnan(r.get(k))) else None  # noqa: E731
    pre = "col" if f("col_t") is not None else "ttc" if f("ttc_t") is not None else None
    if pre is None:
        return "unresolved"
    dh = abs(f(f"{pre}_dh") or 0.0)
    dy0 = f(f"{pre}_dy0")
    if not f(f"{pre}_agent"):
        return "static object"
    if f(f"{pre}_type") in ("PEDESTRIAN", "BICYCLE"):
        return "VRU"
    ct = f(f"{pre}_ctype")
    if ct == "STOPPED_TRACK_COLLISION" or (pre == "ttc" and (f("ttc_obj_v") or 0) < 0.5 and dh < 30):
        return "stopped vehicle ahead"
    if ct == "ACTIVE_LATERAL_COLLISION":
        return "sideswipe (ego across lanes)"
    if dh > 150:
        return "oncoming"
    if dh >= 30:
        return "crossing / turn conflict"
    if dy0 is not None and abs(dy0) > 1.5 and dh < 45:
        return "cut-in"
    return "lead vehicle"


def mechanism(bench, pd, stats, tabs, rp, geo, fut, F, toks, g):
    """Per direction and arm: DAC side (inside / outside), raw vs LQR-only, graze, t0 speed, heading gain, turn-in offsets, decision-147 split
    (navtest eval tokens); collision types, plan vs path speed, WA overlap. Per-token rows -> nav_tokens_<bench>.csv."""
    pos = {t: i for i, t in enumerate(toks)}
    P = {k: np.stack([plans_cache(bench)[k][t] for t in toks]) for k in ("P2Hs0", "P2Hs1", "WA")}
    kin = {k: plan_kin(P[k], fut) for k in P}
    farc = _arc(fut)
    res, case_rows = {}, []
    dec = None
    if bench == "navtest":
        d = pd.read_parquet(D / "runs/op_probe/joint/decoders.parquet")
        dec = d[d.obj == "hinge10"].pivot_table(index="token", columns="stage", values="DAC")
    dsets = {"D1 sharp": geo.sharp, "D1' >45": geo.sharp45, "D2 wide": geo.wide, "D2a curve": geo.wide_curve, "D2b wide turn": geo.wide_turn,
             "straight": ~geo.turning}
    for arm, ks in (("P2H", ["P2Hs0", "P2Hs1"]), ("WA", ["WA"])):
        for dname, gm in dsets.items():
            recs = []
            for k in ks:
                q = rp[k].reindex((F[k].dac & gm).pipe(lambda m: m.index[m]))
                ii = np.array([pos[t] for t in q.index], int)
                sgn = np.sign(geo.dpsi.to_numpy()[ii])
                kk = {c: v[ii] for c, v in kin[k].items()}
                recs.append(pd.DataFrame(dict(
                    token=q.index, log=g[ii], key=k, dpsi=geo.dpsi.to_numpy()[ii], R_min=geo.R_min.to_numpy()[ii],
                    inside=(q.lqr_side.to_numpy(float) == sgn).astype(float), raw=q.raw_out.astype(float).to_numpy(),
                    lqr_only=(~q.raw_out.astype(bool) & q.lqr_out.astype(bool)).astype(float).to_numpy(),
                    graze=(q.lqr_depth < 0.3).astype(float).to_numpy(), depth=q.lqr_depth.to_numpy(float), t_first=q.lqr_t.to_numpy(float),
                    front=q.lqr_corner.isin([0, 3]).astype(float).to_numpy(), v0=geo.v0.to_numpy()[ii], ep=tabs[k].loc[q.index, "ego_progress"].to_numpy(float),
                    wa_fail=F["WA"].dac.to_numpy()[ii].astype(float), **kk)))
            r = pd.concat(recs, ignore_index=True)
            if len(r) < 3:
                continue
            e = dict(n=len(r) / len(ks))
            for c in ("inside", "raw", "lqr_only", "graze", "wa_fail", "front"):
                e[c] = _share(stats, r[c], r.log)
            for c in ("depth", "v0", "gain", "lat1", "lat2", "lat4", "speed_ratio", "t_first", "ep"):
                e[c + "_med"] = float(np.nanmedian(r[c]))
            if dname != "straight":
                fin = np.isfinite(r.gain)
                e["undershoot"] = _share(stats, (r.gain < 0.9)[fin], r.log[fin])
                e["overshoot"] = _share(stats, (r.gain > 1.1)[fin], r.log[fin])
                e["outside_undershoot"] = _share(stats, ((r.inside == 0) & (r.gain < 0.9))[fin], r.log[fin])
                e["inside_early_or_over"] = _share(stats, (r.inside == 1) & ((r.gain > 1.1) | (r.lat2 > 0.3)), r.log)
                ok = []
                for k in ks:
                    mm = (~F[k].dac & gm).to_numpy()
                    ok.append(pd.DataFrame(dict(gain=kin[k]["gain"][mm], lat2=kin[k]["lat2"][mm], v0=geo.v0.to_numpy()[mm])))
                ok = pd.concat(ok)
                e.update(pass_gain_med=float(np.nanmedian(ok.gain)), pass_lat2_med=float(np.nanmedian(ok.lat2)), pass_v0_med=float(np.nanmedian(ok.v0)),
                         pass_undershoot=float(np.nanmean((ok.gain < 0.9)[np.isfinite(ok.gain)])))
            if dec is not None and arm == "P2H":
                q = r[r.token.isin(dec.index)]
                if len(q) >= 5:
                    dv = dec.loc[q.token]
                    e["d147_n"] = len(q)
                    e["d147_encoder_fails"] = _share(stats, dv["P2-V"].to_numpy() < 1, q.log)
                    e["d147_head_fails"] = _share(stats, dv["P2-H"].to_numpy() < 1, q.log)
                    e["d147_WAenc_passes"] = _share(stats, dv["WA-Cf"].to_numpy() >= 1, q.log)
            res[f"{arm}|{dname}"] = e
            r["bucket"], r["arm"] = dname, arm
            case_rows.append(r)
        recs = []
        for k in ks:
            q = rp[k].reindex(F[k].col.pipe(lambda m: m.index[m]))
            ii = np.array([pos[t] for t in q.index], int)
            col_t = q["col_t"] if "col_t" in q else pd.Series(np.nan, q.index)
            ttc_t = q["ttc_t"] if "ttc_t" in q else pd.Series(np.nan, q.index)
            t_ev = col_t.fillna(ttc_t).to_numpy(float)
            pre = np.where(col_t.notna(), "col", "ttc")
            get = lambda c: np.array([q[f"{p}_{c}"].iloc[j] if f"{p}_{c}" in q else np.nan for j, p in enumerate(pre)], float)  # noqa: E731
            # plan progress vs path progress at the event time (plan poses interpolated in time)
            tt = np.r_[0, T_POSE]
            pa, fa = _arc(P[k][ii]), farc[ii]
            prog = np.array([np.interp(t, tt, a) / max(np.interp(t, tt, b), 0.5) if np.isfinite(t) else np.nan for t, a, b in zip(t_ev, pa, fa)])
            recs.append(pd.DataFrame(dict(
                token=q.index, log=g[ii], key=k, cls=[ctype_class(x) for x in q.to_dict("records")], src=pre,
                nc=(tabs[k].loc[q.index, "no_at_fault_collisions"] < 1).to_numpy(), turning=geo.turning.to_numpy()[ii], dpsi=geo.dpsi.to_numpy()[ii],
                spd=kin[k]["speed_ratio"][ii], prog_at_event=prog, v0=geo.v0.to_numpy()[ii], wa_fail=F["WA"].col.to_numpy()[ii], t=t_ev,
                ego_v=get("ego_v"), obj_v=get("obj_v"), dh=get("dh"), dx=get("dx"), dy=get("dy"), dy0=get("dy0"))))
        r = pd.concat(recs, ignore_index=True)
        e = dict(n=len(r) / len(ks), nc_share=_share(stats, r.nc, r.log), turning=_share(stats, r.turning, r.log), wa_fail=_share(stats, r.wa_fail, r.log),
                 faster=_share(stats, r.spd > 1.1, r.log), slower=_share(stats, r.spd < 0.9, r.log), spd_med=float(np.nanmedian(r.spd)),
                 prog_med=float(np.nanmedian(r.prog_at_event)), ahead_at_event=_share(stats, (r.prog_at_event > 1.1)[np.isfinite(r.prog_at_event)],
                                                                                         r.log[np.isfinite(r.prog_at_event)]))
        e["types"] = {c: dict(**_share(stats, r.cls == c, r.log), spd_med=float(np.nanmedian(r.spd[r.cls == c])),
                              prog_med=float(np.nanmedian(r.prog_at_event[r.cls == c])), wa_fail=float(r.wa_fail[r.cls == c].mean()),
                              turning=float(r.turning[r.cls == c].mean())) for c in sorted(r.cls.unique())}
        res[f"{arm}|D3 collision"] = e
        r["bucket"], r["arm"] = "D3 collision", arm
        case_rows.append(r)
    pd.concat(case_rows, ignore_index=True).to_csv(RES / f"nav_tokens_{bench}.csv", index=False, float_format="%.4f")
    return res


_PC = {}


def plans_cache(bench):
    if bench not in _PC:
        _PC[bench] = plans(bench)
    return _PC[bench]


def cmd_analyze(a):
    import pandas as pd
    from jevdrive import stats
    from pp_gap_tables import shapley
    RES.mkdir(parents=True, exist_ok=True)
    summary = {b: analyze_bench(b, pd, stats, shapley) for b in a.bench}
    old = json.load(open(RES / "nav_summary.json")) if (RES / "nav_summary.json").exists() else {}
    json.dump(old | summary, open(RES / "nav_summary.json", "w"), indent=1, default=float)
    print(json.dumps({b: {k: v for k, v in s.items() if k != "mechanism"} for b, s in summary.items()}, indent=1, default=float))


# ---------------------------------------------------------------- cases (navsim2) and figures (.venv)
def pick_cases(pd):
    """Typical cases (rule fixed before drawing): navtest, P2H seed 0. D1 / D2: failing tokens of the bucket with v0 > 2 m/s whose side is
    the bucket's majority side (D1 outside, D2 by majority), the token at the median LQR depth. D3: tokens of the most frequent collision type,
    the one at the median ego speed at the event."""
    t = pd.read_csv(RES / "nav_tokens_navtest.csv")
    t = t[(t.arm == "P2H") & (t.key == "P2Hs0")]
    picks = []
    for b in ("D1 sharp", "D2 wide"):
        q = t[(t.bucket == b) & (t.v0 > 2)]
        side = 0.0 if b == "D1 sharp" else float(q.inside.mode().iloc[0])
        q = q[q.inside == side].sort_values("depth")
        picks.append(dict(bucket=b, token=q.token.iloc[len(q) // 2], key="P2Hs0"))
    q = t[t.bucket == "D3 collision"]
    q = q[q.cls == q.cls.mode().iloc[0]].sort_values("ego_v")
    picks.append(dict(bucket="D3 collision", token=q.token.iloc[len(q) // 2], key="P2Hs0", cls=q.cls.iloc[len(q) // 2]))
    return picks


def cmd_cases(a):
    import pandas as pd
    from shapely.geometry import box
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    picks = pick_cases(pd)
    rep = pd.read_parquet(OUT / "replay_navtest.parquet")
    S = np.load(OUT / "replay_navtest_states.npz", allow_pickle=True)
    srow = {(k, t): i for i, (k, t) in enumerate(zip(S["key"], S["token"]))}
    P = plans("navtest")
    cp = {Path(p).parent.name: p for p in glob.glob(str(MC["navtest"] / "*/*/*/metric_cache.pkl"))}
    out = []
    for p in picks:
        tok = p["token"]
        with lzma.open(cp[tok], "rb") as f:
            mc = pickle.load(f)
        o = np.array(mc.ego_state.rear_axle.serialize())
        c, s = np.cos(o[2]), np.sin(o[2])
        to_ego = lambda xy: np.stack([c * (xy[:, 0] - o[0]) + s * (xy[:, 1] - o[1]), -s * (xy[:, 0] - o[0]) + c * (xy[:, 1] - o[1])], 1)  # noqa: E731
        am = mc.drivable_area_map
        idc = am.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
        win = box(o[0] - 60, o[1] - 60, o[0] + 60, o[1] + 60)
        polys = []
        for k in idc:
            gg = am._geometries[k].intersection(win)
            for q in getattr(gg, "geoms", [gg]):
                if q.geom_type == "Polygon" and not q.is_empty:
                    polys.append(to_ego(np.asarray(q.exterior.coords)))
        r = rep[(rep.key == p["key"]) & (rep.token == tok)].iloc[0].to_dict()
        fin = lambda v: v is not None and v == v  # noqa: E731
        tev = next((r[k] for k in ("col_t", "ttc_t", "lqr_t") if fin(r.get(k))), 2.0)
        ti = int(round(10 * tev))
        agents = []
        ob = mc.observation[ti]
        for tk in ob.tokens:
            q = ob[tk]
            if q.distance(win.centroid) < 60:
                agents.append(to_ego(np.asarray(q.exterior.coords)))
        out.append(dict(p, polys=polys, agents=agents, t_draw=0.1 * ti, plan=P[p["key"]][tok], wa=P["WA"][tok], hum=P["HUM"][tok],
                        lqr=S["states"][srow[(p["key"], tok)]], lqr_wa=S["states"][srow[("WA", tok)]] if ("WA", tok) in srow else None,
                        row={k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}))
        print("case", p, flush=True)
    pickle.dump(out, open(OUT / "cases_navtest.pkl", "wb"))


def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    figs = RES / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    C = {"P2H": "#c0504d", "WA": "#4f81bd"}
    # 1. DAC and collision failure rate by logged radius and heading bin
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, bench in enumerate(("navtest", "navhard")):
        r = pd.read_csv(RES / f"nav_rates_{bench}.csv")
        for i, (by, order) in enumerate((("rbin", ["<8", "8-15", "15-30", "30-60", "not turning or >60"]), ("bin", ["<5", "5-20", "20-45", ">45"]))):
            a_ = ax[i, j]
            for o_, (arm, mk) in enumerate((("P2H", "o"), ("WA", "s"))):
                for metric, ls in (("dac", "-"), ("col", ":")):
                    q = r[(r.by == by) & (r.arm == arm) & (r.metric == metric)].set_index("bin").reindex(order)
                    x = np.arange(len(order)) + (o_ - 0.5) * 0.12 + (0.06 if metric == "col" else 0)
                    a_.errorbar(x, q.rate, yerr=[q.rate - q.lo, q.hi - q.rate], color=C[arm], ls=ls, marker=mk, ms=4, capsize=2,
                                label=f"{arm} {'DAC fail' if metric == 'dac' else 'NC/TTC fail'}")
            a_.set_xticks(range(len(order)))
            a_.set_xticklabels([f"{b}\n(n {int(r[(r.by == by) & (r.bin == b)].n.iloc[0])})" if (r.by == by).any() and b in set(r[r.by == by].bin) else b for b in order], fontsize=7)
            a_.set_xlabel("min path radius over 4 s (m)" if by == "rbin" else "|heading change| over 4 s (deg)")
            a_.set_ylabel("failure rate (%)")
            a_.set_title(f"{bench} ({'logged path' if bench == 'navtest' else 'PDM reference path'})", fontsize=9)
            a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "nav_rates.png", dpi=130)
    plt.close(fig)
    # 2. DAC failure anatomy per bucket: inside / outside x raw-plan / LQR-only, for P2H and WA
    sm = json.load(open(RES / "nav_summary.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, bench in enumerate(("navtest", "navhard")):
        m = sm[bench]["mechanism"]
        labels, vals = [], []
        for b in ("D1 sharp", "D2a curve", "D2b wide turn", "straight"):
            for arm in ("P2H", "WA"):
                e = m.get(f"{arm}|{b}")
                if not e:
                    continue
                labels.append(f"{arm}\n{b}\n(n {e['n']:.0f})")
                vals.append([e["inside"]["share"], e["raw"]["share"], e["graze"]["share"], e.get("undershoot", {}).get("share", np.nan)])
        V = np.array(vals)
        x = np.arange(len(labels))
        for k, (nm, col) in enumerate((("inside of the turn", "#9bbb59"), ("raw plan leaves (not LQR-only)", "#8064a2"), ("graze < 0.3 m", "#f79646"),
                                       ("heading gain < 0.9", "#4bacc6"))):
            ax[j].bar(x + (k - 1.5) * 0.2, V[:, k], 0.2, color=col, label=nm)
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(labels, fontsize=6)
        ax[j].set_ylabel("share of DAC failures (%)")
        ax[j].set_title(bench, fontsize=9)
        ax[j].set_ylim(0, 100)
        ax[j].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "nav_dac_anatomy.png", dpi=130)
    plt.close(fig)
    # 3. collision types
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, bench in enumerate(("navtest", "navhard")):
        m = sm[bench]["mechanism"]
        types = sorted(set(m["P2H|D3 collision"]["types"]) | set(m["WA|D3 collision"]["types"]))
        for o_, arm in enumerate(("P2H", "WA")):
            e = m[f"{arm}|D3 collision"]
            v = np.array([e["types"].get(t, {}).get("share", 0) * e["n"] / 100 for t in types])
            ax[j].barh(np.arange(len(types)) + (o_ - 0.5) * 0.4, v, 0.4, color=C[arm], label=f"{arm} (n {e['n']:.0f})")
        ax[j].set_yticks(range(len(types)))
        ax[j].set_yticklabels(types, fontsize=7)
        ax[j].set_xlabel("failing tokens (NC < 1 or TTC < 1; P2H seed mean)")
        ax[j].set_title(bench, fontsize=9)
        ax[j].legend(fontsize=7)
        ax[j].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "nav_collision_types.png", dpi=130)
    plt.close(fig)
    # 4. typical BEV cases
    cp = OUT / "cases_navtest.pkl"
    if cp.exists():
        cases = pickle.load(open(cp, "rb"))
        fig, ax = plt.subplots(1, len(cases), figsize=(5 * len(cases), 5))
        for a_, c in zip(np.atleast_1d(ax), cases):
            for p in c["polys"]:
                a_.fill(p[:, 1], p[:, 0], color="#dddddd", ec="#aaaaaa", lw=0.5)
            for p in c["agents"]:
                a_.fill(p[:, 1], p[:, 0], color="#999999", ec="k", lw=0.6, alpha=0.8)
            for nm, tr, col, ls in (("logged", c["hum"], "k", "--"), ("P2H plan", c["plan"], C["P2H"], "-"), ("WA-JEPA plan", c["wa"], C["WA"], "-")):
                q = np.vstack([[0, 0, 0], tr])
                a_.plot(q[:, 1], q[:, 0], ls, color=col, lw=1.5, marker=".", label=nm)
            a_.plot(c["lqr"][:, 1], c["lqr"][:, 0], ":", color=C["P2H"], lw=1.5, label="P2H LQR replay")
            r = c["row"]
            ext = np.abs(np.vstack([c["hum"][:, :2], c["plan"][:, :2]])).max() + 8
            a_.set_xlim(ext, -ext)
            a_.set_ylim(-8, 2 * ext - 8)
            a_.set_aspect("equal")
            ttl = f"{c['bucket']}: {c['token']}\nDAC {r['drivable_area_compliance']:.0f} NC {r['no_at_fault_collisions']:.1f} TTC {r['time_to_collision_within_bound']:.0f}"
            if c["bucket"].startswith("D3"):
                ttl += f", {c.get('cls')}; agents at t = {c['t_draw']:.1f} s"
            else:
                ttl += f", depth {r.get('lqr_depth', 0):.2f} m at t = {r.get('lqr_t', 0):.1f} s"
            a_.set_title(ttl, fontsize=8)
            a_.legend(fontsize=6, loc="lower left")
        fig.tight_layout()
        fig.savefig(figs / "nav_cases.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("replay")
    p.add_argument("--bench", choices=["navtest", "navhard"], required=True)
    p.add_argument("--procs", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("analyze")
    p.add_argument("--bench", nargs="+", default=["navtest", "navhard"])
    sp.add_parser("cases")
    sp.add_parser("figs")
    a = ap.parse_args()
    {"replay": cmd_replay, "analyze": cmd_analyze, "cases": cmd_cases, "figs": cmd_figs}[a.cmd](a)
