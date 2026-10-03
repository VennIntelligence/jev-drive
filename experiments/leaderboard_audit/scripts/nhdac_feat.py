"""navhard DAC attribution, step 1 (CPU, navsim2 env). Pre-registration: plans/2026-10-04-navhard-dac-prereg.md.

For every navhard token: reference kinematics (PDM-Closed), each arm's plan kinematics, and for the arm's DAC failures
(scorer's own LQR + bicycle replay) the departure geometry plus four intervention checks, all geometry only:
  map_narrow   every off-polygon corner lies in polygon U nuPlan generic_drivable_areas U carpark_areas (0.1 m tolerance)
  raw_ok       the plan itself (8 poses -> 41 states, plan heading) stays inside: an ideal tracker would pass
  retime_ok    the plan path re-timed to the reference's distance-time curve stays inside under the LQR replay
  rot0_ok      the same model on history-rotation-removed inputs (al-rot0 rollout) stays inside
  feasible     some constant-curvature arc (41 curvatures x {plan, reference} distance-time curve) from the same start stays inside
               under the same LQR replay (prereg addendum 1)
Also exports the devkit post-aggregation score tables (native, best, ref) and the two-stage mapping for step 2.

  DATA_DIR=... PYTHONPATH=$DATA_DIR/third_party/navsim:$DATA_DIR/third_party/nuplan-devkit \
    $DATA_DIR/envs/navsim2/bin/python experiments/leaderboard_audit/scripts/nhdac_feat.py [--procs 64] [--limit 50]
Output: $DATA_DIR/runs/leaderboard_audit/navhard_dac/{feat.pkl, scores.pkl}
"""
import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402
import offroad_replay_cf as R  # noqa: E402

PRED = L.D / "runs/op_lb/lb_navhard/preds"
ARMS = {"native": "gimm-cinque__base.npz", "best": "gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base.npz"}
ROT0 = {"native": "gimm-cinque_al-rot0__base.npz", "best": "gimm-cinque_Oit_dw3-s0_al-rot0__base.npz"}
OUTD = L.D / "runs/leaderboard_audit/navhard_dac"
_W = {}


def _init():
    import glob
    sim, _, _, _, samp = L.setup_scoring()
    _W.update(sim=sim, samp=samp, maps={}, cp={Path(p).parent.name: p for p in glob.glob(str(L.MCACHE / "*/*/*/metric_cache.pkl"))},
              P={a: L.poses_by_token(PRED / f) for a, f in ARMS.items()}, P0={a: L.poses_by_token(PRED / f) for a, f in ROT0.items()},
              idx={e["token"]: e for e in L.index()})


def raw_layers(name):
    if name not in _W["maps"]:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        from shapely.strtree import STRtree
        a = get_maps_api(str(L.D / "datasets/navsim/maps"), "nuplan-maps-v1.0", name)
        lay = {}
        for k, ln in (("gda", "generic_drivable_areas"), ("walk", "walkways"), ("park", "carpark_areas")):
            g = [x for x in a._load_vector_map_layer(ln).geometry.values if x is not None and not x.is_empty]
            lay[k] = (STRtree(g), g)
        _W["maps"][name] = lay
    return _W["maps"][name]


def kin(d):
    """Kinematics of a dense (41, 3) trajectory relative to its own start pose."""
    h = np.unwrap(d[:, 2])
    c, s = np.cos(h[0]), np.sin(h[0])
    rel = d[:, :2] - d[0, :2]
    lat = -s * rel[:, 0] + c * rel[:, 1]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(d[:, :2], axis=0).T))]
    return dict(dpsi1=float(h[10] - h[0]), dpsi2=float(h[20] - h[0]), dpsi4=float(h[40] - h[0]), y4=float(lat[-1]), s4=float(seg[-1]),
                s1=float(seg[10]), s2=float(seg[20]))


def to_global(mc, d):
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    st = np.zeros((len(d), 11))
    st[:, 0] = o[0] + c * d[:, 0] - s * d[:, 1]
    st[:, 1] = o[1] + s * d[:, 0] + c * d[:, 1]
    st[:, 2] = d[:, 2] + o[2]
    return st


def inside_of(mc, g, states):
    cor = R.corners_of(mc, states)
    return cor, g["am"].points_in_polygons(cor[None, :, :-1, :])[g["area"]].any(axis=0)[0]     # (41, 4)


def retime(d, ref):
    """Plan path (x, y, heading) re-timed to the reference's distance-time curve."""
    sp = np.r_[0, np.cumsum(np.hypot(*np.diff(d[:, :2], axis=0).T))]
    sp = np.maximum.accumulate(sp + np.arange(len(sp)) * 1e-6)
    sr = np.r_[0, np.cumsum(np.hypot(*np.diff(ref[:, :2], axis=0).T))]
    tgt = np.minimum(sr, sp[-1])
    h = np.unwrap(d[:, 2])
    return np.stack([np.interp(tgt, sp, d[:, 0]), np.interp(tgt, sp, d[:, 1]), np.interp(tgt, sp, h)], -1)


KAP = np.linspace(-0.2, 0.2, 41)


def arcs(sd):
    """Constant-curvature arcs (len(KAP), 41, 3) along the distance profile sd (41,)."""
    k = KAP[:, None]
    z = np.abs(k) < 1e-9
    kk = np.where(z, 1.0, k)
    ks = k * sd[None]
    return np.stack([np.where(z, sd[None], np.sin(ks) / kk), np.where(z, 0.0, (1 - np.cos(ks)) / kk), ks], -1)


def feasible(mc, g, profiles):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    W = _W
    arr = [get_trajectory_as_array(transform_trajectory(Trajectory(L.poses_from_dense(a)), mc.ego_state), W["samp"], mc.ego_state.time_point)
           for sd in profiles for a in arcs(sd)]
    st = W["sim"].simulate_proposals(np.stack(arr), mc.ego_state)
    ok = np.array([inside_of(mc, g, s)[1].all() for s in st])
    return bool(ok.any()), int(ok.sum())


def dist_profile(d):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(d[:, :2], axis=0).T))]


def depart(mc, g, states, cor, inside):
    """Departure geometry of a failing replay."""
    import shapely
    from shapely.ops import unary_union
    out = ~inside
    pts = cor[:, :4][out]
    dist = shapely.distance(g["area_u"], shapely.points(pts))
    ost = ~inside.all(axis=1)
    first = int(np.flatnonzero(ost)[0])
    fd = shapely.distance(g["area_u"], shapely.points(cor[first, :4]))
    c = int(np.argmax(np.where(out[first], fd, -1)))
    h = states[first, 2]
    rel = cor[first, c] - states[first, :2]
    side = float(-np.sin(h) * rel[0] + np.cos(h) * rel[1])                    # + = left corner of the car
    cen, ptc = shapely.points(cor[first, 4]), shapely.points(cor[first, c])
    junction = bool((g["inter_u"] is not None and (shapely.distance(g["inter_u"], cen) <= 3 or shapely.distance(g["inter_u"], ptc) <= 3))
                    or (g["conn_u"] is not None and (shapely.contains(g["conn_u"], cen) or shapely.distance(g["conn_u"], ptc) < 0.01)))
    lay = raw_layers(mc.map_parameters.map_name)
    box = shapely.box(*pts.min(0) - 5, *pts.max(0) + 5)
    near = {k: [lay[k][1][i] for i in lay[k][0].query(box)] for k in lay}
    ext = unary_union([g["area_u"], *near["gda"], *near["park"]])
    mp_ = shapely.points(pts)
    map_narrow = bool((shapely.distance(ext, mp_) <= 0.1).all())
    walk = bool(near["walk"]) and bool((shapely.distance(unary_union(near["walk"]), mp_) <= 0.0).any())
    return dict(first=first, start_outside=bool(ost[0]), side=side, d_worst=float(dist.max()), d_first=float(fd.max()), junction=junction,
                map_narrow=map_narrow, walk=walk, n_out_states=int(ost.sum()))


def work(token):
    W = _W
    mc = L.load_cache(W["cp"][token])
    g = R.geoms(mc)
    ref = L.pdm_ref_ego(mc, W["samp"])
    ref[:, 2] = np.unwrap(ref[:, 2])
    e = W["idx"][token]
    out = dict(token=token, stage=e["stage"], cmd=int(np.argmax(e["cmd"][-1])), start=R.start_features(mc), ref=kin(ref), arms={})
    for a in ARMS:
        p8 = W["P"][a][token]
        d = L.dense_from_poses(p8)
        f = kin(d)
        f["end_y"] = float(p8[-1, 1])
        st = L.simulate(W["sim"], mc, p8)
        cor, ins = inside_of(mc, g, st)
        f["dac_geom"] = bool(ins.all())
        if not f["dac_geom"]:
            f.update(depart(mc, g, st, cor, ins))
            f["raw_ok"] = bool(inside_of(mc, g, to_global(mc, d))[1].all())
            f["retime_ok"] = bool(inside_of(mc, g, L.simulate(W["sim"], mc, L.poses_from_dense(retime(d, ref))))[1].all())
            f["rot0_ok"] = bool(inside_of(mc, g, L.simulate(W["sim"], mc, W["P0"][a][token]))[1].all())
            f["feasible"], f["n_feasible"] = feasible(mc, g, [dist_profile(d), dist_profile(ref)])
        out["arms"][a] = f
    out["ref_end_y"] = float(ref[-1, 1])
    return out


def export_scores():
    """Post-aggregation devkit token tables (score, sub-scores, weight) of native / best / ref, plus the mapping."""
    _, _, _, mapping, samp = L.setup_scoring()
    prep = pickle.load(open(L.D / "runs/leaderboard_audit/loss_budget/prep_navhard.pkl", "rb"))
    tokens = list(prep)
    T = {}
    for name in ["native", "best", "ref"]:
        comb, s1, s2, df = R.aggregate([prep[t]["rows"][name] for t in tokens], mapping, samp)
        keep = [c for c in df.columns if c not in ("ego_simulated_states",) and df[c].dtype != object or c == "token"]
        T[name] = df[keep].set_index("token", drop=False)
        T[name + "_official"] = dict(combined=float(comb["score"]), s1=float(s1["score"]), s2=float(s2["score"]))
        print(name, T[name + "_official"], flush=True)
    T["mapping"] = mapping
    pickle.dump(T, open(OUTD / "scores.pkl", "wb"), protocol=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-scores", action="store_true")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if not a.skip_scores and not a.limit:
        export_scores()
    tokens = [e["token"] for e in L.index()]
    if a.limit:
        tokens = tokens[::len(tokens) // a.limit][:a.limit]
    t0, res = time.time(), {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tokens, chunksize=4)):
            res[r["token"]] = r
            if i % 500 == 0:
                print(i, len(tokens), f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(res, open(OUTD / f"feat{'_dbg' if a.limit else ''}.pkl", "wb"), protocol=4)
    print("done", len(res), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
