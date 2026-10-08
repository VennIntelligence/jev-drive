"""AlpaSim's route for NAVSIM tokens, rebuilt from the NAVSIM logs with AlpaSim's own code (nothing of theirs is rewritten).

AlpaSim (0bb4c4b) builds a rollout's route once, in runtime/route_generator.RouteGeneratorMap: the 5.5 s recorded ego track of the scene
(56 poses at 0.1 s; a scene `<log>-<token>` runs from the token's t0 - 1.5 s to t0 + 4 s) projected onto lane centres of the trajdata
nuPlan map, extended past the end of the recording along the lane graph (successor with the closest heading), and at every decision
resampled from the ego's projection: 20 waypoints 4.21 m apart, those before 40 m dropped, NaN-padded to 20, in the rig frame.

Here the same class runs on a token's track taken from the NAVSIM 2 Hz logs (12 frames, cubic spline to 0.1 s) and the same cached trajdata
map ($DATA_DIR/datasets/alpasim_nuplan/nuplan_test/maps, moved to a per-city local frame with TrajdataDataSource's own transform). A token
used as decision k of a rollout belongs to the scene that starts k frames before it, so its route depends on k (where the recording ends):

  build  per op_parity cache dir, routes for 4 variants j = 0..3 (m = j + 1 keyframes): k = 0, 1, 2 and, for m = 4, k = 3 (--k4 fixed: the
         AlpaSim scene of that token itself) or 3 + crc32(token) % 7 (--k4 hash: one of the decisions 3..9, navtrain)
         -> $DATA_DIR/runs/alpasim/ap2/route/<data>.npz: names, wp (N, 4, 20, 2) rig-frame waypoints (NaN = padding / no route), ok (N, 4), k (N, 4)
  check  against a tapped run (run.sh --tap): every tapped route message vs the rebuild at the tapped pose (the polyline) and at the log
         pose of that decision (what training uses), waypoint distance and command agreement -> <out>/route_check.json

  $DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim/scripts/ap2_route.py build --data lb_navtest --k4 fixed
"""
import argparse
import copy
import json
import pickle
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_R), str(_R / "experiments/alpasim/lib")]
import ap2_inputs as AI  # noqa: E402
from jevdrive import cache, par  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

LOC = {"us-nv-las-vegas-strip": "las_vegas", "us-ma-boston": "boston", "us-pa-pittsburgh-hazelwood": "pittsburgh", "sg-one-north": "singapore"}
N_FR, N_PT, OFFSET_M = 12, 56, 40.0                 # frames of a scene at 2 Hz, its poses at 0.1 s, route_start_offset_m (base.yaml)
_MAP, _LOG = {}, {}


def out_root(*p) -> Path:
    d = data_dir() / "runs/alpasim/ap2" / Path(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


class _Traj:
    positions = np.zeros((1, 3))

    def __len__(self):
        return 1


def city_map(loc: str):
    """The cached trajdata nuPlan map of a city in a local frame (origin = its first lane point, rounded to 1 km), prepared with the steps
    of TrajdataDataSource._load_map_from_map_api -> (VectorMap, origin xy)."""
    if loc not in _MAP:
        from alpasim_utils.trajdata_data_source import TrajdataDataSource as TDS
        from trajdata.maps import MapAPI
        vm = copy.deepcopy(MapAPI(data_dir() / "datasets/alpasim_nuplan").get_map(f"nuplan_test:{LOC[loc]}", incl_road_lanes=True, incl_road_areas=False,
                                                                                 incl_ped_crosswalks=False, incl_ped_walkways=False))
        org = np.round(np.asarray(vm.lanes[0].center.points[0, :2], np.float64) / 1000.0) * 1000.0
        T = np.eye(4)
        T[:2, 3] = -org
        stub = SimpleNamespace(_rig=SimpleNamespace(world_to_nre=T), rig=SimpleNamespace(trajectory=_Traj()))
        stub._transform_map_points = lambda *a: TDS._transform_map_points(stub, *a)
        TDS._apply_coordinate_transform_to_map(stub, vm)
        vm.__post_init__()
        vm.compute_search_indices()
        TDS._fix_map_datatypes(stub, vm)
        _MAP[loc] = (vm, org)
    return _MAP[loc]


def load_log(split_dir: str, log: str) -> tuple:
    """-> (frames: token, time (s), position (n, 3), yaw (n,), map location; token -> frame index) of a NAVSIM log."""
    k = (split_dir, log)
    if k not in _LOG:
        _LOG.clear()                                # one log in memory per worker
        with open(data_dir() / "datasets/navsim/navsim_logs" / split_dir / f"{log}.pkl", "rb") as f:
            L = pickle.load(f)
        q = np.array([x["ego2global_rotation"] for x in L], np.float64)
        fr = SimpleNamespace(t=np.array([x["timestamp"] for x in L], np.float64) * 1e-6, p=np.array([x["ego2global_translation"] for x in L], np.float64),
                             yaw=np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)), loc=L[0]["map_location"],
                             cmd=np.array([np.argmax(x["driving_command"]) for x in L]))
        _LOG[k] = (fr, {x["token"]: i for i, x in enumerate(L)})
    return _LOG[k]


def scene_track(fr, i0: int):
    """The 5.5 s track of the scene whose first frame is i0 -> (56, 3) world positions at 0.1 s, or None when the log does not hold its 12
    frames 0.5 s apart."""
    from scipy.interpolate import CubicSpline
    if i0 < 0 or i0 + N_FR > len(fr.t):
        return None
    t = fr.t[i0:i0 + N_FR] - fr.t[i0]
    if np.abs(np.diff(t) - 0.5).max() > 0.05:
        return None
    return CubicSpline(t, fr.p[i0:i0 + N_FR])(np.arange(N_PT) * 0.1)


def generator(fr, i0: int):
    """AlpaSim's RouteGeneratorMap of the scene starting at frame i0 -> (generator, local-frame offset (3,)); None when the scene cannot
    exist or their generator raises (off-map / fold-back scenes fail in AlpaSim as well)."""
    from alpasim_runtime.route_generator import RouteGeneratorMap
    P = scene_track(fr, i0)
    if P is None:
        return None
    vm, org = city_map(fr.loc)
    off = np.r_[org, P[0, 2]]
    try:
        return RouteGeneratorMap((P - off).astype(np.float32), vm, route_start_offset_m=OFFSET_M), off
    except Exception:                               # noqa: BLE001  ValueError (sanity checks) and map look-up failures
        return None


def route_at(g, xyz_local, yaw: float, ts: int = 0) -> np.ndarray:
    """Their generate_route + prepare_for_policy at a rig pose in the generator's local frame -> (20, 2) rig-frame waypoints, NaN-padded."""
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    pose = Pose(np.asarray(xyz_local, np.float32), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32))
    return np.asarray(RouteGenerator.prepare_for_policy(g.generate_route(ts, pose)).waypoints, np.float32)[:, :2]


def _work(u):
    """One log: (split dir, log, [(row, token)], k4 mode) -> rows, wp (n, 4, 20, 2), ok (n, 4), k (n, 4)."""
    import logging
    logging.disable(logging.WARNING)                # "Unable to find a preferable lane id" once per ambiguous waypoint
    split_dir, log, items, k4 = u
    fr, pos = load_log(split_dir, log)
    wp = np.full((len(items), 4, AI.N_WP, 2), np.nan, np.float32)
    ok, kk = np.zeros((len(items), 4), bool), np.zeros((len(items), 4), np.int8)
    for n, (_, tok) in enumerate(items):
        i = pos.get(tok)
        if i is None:
            continue
        kk[n] = (0, 1, 2, 3 if k4 == "fixed" else 3 + zlib.crc32(tok.encode()) % 7)
        for j, k in enumerate(kk[n]):
            g = generator(fr, i - int(k))
            if g is None:
                continue
            try:
                wp[n, j], ok[n, j] = route_at(g[0], fr.p[i] - g[1], fr.yaw[i]), True
            except Exception:                       # noqa: BLE001
                pass
    return np.array([r for r, _ in items]), wp, ok, kk


def cmd_build(a):
    tab = np.load(data_dir() / "runs/op_parity/cache" / a.data / "tab.npz")
    names, logs = tab["names"].tolist(), tab["log"].tolist()
    split_dir = "test" if (data_dir() / "datasets/navsim/navsim_logs/test" / f"{logs[0]}.pkl").exists() else "trainval"
    by = {}
    for r, (t, lg) in enumerate(zip(names, logs)):
        by.setdefault(lg, []).append((r, t))
    with Run("alpasim", f"ap2-route-{a.data}", config=vars(a)) as run:
        def make():
            res = par.pmap(_work, [(split_dir, lg, it, a.k4) for lg, it in sorted(by.items())], run=run, desc="logs", workers=a.workers or None)
            res.raise_if_failed()
            wp = np.full((len(names), 4, AI.N_WP, 2), np.nan, np.float32)
            ok, kk = np.zeros((len(names), 4), bool), np.zeros((len(names), 4), np.int8)
            for rows, w, o, k in res.values:
                wp[rows], ok[rows], kk[rows] = w, o, k
            return dict(names=np.array(names), wp=wp, ok=ok, k=kk)
        z = cache.cached(out_root("route", f"{a.data}.npz"), cache.key(params=dict(data=a.data, k4=a.k4, n=len(names), v="r1"), code=[_work, generator, route_at, city_map]),
                         make, force=a.force)
        cr, cn = np.argmax(AI.route_cmd(z["wp"][:, 3]), -1), np.argmax(tab["cmd"][:, -1], -1)
        v = z["ok"][:, 3]
        conf = [[int(((cn == i) & (cr == j) & v).sum()) for j in range(3)] for i in range(4)]
        run.summary |= {"n": len(names), "ok_frac": z["ok"].mean(0).round(4).tolist(), "cmd_agree_m4": float((cn == cr)[v].mean()),
                        "confusion_navsim_rows_route_cols": conf, "n_wp_mean": float((~np.isnan(z["wp"][:, 3, :, 0]))[v].sum(1).mean())}
        run.info(json.dumps(run.summary))


def cmd_check(a):
    run = Path(a.run)
    S = {}
    for line in open(run / "driver_tap.jsonl"):
        r = json.loads(line)
        if r["method"] == "start_session":
            S[r["session"]] = {"scene": r["request"]["debug_info"]["scene_id"], "pose": {}, "route": {}}
        elif r["method"] == "submit_egomotion_observation":
            p = r["last_pose"]["pose"]
            S[r["session"]]["pose"][int(r["ts_us"][-1])] = ([p["vec"].get(c, 0.0) for c in "xyz"], 2 * np.arctan2(p["quat"].get("z", 0.0), p["quat"].get("w", 1.0)))
        elif r["method"] == "submit_route":
            S[r["session"]]["route"][int(r["timestamp_us"])] = np.array(r["waypoints"], np.float32).reshape(-1, 3)[:, :2]
    rows = []
    for s in S.values():
        log, tok = s["scene"].rsplit("-", 1)
        fr, pos = load_log("test", log)
        i = pos[tok]
        g = generator(fr, i - 3)
        for k, ts in enumerate(sorted(s["route"])):
            tap = s["route"][ts]
            rec = {"scene": s["scene"], "k": k, "tap_n": int((~np.isnan(tap[:, 0])).sum()), "tap_cmd": int(np.argmax(AI.route_cmd(tap))), "built": g is not None,
                   "navsim_cmd": int(fr.cmd[i - 3 + k])}
            if g is not None:
                xyz, yaw = s["pose"][ts]
                first = fr.p[i - 3] - g[1]                                  # the rollout's local origin (its first pose) in the generator's frame
                for name, w in (("pose", route_at(g[0], np.asarray(xyz) + first, yaw)), ("log", route_at(g[0], fr.p[i - 3 + k] - g[1], fr.yaw[i - 3 + k]))):
                    both = ~np.isnan(tap[:, 0]) & ~np.isnan(w[:, 0])
                    d = np.linalg.norm(tap[both] - w[both], axis=1)
                    rec |= {f"{name}_n": int((~np.isnan(w[:, 0])).sum()), f"{name}_cmd": int(np.argmax(AI.route_cmd(w))),
                            f"{name}_d_mean": float(d.mean()) if both.any() else None, f"{name}_d_max": float(d.max()) if both.any() else None}
            rows.append(rec)
    B = [r for r in rows if r["built"]]
    f = lambda key, sel=lambda r: True: [r[key] for r in B if sel(r) and r.get(key) is not None]  # noqa: E731
    summ = {"sessions": len(S), "messages": len(rows), "rebuilt": len(B),
            "at_tapped_pose": {"d_mean": float(np.mean(f("pose_d_mean"))), "d_p95": float(np.quantile(f("pose_d_mean"), 0.95)), "d_max": float(np.max(f("pose_d_max"))),
                               "same_n": float(np.mean([r["pose_n"] == r["tap_n"] for r in B])), "cmd_agree": float(np.mean([r["pose_cmd"] == r["tap_cmd"] for r in B]))},
            "at_log_pose_k01": {"d_mean": float(np.mean(f("log_d_mean", lambda r: r["k"] < 2))), "d_max": float(np.max(f("log_d_max", lambda r: r["k"] < 2))),
                                "cmd_agree": float(np.mean([r["log_cmd"] == r["tap_cmd"] for r in B if r["k"] < 2]))},
            "at_log_pose_all_k": {"d_mean": float(np.mean(f("log_d_mean"))), "cmd_agree": float(np.mean([r["log_cmd"] == r["tap_cmd"] for r in B])),
                                  "cmd_agree_by_k": [float(np.mean([r["log_cmd"] == r["tap_cmd"] for r in B if r["k"] == k])) for k in range(10)]},
            "tap_cmd_vs_navsim_driving_command": {"agree_all_k": float(np.mean([r["tap_cmd"] == r["navsim_cmd"] for r in rows])),
                                                  "agree_k3": float(np.mean([r["tap_cmd"] == r["navsim_cmd"] for r in rows if r["k"] == 3]))},
            "tap_cmd_counts_LSRU": np.bincount([r["tap_cmd"] for r in rows], minlength=4).tolist(), "tap_n_waypoints_mean": float(np.mean([r["tap_n"] for r in rows]))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "route_check.json").write_text(json.dumps({"run": str(run), "summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True, help="op_parity cache dir (tab.npz: names, log)")
    b.add_argument("--k4", default="hash", choices=["hash", "fixed"])
    b.add_argument("--workers", type=int, default=0)
    cli_args(b)
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
