"""Brake spin probe: the human driver's straight braking plan against AlpaSim's own controller (nonlinear MPC + vehicle model,
alpasim_controller.system.System, the controller service's code and config) in a closed loop without renderer or physics.

Each case cruises straight at --v for 3 s (plan accel --pre), then brakes: brake input b(t) = min(1, (t - t_b) / ramp), plan accel
a = -cap * b - drag(v) (drag = lib/human_driver.py's 0.25 + 0.0008 v^2, --drag 0 drops it), the plan is lib/human_driver.py's
kinematic bicycle rollout (41 poses, 0.1 s, inputs held, v clamped at 0), re-planned every 0.1 s from the current pose and speed,
as human_driver.drive does. The request mirrors alpasim_runtime/events/controller.py (rig-frame reference, coerce only on the first
step, 0.1 s control step, pose reporting 0.1 s). --noise perturbs the fed-back pose each step (yaw N(0, noise mrad), lateral
N(0, noise cm)), a stand-in for the physics service's ground correction: without it the problem is exactly symmetric and the MPC
never steers. Per case: peak |yaw rate| in the brake phase (the plan's yaw rate is 0), peak
|heading|, peak |lateral offset| (and their signs: + = left, the rig frame), peak |steering command|, IPOPT solves that did not succeed. spin = peak |yaw rate| > 0.5 rad/s
or |heading| > 0.2 rad; kink (the "drift") = peak |yaw rate - plan yaw rate| > 0.1 rad/s or |heading error vs the plan| > 0.03 rad.

--gate -5 passes the plan through lib/human_driver.py's HUMAN_TRACK_GATE path (track_ref gate, amin -5) with the measured acceleration.
stop_m = distance from the brake onset to the stop (speed integral).

Runs inside alpasim-base (needs alpasim_controller, alpasim_grpc, do_mpc), CPU only:
    docker run --rm --network none --cpus 6 -v $SRC/src:/repo/src -v $SRC/plugins:/repo/plugins -v $REPO:$REPO -v $OUT:$OUT \
      --entrypoint /repo/.venv/bin/python alpasim-base:0.89.0 $REPO/experiments/alpasim_human/scripts/brake_probe.py --out $OUT
Writes <out>/cases.jsonl (one summary per case) and <out>/trace_<case>.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ["HUMAN_TRACK_GATE"] = "1"                     # import the gate; each case decides whether it is used
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from human_driver import gate_ref, rollout  # noqa: E402  the human driver's own plan and its HUMAN_TRACK_GATE path

DT_US = 100_000


def run_case(c: dict) -> dict:
    import do_mpc
    from alpasim_controller.mpc_controller import ControllerConfig
    from alpasim_controller.system import create_system
    from alpasim_grpc.v0 import common_pb2, controller_pb2

    if c["max_iter"] != 30:                              # mechanism check only: the shipped solver budget is 30
        setup = do_mpc.controller.MPC.setup
        def patched(self, *a, **k):
            self.settings.nlpsol_opts = {**self.settings.nlpsol_opts, "ipopt.max_iter": c["max_iter"]}
            return setup(self, *a, **k)
        do_mpc.controller.MPC.setup = patched
    t, v = 10**12, c["v"]
    init = common_pb2.StateAtTime(timestamp_us=t, pose=common_pb2.Pose(vec=common_pb2.Vec3(), quat=common_pb2.Quat(w=1.0)),
                                  state=common_pb2.DynamicState(linear_velocity=common_pb2.Vec3(x=v)))
    cfg = ControllerConfig(mpc_implementation="nonlinear")
    for kv in filter(None, c["gains"].split(";")):                # mechanism check only: the shipped gains are the defaults
        k, v_ = kv.split("=")
        setattr(cfg.gains, k, type(getattr(cfg.gains, k))(float(v_)))
    sysm = create_system(f"{c['out']}/ctl_{c['name']}.csv", init, cfg)
    ctl, stats = sysm._controller, []
    orig = ctl.compute_control
    def wrapped(inp):
        o = orig(inp)
        s = ctl._mpc.solver_stats if ctl._mpc is not None else {}
        stats.append((bool(s.get("success", False)), int(s.get("iter_count", -1)), str(s.get("return_status", o.status))))
        return o
    ctl.compute_control = wrapped
    pose = init.pose
    rng = np.random.default_rng(c["seed"])
    tb, rows, n_bad_brake, a0 = t + 3_000_000, [], 0, 0.0
    L = 2.9
    while True:
        tt = (t - tb) / 1e6
        if tt < 0:
            a, b = c["pre"], 0.0
        else:
            b = min(1.0, tt / c["ramp"]) if c["ramp"] > 0 else 1.0
            a = -c["cap"] * b - (0.25 + 8e-4 * v * v) * (v > 0.05) * c["drag"]
        plan = rollout(0.0, 0.0, 0.0, v, c["delta"], a, L)          # rig frame of the current pose: the plan starts at the car
        if c["gate"] != "off":                                     # HUMAN_TRACK_GATE=1, deceleration bound amin = c["gate"]
            g = [float(x) for x in c["gate"].split(":")]           # amin[:jlon]
            plan = gate_ref(plan, 0.0, 0.0, 0.0, v, a0, amin=g[0], **({"jlon": g[1]} if len(g) > 1 else {}))
        traj = common_pb2.Trajectory(poses=[common_pb2.PoseAtTime(timestamp_us=t + i * DT_US, pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=float(p[0]), y=float(p[1]), z=0.0),
            quat=common_pb2.Quat(w=math.cos(p[2] / 2), z=math.sin(p[2] / 2)))) for i, p in enumerate(plan)])
        req = controller_pb2.RunControllerAndVehicleModelRequest(session_uuid="probe", future_time_us=t + DT_US,
                                                                 coerce_dynamic_state=not rows, pose_reporting_interval_us=DT_US)
        req.state.timestamp_us = t
        req.state.pose.CopyFrom(pose)
        req.planned_trajectory_in_rig.CopyFrom(traj)
        req.state.state.linear_velocity.x = v
        n0 = len(stats)
        out = sysm.run_controller_and_vehicle_model(req).states[-1]
        pose, t = out.pose_local_to_rig, out.timestamp_us
        if c["noise"] > 0:                               # stand-in for the physics service's ground correction of the pose
            dyaw = rng.normal(0, c["noise"] * 1e-3)
            q = pose.quat
            q.w, q.z = (q.w * math.cos(dyaw / 2) - q.z * math.sin(dyaw / 2)), (q.z * math.cos(dyaw / 2) + q.w * math.sin(dyaw / 2))
            pose.vec.y += rng.normal(0, c["noise"] * 1e-2)
        d = out.dynamic_state
        v, a0 = math.hypot(d.linear_velocity.x, d.linear_velocity.y), d.linear_acceleration.x
        q = pose.quat
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        ok = all(s[0] for s in stats[n0:])
        n_bad_brake += (not ok) and tt >= 0
        rows.append(dict(t=round(tt + 0.1, 2), v=v, b=b, a_plan=a, wz=d.angular_velocity.z, yaw=yaw, y=pose.vec.y,
                         u_steer=float(sysm.control_input[0]), u_acc=float(sysm.control_input[1]),
                         ok=ok, iters=stats[-1][1] if stats else -1, status=stats[-1][2] if stats else ""))
        if (tt >= 0 and v < 0.3) or tt > 8.0:
            break
    br = [r for r in rows if r["t"] > 0]
    with open(f"{c['out']}/trace_{c['name']}.csv", "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(round(x, 5)) if isinstance(x, float) else str(x) for x in r.values()) + "\n")
    pre = [r for r in rows if r["t"] <= 0 and r["t"] > -1.0]
    res = {k: c[k] for k in ("name", "v", "cap", "ramp", "drag", "pre", "max_iter", "noise", "delta", "seed", "gains", "gate")}
    res.update(peak_wz=max(abs(r["wz"]) for r in br), peak_yaw=max(abs(r["yaw"]) for r in br), peak_y=max(abs(r["y"]) for r in br),
               peak_steer=max(abs(r["u_steer"]) for r in br), min_uacc=min(r["u_acc"] for r in br), bad_solves=n_bad_brake,
               n_brake=len(br), pre_bad=sum(not r["ok"] for r in pre), stop_s=br[-1]["t"] if br[-1]["v"] < 0.3 else None,
               t_peak_wz=max(br, key=lambda r: abs(r["wz"]))["t"],
               y_signed=max(br, key=lambda r: abs(r["y"]))["y"], yaw_signed=max(br, key=lambda r: abs(r["yaw"]))["yaw"],
               y_end=br[-1]["y"], yaw_end=br[-1]["yaw"], y_1s=next((r["y"] for r in br if r["t"] >= 1.0), br[-1]["y"]),
               yaw_1s=next((r["yaw"] for r in br if r["t"] >= 1.0), br[-1]["yaw"]))
    exc = [r["wz"] - r["v"] * math.tan(c["delta"]) / L for r in br]         # yaw rate minus the plan's
    he = np.cumsum(exc) * 0.1                                                 # heading error vs the plan, rad
    k = int(np.argmax(np.abs(exc)))
    res.update(exc_peak=exc[k], head_err_peak=float(he[np.argmax(np.abs(he))]), head_err_1s=float(he[min(9, len(he) - 1)]))
    x_stop = None
    if br[-1]["v"] < 0.3:                                        # distance from the brake onset to the stop, along the road
        x_stop = sum(0.5 * (p["v"] + q["v"]) * 0.1 for p, q in zip([rows[len(rows) - len(br) - 1]] + br[:-1], br))
    res.update(stop_m=x_stop, v_onset=rows[len(rows) - len(br) - 1]["v"])
    res["spin"] = res["peak_wz"] > 0.5 or res["peak_yaw"] > 0.2
    res["kink"] = abs(res["exc_peak"]) > 0.1 or abs(res["head_err_peak"]) > 0.03
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--v", default="15,25,33")
    ap.add_argument("--cap", default="8,6,5,4")
    ap.add_argument("--ramp", default="0,0.3,0.6")
    ap.add_argument("--drag", default="1")
    ap.add_argument("--pre", default="0")
    ap.add_argument("--max-iter", default="30")
    ap.add_argument("--noise", default="1", help="pose noise per step: N(0, noise mrad) yaw, N(0, noise cm) lateral")
    ap.add_argument("--delta", default="0", help="plan road-wheel angle, rad")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--gains", default="", help="MPCGains overrides, k=v;k=v, several sets separated by '|'")
    ap.add_argument("--gate", default="off", help="off | track_ref amin[:jlon] values, e.g. off,-5,-8:20")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fl = lambda s: [float(x) for x in s.split(",")]
    cases = [dict(v=v, cap=cap, ramp=r, drag=dr, pre=pre, max_iter=int(mi), noise=nz, delta=de, seed=int(sd), gains=gs, gate=gt, out=a.out,
                  name=f"v{v:g}_c{cap:g}_r{r:g}_d{dr:g}_p{pre:g}_i{int(mi)}_n{nz:g}_s{de:g}_r{int(sd)}" + (f"_g{gi}" if gs else "") + (f"_gate{gt}" if gt != "off" else ""))
             for v, cap, r, dr, pre, mi, nz, de, sd, (gi, gs), gt in itertools.product(
                 fl(a.v), fl(a.cap), fl(a.ramp), fl(a.drag), fl(a.pre), fl(a.max_iter), fl(a.noise), fl(a.delta), fl(a.seed),
                 enumerate(a.gains.split("|")), a.gate.split(","))]
    with mp.get_context("spawn").Pool(a.j) as pool, open(f"{a.out}/cases.jsonl", "a") as f:
        for res in pool.imap_unordered(run_case, cases):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(json.dumps(res), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
