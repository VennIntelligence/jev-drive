#!/usr/bin/env python3
"""STEER-LONG handoff replay (amendment 4, second step): AlpaSim's NonlinearMPC (as shipped, alpasim-base image) re-run through the force-GT
phase and the handoff the way the runtime drives it: steps before closed_loop_start use the logged ego trajectory (traffic_session_request,
logged object 0) as reference and the state coerced from the force-GT dynamics (rollout.asl driver_ego_trajectory dynamic states; v_cg_y =
v_y + l_rig_to_cg w, steering = atan(w L / v)); the handoff step coerces once more and switches to the driver's plan; later steps use the
logged controller states. Arm gt_ref keeps the logged ego trajectory as the reference after the handoff too (default gains): is the
jump the driver's plan or the handoff itself; arm plan_vego serves the plan's path at the ego's current speed (default gains):
is it the plan's speed. Prints the replayed and logged steering command at the handoff and after, per arm.

  docker run --rm -v /data:/data alpasim-base:0.89.0 bash -c 'cd /repo && uv run python <this> <pull dir> <events.json> <out.json>'
events.json: [{"label", "scene", "session"}] of PAI rollouts that kept their rollout.asl.
"""
import asyncio
import glob
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/data/runs/steer_long")
from sl_mpc import ARMS, load, served_local, yaw_q  # noqa: E402

FORCE_US, L, LCG = 1_700_000, 2.85, 1.59


async def read_asl(f):
    from alpasim_utils.logs import async_read_pb_log
    dyn, gt = {}, None
    async for e in async_read_pb_log(f):
        k = e.WhichOneof("log_entry")
        if k == "driver_ego_trajectory":
            m = e.driver_ego_trajectory
            for p, d in zip(m.trajectory.poses, m.dynamic_states):
                dyn[p.timestamp_us] = (d.linear_velocity.x, d.linear_velocity.y, d.angular_velocity.z, d.linear_acceleration.x)
        elif k == "traffic_session_request" and gt is None:
            tr = e.traffic_session_request.logged_object_trajectories[0].trajectory
            gt = np.array([[q.timestamp_us, q.pose.vec.x, q.pose.vec.y, yaw_q(q.pose.quat.x, q.pose.quat.y, q.pose.quat.z, q.pose.quat.w)] for q in tr.poses])
    return dyn, gt


def main():
    from alpasim_controller.mpc_controller import ControllerInput, MPCGains
    from alpasim_controller.mpc_impl.nonlinear_mpc import NonlinearMPC
    from alpasim_utils.geometry import Pose, Trajectory
    pull, evs, outf = sys.argv[1:4]
    evs = json.load(open(evs))
    res = []
    for e in evs:
        f = glob.glob(f"/data/runs/onpol_long/pai281/{e['label']}/runs/*/sim/rollouts/{e['scene']}/{e['session']}/rollout.asl")
        if not f:
            continue
        dyn, gt = asyncio.run(read_asl(f[0]))
        D, C = load(pull, e["label"], {e["session"]})
        recs = sorted(D[e["session"]], key=lambda r: r["now"])
        now = np.array([r["now"] for r in recs], np.int64)
        c = C[e["session"]]
        ts = c["timestamp_us"].astype(np.int64)
        tstart = int(ts[0]) - 100_000
        t_ho = tstart + FORCE_US
        steps = [tstart + 100_000 * i for i in range(int((t_ho - tstart) / 100_000) + 6)]
        tk = np.array(sorted(dyn))
        out = {"label": e["label"], "session": e["session"], "steps": [(s - tstart) / 1e6 for s in steps], "u_log": [], "arms": {}}
        for s in steps:
            i = int(np.searchsorted(ts, s))
            out["u_log"].append(float(c["u_steering_angle"][i]) if i < len(ts) and ts[i] == s + 100_000 else
                                (float(c["u_steering_angle"][i + 1]) if i + 1 < len(ts) else None))
        for arm, (g, it) in {**ARMS, "gt_ref": ({}, 30), "plan_vego": ({}, 30), "plan_geo": ({}, 30), "plan_vego_geo": ({}, 30)}.items():
            mpc, us = NonlinearMPC(gains=MPCGains(**g)), []
            for s in steps:
                pose_t = s
                if s <= t_ho:                                            # pose of the ego = GT during force-GT (and at the handoff)
                    x0, y0, w0 = (np.interp(s, gt[:, 0], gt[:, j]) for j in (1, 2, 3))
                    kd = tk[np.argmin(np.abs(tk - s))]
                    vx, vy, wz, ax = dyn[kd]
                    state = np.array([0, 0, 0, vx, vy + LCG * wz, wz, math.atan(wz / vx * L) if vx > 0.25 else 0.0, ax], float)
                else:
                    i = int(np.searchsorted(ts, s))
                    x0, y0, w0 = c["x"][i], c["y"][i], yaw_q(c["qx"][i], c["qy"][i], c["qz"][i], c["qw"][i])
                    state = np.array([0, 0, 0, c["vx"][i], c["vy"][i], c["wz"][i], c["front_steering_angle"][i], c["acceleration"][i]], float)
                if s < t_ho or (arm == "gt_ref"):
                    m = (gt[:, 0] >= s) & (gt[:, 0] <= s + 5_000_001)
                    tt, X, Y, W = gt[m, 0].astype(np.int64), gt[m, 1], gt[m, 2], np.unwrap(gt[m, 3])
                else:
                    j = int(np.searchsorted(now, s, side="right")) - 1
                    tt, X, Y, W = served_local(recs[j])
                    keep = tt >= recs[j]["now"]
                    tt, X, Y, W = tt[keep], X[keep], Y[keep], W[keep]
                    if arm in ("plan_geo", "plan_vego_geo"):                 # heading from the path's own geometry, not the plan's yaw output
                        W = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
                    if arm in ("plan_vego", "plan_vego_geo"):                               # the plan's path re-timed to the ego's current speed
                        sa = np.r_[0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]
                        sq = np.clip(state[3] * (tt - tt[0]) / 1e6, 0, sa[-1])
                        X, Y, W = np.interp(sq, sa, X), np.interp(sq, sa, Y), np.interp(sq, sa, W)
                        if arm == "plan_vego_geo":
                            W = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
                cc, ss = math.cos(w0), math.sin(w0)
                poses = [Pose(np.array([cc * (x - x0) + ss * (y - y0), -ss * (x - x0) + cc * (y - y0), 0.0], np.float32),
                              np.array([0, 0, math.sin((w - w0) / 2), math.cos((w - w0) / 2)], np.float32)) for x, y, w in zip(X, Y, W)]
                ref = Trajectory.from_poses(timestamps=tt.astype(np.uint64), poses=poses)
                inp = ControllerInput(state=state, reference_trajectory=ref, timestamp_us=int(s))
                if mpc._mpc is None:
                    mpc._current_reference = inp
                    mpc._setup_mpc()
                    if it != 30:
                        mpc._mpc.settings.nlpsol_opts = {"ipopt.max_iter": it}
                        mpc._mpc.setup()
                    mpc._mpc.x0 = state.reshape(-1, 1)
                    mpc._mpc.set_initial_guess()
                us.append(float(mpc.compute_control(inp).control[0]))
            out["arms"][arm] = us
        res.append(out)
        k = out["steps"].index(1.7)
        print(e["label"], e["session"][:8], "handoff u_log", round(out["u_log"][k], 3), {a: round(v[k], 3) for a, v in out["arms"].items()},
              "next", round(out["u_log"][k + 1] or 0, 3), {a: round(v[k + 1], 3) for a, v in out["arms"].items()}, flush=True)
    json.dump(res, open(outf, "w"))


if __name__ == "__main__":
    main()
