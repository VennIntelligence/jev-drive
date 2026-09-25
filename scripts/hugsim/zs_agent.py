#!/usr/bin/env python
"""HUGSIM agent process for the zero-shot exam (todos/2026-09-25-hugsim-exam): speaks closed_loop.py's FIFO
protocol and forwards every step to a resident policy server (scripts/hugsim_zs_server.py) over a unix socket.
Adapter geometry: jevdrive.hugsim_zs. Configuration comes from the environment (set by scripts/hugsim/zs_run.py):

  HUGSIM_ZS_MODEL    alpamayo | cinque | lebowski
  HUGSIM_ZS_SOCKET   the server's unix socket
  HUGSIM_ZS_CAMYAML  configs/sim/<dataset>_camera.yaml (for cam_rect)
  HUGSIM_ZS_DATASET  nuscenes | waymo | kitti360 | pandaset
  HUGSIM_ZS_OPTS     JSON: nav (Alpamayo nav text, default true), desire (openpilot desire, default true),
                     traffic ([1, 0] right-hand / [0, 1] left-hand), op_clock (openpilot frame synthesis: "dilate",
                     default, one 4 Hz frame per 0.2 s context step, model clock 1.25x fast; "hold", Cinque only, the
                     20 Hz clock at face value with every frame held 5 steps), dilation (default 1.25),
                     warmup_s (openpilot, model seconds of the first frame before the first plan, default 5),
                     replan (Alpamayo: plan every k-th step, re-issue the last plan in between, default 1),
                     rigid (Alpamayo rear -> camera: rigid body, default true), dump_every (npz every k steps),
                     engage_s (engage while rolling: for the first engage_s simulated seconds the privileged route
                     follower of agent_client.py drives at <= 3 m/s while the model already runs on every frame;
                     default 0; a value past the episode length is shadow mode: the route follower drives the whole
                     episode and the model's plans are only logged), oracle_vmax (route follower speed cap, default 3),
                     forward_only (jevdrive.hugsim_zs.forward_only on every model plan, default true)

Per scenario it writes <output>/zs_steps.jsonl (one line per step: ego state, command, model input summary, the
model's own trajectory, the plan sent, timings) and optional <output>/zs_dump/<step>.npz (model inputs + plans).
"""
import argparse
import json
import os
import pickle
import socket
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "hugsim")]
import zeroshot_wire as wire  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402

OP_T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
OP_CTX_S = 0.2                                     # openpilot context step (5 Hz)


class Agent:
    def __init__(self, model, sock_path, cam_yaml, dataset, opts, out):
        self.model, self.opts, self.out = model, opts, Path(out)
        self.dataset = dataset
        self.rect = Z.rect_matrix(cam_yaml)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)
        wire.send(self.sock, {"cmd": "reset"}, {})
        self.server = wire.recv(self.sock)[0].get("server", {})
        self.hist = Z.History()
        self.frames = []                           # Alpamayo: (t, views) at the simulator's 4 Hz
        self.last = None                           # (t, world points, abs times) of the last model plan
        self.step = 0
        self.seed0 = zlib.crc32(str(self.out.name).encode()) & 0x7FFFFFF
        self.log = open(self.out / "zs_steps.jsonl", "w", buffering=1)
        self.dump_every = int(opts.get("dump_every", 0))
        self.engage_s = float(opts.get("engage_s", 0))
        if self.engage_s > 0:
            from agent_client import RoutePolicy
            self.oracle = RoutePolicy(os.environ["HUGSIM_SCENE_DIR"], v_max=float(opts.get("oracle_vmax", 3.0)),
                                      a_max=1.5)
        if self.dump_every:
            (self.out / "zs_dump").mkdir(exist_ok=True)

    def setup(self, info):
        cams = Z.CAMS if self.dataset in ("nuscenes", "pandaset") else Z.CAMS[:3]   # Waymo / KITTI-360: black backs
        cal = Z.calibs(info["cam_params"], self.rect)
        self.d = Z.rear_offset(info["cam_params"])
        if self.model == "alpamayo":
            self.alp = Z.AlpamayoViews(cal, cams)
            cov = dict(zip(("cross_left", "front_wide", "cross_right", "front_tele"), self.alp.coverage))
        else:
            self.op = Z.OpenpilotFrames(cal)
            cov = {"coverage": self.op.coverage, "sources": self.op.src_frac}
        self.log.write(json.dumps({"setup": True, "model": self.model, "server": self.server, "opts": self.opts,
                                   "rear_offset": self.d, "coverage": cov}) + "\n")

    def call(self, meta, arrays):
        t = time.perf_counter()
        wire.send(self.sock, dict(meta, cmd="plan"), arrays)
        info, out = wire.recv(self.sock)
        info["rtt_ms"] = 1e3 * (time.perf_counter() - t)
        return info, out

    # ---- Alpamayo: 4 frames at 10 Hz from the 4 Hz renders, nearest in time (pre-registered hold rule)
    def alpamayo(self, obs, info, rec):
        t0 = float(info["timestamp"])
        self.frames.append((t0, self.alp.render(obs["rgb"])))
        self.frames = self.frames[-4:]
        ts = np.array([f[0] for f in self.frames])
        slots = [int(np.argmin(np.abs(ts - (t0 - 0.1 * k)) - 1e-6 * ts)) for k in (3, 2, 1, 0)]   # ties -> newer
        frames = np.stack([self.frames[i][1] for i in slots], 1)          # (4 cams, 4 frames, 3, 320, 576)
        hx, hr = self.hist.rear_hist(self.d)
        nav = Z.NAV_TEXT[int(info["command"])] if self.opts.get("nav", True) else None
        r, out = self.call({"nav_text": nav, "seed": self.seed0 + self.step}, {"frames": frames, "hist_xyz": hx,
                                                                              "hist_rot": hr})
        plan = Z.alpamayo_to_plan(out["xyz"], out["yaw"], self.d, rigid=self.opts.get("rigid", True))
        rec.update(nav=nav, slots_dt=[round(t0 - ts[i], 3) for i in slots], cot=r.get("cot"),
                   infer_ms=r.get("infer_ms"), rtt_ms=r["rtt_ms"],
                   model_xy=np.round(out["xyz"][4::5, :2], 3).tolist(), model_yaw=np.round(out["yaw"][4::5], 4).tolist())
        if self.dump_every and self.step % self.dump_every == 0:
            np.savez_compressed(self.out / "zs_dump" / f"{self.step:04d}.npz", frames=frames, hist_xyz=hx,
                                xyz=out["xyz"], yaw=out["yaw"], plan=plan, rgb_front=obs["rgb"]["CAM_FRONT"])
        return plan

    # ---- openpilot: one simulator step = one 0.2 s context step (clock dilated by 1.25)
    def openpilot(self, obs, info, rec):
        img2 = self.op.pack(obs["rgb"])
        hold = self.opts.get("op_clock", "dilate") == "hold" and self.model != "lebowski"
        if hold:            # 20 Hz clock at face value: each 4 Hz frame held for 5 model steps (0.25 s)
            per_ctx, dil = 5, 1.0
            reps = 5 if self.step else int(round(20 * self.opts.get("warmup_s", 5.0)))
        else:               # one simulator step = one 0.2 s context step; Lebowski steps at the context rate
            per_ctx = 1 if self.model == "lebowski" else 4
            dil = float(self.opts.get("dilation", 1.25))
            reps = per_ctx if self.step else per_ctx * int(round(self.opts.get("warmup_s", 5.0) / OP_CTX_S))
        desire = Z.DESIRE[int(info["command"])] if self.opts.get("desire", True) else 0
        r, out = self.call({"desire": desire, "reps": reps, "traffic": self.opts.get("traffic", [1, 0]),
                            "speed": float(info["ego_velo"]) * dil}, {"img2": img2})
        plan = Z.openpilot_to_plan(out["pos"], out["t"], dil)
        rec.update(desire=desire, reps=reps, infer_ms=r.get("infer_ms"), rtt_ms=r["rtt_ms"],
                   lead_prob=r.get("lead_prob"), engaged=r.get("engaged"),
                   model_pos=np.round(out["pos"][[4, 8, 12, 16, 20, 24, 32], :2], 3).tolist(),
                   model_v=np.round(out["vel"][[0, 8, 16, 24]], 3).tolist())
        if self.dump_every and self.step % self.dump_every == 0:
            np.savez_compressed(self.out / "zs_dump" / f"{self.step:04d}.npz", img2=img2, pos=out["pos"],
                                plan=plan, rgb_front=obs["rgb"]["CAM_FRONT"])
        return plan

    def __call__(self, obs, info):
        t0 = time.perf_counter()
        if self.step == 0:
            self.setup(info)
        self.hist.add(info)
        pos, th = Z.ego_pose2d(info)
        rec = {"step": self.step, "t": info["timestamp"], "pos": np.round(pos, 3).tolist(), "theta": round(th, 5),
               "v": round(float(info["ego_velo"]), 3), "steer": round(float(info["ego_steer"]), 4),
               "cmd": int(info["command"]), "n_obj": len(info.get("obj_boxes", []))}
        k = int(self.opts.get("replan", 1))
        if self.model == "alpamayo" and k > 1 and self.step % k and self.last is not None:
            # re-issue the last plan: its world track at the times of this step's waypoints (straight extension)
            tq = info["timestamp"] + Z.plan_times()
            w, ta = self.last
            wt = np.stack([np.interp(tq, ta, w[:, j]) for j in (0, 1)], -1)
            over = tq > ta[-1]
            if over.any():
                vel = (w[-1] - w[-2]) / (ta[-1] - ta[-2])
                wt[over] = w[-1] + (tq[over] - ta[-1])[:, None] * vel
            plan = Z.world_to_plan(wt, pos, th)
            rec["reissued"] = True
        else:
            plan = self.alpamayo(obs, info, rec) if self.model == "alpamayo" else self.openpilot(obs, info, rec)
            if self.opts.get("forward_only", True):
                fwd = Z.forward_only(plan)
                if not np.allclose(fwd, plan):
                    rec["raw_plan"] = np.round(plan, 3).tolist()
                plan = fwd
            ta = info["timestamp"] + np.r_[0.0, Z.plan_times()]
            self.last = (Z.plan_to_world(np.r_[[[0.0, 0.0]], plan], pos, th), ta)
        if self.engage_s > 0 and info["timestamp"] < self.engage_s - 1e-6:
            rec["model_plan"] = np.round(plan, 3).tolist()
            plan = np.asarray(self.oracle(obs, info), np.float64)
            self.last = None
            rec["oracle"] = True
        rec["plan"] = np.round(plan, 3).tolist()
        rec["agent_ms"] = round(1e3 * (time.perf_counter() - t0), 1)
        self.log.write(json.dumps(rec) + "\n")
        self.step += 1
        return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    env = os.environ
    agent = Agent(env["HUGSIM_ZS_MODEL"], env["HUGSIM_ZS_SOCKET"], env["HUGSIM_ZS_CAMYAML"], env["HUGSIM_ZS_DATASET"],
                  json.loads(env.get("HUGSIM_ZS_OPTS", "{}")), a.output)
    obs_pipe, plan_pipe = (os.path.join(a.output, n) for n in ("obs_pipe", "plan_pipe"))
    for p in (obs_pipe, plan_pipe):
        if not os.path.exists(p):
            os.mkfifo(p)
    print(f"zs agent ready: {agent.model} server={agent.server}", flush=True)
    t_wait = 0.0
    while True:
        t0 = time.perf_counter()
        with open(obs_pipe, "rb") as f:
            msg = pickle.loads(f.read())
        t_wait += time.perf_counter() - t0
        if isinstance(msg, str) and msg == "Done":
            print(f"done after {agent.step} steps, {t_wait:.1f} s waiting for observations", flush=True)
            return
        obs, info = msg
        plan = np.asarray(agent(obs, info), dtype=np.float64)
        with open(plan_pipe, "wb") as f:
            f.write(pickle.dumps(plan))


if __name__ == "__main__":
    main()
