#!/usr/bin/env python
"""HUGSIM agent that plans the scene's logged ego trajectory (jevdrive.hugsim_preset.LoggedPlan): the known-good plan
of the controller acceptance (todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md).

It is zs_agent.Agent with the model call replaced, so the plan goes through exactly the path a model's plan takes:
the same step loop, forward_only and straight_stop (HUGSIM_ZS_OPTS, both default on), the same zs_steps.jsonl.
Each step's record also carries the ego's position relative to the log (log_s arc length, log_xt signed cross-track,
+ = right, log_dth heading minus the log's tangent heading, log_v logged speed there, log_t logged time there).

Environment (set by scripts/hugsim/zs_run.py --agent preset): HUGSIM_SCENE_DIR, HUGSIM_ZS_DATASET, HUGSIM_ZS_OPTS
(a_up / a_down: speed-profile bounds of the plan, default 2 / 4 m/s^2; forward_only; straight_stop).
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zs_agent  # noqa: E402  (puts the repo root on sys.path)
from jevdrive.hugsim_preset import LoggedPlan  # noqa: E402


class PresetAgent(zs_agent.Agent):
    def __init__(self, scene_dir, dataset, opts, out):          # no model server, no cameras
        self.model, self.opts, self.out, self.dataset = "preset", opts, Path(out), dataset
        self.hist, self.frames, self.last, self.step = zs_agent.Z.History(), [], None, 0
        self.dump_every, self.engage_s = 0, 0.0
        self.log = open(self.out / "zs_steps.jsonl", "w", buffering=1)
        self.src = LoggedPlan(scene_dir, a_up=float(opts.get("a_up", 2.0)), a_down=float(opts.get("a_down", 4.0)))

    def setup(self, info):
        self.log.write(json.dumps({"setup": True, "model": self.model, "opts": self.opts,
                                   "log": self.src.summary()}) + "\n")

    def openpilot(self, obs, info, rec):                        # the model slot of Agent.__call__
        t = time.perf_counter()
        plan, meta = self.src(info)
        rec.update(meta, infer_ms=round(1e3 * (time.perf_counter() - t), 2))
        return plan


def main():
    out = sys.argv[sys.argv.index("--output") + 1]
    env = os.environ
    agent = PresetAgent(env["HUGSIM_SCENE_DIR"], env.get("HUGSIM_ZS_DATASET", ""), json.loads(env.get("HUGSIM_ZS_OPTS", "{}")),
                        out)
    obs_pipe, plan_pipe = (os.path.join(out, n) for n in ("obs_pipe", "plan_pipe"))
    for p in (obs_pipe, plan_pipe):
        if not os.path.exists(p):
            os.mkfifo(p)
    print(f"preset agent ready: {agent.src.summary()}", flush=True)
    while True:
        with open(obs_pipe, "rb") as f:
            msg = pickle.loads(f.read())
        if isinstance(msg, str) and msg == "Done":
            print(f"done after {agent.step} steps", flush=True)
            return
        obs, info = msg
        plan = np.asarray(agent(obs, info), dtype=np.float64)
        with open(plan_pipe, "wb") as f:
            f.write(pickle.dumps(plan))


if __name__ == "__main__":
    main()
