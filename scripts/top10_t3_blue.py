#!/usr/bin/env python
"""Top-10 T3: BLUE (SimLingo epoch=013 + trained language gate, threshold 0.66) in shadow on the re-recorded P5 v1 BA
worlds, offline ([T3] in todos/2026-09-26-top10-intersection.md). Env: envs/blue.

Per world, the author's agent (team_code/agent_simlingo.py LingoAgent) is set up as the leaderboard would set it up,
given the recorded global plan, and fed every recorded tick in order: its own `tick` (JPEG round trip, InternVL
tiling, UKF, route planner, command history, prompt) on every tick, its model on the camera ticks the exam
references. Two adapter choices, as for BridgeDrive: the UKF is fed the control that was actually applied (the
expert's, one tick earlier; the author's first call brakes, as run_step does), and the image on a tick the model does
not read is a black 64 x 32 stand-in (only the model reads the image, and the model only reads the current one).
`--ref` instead calls the author's `run_step` on every tick (images held from the last camera tick): the equivalence
reference for that shortcut.

Setup once per process: the model is instantiated and loaded once and handed to every later world's `setup`
(hydra.utils.instantiate and the checkpoint torch.load are served from that cache; `--no-cache` re-loads per world).

  python scripts/top10_t3_blue.py --plan <blue_plan.json> --out <dir> [--shard i/n] [--ref] [--worlds a,b]
Output per world: <out>/<rid>.json with, per referenced tick k: speed waypoints (10 x 0.25 s), path (20 x 1 m), gate
decision / score, language, prompt, and the target points.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BLUE = Path(os.environ.get("BLUE_ROOT", Path.home() / "data/third_party/blue"))
CKPT = os.environ.get("BLUE_SIMLINGO_CKPT", str(Path(os.environ["DATA_DIR"]) / "cache/huggingface/hub/models--RenzKa--simlingo/"
                      "snapshots/26c7c89e797d4e25bbf640013317af8da26a5454/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"))
GATE = str(BLUE / "gate/weights/blue_simlingo_gate.pt")
THRESHOLD = 0.66                                   # docs/MODEL_ZOO.md recommended threshold
_CACHE = {}


def _imports():
    sys.path[:0] = [str(BLUE), str(BLUE / "team_code"), str(BLUE / "Bench2Drive/leaderboard"),
                    str(BLUE / "Bench2Drive/scenario_runner"), os.environ["CARLA_ROOT"] + "/PythonAPI/carla"]
    os.chdir(BLUE)                                 # the agent resolves ./pretrained/InternVL2-1B relative to cwd
    os.environ.update(BLUE_MODE="trained_gate", BLUE_GATE_CKPT=GATE, BLUE_GATE_THRESHOLD=str(THRESHOLD), ROUTES="")
    os.environ.setdefault("SAVE_PATH", "/tmp/t3_blue")
    import torch
    # torch >= 2.6 loads weights_only; the gate .pt stores numpy scalars in its config (the smoke's workaround)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, *[type(np.dtype(t)) for t in "fdil?"]])
    import agent_simlingo as A
    return A, torch


def make_agent(A, torch, cache=True):
    """LingoAgent.setup as the leaderboard calls it; the model object and checkpoint come from the process cache."""
    import hydra
    inst, load = hydra.utils.instantiate, torch.load

    def cached_inst(*a, **k):
        if "model" not in _CACHE:
            _CACHE["model"] = inst(*a, **k)
        return _CACHE["model"]

    def cached_load(f, *a, **k):
        if f != CKPT:
            return load(f, *a, **k)
        if "sd" not in _CACHE:
            _CACHE["sd"] = load(f, *a, **k)
        return _CACHE["sd"]

    agent = A.LingoAgent.__new__(A.LingoAgent)
    if cache:
        A.hydra.utils.instantiate, A.torch.load = cached_inst, cached_load
    try:
        agent.setup(CKPT + "+t3")
    finally:
        A.hydra.utils.instantiate, A.torch.load = inst, load
    agent.model.eval()
    return agent


def world(A, torch, rid, adir, ks, ref=False, cache=True):
    import carla
    from agents.navigation.local_planner import RoadOption
    adir = Path(adir)
    agent = make_agent(A, torch, cache)
    plan = json.loads((adir / "plan.json").read_text())
    agent.set_global_plan([({"lat": la, "lon": lo, "z": z}, RoadOption(o)) for la, lo, z, o in plan["gps"]],
                          [(carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=p, yaw=yw, roll=r)), RoadOption(o))
                           for x, y, z, p, yw, r, o in plan["world"]])
    ctrl = [json.loads(l) for l in open(adir / "pose.jsonl")]
    rows = [json.loads(l) for l in open(adir / "blue_inputs.jsonl")]
    imgs = {json.loads(l)["k"]: json.loads(l)["blue"] for l in open(adir / "frames.jsonl")}
    ks, out, t_model = set(ks), [], []
    dummy = np.zeros((32, 64, 4), np.uint8)
    held = dummy
    captured = {}
    if ref:
        fwd = agent.model.forward

        def hook(*a, **k):
            captured["out"] = fwd(*a, **k)
            return captured["out"]
        agent.model.forward = hook
        agent.get_metric_info = lambda: {"acceleration": 0.0}
    import cv2
    for i, r in enumerate(rows):
        k, f = r["k"], r["frame"]
        read = k in ks and imgs.get(k)
        img = dummy
        if read or (ref and imgs.get(k)):
            bgr = cv2.imread(str(adir / imgs[k]), cv2.IMREAD_UNCHANGED)
            img = held = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
        elif ref:
            img = held
        inp = {"rgb_0": (f, img), "gps": (f, np.array(r["gps"])), "imu": (f, np.array(r["imu"])),
               "speed": (f, {"speed": r["speed"]})}
        if i > 0:
            c = ctrl[i - 1]
            agent.control = carla.VehicleControl(throttle=c["throttle"], steer=c["steer"], brake=c["brake"])
        if ref:
            captured.clear()
            agent.run_step(inp, r["t"])
            res = captured.get("out")
        elif not agent.initialized:                # run_step's first call: initialise, brake, tick
            agent._init()
            agent.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            agent.tick(inp)
            continue
        else:
            agent.tick(inp)
            res = None
            if read:
                t0 = time.perf_counter()
                with torch.no_grad():
                    res = agent.model(A.DrivingInput(**agent.DrivingInput))
                t_model.append(time.perf_counter() - t0)
        if res is None or k not in ks:
            continue
        wps, route, lang = res
        m = agent.model
        out.append({"k": k, "wps": np.round(wps.float()[0].cpu().numpy(), 4).tolist(),
                    "route": np.round(route.float()[0].cpu().numpy(), 4).tolist(),
                    "gate": int(m.gate_decisions[0]) if getattr(m, "gate_decisions", None) else None,
                    "gate_score": float(m.gate_scores[0]) if getattr(m, "gate_scores", None) else None,
                    "language": lang[0] if lang is not None and len(lang) else None,
                    "prompt": agent.prompt, "target_points": np.round(np.array(agent.target_points), 3).tolist(),
                    "speed_in": r["speed"]})
    return out, t_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="JSON list of {rid, adir, ks}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--worlds", default="", help="comma list of route ids (default: all in the plan)")
    ap.add_argument("--ref", action="store_true", help="the author's run_step on every tick (equivalence reference)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    if a.worlds:
        keep = set(a.worlds.split(","))
        plan = [p for p in plan if p["rid"] in keep]
    i, n = map(int, a.shard.split("/"))
    plan = plan[i::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, torch = _imports()
    t_all = time.time()
    for j, p in enumerate(plan):
        dst = out / (p["rid"] + ".json")
        if dst.exists():
            continue
        t0 = time.time()
        rows, tm = world(A, torch, p["rid"], p["adir"], p["ks"], ref=a.ref, cache=not a.no_cache)
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rid": p["rid"], "adir": p["adir"], "ref": a.ref, "threshold": THRESHOLD,
                                   "wall_s": round(time.time() - t0, 2),
                                   "model_ms_mean": round(1e3 * float(np.mean(tm)), 1) if tm else None,
                                   "frames": rows}))
        tmp.rename(dst)
        print(json.dumps({"world": p["rid"], "done": j + 1, "of": len(plan), "frames": len(rows),
                          "wall_s": round(time.time() - t0, 1), "elapsed_s": round(time.time() - t_all)}), flush=True)


if __name__ == "__main__":
    main()
