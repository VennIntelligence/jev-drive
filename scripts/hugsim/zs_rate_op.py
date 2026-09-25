#!/usr/bin/env python
"""How much does HUGSIM's 4 Hz frame rate cost openpilot? Offline on real comma video (comma1M, the 8 segments of the
rig study, native model frames already rendered by scripts/openpilot_rig_study.py). todos/2026-09-25-hugsim-exam.

Variants (all score the plan at the frames where a 4 Hz simulator would deliver one, frames 5k >= 100, against the
localizer's future at REAL horizons 1 / 2 / 4 s):
  native     modeld as on the car: every 20 Hz frame, plan read at frame 5k.
  ctx5-hold  every 4th frame (5 Hz, the context rate) held for 4 steps: each output phase sees exactly what it sees
             on the car, so this must equal native up to the missing in-between phases (sanity check).
  h4-dilate  the HUGSIM adapter: frames 5k (0.25 s apart) fed as consecutive 0.2 s context steps, i.e. the model's
             clock runs 1.25x fast; plan time tau is read as real time 1.25 tau.
  h4-hold    the naive alternative: 20 Hz clock, the 4 Hz frame held for 5 steps, plan read after the 5th step at
             face value (the model sees 0.25 s of motion between frames it believes are 0.2 s apart).

    CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/openpilot/bin/python scripts/hugsim/zs_rate_op.py --models cinque lebowski
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from jevdrive.openpilot.frames import load_segment_meta  # noqa: E402
from jevdrive.openpilot.model import T_IDXS, OPModel, decode  # noqa: E402
from openpilot_replay import ACTION_T, ground_truth  # noqa: E402

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
FR = D / "runs" / "openpilot_rigs" / "frames"
COMMA = D / "datasets" / "comma1M"
H = (1.0, 2.0, 4.0)
DIL = 1.25


def interp_plan(pos, tau):
    """(33, 3) plan at model time tau -> (x, y)."""
    return np.array([np.interp(tau, T_IDXS, pos[:, 0]), np.interp(tau, T_IDXS, pos[:, 1])])


def rollout(m, frames, variant):
    """Returns {frame index 5k: plan_pos (33, 3)}."""
    n = len(frames)
    out = {}
    m.reset()
    if variant == "native":
        for i in range(n):
            raw = m.step(frames[i], action_t=ACTION_T)
            if i % 5 == 0:
                out[i] = decode(raw, m.slices, 10.0)["plan_pos"]
    elif variant == "h4-hold":
        for i in range(n):
            raw = m.step(frames[5 * (i // 5)], action_t=ACTION_T)
            if i % 5 == 4:
                out[i - 4] = decode(raw, m.slices, 10.0)["plan_pos"]
    else:
        step = 4 if variant == "ctx5-hold" else 5
        reps = 1 if m.skip == 1 else 4
        for i in range(0, n, step):
            for _ in range(reps):
                raw = m.step(frames[i], action_t=ACTION_T)
            if i % 5 == 0:
                out[i] = decode(raw, m.slices, 10.0)["plan_pos"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cinque", "lebowski"])
    ap.add_argument("--out", default=str(D / "runs" / "hugsim-exam" / "rate_op.jsonl"))
    a = ap.parse_args()
    segs = [p.name for p in sorted(FR.iterdir()) if (p / "native.npy").exists()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    for name in a.models:
        models = {"20hz": OPModel(name, "trt")}
        if name == "lebowski":
            models["ctx"] = OPModel(name, "trt", context_rate=True)
        for sid in segs:
            meta = load_segment_meta(COMMA / sid)
            frames = np.load(FR / sid / "native.npy", mmap_mode="r")
            n = len(frames)
            gt = ground_truth(meta, n)
            frames = np.asarray(frames)
            for v in ("native", "ctx5-hold", "h4-dilate", "h4-hold"):
                m = models["ctx"] if v in ("ctx5-hold", "h4-dilate") and "ctx" in models else models["20hz"]
                t0 = time.time()
                plans = rollout(m, frames, v)
                dil = DIL if v == "h4-dilate" else 1.0
                for i, p in plans.items():
                    if i < 100:
                        continue
                    row = {"model": name, "seg": sid, "variant": v, "i": i, "speed": float(gt["speed"][i])}
                    for h in H:
                        g = np.array([np.interp(h, T_IDXS, gt["gt_pos"][i][:, k]) for k in (0, 1)])
                        if not np.isfinite(g).all():
                            continue
                        q = interp_plan(p, h / dil)
                        row[f"dlon@{h:g}"], row[f"dlat@{h:g}"] = float(q[0] - g[0]), float(q[1] - g[1])
                        row[f"gtlon@{h:g}"] = float(g[0])
                    fh.write(json.dumps(row) + "\n")
                print(f"{name} {sid[:8]} {v}: {len(plans)} plans, {time.time() - t0:.0f} s", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
