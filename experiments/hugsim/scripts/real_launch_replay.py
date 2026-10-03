#!/usr/bin/env python
"""Replay shipped Cinque on real comma1M launches at the native 20 Hz and save plans + localizer motion.
Per event (onset j from real_launch_scan.py) two input conditions, frames j-HIST .. j+POST:
  real    the recorded stream (real history: creep, camera vibration, vehicle yaw noise)
  frozen  frames before j replaced by the last static frame j-1 (HUGSIM's static warm-up); frames from j on real
Saved per step: plan_pos (33, 3) from decode(), so any readout (direction at 0.8 / 1.0 s ...) is computed later.

  CUDA_VISIBLE_DEVICES=1 python real_launch_replay.py <events.json> <out_dir> [--backend trt] [--shard K --nshard N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HIST, POST = 160, 80
ROOT = Path.home() / "data/datasets/comma1M"


def run(m, frames, tc=(1, 0)):
    from jevdrive.openpilot.model import decode
    m.reset()
    dv = np.zeros(8, np.float32)
    out = []
    for f in frames:
        raw = m.step(f, desire=dv, traffic=tc)
        out.append(decode(raw, m.slices, 0.0)["plan_pos"].astype(np.float32))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("out")
    ap.add_argument("--backend", default="trt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    from jevdrive.openpilot.frames import segment_model_frames
    from jevdrive.openpilot.model import OPModel
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = json.load(open(a.events))
    sids = sorted(ev)[a.shard::a.nshard]
    m = OPModel("cinque", a.backend)
    for sid in sids:
        if not all((ROOT / sid / f).exists() for f in ("fcamera.hevc", "ecamera.hevc")):
            print("no video", sid, flush=True)
            continue
        todo = [e for e in ev[sid] if not (out / f"{sid}_{e['onset']}.npz").exists()]
        if not todo:
            continue
        top = max(e["onset"] for e in todo) + POST
        fr, _ = segment_model_frames(ROOT / sid, max_frames=min(top, 1200))
        for e in todo:
            j = e["onset"]
            lo, hi = max(0, j - HIST), min(len(fr), j + POST)
            real = fr[lo:hi]
            frozen = real.copy()
            frozen[: j - lo] = fr[j - 1]
            pa, pb = run(m, real), run(m, frozen)
            np.savez_compressed(out / f"{sid}_{j}.npz", lo=lo, onset=j, real=pa, frozen=pb)
            print("done", sid, j, flush=True)
        del fr


if __name__ == "__main__":
    main()
