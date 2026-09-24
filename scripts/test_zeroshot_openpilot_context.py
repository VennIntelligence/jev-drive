"""openpilot external-queue model stepped at the 5 Hz context rate reproduces modeld's 20 Hz outputs on that phase.

Runs on the box in envs/openpilot on a comma1M segment:
    ~/data/envs/openpilot/bin/python scripts/test_zeroshot_openpilot_context.py [--backend trt]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.openpilot.frames import segment_model_frames  # noqa: E402
from jevdrive.openpilot.model import FRAME_SKIP, OPModel  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lebowski")
    p.add_argument("--backend", default="trt")
    p.add_argument("--segment", default=str(Path.home() / "data/datasets/comma1M/0045b4fe97931624e169d500d6009784"))
    p.add_argument("--frames", type=int, default=400)
    a = p.parse_args()
    imgs, _ = segment_model_frames(a.segment, max_frames=a.frames)
    desire = np.zeros((len(imgs), 8), np.float32)
    desire[150:170, 1] = 1   # a turn-left pulse inside the window, to exercise the desire max-pool alignment
    full = OPModel(a.model, a.backend)
    ref = [full.step(imgs[i], desire[i]) for i in range(len(imgs))]
    ctx = OPModel(a.model, a.backend, context_rate=True)
    # One context step per 4 modeld steps; its desire is the max over the 4 modeld steps it stands for, which is
    # what modeld's max-pool sees (the rising edge falls in the window that contains it).
    worst = 0.0
    for i in range(0, len(imgs), FRAME_SKIP):
        d = desire[max(0, i - FRAME_SKIP + 1):i + 1].max(0)
        out = ctx.step(imgs[i], d)
        worst = max(worst, float(np.abs(out - ref[i]).max()))
    print("max |context-rate - modeld| over %d phase-0 steps: %.3g" % (len(range(0, len(imgs), FRAME_SKIP)), worst))
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
