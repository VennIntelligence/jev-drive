"""Frames of the negative-route set (experiments/op_route_cmd/results/negatives_sample.npz, 158 navtrain tokens) on op_lb's NAVSIM
protocol (4 keyframes + 6 GIMM context frames, the navtest headline protocol), as op_lb data `lb_guardneg`. Built once, shared by every
candidate; op_lb's own prep / synth do the work (as experiments/op_adapt_h/scripts/h_nav_pool.py does for lb_h1train).

  prep   (envs/openpilot)  runs/op_lb/lb_guardneg/{meta.json, tokens.txt, keys.npy}
  synth  (envs/vfi, GPU)   runs/op_lb/lb_guardneg/gimm.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import op_lb as B  # noqa: E402

DATA = "lb_guardneg"
B.SPLITS[DATA] = "navtrain"
NEG = REPO / "experiments" / "op_route_cmd" / "results" / "negatives_sample.npz"


def tokens() -> list[str]:
    return list(dict.fromkeys(np.load(NEG, allow_pickle=True)["id"].astype(str)))


def cmd_prep(a):
    from jevdrive import navsim_zs as Z
    sel = set(tokens())
    Z.nonav_subset = lambda idx_, per_command=0, seed=0: sel                       # op_lb.cmd_prep's navtrain draw -> this set
    B.cmd_prep(argparse.Namespace(data=DATA, per_cmd=0, seed=0, workers=a.workers))
    got = B.meta(DATA)["names"]
    assert set(got) == sel, f"{len(sel - set(got))} negative tokens missing from the navtrain index"


def cmd_synth(a):
    B.cmd_synth(argparse.Namespace(data=[DATA], method="gimm", gpu=a.gpu, vram_gb=a.vram_gb, cap_gb=a.cap_gb, chunk=32,
                                   batch=8, workers=1, limit_chunks=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("synth")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--vram-gb", type=float, default=14.0)
    p.add_argument("--cap-gb", type=float, default=80.0)
    a = ap.parse_args()
    {"prep": cmd_prep, "synth": cmd_synth}[a.cmd](a)
