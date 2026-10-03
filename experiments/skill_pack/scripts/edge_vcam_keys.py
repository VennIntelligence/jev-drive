"""Lane EDGE-FIX (plans/2026-10-04-roadedge-diagnosis-plan.md, addendum 4): a run dir whose 4 keyframes are rendered for the
virtual camera (height H m above the road, pitch P deg), so that op_lb's GIMM synth / run work on it unchanged.
Writes $DATA_DIR/runs/op_lb/<src>_vh<cm>/{meta.json, tokens.txt, keys.npy}. envs/openpilot, CPU.

  edge_vcam_keys.py --src lb_navtrain --h 1.40 --workers 48
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]


def keys(args):
    import op_lb
    return op_lb._vcam_job(args)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="lb_navtrain")
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    import op_lb
    from jevdrive import navsim_zs as Z
    src = op_lb.root(a.src)
    dst = op_lb.root(f"{a.src}_vh{round(a.h * 100)}" + (f"p{a.pitch:g}".replace("-", "m") if a.pitch else ""))
    mt = json.loads((src / "meta.json").read_text())
    idx = Z.load_index(mt["split"], slim=True)
    ents = [idx[k] for k in mt["index"]]
    for f in ("meta.json", "tokens.txt"):
        shutil.copy(src / f, dst / f)
    out = np.lib.format.open_memmap(dst / "keys.tmp.npy", "w+", np.uint8, (len(ents), 4) + op_lb.FRAME)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, k in enumerate(ex.map(keys, [(e, a.h, a.pitch) for e in ents], chunksize=8)):
            out[i] = k
    out.flush()
    del out
    (dst / "keys.tmp.npy").replace(dst / "keys.npy")
    print(dst, len(ents))


if __name__ == "__main__":
    main()
