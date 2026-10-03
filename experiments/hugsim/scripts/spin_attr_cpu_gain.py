"""Local and large-signal loop gain of shipped Cinque on logged HUGSIM frames, CPU ONNX (no GPU).  Plan: experiments/hugsim/plans/2026-10-04-spin-attribution-prereg.md (Q2).
Same measurement as lean_probe.py (decision 100), steps 1..S only, fake yaw +-w only, on onnxruntime CPU instead of TensorRT:
  local  at each logged step s: the logged history as it was plus a fake rate +-w (deg / model-s) on every history frame; s_local = (phi1(+w) - phi1(-w)) / 2 / (1.2 w)
  derot  at each step s: the history re-rendered at the step's heading (decision 96 rule); large-signal gain s = (phi1(normal) - phi1(derot)) / H
phi1 = direction of the HUGSIM plan point at 1 s (deg, + left).  normal = local at w = 0.
    python spin_attr_cpu_gain.py <jobs.json> <key> <out.json> [--threads 4] [--steps 10] [--w 2.0]
jobs.json: [{"key", "scenario", "run_dir", "dataset"}]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import lean_probe as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("key")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--w", type=float, default=2.0)
    a = ap.parse_args()
    job = next(j for j in json.load(open(a.jobs)) if j["key"] == a.key)
    from jevdrive.openpilot.model import OPModel
    m = OPModel("cinque", "cpu", threads=a.threads)
    t0 = time.time()
    R = P.LoggedRun(job, a.steps + 1)
    res = {"key": a.key, "scenario": job["scenario"], "theta": R.th.tolist(), "v": R.v.tolist(), "local": {}, "derot": {}}
    for s in range(1, R.n):
        j0 = max(0, s - P.CTX)
        for w in (0.0, a.w, -a.w):
            seq = [(R.frame(j, w * (j - s) * 0.2), P.WARM if j == j0 else P.PER, R.des[j], j == s) for j in range(j0, s + 1)]
            res["local"][f"{s}|{w:+g}"] = P.feed(m, seq, R.tc)[0]
        seq = [(R.frame(j, float(np.degrees(R.th[j] - R.th[s]))), P.WARM if j == j0 else P.PER, R.des[j], j == s) for j in range(j0, s + 1)]
        res["derot"][str(s)] = P.feed(m, seq, R.tc)[0]
        print(f"{a.key} step {s}/{R.n - 1} {time.time() - t0:.0f}s", flush=True)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(a.out, "w"))
    os._exit(0)


if __name__ == "__main__":
    main()
