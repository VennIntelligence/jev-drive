#!/usr/bin/env python3
"""Lane FIX1, PAI code-path check without a simulator: the driver-side messages COL1 extracted from a finished run (col1_pai_extract.py:
<x dir>/msgs/<scene>.pkl) fed again through the real pai_driver.Driver with the serving switches of the environment (JEV_VCONT,
JEV_LEAD). The driver writes its usual drive.jsonl (with the `fix` record per decision) into --out. Open loop: the ego does not react.
  docker run --rm --gpus device=0 -v <repo>/experiments/alpasim/lib/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro (same for
      pai_driver.py, serve_fix.py) -v <x dir>:/x:ro -v <out>:/out -v <this file>:/r.py:ro -e SH30_TAG=P2H10-F-s0 -e JEV_VCONT=1.0 -e JEV_LEAD=1 \
      <image> python /r.py --msgs /x/msgs --out /out --limit 2
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/jev-drive/experiments/alpasim/lib")


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs", required=True), ap.add_argument("--out", required=True), ap.add_argument("--tag", default="P2H10-F-s0")
    ap.add_argument("--limit", type=int, default=2)
    a = ap.parse_args()
    import pai_driver as D
    core = D.C.Core(a.tag, "cuda")
    core.lead_out = D.FX.LEAD
    drv, pb, ctx = D.Driver(core, Path(a.out), "backwarp", 1, 0, False), D.egodriver_pb2, Ctx()
    for f in sorted(Path(a.msgs).glob("*.pkl"))[:a.limit]:
        n = fail = 0
        for kind, raw in pickle.load(open(f, "rb")):
            if kind == "driver_session_request":
                req = pb.DriveSessionRequest.FromString(raw)
                drv.start_session(req, ctx)
            elif kind == "driver_camera_image":
                drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
            elif kind == "driver_ego_trajectory":
                drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
            elif kind == "route_request":
                drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
            elif kind == "driver_request":
                n += 1
                try:
                    drv.drive(pb.DriveRequest.FromString(raw), ctx)
                except RuntimeError as e:
                    fail += 1
                    if fail <= 3:
                        print(f.stem, n, "drive failed:", e, flush=True)
        print(f.stem, n, "drive calls,", fail, "failed", flush=True)
    R = [json.loads(l) for l in open(Path(a.out) / "drive.jsonl")]
    R = [r for r in R if r.get("kind") == "drive" and r.get("infer")]
    ms = np.array([r["fix"]["ms"] for r in R if r.get("fix") and "ms" in r["fix"]])
    tot = np.array([r["total_ms"] for r in R])
    cut = np.array([r["fix"]["lead"]["cut"] for r in R if r.get("fix") and "lead" in r["fix"]])
    print(json.dumps({"decisions": len(R), "switches": D.FX.SUFFIX, "serve_ms_median_p99": [float(np.median(ms)), float(np.percentile(ms, 99))] if len(ms) else None,
                      "drive_ms_median_p90": [float(np.median(tot)), float(np.percentile(tot, 90))], "lead_cut_decisions": int((cut >= 0.5).sum()) if len(cut) else None}))


if __name__ == "__main__":
    main()
