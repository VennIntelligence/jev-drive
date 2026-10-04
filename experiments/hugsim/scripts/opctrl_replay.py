"""Replay the exam baseline (cinque-fixed) HUGSIM runs open loop on CPU and record what openpilot's own control path would have been fed:
the action head's desired curvature (modeld: action[0] / max(1, v)^2, v = the model's dilated speed 1.25 v) and the plan-derived curvature
(drive_helpers.get_curvature_from_plan at action_t, the path for models without an action head), next to the plan's 1 s direction.
Plan: experiments/hugsim/plans/2026-10-05-op-control-stack-prereg.md (offline part). Frames: the run's video.mp4 (lean_probe.LoggedRun),
same feeding protocol as the exam (100 warm-up model steps on frame 0, then 4 per simulator step).
    python opctrl_replay.py <results.csv> <out_dir> --part i/n [--steps 40] [--threads 4]
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import lean_probe as P  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402

DIL = 1.25
ACTION_T = (0.275, 0.525)            # what the exam fed (jevdrive.openpilot.model default): lateralDelay 0.2 + 0.075


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="cinque-fixed")
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel, T_IDXS, curvature_from_plan, decode
    i, n = map(int, a.part.split("/"))
    rows = [r for r in csv.DictReader(open(a.results)) if r["tag"] == a.tag]
    rows = sorted({r["scenario"]: r for r in rows}.values(), key=lambda r: r["scenario"])[i::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = OPModel("cinque", "cpu", threads=a.threads)
    for r in rows:
        f = out / f"{r['scenario']}.json"
        if f.exists():
            continue
        t0 = time.time()
        R = P.LoggedRun({"scenario": r["scenario"], "run_dir": r["run_dir"], "dataset": r["dataset"]}, a.steps)
        m.reset()
        res = {"scenario": r["scenario"], "v": R.v.tolist(), "theta": R.th.tolist(), "steer": [x["steer"] for x in R.recs[:R.n]],
               "plan_logged": [x["plan"][:2] for x in R.recs[:R.n]], "k_act": [], "k_plan": [], "phi1": [], "p1": [], "pos_err": [], "lat25_err": []}
        for j in range(R.n):
            dv = np.zeros(8, np.float32)
            dv[R.des[j]] = 1
            img = R.frame(j)
            for _ in range(P.WARM if j == 0 else P.PER):
                raw = m.step(img, desire=dv, traffic=tuple(R.tc), action_t=ACTION_T)
            vm = DIL * float(R.v[j])
            d = decode(raw, m.slices, vm, ACTION_T)
            plan = Z.openpilot_to_plan(d["plan_pos"], T_IDXS, DIL)
            res["k_act"].append(d["curvature"])
            res["k_plan"].append(float(curvature_from_plan(d["plan_yaw"], np.asarray(raw[m.slices["plan"]][:495]).reshape(33, 15)[:, 14], vm, ACTION_T[0])))
            res["phi1"].append(float(np.degrees(np.arctan2(plan[1, 0], plan[1, 1]))))          # + right, HUGSIM plan (x right, y fwd)
            res["p1"].append(np.round(plan[1], 4).tolist())
            lp = np.asarray(R.recs[j]["model_pos"], float)
            dp = np.round(d["plan_pos"][[4, 8, 12, 16, 20, 24, 32], :2], 3) - lp
            res["pos_err"].append(float(np.abs(dp).max()))
            res["lat25_err"].append(float(dp[3, 1]))                                             # lateral error at 2.5 s
        res["wall_s"] = round(time.time() - t0, 1)
        json.dump(res, open(f, "w"))
        print(f"{r['scenario']}: {R.n} steps {res['wall_s']} s, max plan err {max(res['pos_err']):.3f} m", flush=True)
        P.rss_guard()
    os._exit(0)


if __name__ == "__main__":
    main()
