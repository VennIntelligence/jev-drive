"""One-step controller replay on the logged states and plans of the HUGSIM zero-shot exam (CPU, no rendering).

For every logged step of an openpilot run (state v, steer; plan as sent) the actual controller variants are asked what they would
do from that same state: official (transposed heading), fixed (PR #57), fixed2 (PR #57 + tracker v2), each through the
nuPlan iLQR of the HUGSIM tree plus one 0.25 s step of hug_sim's kinematic bicycle. The reference is the "ideal"
heading change, the direction of the plan's 0.5 s point. If the heading-fixed controllers realise (roughly) the
direction the plan asks for and never more, a spin that follows a plan pointing off the route is the plan's, not the tracker's.
    cd $DATA_DIR/third_party/HUGSIM-zs/fixed && $DATA_DIR/envs/hugsim/bin/python spin_controller_replay.py \
        <runs_root> <out.csv> [--workers 16]
runs_root holds scored-op/{cinque,lebowski}-{official,fixed}/zs/<run>/zs_steps.jsonl.
"""
import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/hugsim/archive"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_offline as C  # noqa: E402

DT, L = 0.25, 2.7
VARS = {"official": "official", "fixed": "fixed", "fixed2": "fixed-dt-u"}


def one_run(job):
    d, tag = job
    recs = [json.loads(x) for x in open(Path(d) / "zs_steps.jsonl")][1:]
    sol = {k: C.solver(*C.VARIANTS[v][1:3], *(C.VARIANTS[v][3:])) for k, v in VARS.items()}
    rows = []
    for k in range(len(recs) - 1):
        r, nx = recs[k], recs[k + 1]
        plan = np.array(r["plan"], np.float64)
        v, steer = float(r["v"]), float(r["steer"])
        th = r["theta"]
        dth_log = nx["theta"] - th
        row = dict(run=Path(d).name, tag=tag, step=k, v=v, dth_logged=dth_log,
                   phi05=math.atan2(plan[0, 0], plan[0, 1]) if np.linalg.norm(plan[0]) > 0.05 else float("nan"),
                   plan_len=float(np.linalg.norm(plan[-1])))
        for name, var in VARS.items():
            heading, dt, cap, *extra = C.VARIANTS[var]
            s = sol[name].solve(np.array([0.0, 0.0, 0.0, v, steer]), C.reference(plan, heading, dt))
            acc, sr = s[-1].input_trajectory[0]
            v1, st1 = v + acc * DT, steer + sr * DT
            row["dth_" + name] = v1 * math.tan(st1) / L * DT
            row["steer_" + name] = st1
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = []
    for tag in ("cinque-official", "cinque-fixed", "lebowski-official", "lebowski-fixed"):
        for d in sorted((Path(a.root) / "scored-op" / tag / "zs").iterdir()):
            if (d / "zs_steps.jsonl").exists():
                jobs.append((str(d), tag))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(one_run, jobs) for r in rs]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "steps from", len(jobs), "runs")


if __name__ == "__main__":
    main()
