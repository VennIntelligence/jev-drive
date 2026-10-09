"""M1: compact per-decision table out of the driver logs of finished AlpaSim runs (box, plain python3 + numpy).

  m1_decisions.py --out <npz> name=<run dir or glob>[,<...>] ...

Per driver name: scene, k, anchor (x, y, yaw of the rear axle in the rollout's local frame at the decision), the plan (8, 3), command,
the route's first waypoint (rig frame), speed, and the scene scores / failure reasons of results-summary.json. First run dir wins when
a scene is in several.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np


def main():
    out, specs = sys.argv[2], sys.argv[3:]
    Z = {}
    for sp in specs:
        name, dirs = sp.split("=", 1)
        rows, score, seen = [], {}, set()
        for D in (Path(d) for pat in dirs.split(",") for d in sorted(glob.glob(pat)) if (Path(d) / "aggregate/results-summary.json").exists()):
            S = {r["clipgt_id"]: r for r in json.loads((D / "aggregate/results-summary.json").read_text())["rollouts"]}
            new = set(S) - seen
            for s in new:
                score[s] = (S[s]["score"], S[s].get("failure_reason") or "")
            for line in (D / "driver-logs/drive.jsonl").open():
                if '"kind": "drive"' not in line:
                    continue
                r = json.loads(line)
                if r["scene"] in new:
                    rows.append((r["scene"], r["k"], *r["anchor"], np.array(r["poses"], np.float32), r.get("cmd", -1),
                                 *(r.get("route0") or [np.nan, np.nan]), 10.0 * r["ego"][4] if "ego" in r else np.nan))
            seen |= new
        rows.sort(key=lambda x: (x[0], x[1]))
        Z[f"{name}|scene"] = np.array([x[0] for x in rows])
        Z[f"{name}|num"] = np.array([[x[1], x[2], x[3], x[4], x[6], x[7], x[8], x[9]] for x in rows], np.float64)   # k, x, y, yaw, cmd, route x, y, v
        Z[f"{name}|plan"] = np.stack([x[5] for x in rows])
        Z[f"{name}|scored"] = np.array(sorted(score))
        Z[f"{name}|score"] = np.array([score[s][0] for s in sorted(score)])
        Z[f"{name}|fail"] = np.array([score[s][1] for s in sorted(score)])
        print(name, len(rows), "decisions", len(score), "scenes", flush=True)
    np.savez_compressed(out, **Z)


if __name__ == "__main__":
    main()
