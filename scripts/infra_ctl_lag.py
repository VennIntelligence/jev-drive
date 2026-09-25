#!/usr/bin/env python3
"""Longitudinal lag diagnosis for the B2D controller acceptance (b2d-controllers-p5.md, descriptive, added after the
runs). Per arm and route, from simulator truth against expert a's rear-axle path:
  departure  time the car first passes 1 m of the expert's arc minus the expert's time to do the same (s)
  time_lag   median over arc positions 5, 10, ... m of (car time - expert time) to reach them (s)
  max_lag    the largest of those

    python scripts/infra_ctl_lag.py $DATA_DIR/runs/infra-accept/b2d-ctl-v2 [--arms f2t,p5x2,p5x5,p5x1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infra_ctl_score as S  # noqa: E402

REAR = -1.388633220199954


def first_time(s, t, d):
    k = np.flatnonzero(s >= d)
    return t[k[0]] if len(k) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arms", default="f2t,p5x2,p5x5,p5x1")
    a = ap.parse_args()
    run = Path(a.run)
    routes = sorted(p.stem for p in (run / "expert-logs").glob("*.jsonl"))
    for arm in a.arms.split(","):
        rows = []
        for rid in routes:
            te, xe, _, _ = S.expert_track(S.last_attempt(run / "expert-a", rid), REAR)
            se = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xe, axis=0).T))]
            att = S.last_attempt(run / ("arm-" + arm), rid)
            ticks = [json.loads(line) for line in open(att / "ticks.jsonl")]
            t = np.array([x["t"] for x in ticks])
            t -= t[0]
            sa = S.project(xe, se, np.array([x["truth"][:2] for x in ticks]))[:, 0]
            marks = np.arange(5.0, min(sa.max(), se.max()) - 1, 5.0)
            lag = np.array([first_time(sa, t, d) - first_time(se, te, d) for d in marks])
            rows.append((rid, first_time(sa, t, 1.0) - first_time(se, te, 1.0), np.nanmedian(lag), np.nanmax(lag)))
        r = np.array([x[1:] for x in rows], float)
        print("%s  departure median %.2f s  time_lag median %.2f s" % (arm, np.nanmedian(r[:, 0]), np.nanmedian(r[:, 1])))
        for rid, dep, lag, mx in rows:
            print("   %6s  departure %5.2f  time_lag %6.2f  max_lag %6.2f" % (rid, dep, lag, mx))


if __name__ == "__main__":
    main()
