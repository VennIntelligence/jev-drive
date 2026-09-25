#!/usr/bin/env python3
"""Pre-registered choice of the openpilot full-run configuration from smoke3 (todos/2026-09-24-zeroshot-exam/
openpilot-migration.md, section D3, frozen before smoke3). Driving Score never decides.

    python3 zeroshot_b2d_op_choose.py <b2d-op dir> --out full-choice.json

Candidates in order: smoke3-partner-lebowski (Zoo PID, primary), smoke3-partner-native-lebowski (openpilot's own
curvature / accel execution). The first one meeting all four adapter criteria on the 9 smoke3 routes (last attempt each):
  E1 start     the ego moves (v > 0.5 m/s) within 20 s of the route start on >= 8 of 9 routes
  E2 turns     LEFT / RIGHT route turns passed / reached >= 0.75, pooled
  E3 handover  no collision while the control is being blended between drivers
  E4 share     the model drives >= 50 % of the distance, pooled (else the score would be mostly TCP's)
None meets them -> "controller": null and exit 1 (a decision for the user; no full run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_b2d_op_accept import last_attempts, route_checks  # noqa: E402
from zeroshot_b2d_junctions import attempt_summary  # noqa: E402

CANDIDATES = (("partner-lebowski", "zoo_pid"), ("partner-native-lebowski", "native"))
COLLISIONS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")


def criteria(run):
    atts = last_attempts(run)
    rc = [route_checks(a) for a in atts]
    sm = [attempt_summary(a) for a in atts]
    turns = [p for r in rc for _, p in r["a7_turns"]]
    dist = {d: sum(s["dist_m_" + d] for s in sm) for d in ("model", "partner", "blend")}
    blend_col = sum(k in COLLISIONS for s in sm for k in json.loads(s["infractions_by_driver"]).get("blend", []))
    e = dict(E1_started=sum(r["a5_start_s"] is not None and r["a5_start_s"] <= 20.0 for r in rc), n_routes=len(rc),
             E2_turns="%d/%d" % (sum(turns), len(turns)), E3_blend_collisions=blend_col,
             E4_dist_share_model=round(dist["model"] / max(sum(dist.values()), 1e-9), 3))
    e["ok"] = bool(len(rc) == 9 and e["E1_started"] >= 8 and turns and sum(turns) / len(turns) >= 0.75
                   and blend_col == 0 and e["E4_dist_share_model"] >= 0.5)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, choice = {}, None
    for phase, ctl in CANDIDATES:
        res[phase] = criteria(a.root / ("smoke3-" + phase))
        if choice is None and res[phase]["ok"]:
            choice = ctl
    res["controller"] = choice
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if choice else 1)


if __name__ == "__main__":
    main()
