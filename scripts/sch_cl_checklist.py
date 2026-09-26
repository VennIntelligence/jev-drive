#!/usr/bin/env python
"""SCH: the mechanical sanity checklist of one lane-B CL arm x seed (a ~10-route pilot, or the first routes of a live arm).

Reads a b2d_run --out directory (read-only) and lane B's CL1 arm (the P7 ceiling on the same routes) through
jevdrive.nq3_cl_report.collect, and prints / writes one JSON verdict. Items (thresholds are the handoff's, section
"全局调度 (SCH)"):

  completion   finished / requested routes                                   FAIL if < 0.80
  crash        attempts that did not end as a leaderboard record (server died, hung, harness error, timeout, crash)
               / all attempts                                                FAIL if > 0.25
  moves        routes where the ego drove: route completion RC > 5 %          FLAG if < 0.50 of the finished routes
  blocked      finished routes that ended "Agent got blocked"                  FLAG if > 0.50
  ds           DS mean of the arm and of CL1 on the same finished routes      FLAG if the arm's DS is 0 everywhere,
                                                                              or it beats CL1 by > 25 points
  plans        head / openpilot / Alpamayo arms: routes whose plans.jsonl is empty  FAIL if > 0.20 of finished routes

verdict: FAIL (the pipeline is broken: drop the arm and escalate), FLAG (the pipeline runs but the behaviour is
degenerate: escalate; registered no-gate arms such as CL2 / CL7 keep running), PASS.

    .venv/bin/python scripts/sch_cl_checklist.py <arm dir> [--min-routes 10] [--out verdict.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import nq3_cl_report as R  # noqa: E402

PLAN_ARMS = {"cl2", "cl3", "cl4", "cl5", "cl5d", "cl6", "cl7", "cl8", "mc_real0"}


def attempts(d: Path) -> list[dict]:
    out = []
    for f in d.glob("attempts/*/*/attempt.json"):
        try:
            out.append(json.loads(f.read_text()))
        except ValueError:
            pass
    return out


def check(d: Path, min_routes: int = 10) -> dict:
    arm, seed = d.parent.name, int(d.name[1:])
    df = R.collect(d.parent.parent)
    df = df[(df.arm == arm) & (df.seed == seed)]
    fin = df[df.finished]
    if (d / "requested.json").exists() and len(json.loads((d / "requested.json").read_text())) > 3 * min_routes:
        busy = {f.stem for f in (d / "claims").glob("*.lock")}  # a live arm: judge the routes that are settled
        df = df[df.finished | (df.attempts.gt(0) & ~df.route.isin(busy))]
    att = attempts(d)
    bad = [a for a in att if a.get("status") not in ("finished",) and a.get("status") != "cancelled_by_user"]
    cl1 = R.collect(R.root() / "arms")
    cl1 = cl1[(cl1.arm == "cl1") & (cl1.seed == 0) & cl1.finished].set_index("route")
    common = [r for r in fin.route if r in cl1.index]
    ds_arm = float(fin.set_index("route").loc[common, "DS"].mean()) if common else float("nan")
    ds_cl1 = float(cl1.loc[common, "DS"].mean()) if common else float("nan")
    blocked = float(fin.status.astype(str).str.contains("blocked").mean()) if len(fin) else float("nan")
    moves = float((fin.RC > 5).mean()) if len(fin) else float("nan")
    empty_plans = 0
    if arm in PLAN_ARMS:
        for rid in fin.route:
            a = json.loads((d / "done" / f"{rid}.json").read_text())["attempt"]
            p = d / "attempts" / rid / str(a) / "plans.jsonl"
            empty_plans += (not p.exists()) or p.stat().st_size == 0
    items = {
        "completion": {"value": len(fin) / max(len(df), 1), "n": f"{len(fin)} / {len(df)}"},
        "crash": {"value": len(bad) / max(len(att), 1), "n": f"{len(bad)} / {len(att)}",
                  "statuses": sorted({str(a.get("status")) for a in bad})},
        "moves": {"value": moves},
        "blocked": {"value": blocked},
        "ds": {"arm": ds_arm, "cl1_same_routes": ds_cl1, "n_common": len(common),
               "range": [float(fin.DS.min()), float(fin.DS.max())] if len(fin) else None},
        "plans": {"empty": int(empty_plans), "of": len(fin)} if arm in PLAN_ARMS else None,
    }
    fail, flag = [], []
    if len(df) < min_routes and not (d / "DONE").exists():
        return {"arm": arm, "seed": seed, "dir": str(d), "verdict": "PENDING", "items": items}
    if items["completion"]["value"] < 0.80:
        fail.append("completion < 0.80")
    if items["crash"]["value"] > 0.25:
        fail.append("crash rate > 0.25")
    if arm in PLAN_ARMS and len(fin) and empty_plans / len(fin) > 0.20:
        fail.append("empty plans.jsonl on > 20 % of routes")
    if len(fin) and moves < 0.50:
        flag.append("ego drove (RC > 5 %) on < 50 % of routes")
    if len(fin) and blocked > 0.50:
        flag.append("> 50 % of routes blocked")
    if len(fin) and float(fin.DS.max()) == 0.0:
        flag.append("DS is 0 on every route")
    if common and ds_arm > ds_cl1 + 25:
        flag.append("DS beats the CL1 ceiling by > 25 points")
    verdict = "FAIL" if fail else "FLAG" if flag else "PASS"
    return {"arm": arm, "seed": seed, "dir": str(d), "verdict": verdict, "fail": fail, "flag": flag, "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-routes", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    v = check(Path(a.dir), a.min_routes)
    s = json.dumps(v, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(s)
    print(s)
    sys.exit({"PASS": 0, "FLAG": 3, "FAIL": 2, "PENDING": 4}[v["verdict"]])


if __name__ == "__main__":
    main()
