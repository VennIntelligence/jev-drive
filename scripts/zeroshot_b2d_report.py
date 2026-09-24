#!/usr/bin/env python
"""Per-route table of a zero-shot Bench2Drive run (scripts/b2d_run.py + b2d_zeroshot_agent.py): official DS / RC /
infractions from the evaluator's results.json, and the cost of each route (wall, ticks, s/tick, planning calls).
Standard library only.

    python3 scripts/zeroshot_b2d_report.py <run dir> [<run dir> ...] --out results.csv
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def last_attempt(run, rid):
    attempts = sorted((run / "attempts" / rid).glob("*"), key=lambda p: int(p.name))
    return attempts[-1] if attempts else None


def row(run, rid):
    a = last_attempt(run, rid)
    r = {"run": run.name, "route_id": rid, "attempts": len(list((run / "attempts" / rid).glob("*")))}
    if a is None:
        return r
    res = json.loads((a / "results.json").read_text()) if (a / "results.json").exists() else {}
    rec = (res.get("_checkpoint", {}).get("records") or [{}])[0]
    sc = rec.get("scores", {})
    inf = {k: len(v) for k, v in rec.get("infractions", {}).items() if v}
    rr = json.loads((a / "route_result.json").read_text()) if (a / "route_result.json").exists() else {}
    prof = rr.get("profile", {})
    plans = [json.loads(l) for l in open(a / "plans.jsonl")] if (a / "plans.jsonl").exists() else []
    ticks = prof.get("ticks") or 0
    rt = [p["round_trip_ms"] for p in plans]
    inf_ms = [p["infer_ms"] for p in plans if "infer_ms" in p]
    r.update(status=rec.get("status", rr.get("status")), ds=sc.get("score_composed"), rc=sc.get("score_route"),
             penalty=sc.get("score_penalty"), infractions=json.dumps(inf, sort_keys=True),
             game_s=rec.get("meta", {}).get("duration_game"), wall_s=rr.get("wall_s"), ticks=ticks,
             s_per_tick=round(rr["wall_s"] / ticks, 4) if ticks and rr.get("wall_s") else None,
             world_tick_ms=prof.get("world_tick_ms_mean"), tree_ms=prof.get("tree_ms_mean"),
             agent_ms=prof.get("agent_ms_mean"), n_plans=len(plans),
             plan_rt_ms=round(statistics.mean(rt), 1) if rt else None,
             infer_ms=round(statistics.mean(inf_ms), 1) if inf_ms else None,
             queue_ms=round(statistics.mean(p.get("queue_ms", 0) for p in plans), 1) if plans else None)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="")
    a = p.parse_args()
    rows = []
    for run in map(Path, a.runs):
        for rid in sorted(d.name for d in (run / "attempts").iterdir()):
            rows.append(row(run, rid))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
