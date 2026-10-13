#!/usr/bin/env python
"""Per-route leaderboard results of a human_drive.sh run: DS, RC, IS, status, infractions, and the agent's sim pacing.

    python experiments/phone_wheel/scripts/human_results.py /data/runs/human_drive/<stamp>

Reads <run>/done/<id>.json (scripts/b2d_run.py) -> attempts/<id>/<attempt>/{results.json, human_summary.json}; writes
<run>/results.json and <run>/results.md and prints the table. Routes requested but never finished are listed, not scored.
"""
import json
import sys
from pathlib import Path


def main(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    rows = []
    for done in sorted((run / "done").glob("*.json")):
        d = json.loads(done.read_text())
        att = run / "attempts" / str(d["route_id"]) / str(d["attempt"])
        rec = json.loads((att / "results.json").read_text())["_checkpoint"]["records"][0]
        hs = json.loads((att / "human_summary.json").read_text()) if (att / "human_summary.json").exists() else {}
        rows.append({"route": str(d["route_id"]), "town": rec.get("town_name"), "scenario": rec.get("scenario_name"),
                     "DS": round(rec["scores"]["score_composed"], 2), "RC": round(rec["scores"]["score_route"], 2),
                     "IS": round(rec["scores"]["score_penalty"], 3), "status": rec["status"],
                     "infractions": {k: len(v) for k, v in rec["infractions"].items() if v},
                     "infraction_text": [x for v in rec["infractions"].values() for x in v],
                     "sim_s": hs.get("sim_s"), "wall_per_sim": hs.get("wall_per_sim"), "raw_per_sim": hs.get("raw_per_sim"),
                     "wait_s": hs.get("wait_s"), "stale_ticks": hs.get("stale_ticks")})
    order = {r: i for i, r in enumerate(summary.get("attempts", {}))}
    rows.sort(key=lambda r: order.get(r["route"], 1e9))
    never = summary.get("routes_never_finished", [])
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 2) if rows else None  # noqa: E731
    out = {"run": str(run), "routes": rows, "never_finished": never, "n": len(rows), "mean_DS": mean("DS"), "mean_RC": mean("RC")}
    (run / "results.json").write_text(json.dumps(out, indent=1))
    lines = ["| route | town | DS | RC | IS | status | infractions | sim s | wall/sim (raw) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %.1f | %.1f | %.2f | %s | %s | %s | %s (%s) |" % (
            r["route"], r["town"], r["DS"], r["RC"], r["IS"], r["status"],
            ", ".join("%s %d" % kv for kv in r["infractions"].items()) or "-", r["sim_s"], r["wall_per_sim"], r["raw_per_sim"]))
    lines.append("")
    lines.append("mean over %d finished routes: DS %s, RC %s; never finished: %s" % (len(rows), out["mean_DS"], out["mean_RC"],
                                                                                    ", ".join(never) or "none"))
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
