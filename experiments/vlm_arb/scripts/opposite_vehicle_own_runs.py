"""Collect every vlm_arb attempt on the opposite-vehicle routes of bench2drive_0.0.4_val (run on the box).

usage: opposite_vehicle_own_runs.py <runs/vlm_arb dir> <out.csv>
One row per attempt: arm, route, scenario, status, DS, RC, penalty, and the infraction counts that matter here.
"""
import csv, glob, json, os, sys
ROUTES = {"9196", "8859", "9102", "9218", "9395", "6673"}
root, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(f"{root}/arms/*/attempts/*/*/results.json")):
    parts = p.split("/"); arm, route, att = parts[-5], parts[-3], parts[-2]
    if route not in ROUTES: continue
    try: r = json.load(open(p))
    except Exception: continue
    recs = r.get("_checkpoint", {}).get("records", [])
    if not recs: continue
    rec = recs[0]; inf = rec["infractions"]; sc = rec["scores"]
    rows.append([arm, route, att, rec["scenario_name"], rec["status"], sc["score_composed"], sc["score_route"], sc["score_penalty"],
                 len(inf["collisions_vehicle"]), len(inf["red_light"]), len(inf["stop_infraction"]), len(inf["vehicle_blocked"]),
                 len(inf["route_timeout"]), len(inf["outside_route_lanes"])])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "route", "attempt", "scenario", "status", "ds", "rc", "penalty", "coll_vehicle", "red_light", "stop", "blocked", "timeout", "outside_lanes"])
    w.writerows(rows)
print(len(rows), "attempts")
