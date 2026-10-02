"""Collect attack_counterfactual.py logs into one table (local, stdlib + csv).

usage: attack_counterfactual_table.py <dir with *.log> <out.csv>
Joins each log's SUMMARY line with the recorded outcomes of the four exam agents (PR#57 controller) from
results/hugsim-exam/scored_op.csv and scored_base.csv.
"""
import csv, glob, json, os, re, sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1] / "results/hugsim-exam"
sc = {r["scenario"]: r for r in csv.DictReader(open(R / "scenarios.csv"))}
out = {}; by_dir = {}; steps = {}
for f in ("scored_op.csv", "scored_base.csv"):
    for r in csv.DictReader(open(R / f)):
        if r["controller"] == "fixed": out.setdefault(r["scenario"], {})[r["agent"]] = r["end"]; by_dir[os.path.basename(r["run_dir"])] = r["scenario"]
        if r["tag"] == "cinque-fixed": steps[r["scenario"]] = r["steps"]
rows = []
for log in sorted(glob.glob(sys.argv[1] + "/*.log")):
    s = [l for l in open(log) if l.startswith("SUMMARY")]
    if not s: continue
    d = json.loads(s[0][8:]); run = os.path.basename(d["run"])
    scen = by_dir[run]
    k = d["tried_free_free_on_ground_by_kind"]
    rows.append([scen, sc[scen]["difficulty"], sc[scen]["planners"], ("%d:%d" % tuple(d["replay_hit"]) if d["replay_hit"] else "none"), steps[scen], round(d["replay_err_m"], 3),
                 k["brake_hold"][1], k["reverse"][1], k["swerve"][1], k["brake_hold"][2], k["reverse"][2], k["swerve"][2]] +
                [out[scen].get(a, "") for a in ("cinque", "lebowski", "ltf", "cv")])
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "difficulty", "planners", "replay_first_overlap_step:actor", "recorded_cinque_steps", "replay_actor_err_m", "brake_free", "reverse_free", "swerve_free", "brake_free_ground", "reverse_free_ground", "swerve_free_ground", "cinque", "lebowski", "ltf", "cv"])
    w.writerows(rows)
print(len(rows))
