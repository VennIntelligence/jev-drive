#!/usr/bin/env python
"""Official Bench2Drive scores of a human_drive.sh run (complete or partial), computed by Bench2Drive's own tools as shipped.

    /data/envs/carla/bin/python experiments/phone_wheel/scripts/human_official.py /data/runs/human_drive/<stamp>

Pipeline (README "Tools" section of Bench2Drive 0.0.4, run unmodified from $BENCH2DRIVE_ROOT):
  1. every finished route's leaderboard checkpoint (attempts/<id>/<n>/results.json, the attempt named in done/<id>.json)
     -> <run>/official/routes/<id>.json
  2. tools/merge_route_json.py -f <run>/official/routes            -> merged.json: "driving score" = sum DS / 220,
     "success rate" = routes Completed / Perfect with no infraction except min_speed, / 220 (missing routes count 0)
  3. tools/ability_benchmark.py -f bench2drive_0.0.4_val.xml -r merged.json -> merged_ability.json: the five abilities over the
     routes present (missing routes are listed as "crashed", not counted)
  4. tools/efficiency_smoothness_benchmark.py -f merged_metric.json -m <run>/official/metric -> Driving Efficiency / Smoothness
     (comfort), over the routes that have metric_info.json (human_agent.py logs it from 2026-10-13 ~23:50; earlier routes lack it)
Writes <run>/official/official.md (the human's row next to the v0.0.4 leaderboard rows) and prints it.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

B2D = Path(os.environ.get("BENCH2DRIVE_ROOT", "/data/third_party/Bench2Drive"))
CARLA = Path(os.environ.get("CARLA_ROOT", "/data/third_party/carla/CARLA_0.9.15"))
XML = B2D / "leaderboard/data/bench2drive_0.0.4_val.xml"
ABILITIES = ["Overtaking", "Merging", "Emergency_Brake", "Give_Way", "Traffic_Signs"]
# Bench2Drive docs/v004_update.md (220-route v0.0.4 val set), as shipped in the repo on 2026-10-13
LEADERBOARD = [  # method, DS, SR, Efficiency, Comfort, Overtaking, Merging, Emergency_Brake, Give_Way, Traffic_Signs, Mean
    ("LEAD", 95.59, 87.28, 201.55, 20.56, 89.86, 73.44, 98.25, 50.00, 97.67, 81.84),
    ("Simlingo", 86.55, 70.45, 251.72, 35.47, 63.77, 60.94, 91.23, 50.00, 90.70, 71.34),
    ("Drive-pi0", 69.71, 45.91, 178.07, 10.80, 24.64, 39.06, 73.68, 33.33, 66.86, 47.52),
    ("DriveTransformer", 60.20, 35.91, 168.87, 17.22, 30.43, 39.06, 49.12, 0.00, 61.63, 36.05),
    ("DriveAdapter", 54.96, 30.45, 91.82, 20.63, 26.09, 21.88, 64.90, 33.33, 69.18, 43.08),
    ("ThinkTwice", 52.84, 27.73, 90.73, 19.76, 20.29, 20.31, 59.64, 33.33, 66.86, 40.09),
    ("TCP", 51.44, 26.36, 75.67, 30.16, 14.49, 14.06, 59.64, 33.33, 63.95, 37.10),
    ("UniAD", 38.69, 19.09, 120.16, 64.09, 11.59, 10.94, 40.35, 33.33, 8.14, 20.87),
    ("VAD", 38.65, 17.27, 163.74, 67.34, 13.04, 17.19, 29.82, 16.67, 19.77, 19.30),
    ("ADMLP", 35.39, 0.00, 64.92, 20.95, 0.00, 0.00, 0.00, 0.00, 45.35, 9.07),
]


def tool(args, log):
    env = dict(os.environ, CARLA_ROOT=str(CARLA), PYTHONPATH=os.pathsep.join(
        [str(CARLA / "PythonAPI/carla"), str(B2D / "leaderboard"), str(B2D / "scenario_runner"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable] + args, cwd=B2D, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write("$ %s\n%s\n" % (" ".join(args), p.stdout))
    return p.returncode, p.stdout


def main(run):
    run = Path(run)
    off = run / "official"
    shutil.rmtree(off, ignore_errors=True)
    (off / "routes").mkdir(parents=True)
    (off / "metric").mkdir()
    with_metric = []
    for done in sorted((run / "done").glob("*.json")):
        d = json.loads(done.read_text())
        att = run / "attempts" / str(d["route_id"]) / str(d["attempt"])
        if not (att / "results.json").exists():
            continue
        res = json.loads((att / "results.json").read_text())
        (off / "routes" / ("%s.json" % d["route_id"])).write_text(json.dumps(res))
        rec = res["_checkpoint"]["records"][0]
        if (att / "metric_info.json").exists() and json.loads((att / "metric_info.json").read_text()):
            (off / "metric" / rec["save_name"]).symlink_to(att.resolve())
            with_metric.append(rec["save_name"])
    n = len(list((off / "routes").glob("*.json")))
    log = open(off / "tools.log", "w")
    if n == 0:
        print("no finished routes in %s" % run)
        return
    tool(["tools/merge_route_json.py", "-f", str(off / "routes")], log)
    merged = json.loads((off / "routes/merged.json").read_text())
    recs = merged["_checkpoint"]["records"]
    ds_n = sum(r["scores"]["score_composed"] for r in recs) / n
    sr_n = merged["success rate"] * 220 / n
    rc, _ = tool(["tools/ability_benchmark.py", "-f", str(XML), "-r", str(off / "routes/merged.json")], log)
    ab_note = ""
    if rc != 0:  # partial run: an ability with no driven route divides by zero; rerun a copy with only that guarded
        src = (B2D / "tools/ability_benchmark.py").read_text()
        old = "Ability_Res = {k: v[0] / v[1] for k, v in Ability_Statistic.items()}"
        assert src.count(old) == 1
        (off / "ability_benchmark_partial.py").write_text(src.replace(old, "Ability_Res = {k: v[0] / v[1] for k, v in Ability_Statistic.items() if v[1]}"))
        rc, _ = tool([str(off / "ability_benchmark_partial.py"), "-f", str(XML), "-r", str(off / "routes/merged.json")], log)
        ab_note = ("Abilities: partial run, the shipped ability_benchmark.py divides by zero for an ability with no driven route; "
                   "official/ability_benchmark_partial.py is the shipped file with only that division guarded (ability left out, "
                   "Mean over the abilities present).")
    ab_path = off / "routes/merged_ability.json"
    ab = json.loads(ab_path.read_text()) if rc == 0 and ab_path.exists() else {}
    eff = smooth = None
    if with_metric:
        sub = dict(merged, _checkpoint={"records": [r for r in recs if r["save_name"] in set(with_metric)]})
        (off / "merged_metric.json").write_text(json.dumps(sub))
        rc, out = tool(["tools/efficiency_smoothness_benchmark.py", "-f", str(off / "merged_metric.json"), "-m", str(off / "metric")], log)
        m1, m2 = re.search(r"Driving Efficiency=([\d.]+)", out), re.search(r"Driving Smoothness=([\d.]+)", out)
        eff, smooth = (float(m1.group(1)) if m1 else None), (100 * float(m2.group(1)) if m2 else None)
    f = lambda v, k=1: "-" if v is None else "%.*f" % (k, v)  # noqa: E731
    abil = [100 * ab[k] if k in ab else None for k in ABILITIES] + [100 * ab["mean"] if "mean" in ab else None]
    hdr = "| Method | Driving Score | Success Rate | Efficiency | Comfort | " + " | ".join(ABILITIES) + " | Mean |"
    lines = ["# Official Bench2Drive v0.0.4 scores: %s" % run, "",
             "%d of 220 routes driven; %d with metric_info (efficiency / comfort over those)." % (n, len(with_metric)),
             "Driving Score and Success Rate as merge_route_json.py reports them (/ 220, undriven routes count 0) and, in the",
             "second row, over the driven routes only; abilities over the driven routes (ability_benchmark.py skips missing ones).",
             "", hdr, "|" + "---|" * 11,
             "| **human (phone), /220** | %s | %s | %s | %s | %s |" % (f(merged["driving score"], 2), f(100 * merged["success rate"], 2),
                                                                        f(eff, 2), f(smooth, 2), " | ".join(f(v, 2) for v in abil)),
             "| **human (phone), over %d driven** | %s | %s |%s" % (n, f(ds_n, 2), f(100 * sr_n, 2), " - |" * 8)]
    lines += ["| %s | %s |" % (m, " | ".join("%.2f" % v for v in row)) for m, *row in LEADERBOARD]
    lines += ["", "Leaderboard rows: Bench2Drive docs/v004_update.md (shipped repo). Tool output: official/tools.log.", ab_note]
    (off / "official.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
