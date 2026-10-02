"""Summarise spin_input_replay.py output: plan direction under input variants on the same logged frames.
    python spin_input_summary.py <replay_out_dir> <jobs.json> <spin_episodes.csv> <out.md>
Window = steps 1..t_yaw5 of each episode (until the logged heading error first reaches 5 deg, so the frames are still
aligned with the route); for the long normal episodes the first 30 steps. Plan direction phi = atan2(x, y) of the
plan point at 1 s (+ = right), lateral = plan x at 3 s (m, + = right), both relative to the ego heading.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np


def phi(p, i=1):
    p = np.asarray(p)
    return np.degrees(np.arctan2(p[:, i, 0], np.maximum(p[:, i, 1], 1e-6)))


def main(rd, jobs, spin_csv, out):
    sp = {r["scenario"]: r for r in csv.DictReader(open(spin_csv)) if r["agent"] == "cinque" and r["controller"] == "fixed"}
    keys = ["logged", "base", "front", "yaw:2", "yaw:-2", "yaw:4", "yaw:-4"]
    rows = []
    for j in json.load(open(jobs)):
        R = json.load(open(glob.glob(f"{rd}/{j['tag']}__{Path(j['run_dir']).name}.json")[0]))
        ty = int(sp[j["scenario"]]["t_yaw5"])
        n = len(R["logged"])
        w = slice(1, min(ty + 1, n)) if j["tag"] == "spin" else slice(1, n)
        w = slice(w.start, max(w.stop, 3))
        rows.append(dict(sc=j["scenario"], tag=j["tag"], ds=j["dataset"], ty=ty,
                         **{k: (float(np.mean(phi(R[k][w]))), float(np.mean(np.asarray(R[k])[w, -1, 0]))) for k in keys}))
    L = ["# spin_input_summary (generated)", "",
         "Mean plan direction at 1 s (deg, + = right) / 3 s lateral (m, + = right) over the pre-divergence window.", "",
         "| scenario | group | t_yaw5 | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 3)]
    for r in rows:
        L.append(f"| {r['sc']} | {r['tag']} | {r['ty']} | " + " | ".join("%.1f / %.1f" % r[k] for k in keys) + " |")
    L += ["", "| group | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for g in ("spin", "norm"):
        X = [r for r in rows if r["tag"] == g]
        L.append(f"| {g} (mean of signed values) | {len(X)} | " + " | ".join(
            "%.1f / %.1f" % (np.mean([r[k][0] for r in X]), np.mean([r[k][1] for r in X])) for k in keys) + " |")
        L.append(f"| {g} (mean of abs values) | {len(X)} | " + " | ".join(
            "%.1f / %.1f" % (np.mean([abs(r[k][0]) for r in X]), np.mean([abs(r[k][1]) for r in X])) for k in keys) + " |")
    # agreement of the replayed baseline with the logged plans (video compression is the only difference)
    d = [abs(r["base"][1] - r["logged"][1]) for r in rows]
    L += ["", f"Replay vs logged 3 s lateral, mean |difference| over episodes: {np.mean(d):.2f} m (median {np.median(d):.2f}).", ""]
    # ramp response
    L += ["## Response of the plan to a synthetic yaw ramp (camera turned left by R deg per step, steps 6..13)", "",
          "| scenario | R=0 phi | R=+0.5 | R=-0.5 | R=+1 | R=-1 |", "|---|---|---|---|---|---|"]
    g = []
    for j in json.load(open(jobs)):
        R = json.load(open(glob.glob(f"{rd}/{j['tag']}__{Path(j['run_dir']).name}.json")[0]))
        if "ramp:1" not in R or len(R["ramp:1"]) < 14:
            continue
        v = [float(np.mean(phi(R[k][6:14]))) for k in ("base", "ramp:0.5", "ramp:-0.5", "ramp:1", "ramp:-1")]
        base = float(np.mean(phi(R["base"][6:14])))
        v[0] = base
        g.append(v)
        L.append(f"| {j['scenario']} | " + " | ".join("%.1f" % x for x in v) + " |")
    if g:
        m = np.mean(g, 0)
        L.append("| mean | " + " | ".join("%.1f" % x for x in m) + " |")
        L += ["", "Slope of the plan direction against the ramp rate (deg of plan direction per deg/step of ramp, mean over "
              f"{len(g)} episodes): {(m[3] - m[4]) / 2.0:.2f} (R = +-1), {(m[1] - m[2]) / 1.0:.2f} (R = +-0.5)."]
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:5])
