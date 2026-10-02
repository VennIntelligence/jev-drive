"""Ego vs scripted-actor tracks of a HUGSIM run (run on the box with the hugsim env: numpy only).

usage: attack_tracks.py <run_dir> [out.csv]
infos.pkl holds per step: ego_pos / ego_rot / ego_velo / accelerate / obj_boxes ([x, y, z, w, l, h, yaw] in the scene
frame, x forward at the start, y lateral). Prints the per-step table (0.25 s) of ego speed, each actor's position,
speed (finite difference), distance and closing speed, and the step of the first overlap.
"""
import csv, math, pickle, sys
import numpy as np
d = pickle.load(open(sys.argv[1] + "/infos.pkl", "rb")); dt = 0.25
rows = []
prev = None
for i, s in enumerate(d):
    eb = s["ego_box"]; ex, ey, eyaw = eb[0], eb[1], eb[6]
    row = {"step": i, "t": round(i * dt, 2), "ego_x": ex, "ego_y": ey, "ego_v": s["ego_velo"], "ego_acc": s["accelerate"], "ego_yaw_deg": math.degrees(-eyaw)}
    for k, o in enumerate(s["obj_boxes"]):
        row[f"a{k}_x"], row[f"a{k}_y"], row[f"a{k}_yaw_deg"] = o[0], o[1], math.degrees(o[6])
        if prev is not None:
            p = prev["obj_boxes"][k]; row[f"a{k}_v"] = math.hypot(o[0] - p[0], o[1] - p[1]) / dt
        else: row[f"a{k}_v"] = float("nan")
        row[f"a{k}_dist"] = math.hypot(o[0] - ex, o[1] - ey)
        row[f"a{k}_lat"] = o[1] - ey
    rows.append(row); prev = s
keys = list(rows[0].keys())
w = csv.DictWriter(open(sys.argv[2], "w", newline="") if len(sys.argv) > 2 else sys.stdout, fieldnames=keys)
w.writeheader()
for r in rows: w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
