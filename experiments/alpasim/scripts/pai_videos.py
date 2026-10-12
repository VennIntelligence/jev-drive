#!/usr/bin/env python3
"""Flat symlink index of PAI rollout videos for watching on the Tokyo desktop.

Writes ~/Desktop/pai_videos/<arm>/<score>_<zero kind>_<scene8>.mp4 -> the front-wide mp4 of that rollout,
and ~/Desktop/pai_videos/by_scene/<scene8>/<arm>_<score>.mp4 for side-by-side comparison of arms.
Idempotent: rerun to pick up new chunks. Usage: pai_videos.py [ROOT] (default /data/runs/onpol_long/pai281).
"""
import glob, json, os, shutil, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/runs/onpol_long/pai281")
OUT = Path.home() / "Desktop" / "pai_videos"
FAIL = ["collision_at_fault", "offroad", "left_corridor_laterally"]
SHORT = {"collision_at_fault": "crash", "offroad": "offroad", "left_corridor_laterally": "corridor"}

if OUT.exists():
    shutil.rmtree(OUT)
n = 0
for summ in sorted(glob.glob(str(ROOT / "*/runs/*/sim/aggregate/results-summary.json"))):
    summ = Path(summ)
    run_dir = summ.parents[2]
    arm = summ.parents[4].name
    for r in json.loads(summ.read_text())["rollouts"]:
        scene, rid, score = r["clipgt_id"], r["rollout_id"], r["score"]
        mp4 = sorted((run_dir / "sim/rollouts" / scene / rid).glob("*camera_front_wide*.mp4"))
        if not mp4 or score is None:
            continue
        kind = next((SHORT[f] for f in FAIL if (r.get("score_metrics") or {}).get(f)), "ok" if score > 0 else "zero")
        s8 = scene.replace("clipgt-", "")[:8]
        a = OUT / arm / f"{score:.2f}_{kind}_{s8}.mp4"
        b = OUT / "by_scene" / s8 / f"{arm}_{score:.2f}_{kind}.mp4"
        for ln in (a, b):
            ln.parent.mkdir(parents=True, exist_ok=True)
            ln.symlink_to(mp4[0])
        n += 1
print(f"{n} videos linked under {OUT}")
