"""Figure: the raw 3-camera strip and the two openpilot model inputs (road on top, wide below, luma) at the step the plan starts to
lean, for spinning and normal Cinque episodes. Reads PNGs written by spin_input_replay.py --images.
    python spin_input_figs.py <replay_out_dir> <jobs.json> <spin_episodes.csv> <out.png>
Each row: raw FRONT_LEFT | FRONT | FRONT_RIGHT (left), road input (512x256) over wide input (512x256) (right)."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(rd, jobs, spin_csv, out):
    rd = Path(rd)
    sp = {r["scenario"]: r for r in csv.DictReader(open(spin_csv)) if r["agent"] == "cinque" and r["controller"] == "fixed"}
    J = json.load(open(jobs))
    pick = [j for j in J if j["scenario"] in ("scene-0013-medium-00", "scene-0528-medium-00", "scene-0254-extreme-00",
                                              "scene-152217047339-medium-00", "scene-0411-easy-00", "scene-0418-hard-00")]
    rows = []
    for j in pick:
        run = Path(j["run_dir"]).name
        k = j["img_steps"][2] if j["tag"] == "spin" else j["img_steps"][1]
        raw = cv2.imread(str(rd / f"{j['tag']}__{run}__raw__step{k}.png"))
        mi = cv2.imread(str(rd / f"{j['tag']}__{run}__base__step{k}.png"))
        raw = cv2.resize(raw, (1200, 225))
        mi = cv2.resize(mi, (512, 256 * 2 // 1))
        mi = cv2.resize(mi, (450, 450 * 512 // 512))
        row = np.full((450, 1200 + 450 + 10, 3), 255, np.uint8)
        row[:225, :1200] = raw
        row[:, 1210:] = mi
        label = f"{'SPIN' if j['tag'] == 'spin' else 'normal'} {j['scenario']} step {k} (t_yaw5 {sp[j['scenario']]['t_yaw5']})"
        cv2.putText(row, label, (8, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        rows.append(row)
    cv2.imwrite(out, cv2.resize(np.concatenate(rows, 0), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main(*sys.argv[1:5])
