"""Per-run behaviour table for the WA-JEPA reference (results/wajepa_ref.md), run on the box (needs infos.pkl, data.pkl, viz/).
For wajepa (runs/hugsim-wajepa) and cinque-fixed (runs/hugsim-exam/scored-op): spin (heading error vs the recorded route >= 60 deg, the
definition of results/controller_spin.md, scripts/spin_analysis.py), launch speed profile, standing share, WA-JEPA fallback count and the
mean brightness of its four input tiles (the 4th is CAM_BACK).
    $DATA_DIR/envs/hugsim/bin/python experiments/hugsim/scripts/wajepa_extract.py <out_dir>
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spin_analysis as SA  # noqa: E402

D = Path(os.environ["DATA_DIR"])
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
routes_json = out / "routes.json"
if not routes_json.exists():
    subprocess.run([sys.executable, str(HERE / "spin_export_routes.py"), str(HERE.parent / "results/hugsim-exam/scored_op.csv"), str(routes_json)], check=True)
routes = json.load(open(routes_json))
rows = []
srcs = [("wajepa", D / "runs/hugsim-wajepa/results.csv", None), ("cinque-fixed", HERE.parent / "results/hugsim-exam/scored_op.csv", "cinque-fixed")]
for tag, csvp, want in srcs:
    for r in csv.DictReader(open(csvp)):
        if (want or tag) != r["tag"]:
            continue
        d = Path(r["run_dir"])
        if not (d / "infos.pkl").exists():
            print("missing", d)
            continue
        pos, th, v, steer, plans = SA.load_run(d, r["agent"])
        res, ser = SA.analyse(pos, th, v, steer, plans, routes[r["scene"]])
        e = ser["e"]
        row = dict(tag=tag, scenario=r["scenario"], max_abs_e=res["max_abs_e"], spin=bool(res["spin"]), k60=res.get("k60", -1),
                   v_max40=float(v[:40].max()), v_5s=float(v[min(20, len(v) - 1)]), v_max=float(v.max()), v_end=float(v[-1]),
                   standing=float((v < 0.3).mean()), n=len(v), end_heading_err=float(np.degrees(e[-1])))
        if tag == "wajepa":
            ps = json.load(open(d / "planner_stats.json"))
            row.update(n_failures=ps["n_failures"], n_hist_pad=ps["n_hist_pad"], lat_p50=ps.get("latency_summary", {}).get("p50_ms", ""))
            try:
                import cv2
                im = cv2.imread(str(d / "viz/input/0010.jpg"))
                h, w = im.shape[0] // 2, im.shape[1] // 2
                row["back_mean"] = float(im[h:, w:].mean())            # 2x2 mosaic, 4th tile = cam_b0 = CAM_BACK
                row["front_mean"] = float(im[:h, w:].mean())            # 2nd tile
            except Exception as ex:
                row["back_mean"] = row["front_mean"] = ""
        rows.append(row)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out / "wajepa_extract.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "rows ->", out / "wajepa_extract.csv")
