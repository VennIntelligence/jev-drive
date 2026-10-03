"""HUGSIM camera height above ground per dataset, from a run's ground.ply (median ground y under the ego, OpenCV y down).
CPU. $DATA_DIR/envs/hugsim/bin/python experiments/leaderboard_audit/scripts/bv_hugsim_height.py"""
import csv, os, pickle, json
from pathlib import Path
import numpy as np
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
R = Path(__file__).resolve().parents[3]
rows = [r for r in csv.DictReader(open(R / "experiments/leaderboard_audit/results/loss_budget/hugsim_inputs/exam_scored_op.csv")) if r["tag"] == "cinque-fixed"]
out = {}
for ds in ("nuscenes", "pandaset", "waymo", "kitti360"):
    hs = []
    for r in [r for r in rows if r["dataset"] == ds][:3]:
        d = Path(r["run_dir"].replace("/root/autodl-tmp/ujs", str(D)))
        try:
            from plyfile import PlyData
            infos = pickle.load(open(d / "infos.pkl", "rb"))
            v = PlyData.read(str(d / "ground.ply"))["vertex"]
            P = np.stack([v["x"], v["y"], v["z"]], 1)
            for i in (0, len(infos) // 2):
                e = np.asarray(infos[i]["ego_pos"], float)
                m = np.hypot(P[:, 0] - e[0], P[:, 2] - e[2]) < 3.0
                if m.sum() > 20:
                    hs.append(float(np.median(P[m, 1]) - e[1]))
        except Exception as ex:
            print(ds, d, repr(ex)[:150])
    out[ds] = hs
print(json.dumps(out))
