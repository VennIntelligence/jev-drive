"""WOD ground height in the vehicle frame: box bottoms (z - h/2) of labelled vehicles near the ego, per segment, from the
perception v2 parquet (lidar_box + camera_calibration). Also the E2E front-camera extrinsic z from op_calib.json.
CPU. $DATA_DIR/envs/p3-wodprep/bin/python experiments/leaderboard_audit/scripts/sc_wod_ground.py [out.json]"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
V2 = D / "datasets/waymo_perception/v2"
P = "[LiDARBoxComponent]."
out = {"seg": []}
allb = []
for split in ("training", "validation"):
    for f in sorted(glob.glob(str(V2 / split / "lidar_box" / "*.parquet"))):
        seg = Path(f).stem
        t = pq.read_table(f, columns=["key.frame_timestamp_micros", P + "type", P + "box.center.x", P + "box.center.y",
                                      P + "box.center.z", P + "box.size.z", P + "speed.x", P + "num_lidar_points_in_box"]).to_pydict()
        x, y, z, h = (np.array(t[P + f"box.{k}"]) for k in ("center.x", "center.y", "center.z", "size.z"))
        ty, n, ts = np.array(t[P + "type"]), np.array(t[P + "num_lidar_points_in_box"]), np.array(t["key.frame_timestamp_micros"])
        m = (ty == 1) & (np.hypot(x, y) < 30) & (np.hypot(x, y) > 4) & (n > 50) & (h < 3.5)
        cal = pq.read_table(str(V2 / split / "camera_calibration" / (seg + ".parquet"))).to_pydict()
        i = [k for k, c in enumerate(cal["key.camera_name"]) if c == 1][0]
        ez = float(cal["[CameraCalibrationComponent].extrinsic.transform"][i][11])
        ex = float(cal["[CameraCalibrationComponent].extrinsic.transform"][i][3])
        if m.sum() < 10:
            continue
        out["seg"].append(dict(seg=seg, split=split, n=int(m.sum()), bottom_med=float(np.median((z - h / 2)[m])), cam_z=ez, cam_x=ex))
        allb.append((z - h / 2)[m])
b = np.concatenate(allb)
s = out["seg"]
bm = np.array([r["bottom_med"] for r in s]); cz = np.array([r["cam_z"] for r in s])
out["summary"] = dict(n_seg=len(s), n_boxes=int(len(b)), bottom_pooled_med=float(np.median(b)), bottom_q=np.percentile(b, [5, 25, 75, 95]).tolist(),
                      bottom_seg_med_of_med=float(np.median(bm)), bottom_seg_q=np.percentile(bm, [5, 25, 75, 95]).tolist(),
                      cam_z_med=float(np.median(cz)), cam_z_q=np.percentile(cz, [0, 5, 50, 95, 100]).tolist(),
                      cam_height_above_ground_med=float(np.median(cz - bm)), cam_height_q=np.percentile(cz - bm, [5, 25, 50, 75, 95]).tolist())
oc = D / "processed/wod_zeroshot/op_calib.json"
if oc.exists():
    c = json.load(open(oc))
    z = np.array([np.array(v["1"]["extrinsic"]).reshape(4, 4)[2, 3] for v in c.values()])
    out["e2e_cam_z"] = dict(n_seq=len(z), q=np.percentile(z, [0, 5, 50, 95, 100]).tolist())
print(json.dumps(out["summary"], indent=1)); print(json.dumps(out.get("e2e_cam_z")))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"))
