#!/usr/bin/env python3
"""AUD1 task 2: control vs donor-calibration runs of the same scenes (box, numpy): scores, GT distance, the calibration the driver received,
the first plans' lateral offsets, and whether the rendered CAM_F0 JPEG changed.

  aud1_calib_compare.py <control run dir> <arm run dir> [<arm run dir> ...]  -> json on stdout
"""
import io
import json
import sys
from pathlib import Path

import numpy as np


def read(D):
    D = Path(D)
    R = {r["clipgt_id"]: r for r in json.loads((D / "aggregate/results-summary.json").read_text())["rollouts"]}
    st, plans = {}, {}
    for line in (D / "driver-logs/drive.jsonl").open():
        x = json.loads(line)
        if x["kind"] == "start":
            st[x["scene"]] = x
        elif x["kind"] == "drive":
            plans.setdefault(x["scene"], {})[x["k"]] = np.array(x["poses"])
    dump = {}
    for f in sorted((D / "driver-logs/dump").glob("s*_k0.npz")):
        z = np.load(f, allow_pickle=True)
        dump[str(z["scene"])] = z
    return R, st, plans, dump


def main():
    C = read(sys.argv[1])
    out = {}
    for arm in sys.argv[1:]:
        A = read(arm)
        res = {}
        for s in sorted(C[0]):
            r, m = A[0][s], A[0][s]["metrics"]
            row = {"score": r["score"], "status": r["status"], "dist_to_gt_m": m.get("dist_to_gt_trajectory"),
                   "lat_dist_to_gt_m": m.get("lateral_dist_to_gt_trajectory"), "fail": r.get("failure_reason")}
            if s in A[1] and "cam" in A[1][s]:
                c = A[1][s]["cam"]
                row["cam"] = {"fx": c["K"][0][0], "fy": c["K"][1][1], "t": c["t"],
                              "pitch_deg": float(np.degrees(np.arcsin(-np.array(c["R"])[2][2]))),
                              "yaw_deg": float(np.degrees(np.arctan2(np.array(c["R"])[1][2], np.array(c["R"])[0][2])))}
            P, P0 = A[2].get(s, {}), C[2].get(s, {})
            # plan k (8 x 0.5 s in the decision's own frame): end point and its lateral difference to the control's plan k
            row["plan_end_y"] = {k: round(float(P[k][-1, 1]), 3) for k in sorted(P)[:4]}
            row["plan_end_x"] = {k: round(float(P[k][-1, 0]), 3) for k in sorted(P)[:4]}
            row["dy_vs_control"] = {k: round(float(np.abs(P[k][:, 1] - P0[k][:, 1]).max()), 3) for k in sorted(P)[:4] if k in P0}
            row["dx_end_vs_control"] = {k: round(float(P[k][-1, 0] - P0[k][-1, 0]), 3) for k in sorted(P)[:4] if k in P0}
            if s in A[3] and s in C[3] and "jpeg" in A[3][s].files:
                a, b = A[3][s]["jpeg"].tobytes(), C[3][s]["jpeg"].tobytes()
                row["k0_jpeg_identical_to_control"] = a == b
                try:
                    from PIL import Image
                    ia = np.asarray(Image.open(io.BytesIO(a)).convert("L"), np.float32)
                    ib = np.asarray(Image.open(io.BytesIO(b)).convert("L"), np.float32)
                    row["k0_jpeg_mean_abs_diff"] = float(np.abs(ia - ib).mean()) if ia.shape == ib.shape else f"shape {ia.shape} vs {ib.shape}"
                except Exception as e:  # noqa: BLE001
                    row["k0_jpeg_mean_abs_diff"] = str(e)
            if s in A[3] and s in C[3]:
                fa, fb = A[3][s]["cur"][-1].astype(np.float32), C[3][s]["cur"][-1].astype(np.float32)
                row["k0_model_frame_mean_abs_diff"] = float(np.abs(fa - fb).mean())   # newest slot, road + wide, 0-255
            res[s] = row
        out[Path(arm).parent.name] = res
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
