#!/usr/bin/env python3
"""CPU-only check that the CARLA openpilot rig feeds modeld the exact equivalent of a converged calibration.

Reference: openpilot master (2026-09-29) common/transformations/{camera,model}.py and modeld.py, re-derived here
from the published constants with independent code (explicit Rz Ry Rx, K, view-from-device matrices), not imported.
Compares against jevdrive.openpilot.frames (what scripts/zeroshot_policy_server.py uses) and the CARLA sensor specs
in scripts/zeroshot_rigs.py. Prints one JSON summary and exits non-zero on any failed check.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import zeroshot_rigs as rigs  # noqa: E402
from jevdrive.openpilot import frames as opf  # noqa: E402

# --- reference (transcribed from openpilot master) ----------------------------------------------------------
DEVICE_FROM_VIEW = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
VIEW_FROM_DEVICE = DEVICE_FROM_VIEW.T


def ref_rot(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def K(w, h, f):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]])


MED = np.array([[910., 0, 256], [0, 910., 47.6], [0, 0, 1]])
SBIG = np.array([[455., 0, 256], [0, 455., 0.5 * (256 + 47.6)], [0, 0, 1]])


def ref_warp(rpy, cam_K, big):
    calib_from_model = np.linalg.inv((SBIG if big else MED) @ VIEW_FROM_DEVICE)   # frame_from_calib(0,0,0,0)[:, :3]
    return cam_K @ VIEW_FROM_DEVICE @ ref_rot(rpy) @ calib_from_model


def project(M, uv):
    p = M @ np.array([uv[0], uv[1], 1.])
    return p[:2] / p[2]


def main():
    out, ok = {}, True
    w, h = rigs.OP_CAMERA_WH
    specs = {s["id"]: s for s in rigs.openpilot_sensor_specs()}
    r_spec, w_spec = specs["OP_ROAD"], specs["OP_WIDE"]

    # 1. warp matrices: ours vs reference, converged calibration (rpy = 0) and non-zero
    cases = [np.zeros(3), np.array([0., 0.02, 0.01]), np.array([0.01, -0.03, 0.05])]
    d = [np.abs(opf.get_warp_matrix(c, opf.intrinsics(w, h, f), big) - ref_warp(c, K(w, h, f), big)).max()
         for c in cases for f, big in ((rigs.OP_FOCAL["road"], False), (rigs.OP_FOCAL["wide"], True))]
    out["warp_max_abs_diff"] = float(max(d))
    ok &= max(d) < 1e-9

    # 2. CARLA fov attribute (float32 in CARLA) -> focal actually rendered
    f_carla = {k: w / 2 / math.tan(math.radians(float(np.float32(s["fov"]))) / 2)
               for k, s in (("road", r_spec), ("wide", w_spec))}
    out["carla_focal_px"] = {k: round(v, 3) for k, v in f_carla.items()}
    out["focal_rel_err"] = {k: abs(f_carla[k] / rigs.OP_FOCAL[k] - 1) for k in f_carla}
    ok &= max(out["focal_rel_err"].values()) < 1e-5

    # 3. horizon / principal point: model row 47.6 (road) and 151.8 (wide) must map to the camera principal row
    M_r = ref_warp(np.zeros(3), K(w, h, rigs.OP_FOCAL["road"]), False)
    M_w = ref_warp(np.zeros(3), K(w, h, rigs.OP_FOCAL["wide"]), True)
    out["horizon_model_row_to_cam_px"] = {"road": project(M_r, (256, 47.6)).tolist(),
                                          "wide": project(M_w, (256, 0.5 * (256 + 47.6))).tolist()}
    ok &= np.allclose(project(M_r, (256, 47.6)), [w / 2, h / 2]) and np.allclose(project(M_w, (256, 151.8)), [w / 2, h / 2])

    # 4. ground point at X m, camera height H (level CARLA camera): the model row it lands on
    #    vs openpilot's expectation row = cy_model + f_model * H / X (a level calib frame)
    H = rigs.OP_MOUNT_RIG[2]
    rows = []
    for X in (10., 20., 40., 80.):
        for (M, fm, cym, fc) in ((M_r, 910., 47.6, rigs.OP_FOCAL["road"]), (M_w, 455., 151.8, rigs.OP_FOCAL["wide"])):
            v_cam = h / 2 + fc * H / X                                    # CARLA pinhole render
            u_m = np.linalg.inv(M) @ np.array([w / 2, v_cam, 1.])
            rows.append(abs(u_m[1] / u_m[2] - (cym + fm * H / X)))
    out["ground_row_err_px"] = float(max(rows))
    ok &= max(rows) < 1e-6
    # what the same ground point would be at openpilot's nominal 1.22 m: the row shift the model has to absorb
    out["row_shift_vs_nominal_height_px_at_20m"] = {"road": 910 * (H - 1.22) / 20, "wide": 455 * (H - 1.22) / 20}

    # 5. road / wide relative extrinsics in the CARLA specs: colocated, same orientation => wide_from_device = 0
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    out["road_wide_spec_delta"] = {k: r_spec[k] - w_spec[k] for k in keys}
    out["camera_orientation_deg"] = {k: r_spec[k] for k in ("roll", "pitch", "yaw")}
    ok &= all(v == 0 for v in out["road_wide_spec_delta"].values()) and all(r_spec[k] == 0 for k in ("roll", "pitch", "yaw"))

    # 6. server's precomputed gather indices (zeros calibration) vs the reference matrix
    for name, M, big in (("road", M_r, False), ("wide", M_w, True)):
        mine = opf._nn_index(opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, rigs.OP_FOCAL[name]), big),
                             (opf.MODEL_W, opf.MODEL_H), (w, h))
        ref = opf._nn_index(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
        out["gather_idx_equal_" + name] = bool(np.array_equal(mine, ref))
        ok &= out["gather_idx_equal_" + name]

    # 7. size of a mis-calibration in the model frame (what a 'wrong rpy' arm would change): pixel shift of the
    #    model-frame image of the horizon and of a point 30 deg... in the road model frame, rows / columns
    def shift(rpy):
        M = ref_warp(rpy, K(w, h, rigs.OP_FOCAL["road"]), False)
        p = project(np.linalg.inv(M), (w / 2, h / 2))                      # cam principal point in the model frame
        return (p - [256, 47.6]).tolist()
    out["road_model_frame_shift_px"] = {"pitch+0.5deg": shift([0, math.radians(.5), 0]),
                                        "yaw+0.5deg": shift([0, 0, math.radians(.5)]),
                                        "pitch+1deg": shift([0, math.radians(1), 0])}
    out["all_ok"] = bool(ok)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
