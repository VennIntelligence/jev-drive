#!/usr/bin/env python3
"""AUD1 task 2: the CAM_F0 calibration of a donor scene (K, R, t as the driver logged it) -> a wizard override that renders and serves
CAM_F0 with that calibration on every scene of the run (runtime.extra_cameras, src/runtime/alpasim_runtime/camera_catalog.py).

  aud1_calib_arms.py <per_scene.csv> <donor scene>      -> prints the override (one argv word)
"""
import csv
import json
import sys

import numpy as np


def quat_xyzw(R):
    """Rotation matrix -> unit quaternion (x, y, z, w), the inverse of sh30_driver.quat_R."""
    R = np.asarray(R, np.float64)
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    rows = {r["scene"]: r for r in csv.DictReader(open(sys.argv[1]))}
    r = rows[sys.argv[2]]
    R, t = np.array(json.loads(r["R"])), json.loads(r["t"])
    q = quat_xyzw(R)
    err = np.abs(quat_R(*q) - R).max()
    assert err < 2e-3, err                      # R is logged at 6 decimals and only approximately orthonormal
    f = lambda v: ",".join(f"{x:.6f}" for x in v)  # noqa: E731
    print(f"runtime.extra_cameras=[{{logical_id:CAM_F0,rig_to_camera:{{translation_m:[{f(t)}],rotation_xyzw:[{f(q)}]}},"
          f"intrinsics:{{model:opencv_pinhole,opencv_pinhole:{{focal_length:[{f([float(r['fx']), float(r['fy'])])}],"
          f"principal_point:[{f([float(r['cx']), float(r['cy'])])}],radial:[0,0,0,0,0,0],tangential:[0,0],thin_prism:[0,0,0,0]}}}}}}]")


if __name__ == "__main__":
    main()
