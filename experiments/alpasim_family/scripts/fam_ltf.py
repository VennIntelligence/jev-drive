"""FAMILY: the shipped SimScale LTF sample (navsim_transfuser_challenge.driver, imported and run unchanged) with FAM_ARM applied
(fam_perturb.py): calibration arms rewrite every announced camera of the start_session request (the sample does not read them; this
shows it), image arms rewrite the CAM_L0 / F0 / R0 JPEGs before its submit_image_observation. Run by
experiments/alpasim/scripts/drivers/famltf.sh (envs/alpasim-ltf, cwd = the sample directory)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fam_perturb as P  # noqa: E402
from navsim_transfuser_challenge import driver as L  # noqa: E402

A = P.arm()
_start, _img = L.NavsimTransfuserDriver.start_session, L.NavsimTransfuserDriver.submit_image_observation


def _quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)


def start(self, req, ctx):
    for c in req.rollout_spec.vehicle.available_cameras:
        p, q, v = c.intrinsics.opencv_pinhole_param, c.rig_to_camera.quat, c.rig_to_camera.vec
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        cam = P.told({"K": [[p.focal_length_x, 0, p.principal_point_x], [0, p.focal_length_y, p.principal_point_y], [0, 0, 1]],
                      "R": R, "t": [v.x, v.y, v.z], "D": [0, 0, 0, 0, 0]}, "", {k: x_ for k, x_ in A.items() if k != "D"})
        p.focal_length_x, p.focal_length_y, p.principal_point_x, p.principal_point_y = map(float, (cam["K"][0, 0], cam["K"][1, 1], cam["K"][0, 2], cam["K"][1, 2]))
        q.w, q.x, q.y, q.z = map(float, _quat(cam["R"].astype(np.float64)))
        v.x, v.y, v.z = map(float, cam["t"])
    return _start(self, req, ctx)


def submit_image(self, req, ctx):
    if req.camera_image.logical_id in ("CAM_L0", "CAM_F0", "CAM_R0"):
        req.camera_image.image_bytes = P.image(req.camera_image.image_bytes, A)
    return _img(self, req, ctx)


L.NavsimTransfuserDriver.start_session = start
L.NavsimTransfuserDriver.submit_image_observation = submit_image

if __name__ == "__main__":
    L.LOGGER.warning("FAM_ARM %s", A or "control")
    L.main()
