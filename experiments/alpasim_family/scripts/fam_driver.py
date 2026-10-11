"""FAMILY: the served sh30 driver (experiments/alpasim/lib/sh30_driver.py, imported and run unchanged) with FAM_ARM applied
(fam_perturb.py): the CAM_F0 calibration a session is told is replaced right after Session() reads it, and image arms rewrite the
JPEGs before submit_image_observation. The start record in drive.jsonl logs the calibration actually used. Run by
experiments/alpasim/scripts/drivers/fam.sh (envs/op-train)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "alpasim/lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fam_perturb as P  # noqa: E402
import sh30_driver as S  # noqa: E402

A = P.arm()
_init, _img = S.Session.__init__, S.Driver.submit_image_observation


def init(self, req, n):
    _init(self, req, n)
    self.cam = P.told(self.cam, self.scene, A)


def submit_image(self, req, ctx):
    if req.camera_image.logical_id == S.CAM:
        req.camera_image.image_bytes = P.image(req.camera_image.image_bytes, A)
    return _img(self, req, ctx)


S.Session.__init__ = init
S.Driver.submit_image_observation = submit_image

if __name__ == "__main__":
    logging.getLogger("sh30").warning("FAM_ARM %s", A or "control")
    S.main()
