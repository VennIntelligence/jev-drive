"""BridgeDrive as shipped for the CL10 closed-loop arm (todos/2026-09-26-night-queue-3.md, section CL).

The author's LEAD `SensorAgent` from `lead/inference/sensor_agent_bridgedrive.py` (BridgeDrive 85aa089 copied into
its pinned lead a41d116), with its own rig, preprocessing, model, route + target-speed PIDs and post-processors.
Nothing that touches control is changed. Three start-up shims, the same ones as the T3 smoke
($DATA_DIR/third_party/bridgedrive/jev_smoke.py), because the released code does not start otherwise:
  1. TrainingConfig.gpu_name raises on any GPU outside the author's list (ours: RTX PRO 6000 Blackwell); it only
     feeds training-time mixed-precision switches, so an unknown card maps to "" (the author's no-GPU value).
  2. config_closed_loop.py reads self.debug_mode in every produce_* property but never defines it; define it False,
     as the author's '# Shu' branches intend for evaluation (every produce_* output is then off).
  3. setup() refuses to start without ffmpeg, which only compresses videos that are never produced.
The anchor path (relative to the lead root in the author's script) is made absolute through the author's own
LEAD_TRAINING_CONFIG by scripts/nq3_b_cl10.sh, not here. SAVE_PATH defaults to <attempt>/lead_save, so the
author's per-route metric_info.json / infractions.json land with the attempt instead of one shared directory.
Env: envs/bridgedrive, PYTHONPATH=<lead>. Agent config: the model dir ($DATA_DIR/models/bridgedrive).
"""
import os
from unittest import mock

import lead.inference.config_closed_loop as cc
import lead.training.config_training as ct
from lead.inference.sensor_agent_bridgedrive import SensorAgent

cc.ClosedLoopConfig.debug_mode = False
_gpu_name = ct.TrainingConfig.gpu_name.fget


def _safe_gpu_name(self):
    try:
        return _gpu_name(self)
    except Exception:
        return ""


ct.TrainingConfig.gpu_name = property(_safe_gpu_name)


def get_entry_point():
    return "BridgeDriveAgent"


class BridgeDriveAgent(SensorAgent):
    def setup(self, path_to_conf_file, *args, **kwargs):
        if "B2D_ATTEMPT_OUT" in os.environ:
            os.environ["SAVE_PATH"] = os.path.join(os.environ["B2D_ATTEMPT_OUT"], "lead_save")
        with mock.patch("shutil.which", return_value="/bin/true"):
            super().setup(path_to_conf_file, *args, **kwargs)
