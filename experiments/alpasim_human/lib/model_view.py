"""A served model driver with its own view, for the human exam's model mode (README "Model mode"): the driver of a submission image runs
unchanged, and what it fed the policy is written to the live dir for the viewer. Display only: nothing it does reaches the plan.

    python model_view.py pai | nuplan          (inside a submission image; scripts/model_driver.sh starts it)

pai     the image's experiments/alpasim/lib/pai_driver.py (PAI images: the code of the submission, the served switches baked into the
        image's ENV). Hook: pai_core.plan, called by pai_driver with the 8 slot frames it built from the rendered 10 Hz front camera (the
        session's own f-theta Maps at submit time) and their validity.
nuplan  the image's sh30_driver.py. Hook: sh30_core.Core.plan, which synthesises the 8 slots on its 0.2 s clock from the 2 Hz keyframes
        and returns them ("cur", "valid").
Both: only calls made inside the driver's `drive` count (the warm-up passes do not). Per decision the hook copies only the wide model
frame of the newest slot (the frame at the decision time t0, 196 KB) and a writer thread (latest value only: a decision that arrives
while the previous one is still being written replaces it, so a slow disk never delays the model) puts into HUMAN_LIVE (atomic renames):
  model.jpg        the wide model frame at t0 (512 x 256, RGB, quality 80): the widest view the policy gets
  model.json       {t, n, track, tag, valid (8), poses (8, 3: the model's own plan, rear axle x, y, yaw at 0.5 .. 4 s in the t0 frame, before
                   any serving switch), cam_t, K_wide, view_from_device}: the viewer projects a rig-frame point p into the frame
                   as K_wide @ view_from_device @ ((p - cam_t) * [1, -1, -1])
Env: HUMAN_LIVE (required), SH30_TAG (the tag, as the drivers read it), JEV_LIB (the image's driver directory), HUMAN_MODEL_VIEW (1;
0 = no hook, the image's driver alone: for measuring the view's cost).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

LIB = os.environ.get("JEV_LIB", "/app/jev-drive/experiments/alpasim/lib")
sys.path.insert(0, LIB)
LIVE = Path(os.environ["HUMAN_LIVE"])
LOG = logging.getLogger("model_view")
TLS = threading.local()                    # TLS.on: inside the driver's drive()


def put(p: Path, data: bytes):
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


class View:
    """Latest-value writer: put() is O(copy) on the gRPC thread, the encoding happens here."""

    def __init__(self, track: str, I):
        self.track, self.I, self.item, self.cv, self.n = track, I, None, threading.Condition(), 0
        from jevdrive.openpilot.frames import SBIGMODEL_K, VIEW_FROM_DEVICE
        self.K = {"K_wide": SBIGMODEL_K.tolist(), "view_from_device": VIEW_FROM_DEVICE.tolist()}
        threading.Thread(target=self._run, daemon=True, name="model-view").start()

    def put(self, cur, valid, poses, cam_t):
        if not getattr(TLS, "on", False):
            return
        try:
            w = cur[-1, 1]                                         # newest slot, wide frame: (6, 128, 256) packed YUV420
            w = w.cpu().numpy() if hasattr(w, "cpu") else np.array(w)
            item = (time.time(), w, np.asarray(valid, bool).copy(), np.asarray(poses, float).copy(), list(map(float, cam_t)))
        except Exception as e:  # noqa: BLE001  (display only)
            LOG.warning("view put: %r", e)
            return
        with self.cv:
            self.item = item
            self.cv.notify()

    def rgb(self, p):
        """(6, 128, 256) packed YUV420 -> (256, 512, 3) BGR (JPEG YCbCr, chroma replicated), as frame_fill.model_rgb."""
        import cv2
        Y, U, V = self.I.unpack(p)
        U, V = (cv2.resize(c, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_NEAREST) for c in (U, V))
        return cv2.cvtColor(np.dstack([Y, V, U]), cv2.COLOR_YCrCb2BGR)

    def _run(self):
        import cv2
        cv2.setNumThreads(1)
        while True:
            with self.cv:
                while self.item is None:
                    self.cv.wait()
                (t, wide, valid, poses, cam_t), self.item = self.item, None
            try:
                put(LIVE / "model.jpg", cv2.imencode(".jpg", self.rgb(wide), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                self.n += 1
                put(LIVE / "model.json", json.dumps({"t": t, "n": self.n, "track": self.track, "tag": os.environ.get("SH30_TAG"),
                                                     "valid": valid.tolist(), "poses": poses.round(3).tolist(), "cam_t": cam_t, **self.K}).encode())
            except Exception as e:  # noqa: BLE001
                LOG.warning("view write: %r", e)
                time.sleep(0.2)


def in_drive(cls):
    """cls.drive with TLS.on set for its duration."""
    f = cls.drive

    def drive(self, req, ctx):
        TLS.on = True
        try:
            return f(self, req, ctx)
        finally:
            TLS.on = False
    cls.drive = drive


def main():
    track = sys.argv[1]
    LIVE.mkdir(parents=True, exist_ok=True)
    if os.environ.get("HUMAN_MODEL_VIEW", "1") == "0" and track in ("pai", "nuplan"):   # no hook: the image's driver as it ships
        __import__("pai_driver" if track == "pai" else "sh30_driver").main()
        return
    if track == "pai":
        import pai_driver as D
        import pai_core as PC
        view, plan0 = View(track, PC.I), PC.plan
        sig = inspect.signature(plan0)

        def plan(*a, **k):
            o = plan0(*a, **k)
            b = sig.bind(*a, **k).arguments
            view.put(b["cur"], b["valid"], o["poses"], b["cam_t"])
            return o
        PC.plan = plan                     # pai_driver calls PC.plan(...) through the module
    elif track == "nuplan":
        import sh30_driver as D
        C = D.C
        view, plan0 = View(track, C.I), C.Core.plan
        sig = inspect.signature(plan0)

        def plan(self, *a, **k):
            o = plan0(self, *a, **k)
            view.put(o["cur"], o["valid"], o["poses"], sig.bind(self, *a, **k).arguments["cam_t"])
            return o
        C.Core.plan = plan
    else:
        sys.exit("track pai | nuplan")
    in_drive(D.Driver)
    D.main()


if __name__ == "__main__":
    main()
