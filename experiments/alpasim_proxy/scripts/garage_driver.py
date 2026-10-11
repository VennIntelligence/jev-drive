"""Starts the shipped py123d_garage AlpaSim driver (py123d_garage.evaluation.alpasim.evaluate, v0.1.1) unchanged, plus an observer that
writes one close record per session to $ALPASIM_DRIVER_LOG_DIR/drive.jsonl in the format c0b_chain.driver_health reads.

The garage driver only logs: a failed inference (`session X failed to plan.`) or missing inputs (`session X has [..] of [..] cameras and
no ego pose; not planning.`) make it serve the previous plan or a straight line, and nothing else records it. The observer counts those
log records per session (inference_error / input_error), plus drive calls, successful plans and used camera frames, by wrapping
GarageDriver methods. It passes every argument and return value through and touches no driver state.
  python garage_driver.py <the evaluate.py hydra overrides>
"""
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from py123d_garage.evaluation.alpasim import evaluate
from py123d_garage.evaluation.alpasim.help import driver_service as ds

LOG_DIR = Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "driver-logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT = (LOG_DIR / "drive.jsonl").open("a", buffering=1)
LOCK = threading.Lock()
STATS: dict[str, dict] = {}
ERR = re.compile(r"session (\S+) (failed to plan|has .* not planning)")


def bump(uuid, key):
    with LOCK:
        s = STATS.setdefault(uuid, dict(drive=0, inference=0, inference_error=0, input_error=0, images=0, t0=time.time()))
        s[key] += 1


class ErrorCounter(logging.Handler):
    def emit(self, record):
        m = ERR.match(record.getMessage())
        if m:
            bump(m[1], "inference_error" if m[2] == "failed to plan" else "input_error")


def after(name, fn):
    """Replaces GarageDriver.<name> by a wrapper that calls the original, then fn(self, args, result), and returns the original's result."""
    orig = getattr(ds.GarageDriver, name)

    def wrapped(self, *args):
        out = orig(self, *args)
        fn(self, args, out)
        return out
    setattr(ds.GarageDriver, name, wrapped)


def close(self, args, _):
    uuid = args[0].session_uuid
    with LOCK:
        s = STATS.pop(uuid, None) or dict(drive=0, inference=0, inference_error=0, input_error=0, images=0, t0=time.time())
    t0 = s.pop("t0")
    OUT.write(json.dumps(dict(kind="close", t=time.time(), session=uuid, wall_s=round(time.time() - t0, 2), **s)) + "\n")


logging.getLogger(ds.__name__).addHandler(ErrorCounter(logging.WARNING))
after("drive", lambda self, a, _: bump(a[0].session_uuid, "drive"))
after("_plan", lambda self, a, _: bump(a[2], "inference"))
after("submit_image_observation",
      lambda self, a, _: a[0].camera_image.logical_id in self._camera_id_by_alpasim_id and bump(a[0].session_uuid, "images"))
after("close_session", close)

if __name__ == "__main__":
    evaluate.main()
