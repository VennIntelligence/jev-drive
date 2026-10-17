"""sh30_driver.py with JEV_HOLD (lib/static_hold.py): the same driver, its served plan passed through static_hold.step before it becomes
the trajectory. sh30_driver.py itself is not changed (arena reads were running it when this was added): this module wraps it.

JEV_HOLD unset / off: sh30_driver.main() runs with nothing patched (the served output is sh30_driver's by construction).
JEV_HOLD on: three patches, all inside this process -
  Driver.drive         records the session of the call (thread-local) and keeps one static_hold state per session
  C.Core.plan          keeps the core's outputs of the call (thread-local); for JEV_HOLD=lead also returns the lead outputs (lead_out)
  make_cached_plan     the poses it receives (the plan after every other serving switch) -> static_hold.step -> the original function
Logged per decision to $ALPASIM_DRIVER_LOG_DIR/hold.jsonl (session, scene, t0, the step record; the served poses when cut); drive.jsonl's
`poses` stay the model's plan. JEV_TRACK is refused (its reference does not go through make_cached_plan).
Run: run.sh <run dir> sh30hold ... (scripts/drivers/sh30hold.sh), JEV_HOLD in the environment.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

import sh30_driver as D
import static_hold as HD

TL = threading.local()


def install(log_dir: Path) -> None:
    assert not D.TR.ON, "JEV_HOLD: JEV_TRACK must be off (the feasible reference bypasses make_cached_plan)"
    Path(log_dir).mkdir(parents=True, exist_ok=True)              # the Driver makes it later; install runs first
    out = open(Path(log_dir) / "hold.jsonl", "a", buffering=1)
    states, lock = {}, threading.Lock()
    drive0, plan0, make0 = D.Driver.drive, D.C.Core.plan, D.make_cached_plan

    def drive(self, req, ctx):
        TL.s, TL.uuid, TL.o = self.sessions.get(req.session_uuid), req.session_uuid, None
        return drive0(self, req, ctx)

    def plan(self, *a, **k):
        if HD.NEED_LEAD:
            self.lead_out = True
        o = plan0(self, *a, **k)
        TL.o = o
        return o

    def make_cached_plan(t0, anchor, poses):
        s = getattr(TL, "s", None)
        if s is None:                                    # not inside a drive call (never in serving)
            return make0(t0, anchor, poses)
        with lock:
            st = states.setdefault(TL.uuid, HD.new())
        v = float(np.hypot(*s.velocity(t0)[0]))
        served, info = HD.step(st, poses, v, (anchor.pose.vec.x, anchor.pose.vec.y), t0, TL.o)
        rec = dict(session=TL.uuid, scene=s.scene, t0=t0, **info)
        if info.get("cut"):
            rec["poses"] = np.asarray(served).round(4).tolist()
        out.write(json.dumps(rec) + "\n")
        return make0(t0, anchor, served)

    D.Driver.drive, D.C.Core.plan, D.make_cached_plan = drive, plan, make_cached_plan
    D.LOG.info("serving JEV_HOLD=%s (lib/static_hold.py): latch below %g m/s at the %g s call, leash %g m, release arc %g m / lead %g m/s, "
               "stop leash below %g m/s^2 within %g s", HD.MODE, HD.V_LATCH, HD.T_HAND, HD.CAP, HD.REL_ARC, HD.V_LEAD_GO, HD.A_STOP, HD.T_STOP)


if __name__ == "__main__":
    if HD.ON:
        install(Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver")))
    D.main()
