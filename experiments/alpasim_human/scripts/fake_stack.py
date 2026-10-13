#!/usr/bin/env python3
"""Stand-in for human_stack.sh in the foreground (HUMAN_FG=1) to test scripts/human_loop.py and the panel without a GPU stack.

    HUMAN_STACK_SCRIPT=experiments/alpasim_human/scripts/fake_stack.py python3 experiments/alpasim_human/scripts/human_loop.py --run ...

Same call (<track> <out dir> <scene list>) and the same files the loop reads: <out>/driver/sessions.jsonl and drive.jsonl close
records, <live>/state.json (run = out dir, scene, waiting, closed), sim/rollouts/<scene>/<id>/{metrics.parquet, _complete} and
sim/eval-config.yaml, ERROR on failure. Per scene: FAKE_LOAD_S of loading, a hold until the start (FAKE_START_S after the hold
begins, or a throttle > 0.2 on HUMAN_WHEEL_PORT; never while control.json says paused), FAKE_DRIVE_S of driving (frozen while
paused), then the rollout files. abort.json skips the scene as lib/human_driver.py does. metrics.parquet is copied from a real
earlier rollout of the same scene (FAKE_METRICS_FROM: a run dir with sim/rollouts) so the loop's scoring runs AlpaSim's own
aggregation; scenes without one get an empty file (scoring then fails, as it should). Stdlib only.
"""
import json
import os
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path

E = os.environ.get
track, out, lst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
live = Path(E("HUMAN_LIVE", "/data/runs/alpasim_human/live"))
ctl = Path(E("HUMAN_CONTROL", str(live / "control.json")))
src = Path(E("FAKE_METRICS_FROM", "/data/runs/alpasim_human/pai_1013-2305_rt"))
LOAD, START, DRIVE = float(E("FAKE_LOAD_S", "3")), float(E("FAKE_START_S", "2")), float(E("FAKE_DRIVE_S", "6"))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
ids = [x for x in lst.read_text().split() if x]
(out / "driver").mkdir(parents=True, exist_ok=True)
(out / "sim").mkdir(exist_ok=True)
shutil.copy(str(next(src.glob("sim/eval-config.yaml"))), str(out / "sim/eval-config.yaml"))
(out / "sim/docker-compose.yaml").write_text("services: {}\n")
sess, drive = open(str(out / "driver/sessions.jsonl"), "a", buffering=1), open(str(out / "driver/drive.jsonl"), "a", buffering=1)


def j(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return d


def state(**kw):
    tmp = live / ".state.fake"
    tmp.write_text(json.dumps(dict({"run": str(out), "t": time.time(), "track": track, "n_scenes": len(ids)}, **kw)))
    os.replace(str(tmp), str(live / "state.json"))


time.sleep(float(E("FAKE_STACK_START_S", "4")))
for n, scene in enumerate(ids):
    state(scene=scene, closed=True, scene_i=n)
    time.sleep(LOAD)
    sid, t_created = str(uuid.uuid4()), time.time()
    sess.write(json.dumps({"kind": "start", "t": t_created, "session": sid, "n": n, "scene": scene}) + "\n")
    skip = lambda: (j(live / "abort.json", {}) or {}).get("scene") == scene and j(live / "abort.json")["t"] > t_created  # noqa: E731
    t_hold, aborted = time.time(), False
    while True:                                                  # hold
        paused = (j(ctl, {}) or {}).get("paused")
        state(scene=scene, waiting=True, paused=bool(paused), scene_i=n + 1)
        if skip():
            aborted = True
            break
        if paused:
            t_hold = time.time()
        elif time.time() - t_hold > START:
            break
        time.sleep(0.1)
    t_drive, t_p = 0.0, 0.0
    while t_drive < (1.0 if aborted else DRIVE):                 # drive (a skipped scene runs out fast)
        paused = (j(ctl, {}) or {}).get("paused") and not aborted
        if not aborted and skip():
            aborted = True
            t_drive = 0.0
        state(scene=scene, waiting=False, started=True, paused=bool(paused), aborted=aborted, scene_i=n + 1)
        time.sleep(0.1)
        if paused:
            t_p += 0.1
        else:
            t_drive += 0.1
    state(scene=scene, closed=True, scene_i=n + 1)
    drive.write(json.dumps({"kind": "close", "scene": scene, "sim_s": 20.0, "wall_per_sim": 1.0, "stale": 0, "wait_s": round(START, 1),
                            "aborted": aborted, "paused_s": round(t_p, 1)}) + "\n")
    sess.write(json.dumps({"kind": "close", "t": time.time(), "session": sid, "n": n, "scene": scene, "started": True,
                           "aborted": aborted}) + "\n")
    rd = out / "sim/rollouts" / scene / str(uuid.uuid4())
    rd.mkdir(parents=True)
    real = next(iter(sorted(src.glob("sim/rollouts/%s/*/metrics.parquet" % scene))), None)
    shutil.copy(str(real), str(rd / "metrics.parquet")) if real else (rd / "metrics.parquet").write_bytes(b"")
    (rd / "_complete").touch()
if E("FAKE_FAIL"):
    (out / "ERROR").write_text("fake failure\n")
    sys.exit(1)
(out / "DONE").write_text(time.ctime() + "\n")
