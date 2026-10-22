#!/usr/bin/env python3
"""Stand-in for a resident human_stack.sh (HUMAN_SERVE=1) to test scripts/human_loop.py and the panel without a GPU stack.

    HUMAN_STACK_SCRIPT=experiments/alpasim_human/scripts/fake_stack.py python3 experiments/alpasim_human/scripts/human_loop.py --run ...

Same call (<track> <out dir> <scene list>) and the same files the loop reads: <out>/phase.json (driver, wizard, containers, services,
ready: FAKE_STACK_START_S in all), the feeder protocol of scripts/human_feed.py under <out>/feed/ (req/, res/, busy.json, lease-*,
ready.json, alive.json, stop; idle exit after HUMAN_IDLE_S), <out>/driver/{sessions,drive}.jsonl and thumbs/<rollout>.jpg (a copy
of FAKE_THUMB, default <live>/front.jpg), <live>/state.json (run = out dir, scene, waiting, started, closed),
sim/rollouts/<scene>/<id>/{metrics.parquet, _complete} and sim/eval-config.yaml, ERROR on failure. Per request: FAKE_LOAD_S of
loading, a hold until the start (FAKE_START_S after the hold begins, or a throttle > 0.2 on HUMAN_WHEEL_PORT; never while
control.json says paused), FAKE_DRIVE_S of driving (frozen while paused), then the rollout files. abort.json ends the scene as
lib/human_driver.py does (only when written after the session started). metrics.parquet is copied from a real earlier rollout of
the same scene (FAKE_METRICS_FROM: a run dir with sim/rollouts) so the loop's scoring runs AlpaSim's own aggregation; scenes
without one get an empty file (scoring then fails, as it should). FAKE_FAIL=n: the stack dies during its n-th request. Stdlib only.
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
IDLE, FAIL = float(E("HUMAN_IDLE_S", "1200")), int(E("FAKE_FAIL", "0") or 0)
thumb = Path(E("FAKE_THUMB", str(live / "front.jpg")))
ids = set(x for x in lst.read_text().split() if x)
F, t0 = out / "feed", time.time()
for d in (out / "driver", out / "sim", F / "req", F / "res"):
    d.mkdir(parents=True, exist_ok=True)


def j(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return d


def put(p, d):
    tmp = p.with_name("." + p.name + ".fake")
    tmp.write_text(json.dumps(d))
    os.replace(str(tmp), str(p))


def die(msg, rc=1):
    (out / "ERROR").write_text(msg + "\n")
    put(out / "phase.json", {"phase": "error", "msg": msg, "t0": t0, "t": time.time()})
    sys.exit(rc)


signal.signal(signal.SIGTERM, lambda *_: die("interrupted", 143))
(out / "pid").write_text("%d\n" % os.getpid())
start = float(E("FAKE_STACK_START_S", "4"))
for ph, msg in (("driver", "starting the driver container"), ("wizard", "wizard: writing the configs for %d scenes" % len(ids)),
                ("containers", "starting the containers"), ("services", "containers up; waiting for renderer runtime")):
    put(out / "phase.json", {"phase": ph, "msg": msg, "t0": t0, "t": time.time()})
    time.sleep(start / 4)
shutil.copy(str(next(src.glob("sim/eval-config.yaml"))), str(out / "sim/eval-config.yaml"))
sess, drive = open(str(out / "driver/sessions.jsonl"), "a", buffering=1), open(str(out / "driver/drive.jsonl"), "a", buffering=1)
put(F / "ready.json", {"t": time.time()})
put(out / "phase.json", {"phase": "ready", "msg": "ready", "t0": t0, "t": time.time()})


def lease_age(owner=None):
    ts = [(j(p, {}) or {}).get("t", 0) for p in F.glob("lease-%s.json" % (owner or "*"))]
    return time.time() - max(ts or [0])


def state(**kw):
    put(live / "state.json", dict({"run": str(out), "t": time.time(), "track": track, "n_scenes": 0}, **kw))


def rollout(scene, n):
    """One session as lib/human_driver.py runs it; returns its uuid."""
    sid, t_created = str(uuid.uuid1()), time.time()           # session uuid = rollout dir name, as in AlpaSim
    sess.write(json.dumps({"kind": "start", "t": t_created, "session": sid, "n": n, "scene": scene}) + "\n")
    skip = lambda: (j(live / "abort.json", {}) or {}).get("scene") == scene and j(live / "abort.json")["t"] > t_created  # noqa: E731
    time.sleep(LOAD)
    t_hold, aborted = time.time(), False
    while True:                                                  # hold
        put(F / "alive.json", {"t": time.time()})
        paused = (j(ctl, {}) or {}).get("paused")
        state(scene=scene, waiting=True, paused=bool(paused), scene_i=n + 1, session=sid)
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
        put(F / "alive.json", {"t": time.time()})
        paused = (j(ctl, {}) or {}).get("paused") and not aborted
        if not aborted and skip():
            aborted, t_drive = True, 0.0
        state(scene=scene, waiting=False, started=True, paused=bool(paused), aborted=aborted, scene_i=n + 1, session=sid)
        time.sleep(0.1)
        if paused:
            t_p += 0.1
        else:
            t_drive += 0.1
    state(scene=scene, closed=True, scene_i=n + 1)
    drive.write(json.dumps({"kind": "close", "session": sid, "scene": scene, "sim_s": 20.0, "wall_per_sim": 1.0, "stale": 0,
                            "wait_s": round(START, 1), "aborted": aborted, "paused_s": round(t_p, 1)}) + "\n")
    sess.write(json.dumps({"kind": "close", "t": time.time(), "session": sid, "n": n, "scene": scene, "started": True,
                           "aborted": aborted}) + "\n")
    rd = out / "sim/rollouts" / scene / sid
    if thumb.is_file():                                          # the driver's handoff front frame (scene gallery)
        (out / "driver/thumbs").mkdir(exist_ok=True)
        shutil.copy(str(thumb), str(out / "driver/thumbs" / (sid + ".jpg")))
    rd.mkdir(parents=True)
    real = next(iter(sorted(src.glob("sim/rollouts/%s/*/metrics.parquet" % scene))), None)
    shutil.copy(str(real), str(rd / "metrics.parquet")) if real else (rd / "metrics.parquet").write_bytes(b"")
    (rd / "_complete").touch()
    time.sleep(0.5)                                              # the runtime's in-process eval before the request returns
    return sid


n, last = 0, time.time()
while not (F / "stop").exists():
    put(F / "alive.json", {"t": time.time()})
    todo = sorted(p for p in (F / "req").glob("*.json") if not (F / "res" / p.name).exists())
    if not todo:
        if lease_age() < 20:
            last = time.time()
        elif time.time() - last > IDLE:
            break
        time.sleep(0.1)
        continue
    r = j(todo[0], {}) or {}
    rec = {"scene": r.get("scene"), "owner": r.get("owner"), "t_sent": time.time(), "rollouts": [], "error": ""}
    if r.get("scene") not in ids or lease_age(r.get("owner")) > 20:
        rec.update(error="dropped" if r.get("scene") in ids else "INVALID_ARGUMENT: unknown scene", t_done=time.time())
        put(F / "res" / todo[0].name, rec)
        continue
    put(F / "busy.json", {"name": todo[0].name, "scene": r["scene"], "t_sent": rec["t_sent"]})
    if FAIL and n + 1 == FAIL:
        time.sleep(LOAD)
        die("fake failure")
    rec["rollouts"] = [{"uuid": rollout(r["scene"], n), "success": True, "error": ""}]
    rec["t_done"] = time.time()
    put(F / "res" / todo[0].name, rec)
    (F / "busy.json").unlink()
    n, last = n + 1, time.time()
put(out / "phase.json", {"phase": "stopped", "msg": "stopped", "t0": t0, "t": time.time()})
(out / "DONE").write_text(time.ctime() + "\n")
