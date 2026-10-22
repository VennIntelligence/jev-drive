#!/usr/bin/env python3
"""Probe of the scene-to-scene gap with and without prefetch (README "Scene-to-scene gap"), beside the user's stack: its own live dir,
ports, container prefix, resident dir and card (Tokyo card 1). Runs scripts/human_loop.py of <repo> on a fixed playlist, then stops the
loop and its resident stack and writes <out>/summary.json (per scene: score, wall / sim; the gap table of scripts/gap_probe.py; card peak).

    python3 prefetch_probe.py <repo> <out> <driver: human | tag> [--track pai] [--scenes id ...] [--semantics] [K=V ...]

human: a fake wheel (scripts/fake_wheel.py, constant throttle 0.3, straight) on HUMAN_WHEEL_PORT starts each held scene, paced at real
time. K=V are env for the loop and the stack (HUMAN_PREFETCH=1, HUMAN_HARMONIZER via --harm, ...).
--semantics (prefetch, human): Next while a scene drives and the next one is prefetched (the prefetched scene must be shown at once),
play of another scene while one is prefetched (the prefetched one must be dropped unscored), Restart, then Stop.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("out")
ap.add_argument("driver")
ap.add_argument("--track", default="pai")
ap.add_argument("--scenes", nargs="*", default=["clipgt-559d4ad6-aed7-4065-b25c-cff64aba28c0", "clipgt-478d6ec4-f02b-4c82-97af-ddc0fbe5027f",
                                                 "clipgt-054b5901-fdcb-4146-b125-eb2bb333cf02", "clipgt-4ecc114d-4c76-49bb-874c-601561d738bb",
                                                 "clipgt-f18c6340-ab0d-4f06-8185-260b90430062"])
ap.add_argument("--harm", action="store_true")
ap.add_argument("--semantics", action="store_true")
ap.add_argument("--timeout", type=float, default=1500)
ap.add_argument("--summary", action="store_true", help="only (re)write summary.json of a finished probe")
ap.add_argument("env", nargs="*")
a = ap.parse_args()
S, R = a.scenes, Path(a.out)
L, RES = R / "live", R / "res"
human = a.driver == "human"
if not a.summary:
    R.mkdir(parents=True, exist_ok=True)
    L.mkdir(exist_ok=True)
    json.dump({"track": a.track, "set": "custom", "label": "prefetch probe", "ids": S, "created": time.strftime("%F")}, open(R / "exam.json", "w"))
    ctl = {"paused": False, "playlist": True, "record": True, "order": S, "actions": [],
           "options": {"driver": a.driver, "render": "auto", "harmonizer": a.harm, "shadow": False}}
    json.dump(ctl, open(L / "control.json", "w"))
    WP = "47898"
    env = dict(os.environ, HUMAN_LIVE=str(L), HUMAN_GPU="1", HUMAN_ALLOW_CARD1="1", HUMAN_PREFIX="hpf", HUMAN_JIT_DIR="/data/tmp/hmp_jit",
               HUMAN_BASEPORT="6900", HUMAN_DRV_PORT="6990", HUMAN_MODEL_PORT="6991", HUMAN_RESIDENT_DIR=str(RES),
               HUMAN_WHEEL_PORT=WP, HUMAN_IDLE_S="120")
    env.update(x.split("=", 1) for x in a.env)                 # K=V may override the probe defaults (prefix, ports, card)
    fake = subprocess.Popen(["python3", a.repo + "/experiments/alpasim_human/scripts/fake_wheel.py", "--port", WP, "--straight", "1000",
                             "--live", str(L)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if human else None
    t_start = time.time()
    loop = subprocess.Popen(["python3", a.repo + "/experiments/alpasim_human/scripts/human_loop.py", "--run", str(R)], env=env,
                            stdout=open(R / "loop.log", "a"), stderr=subprocess.STDOUT)
    key = (lambda s: s) if human else (lambda s: "%s@%s" % (s, a.driver))
    done = lambda: [s for s in S if (R / "done" / (key(s) + ".json")).exists()]  # noqa: E731
    aid = [100]
    log = open(R / "probe.log", "a")


    def say(msg):
        log.write("%s %s\n" % (time.strftime("%T"), msg))
        log.flush()


    def act(kind, item=None):
        c = json.load(open(L / "control.json"))
        aid[0] += 1
        c["actions"].append({"id": aid[0], "kind": kind, "item": item, "t": time.time()})
        tmp = L / ".control.tmp"
        json.dump(c, open(tmp, "w"))
        os.replace(tmp, L / "control.json")
        say("action %s %s" % (kind, item))


    def rd(name):
        try:
            return json.load(open(L / name))
        except (OSError, ValueError):
            return {}


    def wait_for(cond, t_max=300):
        t0 = time.time()
        while time.time() - t0 < t_max and loop.poll() is None:
            v = cond()
            if v:
                return v
            time.sleep(0.2)
        return None


    def two_open_one_started():
        o = rd("sessions.json").get("open", [])
        st = rd("state.json")
        return len([x for x in o if not x["aborted"]]) >= 2 and st.get("started") and (st.get("sim_t") or 0) > 4 and st


    checks = {}
    if a.semantics:
        # 1. Next while driving with a prefetched neighbour: the neighbour is shown at once
        st = wait_for(two_open_one_started)
        if st:
            cur = st["session"]
            nb = [x for x in rd("sessions.json")["open"] if x["session"] != cur][0]
            t0 = time.time()
            act("next")
            sh = wait_for(lambda: rd("state.json").get("session") == nb["session"] and rd("state.json"), 30)
            checks["next"] = {"skipped": cur, "scene": st["scene"], "neighbour": nb["scene"], "shown_after_s": round(time.time() - t0, 2) if sh else None}
            say("next: %s" % checks["next"])
        # 2. play another scene while one drives and one is prefetched: the prefetched one is dropped, unscored
        st = wait_for(two_open_one_started)
        if st:
            opn = rd("sessions.json")["open"]
            nb = [x for x in opn if x["session"] != st["session"]][0]
            other = next(s for s in reversed(S) if s not in (st["scene"], nb["scene"]) and s not in done())
            act("play", other)
            gone = wait_for(lambda: nb["session"] not in [x["session"] for x in rd("sessions.json").get("open", [])], 60)
            checks["play"] = {"driving": st["scene"], "dropped": nb["scene"], "played": other, "dropped_closed": bool(gone)}
            say("play: %s" % checks["play"])
        # 3. restart while driving: the same scene comes back first
        st = wait_for(two_open_one_started)
        if st:
            act("restart")
            again = wait_for(lambda: rd("state.json").get("scene") == st["scene"] and rd("state.json").get("session") != st["session"]
                             and rd("state.json"), 120)
            checks["restart"] = {"scene": st["scene"], "again": bool(again)}
            say("restart: %s" % checks["restart"])
        wait_for(lambda: len(done()) >= 3, a.timeout)
    else:
        while len(done()) < len(S) and time.time() - t_start < a.timeout and loop.poll() is None:
            time.sleep(3)
            st = rd("loop.json")
            if st.get("state") == "idle" and "try again" in st.get("msg", ""):
                say("loop idle with an error: %s" % st.get("msg"))
                break
    time.sleep(12)                                                    # scoring of the last scene
    act("stop")
    try:
        loop.wait(120)
    except subprocess.TimeoutExpired:
        loop.kill()
    if fake:
        fake.terminate()
    for d in glob.glob(os.path.join(str(RES), "*", "")):
        if not os.path.exists(d + "DONE"):
            open(d + "feed/stop", "w").close()
    for _ in range(60):
        if subprocess.run("docker ps --format '{{.Names}}' | grep -q '^hpf-'", shell=True).returncode != 0:
            break
        time.sleep(2)
    subprocess.run("docker ps --format '{{.Names}}' | grep '^hpf-' | xargs -r docker rm -f >/dev/null", shell=True)
else:
    checks = (json.load(open(R / "summary.json")) if (R / "summary.json").exists() else {}).get("checks", {})
# ---- summary ----
out = {"driver": a.driver, "env": a.env, "harm": a.harm, "checks": checks, "scenes": [], "gaps": {}, "peak_mib": None}
for f in sorted(glob.glob(str(R / "done" / "*.json"))):
    d = json.load(open(f))
    out["scenes"].append({k: d.get(k) for k in ("scene", "score", "why", "sim_s", "wall_per_sim", "wait_s", "driver", "attempt")})
out["tried"] = {Path(f).stem: json.load(open(f)).get("status") for f in glob.glob(str(R / "tried" / "*.json"))}
sys.path.insert(0, a.repo + "/experiments/alpasim_human/scripts")
import gap_probe  # noqa: E402
closes = []
for rdir in sorted(glob.glob(os.path.join(str(RES), "*", ""))):
    out["gaps"][rdir] = gap_probe.analyse(rdir)
    closes += [x for x in gap_probe.jl(Path(rdir) / "driver/drive.jsonl") if x.get("kind") == "close"]
    u = gap_probe.jl(Path(rdir) / "usage.jsonl")
    out["peak_mib"] = max([out["peak_mib"] or 0] + [x["peak_mib"] for x in u])
out["closes"] = [{k: c.get(k) for k in ("scene", "aborted", "drives", "sim_s", "wall_per_sim", "unpaced_per_sim", "late", "wait_s", "standby_s",
                                        "why_fin", "refinish", "gt_end", "t_first", "last_now", "dt_us", "driver")} for c in closes]
json.dump(out, open(R / "summary.json", "w"), indent=1)
print("summary written: %s" % (R / "summary.json"))
