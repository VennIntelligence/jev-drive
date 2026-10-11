#!/usr/bin/env python3
"""Lane JOINT: keep this lane's queued pool jobs on cards 1-5 so that card 0 is free for lane LONG-PROBE (the lane brief: LONG-PROBE holds one card,
JOINT uses up to five). Every 30 s, each queued job of this lane (owners alpasim-joint1 / eyes_sense / alpasim-nuplan1491, and the bench jobs of
its tags) that may run anywhere is retargeted to cards 1-5 (`jevdrive.cl retarget`). The pool still widens such a job to card 0 when card 0 is idle
(auto_retarget), so no card idles while work waits. Stops by itself after --hours.
  scripts/tmux_run.sh joint-cards .venv/bin/python experiments/alpasim/scripts/joint_cards.py
"""
import argparse
import json
import re
import subprocess
import sys
import time

OWNERS = ("alpasim-joint1", "eyes_sense", "alpasim-nuplan1491")
TAGS = re.compile(r"(JAPS|ES[abcd]-)")


def cl(*a):
    return subprocess.run([sys.executable, "-m", "jevdrive.cl", *a], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,3,4,5")
    ap.add_argument("--hours", type=float, default=30)
    a = ap.parse_args()
    t0, done = time.time(), set()
    while time.time() - t0 < 3600 * a.hours:
        for ln in cl("queue").stdout.splitlines():
            f = ln.split()
            if len(f) < 2 or f[1] != "queued" or f[0] in done:
                continue
            try:
                j = json.loads(cl("show", f[0]).stdout)
            except ValueError:
                continue
            s = j.get("spec", {})
            if s.get("gpus") or not (s.get("owner") in OWNERS or TAGS.search(s.get("name", ""))):
                done.add(f[0])
                continue
            r = cl("retarget", f[0], "--gpus", a.gpus)
            print(time.strftime("%F %T"), f[0], s.get("name"), s.get("owner"), "->", a.gpus, "rc", r.returncode, (r.stdout + r.stderr).strip()[-200:], flush=True)
            done.add(f[0])
        time.sleep(30)


if __name__ == "__main__":
    main()
