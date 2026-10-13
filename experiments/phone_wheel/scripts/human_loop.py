#!/usr/bin/env python
"""Long-lived human drive loop for the desktop panel: one b2d_run Runner per route on one kept off-screen CARLA server.

    /data/envs/carla/bin/python experiments/phone_wheel/scripts/human_loop.py --run /data/runs/human_drive/<stamp>

It takes commands from <live>/control.json (written only by the panel; protocol in lib/exam_panel.py) and reports in
<live>/loop.json. Route order = the panel's current order (control "order"), else <run>/routes.txt. A route counts as finished
when <run>/done/<id>.json exists; results land in the run dir exactly as with human_drive.sh (attempts/<id>/<n>, done/,
results.md via human_results.py after every route, official/ via human_official.py in the background).
- Playlist on: after a route ends, the next unfinished route in order loads (and holds until the first throttle press).
  Playlist off: the loop waits idle until a row is double-clicked or the driver presses throttle on the phone.
- Pause: no new route starts; a running route is held at its current tick by the agent (HUMAN_CONTROL).
- Actions: play <id> (drive it next, even if finished), next (abort the route, skip it for this session), restart (abort and
  drive it again), stop (abort and exit). An aborted route is not finished and can be driven again later.
- Option stop_on_collision -> HUMAN_STOP_ON=collision for the next route (default off = official full routes).
CARLA is reloaded for every route (load_world, as the leaderboard ships); only the server process is kept.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "lib"), str(REPO / "experiments/phone_wheel/lib")]
import b2d_run as R  # noqa: E402
from exam_panel import Control, read_json, write_json  # noqa: E402
from wheel_client import WheelClient  # noqa: E402

LIVE = Path(os.environ.get("HUMAN_LIVE", "/data/runs/human_drive/live"))
XML = str(R.BENCH2DRIVE / "leaderboard/data/bench2drive_0.0.4_val.xml")
PY = "/data/envs/carla/bin/python"


class KeptServer(R.Server):
    """While `keep` is set (an abort from the panel), the Runner's stop() / move() leave the server running."""
    keep = False

    def stop(self):
        if not self.keep:
            super().stop()

    def move(self):
        if not self.keep:
            super().move()


def gpu_check(server, gpu, out):
    """The server must sit on card `gpu` (CARLA ignores CUDA_VISIBLE_DEVICES); stop the loop if it does not."""
    for _ in range(120):
        time.sleep(5)
        if server.proc is None:
            continue
        kids = subprocess.run(["pgrep", "-g", str(server.proc.pid), "CarlaUE4"], stdout=subprocess.PIPE, text=True).stdout.split()
        smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True).stdout
        cards = [ln.split()[1] for ln in smi.splitlines() if ln.startswith("|") and any(" %s " % k in ln for k in kids)
                 and len(ln.split()) > 2 and ln.split()[1].isdigit()]
        if cards:
            with open(str(out / "gpu_check.txt"), "a") as fh:
                fh.write("%s CARLA server (%s) on card %s, wanted %s\n" % (time.strftime("%F %T"), "|".join(kids), cards[0], gpu))
            if cards[0] != str(gpu):
                os.kill(os.getpid(), signal.SIGTERM)
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--server-index", type=int, default=int(os.environ.get("HUMAN_SERVER_INDEX", 4)))
    p.add_argument("--gpu", type=int, default=int(os.environ.get("HUMAN_GPU", 0)))
    o = p.parse_args()

    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    run = Path(o.run)
    ids = [i for i in (run / "routes.txt").read_text().strip().split(",") if i]
    town = dict(R.select_routes(XML, "all", "", 0))
    LIVE.mkdir(parents=True, exist_ok=True)
    ctl, state_f = Control(LIVE), LIVE / "loop.json"
    consumed = [a["id"] for a in ctl.read()["actions"]]          # actions older than this loop are not replayed
    st = {"run": str(run), "item": None, "state": "idle", "msg": "", "consumed": consumed, "pid": os.getpid()}

    def report(**kw):
        st.update(kw, t=time.time())
        st["consumed"] = st["consumed"][-200:]
        write_json(state_f, st)
    os.environ.update(B2D_PREVIEW_DIR=str(LIVE), HUMAN_CONTROL=str(LIVE / "control.json"), B2D_SENSOR_TICK="1",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    os.environ.setdefault("WHEEL_STALE_S", "1.0")
    os.environ.setdefault("WHEEL_BRAKE_RAMP_S", "1.0")
    common = ["--routes", XML, "--workers", "1", "--server-index", str(o.server_index), "--gpu-rank", str(o.gpu), "--tm-seed", "0",
              "--no-reap", "--no-spectator", "--client-threads", "8", "--max-attempts", "3", "--stall-s", "600",
              "--route-timeout-s", "86400", "--out", str(run), "--python", PY, "--fast-copy", "--cache-lights",
              "--agent", str(REPO / "experiments/phone_wheel/lib/human_agent.py")]
    server = KeptServer(o.server_index, run / "servers", "Epic", o.gpu, stride=1, windowed=False, base=o.server_index, span=1)
    threading.Thread(target=gpu_check, args=(server, o.gpu, run), daemon=True).start()
    wheel = WheelClient()
    plays, skipped = [], set()
    cur = {"runner": None, "abort": None}

    def take(kinds):
        """Pending actions of these kinds, marked consumed."""
        got = [a for a in ctl.pending(st["consumed"]) if a["kind"] in kinds]
        st["consumed"] += [a["id"] for a in got]
        return got

    def watch():                 # next / restart / stop abort the running route (SIGINT to its group, CARLA kept)
        while True:
            time.sleep(0.3)
            run_ = cur["runner"]
            if run_ is None or run_.stop_flag:
                continue
            for a in take(("next", "restart", "stop", "play")):
                if a["kind"] == "play":
                    plays.append(a["item"])
                    continue
                cur["abort"] = a["kind"]
                server.keep = a["kind"] != "stop"
                run_.stop_flag = True
                break
    threading.Thread(target=watch, daemon=True).start()

    def done(rid):
        return (run / "done" / (rid + ".json")).exists()

    def next_unfinished():
        order = [i for i in ctl.read()["order"] if i in town and i in ids] or ids
        return next((i for i in order if not done(i) and i not in skipped), None)

    def results():
        subprocess.run([PY, str(REPO / "experiments/phone_wheel/scripts/human_results.py"), str(run)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen([PY, str(REPO / "experiments/phone_wheel/scripts/human_official.py"), str(run)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        while True:
            c = ctl.read()
            for a in take(("play", "next", "restart", "stop")):
                if a["kind"] == "stop":
                    raise KeyboardInterrupt
                if a["kind"] == "play" and a["item"] in town:
                    plays.append(a["item"])
            if c["paused"]:
                report(item=None, state="paused", msg="resume on the panel")
                time.sleep(0.5)
                continue
            rid, fresh = None, False
            if plays:
                rid, fresh = plays.pop(0), True
            elif c["playlist"] or (not wheel.state()["stale"] and wheel.state()["raw_throttle"] > 0.2):
                rid = next_unfinished()
            if rid is None:
                left = sum(not done(i) for i in ids)
                report(item=None, state="idle", msg=("%d of %d routes left: double-click a row%s" % (
                    left, len(ids), "" if c["playlist"] else " or press throttle")) if left else "all %d routes finished" % len(ids))
                time.sleep(0.3)
                continue
            stop_on = "collision" if c["options"].get("stop_on_collision") else ""
            os.environ["HUMAN_STOP_ON"] = stop_on
            report(item=rid, state="loading", msg="%s %s%s" % (town[rid], "replay " if fresh and done(rid) else "",
                                                              "stop on collision" if stop_on else ""))
            with open(str(run / "notes.txt"), "a") as fh:
                fh.write("%s loop: route %s stop_on='%s'%s\n" % (time.strftime("%F %T"), rid, stop_on, " (play)" if fresh else ""))
            a = R.parse_args(common + ["--route-ids", rid] + (["--fresh"] if fresh else []))
            runner = R.Runner(a, [(rid, town[rid])], servers=[server])
            cur.update(runner=runner, abort=None)
            report(state="driving")
            rc = runner.run()
            cur["runner"] = None
            kind = cur["abort"]
            server.keep = False
            if kind == "restart":
                plays.insert(0, rid)
            elif kind == "next":
                skipped.add(rid)
            elif kind == "stop":
                raise KeyboardInterrupt
            elif rc == 130:
                raise KeyboardInterrupt          # Ctrl-C in the loop window
            if done(rid):
                skipped.discard(rid)
            results()
    except KeyboardInterrupt:
        pass
    finally:
        report(item=None, state="stopped", msg="")
        server.keep = False
        server.stop()
        results()


if __name__ == "__main__":
    main()
