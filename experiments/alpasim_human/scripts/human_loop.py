#!/usr/bin/env python3
"""Long-lived AlpaSim human drive loop for the desktop panel: human_stack.sh stacks on Tokyo card 0, one scene list per stack.

    /usr/bin/python3 experiments/alpasim_human/scripts/human_loop.py --run /data/runs/alpasim_human/exam_<track>_<stamp>

Commands come from <live>/control.json (written only by the panel; protocol in lib/exam_panel.py); state goes to <live>/loop.json.
A stack costs about a minute to start (compose up 18-27 s + the first scene's renderer start) against 4-5 s between scenes inside one
stack, so a stack runs the whole remaining playlist and the loop only restarts it when the order changes:
- Playlist on: a stack gets every unfinished scene in the panel's order (control "order", else exam.json). Each scene loads and holds
  until the first throttle press (lib/human_driver.py). Playlist off: a stack gets one scene (a double-clicked one, or the next
  unfinished one when the driver presses the throttle while the loop is idle).
- play <id>: drive it next (even if finished). If a stack is running, it is stopped at its next hold point (the next scene loaded and
  waiting for the throttle; nothing of that scene is scored) and a new stack starts with <id> first.
- next: the driver skips the current scene (abort.json: no pacing, full brake, the rollout ends at the stack's own speed, about 10 s);
  the skipped rollout is not counted and the scene is not offered again in this session. Without a running scene: nothing.
- restart: the stack is stopped at once and a new one starts with the same scene first. Between two scenes (one closed, the next
  still loading) next and restart do nothing.
- stop: the stack is stopped, the loop exits.
- Pause: no stack starts; inside a stack the driver holds (a waiting scene stays held, a running one stops advancing).
Scoring: every finished rollout (sim/rollouts/<scene>/<id>/_complete) that was not skipped is scored with AlpaSim's own aggregation
(alpasim-aggregation in the alpasim-base image, CPU, ~7 s) on a copy of its metrics.parquet -> done/<scene>.json; results.md after each.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "lib"), str(REPO / "experiments/alpasim_human/lib"), str(REPO / "experiments/phone_wheel/lib")]
from exam_panel import Control, read_json, write_json  # noqa: E402
from alpasim_exam import Scenes, stack_name, why, sh  # noqa: E402
from wheel_client import WheelClient  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/data"))
LIVE = Path(os.environ.get("HUMAN_LIVE", str(D / "runs/alpasim_human/live")))
SRC = Path(os.environ.get("ALPASIM_SRC", str(D / "third_party/alpasim")))
STACK = Path(os.environ.get("HUMAN_STACK_SCRIPT", str(REPO / "experiments/alpasim_human/scripts/human_stack.sh")))  # fake_stack.py in tests


def base_image():
    v = next((ln.split('"')[1] for ln in (SRC / "pyproject.toml").read_text().splitlines() if ln.startswith("version = ")), "")
    return os.environ.get("BASE", "alpasim-base:%s" % v)


def jsonl(p):
    try:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


class Stack(object):
    """One human_stack.sh run (foreground, own process group) on a list of scenes."""

    def __init__(self, run, k, track, ids):
        self.dir = run / "stacks" / ("%s-s%02d" % (run.name, k))
        self.dir.mkdir(parents=True)
        self.ids, self.name, self.killed = list(ids), stack_name(self.dir), None
        (self.dir / "scenes.txt").write_text("\n".join(ids) + "\n")
        env = dict(os.environ, HUMAN_FG="1", HUMAN_VIEWER="0", HUMAN_CONTROL=str(LIVE / "control.json"), HUMAN_LIVE=str(LIVE),
                   HUMAN_WAIT_MAX_S=os.environ.get("HUMAN_WAIT_MAX_S", "1800"), HUMAN_GPU="0")
        self.log = open(str(self.dir / "stack.out"), "a")
        self.proc = subprocess.Popen(["python3" if STACK.suffix == ".py" else "bash", str(STACK), track, str(self.dir), str(self.dir / "scenes.txt")], env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.t0 = time.time()

    def alive(self):
        return self.proc.poll() is None

    def sessions(self):
        return jsonl(self.dir / "driver/sessions.jsonl")

    def kill(self, why_):
        """Stop the stack: SIGTERM to the script (its trap runs compose down), and compose down / driver rm here as well."""
        self.killed = why_
        try:
            os.kill(self.proc.pid, signal.SIGTERM)
        except OSError:
            pass
        compose = self.dir / "sim/docker-compose.yaml"
        if compose.exists():
            sh("docker compose -p %s -f %s down -t 3 >/dev/null 2>&1" % (self.name, compose))
        sh("docker rm -f %s-drv >/dev/null 2>&1" % self.name)
        for _ in range(120):
            if not self.alive():
                break
            time.sleep(0.5)
        else:
            os.killpg(self.proc.pid, signal.SIGKILL)
            if compose.exists():
                sh("docker compose -p %s -f %s down -t 3 >/dev/null 2>&1" % (self.name, compose))
        self.log.close()


class Scorer(object):
    """Scores finished rollouts of the run's stacks one at a time in a thread (AlpaSim's aggregation on a one-rollout copy)."""

    def __init__(self, run, sc, base):
        self.run, self.sc, self.base, self.seen, self.lock = run, sc, base, set(), threading.Lock()
        for r in jsonl(run / "scores.jsonl"):
            self.seen.add(r["rollout"])

    def pending(self, stack, final=False):
        """(scene, rollout dir, close record) of finished rollouts of this stack not yet handled."""
        closes = {s["scene"]: s for s in stack.sessions() if s["kind"] == "close"}
        out = []
        for c in sorted((stack.dir / "sim/rollouts").glob("*/*/_complete")):
            rd, scene = c.parent, c.parent.parent.name
            if rd.name in self.seen or (scene not in closes and not final):
                continue
            out.append((scene, rd, closes.get(scene, {})))
        return out

    def score(self, stack, scene, rd, close):
        self.seen.add(rd.name)
        if close.get("aborted"):
            write_json(self.run / "tried" / (scene + ".json"), {"status": "skipped (Next item)", "t": time.time()})
            self.log({"rollout": rd.name, "scene": scene, "stack": stack.dir.name, "aborted": True})
            return
        tmp = self.run / "scores" / rd.name
        shutil.rmtree(str(tmp), ignore_errors=True)
        (tmp / "rollouts" / scene / rd.name).mkdir(parents=True)
        for f in ("metrics.parquet", "_complete"):
            shutil.copy(str(rd / f), str(tmp / "rollouts" / scene / rd.name / f))
        shutil.copy(str(stack.dir / "sim/eval-config.yaml"), str(tmp / "eval-config.yaml"))
        out = sh("docker run --rm -v %s:%s %s bash -c 'umask 0000; cd /repo && uv run --offline alpasim-aggregation --array_job_dir %s "
                 "--config_path %s/eval-config.yaml' 2>&1 | tail -5" % (tmp, tmp, self.base, tmp, tmp))
        summ = read_json(tmp / "aggregate/results-summary.json")
        if not summ or not summ.get("rollouts"):
            write_json(self.run / "tried" / (scene + ".json"), {"status": "scoring failed (scores/%s)" % rd.name, "t": time.time()})
            (tmp / "aggregation.txt").write_text(out)
            self.log({"rollout": rd.name, "scene": scene, "stack": stack.dir.name, "error": out[-300:]})
            return
        r = summ["rollouts"][0]
        m, sm = r.get("metrics") or {}, r.get("score_metrics") or {}
        drv = {x.get("scene"): x for x in jsonl(stack.dir / "driver/drive.jsonl") if x.get("kind") == "close"}.get(scene, {})
        prev = read_json(self.run / "done" / (scene + ".json"), {}) or {}
        rec = {"scene": scene, "score": r.get("score") or 0.0, "why": why(r), "rollout": rd.name, "stack": stack.dir.name,
               "progress": round(sm.get("progress_clipped_rel") or 0.0, 3), "driven_m": round(m.get("dist_traveled_m") or 0.0, 1),
               "gt_m": round(m.get("gt_dist_traveled_m") or 0.0, 1), "sim_s": drv.get("sim_s"), "wall_per_sim": drv.get("wall_per_sim"),
               "stale": drv.get("stale"), "wait_s": drv.get("wait_s"), "paused_s": drv.get("paused_s"),
               "attempt": prev.get("attempt", 0) + 1, "t": time.time()}
        write_json(self.run / "done" / (scene + ".json"), rec)
        self.log(rec)
        (self.run / "results.md").write_text(self.sc.table(self.run))

    def log(self, rec):
        with self.lock, open(str(self.run / "scores.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    o = p.parse_args()

    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    run = Path(o.run).resolve()
    ex = json.loads((run / "exam.json").read_text())
    track, ids = ex["track"], ex["ids"]
    for d in ("done", "tried", "scores", "stacks"):
        (run / d).mkdir(exist_ok=True)
    sc = Scenes()
    LIVE.mkdir(parents=True, exist_ok=True)
    ctl, state_f = Control(LIVE), LIVE / "loop.json"
    consumed = [a["id"] for a in ctl.read()["actions"]]          # actions older than this loop are not replayed
    st = {"run": str(run), "item": None, "state": "idle", "msg": "", "consumed": consumed, "pid": os.getpid()}

    def report(**kw):
        st.update(kw, t=time.time())
        st["consumed"] = st["consumed"][-200:]
        write_json(state_f, st)

    def note(msg):
        print("%s %s" % (time.strftime("%F %T"), msg), flush=True)
        with open(str(run / "notes.txt"), "a") as fh:
            fh.write("%s loop: %s\n" % (time.strftime("%F %T"), msg))

    def launcher(state, msg):                                    # the viewer's status line between scenes
        write_json(LIVE / "launcher.json", {"state": state, "msg": msg, "run": str(run), "t": time.time()})

    scorer = Scorer(run, sc, base_image())
    jobs = []                                                    # (stack, scene, rollout dir, close)
    wake = threading.Event()

    def score_worker():
        while True:
            wake.wait(1.0)
            wake.clear()
            while jobs:
                try:
                    scorer.score(*jobs.pop(0))
                except Exception as e:  # noqa: BLE001  (a scoring failure must not stop the drive)
                    note("scoring failed: %r" % e)
    threading.Thread(target=score_worker, daemon=True).start()

    wheel = WheelClient()
    plays, skipped, chosen, driven = [], set(), set(), set()       # driven: finished, not skipped, maybe not scored yet                    # chosen: scenes asked for by play / throttle (playlist off)
    stack, k, hold_error = None, len(list((run / "stacks").glob("*"))), ""

    def done(s):
        return s in driven or (run / "done" / (s + ".json")).exists()

    def unfinished():
        order = [i for i in ctl.read()["order"] if i in ids] or ids
        return [i for i in order if not done(i) and i not in skipped]

    def collect(final=False):
        for scene, rd, close in scorer.pending(stack, final):
            if all(j[2] != rd for j in jobs):
                jobs.append((stack, scene, rd, close))
                if not close.get("aborted"):
                    driven.add(scene)
                    skipped.discard(scene)
        wake.set()

    def view():
        """The driver's live state of the running stack (fresh and from this stack only)."""
        s = read_json(LIVE / "state.json", {}) or {}
        return s if stack and s.get("run") == str(stack.dir) and time.time() - s.get("t", 0) < 5 else {}

    def end_stack(why_):
        nonlocal stack
        if stack.alive():
            note("stopping stack %s: %s" % (stack.dir.name, why_))
            stack.kill(why_)
        collect(final=True)
        stack = None

    try:
        while True:
            c = ctl.read()
            for a in [a for a in ctl.pending(st["consumed"])]:
                st["consumed"].append(a["id"])
                v = view()
                cur = v.get("scene") if v and not v.get("closed") else None    # between scenes next / restart do nothing
                if a["kind"] == "stop":
                    raise KeyboardInterrupt
                if a["kind"] == "play" and a["item"] in ids:
                    plays.append(a["item"])
                    chosen.add(a["item"])
                    hold_error = ""
                elif a["kind"] == "next" and stack and cur:
                    write_json(LIVE / "abort.json", {"scene": cur, "t": time.time()})
                    skipped.add(cur)
                    note("next: skipping %s" % cur)
                elif a["kind"] == "restart" and stack and cur:
                    plays.insert(0, cur)
                    chosen.add(cur)
                    end_stack("restart %s" % cur)
                    note("restart: %s" % cur)
            if stack is not None:
                collect()
                v = view()
                if not stack.alive():
                    rc = stack.proc.returncode
                    err = (stack.dir / "ERROR").read_text().strip() if (stack.dir / "ERROR").exists() else ""
                    note("stack %s ended rc %s %s" % (stack.dir.name, rc, err))
                    if rc and not stack.killed:                   # no automatic relaunch after a failure
                        hold_error = "stack failed: %s (see %s)" % (err or rc, stack.dir)
                    end_stack("ended")
                    continue
                held = v.get("scene") if v.get("waiting") and not v.get("closed") else None
                if held and held in stack.ids:
                    if plays and plays[0] != held:
                        end_stack("play %s next" % plays[0])
                        continue
                    if plays and plays[0] == held:
                        plays.pop(0)
                    if not c["playlist"] and held not in chosen:
                        end_stack("playlist off")
                        continue
                state = "paused" if c["paused"] else "loading" if not v or v.get("closed") else "driving"
                msg = ("%s held: press throttle" % sc.meta.get(held, {}).get("label", held)) if held and not c["paused"] else \
                    "resume on the panel" if c["paused"] else "stack %s, %d scenes" % (stack.dir.name, len(stack.ids))
                report(item=v.get("scene") if v and not v.get("closed") else None, state=state, msg=msg)
                time.sleep(0.3)
                continue
            # ---- no stack ----
            if c["paused"]:
                report(item=None, state="paused", msg="resume on the panel")
                time.sleep(0.5)
                continue
            todo = []
            if plays:
                todo = list(dict.fromkeys(plays + (unfinished() if c["playlist"] else [])))
                plays.clear()
            elif hold_error:
                report(item=None, state="idle", msg=hold_error + "; double-click a row to try again")
                time.sleep(0.5)
                continue
            elif c["playlist"]:
                todo = unfinished()
            else:
                w = wheel.state()
                if not w["stale"] and w["raw_throttle"] > 0.2 and unfinished():
                    todo = unfinished()[:1]
                    chosen.add(todo[0])
            if not c["playlist"]:
                todo = todo[:1]
            if not todo:
                left = sum(not done(i) for i in ids)
                report(item=None, state="idle", msg=("%d of %d scenes left: double-click a row%s" % (
                    left, len(ids), "" if c["playlist"] else " or press throttle")) if left else "all %d scenes driven" % len(ids))
                time.sleep(0.3)
                continue
            k += 1
            launcher("running", "starting a stack for %d scene(s): about a minute" % len(todo))
            stack = Stack(run, k, track, todo)
            note("stack %s: %d scenes from %s" % (stack.dir.name, len(todo), sc.meta.get(todo[0], {}).get("label", todo[0])))
            report(item=todo[0], state="loading", msg="starting the stack (%d scenes)" % len(todo))
    except KeyboardInterrupt:
        pass
    finally:
        report(item=None, state="stopped", msg="")
        if stack is not None:
            end_stack("stop")
        launcher("stopped", "panel loop stopped")
        for _ in range(60):
            if not jobs:
                break
            time.sleep(1)
        time.sleep(1)
        (run / "results.md").write_text(sc.table(run))


if __name__ == "__main__":
    main()
