#!/usr/bin/env python3
"""Long-lived AlpaSim human drive loop for the desktop panel: one scene at a time on a resident render stack (Tokyo card 0).

    /usr/bin/python3 experiments/alpasim_human/scripts/human_loop.py --run /data/runs/alpasim_human/exam_<track>_<stamp>

Commands come from <live>/control.json (written only by the panel; protocol in lib/exam_panel.py); state goes to <live>/loop.json,
progress to <live>/launcher.json (the viewer's banner: state, msg, t0 = start of the current wait, batch = [driven, total]).
Resident stack (README "Resident stack"): human_stack.sh HUMAN_SERVE=1 runs renderer / physics / controller / the runtime daemon
(AlpaSim's shipped wizard.run_mode=SERVER) / the human driver for every scene of the track, detached from this loop and registered
in <live>/resident.json. A loop reuses it when track, harmonizer, render size and card match, so only the first start after it went down costs
the start (about 45 s); a new batch, Continue, play and restart do not. It stops itself after HUMAN_IDLE_S (1200) without a loop.
The loop sends one scene per simulate request (scripts/human_feed.py protocol), so it decides the order itself:
- Playlist on: the next unfinished scene in the panel's order (control "order", else exam.json). Each scene loads and holds until
  the first throttle press (lib/human_driver.py). Playlist off: the next scene only on a double-click or a new throttle press.
- play <id>: drive it next (even if finished). A scene that is loading or held (not started) is dropped for it (abort, the
  rollout runs out unscored at the stack's speed, ~10 s); a scene being driven finishes first.
- next: the driver skips the current scene (abort.json); not counted, not offered again in this session.
- restart: the current scene is aborted and requested again first (~10 s run-out + the scene load, no stack start).
- stop: the current scene is aborted and the loop exits; the resident stack stays warm.
- Pause: no new scene is requested; inside a scene the driver holds.
- Option "harmonizer" (PAI only) and option "render" (front camera height: auto = human_stack.sh's pairing with the harmonizer,
  else HUMAN_H) are fixed for a stack's whole life, so a change restarts the resident stack (about a minute) once no scene is
  being driven; a loading or held scene is requested again on the new stack.
Run dir layout per resident stack used: stacks/<run>-sNN/ with sim/rollouts/<scene>/<uuid>/ (moved out of the resident dir when its
request returns), sim/eval-config.yaml, driver/{sessions,drive}.jsonl (that run's sessions) and driver/thumbs/, resident (its path).
Scoring: every finished rollout that was not aborted is scored with AlpaSim's own aggregation (alpasim-aggregation in the
alpasim-base image, CPU, ~7 s) on a copy of its metrics.parquet -> done/<scene>.json; results.md after each.
"""
import argparse
import json
import os
import re
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
from alpasim_exam import Scenes, why, sh  # noqa: E402
from wheel_client import WheelClient  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/data"))
LIVE = Path(os.environ.get("HUMAN_LIVE", str(D / "runs/alpasim_human/live")))
SRC = Path(os.environ.get("ALPASIM_SRC", str(D / "third_party/alpasim")))
STACK = Path(os.environ.get("HUMAN_STACK_SCRIPT", str(REPO / "experiments/alpasim_human/scripts/human_stack.sh")))  # fake_stack.py in tests
RESIDENTS = Path(os.environ.get("HUMAN_RESIDENT_DIR", str(D / "runs/alpasim_human/resident")))
GPU = os.environ.get("HUMAN_GPU", "0")


def base_image():
    v = next((ln.split('"')[1] for ln in (SRC / "pyproject.toml").read_text().splitlines() if ln.startswith("version = ")), "")
    return os.environ.get("BASE", "alpasim-base:%s" % v)


def jsonl(p):
    try:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                        # a zombie child counts as gone
        return Path("/proc/%d/stat" % pid).read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class Resident(object):
    """A resident human_stack.sh (HUMAN_SERVE=1), detached from the loop that started it; <live>/resident.json names it."""
    REG = LIVE / "resident.json"

    def __init__(self, info, proc=None):
        self.info, self.proc = info, proc
        self.dir, self.pid, self.track, self.harm = Path(info["dir"]), info["pid"], info["track"], info["harm"]
        self.size = info.get("size", "auto")                    # render option (HUMAN_H, or auto)
        self.key = (self.harm, self.size)
        self.ids, self.feed = set(info["ids"]), Path(info["dir"]) / "feed"

    @classmethod
    def start(cls, track, harm, size, ids):
        d = RESIDENTS / ("%s%s%s-%s" % (track, "-harm" if harm else "", "" if size == "auto" else "-h" + size, time.strftime("%m%d-%H%M%S")))
        d.mkdir(parents=True)
        (d / "scenes.txt").write_text("\n".join(ids) + "\n")
        env = dict(os.environ, HUMAN_FG="1", HUMAN_VIEWER="0", HUMAN_SERVE="1", HUMAN_CONTROL=str(LIVE / "control.json"),
                   HUMAN_LIVE=str(LIVE), HUMAN_WAIT_MAX_S=os.environ.get("HUMAN_WAIT_MAX_S", "1800"), HUMAN_GPU=GPU,
                   HUMAN_HARMONIZER="1" if harm else "0", HUMAN_PREFIX=os.environ.get("HUMAN_PREFIX", "hres"))
        if size != "auto":
            env["HUMAN_H"] = size
        with open(str(d / "stack.out"), "a") as log:
            proc = subprocess.Popen(["python3" if STACK.suffix == ".py" else "bash", str(STACK), track, str(d), str(d / "scenes.txt")],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        info = {"dir": str(d), "pid": proc.pid, "track": track, "harm": harm, "size": size, "gpu": GPU, "ids": list(ids), "t": time.time(),
                "prefix": env["HUMAN_PREFIX"]}
        write_json(cls.REG, info)
        return cls(info, proc)

    @classmethod
    def find(cls):
        info = read_json(cls.REG)
        r = cls(info) if info and info.get("dir") else None
        return r if r and r.alive() else None

    def alive(self):
        if self.proc is not None and self.proc.poll() is not None:
            return False
        return pid_alive(self.pid) and not (self.dir / "DONE").exists() and not (self.dir / "ERROR").exists()

    def fits(self, track, key, ids):
        return (self.track, self.key, self.info.get("gpu", "0")) == (track, key, GPU) and set(ids) <= self.ids

    def phase(self):
        return read_json(self.dir / "phase.json", {}) or {}

    def ready(self):
        a = read_json(self.feed / "alive.json", {}) or {}
        return self.phase().get("phase") == "ready" and (self.feed / "ready.json").exists() and time.time() - a.get("t", 0) < 10

    def error(self):
        if (self.dir / "ERROR").exists():
            return (self.dir / "ERROR").read_text().strip()
        tail = (self.dir / "stack.out").read_text().strip().splitlines()[-1:] if (self.dir / "stack.out").exists() else []
        return tail[0] if tail else "exited"

    def lease(self, owner, run):
        self.feed.mkdir(parents=True, exist_ok=True)
        write_json(self.feed / ("lease-%s.json" % owner), {"t": time.time(), "run": str(run)})

    def release(self, owner):
        try:
            (self.feed / ("lease-%s.json" % owner)).unlink()
        except OSError:
            pass

    def submit(self, scene, owner):
        (self.feed / "req").mkdir(parents=True, exist_ok=True)
        name = "%019d-%s.json" % (time.time_ns(), owner)
        write_json(self.feed / "req" / name, {"scene": scene, "owner": owner, "t": time.time()})
        return name

    def result(self, name):
        return read_json(self.feed / "res" / name)

    def busy(self):
        return read_json(self.feed / "busy.json", {}) or {}

    def stop(self):
        """SIGTERM to the stack script (its trap runs compose down); after 60 s its process group is killed and the stack removed."""
        try:
            os.kill(self.pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(120):
            if not pid_alive(self.pid):
                break
            time.sleep(0.5)
        else:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except OSError:
                pass
            name = "%s-%s" % (self.info.get("prefix", "hres"), re.sub(r"[^a-z0-9]", "-", self.dir.name.lower()))   # as human_stack.sh
            sh("docker compose -p %s down -t 3 >/dev/null 2>&1; docker rm -f %s-drv %s-feed >/dev/null 2>&1" % (name, name, name))
        if read_json(self.REG, {}).get("dir") == str(self.dir):
            self.REG.unlink()


class Batch(object):
    """This run's share of a resident stack: stacks/<run>-sNN/, laid out as a one-shot human_stack.sh out dir (scoring, gallery)."""

    def __init__(self, run, k, res):
        self.dir, self.res, self.harm = run / "stacks" / ("%s-s%02d" % (run.name, k)), res, res.harm
        for d in ("sim/rollouts", "driver/thumbs"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        (self.dir / "resident").write_text(str(res.dir) + "\n")
        self.off = {f: self._size(res.dir / "driver" / f) for f in ("sessions.jsonl", "drive.jsonl")}

    @staticmethod
    def _size(p):
        try:
            return p.stat().st_size
        except OSError:
            return 0

    def sessions(self):
        return jsonl(self.dir / "driver/sessions.jsonl")

    def take(self, scene, rec):
        """A returned request: its rollout dirs, driver log lines and thumbnails move / copy into this batch dir."""
        rd = self.res.dir
        if not (self.dir / "sim/eval-config.yaml").exists() and (rd / "sim/eval-config.yaml").exists():
            shutil.copy(str(rd / "sim/eval-config.yaml"), str(self.dir / "sim/eval-config.yaml"))
        uu = {x["uuid"] for x in rec.get("rollouts", []) if x.get("uuid")}
        for f in ("sessions.jsonl", "drive.jsonl"):
            try:
                with open(str(rd / "driver" / f), "rb") as fh:
                    fh.seek(self.off[f])
                    data = fh.read()
            except OSError:
                continue
            data = data[:data.rfind(b"\n") + 1]
            self.off[f] += len(data)
            mine = [ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip() and json.loads(ln).get("session") in uu]
            if mine:
                with open(str(self.dir / "driver" / f), "a") as fh:
                    fh.write("\n".join(mine) + "\n")
        for u in uu:
            src = rd / "sim/rollouts" / scene / u
            if src.is_dir():
                (self.dir / "sim/rollouts" / scene).mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(self.dir / "sim/rollouts" / scene / u))
            if (rd / "driver/thumbs" / (u + ".jpg")).is_file():
                shutil.copy(str(rd / "driver/thumbs" / (u + ".jpg")), str(self.dir / "driver/thumbs" / (u + ".jpg")))
        with open(str(self.dir / "scenes.txt"), "a") as fh:
            fh.write(scene + "\n")


class Scorer(object):
    """Scores finished rollouts of the run's stacks one at a time in a thread (AlpaSim's aggregation on a one-rollout copy)."""

    def __init__(self, run, sc, base):
        self.run, self.sc, self.base, self.seen, self.lock = run, sc, base, set(), threading.Lock()
        self.abort_status = {}                                   # rollout id -> tried status of an aborted rollout (None: none)
        for r in jsonl(run / "scores.jsonl"):
            self.seen.add(r["rollout"])

    def pending(self, stack, final=False):
        """(scene, rollout dir, close record) of finished rollouts of this stack not yet handled."""
        closes = {s["session"]: s for s in stack.sessions() if s["kind"] == "close"}
        out = []
        for c in sorted((stack.dir / "sim/rollouts").glob("*/*/_complete")):
            rd, scene = c.parent, c.parent.parent.name
            if rd.name in self.seen or (rd.name not in closes and not final):
                continue
            out.append((scene, rd, closes.get(rd.name, {})))
        return out

    def score(self, stack, scene, rd, close):
        self.seen.add(rd.name)
        if close.get("aborted"):
            status = self.abort_status.pop(rd.name, "skipped (Next item)")
            if status:
                write_json(self.run / "tried" / (scene + ".json"), {"status": status, "t": time.time()})
            self.log({"rollout": rd.name, "scene": scene, "stack": stack.dir.name, "aborted": True, "status": status})
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
        drv = {x.get("session"): x for x in jsonl(stack.dir / "driver/drive.jsonl") if x.get("kind") == "close"}.get(rd.name, {})
        prev = read_json(self.run / "done" / (scene + ".json"), {}) or {}
        rec = {"scene": scene, "score": r.get("score") or 0.0, "why": why(r), "rollout": rd.name, "stack": stack.dir.name,
               "progress": round(sm.get("progress_clipped_rel") or 0.0, 3), "driven_m": round(m.get("dist_traveled_m") or 0.0, 1),
               "gt_m": round(m.get("gt_dist_traveled_m") or 0.0, 1), "sim_s": drv.get("sim_s"), "wall_per_sim": drv.get("wall_per_sim"),
               "stale": drv.get("stale"), "wait_s": drv.get("wait_s"), "paused_s": drv.get("paused_s"), "harmonizer": stack.harm, "render": stack.res.size,
               "shadow_s": drv.get("shadow_s") or 0.0,
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
    ctl, state_f, owner = Control(LIVE), LIVE / "loop.json", str(os.getpid())
    consumed = [a["id"] for a in ctl.read()["actions"]]          # actions older than this loop are not replayed
    st = {"run": str(run), "item": None, "state": "idle", "msg": "", "consumed": consumed, "pid": os.getpid()}
    catalog = [i for i, m in sc.meta.items() if m.get("track") == track]

    def report(**kw):
        st.update(kw, t=time.time())
        st["consumed"] = st["consumed"][-200:]
        write_json(state_f, st)

    def note(msg):
        print("%s %s" % (time.strftime("%F %T"), msg), flush=True)
        with open(str(run / "notes.txt"), "a") as fh:
            fh.write("%s loop: %s\n" % (time.strftime("%F %T"), msg))

    def label(s):
        return sc.meta.get(s, {}).get("label", s)

    def progress(state, msg, t0=None, item=None):
        """loop.json (the panel's status line, with the elapsed seconds) and launcher.json (the viewer's banner; it adds them itself)."""
        el = "" if t0 is None else ", %.0f s" % (time.time() - t0)
        report(item=item, state=state, msg=msg + el)
        write_json(LIVE / "launcher.json", {"state": state, "msg": msg, "t0": t0, "run": str(run), "t": time.time(),
                                            "batch": [sum(done(i) for i in ids), len(ids)]})

    scorer = Scorer(run, sc, base_image())
    jobs = []                                                    # (batch, scene, rollout dir, close)
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
    plays, skipped, chosen, driven = [], set(), set(), set()     # driven: finished, not skipped, maybe not scored yet
    res, batch, req, hold_error = None, None, None, ""          # req: the scene this loop asked for, until its request returns
    k, armed, t_wait, restart_why = len(list((run / "stacks").glob("*"))), True, time.time(), ""

    def done(s):
        return s in driven or (run / "done" / (s + ".json")).exists()

    def unfinished():
        order = [i for i in ctl.read()["order"] if i in ids] or ids
        return [i for i in order if not done(i) and i not in skipped]

    def view():
        """The driver's live state of this loop's scene (fresh, from the resident stack, that scene only)."""
        s = read_json(LIVE / "state.json", {}) or {}
        ok = res and req and s.get("run") == str(res.dir) and s.get("scene") == req["scene"] and time.time() - s.get("t", 0) < 5
        return s if ok and not s.get("closed") else {}

    def abort(why_, status=None):
        """End this loop's scene: abort.json now, and again once its session exists (the driver ignores older aborts)."""
        req.update(abort=why_, status=status, t_abort=0.0)

    def push_abort(v):
        if req and req.get("abort") and v and not v.get("aborted") and time.time() - req["t_abort"] > 2:
            write_json(LIVE / "abort.json", {"scene": req["scene"], "t": time.time(), "by": req["abort"]})
            req["t_abort"] = time.time()

    def collect():
        for scene, rd, close in scorer.pending(batch, final=True):
            if all(j[2] != rd for j in jobs):
                jobs.append((batch, scene, rd, close))
                if not close.get("aborted"):
                    driven.add(scene)
                    skipped.discard(scene)
        wake.set()

    try:
        while True:
            c = ctl.read()
            harm = track == "pai" and bool(c["options"].get("harmonizer"))
            size = str(c["options"].get("render") or "auto")
            key = (harm, size)
            hs = "harmonizer %s, render %s" % ("on" if harm else "off", size)
            if res:
                res.lease(owner, run)
            v = view()
            started = bool(v.get("started"))                     # the driver took over (first throttle press)
            for a in ctl.pending(st["consumed"]):
                st["consumed"].append(a["id"])
                if a["kind"] == "stop":
                    raise KeyboardInterrupt
                if a["kind"] == "play" and a["item"] in ids:
                    plays.append(a["item"])
                    chosen.add(a["item"])
                    hold_error = ""
                    if req and not req.get("abort") and not started and req["scene"] != a["item"]:
                        abort("play %s" % a["item"])               # a loading / held scene gives way, unscored
                elif a["kind"] == "next" and req and v and not req.get("abort"):
                    abort("next", "skipped (Next item)")
                    skipped.add(req["scene"])
                    note("next: skipping %s" % req["scene"])
                elif a["kind"] == "restart" and req and v and not req.get("abort"):
                    abort("restart")
                    plays.insert(0, req["scene"])
                    chosen.add(req["scene"])
                    note("restart: %s" % req["scene"])
            push_abort(v)
            # ---- the resident stack ----
            if res is not None and not res.alive():
                err = res.error()
                note("resident stack %s ended: %s" % (res.dir.name, err))
                if req:
                    hold_error = "the render stack stopped during %s: %s (see %s)" % (label(req["scene"]), err, res.dir)
                    req = None
                elif not res.ready():
                    hold_error = "the render stack failed to start: %s (see %s)" % (err, res.dir)
                res, batch = None, None
            if res is not None and res.key != key and not started:
                if req and not req.get("abort"):
                    abort(hs, None)
                    if req["scene"] not in plays and (c["playlist"] or req["scene"] in chosen):
                        plays.insert(0, req["scene"])
                if req is None:
                    note("stopping resident stack %s: %s" % (res.dir.name, hs))
                    progress("loading", "%s: stopping the render stack (the renderer restarts with the new setting)" % hs, time.time())
                    res.stop()
                    res, batch, restart_why = None, None, "%s: restarting the render stack (about a minute)" % hs
            if res is None:
                if hold_error:
                    progress("idle", hold_error + "; double-click a row to try again")
                    time.sleep(0.5)
                    continue
                res = Resident.find()
                if res is not None and not res.fits(track, key, ids):
                    note("stopping resident stack %s (%s, harmonizer %s, render %s): this batch needs %s, %s" % (
                        res.dir.name, res.track, res.harm, res.size, track, hs))
                    progress("loading", "stopping the %s render stack for this batch (%s, %s)" % (res.track, track, hs), time.time())
                    res.stop()
                    res, restart_why = None, "switching the render stack to %s, %s (about a minute)" % (track.upper(), hs)
                if res is None:
                    res = Resident.start(track, harm, size, list(dict.fromkeys(catalog + ids)))
                    note("resident stack %s: %s, %s, %d scenes" % (res.dir.name, track, hs, len(res.ids)))
                else:
                    note("reusing resident stack %s" % res.dir.name)
                k += 1
                batch = Batch(run, k, res)
                res.lease(owner, run)
            if not res.ready():
                ph = res.phase()
                why_ = restart_why or "starting the %s render stack (first start; about 45 s)" % track.upper()
                progress("loading", "%s: %s" % (why_, ph.get("msg", "launching")), ph.get("t0") or res.info["t"])
                time.sleep(0.3)
                continue
            restart_why = ""
            # ---- this loop's scene ----
            if req:
                r = res.result(req["name"])
                if r is not None:
                    batch.take(req["scene"], r)
                    bad = r.get("error") or not any(x.get("success") for x in r.get("rollouts", []))
                    if req.get("abort"):                          # aborted: never scored (with or without a complete rollout)
                        for x in r.get("rollouts", []):
                            scorer.abort_status[x["uuid"]] = None
                        if req.get("status"):
                            write_json(run / "tried" / (req["scene"] + ".json"), {"status": req["status"], "t": time.time()})
                    if bad and not req.get("abort"):
                        err = (r.get("error") or "; ".join(x.get("error", "") for x in r.get("rollouts", [])))[:300]
                        write_json(run / "tried" / (req["scene"] + ".json"), {"status": "rollout failed: %s" % err, "t": time.time()})
                        hold_error = "%s failed: %s" % (label(req["scene"]), err)
                        note(hold_error)
                    note("%s returned after %.0f s%s" % (req["scene"], r.get("t_done", 0) - req["t"],
                                                        " (aborted: %s)" % req["abort"] if req.get("abort") else ""))
                    collect()
                    req, armed, t_wait = None, False, time.time()
                    continue
                b = res.busy()
                if req.get("abort"):
                    progress("loading", "ending %s (%s): the rollout runs out unscored" % (label(req["scene"]), req["abort"]),
                             req.get("t_abort_0") or req.setdefault("t_abort_0", time.time()), req["scene"])
                elif b.get("name") != req["name"]:
                    progress("loading", "%s: waiting for the previous scene to end" % label(req["scene"]), req["t"], req["scene"])
                elif not v:
                    progress("loading", "loading %s: renderer loading the scene" % label(req["scene"]), b.get("t_sent"), req["scene"])
                elif v.get("waiting") and not started:
                    progress("paused" if c["paused"] else "driving", "resume on the panel" if c["paused"] else
                             "%s held: press throttle" % label(req["scene"]), None, req["scene"])
                else:
                    state = "paused" if c["paused"] else "driving"
                    progress(state, "resume on the panel" if c["paused"] else "%s%s" % (label(req["scene"]),
                             "   harmonizer on" if res.harm else ""), None, req["scene"])
                if res.key != key:
                    st["msg"] = "%s: the render stack restarts after this scene; %s" % (hs, st["msg"])
                    write_json(state_f, st)
                time.sleep(0.3)
                continue
            # ---- no scene: pick the next ----
            if c["paused"]:
                progress("paused", "resume on the panel")
                time.sleep(0.5)
                continue
            todo = None
            if plays:
                todo = plays.pop(0)
            elif hold_error:
                progress("idle", hold_error + "; double-click a row to try again")
                time.sleep(0.5)
                continue
            elif c["playlist"]:
                todo = (unfinished() or [None])[0]
            else:
                w = wheel.state()
                if not w["stale"] and w["raw_throttle"] < 0.1:
                    armed = True
                if armed and not w["stale"] and w["raw_throttle"] > 0.2 and unfinished():
                    todo = unfinished()[0]
                    chosen.add(todo)
            if todo is None:
                left = sum(not done(i) for i in ids)
                progress("idle", ("%d of %d scenes left: double-click a row%s" % (
                    left, len(ids), "" if c["playlist"] else " or press throttle")) if left else "all %d scenes driven" % len(ids))
                time.sleep(0.3)
                continue
            req = {"name": res.submit(todo, owner), "scene": todo, "t": time.time()}
            note("request %s (%s), %s, %.1f s after the previous scene" % (todo, label(todo), hs, time.time() - t_wait))
            progress("loading", "loading %s" % label(todo), req["t"], todo)
    except KeyboardInterrupt:
        pass
    finally:
        report(item=None, state="stopped", msg="")
        if res is not None:
            if req and res.result(req["name"]) is None:
                if res.busy().get("name") == req["name"]:
                    write_json(LIVE / "abort.json", {"scene": req["scene"], "t": time.time(), "by": "loop stop"})
                else:
                    try:
                        (res.feed / "req" / req["name"]).unlink()
                    except OSError:
                        pass
            res.release(owner)
        write_json(LIVE / "launcher.json", {"state": "stopped", "msg": "panel loop stopped%s" % (
            "; the render stack stays warm for %d min" % (int(os.environ.get("HUMAN_IDLE_S", "1200")) // 60) if res else ""),
            "run": str(run), "t": time.time()})
        for _ in range(60):
            if not jobs:
                break
            time.sleep(1)
        time.sleep(1)
        (run / "results.md").write_text(sc.table(run))


if __name__ == "__main__":
    main()
