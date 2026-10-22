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
- Record off (control "record" false, read when a scene returns): practice. Nothing is scored or kept (no done/ or tried/ record; the
  scene's rollout dirs and thumbnails are deleted); while it stays off, auto-advance skips scenes practised in this loop.
- Option "harmonizer" (PAI only) and option "render" (front camera height: auto = human_stack.sh's pairing with the harmonizer,
  else HUMAN_H) are fixed for a stack's whole life, so a change restarts the resident stack (about a minute) once no scene is
  being driven; a loading or held scene is requested again on the new stack.
- Option "driver" (README "Model mode"): "human" or a checkpoint tag. For a tag, scripts/model_driver.sh serves it (container
  <prefix>-model on the same card, the submission image's driver and served configuration) and <live>/driver.json names the scene and
  the model's address before each request, so lib/human_driver.py delegates that session to it. The scene starts by itself. A change
  takes effect at the next hold point (a loading scene is requested again). Model mode renders the front camera as the preset does
  (1080 px; PAI 10 Hz and the shipped harmonizer resolution also with the harmonizer) and, on PAI, with the submissions' steering weight
  (HUMAN_MODEL_WIZ, default controller.gains.rel_front_steering_angle_weight=10), all fixed per resident stack, so on PAI switching
  between human and model restarts the stack once (nuPlan: only with a render size other than auto). Scores of a model drive go to
  done/<scene>@<tag>.json ("driver" = tag), the human's to done/<scene>.json ("driver" = "human"): the two never mix.
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
import socket
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
MODEL_SCRIPT = REPO / "experiments/alpasim_human/scripts/model_driver.sh"
MODEL_PORT = os.environ.get("HUMAN_MODEL_PORT", "6791")
# README "Scene-to-scene gap": the next scene loads and holds while one drives. Off by default; on with HUMAN_PREFETCH=1 or the flag file
# /data/runs/alpasim_human/prefetch.on (read at loop start; the panel's loops then use it, and the resident stack restarts once).
PREFETCH = os.environ.get("HUMAN_PREFETCH", "1" if (D / "runs/alpasim_human/prefetch.on").exists() else "0") == "1"
MODEL_WIZ = {"pai": os.environ.get("HUMAN_MODEL_WIZ", "controller.gains.rel_front_steering_angle_weight=10").split(), "nuplan": []}


def dkey(scene, driver):
    """done/<key>.json of a scene's attempt by `driver`: the scene id for the human, <scene>@<tag> for a model."""
    return scene if driver in (None, "", "human") else "%s@%s" % (scene, driver)


class ModelSvc(object):
    """The model driver service of model mode (scripts/model_driver.sh, container <prefix>-model), started without blocking the loop.
    The container outlives the loop (a new loop with the same track / tag reuses it; its labels name track, tag, port, card, out
    dir); human mode, a tag switch, a failed start or the resident stack's idle stop remove it."""

    def __init__(self):
        self.name = "%s-model" % os.environ.get("HUMAN_PREFIX", "hres")
        self.track = self.tag = self.proc = None
        self.err, self.t0, self.out, self.ok, self.t_ok = "", 0.0, None, False, 0.0
        lab = self.labels()                                     # a container a previous loop left running
        if lab.get("human.port") == MODEL_PORT and lab.get("human.gpu") == GPU and lab.get("human.tag"):
            self.track, self.tag, self.out = lab.get("human.track"), lab["human.tag"], Path(lab.get("human.out") or "/tmp")

    def labels(self):
        try:
            return json.loads(sh("docker inspect -f '{{if .State.Running}}{{json .Config.Labels}}{{end}}' %s 2>/dev/null" % self.name)
                              .strip() or "{}") or {}
        except ValueError:
            return {}

    def running(self):
        """The container runs and MODEL_PORT accepts a connection (inspect at most every 5 s while true)."""
        if not self.ok or time.time() - self.t_ok > 5:
            up = sh("docker inspect -f '{{.State.Running}}' %s 2>/dev/null" % self.name).strip() == "true"
            if up:
                try:
                    socket.create_connection(("127.0.0.1", int(MODEL_PORT)), timeout=1).close()
                except OSError:
                    up = False
            self.ok, self.t_ok = up, time.time()
        return self.ok

    def ensure(self, track, tag):
        """True once `tag` listens on MODEL_PORT for `track` (model_driver.sh exited 0 for this start, or a reused container that
        accepts connections); starts (or switches) it otherwise. self.err: why it failed."""
        if (self.track, self.tag) == (track, tag):
            if self.proc is not None and self.proc.poll() is not None:
                rc, self.proc = self.proc.returncode, None
                if rc:
                    self.err = (self.out / "start.out").read_text().strip().splitlines()[-1:] or ["exit %d" % rc]
                    self.err = "model driver %s did not start: %s" % (tag, self.err[0])
                self.ok, self.t0 = False, time.time()
            if self.proc is not None or self.err:
                return False
            if self.running():
                return True
            if time.time() - self.t0 < 10:                      # just ready: the port check may lag the driver's log line
                return False
        self.stop()                                             # other tag, or the container went away: (re)start it
        self.track, self.tag, self.err, self.t0 = track, tag, "", time.time()
        self.out = D / "runs/alpasim_human/models" / ("%s-%s-%s" % (track, tag, time.strftime("%m%d-%H%M%S")))
        self.out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HUMAN_GPU=GPU, HUMAN_MODEL_PORT=MODEL_PORT, HUMAN_LIVE=str(LIVE), HUMAN_PREFIX=self.name[:-6])
        with open(str(self.out / "start.out"), "w") as log:
            self.proc = subprocess.Popen(["bash", str(MODEL_SCRIPT), track, tag, str(self.out)], env=env, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        return False

    def detach(self):
        """Loop exit: a started service stays for the next loop; one still starting is stopped."""
        if self.proc is not None or self.err:
            self.stop()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        if self.tag is not None and self.out is not None and self.out.is_dir():
            sh("docker logs %s > %s/driver.log 2>&1" % (self.name, self.out))
        sh("docker rm -f %s >/dev/null 2>&1" % self.name)
        self.track = self.tag = self.proc = None
        self.ok, self.t_ok = False, 0.0
        self.err = ""


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
        self.wiz = list(info.get("wiz", []))                    # extra wizard overrides (model mode: the steering weight)
        self.model = bool(info.get("model"))                    # PAI model mode: the preset's 10 Hz camera (also with the harmonizer)
        self.prefetch = bool(info.get("prefetch"))              # two rollouts at once (human_stack.sh HUMAN_PREFETCH)
        self.key = (self.harm, self.size, tuple(self.wiz), self.model, self.prefetch)
        self.ids, self.feed = set(info["ids"]), Path(info["dir"]) / "feed"

    @classmethod
    def start(cls, track, harm, size, ids, wiz=(), model=False, prefetch=False):
        d = RESIDENTS / ("%s%s%s-%s" % (track, "-harm" if harm else "", "" if size == "auto" else "-h" + size, time.strftime("%m%d-%H%M%S")))
        d.mkdir(parents=True)
        (d / "scenes.txt").write_text("\n".join(ids) + "\n")
        env = dict(os.environ, HUMAN_FG="1", HUMAN_VIEWER="0", HUMAN_SERVE="1", HUMAN_CONTROL=str(LIVE / "control.json"),
                   HUMAN_LIVE=str(LIVE), HUMAN_WAIT_MAX_S=os.environ.get("HUMAN_WAIT_MAX_S", "1800"), HUMAN_GPU=GPU,
                   HUMAN_HARMONIZER="1" if harm else "0", HUMAN_PREFIX=os.environ.get("HUMAN_PREFIX", "hres"),
                   HUMAN_PREFETCH="1" if prefetch else "0")
        if size != "auto":
            env["HUMAN_H"] = size
        if model:                                               # the served camera: 10 Hz, shipped harmonizer resolution (human_stack.sh)
            env.update(HUMAN_FI_MS="100", HUMAN_HARM_RES="")
        with open(str(d / "stack.out"), "a") as log:
            proc = subprocess.Popen(["python3" if STACK.suffix == ".py" else "bash", str(STACK), track, str(d), str(d / "scenes.txt")] + list(wiz),
                                    env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        info = {"dir": str(d), "pid": proc.pid, "track": track, "harm": harm, "size": size, "wiz": list(wiz), "model": model, "prefetch": prefetch, "gpu": GPU, "ids": list(ids), "t": time.time(),
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
        self.buf = {f: [] for f in self.off}                    # lines read but not yet claimed (a prefetched session's, until it returns)

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
            lines = self.buf[f] + [ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()]
            mine = [ln for ln in lines if json.loads(ln).get("session") in uu]
            self.buf[f] = [ln for ln in lines if json.loads(ln).get("session") not in uu][-100000:]
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
        driver = close.get("driver") or drv.get("driver") or "human"
        prev = read_json(self.run / "done" / (dkey(scene, driver) + ".json"), {}) or {}
        rec = {"scene": scene, "score": r.get("score") or 0.0, "why": why(r), "rollout": rd.name, "stack": stack.dir.name,
               "progress": round(sm.get("progress_clipped_rel") or 0.0, 3), "driven_m": round(m.get("dist_traveled_m") or 0.0, 1),
               "gt_m": round(m.get("gt_dist_traveled_m") or 0.0, 1), "sim_s": drv.get("sim_s"), "wall_per_sim": drv.get("wall_per_sim"),
               "stale": drv.get("stale"), "wait_s": drv.get("wait_s"), "paused_s": drv.get("paused_s"), "harmonizer": stack.harm, "render": stack.res.size,
               "shadow_s": drv.get("shadow_s") or 0.0, "driver": driver,
               "attempt": prev.get("attempt", 0) + 1, "t": time.time()}
        write_json(self.run / "done" / (dkey(scene, driver) + ".json"), rec)
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
        report(item=item, state=state, msg=msg + el, scoring=len(jobs))
        practice = "" if ctl.read()["record"] else "practice, not recorded: "
        write_json(LIVE / "launcher.json", {"state": state, "msg": practice + msg, "t0": t0, "run": str(run), "t": time.time(),
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
    practised = set()                                            # driven with record off (not counted)
    res, batch, hold_error = None, None, ""
    reqs = []                                                    # this loop's requests in flight, oldest first (2 with prefetch)
    model, drv = ModelSvc(), "human"                             # model mode: the model service; the panel's driver now
    k, armed, t_wait, restart_why = len(list((run / "stacks").glob("*"))), True, time.time(), ""
    t_push = 0.0

    def done(s):                                                 # by the driver chosen now
        return dkey(s, drv) in driven or (run / "done" / (dkey(s, drv) + ".json")).exists()

    def unfinished(exclude=()):
        c = ctl.read()
        order = [i for i in c["order"] if i in ids] or ids
        return [i for i in order if not done(i) and i not in skipped and i not in exclude and (c["record"] or i not in practised)]

    def live_sessions():
        """The driver's open sessions (<live>/sessions.json; drivers before prefetch do not write it: [])."""
        d = read_json(LIVE / "sessions.json", {}) or {}
        return d.get("open", []) if res and d.get("run") == str(res.dir) and time.time() - d.get("t", 0) < 3600 else []

    def bind(opn):
        """Map requests to their sessions (same scene, created after the request was written, not taken by another request)."""
        taken = {r.get("session") for r in reqs}
        for r in reqs:
            if r.get("session") is None:
                x = next((o for o in opn if o["scene"] == r["scene"] and o["t_created"] >= r["t"] - 0.5 and o["session"] not in taken), None)
                if x:
                    r["session"] = x["session"]
                    taken.add(x["session"])
            o = next((o for o in opn if o["session"] == r.get("session")), None)
            if o:
                r["started"] = r.get("started") or o["started"]

    def view():
        """(request, the driver's live state) of the session the viewer shows, if it is one of this loop's; else (None, {})."""
        s = read_json(LIVE / "state.json", {}) or {}
        if not (res and s.get("run") == str(res.dir) and time.time() - s.get("t", 0) < 5) or s.get("closed"):
            return None, {}
        for r in reqs:
            if (s.get("session") and s.get("session") == r.get("session")) or (r.get("session") is None and s.get("scene") == r["scene"]
                                                                               and not (s.get("session") and any(q.get("session") == s.get("session") for q in reqs))):
                if s.get("started"):
                    r["started"] = True
                r["seen"] = True
                return r, s
        return None, {}

    def abort(r, why_, status=None):
        """End request r: unsent, it is withdrawn; sent, its session is skipped (abort.json, by uuid once known)."""
        r.update(abort=why_, status=status, t_abort=0.0)
        if not any(b.get("name") == r["name"] for b in res.busy().get("items", [res.busy()] if res.busy() else [])):
            try:
                (res.feed / "req" / r["name"]).unlink()
                r["withdrawn"] = time.time()
            except OSError:
                pass

    def push_abort(opn):
        """abort.json: one item per aborted request still open, re-written every 2 s (the driver ignores scene-only items older than a
        session, so an item is written again once the session exists)."""
        nonlocal t_push
        items = []
        for r in reqs:
            if not r.get("abort") or r.get("withdrawn"):
                continue
            o = next((o for o in opn if o["session"] == r.get("session")), None)
            if o and o["aborted"]:
                continue
            items.append({"scene": r["scene"], "t": time.time(), "by": r["abort"], **({"session": r["session"]} if r.get("session") else {})})
        if items and time.time() - t_push > 2:
            write_json(LIVE / "abort.json", {**items[-1], "items": items})
            t_push = time.time()

    def discard(scene, r):
        """Record off: the returned rollouts are never scored and their files go."""
        for u in {x["uuid"] for x in r.get("rollouts", []) if x.get("uuid")}:
            scorer.seen.add(u)
            shutil.rmtree(str(batch.dir / "sim/rollouts" / scene / u), ignore_errors=True)
            try:
                (batch.dir / "driver/thumbs" / (u + ".jpg")).unlink()
            except OSError:
                pass
        practised.add(scene)
        note("practice: %s not recorded (rollouts deleted)" % scene)

    def collect(rq):
        for scene, rd, close in scorer.pending(batch, final=True):
            if all(j[2] != rd for j in jobs):
                jobs.append((batch, scene, rd, close))
                if not close.get("aborted"):
                    driven.add(dkey(scene, close.get("driver") or rq.get("driver")))
                    skipped.discard(scene)
        wake.set()

    def returned(r, out):
        """Request r came back with result `out` (None: withdrawn before it was sent)."""
        nonlocal hold_error
        reqs.remove(r)
        if out is None:
            note("%s withdrawn before it was sent (%s)" % (r["scene"], r.get("abort")))
            return
        batch.take(r["scene"], out)
        bad = out.get("error") or not any(x.get("success") for x in out.get("rollouts", []))
        if not ctl.read()["record"]:
            discard(r["scene"], out)
            return
        if r.get("abort"):                                       # aborted: never scored (with or without a complete rollout)
            for x in out.get("rollouts", []):
                scorer.abort_status[x["uuid"]] = None
            if r.get("status"):
                write_json(run / "tried" / (r["scene"] + ".json"), {"status": r["status"], "t": time.time()})
        if bad and not r.get("abort"):
            err = (out.get("error") or "; ".join(x.get("error", "") for x in out.get("rollouts", [])))[:300]
            write_json(run / "tried" / (r["scene"] + ".json"), {"status": "rollout failed: %s" % err, "t": time.time()})
            hold_error = "%s failed: %s" % (label(r["scene"]), err)
            note(hold_error)
        note("%s returned after %.0f s%s" % (r["scene"], out.get("t_done", 0) - r["t"], " (aborted: %s)" % r["abort"] if r.get("abort") else ""))
        collect(r)

    def submit(todo, prefetch=False):
        nonlocal t_wait
        d = read_json(LIVE / "driver.json", {}) or {}
        sc = {q["scene"]: (d.get("scenes") or {}).get(q["scene"]) for q in reqs if (d.get("scenes") or {}).get(q["scene"])}
        sc[todo] = {"tag": drv, "addr": "127.0.0.1:%s" % MODEL_PORT if mm else ""}
        write_json(LIVE / "driver.json", {"scene": todo, "tag": drv, "addr": sc[todo]["addr"], "track": track, "run": str(run), "t": time.time(),
                                          "scenes": sc})
        r = {"name": res.submit(todo, owner), "scene": todo, "t": time.time(), "driver": drv, "prefetch": prefetch}
        reqs.append(r)
        note("request %s (%s), %s, %s" % (todo, label(todo), hs, "prefetched while %s drives" % reqs[0]["scene"] if prefetch else
                                          "%.1f s after the previous scene" % (time.time() - t_wait)))
        return r

    try:
        while True:
            c = ctl.read()
            harm = track == "pai" and bool(c["options"].get("harmonizer"))
            size = str(c["options"].get("render") or "auto")
            drv = str(c["options"].get("driver") or "human")
            mm = drv != "human"
            if mm and (harm or size != "auto"):                  # model mode: the preset's front camera (auto unharmonised = 1080)
                size = "1080"
            key = (harm, size, tuple(MODEL_WIZ[track] if mm else ()), mm and track == "pai", PREFETCH)
            hs = "harmonizer %s, render %s%s" % ("on" if harm else "off", size, ", driver %s" % drv if mm else "")
            if res:
                res.lease(owner, run)
            opn = live_sessions()
            bind(opn)
            cur, v = view()                                      # the request the viewer shows (the current scene to the user)
            live_reqs = [r for r in reqs if not r.get("abort")]
            if cur is None and live_reqs:
                cur = live_reqs[0]
            started = bool(cur and cur.get("started"))           # the driver took over (first throttle press)
            for a in ctl.pending(st["consumed"]):
                st["consumed"].append(a["id"])
                if a["kind"] == "stop":
                    raise KeyboardInterrupt
                if a["kind"] == "play" and a["item"] in ids:
                    plays.append(a["item"])
                    chosen.add(a["item"])
                    hold_error = ""
                    if model.err:                                # a model driver that failed to start is tried again
                        model.track = model.tag = None
                        model.err = ""
                    for r in reqs:                               # a loading / held / prefetched scene gives way, unscored
                        if not r.get("abort") and not r.get("started") and r["scene"] != a["item"]:
                            abort(r, "play %s" % a["item"])
                elif a["kind"] == "next" and cur and v and not cur.get("abort"):
                    abort(cur, "next", "skipped (Next item)")
                    skipped.add(cur["scene"])
                    note("next: skipping %s" % cur["scene"])
                elif a["kind"] == "restart" and cur and v and not cur.get("abort"):
                    abort(cur, "restart")
                    for r in reqs:                               # the restarted scene goes first: a prefetched one gives way
                        if r is not cur and not r.get("abort") and not r.get("started"):
                            abort(r, "restart %s" % cur["scene"])
                    plays.insert(0, cur["scene"])
                    chosen.add(cur["scene"])
                    note("restart: %s" % cur["scene"])
            push_abort(opn)
            # ---- the resident stack ----
            if res is not None and not res.alive():
                err = res.error()
                note("resident stack %s ended: %s" % (res.dir.name, err))
                if reqs:
                    hold_error = "the render stack stopped during %s: %s (see %s)" % (label(reqs[0]["scene"]), err, res.dir)
                    reqs.clear()
                elif not res.ready():
                    hold_error = "the render stack failed to start: %s (see %s)" % (err, res.dir)
                res, batch = None, None
            for r in reqs:
                if not r.get("abort") and not r.get("started") and (r.get("driver", "human") != drv or (res is not None and res.key != key)):
                    abort(r, "driver %s" % drv if r.get("driver", "human") != drv else hs)   # requested again with the new setting
                    if r["scene"] not in plays and (c["playlist"] or r["scene"] in chosen):
                        plays.insert(0, r["scene"])
            if not mm and model.tag is not None and not reqs:
                note("model driver %s stopped (driver: human)" % model.tag)
                model.stop()
            if res is not None and res.key != key and not reqs:
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
                    res = Resident.start(track, harm, size, list(dict.fromkeys(catalog + ids)), key[2], key[3], PREFETCH)
                    note("resident stack %s: %s, %s, %d scenes%s" % (res.dir.name, track, hs, len(res.ids), ", prefetch" if PREFETCH else ""))
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
            # ---- returned requests ----
            back = False
            busy = res.busy()
            names = {b.get("name") for b in busy.get("items", [busy] if busy else [])}
            for r in list(reqs):
                if r.get("withdrawn") and r["name"] in names:   # the feeder sent it as it was withdrawn: abort its session instead
                    r.pop("withdrawn")
                out = res.result(r["name"])
                if out is not None:
                    returned(r, out)
                    back = True
                elif r.get("withdrawn") and r["name"] not in names and time.time() - r["withdrawn"] > 2:
                    returned(r, None)
                    back = True
            if back:
                armed, t_wait = False, time.time()
                continue
            # ---- the scenes in flight: progress, and the next one prefetched ----
            if reqs:
                r0 = cur if cur is not None else reqs[0]
                if r0.get("abort"):
                    progress("loading", "ending %s (%s): the rollout runs out unscored" % (label(r0["scene"]), r0["abort"]),
                             r0.get("t_abort_0") or r0.setdefault("t_abort_0", time.time()), r0["scene"])
                elif r0["name"] not in names:
                    progress("loading", "%s: waiting for the previous scene to end" % label(r0["scene"]), r0["t"], r0["scene"])
                elif not v:
                    progress("loading", "loading %s: renderer loading the scene" % label(r0["scene"]), r0["t"], r0["scene"])
                elif v.get("model_wait") is not None and not started:
                    progress("loading", "loading %s: waiting for model driver %s to accept connections" % (label(r0["scene"]), r0["driver"]),
                             r0["t"], r0["scene"])
                elif v.get("waiting") and not started:
                    progress("paused" if c["paused"] else "driving", "resume on the panel" if c["paused"] else
                             "%s held: press throttle" % label(r0["scene"]), None, r0["scene"])
                else:
                    model_req = r0.get("driver", "human") != "human"
                    progress("paused" if c["paused"] else "driving", "resume on the panel" if c["paused"] else "%s%s%s%s" % (label(r0["scene"]),
                             "   harmonizer on" if res.harm else "", "   driver: model %s" % r0["driver"] if model_req else "",
                             "   (model + harmonizer at shipped settings: about 4 wall s per sim s, slow motion)" if model_req and res.harm else ""),
                             None, r0["scene"])
                if res.key != key:
                    st["msg"] = "%s: the render stack restarts after this scene; %s" % (hs, st["msg"])
                    write_json(state_f, st)
                # prefetch: one more request while the current scene's session exists (loaded), once nothing aborted is in flight
                live_reqs = [r for r in reqs if not r.get("abort")]
                if (PREFETCH and res.prefetch and len(reqs) == 1 and len(live_reqs) == 1 and (live_reqs[0].get("session") or live_reqs[0].get("seen"))
                        and not hold_error and res.key == key and live_reqs[0].get("driver", "human") == drv and (not mm or model.ensure(track, drv))):
                    nxt = plays[0] if plays else (unfinished(exclude={live_reqs[0]["scene"]}) or [None])[0] if c["playlist"] else None
                    if nxt is not None and nxt != live_reqs[0]["scene"]:
                        if plays:
                            plays.pop(0)
                        submit(nxt, prefetch=True)
                time.sleep(0.3)
                continue
            # ---- no scene: pick the next ----
            if c["paused"]:
                progress("paused", "resume on the panel")
                time.sleep(0.5)
                continue
            if mm and not model.ensure(track, drv):              # model mode: its service first (about 25 s on a switch)
                if model.err and not hold_error:
                    hold_error = model.err + " (see %s)" % model.out
                    note(hold_error)
                progress(*(("idle", hold_error + "; double-click a row to try again") if model.err else
                           ("loading", "starting the model driver %s (%s, about 25 s)" % (drv, track.upper()), model.t0)))
                time.sleep(0.3)
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
                who = "" if not mm else " by %s" % drv
                progress("idle", ("%d of %d scenes left%s: double-click a row%s" % (
                    left, len(ids), who, "" if c["playlist"] else " or press throttle")) if left else "all %d scenes driven%s" % (len(ids), who))
                time.sleep(0.3)
                continue
            r = submit(todo)
            progress("loading", "loading %s" % label(todo), r["t"], todo)
    except KeyboardInterrupt:
        pass
    finally:
        report(item=None, state="stopped", msg="")
        if res is not None:                                     # a warm stack keeps a started model driver for the next loop
            model.detach()
        else:
            model.stop()
        if res is not None:
            busy = res.busy()
            names = {b.get("name") for b in busy.get("items", [busy] if busy else [])}
            items = []
            for r in reqs:
                if res.result(r["name"]) is not None:
                    continue
                if r["name"] in names:
                    items.append({"scene": r["scene"], "t": time.time(), "by": "loop stop", **({"session": r["session"]} if r.get("session") else {})})
                else:
                    try:
                        (res.feed / "req" / r["name"]).unlink()
                    except OSError:
                        pass
            if items:
                write_json(LIVE / "abort.json", {**items[-1], "items": items})
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
