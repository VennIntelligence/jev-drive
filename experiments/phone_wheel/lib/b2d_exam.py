"""Bench2Drive backend of the exam panel (lib/exam_panel.py): the human drives Bench2Drive 0.0.4 val routes on the Tokyo box.

Items = routes, group = town; human score = DS of the route's finished attempt in the run dir (done/<id>.json -> results.json),
model score = the best model's DS (results/model_scores.json: shipped Cinque + FIX1, decision 288; monitor runs only).
The drive side is scripts/human_loop.py in tmux human:loop; the viewer is scripts/human_viewer.py in tmux human:viewer.
"""
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "lib"))
from exam_panel import Backend, Control, read_json  # noqa: E402

RUNS = Path("/data/runs/human_drive")
B2D = Path(os.environ.get("BENCH2DRIVE_ROOT", "/data/third_party/Bench2Drive"))
XML = B2D / "leaderboard/data/bench2drive_0.0.4_val.xml"
PY = "/data/envs/carla/bin/python"
TURNS = re.compile("JunctionLeftTurn|JunctionRightTurn|T_Junction|VanillaSignalizedTurn|VanillaNonSignalizedTurn")
FAIL10 = "334 24948 27005 24944 27870 28147 27297 24416 26023 24519".split()
ENV = "export DISPLAY=:0 XAUTHORITY=/run/user/1001/gdm/Xauthority DATA_DIR=/data CARLA_ROOT=/data/third_party/carla/CARLA_0.9.15 " \
      "BENCH2DRIVE_ROOT=%s" % B2D


def sh(cmd):
    return subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


def town_key(t):
    return (t in ("Town12", "Town13"), t == "Town13", int("".join(c for c in t if c.isdigit()) or 0))


class B2DExam(Backend):
    title = "Bench2Drive human exam"
    live = os.environ.get("HUMAN_LIVE", "/data/runs/human_drive/live")
    options = {"stop_on_collision": ("Stop route at first collision", False)}

    def __init__(self):
        root = ET.parse(str(XML)).getroot()
        self.routes = [(r.get("id"), r.get("town"), r.find("scenarios").find("scenario").get("type")) for r in root.findall("route")]
        self.meta = {i: (t, s) for i, t, s in self.routes}
        self.model = {k: v for k, v in (read_json(REPO / "experiments/phone_wheel/results/model_scores.json", {}) or {}).items()
                      if not k.startswith("_")}
        self.cache = {}

    # ---- route sets: town-grouped, small towns first, Town12 / Town13 last, set order inside a town ----
    def batch_sets(self):
        return [("full", "Full val suite (220)"), ("hard", "Hard list (45)"), ("turns", "Junction turns (54)"),
                ("fail10", "Model failure routes (10)")]

    def set_ids(self, key):
        if key == "full":
            ids = [i for i, _, _ in self.routes]
        elif key == "turns":
            ids = [i for i, _, s in self.routes if TURNS.search(s)]
        elif key == "hard":
            ids = (REPO / "experiments/phone_wheel/results/exam_hard45.txt").read_text().split()
        else:
            ids = list(FAIL10)
        pos = {i: k for k, i in enumerate(ids)}
        return sorted(ids, key=lambda i: (town_key(self.meta[i][0]), pos[i]))

    # ---- runs and the loop ----
    def run_dir(self):
        st = read_json(Path(self.live) / "loop.json", {}) or {}
        if st.get("run") and Path(st["run"], "routes.txt").exists():
            return st["run"]
        runs = sorted((p for p in RUNS.glob("*/routes.txt")), key=lambda p: p.stat().st_mtime)
        return str(runs[-1].parent) if runs else None

    def loop_alive(self):
        st = read_json(Path(self.live) / "loop.json", {}) or {}
        return bool(st.get("pid")) and os.path.exists("/proc/%s" % st["pid"]) and st.get("state") != "stopped"

    def busy_other(self):
        procs = sh("pgrep -af '[s]cripts/b2d_run.py .*--out %s/' | grep -v pgrep" % RUNS).strip()
        if procs:
            out = re.search(r"--out (\S+)", procs)
            return "A human_drive.sh run is active on %s (started from a terminal)." % (out.group(1) if out else "?")
        if self.loop_alive():
            return "The panel loop is already running on %s." % self.run_dir()
        return ""

    def stop_other(self):
        if self.loop_alive():
            Control(self.live).act("stop")
            for _ in range(90):
                if not self.loop_alive():
                    break
                time.sleep(1)
        if sh("pgrep -f '[s]cripts/b2d_run.py .*--out %s/'" % RUNS).strip():
            sh("pkill -INT -f '[s]cripts/b2d_run.py .*--out %s/'" % RUNS)
            for _ in range(60):
                if not sh("pgrep -f '[s]cripts/b2d_run.py .*--out %s/'" % RUNS).strip():
                    break
                time.sleep(1)
            else:
                sh("pkill -TERM -f '[s]cripts/b2d_run.py .*--out %s/'" % RUNS)
            sh("tmux kill-window -t human:drive 2>/dev/null")
        return "stopped (finished routes are kept; the aborted one can be driven again)"

    def start_loop(self, run):
        sh("tmux has-session -t human 2>/dev/null || tmux new-session -d -s human -n main; tmux kill-window -t human:loop 2>/dev/null")
        sh("tmux new-window -d -t human -n loop \"%s; cd %s && %s experiments/phone_wheel/scripts/human_loop.py --run %s 2>&1 | "
           "tee -a %s/loop.log; exec bash\"" % (ENV, REPO, PY, run, run))
        free = sh("nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits").strip()
        return True, "Loop started on %s (card 0, %s MiB free). The first route loads and holds until the throttle." % (run, free)

    def new_batch(self, key):
        ids = self.set_ids(key)
        run = RUNS / time.strftime("%m%d-%H%M%S")
        run.mkdir(parents=True)
        (run / "routes.txt").write_text(",".join(ids))
        (run / "notes.txt").write_text("%s panel: new batch %s (%d routes); stop_on per route in the loop lines below\n"
                                       % (time.strftime("%F %T"), key, len(ids)))
        self.cache.clear()
        return self.start_loop(run)

    def resume_last(self):
        run = self.run_dir()
        if not run:
            return False, "no run with routes.txt in %s" % RUNS
        (Path(run) / "DONE").unlink() if (Path(run) / "DONE").exists() else None
        return self.start_loop(run)

    def ensure_side(self):
        msgs = []
        if sh("tmux has-session -t wheel 2>/dev/null && echo yes").strip() != "yes":
            sh("mkdir -p /data/tmp/wheel && cd %s && tmux new-session -d -s wheel -n server \"/data/envs/wheel/bin/python "
               "experiments/phone_wheel/scripts/wheel_server.py --lan-port 8766 2>&1 | tee -a /data/tmp/wheel/log.txt\"" % REPO)
            msgs.append("wheel daemon started (tmux wheel)")
        if not sh("pgrep -f 'phone_wheel/scripts/human_viewer.py'").strip():
            sh("tmux has-session -t human 2>/dev/null || tmux new-session -d -s human -n main; tmux kill-window -t human:viewer 2>/dev/null; "
               "tmux new-window -d -t human -n viewer \"%s SDL_VIDEO_WINDOW_POS=1340,40; cd %s; while true; do %s "
               "experiments/phone_wheel/scripts/human_viewer.py --dir %s --window-size 2480 1736; sleep 3; done\""
               % (ENV, REPO, PY, self.live))
            msgs.append("viewer started (tmux human:viewer)")
        return "\n".join(msgs)

    # ---- items ----
    def human(self, run, rid):
        d = Path(run, "done", rid + ".json")
        if not d.exists():
            return None, None
        key = (str(d), d.stat().st_mtime)
        if key not in self.cache:
            dj = read_json(d, {}) or {}
            res = read_json(Path(run, "attempts", rid, str(dj.get("attempt")), "results.json"), {}) or {}
            recs = (res.get("_checkpoint") or {}).get("records") or [{}]
            r = recs[0]
            inf = ", ".join("%s %d" % (k.replace("collisions_", "coll_"), len(v)) for k, v in (r.get("infractions") or {}).items()
                            if v and k != "min_speed_infractions")
            self.cache[key] = ((r.get("scores") or {}).get("score_composed"), "%s%s" % (r.get("status", "?"), "; " + inf if inf else ""))
        return self.cache[key]

    def items(self):
        run = self.run_dir()
        if not run:
            return []
        ids = Path(run, "routes.txt").read_text().strip().split(",")
        out = []
        for rid in ids:
            town, scen = self.meta.get(rid, ("?", "?"))
            h, status = self.human(run, rid)
            if h is None and Path(run, "attempts", rid).exists():
                status = "not finished"
            m = self.model.get(rid) or {}
            out.append(dict(id=rid, group=town, scenario=scen, human=h, model=m.get("ds"), status=status or ""))
        return out

    def official(self):
        run = self.run_dir()
        if not run:
            return ""
        sh("%s %s/experiments/phone_wheel/scripts/human_official.py %s >/dev/null 2>&1" % (PY, REPO, run))
        p = Path(run, "official", "official.md")
        return p.read_text() if p.exists() else "no official table (no finished route?)"
