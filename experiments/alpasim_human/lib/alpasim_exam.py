"""AlpaSim backend of the exam panel (lib/exam_panel.py): the human drives AlpaSim scenes (PAI or nuPlan track) on Tokyo card 0.

Items = scenes. Groups: PAI = track + NuRec artifact revision (26.01 / 26.04), nuPlan = the navtest log segment.
Human score = the AlpaSim scene score of the scene's latest finished, not skipped rollout in the run (done/<scene>.json, written by
scripts/human_loop.py). Model score = P2H10-F-s0 on the same scene where a read exists (PAI stage 1: pai1/runs/a10_c4; nuPlan 48:
native_ref/P2H10-F-s0), else the organisers' alpamayo1 (PAI tsv column); the source is shown after the score.
A run dir: /data/runs/alpasim_human/exam_<track>_<stamp>/ with exam.json (track, set, ids), stacks/<one human_stack.sh out dir per
stack launch>, done/<scene>.json, tried/<scene>.json, scores/<rollout id>/ (post-hoc aggregation), scores.jsonl, results.md, loop.log.
The drive side is scripts/human_loop.py in tmux human:alpa-loop; the viewer is scripts/human_viewer.py in tmux human:alpa-viewer.
Plain python3 (stdlib).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "lib"))
from exam_panel import Backend, Control, read_json  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/data"))
RUNS = D / "runs/alpasim_human"
A = D / "runs/alpasim"
PAI_TSV = A / "pai1/pai_scenes_40.tsv"
NUPLAN_LIST = A / "scenes_navtest_full_part001_48.txt"
MODELS = {"pai": [("P2H10", A / "pai1/runs/a10_c4")], "nuplan": [("P2H10", A / "native_ref/P2H10-F-s0")]}
EXTRA = {"pai": [("alpamayo1", "tsv:alpamayo1"), ("organisers' mean", "tsv:ref_mean")],
         "nuplan": [("SH30-F-s0", A / "native_ref/sh30_full48_c8")]}
FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")
ENV = "export DISPLAY=:0 XAUTHORITY=/run/user/1001/gdm/Xauthority DATA_DIR=%s" % D


def sh(cmd):
    return subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


def pai_rows():
    rows = [ln.split("\t") for ln in PAI_TSV.read_text().splitlines()]
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def summary_scores(p):
    """scene -> mean score over the rollouts of a results-summary.json (file, run dir or its sim/ dir)."""
    for c in (p, p / "aggregate/results-summary.json", p / "sim/aggregate/results-summary.json"):
        if c.is_file():
            out = {}
            for r in json.loads(c.read_text())["rollouts"]:
                out.setdefault(r["clipgt_id"], []).append(r["score"] or 0.0)
            return {k: sum(v) / len(v) for k, v in out.items()}
    return {}


def why(r):
    m = r.get("score_metrics") or {}
    z = [f for f in FAIL if m.get(f)]
    if r.get("failure_reason") and not z:
        return "failed: " + str(r["failure_reason"])[:40]
    return ", ".join(z) if z else "pass" if (r.get("score") or 0) >= 1 else "slow"


class Scenes(object):
    """Scene sets, groups and reference scores of both tracks (shared by the panel backend and the loop)."""
    SETS = [("pai1", "PAI stage 1 (10)"), ("pai40", "PAI all 40"), ("nuplan48", "nuPlan navtest 48")]

    def __init__(self):
        self.meta, self.refs = {}, {}
        tsv = pai_rows() if PAI_TSV.exists() else []
        self.pai = tsv
        for k, r in enumerate(tsv):
            self.meta[r["scene_id"]] = dict(track="pai", group="PAI %s" % r["revision"], stage=int(r["stage"]), k=k,
                                            label=r["scene_id"][-12:],
                                            scenario="stage %s, organisers' mean %.2f" % (r["stage"], float(r["ref_mean"])))
        self.nuplan = NUPLAN_LIST.read_text().split() if NUPLAN_LIST.exists() else []
        for k, s in enumerate(self.nuplan):
            seg = s.split("_")[2:4] if s.count("_") >= 3 else ["?", "?"]
            self.meta[s] = dict(track="nuplan", group="nuPlan log %s" % "_".join(seg), k=k, label=s.split("_", 2)[-1][:28],
                                scenario="navtest %s" % s.split("_")[0])
        for track in ("pai", "nuplan"):
            for name, src in MODELS[track] + EXTRA[track]:
                if str(src).startswith("tsv:"):
                    col = str(src)[4:]
                    self.refs[(track, name)] = {r["scene_id"]: float(r[col]) for r in tsv if r.get(col) not in (None, "")}
                else:
                    self.refs[(track, name)] = summary_scores(Path(src))

    def ids(self, key):
        if key.startswith("pai"):
            rows = [r for r in self.pai if key == "pai40" or int(r["stage"]) <= 1]
            ids = [r["scene_id"] for r in rows]
            return sorted(ids, key=lambda i: (self.meta[i]["group"], self.meta[i]["stage"], self.meta[i]["k"]))
        return sorted(self.nuplan, key=lambda i: (self.meta[i]["group"], self.meta[i]["k"]))

    def model(self, scene):
        """(score, source) of the best available model read on this scene: P2H10 first, else the organisers' alpamayo1."""
        track = self.meta.get(scene, {}).get("track", "pai")
        for name, _ in MODELS[track] + EXTRA[track][:1]:
            v = self.refs.get((track, name), {}).get(scene)
            if v is not None:
                return v, ("a1" if name == "alpamayo1" else name)
        return None, None

    def table(self, run):
        """results.md of a panel run: one row per driven scene in run order, means over the driven scenes."""
        run = Path(run)
        ex = read_json(run / "exam.json", {}) or {}
        track = ex.get("track", "pai")
        cols = [n for n, _ in MODELS[track] + EXTRA[track]]
        rows = []
        for sid in ex.get("ids", []):
            d = read_json(run / "done" / (sid + ".json"))
            if d:
                rows.append((sid, d, [self.refs.get((track, n), {}).get(sid) for n in cols]))
        L = ["%s, set %s: %d of %d scenes driven (latest finished attempt per scene; skipped attempts do not count)" % (
            track, ex.get("set"), len(rows), len(ex.get("ids", []))), "",
            "| scene | group | human | why | progress | driven / GT m | wall per sim s | stale | attempts | " + " | ".join(cols) + " |",
            "|---|---|--:|---|--:|--:|--:|--:|--:|" + "--:|" * len(cols)]
        for sid, d, refs in rows:
            L.append("| %s | %s | %.3f | %s | %s | %s / %s | %s | %s | %s | %s |" % (
                self.meta.get(sid, {}).get("label", sid), self.meta.get(sid, {}).get("group", "?"), d["score"], d["why"],
                d.get("progress"), d.get("driven_m"), d.get("gt_m"), d.get("wall_per_sim"), d.get("stale"), d.get("attempt", 1),
                " | ".join("-" if v is None else "%.3f" % v for v in refs)))
        if rows:
            mean = lambda v: sum(v) / len(v)  # noqa: E731
            parts = ["human %.3f" % mean([d["score"] for _, d, _ in rows])]
            for j, n in enumerate(cols):
                v = [r[j] for _, _, r in rows if r[j] is not None]
                parts.append("%s %s%s" % (n, "%.3f" % mean(v) if v else "-", "" if len(v) == len(rows) else " (%d scenes)" % len(v)))
            L += ["", "Means over the %d driven scenes: %s." % (len(rows), "; ".join(parts))]
        L += ["", "Fast mode (front camera only): a human experience, not a leaderboard number (README, section Speed)."]
        return "\n".join(L) + "\n"


class AlpaSimExam(Backend):
    title = "AlpaSim human exam"
    live = os.environ.get("HUMAN_LIVE", str(RUNS / "live"))
    score_fmt = "%.2f"
    options = {}

    def __init__(self):
        self.sc = Scenes()

    def batch_sets(self):
        return list(Scenes.SETS)

    # ---- runs and the loop ----
    def run_dir(self):
        st = read_json(Path(self.live) / "loop.json", {}) or {}
        if st.get("run") and Path(st["run"], "exam.json").exists():
            return st["run"]
        runs = sorted(RUNS.glob("exam_*/exam.json"), key=lambda p: p.stat().st_mtime)
        return str(runs[-1].parent) if runs else None

    def loop_alive(self):
        st = read_json(Path(self.live) / "loop.json", {}) or {}
        return bool(st.get("pid")) and os.path.exists("/proc/%s" % st["pid"]) and st.get("state") != "stopped"

    @staticmethod
    def carla_on_card0():
        uuid = sh("nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader").strip()
        return bool(uuid) and any(ln.startswith(uuid) and "CarlaUE4" in ln for ln in
                                  sh("nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader").splitlines())

    def busy_other(self):
        if self.loop_alive():
            return "The panel loop is already running on %s." % self.run_dir()
        hum = sh("docker ps --format '{{.Names}}' | grep '^hum-'").split()
        if hum:
            return "An AlpaSim human stack started from a terminal is running (%s)." % ", ".join(hum)
        if self.carla_on_card0():
            return "CARLA runs on card 0 (a Bench2Drive exam?)."
        return ""

    def stop_other(self):
        msgs = []
        if self.loop_alive():
            Control(self.live).act("stop")
            for _ in range(90):
                if not self.loop_alive():
                    break
                time.sleep(1)
            msgs.append("panel loop stopped")
        if sh("docker ps --format '{{.Names}}' | grep '^hum-'").strip():
            sh("tmux send-keys -t human:alpa-drive C-c 2>/dev/null")       # human_stack.sh's trap takes its stack down
            for _ in range(60):
                if not sh("docker ps --format '{{.Names}}' | grep '^hum-'").strip():
                    break
                time.sleep(1)
            msgs.append("terminal stack stopped")
        if self.carla_on_card0():
            msgs.append("CARLA on card 0 belongs to the Bench2Drive exam: stop it on the B2D panel; this panel does not touch it")
        return "; ".join(msgs) or "nothing to stop"

    def start_loop(self, run):
        if self.carla_on_card0():
            return False, "Card 0 runs CARLA (the Bench2Drive exam?). Stop it on the B2D panel first."
        sh("tmux has-session -t human 2>/dev/null || tmux new-session -d -s human -n main; tmux kill-window -t human:alpa-loop 2>/dev/null")
        sh("tmux new-window -d -t human -n alpa-loop \"%s; cd %s && /usr/bin/python3 experiments/alpasim_human/scripts/human_loop.py "
           "--run %s 2>&1 | tee -a %s/loop.log; exec bash\"" % (ENV, REPO, run, run))
        free = sh("nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits").strip()
        return True, ("Loop started on %s (card 0, %s MiB free). The stack takes about a minute to start; each scene then holds "
                      "until the first throttle press." % (run, free))

    def new_batch(self, key):
        ids = self.sc.ids(key)
        track = "pai" if key.startswith("pai") else "nuplan"
        run = RUNS / ("exam_%s_%s" % (track, time.strftime("%m%d-%H%M%S")))
        run.mkdir(parents=True)
        (run / "exam.json").write_text(json.dumps({"track": track, "set": key, "label": dict(Scenes.SETS)[key], "ids": ids,
                                                   "created": time.strftime("%F %T")}, indent=1))
        (run / "notes.txt").write_text("%s panel: new batch %s (%d scenes)\n" % (time.strftime("%F %T"), key, len(ids)))
        return self.start_loop(run)

    def resume_last(self):
        run = self.run_dir()
        if not run:
            return False, "no panel run (exam_*/exam.json) in %s" % RUNS
        return self.start_loop(run)

    def ensure_side(self):
        msgs = []
        if sh("tmux has-session -t wheel 2>/dev/null && echo yes").strip() != "yes":
            sh("mkdir -p /data/tmp/wheel && cd %s && tmux new-session -d -s wheel -n server \"/data/envs/wheel/bin/python "
               "experiments/phone_wheel/scripts/wheel_server.py --lan-port 8766 2>&1 | tee -a /data/tmp/wheel/log.txt\"" % REPO)
            msgs.append("wheel daemon started (tmux wheel)")
        if not sh("pgrep -f 'alpasim_human/scripts/human_viewer.py'").strip():
            sh("tmux has-session -t human 2>/dev/null || tmux new-session -d -s human -n main; tmux kill-window -t human:alpa-viewer 2>/dev/null; "
               "tmux new-window -d -t human -n alpa-viewer \"%s SDL_VIDEO_WINDOW_POS=1340,40; cd %s; while true; do /data/envs/carla/bin/python "
               "experiments/alpasim_human/scripts/human_viewer.py --dir %s --window-size 2480 1736; sleep 3; done\"" % (ENV, REPO, self.live))
            msgs.append("viewer started (tmux human:alpa-viewer)")
        return "\n".join(msgs)

    # ---- items ----
    def items(self):
        run = self.run_dir()
        if not run:
            return []
        ex = read_json(Path(run, "exam.json"), {}) or {}
        out = []
        for sid in ex.get("ids", []):
            m = self.sc.meta.get(sid, {})
            d = read_json(Path(run, "done", sid + ".json"))
            t = read_json(Path(run, "tried", sid + ".json"))
            if d:
                status = "%s, progress %s%s" % (d["why"], d.get("progress"), ", attempt %d" % d["attempt"] if d.get("attempt", 1) > 1 else "")
            elif t:
                status = t.get("status", "not finished")
            else:
                status = ""
            ms, src = self.sc.model(sid)
            out.append(dict(id=sid, label=m.get("label", sid), group=m.get("group", "?"), scenario=m.get("scenario", ""),
                            human=d["score"] if d else None, model=ms, model_src=src, status=status))
        return out

    def official(self):
        run = self.run_dir()
        return self.sc.table(run) if run else "no run"


def stack_name(out):
    """Compose project of a human_stack.sh out dir (as the script derives it)."""
    return "hum-" + re.sub(r"[^a-zA-Z0-9]", "-", Path(out).name).lower()
