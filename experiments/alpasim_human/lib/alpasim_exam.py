"""AlpaSim backend of the exam panel (lib/exam_panel.py): the human drives AlpaSim scenes (PAI or nuPlan track) on Tokyo card 0.

Items = scenes. Groups: PAI = track + NuRec artifact revision (26.01 / 26.04), nuPlan = the navtest log segment.
Human score = the AlpaSim scene score of the scene's latest finished, not skipped rollout in the run (done/<scene>.json, written by
scripts/human_loop.py). Model score = R6DAG-L18-s1 (PAI 281 ACC + GAP, results/model_ref_r6dag_281.tsv), else P2H10-F-s0 where a read
exists (PAI stage 1: pai1/runs/a10_c4; nuPlan 48: native_ref/P2H10-F-s0), else the organisers' alpamayo1 (PAI tsv column); the source
is shown after the score. Sets (Scenes): filters over the PAI 441 table (revision, organisers' mean, R6DAG score / loss class) plus a
custom pick in the panel.
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
CKPTS = D / "runs/op_parity/runs"
# Driver choices marked from research/decisions.md "现状" (2026-10-22): tag -> note. Every other checkpoint on Tokyo follows, alphabetical.
CURRENT = {"NTV0-L18-s0": "PAI submitted p1021a", "R8K-L18-s2": "PAI lead family", "R6DAG-L18-s1": "PAI lead family",
           "RSG-L18-s0": "PAI submitted (vennai)", "JTN-L18-s1": "nuPlan submitted"}
NOT_DRIVERS = re.compile(r"^(TB-|ONP-|AP)")                 # trajectory-bank probes, identity tests, AlpaSim-standard (ap2) family


def driver_choices():
    """The panel's Driver choices: [value, label, group]: human, then the current tags (CURRENT), then every other sh30-family checkpoint on
    disk (CKPTS/<tag>/ckpt-final.pt)."""
    tags = sorted(p.parent.name for p in CKPTS.glob("*/ckpt-final.pt") if not NOT_DRIVERS.match(p.parent.name))
    return ([["human", "Human (phone)", ""]] + [[t, "%s  (%s)" % (t, n), "current"] for t, n in CURRENT.items() if t in tags] +
            [[t, t, "other checkpoints"] for t in tags if t not in CURRENT])


def driver_of(d):
    return (d or {}).get("driver") or "human"


def model_drives(run, sid):
    """{tag: done record} of the model drives of a scene in a run (done/<scene>@<tag>.json, scripts/human_loop.py)."""
    return {driver_of(d): d for d in (read_json(p) for p in sorted(Path(run, "done").glob(sid + "@*.json"))) if d}
ENV = "export DISPLAY=:0 XAUTHORITY=/run/user/1001/gdm/Xauthority DATA_DIR=%s" % D


def sh(cmd):
    return subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


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


R6DAG_TSV = REPO / "experiments/alpasim_human/results/model_ref_r6dag_281.tsv"
ALL_TSV = A / "fetch_all/all441.tsv"
NUREC = Path(os.environ.get("NUREC", D / "datasets/nurec"))
# R6DAG-s1 loss classes (round7 loss map, decision 436) -> one "R6DAG fails" set each: (key, label, classes)
FAIL_SETS = [("turnin", "offroad, turn inside", {"off-turn-in"}),
             ("offother", "offroad, straight / junction / outside", {"off-straight", "off-junc-straight", "off-turn-out"}),
             ("overshoot", "stop-line overshoot", {"overshoot"}), ("drift", "drift / lane change", {"drift"}),
             ("slow", "unforced slow-down partial", {"slow", "near1"}), ("rearmove", "rear-ended while moving", {"rear-move"}),
             ("rearlaunch", "slow launch, rear-ended", {"rear-launch"}), ("coll", "at-fault collision", {"coll-front", "coll-side"}),
             ("routeturn", "route turn missed", {"route-turn"})]


def tsv(path):
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()] if Path(path).exists() else []
    return [dict(zip(rows[0], r)) for r in rows[1:]]


class Scenes(object):
    """Scene sets, groups and reference scores of both tracks (shared by the panel backend and the loop).

    PAI scenes = the 441 curated-validation scenes of fetch_all/all441.tsv whose NuRec artifact ($NUREC/all-usdzs/<uuid>.usdz)
    is on disk (all are in sim_scenes.csv / sim_scenes_2604.csv, which human_stack.sh passes), else the 40 of pai1. A set is a
    filter over that table (`PRED`); its label carries the scene count. R6DAG = R6DAG-L18-s1, PAI 281 ACC + GAP read, per-scene
    score and loss class from experiments/round7/results/loss_map.csv (decision 436), copied to results/model_ref_r6dag_281.tsv."""

    def __init__(self):
        self.meta, self.refs = {}, {}
        in40 = {r["scene_id"] for r in tsv(PAI_TSV)}
        rows = tsv(ALL_TSV) or tsv(PAI_TSV)
        self.pai = [r for r in rows if (NUREC / "all-usdzs" / (r["uuid"] + ".usdz")).is_file()]
        r6 = {r["scene_id"]: r for r in tsv(R6DAG_TSV)}
        for k, r in enumerate(self.pai):
            m = r6.get(r["scene_id"], {})
            self.meta[r["scene_id"]] = dict(track="pai", group="PAI %s" % r["revision"], stage=int(r["stage"]), k=k,
                                            label=r["scene_id"][-12:], rev=r["revision"], ref=float(r["ref_mean"]), in40=r["scene_id"] in in40,
                                            r6=float(m["score"]) if m else None, cat=m.get("category"),
                                            scenario="stage %s, organisers %.2f%s" % (r["stage"], float(r["ref_mean"]),
                                                                                   ", R6DAG: %s" % m["category"] if m and m["category"] != "ok" else ""))
        self.nuplan = NUPLAN_LIST.read_text().split() if NUPLAN_LIST.exists() else []
        for k, s in enumerate(self.nuplan):
            seg = s.split("_")[2:4] if s.count("_") >= 3 else ["?", "?"]     # <date>_<vehicle>_<start>_<end>-<token>
            self.meta[s] = dict(track="nuplan", group="nuPlan log %s_%s" % (seg[0], seg[1].split("-")[0]), k=k,
                                label="%s_%s" % (seg[0], seg[1][:14]),
                                scenario="navtest %s" % s.split("_")[0])
        self.refs[("pai", "R6DAG")] = {i: float(r["score"]) for i, r in r6.items()}
        for track in ("pai", "nuplan"):
            for name, src in MODELS[track] + EXTRA[track]:
                if str(src).startswith("tsv:"):
                    col = str(src)[4:]
                    self.refs[(track, name)] = {r["scene_id"]: float(r[col]) for r in self.pai if r.get(col) not in (None, "")}
                elif (track, name) not in self.refs:
                    self.refs[(track, name)] = summary_scores(Path(src))
        pai = lambda f: ("pai", f)  # noqa: E731
        preds = [("pai1", "PAI stage 1", pai(lambda m: m["stage"] <= 1)), ("pai40", "PAI first 40", pai(lambda m: m["in40"])),
                 ("pai441", "PAI all", pai(lambda m: True))]
        preds += [("pai" + v, "PAI %s" % v, pai(lambda m, v=v: m["rev"] == v)) for v in sorted({r["revision"] for r in self.pai})]
        preds += [("hard", "PAI hard (organisers' mean <= 0.2)", pai(lambda m: m["ref"] <= 0.2)),
                  ("easy", "PAI easy (organisers' mean >= 0.8)", pai(lambda m: m["ref"] >= 0.8)),
                  ("r6fail", "R6DAG fails on 281 (score < 1)", pai(lambda m: m["r6"] is not None and m["r6"] < 1))]
        preds += [("r6_" + k, "R6DAG fails: " + t, pai(lambda m, c=c: m["cat"] in c)) for k, t, c in FAIL_SETS]
        preds += [("nuplan48", "nuPlan navtest", ("nuplan", lambda m: True))]
        self.pred = {k: p for k, _, p in preds}
        self.sets = [(k, "%s (%d)" % (t, len(self.ids(k)))) for k, t, _ in preds]
        self.sets = [s for s in self.sets if not s[1].endswith("(0)")]

    def ids(self, key):
        track, f = self.pred[key]
        ids = [i for i, m in self.meta.items() if m["track"] == track and f(m)]
        return sorted(ids, key=lambda i: (self.meta[i]["group"], self.meta[i].get("stage", 0), self.meta[i]["k"]))

    def model(self, scene):
        """(score, source) of the best available model read on this scene: R6DAG-s1 (PAI 281), then P2H10, else the organisers' alpamayo1."""
        track = self.meta.get(scene, {}).get("track", "pai")
        for name, _ in ([("R6DAG", None)] if track == "pai" else []) + MODELS[track] + EXTRA[track][:1]:
            v = self.refs.get((track, name), {}).get(scene)
            if v is not None:
                return v, ("a1" if name == "alpamayo1" else name)
        return None, None

    def catalog(self):
        """Every selectable scene (custom pick), in set order: dict(id, track ("pai" / "nuplan"), group, label, revision (PAI
        "26.01" / "26.04", else None), stage, ref_mean (organisers' mean, PAI), r6dag (R6DAG-s1 281 score or None), fail (R6DAG
        loss class, "" = none / not read), model, model_src (the panel's model column))."""
        out = []
        for i in self.ids("pai441") + self.ids("nuplan48"):
            m = self.meta[i]
            ms, src = self.model(i)
            out.append(dict(id=i, track=m["track"], group=m["group"], label=m["label"], revision=m.get("rev"), stage=m.get("stage"),
                            ref_mean=m.get("ref"), r6dag=m.get("r6"), fail=m.get("cat") if m.get("cat") not in (None, "ok") else "",
                            model=ms, model_src=src))
        return out

    def table(self, run):
        """results.md of a panel run: one row per driven scene in run order, means over the driven scenes."""
        run = Path(run)
        ex = read_json(run / "exam.json", {}) or {}
        track = ex.get("track", "pai")
        cols = (["R6DAG"] if track == "pai" else []) + [n for n, _ in MODELS[track] + EXTRA[track]]
        rows = []
        for sid in ex.get("ids", []):
            d = read_json(run / "done" / (sid + ".json"))
            if d:
                rows.append((sid, d, [self.refs.get((track, n), {}).get(sid) for n in cols]))
        L = ["%s, set %s: %d of %d scenes driven (latest finished attempt per scene; skipped attempts do not count)" % (
            track, ex.get("set"), len(rows), len(ex.get("ids", []))), "",
            "| scene | group | human | why | progress | driven / GT m | wall per sim s | stale | attempts | harmonizer | shadow s | " +
            " | ".join(cols) + " |", "|---|---|--:|---|--:|--:|--:|--:|--:|---|--:|" + "--:|" * len(cols)]
        for sid, d, refs in rows:
            L.append("| %s | %s | %.3f | %s | %s | %s / %s | %s | %s | %s | %s | %.0f | %s |" % (
                self.meta.get(sid, {}).get("label", sid), self.meta.get(sid, {}).get("group", "?"), d["score"], d["why"],
                d.get("progress"), d.get("driven_m"), d.get("gt_m"), d.get("wall_per_sim"), d.get("stale"), d.get("attempt", 1),
                "on" if d.get("harmonizer") else "off", d.get("shadow_s") or 0,
                " | ".join("-" if v is None else "%.3f" % v for v in refs)))
        if rows:
            mean = lambda v: sum(v) / len(v)  # noqa: E731
            parts = ["human %.3f" % mean([d["score"] for _, d, _ in rows])]
            for j, n in enumerate(cols):
                v = [r[j] for _, _, r in rows if r[j] is not None]
                parts.append("%s %s%s" % (n, "%.3f" % mean(v) if v else "-", "" if len(v) == len(rows) else " (%d scenes)" % len(v)))
            L += ["", "Means over the %d driven scenes: %s." % (len(rows), "; ".join(parts))]
        M = [(sid, d) for sid in ex.get("ids", []) for d in model_drives(run, sid).values()]
        if M:                                                    # model mode: never mixed into the human rows or means
            L += ["", "Model drives (driver option, README \"Model mode\"; real-time paced, front camera only: for watching, not leaderboard "
                  "numbers):", "", "| scene | driver | score | why | progress | driven / GT m | attempts | harmonizer |", "|---|---|--:|---|--:|--:|--:|---|"]
            for sid, d in M:
                L.append("| %s | %s | %.3f | %s | %s | %s / %s | %s | %s |" % (self.meta.get(sid, {}).get("label", sid), driver_of(d), d["score"],
                         d["why"], d.get("progress"), d.get("driven_m"), d.get("gt_m"), d.get("attempt", 1), "on" if d.get("harmonizer") else "off"))
            L.append("")
            for tag in sorted({driver_of(d) for _, d in M}):
                v = [d["score"] for _, d in M if driver_of(d) == tag]
                L.append("%s: mean %.3f over %d scenes." % (tag, sum(v) / len(v), len(v)))
        L += ["", "Fast mode (front camera only): a human experience, not a leaderboard number (README, section Speed)."]
        return "\n".join(L) + "\n"


class AlpaSimExam(Backend):
    title = "AlpaSim human exam"
    live = os.environ.get("HUMAN_LIVE", str(RUNS / "live"))
    score_fmt = "%.2f"
    options = {"harmonizer": ("Harmonizer (PAI, slower render)", False),   # human_loop.py: new stack at the next hold
               "render": ("Render size", "auto", ["auto", "1080", "720", "540", "288"]),  # front camera height px (HUMAN_H); auto =
               "shadow": ("Shadow: human log", False)}                         # human_stack.sh's pairing; the viewer only (display; G there too)
    batch_extra = "harmonizer"
    record = True                                                     # human_loop.py honours control "record" (practice)
    pinned = ("shadow",)                                              # the user wants Shadow at hand, not in the drawer (G in the viewer too)
    option_help = {
        "driver": ("Who drives: you with the phone, or a model checkpoint that drives by itself while you watch. Model scores are kept "
                   "apart from yours.", "From the next track. On PAI, switching between you and a model restarts the render stack once "
                   "(about 1 min)."),
        "harmonizer": ("PAI only: makes the rendered camera look more like a real photo. Frames come slower.",
                       "Restarts the render stack (about 1 min) before the next track."),
        "render": ("Height of the front camera picture in pixels. Smaller is faster; auto = the default for the Harmonizer setting.",
                   "Restarts the render stack (about 1 min) before the next track."),
        "shadow": ("Shows where the recorded human driver went, as a ghost path in the viewer.", "Viewer only; applies at once."),
    }

    def __init__(self):
        self.sc = Scenes()
        self.options = dict(type(self).options, driver=("Driver", "human", driver_choices()))   # human_loop.py: at the next hold point

    def batch_sets(self):
        return list(self.sc.sets)

    def catalog(self):
        return self.sc.catalog()

    # ---- runs and the loop ----
    def runs(self):
        return [str(p.parent) for p in sorted(RUNS.glob("exam_*/exam.json"), key=lambda p: -p.stat().st_mtime)]

    def batch_info(self, run):
        ex = read_json(Path(run, "exam.json"), {}) or {}
        ids = ex.get("ids", [])
        done = [d for d in (read_json(Path(run, "done", i + ".json")) for i in ids) if d]
        h = [bool(d.get("harmonizer")) for d in done]
        tags = sorted({p.name.split("@", 1)[1][:-5] for p in Path(run, "done").glob("*@*.json")})
        hx = "-" if not h else "on" if all(h) else "off" if not any(h) else "mixed %d/%d" % (sum(h), len(h))
        return dict(label=ex.get("label") or ex.get("set", "?"), created=(ex.get("created") or "?")[5:16], driven=len(done), total=len(ids),
                    mean=sum(d["score"] for d in done) / len(done) if done else None, extra=hx,
                    badges=["harmonizer %s" % hx] * bool(h) + ["model: %s" % t for t in tags] + [ex.get("track", "pai").upper()])

    def item_label(self, item):
        return self.sc.meta.get(item, {}).get("label", item)

    def delete_batch(self, run):
        ok, msg = Backend.delete_batch(self, run)
        if not ok and Path(run).parent == RUNS and Path(run).name.startswith("exam_"):      # root-owned leftovers of the stack
            base = sh("docker images --format '{{.Repository}}:{{.Tag}}' | grep '^alpasim-base:' | head -1").strip()
            sh("docker run --rm -v %s:%s %s rm -rf %s" % (RUNS, RUNS, base, run)) if base else None
            ok = not Path(run).exists()
            msg = "deleted %s" % run if ok else msg
        return ok, msg

    def default_run(self):
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
            return "The panel loop is already running on %s." % self.default_run()
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

    def new_batch(self, key, ids=None, name=None, start=True):
        """A run dir for set `key`, or for the picked `ids` (key "custom"; one track), labelled `name` if given; then start the loop on it
        (`start`)."""
        if ids is not None:
            ids = list(dict.fromkeys(ids))
            bad = [i for i in ids if i not in self.sc.meta]
            tracks = {self.sc.meta[i]["track"] for i in ids if i in self.sc.meta}
            if bad or len(tracks) != 1:
                return False, ("Unknown scenes: %s" % ", ".join(bad[:5])) if bad else "Pick scenes of one track (PAI or nuPlan)."
            track, label = tracks.pop(), "Custom pick (%d)" % len(ids)
        else:
            ids, track, label = self.sc.ids(key), self.sc.pred[key][0], dict(self.sc.sets)[key]
        label = (name or "").strip()[:80] or label
        run = RUNS / ("exam_%s_%s" % (track, time.strftime("%m%d-%H%M%S")))
        while run.exists():                                                   # two batches within one second
            run = run.with_name(run.name + "b")
        run.mkdir(parents=True)
        (run / "exam.json").write_text(json.dumps({"track": track, "set": key, "label": label, "ids": list(ids),
                                                   "created": time.strftime("%F %T")}, indent=1))
        (run / "notes.txt").write_text("%s panel: new batch %s (%d scenes)\n" % (time.strftime("%F %T"), key, len(ids)))
        return self.start_loop(run) if start else (True, "Playlist \"%s\" created (%d scenes)." % (label, len(ids)))

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
                status = "%s, progress %s%s%s%s" % (d["why"], d.get("progress"), ", attempt %d" % d["attempt"] if d.get("attempt", 1) > 1 else "",
                                                  ", harmonized" if d.get("harmonizer") else "",
                                                  ", shadow on %.0f s" % d["shadow_s"] if d.get("shadow_s") else "")
            elif t:
                status = t.get("status", "not finished")
            else:
                status = ""
            md = model_drives(run, sid)
            if md:                                               # model drives, apart from the human score
                status = "; ".join([status] * bool(status) + ["model %s %.2f %s" % (k, v["score"], v["why"]) for k, v in md.items()])
            ms, src = self.sc.model(sid)
            out.append(dict(id=sid, label=m.get("label", sid), group=m.get("group", "?"), scenario=m.get("scenario", ""),
                            human=d["score"] if d else None, model=ms, model_src=src, status=status,
                            why=d["why"] if d else (t or {}).get("status", ""), attempts=d.get("attempt", 1) if d else 0,
                            sim_s=d.get("sim_s") if d else None, ref=m.get("ref"),
                            drives=[dict(driver=k, score=v["score"], why=v["why"]) for k, v in md.items()]))
        return out

    def official(self):
        run = self.run_dir()
        return self.sc.table(run) if run else "no run"


def stack_name(out):
    """Compose project of a human_stack.sh out dir (as the script derives it)."""
    return "hum-" + re.sub(r"[^a-zA-Z0-9]", "-", Path(out).name).lower()
