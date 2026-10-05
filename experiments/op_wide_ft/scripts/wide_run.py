#!/usr/bin/env python
"""op_wide_ft one-shot lane (plans/2026-10-05-wide-ft-prereg.md): runs in tmux (`scripts/tmux_run.sh wf-lane .venv/bin/python <this> stage1`),
submits every GPU step to the pool and waits on it; STATUS / DONE / ERROR in $DATA_DIR/runs/op_wide_ft/lane/<stage>/. Resumable: a step whose
DONE marker (chain/<phase>/DONE, run dir DONE, unit done files) exists is skipped.

  stage1   wait for the render jobs -> pack (CPU, here) -> bank -> pilot (W116 400 steps, pre-registered checks) -> full training W58 + W116 ->
           ONNX + equivalence -> open-loop 2 x 2 (wide_ol.py) and evalol -> B2D small set (9 turns) -> gate (wide_report.py small) and STOP
  stage2   (after a passed gate) B2D 25 turns x {W58@58, W116@116, W116@58} -> report turns; comma1M + PhysicalAI readouts are separate pool jobs
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.cl import pool as P  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

W = data_dir() / "runs/op_wide_ft"
CH = "bash experiments/op_wide_ft/scripts/wide_chain.sh"
ENV = {"OP_RFT_ROOT": str(W)}
PYT = str(data_dir() / "envs/op-train/bin/python")


class Lane:
    def __init__(self, stage):
        self.d = W / "lane" / stage
        self.d.mkdir(parents=True, exist_ok=True)
        for f in ("DONE", "ERROR"):
            (self.d / f).unlink(missing_ok=True)

    def say(self, msg):
        line = "%s %s" % (time.strftime("%F %T"), msg)
        print(line, flush=True)
        (self.d / "STATUS").write_text(line + "\n")
        with open(self.d / "log.txt", "a") as f:
            f.write(line + "\n")

    def die(self, msg):
        self.say("ERROR " + msg)
        (self.d / "ERROR").write_text(msg + "\n")
        sys.exit(1)

    def job(self, name, cmd, done=None, **kw):
        """Submit (unless `done` exists) and wait; die on failure."""
        if done is not None and Path(done).exists():
            self.say(f"{name}: already done ({done})")
            return
        kw.setdefault("env", dict(ENV))
        jid = P.submit(cmd, name=name, log_dir=str(W / "chain/pool" / name), **kw)
        self.say(f"{name}: pool job {jid}")
        st = P.wait([jid], poll_s=30)
        if st[jid] != "done":
            self.die(f"{name}: pool job {jid} {st[jid]} (log {W / 'chain/pool' / name / 'log.txt'})")

    def jobs(self, specs):
        """Several jobs in parallel: [(name, cmd, done, kw)]."""
        ids = {}
        for name, cmd, done, kw in specs:
            if done is not None and Path(done).exists():
                continue
            kw.setdefault("env", dict(ENV))
            ids[P.submit(cmd, name=name, log_dir=str(W / "chain/pool" / name), **kw)] = name
        if ids:
            self.say("waiting for %s" % ", ".join(ids.values()))
            st = P.wait(list(ids), poll_s=30)
            bad = [ids[i] for i, s in st.items() if s != "done"]
            if bad:
                self.die("failed: %s" % bad)


def losses(run):
    """{step: {loss: value}} from a run's log.txt ('step N: imit x, act y, ...')."""
    out = {}
    for line in open(W / "runs" / run / "log.txt"):
        m = re.search(r"step (\d+): (.*?);", line)
        if m:
            out[int(m.group(1))] = {k: float(v) for k, v in (x.strip().split(" ") for x in m.group(2).split(","))}
    return out


def stage1():
    L = Lane("stage1")
    R = W / "carla/render"
    L.say("waiting for the 3 render pool jobs (chain/pool/render{0,1,2})")
    jobs = [W / "chain/pool" / f"render{j}" for j in range(3)]
    while not all((j / "DONE").exists() for j in jobs):
        bad = [j.name for j in jobs if (j / "ERROR").exists()]
        if bad:
            L.die("render job failed: %s (see %s)" % (bad, R))
        time.sleep(60)
    if not (W / "chain/pack/DONE").exists():
        L.say("pack (CPU)")
        if subprocess.call(["bash", "experiments/op_wide_ft/scripts/wide_chain.sh", "pack"], cwd=REPO) != 0:
            L.die("pack (see %s)" % (W / "chain/pack/log.txt"))
    L.job("wf-bank", f"{CH} bank", W / "bank/w116/teacher.npz", vram_gb=20, cpu=8)
    # pilot: W116 400 steps; checks of the pre-registration (step 0)
    L.job("wf-pilot", f"TAG=pilot {CH} train wf-w116 400 pilot-wf-w116 && OP_RFT_ROOT={W} {PYT} experiments/op_route_ft/scripts/rft.py evalol --models O pilot-wf-w116 --carla w116",
          W / "evalol_w116/pilot-wf-w116.json", vram_gb=18, cpu=12, train=True)
    ls = losses("pilot-wf-w116")
    o, p = (json.load(open(W / f"evalol_w116/{m}.json")) for m in ("O", "pilot-wf-w116"))
    drift = max(v["median"] for k, v in p.items() if k.startswith("drift_") and v.get("median") is not None)
    chk = dict(loss_down=all(ls[400][k] < ls[100][k] for k in ("imit", "act")), exit=p["carla_exit_row"]["mean"], exit_O=o["carla_exit_row"]["mean"],
               exit_ok=p["carla_exit_row"]["mean"] >= o["carla_exit_row"]["mean"] + 0.10, drift_max_median=drift, drift_ok=drift <= 0.15,
               losses={k: ls[k] for k in (100, 400)})
    (W / "lane/pilot.json").write_text(json.dumps(chk, indent=1))
    L.say("pilot checks %s" % json.dumps({k: v for k, v in chk.items() if k != "losses"}))
    if not (chk["loss_down"] and chk["exit_ok"] and chk["drift_ok"]):
        L.die("pilot check failed: %s" % chk)
    # full training, both arms in parallel
    L.jobs([(f"wf-train-{a}", f"TAG={a} {CH} train {a} 0 {a}-s0", W / f"runs/{a}-s0/DONE", dict(vram_gb=18, cpu=12, train=True)) for a in ("wf-w58", "wf-w116")])
    L.jobs([(f"wf-onnx-{a}", f"TAG={a} {CH} onnx {a}-s0", W / f"chain/onnx-{a}/DONE", dict(vram_gb=12, cpu=6)) for a in ("wf-w58", "wf-w116")] +
           [("wf-ol2x2", f"OP_RFT_ROOT={W} {PYT} experiments/op_wide_ft/scripts/wide_ol.py --models O wf-w58-s0 wf-w116-s0",
             REPO / "experiments/op_wide_ft/results/ol2x2_gate.json", dict(vram_gb=18, cpu=8)),
            ("wf-evalol", f"TAG=full {CH} evalol O wf-w58-s0 wf-w116-s0", W / "chain/evalol-full/DONE", dict(vram_gb=18, cpu=8))])
    L.say("B2D small set")
    if subprocess.call([sys.executable, "experiments/op_wide_ft/scripts/wide_lane.py", "--routes", "small", "--arms", "wf-w58-s0@58,wf-w116-s0@116",
                        "--wait"], cwd=REPO) != 0:
        L.die("B2D small-set units (see %s)" % (W / "cl"))
    if subprocess.call([sys.executable, "experiments/op_wide_ft/scripts/wide_report.py", "small", "--ol",
                        str(REPO / "experiments/op_wide_ft/results/ol2x2_gate.json")], cwd=REPO) != 0:
        L.die("report small")
    g = json.load(open(REPO / "experiments/op_wide_ft/results/gate.json"))
    L.say("GATE %s" % json.dumps(g["gate"]))
    (L.d / "DONE").write_text(json.dumps(g) + "\n")


def stage2():
    L = Lane("stage2")
    if subprocess.call([sys.executable, "experiments/op_wide_ft/scripts/wide_lane.py", "--routes", "all", "--arms",
                        "wf-w58-s0@58,wf-w116-s0@116,wf-w116-s0@58", "--wait"], cwd=REPO) != 0:
        L.die("B2D 25-turn units (see %s)" % (W / "cl"))
    if subprocess.call([sys.executable, "experiments/op_wide_ft/scripts/wide_report.py", "turns"], cwd=REPO) != 0:
        L.die("report turns")
    (L.d / "DONE").write_text("ok\n")


if __name__ == "__main__":
    os.chdir(REPO)
    {"stage1": stage1, "stage2": stage2}[sys.argv[1]]()
