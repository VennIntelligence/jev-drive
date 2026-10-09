#!/usr/bin/env python3
"""Lane OT2 closed-loop chain: drivers on the fixed 700-scene public list (c0b/lists, decision 201) as GPU-pool jobs, with c0b_chain.py's
stall watchdog, fill-in of missing scenes and pruning (rollout.asl kept only for failed rollouts).

  scripts/tmux_run.sh ot2-loop-<stage> python3 experiments/alpasim/scripts/ot2_loop.py <stage> <job> [<job> ...]     (box; stdlib only)

job = label:driver:ENV=VALUE[,ENV=VALUE...][:checkpoint tag[+tag...]][:list]
  label        the driver's name in the manifest (e.g. APO-a15m25-s0)
  driver       run.sh driver: sh30 | ap2 | ens
  checkpoint   tags under $DATA_DIR/runs/op_parity/runs whose ckpt-final.pt must exist before the job is submitted (a training that is still
               running); the chain fails when one has not appeared after OT2_WAIT_H hours (default 12)
  list         `chunks` (default: the three interleaved thirds of the 700 scenes, one pool job each) or a list name under c0b/lists (pilot8)
State in $DATA_DIR/runs/alpasim/ot2/<stage>/: STATUS, DONE, ERROR, log.txt, manifest.json {label: [run dirs]}, runs/. At most OT2_MAX_ACTIVE
(3) of this chain's jobs are in the pool at a time (nine AlpaSim stacks at once exhausted host memory on 2026-10-09); declarations are the
measured peaks of the c0b / m1 runs (VRAM 32 GB, RAM 42 GB, 8 cores). Another lane reuses the chain with OT_LANE=<dir under runs/alpasim>
(default ot2; also the pool owner `alpasim-<lane>`) and OT_PRIO (pool priority, default 12). A lane of another topic also sets OT_OUT (the state
directory that replaces $DATA_DIR/runs/alpasim/<lane>), OT_OWNER (the pool owner) and OT_RAM (declared host RAM per job in GiB, instead of
14 + 0.12 per scene).
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c0b_chain as B  # noqa: E402

LANE, PRIO = os.environ.get("OT_LANE", "ot2"), int(os.environ.get("OT_PRIO", 12))
LISTS = B.RA / "c0b/lists"
CK = B.DATA / "runs/op_parity/runs"


class Job(B.Job):
    def __init__(self, label, name, drv, env, scenes, need):
        super().__init__(name, drv, env, scenes, PRIO, 32)
        self.label, self.need, self.t0 = label, need, time.time()

    def ready(self):
        miss = [t for t in self.need if not (CK / t / "ckpt-final.pt").exists()]
        if miss and time.time() - self.t0 > 3600 * float(os.environ.get("OT2_WAIT_H", 12)):
            B.fail(f"{self.name}: checkpoint {miss} did not appear")
        return not miss

    def submit(self):
        self.tries += 1
        if self.tries > B.MAX_TRIES:
            B.fail(f"{self.name}: {B.MAX_TRIES} tries exhausted")
        self.D = B.O / "runs" / self.name / time.strftime("%Y%m%d-%H%M%S")
        self.D.mkdir(parents=True)
        cmd = B.PY + ["submit", "--owner", os.environ.get("OT_OWNER") or f"alpasim-{LANE}", "--name", f"alpasim-{LANE}-{self.drv}", "--vram", str(self.vram), "--cpu", "8",
                      "--ram", os.environ.get("OT_RAM") or str(int(14 + 0.12 * self.n)), "--timeout-h", "3", "--tries", "1", "--priority", str(self.prio), "--log-dir", str(self.D / "pool"),
                      "--", "env", *[f"{k}={v}" for k, v in self.env.items()], "bash", "experiments/alpasim/scripts/run.sh", str(self.D), self.drv,
                      "--scene-list", str(self.scenes), *B.OVERRIDES]
        out = subprocess.run(cmd, cwd=B.REPO, capture_output=True, text=True)
        if out.returncode:
            B.fail(f"submit {self.name}: {out.stderr[-500:]}")
        self.id = out.stdout.split()[-1]
        self.state, self.t_seen, self.t_prog, self.last_n = "active", None, None, 0
        B.log(f"{self.name}: submitted {self.id} try {self.tries} -> {self.D}")

    def finish(self):
        try:
            R = B.load_scores(self.D)
        except Exception as e:
            B.log(f"{self.name}: no results-summary ({e!r})")
            self.submit()
            return False
        missing = sorted(set(self.scenes.read_text().split()) - set(R))
        for sid, r in R.items():
            if not r.get("failure_reason"):
                for f in glob.glob(str(self.D / "rollouts" / sid / "*" / "rollout.asl")):
                    os.remove(f)
        self.dirs.append(self.D)
        if missing and self.tries < B.MAX_TRIES:
            B.log(f"{self.name}: {len(missing)} of {self.n} scenes missing, resubmitting those")
            fill = B.O / "lists" / f"{self.name}.fill{self.tries}.txt"
            fill.write_text("\n".join(missing) + "\n")
            self.scenes, self.n = fill, len(missing)
            self.submit()
            return False
        if missing:
            B.fail(f"{self.name}: {len(missing)} scenes still missing after {self.tries} tries")
        s, d, ie, ip = B.driver_health(self.dirs)
        if ie or ip:
            B.fail(f"{self.name}: driver errors (inference {ie}, input {ip}) in {self.dirs}")
        self.state = "done"
        B.log(f"{self.name}: done, {len(R)} rollouts, {s} sessions, {d} drive calls in {self.D}")
        return True


def main():
    stage, specs = sys.argv[1], sys.argv[2:]
    B.O = (Path(os.environ["OT_OUT"]) if os.environ.get("OT_OUT") else B.RA / LANE) / stage
    (B.O / "lists").mkdir(parents=True, exist_ok=True)
    for f in ("DONE", "ERROR"):
        (B.O / f).unlink(missing_ok=True)
    allsc = set((LISTS / "all.txt").read_text().split())
    assert len(allsc) == 700 and set().union(*(set((LISTS / f"chunk{i}.txt").read_text().split()) for i in range(3))) == allsc, "c0b lists changed"
    man_f = B.O / "manifest.json"
    man = json.loads(man_f.read_text()) if man_f.exists() else {}
    jobs = []
    for sp in specs:
        label, drv, env, *rest = sp.split(":")
        need = [t for t in (rest[0].split("+") if rest and rest[0] else []) if t]
        lst = rest[1] if len(rest) > 1 else "chunks"
        if label in man:                                            # a finished driver of an earlier start of this stage
            B.log(f"{label}: already in the manifest, skipped")
            continue
        for part in (["chunk0", "chunk1", "chunk2"] if lst == "chunks" else [lst]):
            jobs.append(Job(label, f"{label}-{part}", drv, dict(e.split("=", 1) for e in env.split(",") if e), LISTS / f"{part}.txt", need))
    cap = int(os.environ.get("OT2_MAX_ACTIVE", 3))
    while True:
        for j in jobs:
            if j.state == "new" and sum(x.state == "active" for x in jobs) < cap and j.ready():
                j.submit()
        done = [j.poll() for j in jobs]
        for label in dict.fromkeys(j.label for j in jobs):          # publish a driver as soon as all of its chunks are in
            mine = [j for j in jobs if j.label == label]
            if label not in man and all(j.state == "done" for j in mine):
                man[label] = [str(d) for j in mine for d in j.dirs]
                man_f.write_text(json.dumps(man, indent=1))
        fg = B.free_gb()
        B.status(f"{stage}: " + ", ".join(f"{j.name} {j.n_done() if j.state == 'active' and j.D else (j.n if j.state == 'done' else 0)}/{j.n}"
                                          + (" ok" if j.state == "done" else " wait" if j.state == "new" else "") + (f" t{j.tries}" if j.tries > 1 else "")
                                          for j in jobs if j.state != "done" or True) + f"; free {fg:.0f} GB")
        if fg < 150:
            B.fail(f"free disk {fg:.0f} GB < 150 GB; running jobs left alone, nothing new submitted")
        if all(done):
            break
        time.sleep(B.POLL_S)
    (B.O / "DONE").write_text(time.strftime("%F %T") + "\n")
    B.status(f"{stage}: all done")


if __name__ == "__main__":
    main()
