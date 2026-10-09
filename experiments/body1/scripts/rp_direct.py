#!/usr/bin/env python3
"""BODY1 closed-loop chain on the lane's whole-held card, outside the pool (the pool cannot place a job on a held card; when its other cards
are full of another lane's queue this is the lane's only card). Same job specs, run command, overrides, fill-in of missing scenes, pruning
and state files as experiments/alpasim/scripts/ot2_loop.py; the jobs are started here as plain processes instead of `cl submit`.

  scripts/tmux_run.sh body1-<stage> env RP_CARD=2 python3 experiments/body1/scripts/rp_direct.py <stage> <job> [<job> ...]     (box; stdlib only)

job = label:driver:ENV=VALUE[,ENV=VALUE...]::list      (list = chunks | a list name under c0b/lists, as ot2_loop.py | an absolute path; no checkpoint waiting)
Env: RP_CARD (card index, default 2), RP_MAX (jobs at once, default 2: two stacks of <= 32 GB on an 84 GB card), RP_KEEP=1 (keep every
rollout.asl: review strips), OT_OUT (state root, default $DATA_DIR/runs/body1/cl).
State in <OT_OUT>/<stage>/: STATUS, DONE, ERROR, log.txt, manifest.json {label: [run dirs]}, runs/. Before every start the page cache is trimmed to
a 150 GiB margin under the platform's kill line (`cl trim`), the check the pool makes for its own jobs.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/alpasim/scripts"))
import c0b_chain as B  # noqa: E402

CARD, MAXA, KEEP = os.environ.get("RP_CARD", "2"), int(os.environ.get("RP_MAX", 2)), os.environ.get("RP_KEEP") == "1"
LISTS = B.RA / "c0b/lists"
TIMEOUT_S = 3600


ACTIVE = []


def free_cores(n):
    """The n highest cores that no running pool job and no job of this chain is pinned to (a pool job runs under taskset; an unpinned AlpaSim
    stack sizes its thread pools from all 208 host threads and exhausts the container's pid limit: pilot of 2026-10-10)."""
    used = set()
    try:
        for j in json.loads((B.DATA / "runs/pool/state.json").read_text())["jobs"].values():
            if j.get("state") == "running":
                for part in filter(None, str(j.get("cpus") or "").split(",")):
                    a, _, b = part.partition("-")
                    used |= set(range(int(a), int(b or a) + 1))
    except Exception as e:
        B.log(f"pool state unreadable ({e!r}): cores chosen without it")
    for j in ACTIVE:
        if j.state == "active":
            used |= set(j.cpus)
    return sorted(sorted(set(os.sched_getaffinity(0)) - used)[-n:])


class Job:
    def __init__(self, label, name, drv, env, scenes):
        self.label, self.name, self.drv, self.env, self.scenes = label, name, drv, env, scenes
        self.n, self.state, self.tries, self.dirs, self.p, self.D = len(scenes.read_text().split()), "new", 0, [], None, None
        self.cpus = []

    def start(self):
        self.tries += 1
        if self.tries > B.MAX_TRIES:
            B.fail(f"{self.name}: {B.MAX_TRIES} tries exhausted")
        subprocess.run(B.PY + ["trim", "--margin", "150"], cwd=REPO, capture_output=True)
        self.D = B.O / "runs" / self.name / time.strftime("%Y%m%d-%H%M%S")
        (self.D / "pool").mkdir(parents=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=CARD, **self.env)
        self.cpus = free_cores(8)
        self.p = subprocess.Popen(["taskset", "-c", ",".join(map(str, self.cpus)), "bash", "experiments/alpasim/scripts/run.sh", str(self.D), self.drv, "--scene-list", str(self.scenes), *B.OVERRIDES], cwd=REPO, env=env,
                                  stdout=open(self.D / "pool/log.txt", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        self.state, self.t0 = "active", time.time()
        B.log(f"{self.name}: started pid {self.p.pid} on card {CARD} cores {self.cpus[0]}-{self.cpus[-1]} try {self.tries} -> {self.D}")

    def n_done(self):
        return sum(1 for _ in glob.iglob(str(self.D / "rollouts/*/*/_complete"))) if self.D else 0

    def poll(self):
        if self.state != "active":
            return self.state == "done"
        rc = self.p.poll()
        if rc is None:
            if time.time() - self.t0 > TIMEOUT_S:
                B.log(f"{self.name}: no end after {TIMEOUT_S} s, terminating its process group")
                os.killpg(self.p.pid, 15)
                self.p.wait()
                self.start()
            return False
        try:
            R = B.load_scores(self.D)
        except Exception as e:
            B.log(f"{self.name}: rc {rc}, no results-summary ({e!r})")
            self.start()
            return False
        bad = [sid for sid, r in R.items() if (r.get("score_metrics") or {}).get("progress_clipped_rel") is None and "Evaluation failed" in (r.get("failure_reason") or "")]
        if bad:                                                        # the scorer died (not a driving failure): the scene is run again, this run dir is not kept for it
            B.log(f"{self.name}: {len(bad)} rollouts without an evaluation, treated as missing")
            d = json.loads((self.D / "aggregate/results-summary.json").read_text())
            (self.D / "aggregate/results-summary.with-failed-eval.json").write_text(json.dumps(d))
            d["rollouts"] = [r for r in d["rollouts"] if r["clipgt_id"] not in bad]
            (self.D / "aggregate/results-summary.json").write_text(json.dumps(d))
            R = {k: v for k, v in R.items() if k not in bad}
        missing = sorted(set(self.scenes.read_text().split()) - set(R))
        if not KEEP:
            for sid, r in R.items():
                if not r.get("failure_reason"):
                    for f in glob.glob(str(self.D / "rollouts" / sid / "*" / "rollout.asl")):
                        os.remove(f)
        self.dirs.append(self.D)
        if missing and self.tries < B.MAX_TRIES:
            B.log(f"{self.name}: rc {rc}, {len(missing)} of {self.n} scenes missing, rerunning those")
            fill = B.O / "lists" / f"{self.name}.fill{self.tries}.txt"
            fill.write_text("\n".join(missing) + "\n")
            self.scenes, self.n = fill, len(missing)
            self.start()
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
    B.O = Path(os.environ.get("OT_OUT") or B.DATA / "runs/body1/cl") / stage
    (B.O / "lists").mkdir(parents=True, exist_ok=True)
    for f in ("DONE", "ERROR"):
        (B.O / f).unlink(missing_ok=True)
    man_f = B.O / "manifest.json"
    man = json.loads(man_f.read_text()) if man_f.exists() else {}
    jobs = []
    for sp in specs:
        label, drv, env, _, lst = (sp.split(":") + ["", "chunks"])[:5]
        if label in man:
            B.log(f"{label}: already in the manifest, skipped")
            continue
        for part in (["chunk0", "chunk1", "chunk2"] if lst in ("", "chunks") else [lst]):
            jobs.append(Job(label, f"{label}-{Path(part).stem}", drv, dict(e.split("=", 1) for e in env.split(",") if e), Path(part) if part.startswith("/") else LISTS / f"{part}.txt"))
    ACTIVE.extend(jobs)
    while True:
        for j in jobs:
            if j.state == "new" and sum(x.state == "active" for x in jobs) < MAXA:
                j.start()
        done = [j.poll() for j in jobs]
        for label in dict.fromkeys(j.label for j in jobs):
            mine = [j for j in jobs if j.label == label]
            if label not in man and all(j.state == "done" for j in mine):
                man[label] = [str(d) for j in mine for d in j.dirs]
                man_f.write_text(json.dumps(man, indent=1))
        fg = B.free_gb()
        B.status(f"{stage}: " + ", ".join(f"{j.name} {j.n_done() if j.state == 'active' else (j.n if j.state == 'done' else 0)}/{j.n}" + (" ok" if j.state == "done" else " wait" if j.state == "new" else "")
                                          + (f" t{j.tries}" if j.tries > 1 else "") for j in jobs) + f"; free {fg:.0f} GB")
        if fg < 200:
            B.fail(f"free disk {fg:.0f} GB < 200 GB; running jobs left alone, nothing new started")
        if all(done):
            break
        time.sleep(30)
    (B.O / "DONE").write_text(time.strftime("%F %T") + "\n")
    B.status(f"{stage}: all done")


if __name__ == "__main__":
    main()
