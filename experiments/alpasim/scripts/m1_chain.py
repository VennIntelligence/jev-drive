#!/usr/bin/env python3
"""M1 chain: closed-loop AlpaSim runs of the yaw-damping candidates (decision 205) as GPU-pool jobs, with c0b_chain.py's stall watchdog.

  scripts/tmux_run.sh m1 python3 experiments/alpasim/scripts/m1_chain.py <stage> <job> [<job> ...]       (box; plain python3, stdlib only)

job = name:driver:scene list:ENV=VALUE[,ENV=VALUE...][:wizard override[,override...]]
  e.g.  w0:sh30:s1:SH30_TAG=SH30-F-s0,SH30_MOTION=0   g19:sh30:s1:SH30_TAG=SH30-F-s0:controller.gains.idx_start_penalty=19
scene list = a file name under $DATA_DIR/runs/alpasim/m1/lists (without .txt). Runs land in m1/runs/<stage>-<name>/<timestamp>; the stage
writes m1/<stage>.DONE or m1/<stage>.ERROR and keeps m1/STATUS current. rollout.asl is kept for failed rollouts and for the scenes of
m1/lists/keep.txt (the showcase scenes); the rest is pruned.
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

O = B.RA / "m1"
B.O = O                                     # c0b_chain's log / status / fail write here


class Job(B.Job):
    def __init__(self, stage, name, drv, env, scenes, extra=()):
        super().__init__(f"{stage}-{name}", drv, env, scenes, 12)
        self.extra = list(extra)
        k = O / "lists/keep.txt"
        self.keep = set(k.read_text().split()) if k.exists() else set()

    def submit(self):
        self.tries += 1
        if self.tries > B.MAX_TRIES:
            B.fail(f"{self.name}: {B.MAX_TRIES} tries exhausted")
        self.D = O / "runs" / self.name / time.strftime("%Y%m%d-%H%M%S")
        self.D.mkdir(parents=True)
        cmd = B.PY + ["submit", "--name", f"alpasim-m1-{self.name}", "--vram", str(self.vram), "--cpu", "8", "--ram", str(int(10 + 0.12 * self.n)),
                      "--timeout-h", "3", "--tries", "1", "--priority", str(self.prio), "--log-dir", str(self.D / "pool"), "--",
                      "env", *[f"{k}={v}" for k, v in self.env.items()], "bash", "experiments/alpasim/scripts/run.sh", str(self.D), self.drv,
                      "--scene-list", str(self.scenes), *B.OVERRIDES, *self.extra]
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
            if not r.get("failure_reason") and sid not in self.keep:
                for f in glob.glob(str(self.D / "rollouts" / sid / "*" / "rollout.asl")):
                    os.remove(f)
        self.dirs.append(self.D)
        if missing and self.tries < B.MAX_TRIES:
            B.log(f"{self.name}: {len(missing)} of {self.n} scenes missing, resubmitting those")
            fill = O / "lists" / f"{self.name}.fill{self.tries}.txt"
            fill.write_text("\n".join(missing) + "\n")
            self.scenes, self.n = fill, len(missing)
            self.submit()
            return False
        if missing:
            B.fail(f"{self.name}: {len(missing)} scenes still missing after {self.tries} tries")
        sc = [r["score"] for r in R.values()]
        self.state = "done"
        B.log(f"{self.name}: done, {len(R)} rollouts in {self.D}")
        (self.D / "M1_DONE").write_text(json.dumps(dict(n=len(sc))) + "\n")
        return True


def main():
    stage, specs = sys.argv[1], sys.argv[2:]
    O.mkdir(parents=True, exist_ok=True)
    for f in (f"{stage}.DONE", f"{stage}.ERROR", "ERROR"):
        (O / f).unlink(missing_ok=True)
    jobs = []
    for sp in specs:
        name, drv, lst, env, *ex = sp.split(":")
        jobs.append(Job(stage, name, drv, dict(e.split("=", 1) for e in env.split(",") if e), O / "lists" / f"{lst}.txt",
                        [x for x in ",".join(ex).split(",") if x]))
    try:
        for j in jobs:
            j.submit()
        B.wait_all(jobs, stage)
    except SystemExit:
        (O / f"{stage}.ERROR").write_text((O / "ERROR").read_text() if (O / "ERROR").exists() else "failed\n")
        raise
    (O / "manifest.json").write_text(json.dumps({**(json.loads((O / "manifest.json").read_text()) if (O / "manifest.json").exists() else {}),
                                                 **{j.name: [str(d) for d in j.dirs] for j in jobs}}, indent=1))
    (O / f"{stage}.DONE").write_text(time.strftime("%F %T") + "\n")
    B.status(f"{stage}: all done")


if __name__ == "__main__":
    main()
