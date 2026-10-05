#!/usr/bin/env python
"""op_dagger pilot chain (box, .venv python): submits every GPU stage to the GPU pool up front (jevdrive.cl.pool), each stage `after`
the previous one, the independent jobs of a stage in parallel on whichever cards have room; then the report. Resumable: a job whose output
exists is not submitted (and drops out of the next stage's `after`). Per job log dir $DATA_DIR/runs/op_dagger/chain/<job>/ (log.txt, STATUS,
DONE / ERROR); chain STATUS / DONE in $DATA_DIR/runs/op_dagger/chain/ (DONE is written by the report job). --wait blocks until the report
job is final; --dry-run prints the specs.

  stage 1  train dg1 (DAgger on shipped rollouts) | train st1 (static control) | eval shipped (if missing, else idle)
  stage 2  collect dg1 rollouts on train (2 shards) | eval dg1 | eval st1
  stage 3  train dg2 (shipped + dg1 states, from shipped)
  stage 4  eval dg2; report -> $DATA_DIR/runs/op_dagger/report/pilot (box copy, commit from the Mac)

  .venv/bin/python experiments/op_dagger/scripts/dg_chain.py [--steps 400] [--train-vram 24] [--roll-vram 12] [--wait] [--dry-run]
"""
import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
PY = str(DD / "envs/op-train/bin/python")
S = "experiments/op_dagger/scripts/"
OUT = DD / "runs/op_dagger"
CH = OUT / "chain"


def status(msg):
    CH.mkdir(parents=True, exist_ok=True)
    (CH / "STATUS").write_text(f"{time.strftime('%F %T')} {msg}\n")
    print(time.strftime("%T"), msg, flush=True)


def submit_stage(name, jobs, after, a):
    """jobs: list of (args, log, done_path, train); submits the unfinished ones after `after`; returns their pool ids."""
    from jevdrive.cl import pool as P
    ids = []
    for args, log, done, train in jobs:
        if Path(done).exists():
            print(f"{name}: {log} finished ({done})")
            continue
        cmd = f"{shlex.join([PY] + args)} && test -f {shlex.quote(str(done))}"
        kw = dict(owner="op_dagger pilot", vram_gb=a.train_vram if train else a.roll_vram, cpu=12, train=train, after=list(after),
                  env={"OMP_NUM_THREADS": "8"}, log_dir=str(CH / log), cwd=str(REPO))
        if a.dry_run:
            print(f"{name}: would submit {log} {json.dumps(kw)}\n  {cmd}")
            ids.append(f"<{log}>")
            continue
        jid = P.submit(cmd, name=f"dg-{log}", **kw)
        print(f"{name}: {log} -> {jid}")
        ids.append(jid)
    return ids or list(after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--train-vram", type=float, default=24.0, help="VRAM GB a train job declares")
    ap.add_argument("--roll-vram", type=float, default=12.0, help="VRAM GB a collect / eval job declares")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    if not a.dry_run:
        for f in ("DONE", "ERROR"):
            (CH / f).unlink(missing_ok=True)
    R = OUT / "roll"
    while not a.dry_run and not all((R / "shipped" / f"train-collect-{i}of2.npz").exists() for i in (0, 1)):
        status("waiting for the shipped train collection (CPU)")
        time.sleep(60)
    tr = lambda tag, extra: ([S + "dg_train.py", "--tag", tag, "--steps", str(a.steps), "--workers", "12"] + extra, f"train-{tag}", OUT / "runs" / tag / "ckpt-final.pt", True)  # noqa: E731
    ev = lambda m: ([S + "dg_roll.py", "eval", "--model", m, "--set", "heldout", "--workers", "12"], f"eval-{m}", R / m / "heldout-eval-0of1.npz", False)  # noqa: E731
    co = lambda m, i: ([S + "dg_roll.py", "collect", "--model", m, "--set", "train", "--shard", f"{i}/2", "--workers", "12"], f"collect-{m}-{i}", R / m / f"train-collect-{i}of2.npz", False)  # noqa: E731
    after = []
    for name, jobs in (("stage 1", [tr("dg1", ["--rolls", "shipped"]), tr("st1", ["--rolls", "shipped", "--static"]), ev("shipped")]),
                       ("stage 2", [co("dg1", 0), co("dg1", 1), ev("dg1"), ev("st1")]),
                       ("stage 3", [tr("dg2", ["--rolls", "shipped,dg1"])]),
                       ("stage 4", [ev("dg2")])):
        after = submit_stage(name, jobs, after, a)
    rep = shlex.join([PY, S + "dg_report.py", "--models", "shipped", "dg1", "st1", "dg2", "--control", "st1", "--out", str(OUT / "report/pilot")])
    rep += f" && date '+%F %T' > {shlex.quote(str(CH / 'DONE'))}"
    if a.dry_run:
        print(f"report: would submit after {after}\n  {rep}")
        return
    from jevdrive.cl import pool as P
    jid = P.submit(rep, name="dg-report", owner="op_dagger pilot", vram_gb=2, after=after, log_dir=str(CH / "report"), cwd=str(REPO))
    status(f"submitted; report job {jid} (python -m jevdrive.cl queue)")
    if a.wait:
        st = P.wait([jid], 120)[jid]
        status(f"report job {jid}: {st}")
        sys.exit(0 if st == "done" else 1)


if __name__ == "__main__":
    main()
