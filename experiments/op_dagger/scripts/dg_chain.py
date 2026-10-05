#!/usr/bin/env python
"""op_dagger pilot chain (box, .venv python): waits for a lease of 1-3 cards (`jevdrive.cl lease op-dagger`), then runs the GPU stages, the
independent ones in parallel on the leased cards, and releases the lease. Resumable: every stage skips when its output exists.
STATUS / DONE / ERROR in $DATA_DIR/runs/op_dagger/chain/.

  stage 1  train dg1 (DAgger on shipped rollouts) | train st1 (static control) | eval shipped (if missing, else idle)
  stage 2  collect dg1 rollouts on train (2 shards) | eval dg1 | eval st1
  stage 3  train dg2 (shipped + dg1 states, from shipped)
  stage 4  eval dg2; report -> experiments/op_dagger/results/pilot (box copy, commit from the Mac)

  scripts/tmux_run.sh dg-chain .venv/bin/python experiments/op_dagger/scripts/dg_chain.py [--steps 400] [--max-cards 3]
"""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
PY = str(DD / "envs/op-train/bin/python")
S = "experiments/op_dagger/scripts/"
OUT = DD / "runs/op_dagger"
CH = OUT / "chain"
LANE = "op-dagger"


def status(msg):
    CH.mkdir(parents=True, exist_ok=True)
    (CH / "STATUS").write_text(f"{time.strftime('%F %T')} {msg}\n")
    print(time.strftime("%T"), msg, flush=True)


def lease(max_cards):
    """Poll until 1..max_cards cards are granted; returns (cards, {card: cpus})."""
    from jevdrive.cl import lease as L
    while True:
        for k in range(max_cards, 0, -1):
            r = subprocess.run([sys.executable, "-m", "jevdrive.cl", "lease", LANE, "--gpus", str(k), "--status", "op_dagger pilot (train / rollouts)"],
                               capture_output=True, text=True, cwd=REPO)
            if r.returncode == 0 and "granted" in r.stdout:
                ls = L.get(LANE)
                return sorted(ls.cards), {c: ls.cards[c]["cpus"] for c in ls.cards}
        status("waiting for a free card (lease op-dagger)")
        time.sleep(300)


def release(note):
    subprocess.run([sys.executable, "-m", "jevdrive.cl", "release", LANE, note], capture_output=True, text=True, cwd=REPO)


def job(args, card, cpus, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "OMP_NUM_THREADS": "8"}
    cmd = (["taskset", "-c", cpus] if cpus else []) + [PY] + args
    f = open(CH / f"{log}.log", "a")
    f.write(f"$ {' '.join(cmd)}\n")
    f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO)


def run_stage(name, jobs, cards, cpus):
    """jobs: list of (args, log, done_path); runs them over the cards, at most one per card at a time."""
    todo = [j for j in jobs if not Path(j[2]).exists()]
    status(f"{name}: {len(todo)} jobs on cards {cards}")
    running = {}
    while todo or running:
        for c in cards:
            if c not in running and todo:
                a, log, _ = todo.pop(0)
                running[c] = (job(a, c, cpus.get(c, ""), log), log)
        time.sleep(15)
        for c, (p, log) in list(running.items()):
            if p.poll() is not None:
                if p.returncode != 0:
                    raise RuntimeError(f"{name}: {log} exited {p.returncode} (see {CH / log}.log)")
                del running[c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max-cards", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    for f in ("DONE", "ERROR"):
        (CH / f).unlink(missing_ok=True)
    R = OUT / "roll"
    while not all((R / "shipped" / f"train-collect-{i}of2.npz").exists() for i in (0, 1)):
        status("waiting for the shipped train collection (CPU)")
        time.sleep(60)
    tr = lambda tag, extra: ([S + "dg_train.py", "--tag", tag, "--steps", str(a.steps), "--workers", "12"] + extra, f"train-{tag}", OUT / "runs" / tag / "ckpt-final.pt")  # noqa: E731
    ev = lambda m: ([S + "dg_roll.py", "eval", "--model", m, "--set", "heldout", "--workers", "12"], f"eval-{m}", R / m / "heldout-eval-0of1.npz")  # noqa: E731
    co = lambda m, i: ([S + "dg_roll.py", "collect", "--model", m, "--set", "train", "--shard", f"{i}/2", "--workers", "12"], f"collect-{m}-{i}", R / m / f"train-collect-{i}of2.npz")  # noqa: E731
    cards, cpus = lease(a.max_cards)
    try:
        run_stage("stage 1", [tr("dg1", ["--rolls", "shipped"]), tr("st1", ["--rolls", "shipped", "--static"]), ev("shipped")], cards, cpus)
        run_stage("stage 2", [co("dg1", 0), co("dg1", 1), ev("dg1"), ev("st1")], cards, cpus)
        run_stage("stage 3", [tr("dg2", ["--rolls", "shipped,dg1"])], cards, cpus)
        run_stage("stage 4", [ev("dg2")], cards, cpus)
        release("op_dagger pilot done")
        status("report")
        subprocess.run([PY, S + "dg_report.py", "--models", "shipped", "dg1", "st1", "dg2", "--control", "st1", "--out", str(OUT / "report/pilot")],
                       check=True, cwd=REPO, stdout=open(CH / "report.log", "w"), stderr=subprocess.STDOUT)
        (CH / "DONE").write_text(time.strftime("%F %T\n"))
        status("DONE")
    except BaseException:
        release("op_dagger pilot error")
        (CH / "ERROR").write_text(traceback.format_exc())
        status("ERROR")
        raise


if __name__ == "__main__":
    main()
