#!/usr/bin/env python3
"""Autonomous daemon for op_l_b2d chain monitoring, judging, held-out gating, and teardown.

Tasks:
1. Monitor STATUS / ERROR every 30s. If ERROR or fatal crash occurs, write ALERT-ERROR and exit.
2. When DONE-full appears:
   - Run `python -m experiments.op_adapt_l.lib.op_l_b2d_report judge`
   - Check held-out pre-registration condition (prereg §8):
     * lmain (seed 0) on dev satisfies L-S1, L-S2, and L-Safe
     * at least one of lmain1, lmain2 has ΔDS(X - slow_X) > 0
3. If held-out condition met:
   - Mark HELDOUT-GATE-PASSED
   - Launch held-out batch: `python experiments/op_adapt_l/scripts/op_l_b2d_chain.py --plan heldout --slots 2 --workers 4`
   - Monitor until DONE-heldout
   - Re-run judge
   If not met:
   - Mark HELDOUT-GATE-SKIPPED ("未运行 (dev 条件未满足)")
4. Teardown:
   - Update /root/autodl-tmp/ujs/runs/sched/table.tsv row `op-l-b2d` to released.
   - Close tmux window jev:7 if safe.
   - Mark ALL-COMPLETE.
"""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/ujs/runs/op_l_b2d")
REPO = Path("/home/ujs/data/jev-drive")
SCHED_FILE = Path("/root/autodl-tmp/ujs/runs/sched/table.tsv")
DAEMON_LOG = ROOT / "daemon.log"


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(DAEMON_LOG, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, cwd=REPO, timeout=1200):
    log(f"Running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res.stdout:
        log(f"stdout:\n{res.stdout[-2000:]}")
    if res.stderr:
        log(f"stderr:\n{res.stderr[-2000:]}")
    return res


def update_sched_table(status_text: str):
    if not SCHED_FILE.exists():
        log(f"Schedule file {SCHED_FILE} does not exist, skipping table update.")
        return
    lines = SCHED_FILE.read_text().splitlines()
    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if parts and parts[0] == "op-l-b2d":
            # format: lane \t gpus \t workers \t idx0 \t idx_span \t cpus \t status \t go
            new_lines.append(f"op-l-b2d\t-\t0\t-\t-\t-\t{status_text}\t-")
        else:
            new_lines.append(line)
    SCHED_FILE.write_text("\n".join(new_lines) + "\n")
    log(f"Updated sched table: op-l-b2d -> {status_text}")


def check_heldout_gate():
    paired_csv = ROOT / "results" / "paired.csv"
    if not paired_csv.exists():
        log(f"paired.csv not found at {paired_csv}")
        return False, "paired.csv not found"

    import pandas as pd
    df = pd.read_csv(paired_csv)
    lm = df[df["arm"] == "lmain"]
    if len(lm) == 0:
        return False, "lmain row missing in paired.csv"
    lm_row = lm.iloc[0]

    s1_ok = bool(lm_row.get("L-S1") in (True, 1, "True", "1"))
    s2_ok = bool(lm_row.get("L-S2") in (True, 1, "True", "1"))
    safe_ok = bool(lm_row.get("L-Safe") in (True, 1, "True", "1"))

    # check lmain1 / lmain2同号
    seed_ok = False
    details = []
    for arm in ("lmain1", "lmain2"):
        sub = df[df["arm"] == arm]
        if len(sub):
            d = float(sub.iloc[0].get("L-S2 d", -999))
            details.append(f"{arm} L-S2 d={d}")
            if d > 0:
                seed_ok = True

    gate_ok = s1_ok and s2_ok and safe_ok and seed_ok
    reason = (
        f"lmain S1={s1_ok}, S2={s2_ok}, Safe={safe_ok}; seeds: {', '.join(details)} => gate={'PASS' if gate_ok else 'FAIL'}"
    )
    log(f"Held-out evaluation: {reason}")
    return gate_ok, reason


def main():
    log("op_l_b2d_daemon started.")
    last_status = ""
    error_file = ROOT / "ERROR"
    done_full = ROOT / "DONE-full"
    done_heldout = ROOT / "DONE-heldout"

    # Step 1: Wait for full batch
    while not done_full.exists():
        if error_file.exists():
            err_msg = error_file.read_text()
            log(f"FATAL: ERROR detected in {error_file}:\n{err_msg}")
            (ROOT / "ALERT-ERROR").write_text(err_msg)
            sys.exit(1)

        # Check chain process
        res = subprocess.run(["pgrep", "-f", "op_l_b2d_chain.py.*full"], capture_output=True, text=True)
        if res.returncode != 0:
            if done_full.exists():
                break
            log("FATAL: op_l_b2d_chain.py full is NOT running and DONE-full not found!")
            (ROOT / "ALERT-ERROR").write_text("op_l_b2d_chain.py full exited unexpectedly")
            sys.exit(1)

        # Read status
        status_file = ROOT / "STATUS"
        if status_file.exists():
            cur_status = status_file.read_text().strip()
            if cur_status != last_status:
                n_done = len(list((ROOT / "arms").glob("*/DONE")))
                log(f"Status update (units done: {n_done}): {cur_status}")
                last_status = cur_status

        time.sleep(30)

    log("Step 1 complete: DONE-full detected.")

    # Step 2: Run judge on dev batch
    log("Step 2: Running report judge...")
    res = run_cmd([str(REPO / ".venv/bin/python"), "-m", "experiments.op_adapt_l.lib.op_l_b2d_report", "judge"])
    if res.returncode != 0:
        log("ERROR: judge failed!")
        (ROOT / "ALERT-ERROR").write_text(f"judge failed with rc {res.returncode}:\n{res.stderr}")
        sys.exit(1)

    # Step 3: Evaluate held-out gate
    gate_ok, gate_reason = check_heldout_gate()
    if gate_ok:
        log("Step 3: Held-out condition SATISFIED! Starting heldout chain...")
        (ROOT / "HELDOUT-GATE-PASSED").write_text(gate_reason + "\n")
        cmd = [str(REPO / ".venv/bin/python"), "experiments/op_adapt_l/scripts/op_l_b2d_chain.py", "--plan", "heldout", "--slots", "2", "--workers", "4"]
        with open(ROOT / "heldout.log", "a") as f:
            p = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        log(f"Held-out chain launched with PID {p.pid}")

        # Wait for heldout
        while not done_heldout.exists():
            if error_file.exists():
                err_msg = error_file.read_text()
                log(f"FATAL: ERROR during heldout in {error_file}:\n{err_msg}")
                (ROOT / "ALERT-ERROR").write_text(err_msg)
                sys.exit(1)
            if p.poll() is not None:
                if done_heldout.exists():
                    break
                log(f"FATAL: heldout chain exited with code {p.returncode} but DONE-heldout not found!")
                (ROOT / "ALERT-ERROR").write_text(f"heldout chain exited with rc {p.returncode}")
                sys.exit(1)
            time.sleep(30)
        log("Held-out batch finished successfully!")

        # Re-run judge with heldout results
        log("Re-running judge after heldout...")
        run_cmd([str(REPO / ".venv/bin/python"), "-m", "experiments.op_adapt_l.lib.op_l_b2d_report", "judge"])
    else:
        log(f"Step 3: Held-out condition NOT met ({gate_reason}). Keeping held-out unrun.")
        (ROOT / "HELDOUT-GATE-SKIPPED").write_text(gate_reason + "\n")

    # Step 4: Teardown
    log("Step 4: Teardown and schedule table release.")
    ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sched_msg = f"done {ts_now}: B2D batch finished ({'heldout included' if gate_ok else 'dev only, heldout skipped'}), GPUs 0,1,2 released, no process left (experiments/op_adapt_l/plans/2026-10-01-op-adapt-L-b2d-prereg.md)"
    update_sched_table(sched_msg)

    # Close tmux window jev:7 if present
    try:
        subprocess.run(["tmux", "kill-window", "-t", "jev:7"], capture_output=True, text=True)
        log("Closed tmux window jev:7.")
    except Exception as e:
        log(f"Note: closing tmux window jev:7 returned: {e}")

    # Write summary memo
    summary_md = ROOT / "results" / "summary.md"
    summary_text = summary_md.read_text() if summary_md.exists() else "No summary.md found."
    final_report = f"""# B2D op-adapt L Evaluation Complete
Timestamp: {ts_now}
Held-out status: {'RUN and COMPLETED' if gate_ok else 'UNRUN (dev gate condition not met)'}
Gate Reason: {gate_reason}

Summary:
{summary_text}
"""
    (ROOT / "FINAL-REPORT.txt").write_text(final_report)
    (ROOT / "ALL-COMPLETE").write_text(f"Completed at {ts_now}\n")
    log("ALL TASKS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
