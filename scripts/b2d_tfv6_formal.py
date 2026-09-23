"""Run frozen W2 levels 1, 1r and 2, analyzing between levels."""

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "todos/2026-09-23-tfv6-controller/protocol.md"
BUS = Path("/data/runs/b2d/tfv6-w2/bus")


def record(out, kind, **values):
    event = {"t": time.time(), "kind": kind, **values}
    with open(out / "events.jsonl", "a", buffering=1) as stream:
        stream.write(json.dumps(event) + "\n")
    with open(out / "log.txt", "a", buffering=1) as stream:
        stream.write(f"{dt.datetime.now().isoformat()} {kind} {values}\n")
    print(kind, values, flush=True)


def bus_progress(out, stop):
    # Also fire once near each level start; later updates are at most two hours apart.
    started = time.monotonic()
    while not stop.is_set():
        finished = len(list(out.glob("level*/cases/**/done.json")))
        failed = len(list(out.glob("level*/cases/**/failed.json")))
        infra_retries = sum(max(0, len(list(parent.glob("attempt-*/result.json"))) - 1)
                            for parent in out.glob("level*/cases/**/*") if parent.is_dir() and
                            (parent / "attempt-1").exists())
        eta_s = (time.monotonic() - started) * (202 - finished) / finished if finished else None
        message = {"t": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": 2,
                   "step": "campaign_progress", "msg": f"{finished}/202 cases done; {failed} exhausted cases",
                   "numbers": {"cases_done": finished, "cases_total": 202,
                               "infra_retries": infra_retries, "eta_s": eta_s}}
        with open(BUS / "status.jsonl", "a", buffering=1) as stream:
            stream.write(json.dumps(message) + "\n")
        stop.wait(7200)


def run(command, out, label):
    record(out, "step_start", label=label, command=command)
    result = subprocess.run(command, cwd=ROOT)
    record(out, "step_end", label=label, returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f"{label} failed with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-protocol-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency != 2:
        parser.error("W2 phase-1 measurement selected exactly two concurrent servers")
    frozen = subprocess.run(["git", "show", f"{args.frozen_protocol_commit}:{PROTOCOL}"],
                            cwd=ROOT, capture_output=True)
    if frozen.returncode or frozen.stdout != (ROOT / PROTOCOL).read_bytes():
        parser.error("Frozen commit is unavailable or its protocol differs from this worktree")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    record(out, "start", frozen_protocol_commit=args.frozen_protocol_commit, concurrency=2)
    stop = threading.Event()
    progress = threading.Thread(target=bus_progress, args=(out, stop), daemon=True)
    progress.start()
    try:
        for level in ("1", "1r", "2"):
            campaign = out / f"level{level}"
            run([sys.executable, str(ROOT / "scripts/b2d_tfv6_campaign.py"),
                 "--level", level, "--out", str(campaign), "--concurrency", "2",
                 "--server-index", "90", "--frozen-protocol-commit",
                 args.frozen_protocol_commit], out, f"level{level}")
            roots = [out / "level1", out / "level1r"]
            if level == "2":
                roots.append(out / "level2")
            run([sys.executable, str(ROOT / "scripts/b2d_tfv6_analyze.py"),
                 *map(str, roots), "--out", str(out / f"analysis-after-{level}")],
                out, f"analysis-after-{level}")
        record(out, "end", status="complete")
    finally:
        stop.set()
        progress.join(timeout=2)


if __name__ == "__main__":
    main()
