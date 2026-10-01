#!/usr/bin/env python3
"""Read-only GPU and handed-off lane monitoring; run inside tmux."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time
from collections import deque
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration-s", type=float, default=86400)
    parser.add_argument("--interval-s", type=float, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.interval_s <= 0:
        parser.error("duration and interval must be positive")
    data = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
    out = args.out or data / "runs/nq4/cx/gpu-watch"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "GPU watcher already running in this output directory\n")
        (out / "pid").write_text(f"{os.getpid()}\n")
        with (out / "log.txt").open("a", buffering=1) as log, \
                (out / "events.jsonl").open("a", buffering=1) as events, \
                (out / "samples.csv").open("a", newline="", buffering=1) as samples:
            writer = csv.writer(samples)
            if samples.tell() == 0:
                writer.writerow(["time_utc", "index", "uuid", "memory_used_mib",
                                 "memory_total_mib", "utilization_percent"])

            def event(kind, **fields):
                record = {"t": time.time(), "time_utc": stamp(), "kind": kind, **fields}
                events.write(json.dumps(record) + "\n")
                log.write(f"{record['time_utc']} {kind}: {json.dumps(fields)}\n")

            history = deque(maxlen=10)
            start = time.monotonic()
            next_lanes = start
            event("start", pid=os.getpid(), duration_s=args.duration_s,
                  interval_s=args.interval_s)
            try:
                while True:
                    iteration = time.monotonic()
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, check=True)
                        rows = []
                        for row in csv.reader(io.StringIO(result.stdout)):
                            index, uuid, used, total, utilization = [x.strip() for x in row]
                            rows.append({"index": int(index), "uuid": uuid,
                                         "used": int(used), "total": int(total),
                                         "utilization": int(utilization)})
                        if not rows:
                            raise ValueError("nvidia-smi returned no GPU rows")
                        now = stamp()
                        for gpu in rows:
                            writer.writerow([now, gpu["index"], gpu["uuid"], gpu["used"],
                                             gpu["total"], gpu["utilization"]])
                        mean = sum(g["utilization"] for g in rows) / len(rows)
                        history.append(mean)
                        rolling = sum(history) / len(history)
                        lines = [f"# GPU watch ({now})", "",
                                 f"Machine mean utilization: {mean:.1f}%.",
                                 f"Rolling mean ({len(history)} samples, maximum 10): {rolling:.1f}%.", "",
                                 "| GPU | Memory MiB | Utilization | Observation |",
                                 "|:--|--:|--:|:--|"]
                        observations = []
                        for gpu in rows:
                            if gpu["used"] == 0 and gpu["utilization"] == 0:
                                note = "Idle: zero memory and utilization"
                            elif gpu["utilization"] < 10:
                                note = "Low utilization with allocated memory; reservation/ownership unverified"
                            else:
                                note = "Active"
                            observations.append({"index": gpu["index"], "observation": note})
                            lines.append(f"| {gpu['index']} | {gpu['used']} / {gpu['total']} | "
                                         f"{gpu['utilization']}% | {note} |")
                        lines += ["", "Allocated memory does not establish an idle card or permission to launch a job."]
                        temporary = out / "STATUS.md.tmp"
                        temporary.write_text("\n".join(lines) + "\n")
                        temporary.replace(out / "STATUS.md")
                        event("gpu_sample", gpus=rows, machine_mean=mean,
                              rolling_mean=rolling, observations=observations)
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        event("sample_error", error=str(exc))
                    if iteration >= next_lanes:
                        for lane in "abcd":
                            directory = data / "runs/nq3" / lane
                            snapshot = {}
                            for name in ("STATUS.md", "ERROR", "DONE"):
                                path = directory / name
                                try:
                                    snapshot[name] = path.read_text() if path.is_file() else None
                                except OSError as exc:
                                    snapshot[name] = f"Read error: {exc}"
                            event("lane_snapshot", lane=lane, files=snapshot)
                        next_lanes = iteration + 3600
                    if args.once or time.monotonic() - start >= args.duration_s:
                        break
                    delay = min(args.interval_s - (time.monotonic() - iteration),
                                args.duration_s - (time.monotonic() - start))
                    if delay > 0:
                        time.sleep(delay)
            except KeyboardInterrupt:
                event("interrupted")
            finally:
                event("end", elapsed_s=time.monotonic() - start)
                (out / "pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
