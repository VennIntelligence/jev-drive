"""Resumable, route-paired TFv6 campaign using isolated CARLA workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")
RUNTIME = DATA / "runs/b2d/tfv6-repro/runtime/Bench2Drive"
DEV10 = RUNTIME / "leaderboard/data/drivetransformer_bench2drive_dev10.xml"
HOLDOUT = ROOT / "todos/2026-09-22-b2d-controller/results/holdout.xml"
AGENT = ROOT / "scripts/b2d_tfv6_controller_agent.py"
WEIGHTS = DATA / "checkpoints/tfv6/tfv6_resnet34"
PYTHON = DATA / "envs/tfv6/bin/python"
LOCK = threading.Lock()


def routes_in(xml_path):
    return sorted((x.attrib["id"] for x in ET.parse(xml_path).getroot()), key=int)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def record(out, kind, **data):
    event = {"t": time.time(), "kind": kind, **data}
    with LOCK:
        with open(out / "events.jsonl", "a", buffering=1) as stream:
            stream.write(json.dumps(event) + "\n")
        with open(out / "log.txt", "a", buffering=1) as stream:
            stream.write(f"{dt.datetime.now().isoformat()} {kind} {data}\n")
    tqdm.write(f"{kind}: {data}")


def monitor_resources(out, stop):
    with open(out / "resources.jsonl", "a", buffering=1) as stream:
        while not stop.is_set():
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
            ps = subprocess.run(["ps", "-eo", "pcpu,args"], capture_output=True, text=True)
            cpu = 0.0
            for line in ps.stdout.splitlines()[1:]:
                if "CarlaUE4-Linux-Shipping" in line or ("b2d_route.py" in line and str(ROOT) in line):
                    try:
                        cpu += float(line.split(maxsplit=1)[0])
                    except ValueError:
                        pass
            try:
                memory, utilization = [float(x.strip()) for x in gpu.stdout.splitlines()[0].split(",")]
            except (ValueError, IndexError):
                memory = utilization = None
            stream.write(json.dumps({"t": time.time(), "gpu_memory_mib": memory,
                                     "gpu_util_pct": utilization, "process_cpu_pct": cpu}) + "\n")
            stop.wait(2)


def clean_owned_server(run_dir, out):
    for pid_file in (run_dir / "servers").glob("*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except (ValueError, FileNotFoundError, ProcessLookupError):
            continue
        if b"CarlaUE4" not in command:
            continue
        record(out, "orphan_server", pid=pid, source=str(pid_file))
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        if Path(f"/proc/{pid}").exists():
            os.kill(pid, signal.SIGKILL)


def case(out, xml_path, level, route, seed, arm, server_index, max_ticks):
    case_dir = out / "cases" / level / f"route-{route}" / f"seed-{seed}" / arm
    done = case_dir / "done.json"
    if done.exists():
        record(out, "case_skip", level=level, route=route, seed=seed, arm=arm)
        return json.loads(done.read_text())
    attempts = sorted(case_dir.glob("attempt-*/result.json"))
    for number in range(len(attempts) + 1, 5):
        attempt_dir = case_dir / f"attempt-{number}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        run_dir = attempt_dir / "run"
        env = os.environ.copy()
        env.update(DATA_DIR=str(DATA), BENCH2DRIVE_ROOT=str(RUNTIME),
                   LEAD_PROJECT_ROOT=str(DATA / "third_party/lead-cvpr2026"),
                   HF_HOME=str(DATA / "cache/huggingface"), HF_HUB_OFFLINE="1",
                   OPENBLAS_CORETYPE="Barcelona", OMP_NUM_THREADS="4",
                   PYTHONPATH=str(DATA / "third_party/lead-cvpr2026"),
                   B2D_W2_LOG_DIR=str(attempt_dir),
                   LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=True use_kalman_filter=True slower_for_stop_sign=True")
        command = [str(PYTHON), str(ROOT / "scripts/b2d_run.py"), "--routes", str(xml_path),
                   "--route-ids", route, "--out", str(run_dir), "--workers", "1",
                   "--server-index", str(server_index), "--gpu-rank", "0", "--quality", "Epic",
                   "--max-attempts", "1", "--agent", str(AGENT), "--agent-config",
                   f"{WEIGHTS}+{arm}", "--python", str(PYTHON), "--tm-seed", str(seed),
                   "--max-ticks", str(max_ticks)]
        record(out, "case_start", level=level, route=route, seed=seed, arm=arm, attempt=number)
        start = time.monotonic()
        with open(attempt_dir / "runner.log", "w") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        clean_owned_server(run_dir, out)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        reports = summary.get("attempts", {}).get(route, [])
        status = reports[-1]["status"] if reports else "missing_summary"
        official_status = None
        official_path = run_dir / "attempts" / route / "1" / "results.json"
        if official_path.exists():
            try:
                records = json.loads(official_path.read_text())["_checkpoint"]["records"]
                official_status = records[0]["status"] if len(records) == 1 else None
            except (ValueError, KeyError, TypeError):
                official_status = None
        result = {"level": level, "route": route, "seed": seed, "arm": arm,
                  "attempt": number, "status": status, "returncode": process.returncode,
                  "official_status": official_status,
                  "wall_s": time.monotonic() - start, "run_dir": str(run_dir)}
        atomic_json(attempt_dir / "result.json", result)
        record(out, "case_attempt_end", **result)
        official_driving_result = (official_status is not None and
                                   "agent crashed" not in official_status.lower() and
                                   "agent error" not in official_status.lower())
        if status == "finished" and official_driving_result:
            atomic_json(done, {**result, "infra_retries": number - 1})
            return result
        if status == "finished":
            record(out, "missing_official_result", **result)
        if number == 4:
            atomic_json(case_dir / "failed.json", result)
            return result
    raise AssertionError("unreachable")


def group(out, xml_path, level, route, seed, arms, server_index, max_ticks):
    return [case(out, xml_path, level, route, seed, arm, server_index, max_ticks)
            for arm in arms]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("smoke", "1", "1r", "2"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--server-index", type=int, default=90)
    parser.add_argument("--route-id", default="25378", help="smoke only")
    parser.add_argument("--arms", default="ABCD", help="smoke only; formal arms are frozen")
    parser.add_argument("--arm", choices=tuple("ABCD"), help="one smoke arm")
    parser.add_argument("--smoke-replicas", type=int, default=1,
                        help="independent TM seeds for concurrency measurement")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 3:
        parser.error("concurrency must be 1, 2 or 3")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    xml_path = HOLDOUT if args.level == "2" else DEV10
    routes = [args.route_id] if args.level == "smoke" else routes_in(xml_path)
    seeds = tuple(range(args.smoke_replicas)) if args.level == "smoke" else (
        (0,) if args.level == "1r" else (0, 1, 2))
    if args.level != "smoke" and args.arms != "ABCD":
        parser.error("formal arms are fixed by protocol")
    if args.arm and args.level != "smoke":
        parser.error("--arm is only valid for smoke")
    arms = (args.arm or args.arms) if args.level == "smoke" else (
        "A" if args.level == "1r" else "ABCD")
    if not arms or any(arm not in "ABCD" for arm in arms) or len(set(arms)) != len(arms):
        parser.error("--arms must contain distinct letters from ABCD")
    groups = [(route, seed) for route in routes for seed in seeds]
    record(out, "start", level=args.level, groups=len(groups), cases=len(groups) * len(arms),
           concurrency=args.concurrency, xml=str(xml_path))
    all_results = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_resources, args=(out, stop), daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            future_map = {pool.submit(group, out, xml_path, args.level, route, seed, arms,
                                      args.server_index + worker, 350 if args.level == "smoke" else 0):
                          (route, seed, worker)
                          for worker, (route, seed) in enumerate(groups)
                          if worker < args.concurrency}
            pending = iter(groups[args.concurrency:])
            with tqdm(total=len(groups) * len(arms), desc=f"TFv6 level {args.level}") as progress:
                while future_map:
                    for future in as_completed(list(future_map)):
                        route, seed, worker = future_map.pop(future)
                        rows = future.result()
                        all_results.extend(rows)
                        progress.update(len(rows))
                        next_group = next(pending, None)
                        if next_group:
                            index = args.server_index + worker
                            new = pool.submit(group, out, xml_path, args.level, *next_group,
                                              arms, index, 350 if args.level == "smoke" else 0)
                            future_map[new] = (*next_group, worker)
                        break
    finally:
        stop.set()
        monitor.join(timeout=5)
    record(out, "end", level=args.level, cases=len(all_results),
           finished=sum(row["status"] == "finished" for row in all_results))


if __name__ == "__main__":
    main()
