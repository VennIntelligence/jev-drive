#!/usr/bin/env python3
"""Isolated, capacity-gated lane-B pilots; the running lane scripts are unchanged.

Usage: sch_cl_parallel.py plan|run --slots slots.json [--arms cl5 cl6 ...]
slots.json: [{"name":"a","gpu":1,"workers":2,"idx":390,"span":10,
              "cpus":"110-117","model_gb":30,"table_lane":"sch-pilot-b"}]

Every slot must be covered by an existing SCH reservation. GPU 1 remains a
pilot-only card. Stop the old sch_cl_pilots.sh entry point before using run;
this program refuses to coexist with it. Each arm has its own B_DIR and flock,
and explicit route output paths keep nq3_b.sh's EXIT cleanup local. The 1-route
and 10-route stages never share requested.json. Canonical ten-route outputs
and verdicts remain at runs/sched/pilot/b/arms/<arm>/s0 for the existing gate.
No full batches, grants, scientific thresholds, or approvals are written here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import sch_table as SCH

REPO = Path(__file__).resolve().parents[1]
DATA = SCH.DATA
MAIN = DATA / "runs/nq3/b"
PILOT = DATA / "runs/sched/pilot/b"
CONTROL = PILOT / "parallel"
OBSTACLES = {"Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
             "ParkedObstacle", "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays"}
ARMS = ("cl3", "cl4", "cl5", "cl6", "cl8", "tfv6", "bridgedrive", "blue", "simlingo")
STOP = threading.Event()
WRITE_LOCK = threading.Lock()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def processes():
    result = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().decode(errors="replace").strip("\0").split("\0")
            if argv and argv[0]:
                result.append((int(path.parent.name), argv))
        except (OSError, ValueError):
            pass
    return result


def old_pilots():
    return [pid for pid, argv in processes() if any(Path(a).name == "sch_cl_pilots.sh" for a in argv)]


def route_ids(arm):
    xml = DATA / "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
    routes = list(ET.parse(xml).getroot().iter("route"))
    if arm in ("cl5", "cl5d"):
        routes = [r for r in routes if any(s.get("type") in OBSTACLES for s in r.iter("scenario"))][:10]
    else:
        routes = routes[::22][:10]
    ids = [r.get("id") for r in routes]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError(f"{arm}: expected 10 distinct pilot routes")
    return ids


def skipped(arm):
    path = MAIN / "SKIP"
    return path.exists() and f"{arm} 0" in path.read_text().splitlines()


def approved(arm):
    path = MAIN / "APPROVED"
    return path.exists() and arm in path.read_text().splitlines()


def dependency(arm):
    if arm in ("cl5", "cl5d"):
        return (DATA / "runs/nq3/q2/closed_loop_head/READY").exists()
    if arm == "mc_real0":
        return (DATA / "runs/nq3/q4a/PASS").exists()
    return True


def validate_slots(slots, rows):
    if not slots:
        raise ValueError("at least one slot is required")
    conflicts = SCH.conflicts(rows)
    if conflicts:
        raise ValueError("; ".join(conflicts))
    names, blocks, totals = set(), [], {}
    for slot in slots:
        name = slot["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name) or name in names:
            raise ValueError("slot names must be unique lowercase identifiers")
        names.add(name)
        if slot["gpu"] != 1:
            raise ValueError("this entry point only runs validation pilots on GPU 1")
        if not 1 <= slot["workers"] <= slot["span"] or slot["idx"] < 0 or slot["idx"] + slot["span"] > 495:
            raise ValueError(f"{name}: invalid worker count or index block")
        if not slot["cpus"] or slot["model_gb"] <= 0:
            raise ValueError(f"{name}: explicit CPU set and positive model VRAM reserve required")
        row = next((r for r in rows if r["lane"] == slot["table_lane"]), None)
        block = set(range(slot["idx"], slot["idx"] + slot["span"]))
        if row is None or row["status"].startswith(("done", "revoked")) or slot["gpu"] not in SCH.gpus(row):
            raise ValueError(f"{name}: no active SCH reservation")
        if not block <= SCH.indices(row):
            raise ValueError(f"{name}: block is outside SCH reservation")
        for other in blocks:
            if block & (other | {i+120 for i in other} | {i-120 for i in other}):
                raise ValueError(f"{name}: slot index conflict")
        blocks.append(block)
        totals[slot["table_lane"]] = totals.get(slot["table_lane"], 0) + slot["workers"]
        if totals[slot["table_lane"]] > int(row["workers"]):
            raise ValueError(f"{name}: slots exceed reserved worker count")


def capacity(slots, active, rows, probe):
    """Conservatively reserve table workers that have not started yet, too."""
    live = sum(g["carla"] for g in probe["gpus"])
    promised = sum(int(r["workers"]) * len(SCH.gpus(r)) for r in rows
                   if r["workers"].isdigit() and not r["status"].startswith(("done", "revoked")))
    projected = probe["pids"] + max(0, promised - live) * SCH.PIDS_PER_WORKER
    if projected > SCH.PIDS_CAP or probe["cores_used"] > SCH.CPU_CAP:
        return False, f"pending-inclusive pids {projected}; cores {probe['cores_used']:.1f}"
    # These are whole-pilot admission checks. Reserve the peak 10-route stage
    # and all admitted model loads before they become visible to nvidia-smi.
    gpu = next(g for g in probe["gpus"] if g["gpu"] == 1)
    reserve = sum(s["model_gb"] + s["workers"] * SCH.VRAM_PER_WORKER_GB for s in slots if s["name"] in active)
    if gpu["used_gb"] + reserve > SCH.VRAM_CAP_GB:
        return False, f"VRAM used + admission reserves {gpu['used_gb'] + reserve:.1f} GB"
    return True, "capacity available"


def port_block_free(slot):
    # Match b2d_run's port ranges, including traffic-manager scan offsets.
    import socket
    held = []
    try:
        for i in range(slot["idx"], slot["idx"] + slot["span"]):
            for port in [2000 + 50*i + k for k in (0, 1, 2)] + list(range(8000 + 50*i, 8050 + 50*i)):
                sock = socket.socket()
                held.append(sock)
                sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        for sock in held:
            sock.close()


def invocation(arm, slot, one=False):
    ids = route_ids(arm)
    out = (PILOT / "one" if one else PILOT) / "arms" / arm / "s0"
    lane = CONTROL / "lanes" / arm
    env = dict(os.environ, B_DIR=str(lane), B_GO=str(lane / "GO.unused"), B_REPORT="0",
               GPUS=str(slot["gpu"]), WORKERS=str(1 if one else slot["workers"]), B_CPUS=slot["cpus"],
               B_IDX=f"{slot['gpu']}:{slot['idx']}", INDEX_SPAN=str(slot["span"]),
               RUN_FLAGS=f"--fast-copy --cache-lights --index-span {slot['span']}")
    cmd = [str(REPO / "scripts/nq3_b.sh"), "arm", arm, "0", ",".join(ids[:1] if one else ids), "3.0", "0", str(out)]
    return cmd, env, out


def run_command(cmd, env, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as stream:
        child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(logfile.with_suffix(".pid.json"), {"pid": child.pid, "argv": cmd, "started": time.time()})
        while child.poll() is None:
            if STOP.wait(2):
                # nq3_b.sh's TERM trap owns its recorded runner/server cleanup.
                child.send_signal(signal.SIGTERM)
                child.wait()
                return 130
        return child.returncode


def checklist(out, one=False, runner_rc=0):
    dest = out / "verdict.pending.json"
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/sch_cl_checklist.py"), str(out),
           "--min-routes", "1" if one else "10", "--out", str(dest)]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode not in (0, 2, 3, 4) or not dest.exists():
        raise RuntimeError(f"checklist failed: {result.stderr[-1500:]}")
    value = json.loads(dest.read_text())
    if runner_rc and value["verdict"] == "PASS":
        raise RuntimeError(f"runner exited {runner_rc} despite PASS checklist; withholding gate")
    dest.replace(out / "verdict.json")
    return value


def escalate(arm, value, one):
    skip_written = False
    if value["verdict"] == "FAIL" and not one:
        MAIN.mkdir(parents=True, exist_ok=True)
        with (MAIN / "SKIP.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (MAIN / "arms" / arm / "s0/requested.json").exists():
                target = MAIN / "SKIP"
                lines = list(dict.fromkeys(target.read_text().splitlines())) if target.exists() else []
                lines += [f"{arm} {seed}" for seed in range(3) if f"{arm} {seed}" not in lines]
                fd, name = tempfile.mkstemp(prefix=".SKIP.", dir=MAIN)
                with os.fdopen(fd, "w") as stream:
                    stream.write("\n".join(lines) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, target)
                skip_written = True
    with WRITE_LOCK:
        msg = f"{arm} s0 {'one-route' if one else 'ten-route'}: {value['verdict']} " + "; ".join(value.get("fail", []) + value.get("flag", []))
        if skip_written:
            msg += "; seeds 0-2 appended to SKIP"
        with (DATA / "runs/sched/ESCALATE.md").open("a") as stream:
            stream.write(f"- {time.strftime('%F %T')} [SCH parallel pilot] {msg} ({value['dir']}/verdict.json)\n")
    # A one-route failure stops here for review; automatic three-seed SKIP is
    # the existing ten-route policy only.


def pilot(arm, slot):
    claim = CONTROL / "claims" / f"{arm}.lock"
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"arm": arm, "state": "claimed elsewhere"}
        final = PILOT / "arms" / arm / "s0/verdict.json"
        if final.exists() or skipped(arm):
            return {"arm": arm, "state": "already settled"}
        if (MAIN / "arms" / arm / "s0/requested.json").exists():
            return {"arm": arm, "state": "main already running; inspect in place"}
        for one in (True, False):
            cmd, env, out = invocation(arm, slot, one)
            if (out / "verdict.json").exists():
                value = json.loads((out / "verdict.json").read_text())
            else:
                if not port_block_free(slot):
                    return {"arm": arm, "state": "ports occupied; deferred"}
                rc = run_command(cmd, env, CONTROL / "logs" / f"{arm}-{'one' if one else 'ten'}.log")
                if STOP.is_set() or not (out / "requested.json").exists():
                    return {"arm": arm, "state": "runner failed", "returncode": rc, "out": str(out)}
                value = checklist(out, one, rc)
            if value["verdict"] != "PASS":
                escalate(arm, value, one)
                if value["verdict"] != "FLAG" or not approved(arm):
                    return {"arm": arm, "state": value["verdict"], "out": str(out)}
        return {"arm": arm, "state": "pilot complete"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("plan", "run"))
    ap.add_argument("--slots", required=True)
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = ap.parse_args()
    slots = json.loads(Path(args.slots).read_text())
    validate_slots(slots, SCH.load())
    jobs = [arm for arm in dict.fromkeys(args.arms) if not skipped(arm) and dependency(arm)
            and not (PILOT / "arms" / arm / "s0/verdict.json").exists()]
    if args.command == "plan":
        print(json.dumps({"slots": slots, "arms": jobs, "old_pilot_pids": old_pilots(),
                          "commands": {a: invocation(a, slots[0])[0] for a in jobs}}, indent=2))
        return
    if old_pilots():
        sys.exit("old sch_cl_pilots.sh is active; finish/clean it up by recorded PID first")
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(CONTROL / "dispatcher.pid.json", {"pid": os.getpid(), "started": time.time()})
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: STOP.set())
        active, results = {}, []
        pool = concurrent.futures.ThreadPoolExecutor(len(slots))
        try:
            while (jobs or active) and not STOP.is_set():
                for name, future in list(active.items()):
                    if future.done():
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            results.append({"slot": name, "state": "dispatcher error", "error": str(exc)})
                            STOP.set()
                        del active[name]
                        atomic_json(CONTROL / "results.json", results)
                if STOP.is_set():
                    break
                rows = SCH.load()
                validate_slots(slots, rows)
                probe = SCH.probe()
                blocked = {}
                for slot in slots:
                    if not jobs or slot["name"] in active:
                        continue
                    ok, why = capacity(slots, set(active) | {slot["name"]}, rows, probe)
                    if not ok:
                        blocked[slot["name"]] = why
                        continue
                    if not port_block_free(slot):
                        blocked[slot["name"]] = "ports occupied"
                        continue
                    arm = jobs.pop(0)
                    active[slot["name"]] = pool.submit(pilot, arm, slot)
                atomic_json(CONTROL / "status.json", {"time": time.time(), "waiting": jobs, "active": list(active),
                                                       "blocked": blocked, "probe": probe, "results": results})
                STOP.wait(30 if jobs or active else 0)
        finally:
            STOP.set()
            pool.shutdown(wait=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
