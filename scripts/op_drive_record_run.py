"""Rerun seven seed-0 op-drive failures with one CARLA at a time on GPU 5."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["DATA_DIR"])
OUT = DATA / "runs/op_drive_record_seed0"
ROUTES = ["27870", "27043", "27297", "9196", "24944", "37969", "24497"]
PY_OP = DATA / "envs/openpilot/bin/python"
PY_CARLA = DATA / "envs/carla/bin/python"
SOCKET = OUT / "srv/op.sock"


def event(kind, **fields):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "kind": kind, **fields}) + "\n")
    print(time.strftime("%F %T"), kind, fields, flush=True)


def run_route(route):
    run = OUT / "routes" / route
    event("route_start", route=route)
    (OUT / "STATUS").write_text("running route " + route)
    command = [str(PY_CARLA), "scripts/b2d_run.py", "--routes",
               str(DATA / "third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml"),
               "--route-ids", route, "--towns", "all", "--workers", "1", "--server-index", "20",
               "--index-span", "1", "--gpu-rank", "5", "--tm-seed", "0", "--no-spectator",
               "--no-reap", "--client-threads", "4", "--max-attempts", "2", "--stall-s", "480",
               "--route-timeout-s", "2400", "--out", str(run), "--python", str(PY_CARLA),
               "--agent", "scripts/op_drive_record_agent.py", "--agent-config", str(OUT / "config.json"),
               "--record-dir", str(OUT / "recorders"), "--fast-copy", "--cache-lights"]
    run.mkdir(parents=True, exist_ok=True)
    with (run / "runner.log").open("a") as log:
        subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    done = json.loads((run / "done" / (route + ".json")).read_text())
    attempt = run / "attempts" / route / str(done["attempt"])
    subprocess.run([str(PY_OP), "scripts/op_drive_record_video.py", "annotate", str(attempt)],
                   cwd=str(ROOT), check=True)
    recorders = list((attempt / "recorder").glob("*.log"))
    if not recorders or not all(p.stat().st_size > 0 for p in recorders):
        raise RuntimeError("Missing CARLA recorder for " + route)
    result = json.loads((attempt / "results.json").read_text())["_checkpoint"]["records"][0]
    qa = json.loads((attempt / "video_qa.json").read_text())
    event("route_complete", route=route, attempt=str(attempt), scores=result["scores"],
          status=result["status"], video=qa)
    (OUT / (route + ".done")).write_text(json.dumps({"attempt": str(attempt), "qa": qa}))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "srv").mkdir(exist_ok=True)
    (OUT / "tb").mkdir(exist_ok=True)
    source = DATA / "runs/op_drive/arms/dv-drive-s0/attempts/24497/1/agent_summary.json"
    config = json.loads(source.read_text())["config"]
    config["socket"] = str(SOCKET)
    config["controller_config"] = str(ROOT / "todos/2026-09-23-tfv6-controller/controller-eval/P7.json")
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    (OUT / "manifest.json").write_text(json.dumps({"routes": ROUTES, "seed": 0, "gpu": 5,
        "workers": 1, "source_config": str(source), "config": config,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
        "recording": {"width": 1280, "height": 720, "fps": 10},
        "purpose": "diagnostic rerun; new outcomes are not old-run replay"}, indent=2))
    ready = OUT / "srv/ready"
    ready.unlink(missing_ok=True)
    with (OUT / "srv/log.txt").open("a") as log:
        server = subprocess.Popen([str(PY_OP), "scripts/op_arb_server.py", "cinque", "--pool", "1",
            "--backend", "cuda-iob", "--socket", str(SOCKET), "--ready-file", str(ready)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        (OUT / "srv/server.pid").write_text(str(server.pid))
        try:
            deadline = time.monotonic() + 900
            while not ready.exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("OpenPilot server failed to become ready")
                time.sleep(2)
            for index, route in enumerate(ROUTES):
                if not (OUT / (route + ".done")).exists():
                    run_route(route)
                if index == 0 and not (OUT / "PILOT_REVIEWED").exists():
                    event("pilot_ready", route=route)
                    (OUT / "STATUS").write_text("pilot recorded; waiting for visual review")
                    (OUT / "PILOT_READY").touch()
                    while not (OUT / "PILOT_REVIEWED").exists():
                        time.sleep(5)
                if server.poll() is not None:
                    raise RuntimeError("OpenPilot server exited")
            (OUT / "DONE").touch()
            (OUT / "STATUS").write_text("complete: seven routes recorded")
            event("complete", routes=ROUTES)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    subprocess.run([sys.executable, "scripts/sch_table.py", "finish", "op-drive-record",
                    "seven seed-0 diagnostic recordings complete"], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ERROR").write_text(repr(exc))
        (OUT / "STATUS").write_text("error: " + repr(exc))
        event("error", error=repr(exc))
        raise
