"""Self-advancing master chain script for VLM Arbitration Experiment.

Protocol: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md.
Orchestrates:
  1. OpenJev server lifecycle (vLLM with DiffusionGemma-26B NVFP4 on GPU)
  2. Single-unit verification checklist (Stage 1)
  3. Phase A shadow collection and multi-model evaluation
  4. Stage 2 small-batch rule checklist
  5. Phase B closed-loop evaluation (drive, dslow, jslow, vred, vbyp, vall)
  6. Bootstrap statistics, reports, and plots
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional

REPO = Path(__file__).resolve().parents[3]
for _d in (str(REPO / "lib"), str(REPO)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN_ROOT = DATA_DIR / "runs/vlm_arb"
XML_PATH = DATA_DIR / "third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml"

# Route cohorts
DEV_ROUTES = ["27043", "15102", "24944", "27870", "22535", "37969", "24497", "27297", "9196", "28147"]
HELD_ROUTES = ["26828", "25783", "24416", "469", "25215", "27392", "27005", "25613", "27907", "26023",
               "24207", "26723", "24519", "25753", "24948", "26365", "27916", "24622", "26370"]
DEBUG_ROUTES = ["334", "27787", "24721", "26872", "26537", "17749", "25169", "24955"]

JTYPES = ["NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionLeftTurnEnterFlow", "NonSignalizedJunctionRightTurn",
          "OppositeVehicleTakingPriority", "SignalizedJunctionLeftTurn", "SignalizedJunctionLeftTurnEnterFlow",
          "T_Junction", "MergerIntoSlowTrafficV2", "SignalizedJunctionRightTurn", "VehicleTurningRoute"]

BTYPES = ["Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
          "ParkedObstacle", "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays"]


def update_status(msg: str):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = RUN_ROOT / "STATUS.tmp"
    ts = time.strftime("%F %T")
    tmp.write_text(f"{ts} {msg}\n")
    tmp.replace(RUN_ROOT / "STATUS")
    print(f"[{ts}] STATUS: {msg}", flush=True)


def ensure_openjev_server(gpu_card: int = 0, port: int = 8080, vllm_port: int = 8000) -> Optional[subprocess.Popen]:
    """Ensure vLLM and OpenJev decision servers are running."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
            if resp.status == 200:
                print(f"OpenJev server is already healthy on port {port}.")
                return None
    except Exception:
        pass

    print(f"Starting OpenJev server on GPU {gpu_card} (vLLM:{vllm_port}, API:{port})...")
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "openjev_server.log"
    
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_card),
        "HF_HUB_OFFLINE": "1",
        "VLLM_CACHE_ROOT": str(DATA_DIR / "cache/vllm"),
        "OPENJEV_GPU_UTIL": "0.35", # 0.35 of 83.6 GiB is ~29.2 GB, plenty of room for weights and KV cache
        "PATH": f"{DATA_DIR}/envs/openjev/bin:{env.get('PATH', '')}"
    })
    
    # Launch OpenJev launcher script or entrypoint
    launcher = REPO / "experiments/vlm_arb/scripts/launch_openjev.sh"
    with server_log.open("a") as out:
        proc = subprocess.Popen(["bash", str(launcher)], cwd=str(REPO), env=env, stdout=out, stderr=subprocess.STDOUT)
    
    # Wait for server to become healthy
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    print(f"OpenJev server ready after {time.time() - t0:.1f}s.")
                    return proc
        except Exception:
            time.sleep(3.0)

    raise RuntimeError("OpenJev server failed to become ready within 180s. See logs in " + str(server_log))


def load_route_manifest() -> Dict[str, List[str]]:
    """Build the 58 evaluation routes following registered manifest."""
    root = ET.parse(XML_PATH).getroot()
    rows = [dict(id=r.get("id"), town=r.get("town"), type=next(r.iter("scenario")).get("type")) for r in root.iter("route")]
    banned = set(DEV_ROUTES + HELD_ROUTES + DEBUG_ROUTES)
    
    manifest = {"dev": DEV_ROUTES, "junction": [], "obstacle": []}
    for t in JTYPES:
        matching = [r["id"] for r in rows if r["type"] == t and r["id"] not in banned][:3]
        manifest["junction"].extend(matching)
    for t in BTYPES:
        matching = [r["id"] for r in rows if r["type"] == t and r["id"] not in banned][:3]
        manifest["obstacle"].extend(matching)

    assert len(manifest["junction"]) == 24
    assert len(manifest["obstacle"]) == 24
    return manifest


def run_unit(arm: str, seed: int, routes: List[str], tag: str, gpu: int = 0, cpus: str = "0-24", workers: int = 2,
             shadow: bool = False, save_frames: bool = False) -> Path:
    """Execute a single unit of routes under specified arm and parameters."""
    unit_name = f"{tag}-{arm}-s{seed}"
    out_dir = RUN_ROOT / "arms" / unit_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment for b2d execution
    env = dict(os.environ)
    env.update({
        "GPU": str(gpu),
        "CPUS": cpus,
        "WORKERS": str(workers),
        "ARMS": arm,
        "SEEDS": str(seed),
        "VLM_ARM": arm,
        "VLM_SHADOW": "1" if shadow else "0",
        "VLM_SAVE_FRAMES": "1" if save_frames else "0",
        "OP_ARB_DIR": str(RUN_ROOT / f"card{gpu}"),
        "OP_ARB_ARMS": str(RUN_ROOT / "arms"),
        "OP_ARB_AGENT": "lib/vlm_arb_agent.py",
        "B2D_PIDS_WAIT": "17000",
        "OPENBLAS_CORETYPE": "Haswell",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1"
    })

    unit_log = RUN_ROOT / f"unit-{unit_name}.log"
    with unit_log.open("a") as log_f:
        cmd = [
            "bash", "experiments/op_closed_loop/archive/op_arb.sh", "arm", arm,
            ",".join(routes), str(out_dir)
        ]
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        rc = proc.wait()

    assert rc == 0, f"Unit {unit_name} failed with return code {rc}. Check {unit_log}"
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Master chain for VLM arbitration")
    parser.add_argument("--skip-server", action="store_true", help="Do not manage OpenJev server")
    parser.add_argument("--gpus", type=str, default="0,1,2", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_route_manifest()

    try:
        # Step 0: Ensure OpenJev server
        if not args.skip_server:
            update_status("Starting OpenJev System One server...")
            ensure_openjev_server(gpu_card=0)

        # Step 1: Stage 1 Checklist (Pilot)
        pilot_done = RUN_ROOT / "DONE-pilot"
        if not pilot_done.exists():
            update_status("Stage 1: Running single-unit pilot checklist on route 334...")
            pilot_dir = run_unit(arm="vred", seed=0, routes=["334"], tag="pilot", workers=1)
            
            # Verify checklist
            from vlm_arb_checks import verify_attempt
            chk = verify_attempt(pilot_dir / "attempts/334/1", arm="vred")
            assert chk["passed"], f"Pilot checklist failed: {chk}"
            pilot_done.write_text(time.strftime("%F %T\n"))
            update_status("Stage 1 Pilot passed successfully.")

        # Step 2: Phase A (Shadow Frame Collection & Multi-Model Evaluation)
        phase_a_done = RUN_ROOT / "DONE-phase-a"
        if not phase_a_done.exists():
            update_status("Phase A: Collecting shadow frames on dev routes...")
            shadow_dir = run_unit(arm="drive", seed=0, routes=DEV_ROUTES, tag="shadow",
                                  workers=2, shadow=True, save_frames=True)
            
            # Multi-model evaluation
            update_status("Phase A: Running multi-model evaluation on collected frames...")
            frames_dir = shadow_dir / "attempts" / DEV_ROUTES[0] / "1/vlm_frames"
            vlm_log = shadow_dir / "attempts" / DEV_ROUTES[0] / "1/vlm_decisions.jsonl"
            
            subprocess.run([
                f"{DATA_DIR}/envs/jevdrive/bin/python",
                "experiments/vlm_arb/scripts/vlm_arb_eval_models.py",
                "--frames-dir", str(frames_dir),
                "--vlm-log", str(vlm_log),
                "--out-dir", str(RUN_ROOT / "results")
            ], cwd=str(REPO), check=True)
            
            phase_a_done.write_text(time.strftime("%F %T\n"))
            update_status("Phase A completed. Multi-model metrics recorded.")

        # Step 3: Phase B Closed-Loop Evaluation
        phase_b_done = RUN_ROOT / "DONE-phase-b"
        if not phase_b_done.exists():
            all_eval_routes = sorted(list(set(manifest["dev"] + manifest["junction"] + manifest["obstacle"])))
            arms = ["drive", "dslow", "jslow", "vred", "vbyp", "vall"]
            
            for arm in arms:
                for seed in (0, 1):
                    update_status(f"Phase B: Running arm {arm} seed {seed} ({len(all_eval_routes)} routes)...")
                    run_unit(arm=arm, seed=seed, routes=all_eval_routes, tag="eval", workers=4)

            phase_b_done.write_text(time.strftime("%F %T\n"))
            update_status("Phase B completed.")

        # Step 4: Final Reporting and Plots
        update_status("Generating final reports and statistical tables...")
        subprocess.run([
            f"{DATA_DIR}/envs/jevdrive/bin/python",
            "experiments/vlm_arb/scripts/vlm_arb_report.py",
            "--run-dir", str(RUN_ROOT),
            "--out-dir", str(RUN_ROOT / "results")
        ], cwd=str(REPO), check=True)

        (RUN_ROOT / "DONE").write_text(time.strftime("%F %T\n"))
        update_status("All stages finished successfully.")

    except Exception as e:
        err_msg = traceback.format_exc()
        (RUN_ROOT / "ERROR").write_text(err_msg)
        update_status(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
