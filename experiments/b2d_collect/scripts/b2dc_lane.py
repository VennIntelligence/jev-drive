#!/usr/bin/env python
"""The b2d_collect lane (plans/2026-10-06-b2d-collect-prereg.md): one chained driver per stage, staged 1 -> ~10 -> all (docs/long-runs.md).
It only submits pool jobs, waits and checks; it runs off the GPU (tmux jev via scripts/tmux_run.sh, or a --vram 0.5 pool job).

  .venv/bin/python experiments/b2d_collect/scripts/b2dc_lane.py --stage smoke|ten|all [--jobs J --workers W]

Per stage, under $DATA_DIR/runs/b2d_collect/:
  data/<stage>/              b2d_run output (attempts/<rid>/<k>/clip/...), index.csv / index.json (labels), check/ (validation)
  lane/<stage>/{STATUS, DONE, ERROR, jobs.txt}   one current line; DONE = gates passed (JSON); ERROR = which gate / job failed
Jobs: collect (CARLA, carla = W workers per job, J jobs sharing one route list through b2d_run's claims) -> labels (CPU) -> check (GPU).
Rerunning a stage resumes: b2d_run skips finished routes, labels skip labelled clips, finished pool jobs are not resubmitted.

Gates (all stages; a failed gate writes ERROR and stops):
  complete     clips with DONE / routes >= 0.9 (smoke: 1 / 1)
  integrity    picture index = tick, consecutive frames, |dt - 0.05| < 1e-4, frozen pictures < 1 %, camera frame = state frame
  alignment    at the launch from the spawn the ground starts moving in the picture within 1 tick of the logged launch on >= 80 % of clips; shipped Cinque's heading
               sign agrees with the log on lane following >= 0.8 (when >= 10 samples; junction turns are the model's open problem, reported)
  labels       logged future footprint inside the drivable SDF (>= -0.3 m) >= 0.9; turn commands (left / right) precede a driven turn of that side >= 0.8
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.cl import pool as P  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
ROOT = D / "runs" / "b2d_collect"
ROUTES = ROOT / "routes" / ("v1" if "--stage" in sys.argv and sys.argv[sys.argv.index("--stage") + 1] in ("smoke", "ten") else "v2")
S = REPO / "experiments" / "b2d_collect" / "scripts"
SIM = D / "third_party" / "simlingo"
PY_SIM = str(D / "envs" / "simlingo" / "bin" / "python")
PY_TRAIN = str(D / "envs" / "op-train" / "bin" / "python")
TEN_TYPES = ["VanillaSignalizedTurnEncounterGreenLight", "NonSignalizedJunctionLeftTurnEnterFlow", "T_Junction", "VehicleTurningRoutePedestrian",
             "SignalizedJunctionRightTurn", "Accident", "ParkingExit", "DynamicObjectCrossing", "MergerIntoSlowTraffic", "SequentialLaneChange"]


def manifest():
    with open(ROUTES / "manifest.csv") as fh:
        return list(csv.DictReader(fh))


def pick(stage):
    rows = manifest()
    if stage == "smoke":                                     # one junction turn, Town12 (the heaviest map, half of the set)
        c = [r for r in rows if r["src"] == "JT" and r["turn"] == "left" and r["town"] == "Town12"]
        return [c[0]["route_id"]]
    if stage == "ten":
        out, towns = [], set()
        for t in TEN_TYPES:
            c = [r for r in rows if r["type"] == t and r["route_id"] not in out]
            c.sort(key=lambda r: (r["town"] in towns, r["route_id"]))  # prefer a town not yet used
            if c:
                out.append(c[0]["route_id"])
                towns.add(c[0]["town"])
        return out
    return [r["route_id"] for r in rows]                     # v2 ids (920000+); the smoke / ten clips (v1) are a separate check set


class Lane:
    def __init__(self, stage):
        self.stage = stage
        self.dir = ROOT / "lane" / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in ("DONE", "ERROR"):
            (self.dir / f).unlink(missing_ok=True)

    def status(self, msg):
        line = f"{time.strftime('%F %T')} b2d_collect {self.stage}: {msg}"
        print(line, flush=True)
        (self.dir / "STATUS").write_text(line + "\n")

    def die(self, msg):
        self.status("ERROR " + msg)
        (self.dir / "ERROR").write_text(msg + "\n")
        sys.exit(1)

    def sub(self, cmd, name, log_dir, **kw):
        log_dir = Path(log_dir)
        if (log_dir / "DONE").exists():
            return None
        (log_dir / "ERROR").unlink(missing_ok=True)
        jid = P.submit(cmd, name=name, owner="b2d_collect", log_dir=str(log_dir), **kw)
        with open(self.dir / "jobs.txt", "a") as fh:
            fh.write(f"{jid} {name}\n")
        return jid

    def wait(self, ids, what):
        ids = [i for i in ids if i]
        if not ids:
            return
        st = P.wait(ids, poll_s=60, on_poll=lambda s: self.status(f"{what}: " + ", ".join(f"{k[-4:]}={v}" for k, v in s.items())))
        bad = {k: v for k, v in st.items() if v != "done"}
        if bad:
            self.die(f"{what}: jobs not done: {bad}")


def gates(stage, data: Path, n_routes: int) -> dict:
    ij = json.loads((data / "index.json").read_text())
    ck = json.loads((data / "check" / "check.json").read_text())
    g = {"complete": ij["clips"] / max(1, n_routes) >= (1.0 if stage == "smoke" else 0.9),
         "integrity": ck["vid_is_tick"] and ck["frames_consecutive"] and ck["dt_max_err_max"] < 1e-4 and ck["frozen_share_max"] < 0.01}
    if ck.get("sensor_frame_eq_all") is not None:
        g["sensor_sync"] = bool(ck["sensor_frame_eq_all"])
    if ck.get("first_launch_lag0_share") is not None:
        g["launch_lag"] = ck["first_launch_lag_le1_share"] >= 0.8
    if (ck.get("lane_samples") or 0) >= 10 and ck.get("lane_sign_agree_P0") is not None:
        g["lane_sign"] = ck["lane_sign_agree_P0"] >= 0.8
    g["footprint_drivable"] = not (ck["fut_footprint_drivable_mean"] < 0.9)
    g["cmd_precision"] = not (ck["cmd_turn_precision_mean"] < 0.8)
    return {"gates": g, "index": ij, "check": ck}


def main(a):
    L = Lane(a.stage)
    data = ROOT / "data" / a.stage
    data.mkdir(parents=True, exist_ok=True)
    ids = pick(a.stage)
    (L.dir / "routes.txt").write_text(",".join(ids) + "\n")
    L.status(f"{len(ids)} routes")
    cfg = ROOT / "agent_config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({"crf": a.crf, "chase": True}, indent=1))
    env = {"BENCH2DRIVE_ROOT": str(SIM / "Bench2Drive"), "WORK_DIR": str(SIM), "IS_BENCH2DRIVE": "1"}
    jobs = 1 if a.stage == "smoke" else a.jobs
    W = 1 if a.stage == "smoke" else min(a.workers, max(1, len(ids) // jobs))
    pool_dir = ROOT / "pool" / a.stage
    cids = []
    for j in range(jobs):
        cmd = P.b2d_cmd(data, ids, routes=str(ROUTES / "routes.xml"), agent=str(S / "b2dc_agent.py"), agent_config=f"{cfg}+{a.stage}",
                        python=PY_SIM, max_attempts=2, stall_s=480, route_timeout_s=3600, extra=["--tm-seed-from-id"])
        cids.append(L.sub(cmd, f"b2dc-{a.stage}-c{j}", pool_dir / f"collect{j}", carla=W, vram_gb=a.vram_per_worker * W,
                          cpu=a.cpu_per_worker * W, ram_gb=6 * W, env=env, timeout_h=a.timeout_h))
    L.status(f"collect: {jobs} job(s) x {W} workers")
    L.wait(cids, "collect")
    lid = L.sub([PY_SIM, str(S / "b2dc_labels.py"), "--data", str(data)], f"b2dc-{a.stage}-labels", pool_dir / "labels", vram_gb=0.5,
                cpu=min(24, max(2, len(ids))), ram_gb=24)
    L.wait([lid], "labels")
    kid = L.sub([PY_TRAIN, str(S / "b2dc_check.py"), "--data", str(data), "--gif", str(a.gif)], f"b2dc-{a.stage}-check", pool_dir / "check",
                vram_gb=12, cpu=8, ram_gb=16)
    L.wait([kid], "check")
    res = gates(a.stage, data, len(ids))
    (L.dir / "gates.json").write_text(json.dumps(res, indent=1))
    failed = [k for k, v in res["gates"].items() if not v]
    if failed:
        L.die(f"gates failed: {failed} (lane/{a.stage}/gates.json)")
    L.status(f"done: {res['index']['clips']} clips, gates {res['gates']}")
    (L.dir / "DONE").write_text(json.dumps(res["gates"]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "ten", "all"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--vram-per-worker", type=float, default=9.0)
    ap.add_argument("--cpu-per-worker", type=int, default=3)
    ap.add_argument("--timeout-h", type=float, default=24.0)
    ap.add_argument("--crf", type=int, default=0)
    ap.add_argument("--gif", type=int, default=4)
    main(ap.parse_args())
