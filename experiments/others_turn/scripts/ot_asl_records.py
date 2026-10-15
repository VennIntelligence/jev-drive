#!/usr/bin/env python3
"""OTHERS-TURN: driver records from rollout.asl for drivers that log no per-decision anchor (the shipped LTF / DiffusionDrive samples, the
py123d-garage driver), so rf_follow.py can read their turns (plans/2026-10-15-others-turn-prereg.md). CPU only, AlpaSim's venv, read-only on the runs.

  ot_asl_records.py view  --run <run dir> --out <view dir>     -> <view dir>: symlinks to every entry of the run except driver-logs, plus
                                                                   driver-logs/drive.jsonl with one start + one drive record per decision
  ot_asl_records.py check --run <run dir with its own drive.jsonl> --out <json>   G1: asl records against the driver's own records

Per rollout.asl: the decisions are the driver_request messages in order; t0 of decision k = the latest pose of the driver's ego trajectory
(driver_ego_trajectory, the poses the driver itself receives, local frame) at or before time_now; anchor = that pose (x, y, yaw); ego = the
last dynamic state's linear velocity / acceleration (vx, vy, ax, ay: wajepa_driver.py's layout); poses = the returned trajectory at
t0 + 0.5 .. 4.0 s in the anchor's frame (x, y, yaw), used by rf_follow only for its plan-deficit column.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_ / "experiments/alpasim/scripts")]


def records(f: str) -> tuple[str, list[dict]]:
    from c1_extract import _read, traj

    async def go():
        ego, reqs, sess = {}, [], None
        async for k, m in _read(f, {"rollout_metadata", "driver_request", "driver_return", "driver_ego_trajectory"}):
            if k == "rollout_metadata":
                sess = m.session_metadata.session_uuid
            elif k == "driver_ego_trajectory":
                d = m.dynamic_states[-1] if len(m.dynamic_states) else None
                for row in traj(m.trajectory):
                    ego[int(row[0])] = (row[1:], d and [d.linear_velocity.x, d.linear_velocity.y, d.linear_acceleration.x, d.linear_acceleration.y])
            elif k == "driver_request":
                ts = [t for t in ego if t <= int(m.time_now_us)]
                reqs.append(dict(now=int(m.time_now_us), t0=max(ts) if ts else None, ego=ego[max(ts)][1] if ts else None))
            elif k == "driver_return" and reqs:
                reqs[-1]["traj"] = traj(m.trajectory)
        return sess, ego, reqs
    sess, ego, reqs = asyncio.run(go())
    out = []
    for k, r in enumerate(reqs):
        if r["t0"] is None:
            continue
        x, y, yaw = ego[r["t0"]][0]
        rec = dict(kind="drive", session=sess, now=r["now"], t0=r["t0"], k=k, anchor=[float(x), float(y), float(yaw)],
                   ego=[float(v) for v in (r["ego"] or [np.nan] * 4)], src="asl")
        T = r.get("traj")
        if T is not None and len(T) > 1:
            tq = r["t0"] + 500_000 * np.arange(1, 9)
            ok = tq <= T[-1, 0]
            c, s = np.cos(yaw), np.sin(yaw)
            dx, dy = np.interp(tq[ok], T[:, 0], T[:, 1]) - x, np.interp(tq[ok], T[:, 0], T[:, 2]) - y
            rec["poses"] = np.c_[c * dx + s * dy, -s * dx + c * dy, np.interp(tq[ok], T[:, 0], np.unwrap(T[:, 3])) - yaw].round(4).tolist()
        out.append(rec)
    return sess, out


def asl_files(run: Path) -> dict:
    """{scene: rollout.asl} (one rollout per scene in these runs)."""
    return {Path(f).parts[-3]: f for f in sorted(glob.glob(str(run / "rollouts/*/*/rollout.asl")))}


def cmd_view(a):
    from concurrent.futures import ProcessPoolExecutor
    run, out = Path(a.run).resolve(), Path(a.out)
    (out / "driver-logs").mkdir(parents=True, exist_ok=True)
    for e in run.iterdir():
        if e.name != "driver-logs" and not (out / e.name).exists():
            os.symlink(e, out / e.name)
    F = asl_files(run)
    with ProcessPoolExecutor(min(a.jobs, max(len(F), 1))) as ex:
        res = list(ex.map(records, F.values()))
    with open(out / "driver-logs/drive.jsonl", "w") as w:
        for n, (sc, (sess, recs)) in enumerate(zip(F, res)):
            w.write(json.dumps(dict(kind="start", session=sess, n=n, scene=sc)) + "\n")
            for r in recs:
                w.write(json.dumps(dict(r, scene=sc)) + "\n")
    print(f"{run}: {len(F)} rollouts, {sum(len(r[1]) for r in res)} decisions -> {out}")


def cmd_check(a):
    sys.path.insert(0, str(R_ / "experiments/anchor_offset/scripts"))
    from ao_e1 import drive_records
    run = Path(a.run)
    own, F, rows = drive_records(run), asl_files(run), []
    for sc, f in F.items():
        if sc not in own:
            continue
        _, recs = records(f)
        A = {r["t0"]: r for r in own[sc]}
        for r in recs:
            o = A.get(r["t0"])
            if o is None:
                rows.append(dict(scene=sc, k=r["k"], t0=r["t0"], matched=False))
                continue
            d = np.asarray(r["anchor"]) - np.asarray(o["anchor"])
            rows.append(dict(scene=sc, k=r["k"], t0=r["t0"], matched=True, dpos=float(np.hypot(d[0], d[1])),
                             dyaw_deg=float(np.degrees(abs((d[2] + np.pi) % (2 * np.pi) - np.pi))), k_own=o["k"]))
    m = [r for r in rows if r["matched"]]
    s = dict(scenes=len({r["scene"] for r in rows}), decisions=len(rows), matched=len(m), own_decisions=sum(len(own[sc]) for sc in F if sc in own),
             dpos_median=float(np.median([r["dpos"] for r in m])) if m else None, dpos_max=float(np.max([r["dpos"] for r in m])) if m else None,
             dyaw_median_deg=float(np.median([r["dyaw_deg"] for r in m])) if m else None,
             dyaw_max_deg=float(np.max([r["dyaw_deg"] for r in m])) if m else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=s, rows=rows), indent=1) + "\n")
    print(json.dumps(s))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["view", "check"])
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    {"view": cmd_view, "check": cmd_check}[a.cmd](a)
