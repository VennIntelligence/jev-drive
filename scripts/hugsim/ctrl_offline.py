#!/usr/bin/env python
"""HUGSIM actuation without rendering: the logged-trajectory plan (jevdrive.hugsim_preset) through variants of
traj2control + nuPlan iLQR + the env's kinematic bicycle, to find what makes the controller miss the plan
(todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md, diagnosis).

    cd $DATA_DIR/third_party/HUGSIM-zs/fixed && $DATA_DIR/envs/hugsim/bin/python \
        $DATA_DIR/jev-drive/scripts/hugsim/ctrl_offline.py --scenarios <list.txt> --out <dir> [--workers 8]

Variants (name: reference heading, iLQR discretization, iLQR wall-clock cap):
  official    transposed arctan2 (upstream),        0.5 s, 0.05 s   = the official controller
  fixed       PR #57 arctan2(d_right, d_forward),   0.5 s, 0.05 s   = the fixed controller
  fixed-T     PR #57,                               0.5 s, none     (only the 50 ms cap removed)
  fixed-dt    PR #57, plan resampled to 0.25 s,     0.25 s, none    (tracker step = simulator step, issue #75)
  central     central-difference tangent,           0.5 s, none
  central-dt  central-difference tangent, 0.25 s,   0.25 s, none
  fixed-dt-sr fixed-dt with the steering-rate limit 0.4 -> 1.0 rad/s
  fixed-dt-h  fixed-dt with the heading state cost 10 -> 30
  fixed-dt-xy fixed-dt with the position state costs 1 -> 3
  fixed-dt-u  fixed-dt with the steering-rate input cost 10 -> 1
  fixed-dt-xyu  fixed-dt-xy and fixed-dt-u together
Each step: the plan from the ego's state, the variant's (acc, steer rate), one 0.25 s step of hug_sim's bicycle.
The start state is the scenario's (start_ab, start_euler, start_velo, start_steer); no collisions, the run ends at
RC >= 1 (nearest logged pose past 90 %), 10 m off the log, or 400 steps. Writes <out>/<variant>/<scenario>/zs_steps.jsonl
in the preset agent's format (so scripts/hugsim/preset_eval.py-style metrics apply) and <out>/offline.csv.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())                                  # a HUGSIM tree (for sim.ilqr)
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "hugsim")]
D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
DT, L = 0.25, 2.7
VARIANTS = {"official": ("official", 0.5, 0.05), "fixed": ("fixed", 0.5, 0.05), "fixed-T": ("fixed", 0.5, None),
            "fixed-dt": ("fixed", 0.25, None), "central": ("central", 0.5, None), "central-dt": ("central", 0.25, None),
            "fixed-dt-sr": ("fixed", 0.25, None, {"max_steering_angle_rate": 1.0}),
            "fixed-dt-h": ("fixed", 0.25, None, {"state_cost_diagonal_entries": [1.0, 1.0, 30.0, 0.0, 0.0]}),
            "fixed-dt-xy": ("fixed", 0.25, None, {"state_cost_diagonal_entries": [3.0, 3.0, 10.0, 0.0, 0.0]}),
            "fixed-dt-u": ("fixed", 0.25, None, {"input_cost_diagonal_entries": [1.0, 1.0]}),
            "fixed-dt-xyu": ("fixed", 0.25, None, {"state_cost_diagonal_entries": [3.0, 3.0, 10.0, 0.0, 0.0],
                                                   "input_cost_diagonal_entries": [1.0, 1.0]})}


def solver(dt, cap, extra=None):
    import sim.ilqr.lqr as base
    from sim.ilqr.lqr_solver import ILQRSolver, ILQRSolverParameters
    from dataclasses import asdict
    p = asdict(base.solver_params)
    p.update(discretization_time=dt, max_solve_time=cap, **(extra or {}))
    return ILQRSolver(solver_params=ILQRSolverParameters(**p), warm_start_params=base.warm_start_params)


def reference(plan, heading, dt):
    """(N+1, 5) iLQR reference [forward, right, heading, 0, 0] from a plan (x right, y forward, 0.5 s spacing)."""
    p = np.r_[[[0.0, 0.0]], np.asarray(plan, np.float64)]
    if dt != 0.5:                                                 # resample the plan to the tracker's step
        t = 0.5 * np.arange(len(p))
        tq = np.arange(0, t[-1] + 1e-9, dt)
        p = np.stack([np.interp(tq, t, p[:, k]) for k in (0, 1)], -1)
    st = np.zeros((len(p), 5))
    st[:, 0], st[:, 1] = p[:, 1], p[:, 0]
    if heading == "central":
        g = np.gradient(p, axis=0)
        st[1:, 2] = np.arctan2(g[1:, 0], g[1:, 1])
    else:
        d = np.diff(p, axis=0)
        a, b = (d[:, 1], d[:, 0]) if heading == "fixed" else (d[:, 0], d[:, 1])
        rot = np.arctan2(b, a)
        rot = np.where(rot > np.pi / 2, rot - np.pi, rot)
        st[1:, 2] = np.where(rot < -np.pi / 2, rot + np.pi, rot)
    return st


def run(args):
    scen, variant, out = args
    import yaml
    from jevdrive import hugsim_zs as Z
    from jevdrive.hugsim_preset import LoggedPlan
    heading, dt, cap, *extra = VARIANTS[variant]
    cfg = yaml.safe_load(open(scen))
    ds = Path(scen).parent.name
    src = LoggedPlan(D / "datasets" / "hugsim" / "scenes" / ds / str(cfg["scene_name"]))
    lqr = solver(dt, cap, *extra)
    a, b = map(float, cfg["start_ab"])
    th = math.radians(float(cfg["start_euler"][1]))
    v, steer = float(cfg["start_velo"]), float(cfg["start_steer"])
    d = Path(out) / variant / Path(scen).stem
    d.mkdir(parents=True, exist_ok=True)
    n_log = len(src.s)
    solve_ms, end = [], "max_steps"
    with open(d / "zs_steps.jsonl", "w") as f:
        for k in range(400):
            info = {"ego_pos": [a, 0.0, b], "ego_rot": [0.0, th, 0.0], "ego_velo": v, "ego_steer": steer,
                    "timestamp": k * DT}
            plan, meta = src(info)
            pos, th_now = Z.ego_pose2d(info)
            f.write(json.dumps(dict(step=k, t=k * DT, pos=np.round(pos, 4).tolist(), theta=round(th_now, 6), v=round(v, 4),
                                    steer=round(steer, 5), plan=np.round(plan, 4).tolist(), **meta)) + "\n")
            i_near = int(np.argmin(np.linalg.norm(src.xz - pos, axis=1)))
            if k and (i_near + 1) / (0.9 * n_log) >= 1:
                end = "complete"
                break
            if abs(meta["log_xt"]) > 10:
                end = "off_route"
                break
            t0 = time.perf_counter()
            sol = lqr.solve(np.array([0.0, 0.0, 0.0, v, steer]), reference(plan, heading, dt))
            solve_ms.append(1e3 * (time.perf_counter() - t0))
            acc, sr = sol[-1].input_trajectory[0]
            v += acc * DT                                          # hug_sim.HUGSimEnv.step, verbatim
            steer += sr * DT
            a += v * math.sin(th) * DT
            b += v * math.cos(th) * DT
            th += v * math.tan(steer) / L * DT
    return dict(scenario=Path(scen).stem, variant=variant, end=end, steps=k + 1,
                solve_ms_med=float(np.median(solve_ms)), solve_capped=float(np.mean(np.array(solve_ms) >= 50)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true", help="re-summarize an existing <out>/offline.csv")
    a = ap.parse_args()
    if a.summary_only:
        import csv
        rows = [dict(r, solve_ms_med=float(r["solve_ms_med"]), solve_capped=float(r["solve_capped"]))
                for r in csv.DictReader(open(Path(a.out) / "offline.csv"))]
        return summarize(Path(a.out), rows)
    scen = [s if s.startswith("/") else str(D / "datasets" / "hugsim" / "scenarios" / s) for s in open(a.scenarios).read().split()]
    jobs = [(s, v, a.out) for v in a.variants.split(",") for s in scen]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    import csv
    with open(Path(a.out) / "offline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(Path(a.out), rows)


def summarize(out, rows):
    """Per variant, pooled over scenarios: the preset_eval tracking metrics -> <out>/offline_summary.csv."""
    import csv
    from preset_eval import q, run_steps
    res = []
    for v in dict.fromkeys(r["variant"] for r in rows):
        R = [r for r in rows if r["variant"] == v]
        S = [s for r in R for s in run_steps(out / v / r["scenario"] / "zs_steps.jsonl")]
        m = {"variant": v, "n_runs": len(R), "n_steps": len(S), "complete": sum(r["end"] == "complete" for r in R)}
        for k in ("lat50", "lon50", "lat25"):
            m[f"{k}_med"], m[f"{k}_p95"] = q([s.get(k) for s in S], 50), q([s.get(k) for s in S], 95)
        m["hd50_med_deg"], m["hd50_p95_deg"] = (float(np.degrees(q([s.get("hd50") for s in S], p))) for p in (50, 95))
        m["xt_med"], m["xt_p95"], m["xt_max"] = (q([s["log_xt"] for s in S], p) for p in (50, 95, 100))
        m["xt_run_med_max"] = max(q([s["log_xt"] for s in run_steps(out / v / r["scenario"] / "zs_steps.jsonl")], 50)
                                  for r in R)
        m["solve_ms_med"] = float(np.median([r["solve_ms_med"] for r in R]))
        m["solve_capped"] = float(np.mean([r["solve_capped"] for r in R]))
        res.append({k: round(x, 4) if isinstance(x, float) else x for k, x in m.items()})
    with open(out / "offline_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    for m in res:
        print(m)


if __name__ == "__main__":
    main()
