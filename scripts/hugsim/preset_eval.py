#!/usr/bin/env python
"""Score the HUGSIM controller acceptance runs (todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md):
tracking error of each controller against the plan it was fed and against the logged path, end reasons, HD-Score vs
the ideal-tracker reference, and the pre-registered pass / fail per controller.

    python scripts/hugsim/preset_eval.py <out_dir> [--static list.txt] [--actor list.txt]

Reads <out_dir>/results.csv (scripts/hugsim/zs_run.py) and every run's zs_steps.jsonl (scripts/hugsim/preset_agent.py).
Writes <out_dir>/steps.csv (one row per step), runs.csv (one per run), summary.json (per controller, with verdicts).

Tracking error vs the fed plan, in the plan frame of step k (x right = lateral, y forward = longitudinal):
  h0.5  the plan's first waypoint (t = 0.5 s) vs the ego two steps later; heading vs the plan's tangent there,
        which for a quadratic through the origin and the first two waypoints is the direction of waypoint 2.
  h0.25 the plan's motion over one step (the same quadratic at t = 0.25 s, what the ideal tracker executes) vs the
        ego one step later; ~0 for the ideal tracker by construction (a check of this code).
Against the log: signed cross-track distance and heading minus the log's tangent, logged by the agent at every step.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REF = "ideal"
CRIT = {  # pre-registered thresholds (hugsim-controllers.md, section "通过标准")
    "lat_med": 0.10, "lat_p95": 0.30, "lon_med": 0.20, "lon_p95": 0.50, "hd_med_deg": 1.5, "hd_p95_deg": 5.0,
    "xt_run_med": 0.30, "xt_run_max": 1.0, "static_mean_hd_drop": 0.05, "static_scene_hd_drop": 0.15,
    "actor_hd_diff": 0.15,
}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_plan(p, pos, th):
    f, r = np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)])
    d = np.asarray(p) - pos
    return np.array([d @ r, d @ f])


def run_steps(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if "step" in r]
    pos = np.array([r["pos"] for r in recs], np.float64)
    th = np.unwrap(np.array([r["theta"] for r in recs], np.float64))
    rows = []
    for k, r in enumerate(recs):
        plan = np.asarray(r["plan"], np.float64)
        p1, p2 = plan[0], plan[1]
        qa, qb = 4 * p1 - p2, 2 * (p2 - 2 * p1)
        row = {"step": r["step"], "t": r["t"], "v": r["v"], "log_xt": r.get("log_xt"), "log_dth": r.get("log_dth"),
               "log_v": r.get("log_v"), "stop": int(bool(r.get("stop"))), "fwd_only": int("raw_plan" in r),
               "plan_len": float(np.linalg.norm(plan[-1]))}
        if k + 1 < len(recs):
            e = to_plan(pos[k + 1], pos[k], th[k]) - (0.25 * qa + 0.0625 * qb)
            v = qa + 0.5 * qb
            hp = np.arctan2(v[0], v[1]) if np.linalg.norm(v) > 1e-6 else 0.0
            row.update(lat25=e[0], lon25=e[1], hd25=float(wrap(th[k + 1] - th[k] - hp)))
        if k + 2 < len(recs):
            e = to_plan(pos[k + 2], pos[k], th[k]) - p1
            hp = np.arctan2(p2[0], p2[1]) if np.linalg.norm(p2) > 1e-6 else 0.0
            row.update(lat50=e[0], lon50=e[1], hd50=float(wrap(th[k + 2] - th[k] - hp)))
        rows.append(row)
    return rows


def q(x, p):
    x = np.abs(np.asarray([v for v in x if v is not None and v == v], np.float64))
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--static", default=None, help="scenario list of the static (no inserted actor) scenes")
    ap.add_argument("--actor", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    names = lambda f: {Path(l).stem for l in open(f).read().split()} if f else set()
    static, actor = names(a.static), names(a.actor)
    res = [r for r in csv.DictReader(open(out / "results.csv"))]
    latest = {}
    for r in res:                                              # a rerun supersedes an earlier row
        latest[(r["scenario"], r["controller"])] = r
    runs, steps = [], []
    for (scen, ctrl), r in sorted(latest.items()):
        f = Path(r["run_dir"]) / "zs_steps.jsonl"
        st = run_steps(f) if f.exists() else []
        for s in st:
            steps.append(dict(scenario=scen, controller=ctrl, **s))
        xt = [s["log_xt"] for s in st]
        runs.append(dict(scenario=scen, controller=ctrl, group="static" if scen in static else "actor" if scen in actor else "other",
                         end=r["end"], steps=r["steps"], hdscore=r["hdscore"], rc=r["rc"], nc=r["nc"], dac=r["dac"],
                         ttc=r["ttc"], c=r["c"], pdms=r["pdms"],
                         lat50_med=q([s.get("lat50") for s in st], 50), lon50_med=q([s.get("lon50") for s in st], 50),
                         hd50_med_deg=np.degrees(q([s.get("hd50") for s in st], 50)),
                         lat25_med=q([s.get("lat25") for s in st], 50),
                         xt_med=q(xt, 50), xt_max=q(xt, 100), dth_med_deg=np.degrees(q([s["log_dth"] for s in st], 50)),
                         n_stop=sum(s["stop"] for s in st), n_fwd_only=sum(s["fwd_only"] for s in st)))
    for name, rows in (("steps.csv", steps), ("runs.csv", runs)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    hd = {(r["scenario"], r["controller"]): float(r["hdscore"] or "nan") for r in runs}
    endr = {(r["scenario"], r["controller"]): r["end"] for r in runs}
    summary = {}
    for ctrl in sorted({r["controller"] for r in runs}):
        R = [r for r in runs if r["controller"] == ctrl]
        S = [s for s in steps if s["controller"] == ctrl and s["scenario"] in static]
        Rs, Ra = [r for r in R if r["group"] == "static"], [r for r in R if r["group"] == "actor"]
        m = {"n_runs": len(R), "n_static": len(Rs), "n_actor": len(Ra), "n_static_steps": len(S)}
        for k in ("lat50", "lon50", "lat25", "lon25"):
            m[f"{k}_med"], m[f"{k}_p95"] = q([s.get(k) for s in S], 50), q([s.get(k) for s in S], 95)
        for k in ("hd50", "hd25"):
            m[f"{k}_med_deg"] = float(np.degrees(q([s.get(k) for s in S], 50)))
            m[f"{k}_p95_deg"] = float(np.degrees(q([s.get(k) for s in S], 95)))
        m["xt_med"], m["xt_p95"], m["xt_max"] = (q([s["log_xt"] for s in S], p) for p in (50, 95, 100))
        m["dth_med_deg"] = float(np.degrees(q([s["log_dth"] for s in S], 50)))
        m["ends"] = {g: {e: sum(r["end"] == e for r in G) for e in sorted({r["end"] for r in G})}
                     for g, G in (("static", Rs), ("actor", Ra))}
        for g, G in (("static", Rs), ("actor", Ra), ("all", R)):
            for k in ("hdscore", "rc", "nc", "dac", "ttc", "c", "pdms"):
                v = [float(r[k]) for r in G if r[k] not in ("", None)]
                m[f"{g}_{k}_mean"] = float(np.mean(v)) if v else float("nan")
        m["n_stop_plans"] = sum(r["n_stop"] for r in R)
        m["n_fwd_only_plans"] = sum(r["n_fwd_only"] for r in R)
        if ctrl != REF:
            ds = [hd[(r["scenario"], ctrl)] - hd.get((r["scenario"], REF), np.nan) for r in Rs]
            da = [hd[(r["scenario"], ctrl)] - hd.get((r["scenario"], REF), np.nan) for r in Ra]
            same_s = sum(endr.get((r["scenario"], REF)) == r["end"] for r in Rs)
            same_a = sum(endr.get((r["scenario"], REF)) == r["end"] for r in Ra)
            n_s, n_a = len(Rs), len(Ra)
            c = CRIT
            chk = {
                "P1_plan_tracking": m["lat50_med"] <= c["lat_med"] and m["lat50_p95"] <= c["lat_p95"]
                and m["lon50_med"] <= c["lon_med"] and m["lon50_p95"] <= c["lon_p95"]
                and m["hd50_med_deg"] <= c["hd_med_deg"] and m["hd50_p95_deg"] <= c["hd_p95_deg"],
                "P2_path_tracking": all(r["xt_med"] <= c["xt_run_med"] for r in Rs)
                and sum(r["xt_max"] <= c["xt_run_max"] for r in Rs) >= n_s - 1,
                "P3_end_reasons": m["ends"]["static"].get("complete", 0) >= n_s - 1 and same_s >= n_s - 1,
                "P4_hdscore": bool(np.nanmean(ds) >= -c["static_mean_hd_drop"])
                and all(d >= -c["static_scene_hd_drop"] for d in ds)
                and same_a >= n_a - 1 and sum(abs(d) <= c["actor_hd_diff"] for d in da) >= n_a - 1,
            }
            m.update(static_hd_diff_mean=float(np.nanmean(ds)), static_hd_diff_min=float(np.nanmin(ds)),
                     actor_hd_diff=[round(d, 3) for d in da], static_same_end=same_s, actor_same_end=same_a,
                     checks=chk, verdict="pass" if all(chk.values()) else "fail")
        summary[ctrl] = m
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
