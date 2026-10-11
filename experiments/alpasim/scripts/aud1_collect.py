#!/usr/bin/env python3
"""AUD1 (private-gap audit): per-scene table of the stored public-suite rollouts of APY10m10-AB-s0, P2H10-F-s0 and the WA-JEPA reference,
with the organisers' run-level metrics recomputed on the pooled 1491 scenes, plus scene metadata from the driver logs.

  aud1_collect.py --out DIR      (box; the AlpaSim venv: numpy; reads the ot3 / cf1 / c0c manifests, writes nothing else)

Definitions (src/eval at 0bb4c4b): per rollout, dist_to_gt_trajectory = MAX over time of the distance (m) from the ego polygon centroid to
the full logged ego path (shapely LineString.distance), dist_traveled_m = LAST arc length of the ego's own path, offroad_or_collision_at_fault
= MAX of the per-step flag; timesteps after the first dist_to_gt_trajectory >= 4 m are dropped before time aggregation (RemoveTimestepsAfterEvent).
Run level (aggregation/processing.py): dist_to_gt_trajectory = mean over clips; avg_dist_between_incidents_at_fault = sum(dist_traveled_m) /
sum(offroad_or_collision_at_fault) / 1000 (km per at-fault incident). Rollouts that fail before the driver runs (route fold) carry no
metrics and are excluded from both; the scene score counts them as 0 (e2e_challenge/local_evaluation/evaluate.py averages scene scores).
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

RA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/alpasim"
M = ("dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory", "dist_traveled_m", "gt_dist_traveled_m", "offroad_or_collision_at_fault",
     "collision_at_fault", "offroad", "left_corridor_laterally", "progress_clipped_rel", "dist_to_gt_location", "plan_deviation")


def dirs():
    o3 = json.loads((RA / "ot3/a/manifest.json").read_text())
    cf = json.loads((RA / "cf1/a/manifest.json").read_text())
    c0c = json.loads((RA / "c0c/manifest.json").read_text())
    d = {k: o3.get(k, []) + o3.get(k + "@new", []) + cf.get(k + "@new", []) + cf.get(k + "@fresh", []) for k in ("APY10m10-AB-s0", "P2H10-F-s0")}
    d["WA-JEPA"] = c0c["WA-JEPA (reference)"]
    return d


def load(run_dirs):
    """scene -> rollout record, scene -> {k: drive record}, scene -> start record; first run dir wins; also the per-run metrics_results."""
    R, K, S, runs = {}, {}, {}, []
    for D in map(Path, run_dirs):
        d = json.loads((D / "aggregate/results-summary.json").read_text())
        runs.append({"dir": str(D), **{k: (d["metrics_results"][0].get(k) if d["metrics_results"] else None)
                                         for k in ("n_clips", "avg_dist_between_incidents_at_fault", "dist_to_gt_trajectory")}})
        new = set()
        for r in d["rollouts"]:
            if r["clipgt_id"] not in R:
                R[r["clipgt_id"]] = r
                new.add(r["clipgt_id"])
        f = D / "driver-logs/drive.jsonl"
        if f.exists():
            for line in f.open():
                x = json.loads(line)
                s = x.get("scene")
                if s not in new:
                    continue
                if x["kind"] == "start":
                    S.setdefault(s, x)
                elif x["kind"] == "drive":
                    K.setdefault(s, {}).setdefault(x["k"], x)
    return R, K, S, runs


def runlevel(R):
    m = [r["metrics"] for r in R.values() if r["metrics"].get("dist_traveled_m") is not None]
    dt, af = sum(x["dist_traveled_m"] for x in m), sum(x["offroad_or_collision_at_fault"] for x in m)
    g = np.array([x["dist_to_gt_trajectory"] for x in m])
    return {"n_rollouts": len(R), "n_clips": len(m), "avg_scene_score": float(np.mean([r["score"] for r in R.values()])),
            "zeros": int(sum(r["score"] == 0 for r in R.values())), "sum_dist_traveled_km": dt / 1000, "at_fault_incidents": af,
            "avg_dist_between_incidents_at_fault_km": dt / af / 1000 if af else float("inf"), "dist_to_gt_trajectory_mean_m": float(g.mean()),
            "dist_to_gt_trajectory_median_m": float(np.median(g)), "dist_to_gt_trajectory_p90_m": float(np.quantile(g, .9)),
            "frac_dist_to_gt_ge_4m": float((g >= 4).mean()), "mean_dist_traveled_m": dt / len(m),
            "lateral_dist_to_gt_trajectory_mean_m": float(np.mean([x["lateral_dist_to_gt_trajectory"] for x in m]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {k: load(v) for k, v in dirs().items()}
    summ = {k: {**runlevel(R), "runs": runs} for k, (R, K, S, runs) in data.items()}
    (out / "runlevel.json").write_text(json.dumps(summ, indent=1))
    scenes = sorted(set().union(*(set(v[0]) for v in data.values())))
    rows = []
    for s in scenes:
        row = {"scene": s, "log": "_".join(s.split("_")[:2]), "vehicle": s.split("_")[1]}
        RA_, KA, SA, _ = data["APY10m10-AB-s0"]
        RW, KW, _, _ = data["WA-JEPA"]
        st = SA.get(s) or data["P2H10-F-s0"][2].get(s)
        if st and "cam" in st:
            c = st["cam"]
            row |= {"fx": c["K"][0][0], "fy": c["K"][1][1], "cx": c["K"][0][2], "cy": c["K"][1][2], "R": json.dumps(c["R"]), "t": json.dumps(c["t"]),
                    "D": json.dumps(c["D"])}
        k0 = KA.get(s, {}).get(0) or data["P2H10-F-s0"][1].get(s, {}).get(0)
        if k0:
            r0 = k0.get("route0") or [None, None]
            row |= {"route0_x": r0[0], "route0_y": r0[1], "cmd_ap2": k0.get("cmd"), "n_wp0": k0.get("n_wp")}
        w0 = KW.get(s, {}).get(0)
        if w0:
            v, acc = w0["ego"][:2], w0["ego"][2:4]
            row |= {"v0": float(np.hypot(*v)), "vx0": v[0], "vy0": v[1], "a0x": acc[0], "cmd_wajepa": w0.get("cmd")}
        for k, (R, K, S, _) in data.items():
            r = R.get(s)
            if r is None:
                continue
            mm = r["metrics"]
            row |= {f"{k}|score": r["score"], f"{k}|status": r["status"], f"{k}|n_decisions": len(K.get(s, {}))}
            row |= {f"{k}|{n}": mm.get(n) for n in M}
            p = K.get(s, {}).get(0, {}).get("poses")
            if p:
                row |= {f"{k}|plan0_end_x": p[-1][0], f"{k}|plan0_end_y": p[-1][1]}
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "per_scene.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: {x: y for x, y in v.items() if x != "runs"} for k, v in summ.items()}, indent=1))


if __name__ == "__main__":
    main()
