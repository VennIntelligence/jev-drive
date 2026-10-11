#!/usr/bin/env python3
"""AUD1-LHT: APY10m10-AB-s0 (ap2 driver, interpreted passes SH30_COMPILE=0, backwarp, serving switches off) on the 220 public Singapore
scenes (city sg-one-north in navtest/configs) with the left-hand-traffic flag on (SH30_LHT=1) and off, plus the stored CF1 run (compiled
passes, flag off) as a second control.

  aud1_lht_report.py --out DIR      (box; repo .venv; reads aud1_lht/sg220/lht{0,1} and the CF1 fresh run, writes DIR only)

Run-level metrics as aud1_collect.py (src/eval at 0bb4c4b): scene score = mean over scenes (failed rollouts count 0); GT distance = mean
over clips with metrics of dist_to_gt_trajectory; at-fault distance (km) = sum dist_traveled_m / sum offroad_or_collision_at_fault / 1000.
CIs: cluster bootstrap over log segments (the scene id before '-', 13 segments in 2 logs; 2 logs are too few clusters), 10 000 draws,
seed 0; paired differences resample the same segments in both arms.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path.home() / "data/jev-drive"))
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402

RA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/alpasim"
L = RA / "aud1_lht"
ARMS = {"lht1": L / "sg220/lht1", "lht0": L / "sg220/lht0",
        "stored": RA / "cf1/a/runs/APY10m10-AB-s0@fresh-cf1fresh/20261009-203623"}
PAIRS = [("lht1", "lht0"), ("lht1", "stored"), ("lht0", "stored")]


def load(D, scenes):
    R = {r["clipgt_id"]: r for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]}
    miss = [s for s in scenes if s not in R]
    assert not miss, f"{D}: {len(miss)} scenes missing"
    return [R[s] for s in scenes]


def cls(r):
    if r["score"] > 0:
        return ""
    f = r.get("failure_reason") or "zero_no_reason"
    return "route_fold" if f.startswith("Waypoint") else f


def runlevel(rs, idx=None):
    idx = range(len(rs)) if idx is None else idx
    sc, dt, af, gt = [], 0.0, 0.0, []
    for i in idx:
        r = rs[i]
        sc.append(r["score"])
        m = r.get("metrics") or {}
        if m.get("dist_traveled_m") is not None:
            dt += m["dist_traveled_m"]; af += m["offroad_or_collision_at_fault"]; gt.append(m["dist_to_gt_trajectory"])
    return np.array([np.mean(sc), dt / af / 1000 if af else np.inf, np.mean(gt), af, len(gt)])


def boot_runlevel(arms, groups, B=10_000, seed=0):
    """Segment-clustered bootstrap of the run-level triple per arm and of the paired differences. -> {key: (lo, hi) per stat}"""
    seg = sorted(set(groups))
    gi = {g: [i for i, x in enumerate(groups) if x == g] for g in seg}
    rng = np.random.default_rng(seed)
    draws = {a: [] for a in arms}
    for _ in range(B):
        idx = [i for g in rng.choice(seg, len(seg)) for i in gi[g]]
        for a, rs in arms.items():
            draws[a].append(runlevel(rs, idx)[:3])
    draws = {a: np.array(v) for a, v in draws.items()}
    out = {a: np.percentile(v, [2.5, 97.5], 0) for a, v in draws.items()}
    for a, b in PAIRS:
        d = draws[a] - draws[b]
        d[~np.isfinite(d)] = np.nan
        out[f"{a}-{b}"] = np.nanpercentile(d, [2.5, 97.5], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    o = Path(ap.parse_args().out)
    o.mkdir(parents=True, exist_ok=True)
    scenes = (L / "lists/sg220.txt").read_text().split()
    seg = [s.rsplit("-", 1)[0] for s in scenes]
    log = ["_".join(s.split("_")[:2]) for s in scenes]
    A = {a: load(D, scenes) for a, D in ARMS.items()}
    # per-scene table
    with open(o / "per_scene.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "log", "segment"] + [f"{a}|{k}" for a in A for k in ("score", "class", "dist_to_gt_trajectory", "dist_traveled_m", "at_fault")])
        for i, s in enumerate(scenes):
            row = [s, log[i], seg[i]]
            for a in A:
                r = A[a][i]; m = r.get("metrics") or {}
                row += [r["score"], cls(r), m.get("dist_to_gt_trajectory"), m.get("dist_traveled_m"), m.get("offroad_or_collision_at_fault")]
            w.writerow(row)
    res = {"scenes": len(scenes), "segments": len(set(seg)), "logs": dict(Counter(log)), "runs": {a: str(D) for a, D in ARMS.items()}, "arms": {}, "pairs": {}}
    ci = boot_runlevel(A, seg)
    for a, rs in A.items():
        rl = runlevel(rs)
        res["arms"][a] = {"avg_scene_score": rl[0], "avg_dist_between_incidents_at_fault_km": rl[1], "dist_to_gt_trajectory_mean_m": rl[2],
                          "at_fault_incidents": int(rl[3]), "clips_with_metrics": int(rl[4]),
                          "ci95": {"score": ci[a][:, 0].tolist(), "at_fault_km": ci[a][:, 1].tolist(), "gt_m": ci[a][:, 2].tolist()},
                          "zeros": dict(Counter(c for c in map(cls, rs) if c)), "n_zero": sum(r["score"] == 0 for r in rs),
                          "collision_at_fault": int(sum((r.get("metrics") or {}).get("collision_at_fault") or 0 for r in rs)),
                          "offroad": int(sum((r.get("metrics") or {}).get("offroad") or 0 for r in rs)),
                          "frac_gt_ge_4m": float(np.mean([(r.get("metrics") or {})["dist_to_gt_trajectory"] >= 4 for r in rs if (r.get("metrics") or {}).get("dist_traveled_m") is not None]))}
    for a, b in PAIRS:
        x, y = (np.array([r["score"] for r in A[k]]) for k in (a, b))
        p = stats.paired(x, y, groups=seg)
        p2 = stats.paired(x, y, groups=log)
        ra, rb = runlevel(A[a]), runlevel(A[b])
        res["pairs"][f"{a}-{b}"] = {"score_diff": p, "score_diff_2log_cluster": p2,
                                     "score_diff_scene_unit": stats.paired(x, y),
                                     "at_fault_km_diff": [ra[1] - rb[1], *ci[f"{a}-{b}"][:, 1].tolist()],
                                     "gt_m_diff": [ra[2] - rb[2], *ci[f"{a}-{b}"][:, 2].tolist()],
                                     "scenes_up": int((x > y).sum()), "scenes_down": int((x < y).sum()), "scenes_equal": int((x == y).sum()),
                                     "zero_to_pos": [scenes[i] for i in np.where((y == 0) & (x > 0))[0]],
                                     "pos_to_zero": [scenes[i] for i in np.where((x == 0) & (y > 0))[0]]}
    (o / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    for a, v in res["arms"].items():
        print(f"{a}: score {v['avg_scene_score']:.4f} {np.round(v['ci95']['score'], 3)} zeros {v['n_zero']} {v['zeros']} at-fault {v['at_fault_incidents']} "
              f"km {v['avg_dist_between_incidents_at_fault_km']:.3f} {np.round(v['ci95']['at_fault_km'], 3)} GT {v['dist_to_gt_trajectory_mean_m']:.3f} "
              f"{np.round(v['ci95']['gt_m'], 3)} >=4m {v['frac_gt_ge_4m']:.3f}")
    for k, v in res["pairs"].items():
        p = v["score_diff"]
        print(f"{k}: dscore {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] ({p['units']} segs); 2-log [{v['score_diff_2log_cluster']['lo']:+.4f}, "
              f"{v['score_diff_2log_cluster']['hi']:+.4f}]; scene-unit [{v['score_diff_scene_unit']['lo']:+.4f}, {v['score_diff_scene_unit']['hi']:+.4f}]; "
              f"up/down/eq {v['scenes_up']}/{v['scenes_down']}/{v['scenes_equal']}; zero->pos {len(v['zero_to_pos'])} pos->zero {len(v['pos_to_zero'])}; "
              f"d at-fault km {np.round(v['at_fault_km_diff'], 3)} d GT m {np.round(v['gt_m_diff'], 3)}")


if __name__ == "__main__":
    main()
