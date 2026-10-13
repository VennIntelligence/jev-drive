#!/usr/bin/env python3
"""ROUTE-ONLY BEV figures (plans/2026-10-13-route-only-prereg.md): one scene, one junction decision of the JEV_ROUTE run, in that
decision's rear-axle frame: the model's plan (`poses_model`), the route path (`route_path`), the served poses, and the ego tracks
(logged anchors over the whole rollout) of every given run. Read-only on the run dirs. Box, AlpaSim's venv (matplotlib).

  ro_bev.py --route LABEL=<run dirs> --ref LABEL=<run dirs> [--ref ...] --scene <id> [--k K] --out <png>
K defaults to the covered decision with the largest |lateral change at 4 s| (route path vs plan).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def recs(dirs, scene):
    """The drive records of the scene's last session in any of the run dirs, by decision index."""
    out = {}
    for d in dirs.split(","):
        f = Path(d) / "driver-logs/drive.jsonl"
        if not f.exists():
            continue
        S = {}
        for line in f.open():
            if scene in line:
                o = json.loads(line)
                if o.get("kind") == "drive" and o["scene"] == scene:
                    S.setdefault(o["session"], {})[o["k"]] = o
        if S:
            out = max(S.values(), key=lambda v: max(r["t"] for r in v.values()))
    return out


def to_frame(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, float).reshape(-1, 2) - np.asarray(a[:2], float)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--k", type=int)
    ap.add_argument("--score", action="append", default=[], help="LABEL=score text for the legend")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab, dirs = a.route.split("=", 1)
    R = recs(dirs, a.scene)
    assert R, f"no records of {a.scene} in {dirs}"
    if a.k is None:
        cand = [(abs(r["route"]["dlat"][2]), k) for k, r in R.items() if r.get("route", {}).get("cov", 0) > 0 and r["route"].get("dlat")]
        a.k = max(cand)[1] if cand else min(R)
    r = R[a.k]
    A = r["anchor"]
    sc = dict(s.split("=", 1) for s in a.score)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    if r.get("route_path"):
        P = np.asarray(r["route_path"])
        ax.plot(P[:, 1], P[:, 0], "-", c="0.55", lw=6, alpha=0.45, label="route path (carried memory + blend)", zorder=1)
    pm = np.asarray(r.get("poses_model", r["poses"]))
    ax.plot(np.r_[0, pm[:, 1]], np.r_[0, pm[:, 0]], "o--", c="#d1495b", ms=4, lw=1.4, label="model plan (RSG-L18-s0)", zorder=3)
    ps = np.asarray(r["poses"])
    ax.plot(np.r_[0, ps[:, 1]], np.r_[0, ps[:, 0]], "s-", c="#00798c", ms=4, lw=1.6, label=f"served poses ({lab}), on route {r['route'].get('cov')} / 8", zorder=4)
    cols = ["#30638e", "#edae49", "#7a5195"]
    for i, (l, d) in enumerate([(lab, None)] + [s.split("=", 1) for s in a.ref]):
        Q = R if d is None else recs(d, a.scene)
        if not Q:
            continue
        T = to_frame(A, [Q[k]["anchor"][:2] for k in sorted(Q)])
        ax.plot(T[:, 1], T[:, 0], ".-", c=cols[i % 3], lw=1.0, ms=5, label=f"ego track {l}" + (f" (score {sc[l]})" if l in sc else ""), zorder=2)
    ax.plot(0, 0, "k^", ms=9, zorder=5)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("y left (m)")
    ax.set_ylabel("x forward (m)")
    ri = r["route"]
    v = np.hypot(*(np.subtract(A[:2], R[a.k - 1]["anchor"][:2]))) / 0.5 if a.k - 1 in R else 0.0
    ax.set_title(f"{a.scene[-16:]}  k {a.k}  v {v:.1f} m/s  gap {ri.get('gap')} m  Lb {ri.get('lb')} m", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out, "k", a.k, ri)


if __name__ == "__main__":
    main()
