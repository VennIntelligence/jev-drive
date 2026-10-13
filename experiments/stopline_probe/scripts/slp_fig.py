#!/usr/bin/env python3
"""STOPLINE-PROBE figure (Mac, .venv): probe AUC (tokens, logistic on PCA-256) against the position-only baseline (d, v, a trees) per
stratum, nuPlan navtrain (all rows and the eligible-label subset), plus the PAI frames (nuPlan probe applied unchanged, per class).

  slp_fig.py --res experiments/stopline_probe/results --out experiments/stopline_probe/figs/slp_auc.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.res)
    rows = []
    for name, tag in (("nuplan.json", "nuPlan"), ("nuplan_elig.json", "nuPlan elig")):
        f = R / name
        if f.exists():
            for s, r in json.loads(f.read_text())["strata"].items():
                t, b = r["tok_lr"], r["base"]
                rows.append((f"{tag}: {s}", t["auc"], t["lo"], t["hi"], b["auc"], b["lo"], b["hi"], b["pos"], b["n"]))
    f = R / "pai.json"
    if f.exists():
        P = json.loads(f.read_text())
        for c, r in P["classes"].items():
            if "transfer" in r and "auc" in r["transfer"]:
                t = r["transfer"]
                b = r.get("cv", {}).get("base", {})
                rows.append((f"PAI {c} vs green (nuPlan probe, {r['pos_scenes']} scenes)", t["auc"], t["lo"], t["hi"], b.get("auc", np.nan),
                             b.get("lo", np.nan), b.get("hi", np.nan), r["pos_frames"], r["frames"]))
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, t, tl, th, b, bl, bh, pos, n) in zip(y, rows):
        ax.errorbar(t, yi + 0.12, xerr=[[t - tl], [th - t]], fmt="o", color="#1f5fa8", ms=5, capsize=2)
        if np.isfinite(b):
            ax.errorbar(b, yi - 0.12, xerr=[[b - bl], [bh - b]], fmt="s", color="#9a9a9a", ms=5, capsize=2)
        ax.text(1.005, yi, f"{pos}/{n}", va="center", fontsize=7, color="#555", transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    for x, ls in ((0.85, "--"), (0.70, ":")):
        ax.axvline(x, color="#c44", ls=ls, lw=0.8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC red vs non-red (95% CI, group bootstrap)")
    ax.plot([], [], "o", color="#1f5fa8", label="frozen view_39 tokens (t0 frame), logistic on PCA-256")
    ax.plot([], [], "s", color="#9a9a9a", label="position-only baseline (distance to line, speed, accel)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_title("Is the ego-lane light readable from the frozen openpilot tokens?  (right: red / rows)", fontsize=9)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
