#!/usr/bin/env python
"""Turn-in timing figure of rc-*-pre (box, repo root): open-loop action response before the junction (evalol_ol/<model>.json `turnin`) and
closed-loop turn-in arc relative to the turn start (results/pre_split*.json from pre_report.py split).

  pre_fig.py [--split results/pre_split_doff.json ...] [--out experiments/op_route_ft/figs/pre_turnin_timing.png]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.common import data_dir  # noqa: E402

MODELS = ("O", "rc-ctl-s0", "rc-bear-s0", "rc-ctl-pre-s0", "rc-bear-pre-s0")
COL = {"O": "#7f7f7f", "shipped": "#7f7f7f", "rc-ctl-s0": "#2a78d6", "rc-bear-s0": "#eb6834", "rc-ctl-pre-s0": "#1baf7a", "rc-bear-pre-s0": "#e87ba4"}
MK = {"O": "o", "shipped": "o", "rc-ctl-s0": "s", "rc-bear-s0": "^", "rc-ctl-pre-s0": "D", "rc-bear-pre-s0": "v"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="*", default=[])
    ap.add_argument("--out", default=str(REPO / "experiments/op_route_ft/figs/pre_turnin_timing.png"))
    a = ap.parse_args()
    ev = data_dir() / "runs/op_route_ft/evalol_ol"
    profs = ("creep", "brake", "cruise")
    ncl = len(a.split)
    fig, axs = plt.subplots(1, len(profs) + ncl, figsize=(4.2 * (len(profs) + ncl), 4.2))
    tgt_done = set()
    for m in MODELS:
        p = ev / f"{m}.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text())["turnin"]
        for ax, pf in zip(axs, profs):
            ds = [d for d in (10, 20, 30) if f"{pf}_d{d}" in t]
            y = [t[f"{pf}_d{d}"]["k_signed"]["mean"] for d in ds]
            lo = [t[f"{pf}_d{d}"]["k_signed"]["lo"] for d in ds]
            hi = [t[f"{pf}_d{d}"]["k_signed"]["hi"] for d in ds]
            ax.errorbar(ds, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color=COL[m], marker=MK[m], ms=7, lw=2, capsize=3,
                        label="shipped" if m == "O" else m)
            if pf not in tgt_done:
                ax.plot(ds, [t[f"{pf}_d{d}"]["target_signed"] or 0 for d in ds], color="k", ls="--", lw=1.5, label="turn-in target")
                tgt_done.add(pf)
    for ax, pf in zip(axs, profs):
        ax.axhline(0, color="#bbbbbb", lw=1)
        ax.set_title(f"open loop, CARLA dev, {pf} poses", fontsize=10)
        ax.set_xlabel("distance to the junction d (m)")
        ax.set_xticks([10, 20, 30])
        ax.grid(alpha=.25)
    axs[0].set_ylabel("desired curvature towards the commanded exit (1/m)")
    axs[0].legend(fontsize=8)
    for ax, sp in zip(axs[len(profs):], a.split):
        rows = json.loads(Path(sp).read_text())
        arms = list(dict.fromkeys(r["arm"] for r in rows))
        for i, arm in enumerate(arms):
            rr = [r for r in rows if r["arm"] == arm and r["entered"]]
            v = np.array([r["tin_m"] for r in rr if r["tin_m"] is not None], float)
            took = np.array([r["branch"] == "yes" for r in rr if r["tin_m"] is not None])
            base = arm.split("@")[0]
            jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
            ax.scatter(v[~took], i + jit[~took], s=36, facecolors="none", edgecolors=COL.get(base, "k"), lw=1.5)
            ax.scatter(v[took], i + jit[took], s=40, color=COL.get(base, "k"), marker=MK.get(base, "o"))
            if len(v):
                ax.plot([np.median(v)] * 2, [i - .3, i + .3], color="k", lw=2)
            ax.text(30, i + 0.32, f"{len(v)} / {len(rr)} steered", fontsize=8, va="center")
        ax.axvline(0, color="#bbbbbb", lw=1)
        ax.axvline(3, color="#bbbbbb", lw=1, ls=":")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms, fontsize=8)
        ax.set_xlim(-15, 45)
        ax.set_xlabel("arc at turn-in, m after the turn start (< 0: before)")
        ax.set_title("closed loop, B2D 25 turns: " + Path(sp).stem.replace("pre_split_", "desire ").replace("pre_split", "desire on"), fontsize=10)
        ax.grid(alpha=.25)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
