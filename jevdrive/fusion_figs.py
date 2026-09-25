"""Figures for the fusion diagnostics (todos/2026-09-25-fusion-diagnostics.md), from the committed small results in
research/results/fusion-diagnostics/ (so they run on the Mac). Style: research/plot_style.py via p4_carla._style.

  q1   concat / late fusion minus the best single arm, WOD pre-onset ADE and RFS, P5 pooled flip rate
  q4   SAM 3.1 recall by distance, P5 and nuScenes; flat-ground vs oracle-height lift

  python -m jevdrive.fusion_figs q1 q4 --out research/figs
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .p4_carla import _shrink_png, _style

RES = Path(__file__).resolve().parents[1] / "research" / "results" / "fusion-diagnostics"
MODEL = {"cinque": ("Cinque", "#0072B2"), "lebowski": ("Lebowski", "#D55E00")}


def _model(arm: str) -> str:
    return "cinque" if "cinque" in arm else "lebowski"


def q1(out: Path):
    import matplotlib.pyplot as plt
    ps = _style()
    rows = [("(a) ridge", "q1a", "ridge"), ("(a) cls", "q1a", "cls"), ("(b) ridge", "q1b", "ridge"), ("(b) cls", "q1b", "cls")]
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.0), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, readout, xlabel in ((axs[0], "pre-onset dec1-9 ADE", r"$\Delta$ pre-onset ADE, deciles 1-9 (m)"),
                                (axs[1], "RFS frame", r"$\Delta$ RFS (rater frames)")):
        for i, (label, f, fam) in enumerate(rows):
            t = pd.read_csv(RES / "q1" / f"{f}_concat_vs_best.csv")
            t = t[(t.family == fam) & (t.readout == readout) & t.is_best_single]
            for _, r in t.iterrows():
                m = _model(r.arm)
                y = i + (-0.18 if m == "cinque" else 0.18) + (0.06 if r.kind == "late" else -0.06)
                ax.errorbar(r.delta, y, xerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o" if r.kind == "concat" else "o",
                            mfc=MODEL[m][1] if r.kind == "concat" else "white", mec=MODEL[m][1], ecolor=MODEL[m][1],
                            ms=3, elinewidth=0.6, capsize=1.0)
        ax.axvline(0, color=ps.BASELINE, lw=0.5)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    t = pd.read_csv(RES / "q1" / "p5_concat_vs_best.csv")
    t = t[(t.readout == "flip rate (pooled)") & t.is_best_single]
    ax = axs[2]
    for j, (_, r) in enumerate(t.iterrows()):
        m = _model(r.arm)
        y = (0 if m == "cinque" else 1) + (0.12 if r.kind == "late" else -0.12)
        ax.errorbar(100 * r.delta, y, xerr=[[100 * (r.delta - r.lo)], [100 * (r.hi - r.delta)]], fmt="o",
                    mfc=MODEL[m][1] if r.kind == "concat" else "white", mec=MODEL[m][1], ecolor=MODEL[m][1], ms=3,
                    elinewidth=0.6, capsize=1.0)
    ax.axvline(0, color=ps.BASELINE, lw=0.5)
    ax.set_yticks([0, 1], ["P5 Cinque", "P5 Lebowski"])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ pooled flip rate (pp)")
    ax.grid(axis="y", visible=False)
    for ax, lab in zip(axs, ("(a)", "(b)", "(c)")):
        ps.panel(ax, lab)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=MODEL["cinque"][1], ms=3, label="Cinque"),
         Line2D([], [], marker="o", ls="", color=MODEL["lebowski"][1], ms=3, label="Lebowski"),
         Line2D([], [], marker="o", ls="", mfc="#444444", mec="#444444", ms=3, label="concat (+ Qwen L18_last)"),
         Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", ms=3, label="late fusion (averaged predictions)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    stem = out / "fusion-q1-concat-vs-best"
    ps.save(fig, stem)
    _shrink_png(stem)
    return stem


def q4(out: Path):
    """Recall of the background objects in image, <= 40 m, by GT distance: flat-ground lift (registered, solid) and the
    oracle-height lift (dashed, the flat-ground error removed), P5 and nuScenes val (visibility token >= 3)."""
    import matplotlib.pyplot as plt
    ps = _style()
    bins = ["0-10 m", "10-20 m", "20-40 m"]
    cls_c = {"pedestrian": "#CC79A7", "vehicle": "#0072B2", "cyclist": "#009E73", "cone": "#E69F00"}
    main_r, orc_r = "recall (ii), <= 40 m, in image", "side: recall (ii) with the oracle-height lift"
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN * 0.75, 2.1), sharey=True)
    for ax, f, classes, lab in ((axs[0], "p5_recall.csv", ("pedestrian", "vehicle"), "(a) P5 (CARLA)"),
                                (axs[1], "nusc_vis3_recall.csv", ("pedestrian", "vehicle", "cyclist", "cone"), "(b) nuScenes val")):
        t = pd.read_csv(RES / "q4" / f)
        for cls in classes:
            for rd, ls, mfc in ((main_r, "-", cls_c[cls]), (orc_r, "--", "white")):
                h = t[(t.cls == cls) & (t.reading == rd) & t.dbin.notna() & t.weather.isna()]
                h = h.set_index("dbin").reindex(bins)
                ax.plot(np.arange(3), h.recall, "o" + ls, color=cls_c[cls], mfc=mfc, ms=3)
        ax.set_xticks(range(3), bins)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("GT distance")
        ps.panel(ax, lab)
    axs[0].set_ylabel("Recall, BEV gate max(2 m, 0.1 d)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", ms=3, label=k) for k, c in cls_c.items()]
    h += [Line2D([], [], color="#444444", ls="-", label="flat ground"),
          Line2D([], [], color="#444444", ls="--", marker="o", mfc="white", ms=3, label="oracle height")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    stem = out / "fusion-q4-sam-recall"
    ps.save(fig, stem)
    _shrink_png(stem)
    return stem


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=("q1", "q4"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "research" / "figs"))
    a = ap.parse_args()
    for f in a.figs:
        print(globals()[f](Path(a.out)))


if __name__ == "__main__":
    main()
