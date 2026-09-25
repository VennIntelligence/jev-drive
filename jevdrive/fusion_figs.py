"""Figures for the fusion diagnostics (todos/2026-09-25-fusion-diagnostics.md), from the committed small results in
research/results/fusion-diagnostics/ (so they run on the Mac). Style: research/plot_style.py via p4_carla._style.

  q1   concat / late fusion minus the best single arm, WOD pre-onset ADE and RFS, P5 pooled flip rate
  q4   SAM 3.1 recall by distance: P5 hazard (reading (i)) and background (ii), nuScenes; flat-ground vs oracle-height lift
  q8   reaction window per P5 family against the latency tiers

  python -m jevdrive.fusion_figs q1 q4 q8 --out research/figs
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


def q8(out: Path, run_csv: Path | None = None):
    import matplotlib.pyplot as plt
    ps = _style()
    p = pd.read_csv(Path(run_csv or RES / "q8" / "q8_pairs.csv"))
    fams = p.groupby("family").window_s.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.4))
    data = [p[p.family == f].window_s.to_numpy() for f in fams]
    ax.boxplot(data, vert=False, widths=0.5, showfliers=True, flierprops={"ms": 2},
               medianprops={"color": "#000000", "lw": 0.8}, boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6},
               capprops={"lw": 0.6})
    tiers = pd.read_json(RES / "q8" / "q8_tiers.json", typ="series")
    colors = ["#009E73", "#0072B2", "#56B4E9", "#D55E00"]
    for (name, lat), c in zip(tiers.items(), colors):
        ax.axvline(lat + 0.5, color=c, lw=0.7, ls="--", label=f"{name} + 0.5 s")
    ax.set_yticks(range(1, len(fams) + 1), fams, fontsize=6.5)
    ax.set_xlabel(r"Window $t_{div}-t_{vis}$ (s)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    stem = out / "fusion-q8-reaction-windows"
    ps.save(fig, stem)
    _shrink_png(stem)
    return stem


def q4(out: Path):
    import matplotlib.pyplot as plt
    ps = _style()
    haz = pd.read_csv(RES / "q4" / "p5_hazard_by_dist.csv")
    rec = pd.read_csv(RES / "q4" / "p5_recall.csv")
    nus = pd.read_csv(RES / "q4" / "nusc_vis3_recall.csv")
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.1), sharey=True)
    bins = ["0-10 m", "10-20 m", "20-40 m", "40-80 m"]
    cls_c = {"pedestrian": "#CC79A7", "vehicle": "#0072B2", "cyclist": "#009E73", "cone": "#E69F00"}
    ax = axs[0]
    for cls in ("pedestrian", "vehicle"):
        h = haz[haz.cls == cls].set_index("dbin").reindex(bins)
        x = np.arange(len(bins))
        ax.plot(x, h.recall, "o-", color=cls_c[cls], ms=3, label=f"{cls}, flat ground")
        ax.plot(x, h.recall_oracle_height, "o--", color=cls_c[cls], mfc="white", ms=3, label=f"{cls}, oracle height")
    ax.set_title("")
    ps.panel(ax, "(a) P5 hazard, visible")
    for ax, t, lab, reading in ((axs[1], rec, "(b) P5 background <= 40 m", "recall (ii), <= 40 m, in image"),
                                (axs[2], nus, "(c) nuScenes val, vis >= 3", "recall (ii), <= 40 m, in image")):
        for cls in (["pedestrian", "vehicle"] if ax is axs[1] else ["pedestrian", "vehicle", "cyclist", "cone"]):
            for rd, ls, mfc in ((reading, "-", None), ("side: recall (ii) with the oracle-height lift", "--", "white")):
                h = t[(t.cls == cls) & (t.reading == rd) & t.dbin.notna() & t.weather.isna()] if "weather" in t else \
                    t[(t.cls == cls) & (t.reading == rd) & t.dbin.notna()]
                h = h.drop_duplicates("dbin").set_index("dbin").reindex(bins[:3])
                ax.plot(np.arange(3), h.recall, "o" + ls, color=cls_c[cls], mfc=mfc or cls_c[cls], ms=3)
        ps.panel(ax, lab)
    for ax in axs:
        ax.set_xticks(range(len(bins)) if ax is axs[0] else range(3), bins if ax is axs[0] else bins[:3], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("GT distance")
    axs[0].set_ylabel("Recall (BEV gate max(2 m, 0.1 d))")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", ms=3, label=k) for k, c in cls_c.items()]
    h += [Line2D([], [], color="#444444", ls="-", label="flat-ground lift"),
          Line2D([], [], color="#444444", ls="--", marker="o", mfc="white", ms=3, label="oracle-height lift")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    stem = out / "fusion-q4-sam-recall"
    ps.save(fig, stem)
    _shrink_png(stem)
    return stem


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=("q1", "q4", "q8"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "research" / "figs"))
    a = ap.parse_args()
    for f in a.figs:
        print(globals()[f](Path(a.out)))


if __name__ == "__main__":
    main()
