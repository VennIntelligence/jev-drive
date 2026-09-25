"""Figure for the fast-perception comparison (todos/2026-09-26-fast-perception.md) from the committed small results in
research/results/fast-perception/ (runs on the Mac). Style: research/plot_style.py.

  python -m jevdrive.fastperc_figs --out research/figs
"""
from pathlib import Path

import pandas as pd

from .p4_carla import _shrink_png, _style

RES = Path(__file__).resolve().parents[1] / "research" / "results" / "fast-perception"
# eval tag -> (latency backend spec of the same configuration, label, colour key, marker)
RUNS = {
    "sam31-orig": ("sam31:exact:all:1008:none:bf16", "SAM 3.1 (repo path)", "black", "o"),
    "sam31-batched": ("sam31:batched:all:1008:none:bf16", "SAM 3.1 batched", "black", "s"),
    "esam3-tinyvit": ("esam3:tinyvit:exact:fp32", "EfficientSAM3 TV-M", "purple", "o"),
    "gdino-tiny": ("gdino:0.35:0.25", "Grounding DINO-T", "orange", "D"),
    "yoloe26x-640": ("yoloe:yoloe-26x-seg.pt:640:fp32", "YOLOE-26x 640", "green", "o"),
    "yoloe26x-1280": ("yoloe:yoloe-26x-seg.pt:1280:fp32", "YOLOE-26x 1280", "green", "s"),
    "yoloe26x-640-words": ("yoloe:yoloe-26x-seg.pt:640:fp32:words", "YOLOE-26x 640, class words", "green", "v"),
    "yoloe26x-1280-words": ("yoloe:yoloe-26x-seg.pt:1280:fp32:words", "YOLOE-26x 1280, class words", "green", "^"),
    "yoloe26x-1280-words-half": ("yoloe:yoloe-26x-seg.pt:1280:half:words", "YOLOE-26x 1280, class words, fp16", "green", ">"),
    "yolo26x-640": ("yolo:yolo26x-seg.pt:640:fp32", "YOLO26x-seg 640 (COCO)", "blue", "o"),
    "yolo26x-640-half": ("yolo:yolo26x-seg.pt:640:half", "YOLO26x-seg 640 (COCO), fp16", "blue", "v"),
    "yolo26x-1280": ("yolo:yolo26x-seg.pt:1280:fp32", "YOLO26x-seg 1280 (COCO)", "blue", "s"),
}


def table() -> pd.DataFrame:
    lat = pd.read_csv(RES / "latency.csv")
    rec = pd.read_csv(RES / "recall.csv").set_index("name")
    rows = []
    for tag, (spec, label, col, mk) in RUNS.items():
        if tag not in rec.index:
            continue
        l3 = lat[(lat.backend == spec) & (lat.cams == 3)]
        l1 = lat[(lat.backend == spec) & (lat.cams == 1)]
        rows.append({"tag": tag, "label": label, "col": col, "mk": mk,
                     "p50_3": l3.p50_ms.iloc[0] if len(l3) else float("nan"),
                     "p95_3": l3.p95_ms.iloc[0] if len(l3) else float("nan"),
                     "p50_1": l1.p50_ms.iloc[0] if len(l1) else float("nan"), **rec.loc[tag].to_dict()})
    return pd.DataFrame(rows)


def fig(out: Path):
    import matplotlib.pyplot as plt
    ps = _style()
    t = table()
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.35), sharex=True)
    panels = (("p5_haz_ped_all", "p5_haz_ped_all_oracle", "(a) P5 hazard pedestrians, recall"),
              ("nusc_ped_le40", "nusc_ped_le40_oracle", r"(b) nuScenes pedestrians $\leq$ 40 m, recall"))
    for ax, (flat, orc, title) in zip(axs, panels):
        for _, r in t.iterrows():
            c = ps.PALETTE[r.col] if r.col != "black" else "#000000"
            ax.plot(r.p95_3, r[flat], r.mk, color=c, ms=4.2, label=r.label, zorder=3)
            ax.plot(r.p95_3, r[orc], r.mk, mfc="white", mec=c, ms=4.2, zorder=2)
            ax.plot([r.p95_3] * 2, [r[flat], r[orc]], color=c, lw=0.5, alpha=0.6, zorder=1)
        for x, ls in ((50, "-"), (30, ":")):
            ax.axvline(x, color=ps.BASELINE, lw=0.6, ls=ls, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([20, 30, 50, 100, 200, 500], ["20", "30", "50", "100", "200", "500"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("latency, 3 cameras, batch 1, p95 (ms)")
        ps.panel(ax, title)
    axs[0].set_ylabel("recall")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, handlelength=1.0)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.88, wspace=0.18)
    stem = out / "fastperc-latency-recall"
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    _shrink_png(stem)
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/figs")
    a = ap.parse_args()
    print(fig(Path(a.out)).drop(columns=["col", "mk"]).to_string())
