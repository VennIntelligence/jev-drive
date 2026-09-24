"""Figures for the Alpamayo 1.5 smoke run (reads the small result files pulled into research/results/alpamayo-smoke).

  .venv/bin/python scripts/alpamayo_figs.py [--res research/results/alpamayo-smoke] [--out research/figs]
Writes alpamayo-bev.png (pred vs GT BEV overlays) and alpamayo-latency.png (stage breakdown per config row).
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import plot_style as ps

STAGES = [("vision_ms", "vision", ps.PALETTE["orange"]), ("prefill_ms", "prefill", ps.PALETTE["sky_blue"]),
          ("decode_ms", "reasoning decode", ps.PALETTE["vermillion"]), ("flow_ms", "flow matching", ps.PALETTE["green"]),
          ("other_ms", "other", ps.BASELINE)]


def pick(ev: pd.DataFrame, example: str, k: int = 6) -> list[str]:
    """The shipped example plus clips at spread-out minADE quantiles, so the panels show good and bad cases."""
    rest = ev[ev["clip"] != example].sort_values("minade6_m").reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(rest) - 1, k - 1)).astype(int))
    return [example] + rest["clip"][idx].tolist()


def bev(res: Path, out: Path, example: str):
    ev = pd.read_csv(res / "eval_results.csv")
    z = np.load(res / "preds.npz")
    clips = pick(ev, example)
    fig, axs = plt.subplots(1, len(clips), figsize=(ps.DOUBLE_COLUMN_IN, 2.35), constrained_layout=True)
    for i, (ax, c) in enumerate(zip(axs, clips)):
        pred, gt, hist = z[f"{c}|pred"], z[f"{c}|gt"], z[f"{c}|hist"]
        for j, p in enumerate(pred):  # BEV: lateral y (left positive) on the x axis flipped, forward x up
            ax.plot(-p[:, 1], p[:, 0], color=ps.PREDICTION, lw=.7, alpha=.75, label="pred (6 samples)" if j == 0 else None)
        ax.plot(-gt[:, 1], gt[:, 0], color="black", lw=1.1, ls="--", label="ground truth")
        ax.plot(-hist[:, 1], hist[:, 0], color=ps.BASELINE, lw=1.1, label="history (1.5 s)")
        ax.plot(0, 0, marker="^", color="black", ms=3.5)
        # lateral axis stretched (not equal scale) so sub-metre lateral error is visible
        lat = np.abs(np.r_[pred[..., 1].ravel(), gt[:, 1], hist[:, 1]]).max()
        ax.set_xlim(-max(3, 1.3 * lat), max(3, 1.3 * lat))
        m = ev.set_index("clip").loc[c]
        ax.set_xlabel("right (m)")
        ps.panel(ax, f"({'abcdef'[i]}) {m.minade6_m:.2f} m")
        if i == 0:
            ax.set_ylabel("forward (m)")
    axs[0].plot([], [])  # legend sits in the empty right half of panel (b)
    axs[1].legend(*axs[0].get_legend_handles_labels(), loc="center right", fontsize=6.5, handlelength=1.4)
    info = ps.save(fig, out / "alpamayo-bev")
    return clips, info


def latency(res: Path, out: Path):
    df = pd.read_csv(res / "bench_results.csv")
    df = df[df.p50_ms.notna()]
    order = ["default/n1", "sdpa/n1", "expert-cuda-graph/n1", "flow-5/n1", "flow-2/n1",
             "reason-cap-16/n1", "no-reasoning/n1", "compile-expert/n1", "stack: compile-expert+flow-5/n1",
             "stack + no-reasoning/n1", "sdpa + compile-visual+expert/n1", "sdpa stack: compile-visual+expert+flow-5/n1",
             "sdpa stack + no-reasoning/n1",
             "default/n6", "sdpa/n6", "flow-5/n6", "no-reasoning/n6", "1-rollout x 6 flow samples/n6",
             "stack: compile-expert+flow-5/n6", "sdpa + compile-visual+expert/n6",
             "sdpa stack: compile-visual+expert+flow-5/n6", "sdpa stack + no-reasoning/n6"]
    df = df.set_index("key").reindex([k for k in order if k in set(df.key)])
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 4.0), constrained_layout=True)
    y, left = np.arange(len(df))[::-1], np.zeros(len(df))
    for col, lab, colr in STAGES:
        v = df[col].clip(lower=0).to_numpy()
        ax.barh(y, v, left=left, color=colr, height=.7, label=lab)
        left += v
    ax.scatter(df.p50_ms, y, marker="|", color="black", s=40, zorder=3, label="wall p50")
    ax.set_yticks(y, [k.replace("/n", "  n=") for k in df.index])
    ax.set_xlabel("latency per inference (ms), batch 1, 4 cams x 4 frames")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(.4, 1.0), fontsize=7)
    return ps.save(fig, out / "alpamayo-latency")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=Path("research/results/alpamayo-smoke"))
    ap.add_argument("--out", type=Path, default=Path("research/figs"))
    ap.add_argument("--example", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    a = ap.parse_args()
    ps.apply()
    clips, info = bev(a.res, a.out, a.example)
    cot = json.loads((a.res / "cot.json").read_text())
    for i, c in enumerate(clips):
        print(f"({'abcdef'[i]}) {c}: " + " | ".join(dict.fromkeys(cot[c])))
    print(info)
    if (a.res / "bench_results.csv").exists():
        print(latency(a.res, a.out))
