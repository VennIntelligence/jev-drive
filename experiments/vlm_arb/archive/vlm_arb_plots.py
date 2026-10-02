"""Generate publication-quality figures for VLM arbitration experiment.

Protocol: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_paired_effects(df_paired: pd.DataFrame, out_path: Path):
    """Plot paired DS differences with 95% bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    
    y_pos = np.arange(len(df_paired))
    means = df_paired["delta_mean"]
    errors = [
        means - df_paired["ci_lo"],
        df_paired["ci_hi"] - means
    ]
    
    colors = ["#2b5c8f" if m >= 0 else "#c0392b" for m in means]
    ax.errorbar(means, y_pos, xerr=errors, fmt="o", color="#2b5c8f", ecolor="#7f8c8d",
                elinewidth=2, capsize=4, capthick=1.5, markersize=6)
    
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_paired["contrast"])
    ax.set_xlabel("Paired Δ Driving Score (95% CI)")
    ax.set_title("VLM Arbitration Arms vs Baseline (2000 Route Bootstrap)")
    
    fig.savefig(out_path)
    plt.close(fig)


def plot_latency_comparison(df_models: pd.DataFrame, out_path: Path):
    """Bar chart comparing p50 and p95 latencies across models."""
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    
    x = np.arange(len(df_models))
    width = 0.35
    
    p50 = df_models["p50_latency_ms"]
    p95 = df_models["p95_latency_ms"]
    
    ax.bar(x - width/2, p50, width, label="p50 (median)", color="#3498db")
    ax.bar(x + width/2, p95, width, label="p95 (tail)", color="#e67e22")
    
    ax.axhline(600, color="red", linestyle=":", label="600 ms Closed-Loop Limit")
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("VLM Latency on Batch 1 CARLA Front Cameras")
    ax.set_xticks(x)
    ax.set_xticklabels(df_models["model"], rotation=15, ha="right")
    ax.legend(frameon=False)
    
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot figures for VLM arbitration")
    parser.add_argument("--results-dir", type=str, default="experiments/vlm_arb/results", help="Directory with csv results")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    paired_file = res_dir / "paired_effects.csv"
    if paired_file.exists():
        df_paired = pd.read_csv(paired_file)
        plot_paired_effects(df_paired, res_dir / "paired_effects.png")
        print("Plotted paired_effects.png")

    model_file = res_dir / "model_comparison.csv"
    if model_file.exists():
        df_models = pd.read_csv(model_file)
        plot_latency_comparison(df_models, res_dir / "latency_comparison.png")
        print("Plotted latency_comparison.png")


if __name__ == "__main__":
    main()
