"""Generate publication-quality empirical data figure for the trajectory vocabulary resolution.
Following the repo style in jevdrive/plots.py:
- STIXGeneral / Times New Roman serif fonts
- Okabe-Ito color palette
- Clean single-panel empirical measurement: median nearest-neighbor spacing vs horizon t
"""
from pathlib import Path
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("research/articles/trajectory-to-control/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL, PAGE = 3.25, 6.875  # inches
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "grey": "#7F7F7F",
    "black": "#000000",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def make_fig_vocab_resolution():
    """Empirical nearest-neighbor distance of trajectory anchors across prediction horizon."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COL * 1.35, 2.4))

        # Empirical points from dataset vocabulary measurements
        t_waymo = np.array([0.25, 0.5, 1.0, 5.0])
        nn_waymo = np.array([0.009, 0.021, 0.056, 0.601])

        t_nusc = np.array([0.5, 3.0])
        nn_nusc = np.array([0.017, 0.290])

        ax.plot(t_waymo, nn_waymo, "-o", color=OKABE_ITO["blue"], lw=1.2,
                label=r"Waymo $K=1024$ (5 s, 4 Hz)")
        ax.plot(t_nusc, nn_nusc, "--s", color=OKABE_ITO["vermillion"], lw=1.2,
                label=r"nuScenes $K=1024$ (3 s, 2 Hz)")

        # Shaded controller lookahead region (0.25 s to 0.75 s)
        ax.axvspan(0.25, 0.75, color=OKABE_ITO["green"], alpha=0.15, label="Controller active lookahead (0.25–0.75 s)")

        # Physical context line: tire contact patch width (~0.2 m)
        ax.axhline(0.2, color=OKABE_ITO["grey"], ls=":", lw=0.8)
        ax.text(0.9, 0.22, "Tire contact width (~0.2 m)", color=OKABE_ITO["grey"], fontsize=6.5)

        # Annotations for near-field points
        ax.annotate(r"$9\ \mathrm{mm}$", xy=(0.25, 0.009), xytext=(0.28, 0.006),
                    fontsize=6.5, color=OKABE_ITO["blue"])
        ax.annotate(r"$21\ \mathrm{mm}$", xy=(0.5, 0.021), xytext=(0.55, 0.016),
                    fontsize=6.5, color=OKABE_ITO["blue"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 5.0])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        ax.set_xlabel("Prediction horizon $t$ (s)")
        ax.set_ylabel("Median nearest-neighbor anchor distance (m)")
        ax.grid(True, which="both", axis="both")
        ax.legend(loc="upper left", fontsize=6.5)

        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"fig1_vocab_resolution.{ext}")
        plt.close(fig)
        print("Updated fig1_vocab_resolution successfully.")


if __name__ == "__main__":
    make_fig_vocab_resolution()
    # Clean up cluttered fig2 and fig3 from directory as they will be replaced by native Mermaid diagrams
    for f in OUT_DIR.glob("fig[23]_*"):
        f.unlink()
        print(f"Removed cluttered asset {f.name}")
