"""Figures for the HUGSIM render check (scripts/hugsim/render_check.py output pulled to this Mac).

  python scripts/hugsim/render_figs.py <render_check out dir> research/figs
Writes hugsim-render-check.{pdf,png} (reference vs render, held-out frames) and hugsim-render-psnr.{pdf,png}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jevdrive.plots import COLOR, PAGE, STYLE, legend_below, plt, save  # noqa: E402


def main(src: Path, out: Path):
    plt.rcParams.update(STYLE)
    df = pd.read_csv(src / "render_psnr.csv")
    order = {"CAM_FRONT_LEFT": 0, "CAM_FRONT": 1, "CAM_FRONT_RIGHT": 2}
    cam = df.set_index("idx").cam
    frames = sorted(src.glob("frame_*.npz"), key=lambda f: order.get(cam[int(f.stem.split("_")[1])], 9))
    fig, axes = plt.subplots(2, len(frames), figsize=(PAGE, PAGE * 0.5 * 2 * 450 / 800 / len(frames) * 1.9))
    for j, f in enumerate(frames):
        idx = int(f.stem.split("_")[1])
        d = np.load(f)
        row = df[df.idx == idx].iloc[0]
        for i, (key, lab) in enumerate((("gt", "recorded"), ("pred", "HUGSIM render"))):
            ax = axes[i, j]
            ax.imshow(d[key])
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(lab)
        axes[1, j].set_xlabel(f"{row.cam}, frame {idx} ({row.split}), PSNR {row.psnr:.1f} dB")
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    save(fig, out, "hugsim-render-check")

    fig, ax = plt.subplots(figsize=(PAGE / 2, 1.6))
    for split, c in (("train", COLOR["baseline"]), ("test", COLOR["qwen_last"])):
        s = df[df.split == split]
        ax.scatter(s.t, s.psnr, s=2, color=c, label=f"{split} views (mean {s.psnr.mean():.2f} dB)")
    ax.set_xlabel("time in the recorded log [s]")
    ax.set_ylabel("PSNR vs recorded [dB]")
    legend_below(fig, ax)
    save(fig, out, "hugsim-render-psnr")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
