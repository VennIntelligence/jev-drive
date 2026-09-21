"""The project's one plot style, and the figures made from a planner run's results.csv.

CVPR camera-ready settings: serif text (STIX, the free Times metric-compatible face), 8-9 pt, column width
3.25 in and page width 6.875 in, mathtext for symbols, no title inside the figure (captions live in the
documents), and the colour-blind-safe Okabe-Ito palette with one fixed colour per backbone and grey for every
baseline. Every figure is written twice: PDF for the paper, 300 dpi PNG for research/ and todos/.
"""
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL, PAGE = 3.25, 6.875  # inches
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR = {"qwen_mean": "#0072B2", "qwen_last": "#D55E00", "dinov2": "#009E73", "vision": "#E69F00",
         "baseline": "#7F7F7F", "oracle": "#CC79A7"}
STYLE = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
         "lines.linewidth": 1.0, "lines.markersize": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "grid.linewidth": 0.4, "grid.alpha": 0.4, "figure.dpi": 300, "savefig.bbox": "tight",
         "savefig.pad_inches": 0.01, "axes.prop_cycle": mpl.cycler(color=OKABE_ITO)}


def legend_below(fig, ax, ncol: int | None = None):
    """One legend for the whole figure, under the panels: inside the axes it sits on the curves."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol or len(l), columnspacing=1.4,
               handlelength=1.8, borderaxespad=0.6)


def save(fig, out_dir, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def layer_curve(res: pd.DataFrame, out_dir, k_ref: int):
    """ADE of the vocabulary classifier and of ridge regression against Qwen decoder depth."""
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.1), sharey=True)
        for ax, head in zip(axes, ("cls", "ridge")):
            r = res[(res["head"] == head) & res["features"].str.contains("/L") & res["K"].isin((0, k_ref))]
            for pool in ("mean", "last"):
                q = r[r["features"].str.endswith("_" + pool)].copy()
                if not len(q):
                    continue
                q["layer"] = q["features"].str.extract(r"/L(\d+)_").astype(int)
                q = q.sort_values("layer")
                ax.plot(q.layer, q.ade, "-o", color=COLOR[f"qwen_{pool}"], label=f"Qwen3-VL {pool}-pooled")
            for name, sub, c in (("Qwen ViT (vis_mean)", "vis_mean", COLOR["vision"]),
                                 ("DINOv2 patch mean", "patch_mean", COLOR["dinov2"])):
                v = res[(res["head"] == head) & res["features"].str.endswith(sub)]
                if len(v):
                    ax.axhline(v.ade.iloc[0], color=c, ls="--", label=name)
            for name, c in (("ego state only", COLOR["baseline"]), ):
                v = res[(res["head"] == head) & (res["features"] == "ego")]
                if len(v):
                    ax.axhline(v.ade.iloc[0], color=c, ls=":", label=name)
            ax.set_xlabel("Qwen3-VL decoder layer")
            ax.set_title(f"{'vocabulary classifier' if head == 'cls' else 'waypoint regression'}"
                         f"{f' (K = {k_ref})' if head == 'cls' else ''}", fontsize=8)
            ax.grid(True, axis="y")
        axes[0].set_ylabel("ADE at 3 s (m)")
        legend_below(fig, axes[0])  # inside the axes it covers layers 19-31
        save(fig, out_dir, "layer_curve")


def k_sweep(res: pd.DataFrame, out_dir, best: str):
    """What the vocabulary costs and what the scorer loses on top of it, in displacement and in the
    trust-region miss rate that stands in for Waymo's floored fraction."""
    o = res[res["head"] == "oracle"].sort_values("K")
    cls = res[(res["head"] == "cls") & (res["features"] == best)].sort_values("K")
    base = [("const. turn rate", ("const_turn_rate", "-"), COLOR["baseline"], "--"),
            ("ego-state ridge", ("ridge", "ego"), "#333333", ":")]
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2))
        for ax, col, top, lab in ((axes[0], "ade", "minade10", "ADE at 3 s (m)"),
                                  (axes[1], "miss", "miss10", "outside the trust region (fraction)")):
            ax.plot(o.K, o[col], "-o", color=COLOR["oracle"], label="vocabulary floor (best anchor)")
            ax.plot(cls.K, cls[col], "-o", color=COLOR["qwen_mean"], label="classifier, top-1")
            ax.plot(cls.K, cls[top], "-o", color=COLOR["qwen_last"], label="classifier, best of top 10")
            for name, key, c, ls in base:
                v = res[(res["head"] == key[0]) & (res["features"] == key[1])]
                if len(v):
                    ax.axhline(v[col].iloc[0], color=c, ls=ls, label=name)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(sorted(o.K.unique()))
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.set_xlabel("vocabulary size $K$")
            ax.set_ylabel(lab)
            ax.grid(True, which="major")
        axes[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 4])
        axes[1].set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        legend_below(fig, axes[0])
        save(fig, out_dir, "k_sweep")


def run(res: pd.DataFrame, out_dir, best: str, k_ref: int):
    layer_curve(res, out_dir, k_ref)
    k_sweep(res, out_dir, best)
    return [out_dir / f"{n}.png" for n in ("layer_curve", "k_sweep")]


def probe_layer_curve(res: pd.DataFrame, out_dir, protocol: str = "kfold"):
    """Turn-probe NLL against Qwen decoder depth, for every input width, next to the baselines.
    Left: all samples. Right: the hard subset (the ego is not turning yet), where vision has to do the work.
    """
    r = res[res.protocol == protocol]
    wide = r.pivot_table(index="set", columns="subset", values="nll")
    line = lambda name: wide.loc[name] if name in wide.index else None  # noqa: E731
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.1))
        for ax, subset in zip(axes, ("all", "hard")):
            for width, ls in (("qwen", "-"), ("qwen_w800", "--")):
                for pool in ("mean", "last"):
                    q = wide[wide.index.str.fullmatch(rf"{width}/L\d+_{pool}")].copy()
                    if not len(q):
                        continue
                    layer = q.index.str.extract(r"/L(\d+)_")[0].astype(int)
                    q = q.assign(layer=layer.to_numpy()).sort_values("layer")
                    ax.plot(q.layer, q[subset], ls, marker="o", color=COLOR[f"qwen_{pool}"],
                            label=f"Qwen {pool}-pooled, {'1600' if width == 'qwen' else '800'} px")
            for name, key, c, ls in (("Qwen ViT output", "qwen/vis_mean", COLOR["vision"], "--"),
                                     ("DINOv2 patch mean", "dinov2/patch_mean", COLOR["dinov2"], "--"),
                                     ("ego state only", "ego", COLOR["baseline"], ":"),
                                     ("majority class", "majority", COLOR["baseline"], "-.")):
                v = line(key)
                if v is not None:
                    ax.axhline(v[subset], color=c, ls=ls, label=name)
            ax.set_xlabel("Qwen3-VL decoder layer")
            ax.set_title("all samples" if subset == "all" else "hard subset (not turning yet)", fontsize=8)
            ax.grid(True, axis="y")
        axes[0].set_ylabel("turn NLL (nats, lower is better)")
        axes[1].set_ylabel("turn NLL (nats)")
        axes[0].legend(loc="upper center", ncol=2, fontsize=5.5)
        save(fig, out_dir, "probe_layer_curve")
    return out_dir / "probe_layer_curve.png"


def l0_decile_curve(dec: pd.DataFrame, out_dir, arms=("A ridge_late uniform", "B ego:a1", "D mlp uniform")):
    """Relative gain of each arm over the ego ridge against the evaluation half's own s_ego decile.

    Left panel all evaluation frames, right panel straight_yaw only, so that a lateral confound cannot be
    the explanation. Both directions of the half-val split are drawn, solid and dashed. Bands are the paired
    sequence-bootstrap 95 % interval of the absolute delta, divided by that decile's ego ADE.

    The y axis is clipped: in the lowest deciles the ego prior is already almost exact and vision costs 40 to
    250 % of a very small error, which would flatten the whole range that carries the result. Those values
    are in deciles.csv and are quoted in the document that shows this figure.
    """
    lab = {"A ridge_late uniform": "ridge$_{late}$ uniform", "B ego:a1": r"ridge$_{late}$ $1+s_{ego}$",
           "D mlp uniform": "MLP uniform"}
    col = {"A ridge_late uniform": COLOR["qwen_mean"], "B ego:a1": COLOR["vision"],
           "D mlp uniform": COLOR["oracle"]}
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.3), sharey=True)
        for ax, scope, title in zip(axes, ("all", "straight_yaw"), ("all frames", "straight frames only")):
            ax.axhline(0, color=COLOR["baseline"], lw=0.6, zorder=1)
            for arm in arms:
                for d, ls in ((0, "-"), (1, "--")):
                    g = dec[(dec.scope == scope) & (dec.arm == arm) & (dec.direction == d)].sort_values("decile")
                    if not len(g):
                        continue
                    x = g.decile.to_numpy() + 1
                    ax.plot(x, 100 * g.rel_gain, ls, color=col[arm], marker="o" if d == 0 else None,
                            label=lab[arm] if d == 0 else None)
                    if d == 0:
                        ax.fill_between(x, 100 * g.rel_lo, 100 * g.rel_hi, color=col[arm], alpha=0.15, lw=0)
            ax.set_ylim(32, -14)
            ax.set_xlabel(r"decile of $s_{ego}$ on the evaluation half")
            ax.set_xticks(range(1, 11))
            ax.text(0.03, 0.06, title, transform=ax.transAxes)
        axes[0].set_ylabel("relative gain over ego (%)")
        legend_below(fig, axes[0])
        save(fig, out_dir, "l0-decile-relative-gain")
