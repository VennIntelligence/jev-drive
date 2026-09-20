"""Figures for research/survey-thin-head.md.

Every number is copied from the paper cited in the survey's reference list; nothing here comes from our own
runs. Concept diagrams redraw the mechanism, not a pipeline logo. Style, palette and sizes come from
jevdrive.plots so these figures match the rest of the repo.

Run from the repo root:  .venv/bin/python research/figs/survey/make_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from jevdrive.plots import COL, PAGE, STYLE, legend_below, save

OUT = Path(__file__).parent
# One colour per way of using the backbone, reused across every figure.
C_FROZEN, C_VLM, C_SMALL, C_VIDEO, C_GREY, C_ADAPT = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F", "#E69F00"


# ----------------------------------------------------------------------------------------------- helpers
def box(ax, x, y, w, h, text, fc="white", ec="black", ls="-", fs=7, hatch=None, tc="black", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05", fc=fc, ec=ec,
                                ls=ls, lw=0.7, hatch=hatch))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.25,
            weight=weight)


def arrow(ax, x0, y0, x1, y1, ls="-", color="black", lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=7))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# ----------------------------------------------------------------------------------- 1. world-model roles
def world_model_roles():
    """Three places a driving world model can sit; what still runs at inference differs."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE, 3.1))
        blank(ax, (0, 10), (-0.6, 3.9))
        h, y = 0.55, {0: 3.25, 1: 2.0, 2: 0.25}
        rows = [("Online latent rollout", "World4Drive, 2025-07"),
                ("Generative feature extraction", "DriveLaW, 2025-12"),
                ("Predictive pretraining", "LAW 2024-06; Drive-JEPA 2026-01")]
        for i, (name, ex) in enumerate(rows):
            ax.text(0.0, y[i] + h / 2 + 0.11, name, fontsize=7.5, weight="bold", va="center")
            ax.text(0.0, y[i] + h / 2 - 0.13, ex, fontsize=6.5, va="center", color="#333333")
        x0 = 2.75
        # row 0: rollout kept online
        chain0 = [("past frames", 0.95), ("encoder", 0.85), ("latent future\nper intention", 1.35),
                  ("score and\nselect", 0.95)]
        x = x0
        for j, (t, w) in enumerate(chain0):
            box(ax, x, y[0], w, h, t)
            if j:
                arrow(ax, x - 0.2, y[0] + h / 2, x, y[0] + h / 2)
            x += w + 0.2
        ax.text(x + 0.05, y[0] + h / 2, "trajectory", fontsize=7, va="center")
        arrow(ax, x - 0.2, y[0] + h / 2, x, y[0] + h / 2)
        # row 1: early generative features, decoder skipped
        chain1 = [("past frames", 0.95), ("video DiT,\n1st denoising step", 1.35), ("early\nfeatures", 0.85),
                  ("action DiT", 0.95)]
        x = x0
        xs = []
        for j, (t, w) in enumerate(chain1):
            box(ax, x, y[1], w, h, t)
            xs.append(x)
            if j:
                arrow(ax, x - 0.2, y[1] + h / 2, x, y[1] + h / 2)
            x += w + 0.2
        ax.text(x + 0.05, y[1] + h / 2, "trajectory", fontsize=7, va="center")
        arrow(ax, x - 0.2, y[1] + h / 2, x, y[1] + h / 2)
        bx = xs[2] + 0.1
        box(ax, bx, y[1] - 0.78, 1.05, 0.42, "RGB video\ndecoder", ls="--", ec=C_GREY, tc=C_GREY, fs=6.5)
        arrow(ax, xs[1] + 1.35 - 0.1, y[1], bx + 0.3, y[1] - 0.36, ls="--", color=C_GREY)
        ax.text(bx + 1.15, y[1] - 0.57, "not run at deployment", fontsize=6.5, color=C_GREY, va="center")
        # row 2: prediction only in the loss
        chain2 = [("past frames", 0.95), ("encoder", 0.85), ("planner", 0.85)]
        x = x0
        xs = []
        for j, (t, w) in enumerate(chain2):
            box(ax, x, y[2], w, h, t)
            xs.append(x)
            if j:
                arrow(ax, x - 0.2, y[2] + h / 2, x, y[2] + h / 2)
            x += w + 0.2
        ax.text(x + 0.05, y[2] + h / 2, "trajectory", fontsize=7, va="center")
        arrow(ax, x - 0.2, y[2] + h / 2, x, y[2] + h / 2)
        box(ax, xs[1] - 0.35, y[2] + 0.72, 1.55, 0.42, "future-latent\nprediction loss", hatch="////",
            fc="#F2F2F2", fs=6.5)
        arrow(ax, xs[1] + 0.42, y[2] + 0.72, xs[1] + 0.42, y[2] + h, ls="--")
        ax.text(xs[1] - 0.45, y[2] + 0.93, "training only", fontsize=6.5, color="#333333", va="center", ha="right")
        # legend
        lx, ly = 6.95, -0.2
        box(ax, lx, ly, 0.35, 0.22, "", fs=1)
        ax.text(lx + 0.45, ly + 0.11, "runs at inference", fontsize=6.5, va="center")
        box(ax, lx, ly - 0.32, 0.35, 0.22, "", hatch="////", fc="#F2F2F2", fs=1)
        ax.text(lx + 0.45, ly - 0.21, "training only", fontsize=6.5, va="center")
        box(ax, lx + 1.6, ly, 0.35, 0.22, "", ls="--", ec=C_GREY, fs=1)
        ax.text(lx + 2.05, ly + 0.11, "removed at deployment", fontsize=6.5, va="center")
        save(fig, OUT, "f1_world_model_roles")


# --------------------------------------------------------------------------- 2. VLA output interfaces
def output_interfaces():
    """Four output interfaces of driving VLAs, ordered by how much is generated per decision."""
    cols = [("Full language\ngeneration", "scene description,\nreasoning, trajectory\nall as text",
             "DriveVLM 2024-02\nEMMA 2024-10"),
            ("Short autoregressive\naction sequence", "about 10 action tokens;\nCoT optional",
             "OpenDriveVLA 2025-03\nAutoVLA 2025-06"),
            ("Hidden state to a\ncontinuous head", "diffusion / flow trajectory\nfrom one token;\nreasoning optional",
             "ORION 2025-03\nAlpamayo-R1 2025-10\nQwen-Drive-1.0 2026-09"),
            ("Fixed candidates,\none decision", "nothing generated;\nK candidates scored\nin one forward pass",
             "traffic-concept probes 2026-03\nJev-style scoring 2026-09")]
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE, 2.55))
        blank(ax, (0, 10), (-0.55, 3.15))
        w, gap = 2.05, 0.6
        x = 0.0
        for j, (head, body, ex) in enumerate(cols):
            box(ax, x, 2.45, w, 0.6, head, fc="#EFEFEF", fs=7.5, weight="bold")
            box(ax, x, 1.2, w, 1.1, body, fs=6.8)
            ax.text(x + w / 2, 0.95, "generated at inference", fontsize=6, ha="center", color="#555555",
                    style="italic")
            ax.text(x + w / 2, 0.45, ex, fontsize=6.3, ha="center", va="center", color="#333333", linespacing=1.3)
            if j < 3:
                xa = x + w
                ls = "--" if j == 2 else "-"
                arrow(ax, xa + 0.05, 1.75, xa + gap - 0.05, 1.75, ls=ls, color="black" if j < 2 else C_GREY)
            x += w + gap
        arrow(ax, 0.2, -0.25, 9.75, -0.25, color="#333333")
        ax.text(5.0, -0.45, "fewer tokens generated per decision; the dashed step is not a demonstrated lossless conversion",
                fontsize=6.5, ha="center", color="#333333")
        save(fig, OUT, "f2_output_interfaces")


# ---------------------------------------------------------------------- 3. Alpamayo-R1 latency breakdown
def alpamayo_latency():
    """Same model, same GPU: what the output format costs versus what reasoning costs (Table 14)."""
    labels = ["reasoning (40 tokens)\n+ AR trajectory (127 tokens)", "reasoning (40 tokens)\n+ flow trajectory (5 steps)",
              "flow trajectory only"]
    ms = [312, 99, 29]
    colors = [C_VLM, C_ADAPT, C_FROZEN]
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COL, 1.75))
        y = np.arange(3)[::-1]
        ax.barh(y, ms, color=colors, height=0.6)
        for yi, v in zip(y, ms):
            ax.text(v + 6, yi, f"{v} ms", va="center", fontsize=7)
        ax.set_yticks(y, labels)
        ax.set_xlabel("latency, batch 1 (ms); RTX 6000 Pro Blackwell")
        ax.set_xlim(0, 400)
        ax.grid(True, axis="x")
        ax.annotate("", xy=(99, 1.42), xytext=(312, 1.42),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", mutation_scale=7))
        ax.text(205, 1.55, "$-213$ ms: trajectory serialisation", fontsize=6.5, ha="center", color="#333333")
        ax.annotate("", xy=(29, 0.42), xytext=(99, 0.42),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", mutation_scale=7))
        ax.text(64, 0.55, "$-70$ ms: reasoning", fontsize=6.5, ha="center", color="#333333")
        save(fig, OUT, "f3_alpamayo_latency")


# ---------------------------------------------------------- 4. what is fed to the head: same-planner ablations
def frontend_ablations():
    """Same planner, different front-end, all NAVSIM v1 PDMS. (a) Drive-JEPA Tables 5-6, (b) DriveLaW Table 5,
    (c) DriveLaW Table 6."""
    panels = [("Drive-JEPA: pretraining of the encoder",
               ["ImageNet\nResNet-34", "DINOv2\nViT-L", "V-JEPA 2\nViT-L", "driving\nJEPA"], [76.0, 76.1, 86.1, 89.0],
               [C_SMALL, C_SMALL, C_FROZEN, C_FROZEN]),
              ("DriveLaW: front-end, same action DiT",
               ["BEV", "VLM", "video\ngenerator"], [84.1, 86.5, 89.1], [C_SMALL, C_VLM, C_VIDEO]),
              ("DriveLaW: denoising step fed to the planner",
               ["step 1", "step 5", "step 10"], [89.1, 86.9, 23.2], [C_VIDEO, C_VIDEO, C_VIDEO])]
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(PAGE, 1.9), gridspec_kw=dict(width_ratios=[4, 3, 3]))
        for ax, (title, names, vals, cols) in zip(axes, panels):
            x = np.arange(len(vals))
            ax.bar(x, vals, color=cols, width=0.62)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=6.5)
            ax.set_xticks(x, names, fontsize=6.5)
            ax.set_title(title, fontsize=7)
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y")
        axes[0].set_ylabel("NAVSIM v1 PDMS")
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        save(fig, OUT, "f4_frontend_ablations")


# ------------------------------------------------------------------------ 5. Waymo E2E official-test scatter
WAYMO = [  # name, RFS, ADE@5s (m), group, label offset (points)
    ("RAP", 8.0430, 2.6457, "small", (4, 3)),
    ("Poutine", 7.9860, 2.7419, "vlm", (4, 3)),
    ("SUV", 7.94, 2.90, "video", (4, 2)),
    ("Qwen-Drive-1.0 (RL)", 7.91, 2.67, "vlm", (4, -7)),
    ("Poutine-Base", 7.909, 2.940, "vlm", (4, -7)),
    ("MindVLA-U1", 7.87, 2.66, "vlm", (-4, -8)),
    ("FROST-Drive", 7.8560, 3.5653, "frozen", (4, 3)),
    ("ViT-Adapter-GRU", 7.8493, 2.8888, "small", (4, 4)),
    ("Fast-dDrive", 7.823, 2.907, "vlm", (4, -7)),
    ("UniPlan", 7.7795, 2.8423, "small", (4, -8)),
    ("HMVLM", 7.7367, 3.0715, "vlm", (4, 3)),
    ("DiffusionLTF", 7.717, 2.977, "small", (4, -7)),
    ("dVLM-AD", 7.633, 3.022, "vlm", (4, 3)),
    ("AutoVLA", 7.556, 2.958, "vlm", (4, 3)),
    ("Swin-Trajectory", 7.543, 2.814, "small", (4, -8)),
    ("NaiveEMMA", 7.528, 3.018, "vlm", (4, -8)),
]
GROUP = {"frozen": ("frozen vision encoder + trained head", C_FROZEN, "o"),
         "vlm": ("VLM fine-tuned (SFT, some + RL)", C_VLM, "s"),
         "small": ("task model trained from a vision backbone", C_SMALL, "D"),
         "video": ("video generator + action expert", C_VIDEO, "^")}


def waymo_leaderboard():
    """Official-test RFS against ADE for every entry with a dated public record, coloured by how the
    backbone is used."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE, 3.0))
        for key, (lab, c, m) in GROUP.items():
            pts = [p for p in WAYMO if p[3] == key]
            ax.scatter([p[2] for p in pts], [p[1] for p in pts], color=c, marker=m, s=14, label=lab, zorder=3)
        for name, rfs, ade, g, (dx, dy) in WAYMO:
            ax.annotate(name, (ade, rfs), xytext=(dx, dy), textcoords="offset points", fontsize=5.8,
                        ha="left" if dx > 0 else "right", va="center", color="#222222")
        ax.set_xlabel("ADE at 5 s (m), lower is better")
        ax.set_ylabel("RFS, higher is better")
        ax.set_xlim(2.5, 3.75)
        ax.set_ylim(7.45, 8.12)
        ax.grid(True)
        ax.text(2.53, 8.09, "official test split; papers 2025-06 to 2026-09", fontsize=6.5, color="#555555",
                va="top")
        legend_below(fig, ax, ncol=2)
        save(fig, OUT, "f5_waymo_leaderboard")


# ------------------------------------------------------------------------------- 6. how crowded, by month
TIMELINE = {
    "Latent world models": [("LAW", "2024-06"), ("World4Drive", "2025-07"), ("DriveLaW", "2025-12"),
                            ("Drive-JEPA", "2026-01"), ("NTR", "2026-05"), ("WA-JEPA", "2026-08")],
    "Probing VLM representations": [("Hidden in plain sight", "2025-06"), ("Linear Mechanisms", "2026-01"),
                                    ("Probing Visual Concepts", "2026-03"), ("VLM vs video-gen probing", "2026-05"),
                                    ("Anchor-Align", "2026-07")],
    "Frozen backbone to planner": [("FROST-Drive", "2026-01"), ("Auto-JEPA", "2026-07"),
                                   ("Qwen-Drive vanilla+PE", "2026-09"), ("DiffAdapterVLA", "2026-09")],
    "Fast / slow, memory": [("FASIONAD++", "2025-03"), ("AutoVLA", "2025-06"), ("AdaThinkDrive", "2025-09"),
                            ("CF-VLA", "2025-12"), ("LaST-VLA", "2026-03"), ("ASSCG", "2026-06"),
                            ("Driving on Memory", "2026-08"), ("FIVE-VLA", "2026-09")],
    "Waymo E2E test entries": [("Poutine", "2025-06"), ("HMVLM", "2025-06"), ("AutoVLA", "2025-06"),
                               ("RAP", "2025-10"), ("WOD-E2E baselines", "2025-10"), ("dVLM-AD", "2025-12"),
                               ("FROST-Drive", "2026-01"), ("NoRD", "2026-02"), ("MindVLA-U1", "2026-05"),
                               ("Fast-dDrive", "2026-05"), ("VL-DPO", "2026-05"), ("SUV", "2026-08"),
                               ("Qwen-Drive-1.0", "2026-09")],
}
ROW_COLOR = {"Latent world models": C_VIDEO, "Probing VLM representations": C_ADAPT,
             "Frozen backbone to planner": C_FROZEN, "Fast / slow, memory": C_SMALL,
             "Waymo E2E test entries": C_VLM}


def ym(s):
    y, m = map(int, s.split("-"))
    return y + (m - 0.5) / 12


def crowding_timeline():
    """First arXiv month of every paper the survey leans on, by theme. The band is the last three months."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE, 3.3))
        rows = list(TIMELINE)
        ax.axvspan(2026 + 5.65 / 12, 2026 + 8.65 / 12, color="#F0E442", alpha=0.35, lw=0)
        ax.text(2026 + 7.15 / 12, len(rows) - 0.42, "last 3 months", fontsize=6.5, ha="center", color="#665f00")
        offsets = [0.2, 0.42, -0.24, -0.46]
        for r, name in enumerate(rows):
            yv = len(rows) - 1 - r
            ax.axhline(yv, color="#DDDDDD", lw=0.5, zorder=1)
            ents = sorted(TIMELINE[name], key=lambda e: ym(e[1]))
            seen = {}
            for k, (paper, date) in enumerate(ents):
                x = ym(date) + 0.018 * seen.get(date, 0)
                seen[date] = seen.get(date, 0) + 1
                ax.scatter([x], [yv], color=ROW_COLOR[name], s=11, zorder=3)
                dy = offsets[k % len(offsets)]
                ax.annotate(paper, (x, yv), xytext=(0, dy * 28), textcoords="offset points", fontsize=5.4,
                            ha="center", va="center", color="#222222",
                            arrowprops=dict(arrowstyle="-", lw=0.3, color="#999999", shrinkA=0, shrinkB=1.5))
        ax.set_yticks(range(len(rows)), rows[::-1], fontsize=6.8)
        ticks = [2024.5, 2025.0, 2025.5, 2026.0, 2026.5]
        ax.set_xticks(ticks, ["2024-07", "2025-01", "2025-07", "2026-01", "2026-07"])
        ax.set_xlim(2024.35, 2026.85)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel("first arXiv submission (month)")
        for s in ("left",):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        save(fig, OUT, "f6_crowding_timeline")


# ------------------------------------------------------------------------------ 7. trajectory vocabulary K
def vocab_k():
    """Published K sweeps. Each series is its own system on NAVSIM v1; compare within a series only.
    DiffusionDrive sweeps inference candidates around 20 anchors, not a vocabulary."""
    series = [("Hydra-MDP vocabulary", [4096, 8192], [82.6, 83.0], C_VLM, "o"),
              ("WoTE vocabulary", [64, 128, 256], [84.5, 85.4, 85.6], C_FROZEN, "s"),
              ("DiffusionDrive inference candidates", [10, 20, 40], [84.9, 88.1, 88.2], C_SMALL, "D")]
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COL, 2.0))
        for lab, k, v, c, m in series:
            ax.plot(k, v, "-", marker=m, color=c, label=lab)
            for ki, vi in zip(k, v):
                ax.annotate(f"{vi}", (ki, vi), xytext=(0, 4), textcoords="offset points", fontsize=5.8,
                            ha="center", color=c)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 4096, 16384])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel("number of candidates $K$")
        ax.set_ylabel("NAVSIM v1 PDMS")
        ax.set_ylim(80, 91)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=6)
        save(fig, OUT, "f7_vocab_k")


# --------------------------------------------------------------------------- 8. fast / slow: matched pairs
def fast_slow():
    """(a) AdaThinkDrive Tables II-III, NAVSIM v1 navtest. (b) ASSCG Table 2, nuPlan Hard20 closed-loop."""
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.0))
        a = [("never think", 0.68, 88.3, C_FROZEN), ("always think", 0.86, 88.9, C_VLM),
             ("adaptive", 0.74, 90.3, C_SMALL)]
        for lab, t, s, c in a:
            axes[0].scatter([t], [s], color=c, s=18, zorder=3)
            axes[0].annotate(lab, (t, s), xytext=(5, 3), textcoords="offset points", fontsize=6.5)
        axes[0].set_xlabel("mean inference time (s)")
        axes[0].set_ylabel("NAVSIM v1 PDMS")
        axes[0].set_xlim(0.6, 1.0)
        axes[0].set_ylim(87.8, 91)
        axes[0].set_title("AdaThinkDrive: same VLA, three policies", fontsize=7)
        b = [("slow model every frame", 0.80, 65.00, C_VLM), ("fixed interval", 0.32, 64.27, C_GREY),
             ("learned gate (ASSCG)", 0.32, 67.28, C_SMALL)]
        for lab, t, s, c in b:
            axes[1].scatter([t], [s], color=c, s=18, zorder=3)
            axes[1].annotate(lab, (t, s), xytext=(5, 3 if s < 66 else -8), textcoords="offset points", fontsize=6.5)
        axes[1].set_xlabel("mean compute per frame (s)")
        axes[1].set_ylabel("nuPlan Hard20 closed-loop score")
        axes[1].set_xlim(0.2, 1.0)
        axes[1].set_ylim(63.5, 68.5)
        axes[1].set_title("ASSCG: when to query the slow VLM", fontsize=7)
        for ax in axes:
            ax.grid(True)
        save(fig, OUT, "f8_fast_slow")


# ------------------------------------------------------------------------------------ 9. evidence ladder
def evidence_ladder():
    """Each step to the right needs new evidence; success on the left does not carry over."""
    steps = [("logged open-loop", "nuScenes L2", "matches the log"),
             ("rater-scored open-loop", "Waymo E2E RFS", "matches human\npreference"),
             ("non-reactive simulation", "NAVSIM v1 PDMS", "picks a safe, progressing\ntrajectory"),
             ("pseudo-simulation", "NAVSIM v2 EPDMS", "recovers from\noff-expert states"),
             ("closed-loop simulator", "Bench2Drive DS / SR", "drives and corrects\nits own errors"),
             ("closed-loop under\nperturbation", "Bench2Drive-Robust", "survives latency and\nsensor faults")]
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE, 2.5))
        blank(ax, (0, 12.4), (0, 4.6))
        w = 2.0
        for i, (name, bench, claim) in enumerate(steps):
            x, hgt = i * w + 0.2, 0.45 + 0.42 * i
            ax.add_patch(FancyBboxPatch((x, 0), w - 0.1, hgt, boxstyle="square,pad=0", fc="#E8EEF5", ec="#0072B2",
                                        lw=0.7))
            ax.text(x + (w - 0.1) / 2, hgt / 2, bench, ha="center", va="center", fontsize=6.5, color="#0072B2",
                    weight="bold")
            ax.text(x + (w - 0.1) / 2, hgt + 0.28, name, ha="center", va="bottom", fontsize=7, weight="bold")
            ax.text(x + (w - 0.1) / 2, hgt + 0.95, claim, ha="center", va="bottom", fontsize=6.3, color="#333333",
                    style="italic")
        arrow(ax, 0.3, 4.35, 12.1, 4.35, color="#333333")
        ax.text(6.2, 4.45, "each step needs its own evidence; a score on the left says nothing about the right",
                fontsize=6.5, ha="center", va="bottom", color="#333333")
        save(fig, OUT, "f9_evidence_ladder")


if __name__ == "__main__":
    for f in (world_model_roles, output_interfaces, alpamayo_latency, frontend_ablations, waymo_leaderboard,
              crowding_timeline, vocab_k, fast_slow, evidence_ladder):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
