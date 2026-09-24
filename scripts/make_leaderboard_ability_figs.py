"""Figures for the research article on leaderboard scores versus driving ability.

Three figures, written to research/figs/lb-ability-*.{pdf,png}:

  lb-ability-mechanism-map     where each board's observed score mechanisms sit in the sensor-to-score pipeline
                               (research/results/hack-audit/matrix.csv, first-round audit, present == yes)
  lb-ability-cross-board       published scores of the same method on two protocols, three panels
                               (research/results/leaderboard-text-analysis/w2_cross_board.csv)
  lb-ability-tfv6-two-channels TFv6's scoring channel versus its reacting channel
                               (numbers hard-coded from research/decisions.md, items 31 and 32)
  lb-ability-gain-ledger       the largest ablation gains per board, coloured by what kind of gain they are, and the
                               same ego prior scored open loop versus closed loop
                               (research/results/leaderboard-text-analysis/w1_ablation_ledger.csv)

Run from anywhere: `python -m scripts.make_leaderboard_ability_figs` or `python3 scripts/make_leaderboard_ability_figs.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from jevdrive.plots import COL, PAGE, STYLE, save  # noqa: E402

OUT = ROOT / "research" / "figs"
AUDIT = ROOT / "research" / "results" / "hack-audit" / "matrix.csv"
CROSS = ROOT / "research" / "results" / "leaderboard-text-analysis" / "w2_cross_board.csv"
LEDGER = ROOT / "research" / "results" / "leaderboard-text-analysis" / "w1_ablation_ledger.csv"

GREY, LIGHT = "#7F7F7F", "#BBBBBB"
# Entity colours shared across the three figures.
C_TSPEED, C_WAYPOINT, C_OURS = "#0072B2", "#D55E00", "#009E73"  # TFv6 route+target speed, TFv6 waypoint, our head
# Mechanism classes of the audit taxonomy.
C_CLASS = {"metric-specific": "#D55E00", "routine-layer control": "#0072B2",
           "non-deployable input": "#CC79A7", "protocol": "#009E73"}


# ----------------------------------------------------------------------------------------------------------------
# Figure 1: mechanism map
# ----------------------------------------------------------------------------------------------------------------
STAGES = ["Inputs", "Network /\ntraining", "Candidates /\noutputs", "Selection /\nscoring", "Control &\nrules",
          "Simulator /\nmetric"]
STAGE_OF = {
    "ego_only_open_loop_planning": 0, "ego_state_fusion": 0, "future_label_conditioning": 0,
    "simulator_pose_access": 0, "benchmark_prompt_specification": 0,
    "metric_reward_finetuning": 1, "benchmark_target_shaping": 1, "metric_grid_alignment": 1,
    "benchmark_split_adaptation": 1,
    "offline_candidate_library": 2, "multi_checkpoint_candidate_selection": 2,
    "metric_proxy_candidate_selection": 3, "metric_proxy_test_search": 3, "manual_scorer_reweighting": 3,
    "control_interface_selection": 4, "manual_control_override": 4, "metric_early_termination": 4,
    "inference_failure_brake_fallback": 4,
    "simulator_protocol_dependency": 5,
}
CLASS_OF = {
    "control_interface_selection": "routine-layer control", "manual_control_override": "routine-layer control",
    "future_label_conditioning": "non-deployable input", "simulator_pose_access": "non-deployable input",
    "ego_only_open_loop_planning": "non-deployable input", "ego_state_fusion": "non-deployable input",
    "simulator_protocol_dependency": "protocol", "inference_failure_brake_fallback": "protocol",
}  # everything else is metric-specific
SHORT = {
    "ego_only_open_loop_planning": "ego-only planning", "ego_state_fusion": "ego fusion",
    "future_label_conditioning": "future GT", "simulator_pose_access": "sim pose access",
    "benchmark_prompt_specification": "bench. prompt",
    "metric_reward_finetuning": "reward FT", "benchmark_target_shaping": "target shaping",
    "metric_grid_alignment": "grid align", "benchmark_split_adaptation": "split adaptation",
    "offline_candidate_library": "offline library", "multi_checkpoint_candidate_selection": "multi-ckpt pick",
    "metric_proxy_candidate_selection": "proxy select", "metric_proxy_test_search": "test-time search",
    "manual_scorer_reweighting": "scorer weights",
    "control_interface_selection": "control interface", "manual_control_override": "manual override",
    "metric_early_termination": "early stop", "inference_failure_brake_fallback": "fail\u2192brake",
    "simulator_protocol_dependency": "protocol",
}
REPO_ABBR = {
    "kesai-labs/lead": "TFv6", "George-Ling3/BLUE": "BLUE", "swc-17/SparseDriveV2": "SDv2",
    "autonomousvision/carla_garage": "TF++", "valeoai/DrivoR": "DrivoR", "valeoai/TOAD": "TOAD",
    "vita-epfl/RAP": "RAP", "ZebinX/DriveVLA-M0": "DVLA-M0", "NVlabs/GTRS": "GTRS", "E2E-AD/AD-MLP": "AD-MLP",
    "NVlabs/BEV-Planner": "BEV-P++", "MSunDYY/SparseOccVLA": "SOVLA", "Tsinghua-MARS-Lab/DriveMA": "DriveMA",
    "AFARI-Research/WA-JEPA": "WA-JEPA", "ucla-mobility/AutoVLA": "AutoVLA", "hyzhou404/UniAD_SIM": "UniAD",
    "hyzhou404/NAVSIM": "LTF",
}
BOARDS = [("bench2drive", "Bench2Drive", "closed loop, actions\nchange the world"),
          ("carla_lb2", "CARLA LB2", "closed loop, actions\nchange the world"),
          ("navsim_v1", "NAVSIM v1", "4 s, logged traffic\ndoes not react"),
          ("navsim_v2", "NAVSIM v2", "4 s + pre-generated\nfollow-up scenes"),
          ("nuscenes", "nuScenes", "3 s L2 vs logged ego"),
          ("wod_e2e", "WOD-E2E", "3 s & 5 s points vs\nrater trajectories"),
          ("hugsim", "HUGSIM", "closed loop, actions\nchange the world")]


def load_audit() -> pd.DataFrame:
    m = pd.read_csv(AUDIT)
    m = m[m["present"] == "yes"].copy()
    m["repo"] = m["repo"].str.replace("https://github.com/", "", regex=False)
    m["method"] = m["repo"].map(REPO_ABBR)
    m["stage"] = m["category"].map(STAGE_OF)
    m["cls"] = m["category"].map(CLASS_OF).fillna("metric-specific")
    m["short"] = m["category"].map(SHORT)
    assert m["method"].notna().all() and m["stage"].notna().all(), m[m.isna().any(axis=1)]
    return m


def fig_mechanism_map(m: pd.DataFrame):
    fs, lh = 6.0, 0.105  # entry font size (pt) and line height (in)
    x_board, w_note, gap = 0.54, 0.76, 0.03
    W = PAGE
    w_stage = (W - x_board - w_note - gap) / 6
    x_stage = [x_board + i * w_stage for i in range(6)]
    x_note = x_stage[-1] + w_stage + gap
    y_hdr, y0 = 0.02, 0.62  # header box top, first row top

    pad_l, x_dot, x_txt, pad_r = 0.02, 0.045, 0.08, 0.015  # inside a cell, inches from its left edge
    usable = w_stage - pad_l - x_txt - pad_r

    def measure(fig, ax, e) -> tuple[float, float]:
        """Rendered widths (in) of the bold method name and of the category label."""
        r = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        out = []
        for txt, kw in ((e["method"], {"weight": "bold"}), (e["short"], {})):
            t = ax.text(0, 0, txt, fontsize=fs, **kw)
            bb = t.get_window_extent(renderer=r)
            out.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
            t.remove()
        return out[0], out[1]

    # Pre-pass: an entry whose "method  label" line does not fit its cell wraps the label onto a second line.
    with mpl.rc_context(STYLE):
        probe = plt.figure(figsize=(W, 1))
        pax = probe.add_axes([0, 0, 1, 1])
        pax.set_xlim(0, W)
        widths = {i: measure(probe, pax, e) for i, e in m.iterrows()}
        plt.close(probe)
    m = m.copy()
    m["w_method"] = [widths[i][0] for i in m.index]
    m["lines"] = [1 if widths[i][0] + 0.035 + widths[i][1] <= usable else 2 for i in m.index]

    # Row heights follow the tallest cell in the row.
    order = ["TFv6", "BLUE", "SDv2", "TF++", "TOAD", "DrivoR", "RAP", "DVLA-M0", "GTRS", "AD-MLP", "BEV-P++", "SOVLA",
             "DriveMA", "WA-JEPA"]
    rows = []
    for key, name, note in BOARDS:
        b = m[m["board"] == key].copy()
        b["ord"] = b["method"].map(order.index)
        b = b.sort_values(["stage", "ord", "category"])
        cells = {s: g for s, g in b.groupby("stage")}
        n_max = max([int(g["lines"].sum()) for g in cells.values()] + [2])
        rows.append((name, note, cells, n_max * lh + 0.09))
    H = y0 + sum(h for *_, h in rows) + 0.36

    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

        # Pipeline header: six boxes joined by arrows.
        bh = 0.40
        for i, s in enumerate(STAGES):
            x = x_stage[i] + 0.05
            ax.add_patch(FancyBboxPatch((x, y_hdr), w_stage - 0.10, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc="#F2F2F2", ec="#444444", lw=0.6))
            ax.text(x + (w_stage - 0.10) / 2, y_hdr + bh / 2, s, ha="center", va="center", fontsize=7.5)
            if i < 5:
                ax.add_patch(FancyArrowPatch((x + w_stage - 0.10, y_hdr + bh / 2), (x + w_stage, y_hdr + bh / 2),
                                             arrowstyle="-|>", mutation_scale=6, color="#444444", lw=0.6,
                                             shrinkA=0, shrinkB=0))
        ax.text(x_note + w_note / 2, y_hdr + bh / 2, "What the metric\ncloses the loop on", ha="center",
                va="center", fontsize=6.8, style="italic", color="#333333")
        ax.text(x_board / 2, y_hdr + bh / 2, "Board", ha="center", va="center", fontsize=7.5)

        y = y0
        for r, (name, note, cells, h) in enumerate(rows):
            if r % 2 == 0:
                ax.add_patch(Rectangle((0, y), W, h, fc="#F7F7F7", ec="none", zorder=0))
            counts = sorted((len(g) for g in cells.values()), reverse=True)
            n_top = counts[0] if len(counts) == 1 or counts[0] > counts[1] else None  # no box on a tie
            for s, g in cells.items():
                x = x_stage[s] + pad_l
                if len(g) == n_top:  # the stage this board's mechanisms concentrate in
                    ax.add_patch(Rectangle((x_stage[s] + 0.005, y + 0.03), w_stage - 0.01, h - 0.06, fc="none",
                                           ec="#444444", lw=0.5, ls=(0, (2, 1.5)), zorder=1))
                line = 0
                for _, e in g.iterrows():
                    yy = y + 0.075 + (line + 0.5) * lh
                    ax.plot(x + x_dot, yy, "o", ms=3.0, mfc=C_CLASS[e["cls"]], mec="none", zorder=3)
                    ax.text(x + x_txt, yy, e["method"], ha="left", va="center", fontsize=fs, weight="bold",
                            color="#111111", zorder=3)
                    if e["lines"] == 1:
                        tx, ty = x + x_txt + e["w_method"] + 0.035, yy
                    else:
                        tx, ty = x + x_txt, yy + lh
                    ax.text(tx, ty, e["short"], ha="left", va="center", fontsize=fs, color="#333333", zorder=3)
                    line += int(e["lines"])
            ax.text(0.06, y + h / 2, name, ha="left", va="center", fontsize=7)
            ax.text(x_note + 0.02, y + h / 2, note, ha="left", va="center", fontsize=6.2, color="#333333",
                    linespacing=1.15)
            y += h
            ax.plot([0, W], [y, y], color="#DDDDDD", lw=0.4, zorder=0)

        handles = [Line2D([], [], marker="o", ls="none", ms=4, mfc=c, mec="none", label=k) for k, c in C_CLASS.items()]
        handles.append(Line2D([], [], ls=(0, (2, 1.5)), color="#444444", lw=0.6,
                              label="stage holding most of the board's observed mechanisms"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5,
                   handletextpad=0.4, columnspacing=1.2, handlelength=1.6, borderaxespad=0.15)
        save(fig, OUT, "lb-ability-mechanism-map")


# ----------------------------------------------------------------------------------------------------------------
# Figure 2: cross-board scatter
# ----------------------------------------------------------------------------------------------------------------
def pair(d: pd.DataFrame, a: tuple[str, str, str], b: tuple[str, str, str]) -> pd.DataFrame:
    def col(t):
        s = d[(d["board"] == t[0]) & (d["protocol/split"] == t[1]) & (d["metric"] == t[2])]
        return s.drop_duplicates("method").set_index("method")["score"]
    # 'DrivoR (+134k SimScale)' repeats the DrivoR row with the same numbers on both boards: counted once (W2 note).
    p = pd.concat([col(a).rename("x"), col(b).rename("y")], axis=1, join="inner")
    return p.drop(index=[i for i in p.index if "SimScale" in i], errors="ignore")


def rho_text(p: pd.DataFrame) -> str:
    r = spearmanr(p["x"], p["y"]).statistic
    return f"$n$ = {len(p)}, Spearman $\\rho$ = {r:+.2f}"


def audit_count(m: pd.DataFrame, method: str) -> int:
    """Largest number of present == yes cells the method's repo has in navsim_v1 or navsim_v2."""
    c = m[(m["method"] == method) & m["board"].isin(["navsim_v1", "navsim_v2"])].groupby("board").size()
    return int(c.max()) if len(c) else 0


def label(ax, x, y, text, tx, ty, ha="left", fs=5.8, color="black"):
    """A point label at (tx, ty); a thin grey leader is drawn when the label sits away from its point."""
    ax.text(tx, ty, text, fontsize=fs, ha=ha, va="center", color=color, zorder=4, linespacing=1.1)
    sx, sy = np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]
    if abs(tx - x) / sx > 0.03 or abs(ty - y) / sy > 0.07:
        ax.annotate("", xy=(x, y), xytext=(tx, ty),
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.4, "shrinkA": 3, "shrinkB": 2.5},
                    zorder=1)


def fig_cross_board(d: pd.DataFrame, m: pd.DataFrame):
    pa = pair(d, ("navsim_v1", "navtest, NAVSIM v1", "PDMS"),
              ("navsim_v2", "navhard-two-stage, NAVSIM v2, corrected EPDMS", "EPDMS"))
    pb = pair(d, ("bench2drive", "base-set closed-loop 220 routes", "Driving Score"),
              ("carla_lb2", "Longest6 v2, CARLA LB2 local", "Driving Score"))
    pc = pair(d, ("bench2drive", "base-set open-loop 2s@2Hz", "Avg L2 (m)"),
              ("bench2drive", "base-set closed-loop 220 routes", "Driving Score"))
    assert (len(pa), len(pb), len(pc)) == (20, 13, 9), (len(pa), len(pb), len(pc))

    bins = {0: ("0 yes", "#FFFFFF"), 1: ("1–2 yes", "#56B4E9"), 2: ("1–2 yes", "#56B4E9"), 3: ("3 yes", "#D55E00")}
    audited = {"DrivoR": "DrivoR", "GTRS (V2-99)": "GTRS", "RAP-DINO": "RAP", "DriveVLA-M0": "DVLA-M0"}

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.5), gridspec_kw={"wspace": 0.34})

        # (a) NAVSIM v1 -> v2
        ax = axes[0]
        ax.set_xlim(82.5, 99)
        ax.set_ylim(17, 66)
        used = set()
        for name, r in pa.iterrows():
            base = name.replace(" + TOAD", "")
            is_toad = name.endswith("+ TOAD")
            if name == "PDM-Closed":
                ax.plot(r.x, r.y, "s", ms=4, mfc=GREY, mec=GREY, zorder=3)
                continue
            who = "TOAD" if is_toad else audited.get(base)
            if who is None:
                fc = ec = LIGHT
                ec = GREY
            else:
                used.add(audit_count(m, who))
                fc = ec = bins[audit_count(m, who)][1]
            if is_toad:
                ax.plot(r.x, r.y, "o", ms=4, mfc="white", mec=ec, mew=0.9, zorder=3)
                b = pa.loc[base]
                ax.plot([b.x, r.x], [b.y, r.y], color=ec, lw=0.6, zorder=2)
            else:
                ax.plot(r.x, r.y, "o", ms=4, mfc=fc, mec=ec, mew=0.6, zorder=3)
        pos_a = {  # hand-placed labels: (text x, text y, ha); a label away from its point gets a leader
            "DiffusionDrive": (88.4, 22.7, "left"), "DriveFuture": (89.4, 59.2, "center"),
            "DrivoR": (95.0, 54.2, "left"), "DrivoR + TOAD": (95.2, 58.9, "center"),
            "GTRS (V2-99)": (90.7, 44.4, "left"), "GTRS (V2-99) + TOAD": (90.6, 53.2, "right"),
            "Hydra-MDP": (90.6, 40.9, "right"), "Hydra-MDP + TOAD": (88.8, 44.6, "right"),
            "LTF": (84.1, 26.3, "left"), "LTFv6": (85.7, 27.3, "left"), "LTFv6 + LEAD": (86.7, 30.6, "left"),
            "PDM-Closed": (88.7, 56.6, "right"), "RAP-DINO": (94.1, 38.2, "left"),
            "RAP-DINO + TOAD": (95.3, 47.8, "left"), "TransFuser": (84.3, 21.8, "left"),
            "World4Drive": (85.4, 36.3, "left"), "ZTRS (V2-99)": (86.6, 48.1, "right"),
            "ZTRS (V2-99) + TOAD": (88.7, 50.9, "right"), "iPad": (92.0, 33.6, "left"),
            "iPad + TOAD": (95.3, 51.8, "left")}
        for name, r in pa.iterrows():
            tx, ty, ha = pos_a[name]
            label(ax, r.x, r.y, name.replace(" (V2-99)", "").replace("RAP-DINO + TOAD", "RAP-DINO\n+ TOAD").replace("iPad + TOAD", "iPad\n+ TOAD"), tx, ty, ha)
        ax.set_xlabel("NAVSIM v1 navtest PDMS")
        ax.set_ylabel("NAVSIM v2 navhard two-stage EPDMS")
        ax.text(0.03, 0.97, rho_text(pa), transform=ax.transAxes, va="top", fontsize=7)
        h = [Line2D([], [], marker="o", ls="none", ms=4, mfc=bins[k][1], mec=bins[k][1],
                    label=f"audited, {bins[k][0]} in the NAVSIM cells") for k in sorted({min(k, 3) for k in used})]
        h += [Line2D([], [], marker="o", ls="none", ms=4, mfc=LIGHT, mec=GREY, mew=0.6, label="not audited"),
              Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec="#444444", mew=0.9,
                     label="+TOAD variant, joined to its base"),
              Line2D([], [], marker="s", ls="none", ms=4, mfc=GREY, mec=GREY, label="privileged rule-based expert")]
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5,
                   handletextpad=0.3, columnspacing=1.2, borderaxespad=0.3)

        # (b) Bench2Drive -> Longest6
        ax = axes[1]
        ax.set_xlim(82, 98.5)
        ax.set_ylim(0, 86)
        for name, r in pb.iterrows():
            if name.startswith("TFv6"):
                ax.plot(r.x, r.y, "o", ms=3, mfc=C_TSPEED, mec=C_TSPEED, zorder=3)
            elif name in ("LEAD", "PDM-Lite"):
                ax.plot(r.x, r.y, "s", ms=4, mfc=GREY, mec=GREY, zorder=3)
            else:
                ax.plot(r.x, r.y, "o", ms=4, mfc="black", mec="black", zorder=3)
        pos_b = {"HiP-AD": (87.3, 9.5, "left", "HiP-AD"), "RoG-DAgger": (89.9, 44, "right", "RoG-DAgger"),
                 "SimLingo": (85.5, 25.2, "left", "SimLingo"), "TF++": (84.5, 19.8, "left", "TF++"),
                 "TFv5 (RegNetY-032, 110C+L)": (83.5, 26.4, "center", "TFv5"),
                 "PDM-Lite": (96.3, 77.0, "right", "PDM-Lite (expert)"),
                 "LEAD": (96.3, 68.0, "right", "LEAD (expert, B2D\nvalue estimated)")}
        for name, (tx, ty, ha, txt) in pos_b.items():
            label(ax, pb.loc[name].x, pb.loc[name].y, txt, tx, ty, ha)
        tf = pb[pb.index.str.startswith("TFv6")]
        ax.annotate("TFv6, six same-paper\nsensor / backbone variants", xy=(94.9, 51.2), xytext=(98.0, 33),
                    fontsize=5.8, ha="right", va="center", color=C_TSPEED, linespacing=1.1,
                    arrowprops={"arrowstyle": "-", "color": C_TSPEED, "lw": 0.4, "shrinkA": 2, "shrinkB": 2})
        ax.set_xlabel("Bench2Drive driving score")
        ax.set_ylabel("CARLA Longest6 v2 driving score")
        ax.text(0.03, 0.97, rho_text(pb), transform=ax.transAxes, va="top", fontsize=7)

        # (c) open-loop L2 -> closed-loop DS, Bench2Drive
        ax = axes[2]
        ax.set_xlim(0.5, 4.0)
        ax.set_ylim(12, 76)
        ax.plot(pc.x, pc.y, "o", ms=4, mfc="black", mec="black", zorder=3)
        pos_c = {"AD-MLP": (3.55, 21.5, "right", "AD-MLP"), "DriveAdapter*": (1.1, 66.5, "left", "DriveAdapter*"),
                 "TCP*": (1.78, 40.7, "left", "TCP*"), "TCP-traj w/o distillation": (2.04, 49.3, "left", "TCP-traj\nw/o distillation"),
                 "TCP-traj*": (1.78, 60.4, "left", "TCP-traj*"), "ThinkTwice*": (0.58, 57.3, "left", "ThinkTwice*"),
                 "UniAD-Base": (0.8, 48.2, "left", "UniAD-Base"), "UniAD-Tiny": (0.87, 38.9, "left", "UniAD-Tiny"),
                 "VAD": (0.98, 43.6, "left", "VAD")}
        for name, (tx, ty, ha, txt) in pos_c.items():
            label(ax, pc.loc[name].x, pc.loc[name].y, txt, tx, ty, ha)
        ax.set_xlabel("Bench2Drive open-loop avg L2 (m), lower is better")
        ax.set_ylabel("Bench2Drive closed-loop driving score")
        ax.text(0.03, 0.97, rho_text(pc), transform=ax.transAxes, va="top", fontsize=7)

        for ax, tag in zip(axes, "abc"):
            ax.text(-0.22, 1.02, f"({tag})", transform=ax.transAxes, fontsize=8, weight="bold", va="bottom")
            ax.grid(True, lw=0.3, alpha=0.3)
        save(fig, OUT, "lb-ability-cross-board")


# ----------------------------------------------------------------------------------------------------------------
# Figure 3: TFv6's two channels
# ----------------------------------------------------------------------------------------------------------------
def fig_tfv6_two_channels():
    # (a) Bench2Drive paired DS differences, same TFv6 checkpoint, 48 route x seed pairs, cluster bootstrap 95% CI
    #     over routes (research/decisions.md item 31: A - B = +14.3 [+5.1, +25.9]; W2b C - B = +1.0 [-12.2, +15.4]).
    #     Heuristics on - off (creeping + stop sign + Kalman, one joint README switch) is about +1 with no CI.
    rows = [("route + target speed\n$-$ waypoint interface", 14.3, 5.1, 25.9, C_TSPEED, True),
            ("heuristics on $-$ off\n(creep, stop sign, Kalman)", 1.0, None, None, GREY, False),
            ("our controller $-$ author PID\n(both on the waypoint interface)", 1.0, -12.2, 15.4, C_WAYPOINT, True)]
    # (b) P5 v0 paired exam, directional flip rate (%), per-frame [95% CI] and per-pair (item 32 and its todo).
    flips = [("TFv6\ntarget speed", 2.0, (0.0, 6.5), 4.0, C_TSPEED),
             ("TFv6\nwaypoint", 39.4, (28.5, 50.1), 73.3, C_WAYPOINT),
             ("our head\n(ridge_late)", 0.0, (0.0, 0.0), 0.0, C_OURS)]
    null_floor = 7.1  # out-of-sample false-flip rate on weather-only null pairs (item 32), the measurement floor

    with mpl.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(PAGE * 0.72, 2.0), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1]})

        ys = np.arange(len(rows))[::-1]
        for y, (lab, v, lo, hi, c, has_ci) in zip(ys, rows):
            if has_ci:
                ax.barh(y, v, height=0.55, color=c, ec="none", zorder=2)
                ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="black", elinewidth=0.7, capsize=2,
                            zorder=3)
            else:
                ax.barh(y, v, height=0.55, fc="white", ec=c, lw=0.8, hatch="////", zorder=2)
                ax.text(v + 1.2, y, "no CI, README", fontsize=6, va="center", color="#333333")
        ax.axvline(0, color="black", lw=0.6, ls="--", zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.5, linespacing=1.1)
        ax.set_xlabel("Paired $\\Delta$ driving score (DS points)")
        ax.set_xlim(-16, 30)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.text(14.3, ys[0] + 0.42, "+14.3 [+5.1, +25.9]", fontsize=6, ha="center", va="bottom", color=C_TSPEED)
        ax.text(1.0, ys[2] - 0.42, "+1.0 [$-$12.2, +15.4]", fontsize=6, ha="center", va="top", color=C_WAYPOINT)
        ax.tick_params(axis="y", length=0)

        xs = np.arange(len(flips))
        w = 0.36
        for i, (lab, pf, ci, pp, c) in enumerate(flips):
            bx.bar(i - w / 2, pf, w, color=c, ec="none", zorder=2)
            bx.errorbar(i - w / 2, pf, yerr=[[pf - ci[0]], [ci[1] - pf]], fmt="none", ecolor="black",
                        elinewidth=0.7, capsize=2, zorder=3)
            bx.bar(i + w / 2, pp, w, fc="white", ec=c, lw=0.8, hatch="////", zorder=2)
            bx.text(i + w / 2, pp + 1.5, f"{pp:.1f}" if pp else "0", fontsize=6, ha="center", va="bottom", color=c)
            if pf == 0:
                bx.text(i - w / 2, 1.5, "0", fontsize=6, ha="center", va="bottom", color=c)
            else:
                bx.text(i - w / 2 - 0.02, ci[1] + 1.5, f"{pf:.1f}", fontsize=6, ha="center", va="bottom", color=c)
        bx.axhline(null_floor, color=GREY, lw=0.7, ls="--", zorder=1)
        bx.text(1.5, null_floor + 1.5, "null false-flip floor\n(weather-only pairs, 7$-$8 %)", fontsize=5.6,
                ha="left", va="bottom", color=GREY, linespacing=1.1)
        bx.set_xticks(xs)
        bx.set_xticklabels([f[0] for f in flips], fontsize=6.5, linespacing=1.1)
        bx.set_ylabel("Directional flip rate (%)")
        bx.set_ylim(0, 100)
        bx.set_xlim(-0.6, 2.75)
        h = [Patch(fc="#444444", ec="none", label="per frame [95% CI]"),
             Patch(fc="white", ec="#444444", hatch="////", label="per pair")]
        bx.legend(handles=h, loc="upper left", fontsize=6, handlelength=1.4, borderaxespad=0.2)

        for a, tag in zip((ax, bx), "ab"):
            a.text(-0.02 if a is bx else -0.62, 1.03, f"({tag})", transform=a.transAxes, fontsize=8, weight="bold",
                   va="bottom")
        save(fig, OUT, "lb-ability-tfv6-two-channels")


# ----------------------------------------------------------------------------------------------------------------
# Figure 4: what kind of gain each board's ablations buy
# ----------------------------------------------------------------------------------------------------------------
C_GAIN = {"general ability": "#009E73", "E-layer (data, sensors, memory)": "#D55E00",
          "R-layer / ego prior": "#0072B2", "metric-specific": "#E69F00", "mixed": GREY}
# Each entry: label, ledger key (method, board, component, baseline_score, variant_score), gain class.
# The ledger row is looked up by these fields and the delta is taken from the file, not typed here. A key with two
# rows is a chain (a -> b -> c) and its delta is the sum of the two rows.
LEDGER_ROWS = {
    "Bench2Drive": ("DS", [
        ("BLUE · language gate", [("BLUE", "bench2drive", "language_gate", 85.07, 90.58)], "mixed"),
        ("LinkVLA · action tokenization", [("LinkVLA", "bench2drive", "token_c2f_alignment_sequence", 85.07, 89.57)],
         "general ability"),
        ("RoG-DAgger · offline DAgger", [("RoG-DAgger", "bench2drive", "offline_dagger_posttraining", 86.59, 90.34)],
         "E-layer (data, sensors, memory)"),
        ("TFv6 · LiDAR", [("TFv6", "bench2drive", "sensor_lidar", 91.6, 94.7)], "E-layer (data, sensors, memory)"),
        ("FIVE-VLA · RAM", [("FIVE-VLA", "bench2drive", "recurrent_action_memory", 88.49, 90.95)],
         "E-layer (data, sensors, memory)"),
        ("FIVE-VLA · ego waypoint history", [("FIVE-VLA", "bench2drive", "explicit_ego_waypoint_history", 88.49, 84.54)],
         "R-layer / ego prior")]),
    "NAVSIM v2": ("EPDMS", [
        ("TOAD · CEM search (on iPad)", [("TOAD", "navsim_v2", "test_time_cem_search", 34.7, 49.8)], "metric-specific"),
        ("GTRS · learned scoring", [("GTRS", "navsim_v2", "learned_trajectory_scoring", 25.6, 36.7)], "metric-specific"),
        ("TOAD · scorer re-rank (on iPad)", [("TOAD", "navsim_v2", "postprocessing_and_scorer_choice", 34.7, 45.6)],
         "metric-specific"),
        ("DrivoR · synthetic data", [("DrivoR", "navsim_v2", "synthetic_training_data", 48.3, 52.3),
                                     ("DrivoR", "navsim_v2", "synthetic_training_data", 52.3, 54.6)],
         "E-layer (data, sensors, memory)"),
        ("WA-JEPA · V-JEPA 2 pretraining*", [("WA-JEPA", "navsim_v2", "vision_encoder_pretraining", 83.8, 89.5)],
         "general ability"),
        ("RAP · recovery perturbations", [("RAP-DINO", "navsim_v2", "recovery_perturbations", 32.5, 36.9)],
         "E-layer (data, sensors, memory)")]),
    "nuScenes": ("avg L2", [
        ("AD-MLP · ego acceleration", [("AD-MLP", "nuscenes", "ego_state_navigation_input", 0.97, 0.49)],
         "R-layer / ego prior"),
        ("SparseOccVLA · traj-ego fusion", [("SparseOccVLA", "nuscenes", "planning_fusion", 0.68, 0.30)],
         "R-layer / ego prior"),
        ("BEV-Planner++ · ego in BEV + plan", [("BEV-Planner++", "nuscenes", "ego_state_bev", 0.55, 0.46),
                                               ("BEV-Planner++", "nuscenes", "ego_state_planner", 0.46, 0.35)],
         "R-layer / ego prior"),
        ("UniAD · task coordination", [("UniAD", "nuscenes", "task_coordination", 1.154, 1.004)], "general ability"),
        ("OmniSpace · 3D geometry", [("OmniSpace", "nuscenes", "train_time_3d_geometry_distillation", 0.37, 0.28)],
         "general ability")]),
    "WOD-E2E": ("RFS", [
        ("Poutine · WOD pretraining", [("Poutine", "wod_e2e", "WOD_pretraining", 5.59, 7.95)], "general ability"),
        ("NTR · latent reconstruction", [("NTR", "wod_e2e", "latent_reconstruction", 7.652, 7.974)], "general ability"),
        ("DriveMA · RL with RFS reward", [("DriveMA-4B", "wod_e2e", "reinforcement_learning", 7.893, 7.978)],
         "metric-specific"),
        ("Poutine · GRPO", [("Poutine", "wod_e2e", "grpo_posttraining", 7.91, 7.99)], "metric-specific")]),
}


def ledger_delta(led: pd.DataFrame, keys) -> float:
    total = 0.0
    for method, board, comp, b, v in keys:
        r = led[(led["method"] == method) & (led["board"] == board) & (led["component"] == comp)
                & np.isclose(led["baseline_score"], b) & np.isclose(led["variant_score"], v)]
        assert len(r) >= 1, (method, board, comp, b, v)
        total += float(r["delta"].iloc[0])
    return total


def fig_gain_ledger(led: pd.DataFrame):
    fs = 6.0
    W, H = PAGE, 3.05
    x_lab, w_bar, x_gap = 1.18, 0.80, 0.12      # label column, bar axes, gap between the two columns (inches)
    y_row2, h_row, y_gap = 0.62, 0.85, 0.55      # rows of panel (a)
    xs_col = [x_lab, x_lab + w_bar + x_gap + x_lab]
    ys_row = [y_row2 + h_row + y_gap, y_row2]
    x_b = xs_col[1] + w_bar + 0.62
    w_b = W - x_b - 0.12

    def add(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(W, H))
        axes = [add(xs_col[0], ys_row[0], w_bar, h_row), add(xs_col[1], ys_row[0], w_bar, h_row),
                add(xs_col[0], ys_row[1], w_bar, h_row), add(xs_col[1], ys_row[1], w_bar, h_row)]
        xlabels = {"Bench2Drive": "$\\Delta$ driving score (DS points)", "NAVSIM v2": "$\\Delta$ EPDMS (points)",
                   "nuScenes": "$\\Delta$ avg L2 (m), axis reversed", "WOD-E2E": "$\\Delta$ RFS (points)"}
        for ax, (board, (metric, rows)) in zip(axes, LEDGER_ROWS.items()):
            deltas = [ledger_delta(led, keys) for _, keys, _ in rows]
            lower_better = board == "nuScenes"
            order = np.argsort([-d if lower_better else d for d in deltas])[::-1]  # biggest gain on top
            ys = np.arange(len(rows))[::-1]
            for y, i in zip(ys, order):
                lab, _, cls = rows[i]
                ax.barh(y, deltas[i], height=0.62, color=C_GAIN[cls], ec="none", zorder=2)
                gain = (-deltas[i] if lower_better else deltas[i]) >= 0
                txt = f"{deltas[i]:+.2f}" if abs(deltas[i]) < 1 else f"{deltas[i]:+.1f}"
                ax.text(deltas[i], y, f" {txt} ", fontsize=5.5, va="center", ha="left" if gain else "right",
                        color="#333333", zorder=3)
            ax.set_yticks(ys)
            ax.set_yticklabels([rows[i][0] for i in order], fontsize=fs)
            ax.tick_params(axis="y", length=0, pad=2)
            ax.axvline(0, color="black", lw=0.6, zorder=3)
            ax.set_xlabel(xlabels[board], labelpad=1.5)
            ax.text(0.0, 1.03, f"{board}, {metric}", transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
                    weight="bold")
            ax.spines["left"].set_visible(False)
            ax.grid(True, axis="x", lw=0.3, alpha=0.3)
            lo, hi = min(deltas + [0]), max(deltas + [0])
            span = hi - lo
            if lower_better:
                ax.set_xlim(hi + 0.28 * span, lo - 0.3 * span)  # improvement (negative) points right
                ax.set_xticks([0, -0.25, -0.5])
            else:
                ax.set_xlim(lo - 0.3 * span if lo < 0 else 0, hi + 0.35 * span)
        fig.text(0.995, 0.012, "*WA-JEPA: navtest, not the navhard two-stage split", fontsize=5.5, ha="right",
                 va="bottom", color="#333333")
        h = [Patch(fc=c, ec="none", label=k) for k, c in C_GAIN.items()]
        fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=5, fontsize=6.5, handlelength=1.2,
                   handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)

        # (b) the same ego prior, scored open loop and closed loop, as % of the baseline metric
        bx = add(x_b, y_row2, w_b, ys_row[0] + h_row - y_row2)
        pts = [  # x, gain %, filled (closed loop), marker size, label; ledger rows in the comments
            (0, (0.97 - 0.35) / 0.97 * 100, False, 5, "AD-MLP\nL2 0.97 $\\to$ 0.35 m"),        # AD-MLP rows 27+29
            (1, 1.8 / 84.0 * 100, False, 5, "TransFuser, ego state\nremoved: $-$1.0 to $-$2.6 PDMS"),  # rows 82/83
            (2, (0.549 - 0.373) / 0.549 * 100, False, 5, "FIVE-VLA, + ego history\nspeed ADE 0.549 $\\to$ 0.373 m"),
            (3, -3.95 / 88.49 * 100, True, 5, "FIVE-VLA, + ego history\nDS 88.49 $\\to$ 84.54"),  # row 773
            (3, -8.64 / 73.03 * 100, True, 3.2, "SR 73.03 $\\to$ 64.39"),                       # row 774
        ]
        bx.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
        bx.plot([1, 1], [1.0 / 84 * 100, 2.6 / 84 * 100], color=C_TSPEED, lw=1.0, zorder=2)  # seed range
        for x, g, filled, ms, lab in pts:
            bx.plot(x, g, "o", ms=ms, mfc=C_TSPEED if filled else "white", mec=C_TSPEED, mew=0.9, zorder=3)
        lab_pos = {0: (0.2, 64, "left"), 1: (1.18, 9.5, "left"), 2: (3.55, 41, "right"), 3: (2.8, -6.5, "right"),
                   4: (2.8, -14.5, "right")}
        for k, (x, g, filled, ms, lab) in enumerate(pts):
            tx, ty, ha = lab_pos[k]
            bx.text(tx, ty, lab, fontsize=5.5, ha=ha, va="center", linespacing=1.1, color="#333333")
        bx.set_xticks(range(4))
        bx.set_xticklabels(["nuScenes\nopen loop\n3 s", "NAVSIM v1\nopen loop\n4 s", "NVIDIA aux\nopen loop",
                            "Bench2Drive\nclosed loop"], fontsize=6, linespacing=1.1)
        bx.set_ylabel("Gain from the ego prior (% of baseline metric)")
        bx.set_xlim(-0.4, 3.5)
        bx.set_ylim(-20, 72)
        bx.grid(True, axis="y", lw=0.3, alpha=0.3)
        hb = [Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec=C_TSPEED, mew=0.9, label="open loop"),
              Line2D([], [], marker="o", ls="none", ms=4, mfc=C_TSPEED, mec=C_TSPEED, label="closed loop")]
        bx.legend(handles=hb, loc="lower left", fontsize=6, handletextpad=0.3, borderaxespad=0.2)
        fig.text(0.04 / W, (H - 0.04) / H, "(a)", fontsize=8, weight="bold", va="top")
        fig.text((x_b - 0.42) / W, (H - 0.04) / H, "(b)", fontsize=8, weight="bold", va="top")
        save(fig, OUT, "lb-ability-gain-ledger")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_audit()
    fig_mechanism_map(m)
    fig_cross_board(pd.read_csv(CROSS), m)
    fig_tfv6_two_channels()
    fig_gain_ledger(pd.read_csv(LEDGER))
    for n in ("mechanism-map", "cross-board", "tfv6-two-channels", "gain-ledger"):
        p = OUT / f"lb-ability-{n}.png"
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
