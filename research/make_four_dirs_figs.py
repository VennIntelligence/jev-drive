"""Generate figures for research/four-directions.html following CVPR plot style."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
import plot_style

plot_style.apply()

OUT_DIR = Path("research/figs/four-directions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Fig 1: Current Position across Benchmarks
# -------------------------------------------------------------
def make_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(plot_style.DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    
    # Colors
    c_shipped = '#888888'
    c_p2h = plot_style.PALETTE['blue']
    c_wajepa = plot_style.PALETTE['vermillion']
    
    # Panel 1: navtest EPDMS
    ax = axes[0]
    plot_style.panel(ax, "a  navtest (12,146 tokens)")
    models = ['Shipped\n(P0)', 'P2H\n(Ours)', 'WA-JEPA\n(SOTA)']
    scores = [80.51, 88.67, 91.71]
    yerr = [[0, 0], [0.35, 0.35], [0, 0]] # P2H CI roughly +-0.35
    colors = [c_shipped, c_p2h, c_wajepa]
    bars = ax.bar(models, scores, color=colors, width=0.55, edgecolor='none', zorder=3)
    ax.set_ylabel("EPDMS Score")
    ax.set_ylim(70, 95)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.6, f"{score:.1f}", ha='center', va='bottom', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    # Panel 2: navhard EPDMS (Combined & Stage 1/2)
    ax = axes[1]
    plot_style.panel(ax, "b  navhard (225 groups)")
    x = np.arange(3)
    width = 0.26
    
    # Combined, Stage 1, Stage 2
    shipped_vals = [33.54, 71.79, 47.08]
    p2h_vals = [31.84, 74.73, 43.04]
    wa_vals = [35.41, 81.90, 43.53]
    
    ax.bar(x - width, shipped_vals, width, label='Shipped', color=c_shipped, zorder=3)
    ax.bar(x, p2h_vals, width, label='P2H', color=c_p2h, zorder=3)
    ax.bar(x + width, wa_vals, width, label='WA-JEPA', color=c_wajepa, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Combined', 'Stage 1', 'Stage 2'])
    ax.set_ylabel("EPDMS Score")
    ax.set_ylim(0, 92)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(axis='x', visible=False)
    
    # Panel 3: HUGSIM 64 (HD x 100)
    ax = axes[2]
    plot_style.panel(ax, "c  HUGSIM 64 (HD × 100)")
    x = np.arange(3)
    width = 0.26
    
    # All 64, Turn 23, Straight 41
    # P2H under spec_plan_smooth
    shipped_hd = [39.4, 27.9, 45.9] # spec
    p2h_hd = [43.2, 35.0, 47.8]
    wa_hd = [45.1, 44.2, 45.6]
    
    b1 = ax.bar(x - width, shipped_hd, width, label='Shipped (spec)', color=c_shipped, zorder=3)
    b2 = ax.bar(x, p2h_hd, width, label='P2H (smooth)', color=c_p2h, zorder=3)
    b3 = ax.bar(x + width, wa_hd, width, label='WA-JEPA', color=c_wajepa, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['All 64', 'Turn 23', 'Straight 41'])
    ax.set_ylabel("HD Score (× 100)")
    ax.set_ylim(0, 56)
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(axis='x', visible=False)
    
    stem = OUT_DIR / "fig1_current_position"
    plot_style.save(fig, stem)
    plt.close(fig)
    print(f"Generated {stem}.png")

# -------------------------------------------------------------
# Fig 2: Sizes of Four Directions (navtest & HUGSIM)
# -------------------------------------------------------------
def make_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN_IN, 2.4), constrained_layout=True)
    
    # Panel A: navtest Direction Sizes (points)
    ax = axes[0]
    plot_style.panel(ax, "a  navtest gap share (WA - P2H = 3.04 pts)")
    dirs = ['D1 Sharp\n(R < 15m)', 'D2 Wide\n(R ≥ 15m)', 'D3 Contact\n(NC/TTC)', 'D4 Night\n(Luma < 50)']
    
    # Oracle (a) and Shapley (c)
    oracle_a = [1.04, 1.00, 1.16, 0.00]
    shapley_c = [0.87, 0.66, 0.87, 0.00]
    yerr_a = [[0.31, 0.27, 0.28, 0], [0.33, 0.30, 0.28, 0]]
    
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, oracle_a, width, label='Oracle (a) Replace WA', color=plot_style.PALETTE['sky_blue'], zorder=3, yerr=yerr_a, capsize=2)
    ax.bar(x + width/2, shapley_c, width, label='Shapley Share (c)', color=plot_style.PALETTE['blue'], zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(dirs)
    ax.set_ylabel("EPDMS Points")
    ax.set_ylim(0, 1.8)
    ax.legend(loc='upper right', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    # Panel B: HUGSIM 64 Direction Loss (HD x 100 points)
    ax = axes[1]
    plot_style.panel(ax, "b  HUGSIM 64 loss (Total lost = 56.8 HD)")
    dirs_h = ['D1 Sharp\n(5 sc)', 'D2 Wide\n(2 sc)', 'D3a Oncoming\n(21 sc)', 'D3b Lead\n(10 sc)', 'D4 Night\n(0 sc)']
    
    oracle_ah = [2.7, 1.0, 1.3, 3.8, 0.0]
    loss_c = [5.5, 2.7, 28.4, 13.0, 0.0] # HD set to 1.0 (actual lost HD points)
    
    x_h = np.arange(5)
    ax.bar(x_h - width/2, oracle_ah, width, label='Oracle (a) Replace WA', color=plot_style.PALETTE['orange'], zorder=3)
    ax.bar(x_h + width/2, loss_c, width, label='Total Loss (c) HD=1', color=plot_style.PALETTE['vermillion'], zorder=3)
    
    ax.set_xticks(x_h)
    ax.set_xticklabels(dirs_h)
    ax.set_ylabel("HD × 100 Points")
    ax.set_ylim(0, 32)
    ax.legend(loc='upper right', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    stem = OUT_DIR / "fig2_direction_sizes"
    plot_style.save(fig, stem)
    plt.close(fig)
    print(f"Generated {stem}.png")

# -------------------------------------------------------------
# Fig 3: Key Mechanisms Breakdown
# -------------------------------------------------------------
def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(plot_style.DOUBLE_COLUMN_IN, 2.4), constrained_layout=True)
    
    # Panel A: D1 navtest Sharp Turn Mechanism
    ax = axes[0]
    plot_style.panel(ax, "a  D1 navtest failure modes")
    modes = ['Inside cut\n(54%)', 'Not around\n(26%)', 'Other\n(20%)']
    shares = [54, 26, 20]
    yerr = [[12, 8, 0], [11, 8, 0]]
    colors = [plot_style.PALETTE['vermillion'], plot_style.PALETTE['sky_blue'], '#999999']
    bars = ax.bar(modes, shares, color=colors, width=0.55, zorder=3, yerr=yerr, capsize=2)
    ax.set_ylabel("Share of D1 DAC Failures (%)")
    ax.set_ylim(0, 75)
    for bar, val in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width()/2, val + 6, f"{val}%", ha='center', va='bottom', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    # Panel B: D2 navtest Wide Turn Geometry
    ax = axes[1]
    plot_style.panel(ax, "b  D2 navtest failure geometry")
    modes = ['Graze <0.3m\n(62%)', 'Replay only\n(42%)', 'Inside of turn\n(53%)']
    shares = [62, 42, 53]
    yerr = [[7, 8, 12], [7, 9, 11]]
    colors = [plot_style.PALETTE['orange'], plot_style.PALETTE['blue'], plot_style.PALETTE['green']]
    bars = ax.bar(modes, shares, color=colors, width=0.55, zorder=3, yerr=yerr, capsize=2)
    ax.set_ylabel("Share of D2 DAC Failures (%)")
    ax.set_ylim(0, 80)
    for bar, val in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f"{val}%", ha='center', va='bottom', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    # Panel C: D3 Collision Types on navtest
    ax = axes[2]
    plot_style.panel(ax, "c  D3 navtest collision types")
    types = ['Stopped\nahead', 'Static\nobj', 'Lead\nmoving', 'Cut-in', 'Crossing\n/oncoming']
    shares = [36, 17, 14, 11, 19]
    colors = [plot_style.PALETTE['vermillion'], '#777777', plot_style.PALETTE['orange'], plot_style.PALETTE['sky_blue'], plot_style.PALETTE['purple']]
    bars = ax.bar(types, shares, color=colors, width=0.6, zorder=3)
    ax.set_ylabel("Share of Collision Tokens (%)")
    ax.set_ylim(0, 48)
    for bar, val in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val}%", ha='center', va='bottom', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    stem = OUT_DIR / "fig3_mechanisms"
    plot_style.save(fig, stem)
    plt.close(fig)
    print(f"Generated {stem}.png")

# -------------------------------------------------------------
# Fig 4: WOD-E2E Evaluation (Day vs Night & Diagnostic Arms)
# -------------------------------------------------------------
def make_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE_COLUMN_IN, 2.3), constrained_layout=True)
    
    # Panel A: RFS Comparison
    ax = axes[0]
    plot_style.panel(ax, "a  WOD val: RFS (higher is better)")
    splits = ['All Val\n(479 rater)', 'Day\n(325 rater)', 'Night\n(133 rater)', 'Bias = 0\n(All Val)']
    shipped = [8.005, 8.100, 7.610, 8.005]
    p2h = [7.708, 7.740, 7.404, 7.984]
    
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, shipped, width, label='Shipped', color='#888888', zorder=3)
    ax.bar(x + width/2, p2h, width, label='P2H', color=plot_style.PALETTE['blue'], zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("RFS (0-10)")
    ax.set_ylim(7.0, 8.5)
    ax.legend(loc='lower left', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    # Panel B: ADE@3s Comparison
    ax = axes[1]
    plot_style.panel(ax, "b  WOD val: ADE @ 3s (m, lower is better)")
    splits = ['All Val\n(1,437 frames)', 'Day\n(977 frames)', 'Night\n(397 frames)', 'Bias = 0\n(All Val)']
    shipped_ade = [1.041, 0.921, 1.308, 1.041]
    p2h_ade = [1.324, 1.338, 1.312, 1.098]
    
    ax.bar(x - width/2, shipped_ade, width, label='Shipped', color='#888888', zorder=3)
    ax.bar(x + width/2, p2h_ade, width, label='P2H', color=plot_style.PALETTE['blue'], zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("ADE @ 3s (m)")
    ax.set_ylim(0.7, 1.55)
    ax.legend(loc='upper left', fontsize=7.5)
    ax.grid(axis='x', visible=False)
    
    stem = OUT_DIR / "fig4_wod_comparison"
    plot_style.save(fig, stem)
    plt.close(fig)
    print(f"Generated {stem}.png")

# -------------------------------------------------------------
# Copy and Extract Frame PNGs
# -------------------------------------------------------------
def copy_and_extract_assets():
    # 1. Copy BEV and diagnostic figures from experiments/op_parity/results/four_dirs/figs/
    src_figs = Path("experiments/op_parity/results/four_dirs/figs")
    for name in ["hugsim_bev_d1.png", "hugsim_bev_d2.png", "hugsim_bev_d3.png", 
                 "nav_dac_anatomy.png", "nav_collision_types.png", "night_luma.png"]:
        src = src_figs / name
        dst = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {name} to {dst}")
            
    # 2. Extract PNG frames from gap clips GIFs
    clips_dir = Path("experiments/op_parity/results/gap/clips")
    frame_extractions = [
        ("570_770-easy-00.gif", "frame_570_770_d1.png", 45), # near corner contact
        ("dac-45935e.gif", "frame_dac_45935e_d1.png", 20),
        ("dac-56706b.gif", "frame_dac_56706b_d2.png", 22),
        ("nc-248801.gif", "frame_nc_248801_d3.png", 24),
        ("ttc-3f265d.gif", "frame_ttc_3f265d_d3.png", 22)
    ]
    
    for gif_name, png_name, f_idx in frame_extractions:
        gif_path = clips_dir / gif_name
        if gif_path.exists():
            im = Image.open(gif_path)
            im.seek(min(f_idx, im.n_frames - 1))
            im.convert("RGB").save(OUT_DIR / png_name)
            print(f"Extracted frame {f_idx} from {gif_name} -> {png_name}")

if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    copy_and_extract_assets()
    print("All figures ready in", OUT_DIR)
