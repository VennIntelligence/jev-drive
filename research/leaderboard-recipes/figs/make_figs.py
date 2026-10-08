"""Generate figures for leaderboard-recipes report.

All plotted numbers are hard-coded with citations to primary lit notes and tables.
Style matches research/plot_style.py (serif, STIXGeneral, Okabe-Ito, single/double col).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add repo root to import research.plot_style
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
import research.plot_style as ps

OUT_DIR = Path(__file__).resolve().parent

def setup():
    ps.apply()

# -----------------------------------------------------------------------------
# Figure 1: Score Decomposition Ladders across 3 Benchmarks
# Sources:
# WOD-E2E: lit 1 §1, §3, §4; decisions 180 (WLG test 8.099)
# NAVSIM v1: lit 2 §2, §3, §7.2; decisions 170 (SH30 navtest 89.55)
# NAVSIM navhard (post-fix): lit 2 §2, §7.2; lit 5 §4.1; decisions 148, 170 (SH30 ~33.7)
# -----------------------------------------------------------------------------
def make_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.24, wspace=0.38)

    # Panel (a): WOD-E2E test RFS
    ax = axes[0]
    ps.bars(ax)
    ps.panel(ax, '(a) WOD-E2E (test RFS)')
    steps_a = ['Ego-MLP', '+In-Domain', '+Extra Data', '+Metric RL', '+Ensemble']
    base_a = [0, 7.41, 7.87, 7.99, 8.10]
    deltas_a = [7.41, 0.46, 0.12, 0.11, 0.05]
    colors_a = [ps.BASELINE, ps.PALETTE['sky_blue'], ps.PALETTE['blue'], ps.PALETTE['orange'], ps.PALETTE['green']]
    
    x = np.arange(len(steps_a))
    ax.bar(x, deltas_a, bottom=base_a, color=colors_a, width=0.55, edgecolor='none')
    
    ax.axhline(8.13, color=ps.PALETTE['vermillion'], linestyle='--', linewidth=0.8, label='Human log (8.13)')
    ax.axhline(8.099, color=ps.PALETTE['purple'], linestyle=':', linewidth=0.8, label='Ours WLG (8.099)')
    ax.axhline(8.167, color='#333333', linestyle='-.', linewidth=0.6, label='Top ZSD-Titan (8.17)')
    
    ax.set_ylim(7.0, 8.3)
    ax.set_xticks(x)
    ax.set_xticklabels(steps_a, rotation=35, ha='right', fontsize=6.5)
    ax.set_ylabel('RFS')
    ax.legend(loc='lower right', fontsize=6.0, frameon=False, handlelength=1.5)

    # Panel (b): NAVSIM v1 (navtest PDMS)
    ax = axes[1]
    ps.bars(ax)
    ps.panel(ax, '(b) NAVSIM v1 (PDMS)')
    steps_b = ['Ego-MLP', '+Base Vis.', '+Video-SSL', '+Scorer/RL', '+Sim/Search']
    base_b = [0, 65.6, 84.0, 90.0, 93.7]
    deltas_b = [65.6, 18.4, 6.0, 3.7, 1.6]
    colors_b = [ps.BASELINE, ps.PALETTE['sky_blue'], ps.PALETTE['blue'], ps.PALETTE['orange'], ps.PALETTE['green']]
    
    ax.bar(x, deltas_b, bottom=base_b, color=colors_b, width=0.55, edgecolor='none')
    ax.axhline(94.8, color=ps.PALETTE['vermillion'], linestyle='--', linewidth=0.8, label='Human log (94.8)')
    ax.axhline(89.55, color=ps.PALETTE['purple'], linestyle=':', linewidth=0.8, label='Ours SH30 (89.6)')
    ax.axhline(91.1, color='#666666', linestyle='-.', linewidth=0.6, label='Memory-only (91.1)')
    
    ax.set_ylim(55.0, 100.0)
    ax.set_xticks(x)
    ax.set_xticklabels(steps_b, rotation=35, ha='right', fontsize=6.5)
    ax.set_ylabel('PDMS')
    ax.legend(loc='lower right', fontsize=6.0, frameon=False, handlelength=1.5)

    # Panel (c): NAVSIM navhard (two-stage EPDMS, post-fix)
    ax = axes[2]
    ps.bars(ax)
    ps.panel(ax, '(c) navhard (EPDMS post-fix)')
    steps_c = ['Ego-MLP', '+Base Vis.', '+Scorer', '+SimScale', '+Search']
    base_c = [0, 14.1, 25.1, 36.7, 48.3]
    deltas_c = [14.1, 11.0, 11.6, 11.6, 8.2]
    colors_c = [ps.BASELINE, ps.PALETTE['sky_blue'], ps.PALETTE['orange'], ps.PALETTE['blue'], ps.PALETTE['green']]
    
    ax.bar(x, deltas_c, bottom=base_c, color=colors_c, width=0.55, edgecolor='none')
    ax.axhline(56.6, color=ps.PALETTE['vermillion'], linestyle='--', linewidth=0.8, label='PDM-Closed (56.6)')
    ax.axhline(33.7, color=ps.PALETTE['purple'], linestyle=':', linewidth=0.8, label='Ours SH30 (~33.7)')
    ax.axhline(61.02, color='#333333', linestyle='-.', linewidth=0.6, label='Top RoboTruck (61.0)')
    
    ax.set_ylim(0.0, 68.0)
    ax.set_xticks(x)
    ax.set_xticklabels(steps_c, rotation=35, ha='right', fontsize=6.5)
    ax.set_ylabel('EPDMS')
    ax.legend(loc='lower right', fontsize=6.0, frameon=False, handlelength=1.5)

    ps.save(fig, OUT_DIR / 'fig1_score_decomposition')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 2: World Model Gain Collapse Scatter
# Sources: lit 4 §5a, §5b; lit 2 §4.1, §4.2
# -----------------------------------------------------------------------------
def make_fig2():
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.55))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18)
    ps.panel(ax, 'World-Model Gain Collapse on Strong Baselines')

    pts = [
        (68.7, 12.0, 'DriveVLA-W0 (VQ)'),
        (70.3, 13.6, 'DriveVLA-W0 (ViT)'),
        (77.5, 7.1, 'LAW'),
        (78.1, 8.1, 'Epona'),
        (83.1, 4.6, 'SV-WAM'),
        (85.5, 4.6, 'GraphWorld'),
        (83.2, 2.4, 'WoTE'),
        (84.4, 2.5, 'EponaV2'),
        (83.6, 0.9, 'DriveX'),
        (86.9, 0.6, 'PerceptDrive'),
        (87.0, 1.1, 'WorldDrive'),
        (87.3, 0.8, 'PWM'),
        (87.7, 0.35, 'Latent-WAM'),
        (87.9, 1.0, 'SeerDrive'),
        (88.0, 1.2, 'DriveDreamer'),
        (88.9, 0.7, 'ForeDrive'),
        (88.9, 0.1, 'Metis'),
        (90.84, 0.21, 'EditWM'),
        (91.1, 0.6, 'WA-JEPA'),
        (93.31, 0.37, 'DA-WAM'),
    ]

    x_vals = np.array([p[0] for p in pts])
    y_vals = np.array([p[1] for p in pts])

    # Fit exponential curve
    poly = np.polyfit(x_vals, np.log(y_vals + 0.1), 1)
    x_curve = np.linspace(67, 95, 100)
    y_curve = np.exp(np.polyval(poly, x_curve)) - 0.1

    ax.plot(x_curve, y_curve, color='#888888', linestyle='--', linewidth=0.8, zorder=1, label='Exponential decay')
    
    colors = [ps.PALETTE['vermillion'] if x < 80 else ps.PALETTE['blue'] for x in x_vals]
    ax.scatter(x_vals, y_vals, color=colors, s=26, zorder=2, edgecolors='none')

    # Selective annotations, positioned inside visible area
    annots = {
        'DriveVLA-W0 (ViT)': (1.5, 0.3),
        'LAW': (-6.0, 0.8),
        'Epona': (1.5, 0.8),
        'SV-WAM': (1.5, 0.8),
        'WoTE': (-5.5, 1.2),
        'ForeDrive': (-3.0, 1.6),
        'WA-JEPA': (1.0, 1.5),
        'DA-WAM': (-6.0, 1.2),
    }
    for p in pts:
        name = p[2]
        if name in annots:
            dx, dy = annots[name]
            ax.annotate(name, (p[0], p[1]), xytext=(p[0]+dx, p[1]+dy),
                        fontsize=5.8, color='#333333',
                        arrowprops=dict(arrowstyle='->', color='#999999', lw=0.4))

    ax.axhline(0, color='#999999', linewidth=0.5, zorder=0)
    ax.set_xlim(65, 96)
    ax.set_ylim(-0.5, 16.5)
    ax.set_xlabel('Baseline Score (PDMS / EPDMS)')
    ax.set_ylabel('Net Gain from WM / Future Objective')

    ps.save(fig, OUT_DIR / 'fig2_wm_gain_vs_baseline')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 3: Language at Inference (With vs Without CoT/Language)
# Sources: lit 4 §Q3; lit 5; lit 6
# -----------------------------------------------------------------------------
def make_fig3():
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.7))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.34)
    ps.bars(ax)
    ps.panel(ax, 'Language / CoT at Inference: Minimal or Negative Impact')

    labels = [
        'Poutine\n(WOD RFS)',
        'AutoVLA\n(WOD RFS)',
        'SimLingo\n(B2D DS)',
        'ReCogDrive\n(navtest)',
        'BLUE/SimL\n(B2D SR%)',
        'BLUE/ReCog\n(navtest)',
        'AdaThink\n(navtest)',
        'nuPlan LoRA\n(Score@3s)',
    ]
    deltas_always = np.array([-0.04, +0.041, +0.40, -0.10, -2.64, -1.85, +0.60, -0.02])
    deltas_gated = np.array([np.nan, np.nan, np.nan, np.nan, +6.63, +1.02, +2.00, np.nan])

    x = np.arange(len(labels))
    w = 0.38

    c_always = [ps.PALETTE['vermillion'] if d < 0 else ps.PALETTE['sky_blue'] for d in deltas_always]
    ax.bar(x - w/2, deltas_always, width=w, color=c_always, label='Always CoT / Lang.')
    
    has_gated = ~np.isnan(deltas_gated)
    ax.bar(x[has_gated] + w/2, deltas_gated[has_gated], width=w, color=ps.PALETTE['green'], label='Learned / Adaptive Gate')

    ax.axhline(0, color='#666666', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=5.8)
    ax.set_ylabel(r'$\Delta$ Metric (vs. No-Language Baseline)')
    ax.set_ylim(-3.5, 8.5)
    ax.legend(loc='upper left', fontsize=5.8, frameon=False)

    ps.save(fig, OUT_DIR / 'fig3_language_inference_effect')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 4: Encoder Pretraining Comparison (image-SSL / VL vs video-SSL)
# Sources: lit 4 §Q6; lit 2 §3, §4; lit 5 §5
# -----------------------------------------------------------------------------
def make_fig4():
    fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30, wspace=0.35)

    # Panel (a): WA-JEPA (NAVSIM-v2 navtest EPDMS, Tab. 4a)
    ax = axes[0]
    ps.bars(ax)
    ps.panel(ax, '(a) WA-JEPA (navtest EPDMS)')
    encoders_a = ['MAE\n(image)', 'SigLIP2\n(VLM)', 'DINOv3\n(image)', 'V-JEPA 2\n(video)']
    scores_a = [83.8, 83.1, 83.8, 89.5]
    colors_a = [ps.BASELINE, ps.PALETTE['sky_blue'], ps.BASELINE, ps.PALETTE['blue']]
    x_a = np.arange(len(scores_a))
    ax.bar(x_a, scores_a, color=colors_a, width=0.55)
    ax.set_ylim(80.0, 92.5)
    ax.set_xticks(x_a)
    ax.set_xticklabels(encoders_a, fontsize=6.2)
    ax.set_ylabel('EPDMS')
    ax.annotate('+5.7 pp', xy=(3, 89.5), xytext=(2.0, 90.7),
                fontsize=6.2, weight='bold', color=ps.PALETTE['blue'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['blue'], lw=0.6))

    # Panel (b): Drive-JEPA (NAVSIM v1 navtest PDMS, Tab. 5)
    ax = axes[1]
    ps.bars(ax)
    ps.panel(ax, '(b) Drive-JEPA (navtest PDMS)')
    encoders_b = ['ResNet34\n(Superv.)', 'DINOv2-L\n(image)', 'SigLIP-L\n(VLM)', 'V-JEPA 2\n(video)', '+Driving\nVideo SSL']
    scores_b = [76.0, 76.1, 83.4, 86.1, 89.0]
    colors_b = [ps.BASELINE, ps.BASELINE, ps.PALETTE['sky_blue'], ps.PALETTE['blue'], ps.PALETTE['green']]
    x_b = np.arange(len(scores_b))
    ax.bar(x_b, scores_b, color=colors_b, width=0.55)
    ax.set_ylim(70.0, 92.5)
    ax.set_xticks(x_b)
    ax.set_xticklabels(encoders_b, rotation=25, ha='right', fontsize=5.8)
    ax.set_ylabel('PDMS (Perception-Free)')
    ax.annotate('+10.0 pp', xy=(3, 86.1), xytext=(1.7, 88.0),
                fontsize=6.0, weight='bold', color=ps.PALETTE['blue'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['blue'], lw=0.6))

    # Panel (c): DrivoR (NAVSIM navval PDMS, Tab. 4a, ViT-S)
    ax = axes[2]
    ps.bars(ax)
    ps.panel(ax, '(c) DrivoR (navval PDMS)')
    encoders_c = ['Random\nInit', 'ImageNet-21k\n(Superv.)', 'DINOv2-S\n(image-SSL)']
    scores_c = [70.1, 87.5, 90.0]
    colors_c = [ps.BASELINE, ps.PALETTE['sky_blue'], ps.PALETTE['blue']]
    x_c = np.arange(len(scores_c))
    ax.bar(x_c, scores_c, color=colors_c, width=0.55)
    ax.set_ylim(65.0, 93.5)
    ax.set_xticks(x_c)
    ax.set_xticklabels(encoders_c, fontsize=6.2)
    ax.set_ylabel('PDMS')
    ax.annotate('+19.9 pp', xy=(2, 90.0), xytext=(0.8, 91.0),
                fontsize=6.2, weight='bold', color=ps.PALETTE['blue'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['blue'], lw=0.6))

    ps.save(fig, OUT_DIR / 'fig4_encoder_pretrain_comparison')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 5: Post-RL Sub-score Shifts (EP Up, NC / TTC Down)
# Sources: lit 4 §Q4; lit 6 §1 #21
# -----------------------------------------------------------------------------
def make_fig5():
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.30)
    ps.bars(ax)
    ps.panel(ax, 'Post-RL Shifts: Progress (EP) vs. Safety (NC/TTC)')

    methods = [
        'ReflectDrive-2\n(navtest)',
        'IRL-VLA\n(navhard S1)',
        'ReCogDrive\n(navtest)',
        'PaIR-Drive\n(navtest)',
        'Plan-R1\n(nuPlan CLS)'
    ]
    ep_deltas = np.array([+7.1, +12.3, +6.4, +3.9, +7.2])
    safety_deltas = np.array([-4.7, -1.4, -0.2, -13.5, -0.96])

    x = np.arange(len(methods))
    w = 0.35

    ax.bar(x - w/2, ep_deltas, width=w, color=ps.PALETTE['orange'], label=r'$\Delta$ Progress (EP / Progress)')
    ax.bar(x + w/2, safety_deltas, width=w, color=ps.PALETTE['vermillion'], label=r'$\Delta$ Safety / Comfort (Worst Drop)')

    ax.axhline(0, color='#666666', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=6.0)
    ax.set_ylabel(r'$\Delta$ Sub-score (percentage points)')
    ax.set_ylim(-15.0, 15.0)
    ax.legend(loc='lower left', fontsize=5.8, frameon=False)

    ps.save(fig, OUT_DIR / 'fig5_rl_subscore_tradeoff')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 6: Val vs Test Transfer of RL on WOD-E2E
# Sources: lit 1 §3.3, §4; lit 6 §1 #22, #23, #26, #28
# -----------------------------------------------------------------------------
def make_fig6():
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.30)
    ps.bars(ax)
    ps.panel(ax, 'WOD-E2E: RL Gain on Val vs. Held-Out Test')

    models = [
        'Qwen-Drive-1.0\n(RFS reward)',
        'MindVLA-U1\n(GRPO)',
        'SimWAM\n(LoRA-GRPO)',
        'DiffLTF\n(+val train)',
        'Poutine\n(GRPO)'
    ]
    val_gains = np.array([+0.50, +0.37, +0.30, +0.34, 0.20])
    test_gains = np.array([+0.13, +0.10, +0.07, -0.05, +0.077])

    x = np.arange(len(models))
    w = 0.35

    ax.bar(x - w/2, val_gains, width=w, color=ps.PALETTE['vermillion'], label='Val Gain (In-Sample Target)')
    ax.bar(x + w/2, test_gains, width=w, color=ps.PALETTE['blue'], label='Test Gain (Held-Out Benchmark)')

    ax.axhline(0, color='#666666', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=6.0)
    ax.set_ylabel(r'$\Delta$ RFS')
    ax.set_ylim(-0.10, 0.60)
    ax.legend(loc='upper right', fontsize=5.8, frameon=False)

    for i in range(3):
        ax.annotate(f'~25%\nkept', xy=(x[i]+w/2, test_gains[i]), xytext=(x[i]+w/2, test_gains[i]+0.06),
                    fontsize=5.6, color='#333333', ha='center')
    ax.annotate('−0.05\n(drop)', xy=(x[3]+w/2, test_gains[3]), xytext=(x[3]+w/2, test_gains[3]-0.08),
                fontsize=5.6, color=ps.PALETTE['vermillion'], ha='center')

    ps.save(fig, OUT_DIR / 'fig6_wod_rl_val_vs_test')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 7: Navhard vs Navtest Gain from Scorer & Synthetic Data
# Sources: lit 2 §3.1, §4.1; lit 5 §4.1, §4.2; lit 6 #13, #16, #17
# -----------------------------------------------------------------------------
def make_fig7():
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.30)
    ps.bars(ax)
    ps.panel(ax, 'navhard vs. navtest Gain from Scorer & Recovery Data')

    entries = [
        'SimScale\n(GTRS R34)',
        'SimScale\n(LTF)',
        'RAP\n(Jitter)',
        'DrivoR\n(+SimScale)',
        'TOAD\n(iPad w/ DrivoR)',
        'TOAD\n(DrivoR)'
    ]
    navhard_gains = np.array([+8.6, +5.8, +4.4, +6.3, +15.1, +1.7])
    navtest_gains = np.array([+2.3, +2.9, 0.0, +0.9, +1.7, +0.1])

    x = np.arange(len(entries))
    w = 0.35

    ax.bar(x - w/2, navhard_gains, width=w, color=ps.PALETTE['orange'], label='navhard Gain (EPDMS)')
    ax.bar(x + w/2, navtest_gains, width=w, color=ps.PALETTE['sky_blue'], label='navtest Gain (PDMS/EPDMS)')

    ax.axhline(0, color='#666666', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(entries, rotation=25, ha='right', fontsize=5.8)
    ax.set_ylabel(r'$\Delta$ Metric Score')
    ax.set_ylim(-0.5, 17.0)
    ax.legend(loc='upper right', fontsize=5.8, frameon=False)

    ps.save(fig, OUT_DIR / 'fig7_navhard_vs_navtest_gain')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 8: Parameter Count vs Score (Deployable Entries Only)
# Sources: lit 1 §1.1, §1.2; lit 2 §2, §3; decisions 180
# -----------------------------------------------------------------------------
def make_fig8():
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.20, wspace=0.30)

    # Panel (a): WOD-E2E test RFS vs Parameters
    ax = axes[0]
    ps.panel(ax, '(a) WOD-E2E (test RFS vs. Parameters)')
    
    wod_pts = [
        (1, 7.866, 'BBC (1M)'),
        (7.5, 7.849, 'ViT-Adapter'),
        (12, 7.711, 'E2EDriver'),
        (36, 7.765, 'TrajScorer'),
        (36, 7.543, 'Swin-Traj'),
        (44, 7.982, 'NTR (44M)'),
        (60, 7.780, 'UniPlan'),
        (70, 7.717, 'DiffusionLTF'),
        (86, 7.834, 'Traj-Refine'),
        (105, 7.856, 'FROST-Drive'),
        (382, 8.099, 'Ours WLG (382M)'),
        (888, 8.043, 'RAP-DINO'),
        (888, 8.046, 'NTR (888M)'),
        (2000, 8.060, 'DriveMA-2B'),
        (2200, 8.043, 'TTVLM-2B'),
        (3000, 7.909, 'Poutine-Base'),
        (3000, 7.986, 'Poutine'),
        (3100, 7.924, 'ReflexVLA'),
        (4000, 8.079, 'VMA-plus (4B)'),
        (4000, 8.167, 'ZSD-Titan (4B)*'),
        (5000, 7.910, 'Qwen-Drive (5B)'),
        (6000, 7.943, 'SUV (6B)')
    ]
    p_x = [p[0] for p in wod_pts]
    p_y = [p[1] for p in wod_pts]
    
    colors_wod = [ps.PALETTE['purple'] if 'Ours' in p[2] else (ps.PALETTE['vermillion'] if 'ZSD' in p[2] else (ps.PALETTE['blue'] if p[0] < 500 else ps.PALETTE['orange'])) for p in wod_pts]
    ax.scatter(p_x, p_y, color=colors_wod, s=22, zorder=2)
    ax.set_xscale('log')
    ax.set_xlim(0.7, 9000)
    ax.set_ylim(7.4, 8.35)
    ax.set_xlabel('Parameter Count (M, log scale)')
    ax.set_ylabel('test RFS')
    
    ax.annotate('Ours WLG (8.099)', xy=(382, 8.099), xytext=(80, 8.20),
                fontsize=6.0, weight='bold', color=ps.PALETTE['purple'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['purple'], lw=0.6))
    ax.annotate('BBC (1M: 7.87)', xy=(1, 7.866), xytext=(1.2, 7.62),
                fontsize=5.6, color='#444444',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.5))
    ax.annotate('NTR (44M: 7.98)', xy=(44, 7.982), xytext=(12, 8.06),
                fontsize=5.6, color=ps.PALETTE['blue'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['blue'], lw=0.5))
    ax.annotate('ZSD-Titan (8.17)*\n(no paper)', xy=(4000, 8.167), xytext=(1500, 8.24),
                fontsize=5.6, color=ps.PALETTE['vermillion'], ha='center')

    ax.axhline(8.13, color=ps.PALETTE['vermillion'], linestyle='--', linewidth=0.6, label='Human log (8.13)')
    ax.legend(loc='lower right', fontsize=6.0, frameon=False)

    # Panel (b): NAVSIM v1 PDMS vs Parameters
    ax = axes[1]
    ps.panel(ax, '(b) NAVSIM v1 (PDMS vs. Parameters)')
    nav_pts = [
        (21.8, 91.72, 'iPad (22M)'),
        (21.8, 92.10, 'SparseDriveV2'),
        (41, 94.59, 'DrivoR (41M)'),
        (54.6, 89.0, 'MeanFuser'),
        (60, 88.02, 'DiffusionDrive'),
        (61, 89.9, 'DriveSuprim-R34'),
        (66, 88.9, 'SeerDrive'),
        (68, 88.8, 'ResAD'),
        (74.8, 89.4, 'DiffRefiner'),
        (104, 89.3, 'Latent-WAM'),
        (110, 92.1, 'DriveSuprim-V2'),
        (118, 89.9, 'ForeDrive'),
        (338, 95.31, 'DriveZero-Scale'),
        (345, 93.5, 'DriveSuprim-ViTL'),
        (480, 91.8, 'WA-JEPA'),
        (888, 93.8, 'RAP-DINO'),
        (2000, 90.8, 'ReCogDrive-2B'),
        (2000, 94.85, 'ChainFlow-VLA'),
        (2500, 86.2, 'Epona (2.5B)'),
        (3000, 89.11, 'AutoVLA (3B)'),
        (5000, 90.2, 'SV-WAM (5B)'),
        (8000, 90.4, 'ReCogDrive-8B'),
        (16000, 92.02, 'DriveReferee (16B)')
    ]
    n_x = [p[0] for p in nav_pts]
    n_y = [p[1] for p in nav_pts]
    colors_nav = [ps.PALETTE['blue'] if p[0] < 500 else ps.PALETTE['orange'] for p in nav_pts]
    ax.scatter(n_x, n_y, color=colors_nav, s=22, zorder=2)
    ax.set_xscale('log')
    ax.set_xlim(15, 25000)
    ax.set_ylim(84.0, 97.5)
    ax.set_xlabel('Parameter Count (M, log scale)')
    ax.set_ylabel('PDMS')

    ax.annotate('DrivoR (41M: 94.6)', xy=(41, 94.59), xytext=(25, 96.2),
                fontsize=5.8, weight='bold', color=ps.PALETTE['blue'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['blue'], lw=0.6))
    ax.annotate('DriveZero-Scale\n(338M: 95.3)', xy=(338, 95.31), xytext=(220, 96.5),
                fontsize=5.6, color=ps.PALETTE['blue'], ha='center')
    ax.annotate('DriveReferee\n(16B: 92.0)', xy=(16000, 92.02), xytext=(7000, 89.2),
                fontsize=5.6, color=ps.PALETTE['orange'],
                arrowprops=dict(arrowstyle='->', color=ps.PALETTE['orange'], lw=0.5))

    ax.axhline(94.8, color=ps.PALETTE['vermillion'], linestyle='--', linewidth=0.6, label='Human log (94.8)')
    ax.legend(loc='lower right', fontsize=6.0, frameon=False)

    ps.save(fig, OUT_DIR / 'fig8_params_vs_score')
    plt.close(fig)

def main():
    setup()
    print("Generating Figure 1...")
    make_fig1()
    print("Generating Figure 2...")
    make_fig2()
    print("Generating Figure 3...")
    make_fig3()
    print("Generating Figure 4...")
    make_fig4()
    print("Generating Figure 5...")
    make_fig5()
    print("Generating Figure 6...")
    make_fig6()
    print("Generating Figure 7...")
    make_fig7()
    print("Generating Figure 8...")
    make_fig8()
    print("All 8 figures generated successfully.")

if __name__ == '__main__':
    main()
