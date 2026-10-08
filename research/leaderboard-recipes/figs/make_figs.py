"""Generate figures for the leaderboard-recipes report.

Every plotted number is hard-coded next to a comment naming its source note
(research/lit/2026-10-08-*.md; VER = the verification note, which re-read the
primary tables). Protocols that are not comparable never share an axis:
navhard pre-fix vs post-fix, PDMS vs EPDMS, navval vs navtest, WOD val vs test.
Style: research/plot_style.py.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
import research.plot_style as ps

OUT_DIR = Path(__file__).resolve().parent
P = ps.PALETTE
LEG = dict(fontsize=5.6, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9, handlelength=1.6)


def _vals(ax, xs, tops, labels, dy, fs=5.6):
    for x, t, s in zip(xs, tops, labels):
        ax.text(x, t + dy, s, ha='center', va='bottom', fontsize=fs, color='#333333')


# -----------------------------------------------------------------------------
# Figure 1: cross-paper score ladders, one panel per board.
# (a) WOD note Q2: only 7.41 is a measured entry (Poutine ego-status MLP); the
#     other steps are mid-range picks of cross-paper ranges, not additive.
# (b) NAVSIM note 7.2 + 7.4; DrivoR / DriveZero papers; decision 170 (SH30).
# (c) navhard two-stage EPDMS, post-fix only: NAVSIM note 7.4; DrivoR 48.3 and
#     +134k SimScale 54.6; TOAD 56.3 (VER #13, #14); decision 170 (SH30 33.67).
# -----------------------------------------------------------------------------
def make_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 3.7))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.46, wspace=0.36)
    x = np.arange(5)
    below = dict(loc='upper center', bbox_to_anchor=(0.5, -0.52), **LEG)

    ax = axes[0]
    ps.bars(ax)
    ps.panel(ax, '(a) WOD-E2E, test RFS')
    steps = ['Ego-MLP 7.41\n(measured)', '+in-domain SFT\n(+0.35 to +0.5)', '+extra driving data\n(+0.1 to +0.15)',
             '+RL on val labels\n(+0.07 to +0.17)', '+ensemble\n(about +0.05)']
    base, delta = [0, 7.41, 7.87, 7.99, 8.10], [7.41, 0.46, 0.12, 0.11, 0.05]
    ax.bar(x, delta, bottom=base, width=0.55, edgecolor='none',
           color=[ps.BASELINE, P['sky_blue'], P['blue'], P['orange'], P['green']])
    ax.text(0, 7.43, '7.41', ha='center', va='bottom', fontsize=5.8)
    ax.axhline(8.099, color=P['purple'], ls=':', lw=0.9, label='Ours WLG 8.099')
    ax.axhline(8.167, color='#333333', ls='-.', lw=0.7, label='Top: ZSD-Titan 8.167')
    ax.set_ylim(7.0, 8.3)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=38, ha='right', fontsize=5.4)
    ax.set_ylabel('RFS (test)')
    ax.legend(**below)

    ax = axes[1]
    ps.bars(ax)
    ps.panel(ax, '(b) NAVSIM v1, navtest PDMS')
    steps = ['Ego-MLP', 'TransFuser (IL)', '+head, scorer / RL', 'DrivoR (trainval)', 'DriveZero-Scale']
    base, delta = [0, 65.6, 84.0, 90.0, 93.7], [65.6, 18.4, 6.0, 3.7, 1.6]
    ax.bar(x, delta, bottom=base, width=0.55, edgecolor='none',
           color=[ps.BASELINE, P['sky_blue'], P['orange'], P['orange'], P['green']])
    _vals(ax, x, [65.6, 84.0, 90.0, 93.7, 95.3], ['65.6', '84.0', '90-92', '93.7', '95.3'], 0.4)
    ax.axhline(94.8, color=P['vermillion'], ls='--', lw=0.8, label='Human log 94.8')
    ax.axhline(91.1, color='#666666', ls='-.', lw=0.7, label='MemoryDrivoR, no current cameras 91.1')
    ax.axhline(89.55, color=P['purple'], ls=':', lw=0.9, label='Ours SH30 89.55')
    ax.set_ylim(55.0, 100.0)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=38, ha='right', fontsize=5.6)
    ax.set_ylabel('PDMS (navtest)')
    ax.legend(**below)

    ax = axes[2]
    ps.bars(ax)
    ps.panel(ax, '(c) navhard, EPDMS (post-fix)')
    steps = ['Ego-MLP', 'LTF (IL)', 'DrivoR\n(proposals+scorer)', '+SimScale 134k', '+TOAD search']
    base, delta = [0, 14.1, 25.1, 48.3, 54.6], [14.1, 11.0, 23.2, 6.3, 1.7]
    ax.bar(x, delta, bottom=base, width=0.55, edgecolor='none',
           color=[ps.BASELINE, P['sky_blue'], P['orange'], P['blue'], P['green']])
    _vals(ax, x, [14.1, 25.1, 48.3, 54.6, 56.3], ['14.1', '25.1', '48.3', '54.6', '56.3'], 0.5)
    ax.axhline(61.02, color='#333333', ls='-.', lw=0.7, label='Top: RoboTruck 61.02')
    ax.axhline(56.6, color=P['vermillion'], ls='--', lw=0.8, label='PDM-Closed (privileged) 56.6')
    ax.axhline(33.67, color=P['purple'], ls=':', lw=0.9, label='Ours SH30 33.67')
    ax.set_ylim(0.0, 72.0)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=38, ha='right', fontsize=5.6)
    ax.set_ylabel('EPDMS (navhard two-stage, post-fix)')
    ax.legend(**below)

    ps.save(fig, OUT_DIR / 'fig1_score_decomposition')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: gain from a world-model / future objective vs the baseline it was
# added to. One point per paper, within-paper with / without, single runs.
# Sources: NAVSIM note section 1 item 2 and world-model tables; VER #4, #6-#11.
# (x, y, name, metric, short_schedule)
# -----------------------------------------------------------------------------
def make_fig2():
    pts = [
        (68.7, 12.0, 'DriveVLA-W0 (VQ)', 'PDMS', False),
        (70.3, 13.6, 'DriveVLA-W0 (ViT)', 'PDMS', False),
        (77.5, 7.1, 'LAW', 'PDMS', False),
        (78.1, 8.1, 'Epona', 'PDMS', False),
        (83.1, 4.6, 'SV-WAM', 'EPDMS', False),
        (83.2, 2.4, 'WoTE (rollout scoring)', 'PDMS', True),      # 20-epoch ablation, not the 88.3 headline
        (83.6, 0.9, 'DriveX', 'PDMS', False),
        (84.4, 2.5, 'EponaV2 2B', 'PDMS', False),
        (86.9, 0.6, 'PerceptDrive', 'PDMS', False),
        (87.0, 1.1, 'WorldDrive', 'PDMS', False),
        (87.3, 0.8, 'PWM', 'PDMS', False),
        (87.7, 0.3, 'Latent-WAM', 'EPDMS', False),               # source values +0.3 / +0.4
        (87.9, 1.0, 'SeerDrive', 'PDMS', False),
        (88.0, 1.2, 'DriveDreamer-Policy', 'PDMS', False),
        (88.9, 0.7, 'ForeDrive', 'PDMS', False),
        (88.9, 0.1, 'Metis (video at inf.)', 'PDMS', False),
        (90.84, 0.21, 'EditWM', 'EPDMS', False),
        (90.84, -0.33, 'EditWM, plain rollout', 'EPDMS', False),
        (91.1, 0.6, 'WA-JEPA (flow match.)', 'EPDMS', False),
        (93.31, 0.37, 'DA-WAM (+hard neg.)', 'PDMS', True),  # 20 epochs, single seed
        (93.31, -0.50, 'DA-WAM, shared future', 'PDMS', True),
    ]
    fig = plt.figure(figsize=(ps.DOUBLE_COLUMN_IN, 3.1))
    ax = fig.add_axes([0.075, 0.15, 0.50, 0.75])
    ps.panel(ax, 'Gain from a world-model / future objective vs. baseline score')
    for i, (x, y, _, metric, short) in enumerate(pts, 1):
        color = P['vermillion'] if x < 80 else P['blue']
        ax.scatter([x], [y], s=30, zorder=2, marker='o' if metric == 'PDMS' else 's',
                   facecolors='none' if short else color, edgecolors=color, linewidths=0.9)
        ax.annotate(str(i), (x, y), xytext=(3.2, 2.6), textcoords='offset points', fontsize=5.4, color='#222222')
    ax.axhline(0, color='#999999', lw=0.5, zorder=0)
    ax.axvline(84.0, color='#bbbbbb', lw=0.5, ls=':', zorder=0)
    ax.text(84.2, 11.5, 'TransFuser 84.0', fontsize=5.4, color='#777777')
    ax.set_xlim(65, 96)
    ax.set_ylim(-1.5, 16.5)
    ax.set_xlabel('score of the baseline without the WM / future objective (PDMS or EPDMS, see marker)')
    ax.set_ylabel('gain in the same metric')
    handles = [Line2D([], [], marker='o', ls='', color=P['blue'], markersize=4.5, label='PDMS (v1 navtest)'),
               Line2D([], [], marker='s', ls='', color=P['blue'], markersize=4.5, label='EPDMS (v2 navtest)'),
               Line2D([], [], marker='o', ls='', markerfacecolor='none', color=P['blue'], markersize=4.5,
                      label='short-schedule ablation')]
    ax.legend(handles=handles, loc='upper right', **LEG)
    half = (len(pts) + 1) // 2
    for col, chunk in enumerate((list(enumerate(pts, 1))[:half], list(enumerate(pts, 1))[half:])):
        txt = '\n'.join(f'{i:>2}  {p[2]}  ({p[1]:+g})' for i, p in chunk)
        fig.text(0.60 + 0.20 * col, 0.88, txt, fontsize=5.0, va='top', ha='left', linespacing=1.55, color='#222222')
    ps.save(fig, OUT_DIR / 'fig2_wm_gain_vs_baseline')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: language at inference, on minus off, within one paper. Three panels
# because the metrics differ (RFS / PDMS / Bench2Drive DS and SR).
# Sources: mechanisms note Q3; WOD note (Poutine Table 3, AutoVLA Table S4);
# VER #1, #3, #26, #30 (BLUE on NAVSIM is a head switch, not a CoT switch).
# -----------------------------------------------------------------------------
def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.9),
                             gridspec_kw=dict(width_ratios=[2, 5, 3]))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.30, wspace=0.36)
    fig.suptitle('Language at inference: within noise unless gated', fontsize=8.5, y=0.975)
    c_on, c_neg, c_gate = P['sky_blue'], P['vermillion'], P['green']

    def bar(ax, x, v, gated=False, w=0.36):
        ax.bar(x, v, width=w, color=c_gate if gated else (c_neg if v < 0 else c_on))
        ax.text(x, v + (0.02 if v >= 0 else -0.02) * (ax.get_ylim()[1] - ax.get_ylim()[0]), f'{v:+.2f}'.rstrip('0').rstrip('.'),
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=5.6)

    ax = axes[0]
    ps.bars(ax)
    ps.panel(ax, '(a) WOD-E2E')
    ax.set_ylim(-0.12, 0.12)
    bar(ax, 0, -0.04)     # Poutine: CoT at inference 8.08 vs 8.12, val, 479 frames
    bar(ax, 1, +0.041)    # AutoVLA: 7.406 -> 7.447, test (CoT also in training for that row)
    ax.axhline(0, color='#666666', lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Poutine\n(val)', 'AutoVLA\n(test)'], fontsize=5.8)
    ax.set_ylabel(r'$\Delta$ RFS')

    ax = axes[1]
    ps.bars(ax)
    ps.panel(ax, '(b) NAVSIM navtest')
    ax.set_ylim(-2.8, 2.8)
    bar(ax, 0, -0.10)                       # ReCogDrive: with CoT 90.7 vs trajectory-only 90.8
    bar(ax, 0.82, +0.60)                    # AdaThinkDrive: always 88.9 vs never 88.3
    bar(ax, 1.18, +2.00, gated=True)        # AdaThinkDrive: adaptive 90.3
    bar(ax, 1.82, -1.85)                    # BLUE on ReCogDrive: 84.13 vs 85.98 (text-trajectory head vs IL head)
    bar(ax, 2.18, +1.02, gated=True)        # gated 87.00
    ax.axhline(0, color='#666666', lw=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['ReCogDrive', 'AdaThinkDrive', 'BLUE/ReCogDrive\n(text-traj head vs IL head)'], fontsize=5.8)
    ax.set_ylabel(r'$\Delta$ PDMS')

    ax = axes[2]
    ps.bars(ax)
    ps.panel(ax, '(c) Bench2Drive')
    ax.set_ylim(-4.0, 8.5)
    bar(ax, 0, +0.66)                       # SimLingo: 85.07 +/- 0.95 vs 84.41 +/- 1.76 DS, not significant
    bar(ax, 0.82, -2.64)                    # BLUE on SimLingo SR: always 66.91 vs never 69.55
    bar(ax, 1.18, +6.63, gated=True)        # learned gate 76.18
    ax.axhline(0, color='#666666', lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['SimLingo\n(DS)', 'BLUE/SimLingo\n(SR)'], fontsize=5.8)
    ax.set_ylabel(r'$\Delta$ DS or SR (points)')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c_on, label='always on, positive'),
               plt.Rectangle((0, 0), 1, 1, color=c_neg, label='always on, negative'),
               plt.Rectangle((0, 0), 1, 1, color=c_gate, label='learned / adaptive gate')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, 0.0))
    ps.save(fig, OUT_DIR / 'fig3_language_inference_effect')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 4: within-paper encoder swaps. Three separate metrics / splits.
# (a) WA-JEPA Tab. 4a, NAVSIM-v2 navtest EPDMS (VER #9).
# (b) Drive-JEPA perception-free table, navtest PDMS (NAVSIM note calls it
#     Table 7, the deep-read note Tab. 5; unresolved, not in VER).
# (c) DrivoR Tab. 4a, navval PDMS, ViT-S, 10 epochs (VER #14).
# -----------------------------------------------------------------------------
def make_fig4():
    fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.8))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.32, wspace=0.34)

    def draw(ax, title, names, scores, colors, ylim, ylabel, arrow, fs=5.8, rot=0):
        ps.bars(ax)
        ps.panel(ax, title)
        x = np.arange(len(scores))
        ax.bar(x, scores, color=colors, width=0.55)
        ax.set_ylim(*ylim)
        _vals(ax, x, scores, [f'{s:g}' for s in scores], (ylim[1] - ylim[0]) * 0.012)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=fs, rotation=rot, ha='right' if rot else 'center')
        ax.set_ylabel(ylabel)
        i0, i1, label = arrow
        y = max(scores[i0], scores[i1]) + (ylim[1] - ylim[0]) * 0.12
        ax.annotate('', xy=(i1, y), xytext=(i0, y), arrowprops=dict(arrowstyle='<->', color=P['blue'], lw=0.7))
        ax.text((i0 + i1) / 2, y + (ylim[1] - ylim[0]) * 0.015, label, ha='center', fontsize=6.0, color=P['blue'], weight='bold')

    draw(axes[0], '(a) WA-JEPA, navtest EPDMS',
         ['MAE\n(image)', 'SigLIP2\n(VL)', 'DINOv3\n(image)', 'V-JEPA 2\n(video)'],
         [83.8, 83.1, 83.8, 89.5], [ps.BASELINE, P['sky_blue'], ps.BASELINE, P['blue']], (80.0, 93.0), 'EPDMS (v2 navtest)',
         (2, 3, '+5.7'))
    draw(axes[1], '(b) Drive-JEPA, navtest PDMS',
         ['ImageNet\nResNet34', 'DINOv2\nViT-L', 'SigLIP\nViT-L', 'V-JEPA 2\nViT-L', '+driving\nvideo\npretrain'],
         [76.0, 76.1, 83.4, 86.1, 89.0], [ps.BASELINE, ps.BASELINE, P['sky_blue'], P['blue'], P['green']], (70.0, 94.0),
         'PDMS (navtest, perception-free)', (1, 3, '+10.0'), fs=5.2)
    draw(axes[2], '(c) DrivoR, navval PDMS (ViT-S)',
         ['random\ninit', 'ImageNet-21k\n(supervised)', 'DINOv2\n(image-SSL)'],
         [70.1, 87.5, 90.0], [ps.BASELINE, P['sky_blue'], P['blue']], (65.0, 97.0), 'PDMS (navval)', (0, 2, '+19.9'))
    ps.save(fig, OUT_DIR / 'fig4_encoder_pretrain_comparison')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 5: sub-score change after RL on the board metric, within paper.
# ReflectDrive-2 (navtest): EP 82.2 -> 89.3, NC 97.8 -> 96.3, TTC 93.6 -> 88.9 (VER #21).
# IRL-VLA (navhard Stage-1 sub-metrics): EP 83.9 -> 96.2, NC 98.3 -> 96.9,
#   DAC 92.4 -> 91.3, EC 76.0 -> 72.4 (VER #21).
# ReCogDrive (navtest): EP 80.9 -> 87.3, NC 98.1 -> 97.9, DAC 94.7 -> 97.3 (mechanisms note Q4).
# -----------------------------------------------------------------------------
def make_fig5():
    groups = [('ReflectDrive-2\n(navtest)', [('EP', +7.1), ('NC', -1.5), ('TTC', -4.7)]),
              ('IRL-VLA\n(navhard Stage 1)', [('EP', +12.3), ('NC', -1.4), ('DAC', -1.1), ('EC', -3.6)]),
              ('ReCogDrive\n(navtest)', [('EP', +6.4), ('NC', -0.2), ('DAC', +2.6)])]
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN * 1.25, 2.7))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.20)
    ps.bars(ax)
    ps.panel(ax, 'After RL on the metric: EP up, NC / TTC / EC flat or down')
    x, centers = 0.0, []
    for name, bars in groups:
        xs = x + np.arange(len(bars)) * 0.6
        for xi, (k, v) in zip(xs, bars):
            ax.bar(xi, v, width=0.5, color=P['orange'] if k == 'EP' else (P['vermillion'] if v < 0 else P['sky_blue']))
            ax.text(xi, v + (0.3 if v >= 0 else -0.3), f'{k}\n{v:+.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=5.6)
        centers.append(xs.mean())
        x = xs[-1] + 1.2
    ax.axhline(0, color='#666666', lw=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.2)
    ax.set_ylabel(r'$\Delta$ sub-score after RL (points), within paper')
    ax.set_ylim(-8.5, 15.5)
    ps.save(fig, OUT_DIR / 'fig5_rl_subscore_tradeoff')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 6: WOD-E2E gain of the post-training step on val (in-sample: the step
# was trained on val labels) vs test (held out).
# Qwen-Drive 7.95 -> 8.45 val, 7.78 -> 7.91 test (VER #22); MindVLA-U1 7.83 ->
# 8.20, 7.77 -> 7.87 (VER #23); SimWAM 7.99 -> 8.29, 7.77 -> 7.84 (WOD note);
# Poutine test 7.909 -> 7.986 (board), no val number (val is its RL set);
# DiffusionLTF adds val logs to SUPERVISED training: val 7.86 -> 8.20,
# test 7.54 -> 7.49 (VER #28).
# -----------------------------------------------------------------------------
def make_fig6():
    models = ['Qwen-Drive-1.0\n(RFS + 2xADE reward)', 'MindVLA-U1\n(GRPO)', 'SimWAM\n(LoRA-GRPO)', 'Poutine\n(GRPO)',
              'DiffusionLTF\n(supervised on val logs,\nnot RL)']
    val = [0.50, 0.37, 0.30, np.nan, 0.34]
    test = [0.13, 0.10, 0.07, 0.077, -0.05]
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN * 1.3, 2.9))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.32)
    ps.bars(ax)
    ps.panel(ax, 'WOD-E2E: gain on val (in-sample) vs. test (held-out)')
    w = 0.35
    for i, (v, t) in enumerate(zip(val, test)):
        hatch = '///' if i == 4 else None
        if not np.isnan(v):
            ax.bar(i - w / 2, v, width=w, color=P['vermillion'], hatch=hatch, edgecolor='white', lw=0)
            ax.text(i - w / 2, v + 0.008, f'{v:+.2f}', ha='center', fontsize=5.6)
        else:
            ax.text(i, 0.125, 'val = RL training set,\nno val number', ha='center', va='bottom', fontsize=4.8, color='#666666')
        ax.bar(i + w / 2, t, width=w, color=P['blue'], hatch=hatch, edgecolor='white', lw=0)
        ax.text(i + w / 2, t + (0.008 if t >= 0 else -0.008), (f'{t:+.3f}' if i == 3 else f'{t:+.2f}'), ha='center', va='bottom' if t >= 0 else 'top', fontsize=5.6)
    ax.axvline(3.5, color='#bbbbbb', lw=0.5, ls=':')
    ax.axhline(0, color='#666666', lw=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=5.6)
    ax.set_ylabel(r'$\Delta$ RFS from the post-training step')
    ax.set_ylim(-0.12, 0.62)
    handles = [plt.Rectangle((0, 0), 1, 1, color=P['vermillion'], label='val (in-sample)'),
               plt.Rectangle((0, 0), 1, 1, color=P['blue'], label='test (held-out)')]
    ax.legend(handles=handles, loc='upper right', **LEG)
    ps.save(fig, OUT_DIR / 'fig6_wod_rl_val_vs_test')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 7: navhard gain vs navtest gain, with minus without, same paper.
# Deltas only: pairs mix navhard pre-fix / post-fix and PDMS / EPDMS.
# SimScale Table 1 / 2 (VER #16); RAP Table 6 (VER #17); DrivoR +134k SimScale
# 48.3 -> 54.6, 93.7 -> 94.6 (VER #14); TOAD iPad 34.7 -> 49.8, 91.7 -> 93.4,
# DrivoR-Scale 54.6 -> 56.3, 94.6 -> 94.7 (VER #13).
# -----------------------------------------------------------------------------
def make_fig7():
    entries = ['SimScale on GTRS-Dense R34\n(navhard pre-fix / navtest EPDMS)', 'SimScale on LTF\n(navhard pre-fix / navtest EPDMS)',
               'RAP pose jitter 8.5k\n(navhard pre-fix / navtest PDMS)', 'DrivoR +134k SimScale\n(post-fix / PDMS)',
               'TOAD on iPad (DrivoR scorer)\n(post-fix / PDMS)', 'TOAD on DrivoR-Scale\n(post-fix / PDMS)']
    hard = [8.6, 5.8, 4.4, 6.3, 15.1, 1.7]
    test = [2.3, 2.9, 0.0, 0.9, 1.7, 0.1]
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 3.0))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.36)
    ps.bars(ax)
    ps.panel(ax, 'Gain on navhard vs. navtest from recovery data and scorer search (within paper)')
    x, w = np.arange(len(entries)), 0.35
    ax.bar(x - w / 2, hard, width=w, color=P['orange'], label='navhard gain')
    ax.bar(x + w / 2, test, width=w, color=P['sky_blue'], label='navtest gain')
    _vals(ax, x - w / 2, hard, [f'+{v:g}' for v in hard], 0.2)
    _vals(ax, x + w / 2, test, [f'+{v:g}' if v else '0.0' for v in test], 0.2)
    ax.text(4 + 0.02, 11.6, 'scorer swap\nalone +10.9', ha='left', fontsize=5.4, color='#444444')
    ax.axhline(0, color='#666666', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(entries, rotation=22, ha='right', fontsize=5.6)
    ax.set_ylabel('gain in the board metric,\nwith minus without (same paper)')
    ax.set_ylim(-0.5, 17.5)
    ax.legend(loc='upper left', **LEG)
    ps.save(fig, OUT_DIR / 'fig7_navhard_vs_navtest_gain')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 8: parameter count vs score, board reads + paper-reported sizes,
# cross-system. (a) WOD-E2E test RFS (board JSON 2026-10-08; decision 180 for
# ours). SUV plotted at 6B (board says 5B), RAP at 888M (board says 1B).
# (b) NAVSIM v1 navtest PDMS (HF board + papers).
# kind: 'p' paper, single model; 'e' ensemble; 'n' no public material;
#       'o' ours (not on the public board, 2-seed trajectory mean);
#       't' trainval or +SimScale variant.
# -----------------------------------------------------------------------------
def make_fig8():
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 3.1))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.25, wspace=0.22)
    mk = {'p': 'o', 'e': 'D', 'n': 'x', 'o': '*', 't': '^'}

    def scatter(ax, pts):
        for x, y, _, kind in pts:
            color = P['purple'] if kind == 'o' else ('#555555' if kind == 'n' else (P['blue'] if x < 500 else P['orange']))
            ax.scatter([x], [y], marker=mk[kind], s=70 if kind == 'o' else 22, color=color, zorder=3 if kind == 'o' else 2,
                       linewidths=0.9 if kind == 'n' else 0.3)

    def note(ax, text, xy, xytext, color='#333333'):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=5.4, color=color,
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.4))

    ax = axes[0]
    ps.panel(ax, '(a) WOD-E2E, test RFS vs. parameters')
    wod = [
        (1, 7.866, 'BBC', 'n'), (7.5, 7.849, 'ViT-Adapter-GRU', 'p'), (12, 7.711, 'E2EDriver', 'n'),
        (36, 7.765, 'TrajScorer', 'n'), (36, 7.543, 'Swin-Trajectory', 'p'), (44, 7.982, 'NTR 44M', 'p'),
        (60, 7.780, 'UniPlan', 'p'), (70, 7.717, 'DiffusionLTF', 'p'), (86, 7.834, 'Traj-Refine', 'n'),
        (105, 7.856, 'FROST-Drive', 'p'), (382, 8.099, 'Ours WLG', 'o'), (888, 8.043, 'RAP', 'e'),
        (888, 8.046, 'NTR 888M', 'e'), (915, 8.025, 'QIRL-E2E', 'n'), (2000, 8.060, 'VMA (2B)', 'p'),
        (2000, 8.075, 'DriveMA-2B', 'p'), (2000, 8.090, 'ZSD (2B)', 'n'), (2000, 8.087, 'PlusAI-WorldVLA-2B', 'n'),
        (2000, 8.048, 'Zero-1', 'n'), (2200, 8.071, 'PWVLA-2B', 'n'), (2200, 8.043, 'TTVLM-2B', 'p'),
        (3000, 7.909, 'Poutine-Base', 'p'), (3000, 7.986, 'Poutine', 'p'), (3100, 7.924, 'ReflexVLA-DTS', 'n'),
        (4000, 8.079, 'VMA-plus (4B)', 'p'), (4000, 8.167, 'ZSD-Titan (4B)', 'n'), (5000, 7.910, 'Qwen-Drive', 'p'),
        (6000, 7.943, 'SUV', 'p'), (10000, 8.082, 'VAIL+', 'n'), (10000, 8.061, 'VAIL', 'n'),
    ]
    scatter(ax, wod)
    ax.set_xscale('log')
    ax.set_xlim(0.6, 20000)
    ax.set_ylim(7.45, 8.26)
    ax.set_xlabel('parameters (M, log scale)')
    ax.set_ylabel('test RFS')
    note(ax, 'Ours WLG 8.099 (382M)', (382, 8.099), (40, 8.19), P['purple'])
    note(ax, 'BBC 1M 7.866', (1, 7.866), (1.1, 7.67))
    note(ax, 'NTR 44M 7.982', (44, 7.982), (6, 8.06))
    note(ax, 'ZSD-Titan 8.167', (4000, 8.167), (520, 8.215))
    note(ax, 'NTR 888M / RAP\n8.046 / 8.043', (888, 8.045), (95, 7.93))
    note(ax, 'VAIL+ 8.082', (10000, 8.082), (5200, 8.20))
    note(ax, 'Qwen-Drive 7.91', (5000, 7.910), (4200, 7.74))
    note(ax, 'Swin-Trajectory 7.543', (36, 7.543), (60, 7.50))

    ax = axes[1]
    ps.panel(ax, '(b) NAVSIM v1, navtest PDMS vs. parameters')
    nav = [
        (21.8, 92.10, 'SparseDriveV2', 'p'), (41, 94.59, 'DrivoR +SimScale', 't'), (54.6, 89.0, 'MeanFuser', 'p'),
        (60, 88.02, 'DiffusionDrive', 'p'), (61, 89.9, 'DriveSuprim R34', 'p'), (66, 88.9, 'SeerDrive', 'p'),
        (68, 88.8, 'ResAD', 'p'), (74.8, 89.4, 'DiffRefiner', 'p'), (110, 92.1, 'DriveSuprim V2-99', 'p'),
        (118, 89.9, 'ForeDrive', 'p'), (338, 95.31, 'DriveZero-Scale', 't'), (345, 93.5, 'DriveSuprim ViT-L', 'p'),
        (888, 93.8, 'RAP-DINO', 'p'), (2000, 90.8, 'ReCogDrive-2B', 'p'), (2000, 94.85, 'ChainFlow-VLA', 't'),
        (2500, 86.2, 'Epona', 'p'), (3000, 89.11, 'AutoVLA', 'p'), (5000, 90.2, 'SV-WAM', 'p'),
        (8000, 90.4, 'ReCogDrive-8B', 'p'), (16000, 92.02, 'DriveReferee', 'p'),
    ]
    scatter(ax, nav)
    ax.axhline(94.8, color=P['vermillion'], ls='--', lw=0.6)
    ax.text(75, 94.95, 'Human log 94.8', fontsize=5.4, color=P['vermillion'])
    ax.set_xscale('log')
    ax.set_xlim(15, 30000)
    ax.set_ylim(85.0, 97.0)
    ax.set_xlabel('parameters (M, log scale)')
    ax.set_ylabel('navtest PDMS')
    note(ax, 'DrivoR 41M 94.59\n(+SimScale)', (41, 94.59), (18, 96.0))
    note(ax, 'DriveZero-Scale 338M 95.31', (338, 95.31), (150, 96.4))
    note(ax, 'ChainFlow-VLA 2B 94.85\n(trainval)', (2000, 94.85), (2600, 95.9))
    note(ax, 'RAP-DINO 888M 93.8', (888, 93.8), (1100, 92.9))
    note(ax, 'SparseDriveV2 22M 92.1', (21.8, 92.10), (17, 90.9))
    note(ax, 'DriveReferee 16B 92.02', (16000, 92.02), (3200, 91.4))
    note(ax, 'ReCogDrive 2B / 8B\n90.8 / 90.4', (8000, 90.4), (3800, 88.6))
    note(ax, 'Epona 2.5B 86.2', (2500, 86.2), (500, 85.6))
    note(ax, 'AutoVLA 3B 89.11', (3000, 89.11), (300, 87.6))

    handles = [Line2D([], [], marker='o', ls='', color=P['blue'], markersize=4, label='paper, < 500M'),
               Line2D([], [], marker='o', ls='', color=P['orange'], markersize=4, label='paper, >= 500M'),
               Line2D([], [], marker='D', ls='', color=P['orange'], markersize=4, label='ensemble'),
               Line2D([], [], marker='^', ls='', color=P['blue'], markersize=4.5, label='trainval or +SimScale variant'),
               Line2D([], [], marker='x', ls='', color='#555555', markersize=4.5, label='no public material'),
               Line2D([], [], marker='*', ls='', color=P['purple'], markersize=7, label='ours (not on the public board)')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    ps.save(fig, OUT_DIR / 'fig8_params_vs_score')
    plt.close(fig)


def main():
    ps.apply()
    for i, f in enumerate((make_fig1, make_fig2, make_fig3, make_fig4, make_fig5, make_fig6, make_fig7, make_fig8), 1):
        f()
        print(f'figure {i} done')


if __name__ == '__main__':
    main()
