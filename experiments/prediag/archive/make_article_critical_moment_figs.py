"""Figures for research/articles/vision-at-the-critical-moment/.

Every number comes from the small result tables under research/results/ or, where no table exists,
from research/decisions.md (hard-coded below with the entry it comes from). No experiment is run.

Run from the repo root:  .venv/bin/python experiments/prediag/archive/make_article_critical_moment_figs.py
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("research",)]
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research'))
import plot_style as ps  # noqa: E402

RESULTS_MOVED = {"alpamayo-smoke": "experiments/model_smoke/results/alpamayo-smoke", "b2d": "experiments/cl_infra/results/b2d", "b2d-family": "experiments/leaderboard_audit/results/b2d-family", "b2d-privileged-ceiling": "experiments/b2d_privileged/results", "carla-rewind": "experiments/carla_rewind/results", "cl-lib": "experiments/cl_infra/results/cl-lib", "cn_pair": "experiments/controlnet_pair/results", "cosmos": "experiments/cosmos/results", "driving-backbones": "experiments/driving_backbones/results/driving-backbones", "e0-layer": "experiments/feature_adapter/results/e0-layer", "e1-cosmos": "experiments/feature_adapter/results/e1-cosmos", "elicitation": "experiments/elicitation/results", "fast-perception": "experiments/fastperc/results", "fusion-diagnostics": "experiments/fusion_diag/results", "hack-audit": "experiments/leaderboard_audit/results/hack-audit", "hugsim": "experiments/hugsim/results/hugsim", "hugsim-exam": "experiments/hugsim/results/hugsim-exam", "i3-hugsim-pairs": "experiments/hugsim/results/i3-hugsim-pairs", "infra": "experiments/cl_infra/results/infra", "infra-acceptance": "experiments/cl_infra/results/infra-acceptance", "l0-surprise-weighting": "experiments/prediag/results/l0-surprise-weighting", "leaderboard-text-analysis": "experiments/leaderboard_audit/results/leaderboard-text-analysis", "log-expert-audit": "experiments/log_expert_audit/results", "navhard-deficit": "experiments/op_openloop/results/navhard-deficit", "navsim-openblas-audit": "experiments/op_openloop/results/navsim-openblas-audit", "navsim-zeroshot": "experiments/zeroshot_openloop/results/navsim-zeroshot", "night2": "experiments/night_queue_2/results", "nq3": "experiments/night_queue_3/results", "nq4": "experiments/night_queue_4/results", "nuscenes-zeroshot": "experiments/zeroshot_openloop/results/nuscenes-zeroshot", "op-adapt": "experiments/op_adapt_r1/results", "op-adapt-L": "experiments/op_adapt_l/results", "op-adapt-r2": "experiments/op_adapt_r2/results", "op-interp": "experiments/op_openloop/results/op-interp", "op-lb": "experiments/op_openloop/results/op-lb", "op_arb": "experiments/op_closed_loop/results/op_arb", "op_drive": "experiments/op_closed_loop/results/op_drive", "openpilot-migration": "experiments/model_smoke/results/openpilot-migration", "openpilot-openloop": "experiments/op_openloop/results/openpilot-openloop", "p0-train-split": "experiments/prediag/results/p0-train-split", "p1-judge": "experiments/prediag/results/p1-judge", "p1-judge-ladder": "experiments/prediag/results/p1-judge-ladder", "p2-readout-ladder": "experiments/prediag/results/p2-readout-ladder", "p2p3-subset": "experiments/prediag/results/p2p3-subset", "p3": "experiments/p3_ped_exam/results", "p3-backbone-ladder": "experiments/prediag/results/p3-backbone-ladder", "p3-qwenvid-trainfit": "experiments/prediag/results/p3-qwenvid-trainfit", "p3-qwenvid2b": "experiments/prediag/results/p3-qwenvid2b", "p3e-heads": "experiments/prediag/results/p3e-heads", "p4-carla-gap": "experiments/prediag/results/p4-carla-gap", "p5-carla-pairs": "experiments/reactivity/results/p5-carla-pairs", "p5-v1": "experiments/reactivity/results/p5-v1", "p5-vlm-metaaction": "experiments/reactivity/results/p5-vlm-metaaction", "pai-openpilot": "experiments/zeroshot_openloop/results/pai-openpilot", "qwenvid-train-profile": "experiments/prediag/results/qwenvid-train-profile", "reactivity": "experiments/reactivity/results/reactivity", "reactivity-i4": "experiments/reactivity/results/reactivity-i4", "real-data-transfer": "experiments/real_transfer/results", "skill-pack": "experiments/skill_pack/results", "tfv6-rules-interface": "experiments/tfv6_rules/results/tfv6-rules-interface", "top-decile-audit": "experiments/prediag/results/top-decile-audit", "top10-exams": "experiments/top10/results/top10-exams", "wl": "experiments/world_model/results/wl", "wl-dryrun": "experiments/world_model/results/wl-dryrun", "wl2": "experiments/world_model/results/wl2", "wod-zeroshot": "experiments/zeroshot_openloop/results/wod-zeroshot", "zeroshot-b2d": "experiments/zeroshot_b2d/results"}
OUT = ROOT / 'research' / 'articles' / 'vision-at-the-critical-moment' / 'figs'
P = ps.PALETTE

# One colour per arm family, shared by every figure in the article.
C = {
    'ego': ps.BASELINE,             # ridge ego, the base of every delta
    'A': P['blue'],                 # Qwen3-VL-4B single-frame pooled + ridge (arm A)
    'concat': P['sky_blue'],        # temporal concat of pooled vectors at the readout
    'neural': P['vermillion'],      # non-linear heads (MLP, attention, transformer, gated)
    'q32b': P['purple'],            # Qwen3-VL-32B
    'wan': P['black'],              # Wan2.2-TI2V-5B DiT
    'vjepa': P['green'],            # V-JEPA 2
    'video': P['orange'],           # Qwen3-VL native video path (d'')
    'cls': P['yellow'],             # fixed-vocabulary classification head
    'halfval': '#9A9A9A',           # half-val rehearsal markers (process, not the result)
}
THRESH = -0.05  # practical-effect threshold, decisions.md entry 20


def rj(path, judge='ADE vs log', scope='s_ego deciles 1-9', subset='pre_onset'):
    """Rows of a rejudge table under the entry-22 protocol."""
    head, _, tail = path.partition("/")  # restructure: research/results/<head> moved per topic
    d = pd.read_csv(ROOT / RESULTS_MOVED[head] / tail)
    return d[(d.judge == judge) & (d.scope == scope) & (d.subset == subset)]


def pick(d, arm, direction):
    r = d[(d.arm == arm) & (d.direction == direction)]
    assert len(r) == 1, (arm, direction, len(r))
    r = r.iloc[0]
    return float(r.delta), float(r.lo), float(r.hi), int(r.n) if not pd.isna(r.n) else None


def save(fig, stem):
    info = ps.save(fig, OUT / stem)
    plt.close(fig)
    kb = info['png_bytes'] / 1024
    print(f'{stem}: {info["width_in"]:.2f}x{info["height_in"]:.2f} in, png {kb:.0f} KB')
    assert info['png_under_500KiB'], stem


# --------------------------------------------------------------------------------------------
# Fig 1: vision - ego per subset and DiD, train split (P0) with half-val rehearsal for context.
# --------------------------------------------------------------------------------------------
def fig_subsets():
    paired = pd.read_csv(RES / 'p0-train-split' / 'paired.csv')
    did = pd.read_csv(RES / 'p0-train-split' / 'did.csv')
    arm = 'A ridge_late uniform'
    rows = [('all', 'All frames'), ('straight_yaw', 'Straight'), ('turn_yaw', 'Turning'),
            ('pre_onset', 'Pre-onset')]
    vals = []
    for key, lab in rows:
        r = paired[(paired.arm == arm) & (paired.subset == key)].iloc[0]
        vals.append((lab, int(r.n), r.dade, r.lo, r.hi))
    r = did[did.arm == arm].iloc[0]
    vals.append(('DiD', None, r.did, r.did_lo, r.did_hi))
    # Half-val rehearsal, decisions.md entry 3d (direction 0 / direction 1); turning was not reported there.
    half = {'All frames': [(-0.078, -0.101, -0.058), (-0.034, -0.056, -0.012)],
            'Straight': [(-0.125, -0.162, -0.091), (-0.072, -0.112, -0.033)],
            'Pre-onset': [(-0.027, -0.073, 0.017), (-0.043, -0.090, 0.006)],
            'DiD': [(0.098, 0.041, 0.158), (0.029, -0.036, 0.091)]}

    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.35))
    x = np.arange(len(vals), dtype=float)
    x[-1] += 0.35
    for i, (lab, n, m, lo, hi) in enumerate(vals):
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt='o', ms=4.2, color=C['A'], capsize=2,
                    elinewidth=.9, zorder=3, label='Train fit, full val' if i == 0 else None)
        for k, (hm, hlo, hhi) in enumerate(half.get(lab, [])):
            xo = x[i] + (0.22 if k == 0 else 0.34)
            ax.errorbar(xo, hm, yerr=[[hm - hlo], [hhi - hm]], fmt='o' if k == 0 else 's', ms=2.8,
                        mfc='white', mec=C['halfval'], color=C['halfval'], capsize=1.2, elinewidth=.6,
                        zorder=2, label=(f'Half-val, dir. {k}' if i == 0 else None))
    ax.axvline((x[-2] + x[-1]) / 2 + 0.1, color='#BBBBBB', lw=.5, ls=':')
    ps.zero_line(ax)
    labels = [f'{lab}\n$n$={n:,}' if n else f'{lab}\n(pre $-$ str.)' for lab, n, *_ in vals]
    ax.set_xticks(x + 0.1)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r'$\Delta$ADE, vision $-$ ego (m)')
    ax.set_ylim(-0.19, 0.19)
    ps.bars(ax)
    ax.legend(loc='upper left', fontsize=6.8, handlelength=1.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.97, bottom=0.2)
    save(fig, 'fig1-subset-delta-did')


# --------------------------------------------------------------------------------------------
# Fig 2: relative gain by s_ego decile (train split) + what the top decile is (rater frames).
# --------------------------------------------------------------------------------------------
def fig_decile():
    dec = pd.read_csv(RES / 'p0-train-split' / 'deciles.csv')
    dec = dec[dec.arm == 'A ridge_late uniform']
    bins = pd.read_csv(RES / 'top-decile-audit' / 'bins.csv')
    bins = bins[bins.binned_by == 's_ego (out-of-fold)'].sort_values('bin')

    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.15),
                            gridspec_kw={'width_ratios': [1.25, 1, 1]})
    ax = axs[0]
    for scope, col, ls, lab in [('all', C['A'], '-', 'All frames'),
                                ('straight_yaw', C['A'], '--', 'Straight frames only')]:
        s = dec[dec.scope == scope].sort_values('decile')
        k = s.decile.to_numpy() + 1
        g, lo, hi = (s[c].to_numpy() * 100 for c in ('rel_gain', 'rel_lo', 'rel_hi'))
        ax.plot(k, g, ls, marker='o', ms=2.8, color=col, label=lab, mfc=col if ls == '-' else 'white')
        ax.fill_between(k, lo, hi, color=col, alpha=.14 if ls == '-' else .08, lw=0)
    ax.axvspan(9.5, 10.5, color='#DDDDDD', alpha=.5, lw=0, zorder=0)
    ps.zero_line(ax)
    ax.set_ylim(-19, 12)
    ax.annotate('dec. 1–2: +86% / +33%\n(ego ADE 0.28 / 0.51 m)', xy=(2.75, 11.5), xytext=(4.3, 6.0), fontsize=6.5,
                ha='left', arrowprops=dict(arrowstyle='->', lw=.5, color='#555555'))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(r'$s_\mathrm{ego}$ decile (train fit, full val)')
    ax.set_ylabel(r'Relative gain, $\Delta$ADE / ego ADE (%)')
    ax.legend(loc='lower left', fontsize=6.8)
    ps.panel(ax, '(a)')

    k = bins.bin.to_numpy()
    ax = axs[1]
    ax.plot(k, bins.log_rfs, '-o', ms=2.8, color=P['black'], label='Logged future')
    ax.plot(k, bins.rater_max, ':', marker='^', ms=2.8, color='#555555', mfc='white', label='Best proposal label')
    ax.axhline(4.0, color='#999999', lw=.5, ls='--')
    ax.text(6.6, 4.12, 'RFS floor', fontsize=6.5, color='#666666')
    ax.set_ylim(3.6, 10.4)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(r'$s_\mathrm{ego}$ decile (rater frames, $n\approx48$ each)')
    ax.set_ylabel('RFS')
    ax.legend(loc='lower left', fontsize=6.8, bbox_to_anchor=(0, 0.1))
    ps.panel(ax, '(b)')

    ax = axs[2]
    ax.plot(k, bins.outside_all * 100, '-o', ms=2.8, color=P['vermillion'], label='Outside all 3 trust regions')
    ax.plot(k, bins.log_floored * 100, '--s', ms=2.6, color=P['vermillion'], mfc='white', label='Floored (RFS = 4)')
    ax.set_ylim(-3, 75)
    ax.set_ylabel('Logged future, % of frames')
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(r'$s_\mathrm{ego}$ decile (rater frames)')
    ax.legend(loc='upper left', fontsize=6.8)
    ps.panel(ax, '(c)')
    for a in axs[1:]:
        a.axvspan(9.5, 10.5, color='#DDDDDD', alpha=.5, lw=0, zorder=0)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.2, wspace=0.34)
    save(fig, 'fig2-decile-and-top-decile')


# --------------------------------------------------------------------------------------------
# Fig 3: the same predictions under different judges (P1). Improvement over ridge ego, >0 = better.
# --------------------------------------------------------------------------------------------
def fig_judges():
    d = pd.read_csv(RES / 'p1-judge' / 'judges.csv')
    arms = [('A ridge_late uniform', 'A: ridge, vision + ego', C['A'], 'o'),
            ('D mlp uniform', 'D: MLP, vision + ego', C['neural'], 's'),
            ('cls ego', 'cls: K=1024 classifier, ego', C['cls'], 'D')]
    metre = [('ADE vs log', 'all', 'ADE vs log, all frames'),
             ('ADE vs log', 'dec1-9', r'ADE vs log, $s_\mathrm{ego}$ dec. 1–9'),
             ('ADE vs log', 'dec10', 'ADE vs log, decile 10'),
             ('ADE vs log', 'rater credible', 'ADE vs log, rater-credible'),
             ('ADE vs rater_best', 'rater', 'ADE vs rater_best, rater frames'),
             ('minADE3', 'all', r'minADE$_3$, all frames')]
    rfs = [('RFS', 'rater', 'RFS, rater frames'), ('RFS', 'dec10', 'RFS, decile 10')]

    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.45),
                            gridspec_kw={'width_ratios': [2.0, 1]})
    for ax, rows, sign, xl in [(axs[0], metre, -1, 'Improvement over ridge ego (m)'),
                               (axs[1], rfs, 1, 'Improvement (RFS points)')]:
        ylabs = []
        for i, (judge, scope, lab) in enumerate(rows):
            sub = d[(d.judge == judge) & (d.scope == scope)]
            n = int(sub.n.iloc[0])
            ylabs.append(f'{lab} ($n$={n:,})')
            for j, (arm, _, col, mk) in enumerate(arms):
                r = sub[sub.arm == arm].iloc[0]
                m, lo, hi = sign * r.delta, sign * r.lo, sign * r.hi
                lo, hi = min(lo, hi), max(lo, hi)
                y = i + (j - 1) * 0.22
                ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, ms=3.4, color=col, mec='#333333' if arm == 'cls ego' else col,
                            mew=.4, capsize=1.3, elinewidth=.7, zorder=3)
        ax.axvline(0, color=C['ego'], lw=1.0, zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(ylabs, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        ax.grid(axis='y', visible=False)
    axs[0].text(0.012, -0.62, 'ridge ego', color=C['ego'], fontsize=6.5, ha='left')
    handles = [mpl.lines.Line2D([], [], ls='', marker=mk, color=col, mec='#333333' if 'cls' in lab else col, mew=.4,
                                ms=3.6, label=lab) for _, lab, col, mk in arms]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=7, bbox_to_anchor=(0.6, 1.0))
    fig.subplots_adjust(left=0.255, right=0.985, top=0.88, bottom=0.17, wspace=0.72)
    save(fig, 'fig3-judge-flip')


# --------------------------------------------------------------------------------------------
# Fig 4: the whole ladder, pre-onset delta on s_ego deciles 1-9, both half-val directions.
# --------------------------------------------------------------------------------------------
def fig_ladder():
    p2c, p2e = rj('p1-judge-ladder/rejudge_p2c.csv'), rj('p1-judge-ladder/rejudge_p2e.csv')
    p3, p3d = rj('p3-backbone-ladder/rejudge_p3_all.csv'), rj('p3-backbone-ladder/rejudge_p3d.csv')
    p2b = rj('p3-qwenvid2b/rejudge_p3.csv')
    rows = [  # (label, table, arm, colour, group)
        ('A  Qwen3-VL-4B, 1 frame, pooled', p2c, 'A ridge_late pooled', C['A'], 'P2 readout'),
        ('b  + temporal concat (4 x 0.2 s)', p2c, 'b temporal (subset)', C['concat'], 'P2 readout'),
        ('c  attention head on 144 tokens', p2c, 'c-attn grid', C['neural'], 'P2 readout'),
        ('c  MLP on pooled (compute-matched)', p2c, 'c-mlp pooled (compute-matched)', C['neural'], 'P2 readout'),
        ('c  transformer on tokens + ego', p2c, 'c-tf grid + ego token', C['neural'], 'P2 readout'),
        ('d  temporal + attention head', p2c, 'd temporal + attn grid', C['neural'], 'P2 readout'),
        ('e  gated residual, attention trunk', p2e, 'e gated attn grid', C['neural'], 'P2 readout'),
        ('e  gated residual, MLP trunk', p2e, 'e gated pooled-mlp', C['neural'], 'P2 readout'),
        ('a  Qwen3-VL-32B L32 last', p3, 'a qwen32b L32_last', C['q32b'], 'P3 backbone'),
        ('a  Qwen3-VL-32B L32 mean', p3, 'a qwen32b L32_mean', C['q32b'], 'P3 backbone'),
        ('a  Qwen3-VL-32B L50 last', p3, 'a qwen32b L50_last', C['q32b'], 'P3 backbone'),
        ('a  Qwen3-VL-32B L50 mean', p3, 'a qwen32b L50_mean', C['q32b'], 'P3 backbone'),
        (r'c  Wan2.2-5B DiT, $\sigma$=0.8, block 15', p3, 'c wan s0.8_b15_mean', C['wan'], 'P3 backbone'),
        ('d  V-JEPA 2 ViT-L, 4 frames', p3, 'd vjepa2 mean', C['vjepa'], 'P3 backbone'),
        ('d  V-JEPA 2 ViT-g, 4 frames*', p3d, 'd4 vjepa2 ViT-g', C['vjepa'], 'P3 backbone'),
        ("d'' Qwen3-VL-4B video, L18 last", p3, 'd2 qwenvid L18_last', C['video'], 'P3 backbone'),
        ("d'' Qwen3-VL-4B video, L18 mean", p3, 'd2 qwenvid L18_mean', C['video'], 'P3 backbone'),
        ("d'' Qwen3-VL-2B video, L14 last", p2b, 'd2b qwenvid2b L14_last', C['video'], 'P3 backbone'),
        ("d'' Qwen3-VL-2B video, L14 mean", p2b, 'd2b qwenvid2b L14_mean', C['video'], 'P3 backbone'),
    ]
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 4.1))
    y = 0.0
    ys, labs = [], []
    prev = None
    for lab, tab, arm, col, grp in rows:
        if prev is not None and grp != prev:
            y += 0.6
            ax.axhline(y - 0.8, color='#BBBBBB', lw=.5, ls=':')
        prev = grp
        both = True
        for dirn, mk, off in [(0, 'o', -0.16), (1, 's', 0.16)]:
            m, lo, hi, _ = pick(tab, arm, dirn)
            both &= hi < 0
            ax.errorbar(m, y + off, xerr=[[m - lo], [hi - m]], fmt=mk, ms=3.3, color=col,
                        mfc=col if dirn == 0 else 'white', mec=col, mew=.8, capsize=1.2, elinewidth=.75, zorder=3)
        if 'video' in lab:
            ax.axhspan(y - 0.45, y + 0.45, color=C['video'], alpha=.10, lw=0, zorder=0)
        if both:
            ax.text(0.338, y, 'both CIs < 0', fontsize=6.3, va='center', ha='right', color='#333333')
        ys.append(y)
        labs.append(lab)
        y += 1
    ax.axvline(0, color=C['ego'], lw=1.0, zorder=1)
    ax.axvline(THRESH, color='#444444', lw=.7, ls='--', zorder=1)
    ax.text(THRESH - 0.004, -1.05, '$-$0.05 m threshold', fontsize=6.5, ha='right', va='top')
    ax.text(0.004, -1.05, 'ridge ego', fontsize=6.5, ha='left', va='top', color=C['ego'])
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_ylim(y - 0.4, -1.6)
    ax.set_xlim(-0.24, 0.34)
    ax.set_xlabel(r'Pre-onset $\Delta$ADE vs ridge ego, $s_\mathrm{ego}$ deciles 1–9 (m)')
    ax.grid(axis='y', visible=False)
    for g, yy in [('P2 readout ladder', ys[0]), ('P3 backbone ladder', ys[8])]:
        ax.text(-0.236, yy - 0.62, g, fontsize=7, style='italic', color='#444444', va='bottom')
    handles = [mpl.lines.Line2D([], [], ls='', marker='o', color='#555555', ms=3.3, label='Direction 0'),
               mpl.lines.Line2D([], [], ls='', marker='s', color='#555555', mfc='white', ms=3.3, label='Direction 1')]
    ax.legend(handles=handles, loc='upper right', fontsize=7)
    fig.subplots_adjust(left=0.29, right=0.985, top=0.985, bottom=0.1)
    save(fig, 'fig4-ladder-forest')


# --------------------------------------------------------------------------------------------
# Fig 5: where the 0.6 s of history enters. Same subset, same head, same judge.
# --------------------------------------------------------------------------------------------
def fig_time():
    tabs = {'pre': (rj('p3-backbone-ladder/rejudge_p3_qwenvid.csv'), rj('p1-judge-ladder/rejudge_p2c.csv')),
            'str': (rj('p3-backbone-ladder/rejudge_p3_qwenvid.csv', subset='straight_yaw'),
                    rj('p1-judge-ladder/rejudge_p2c.csv', subset='straight_yaw')),
            'did': (rj('p3-backbone-ladder/rejudge_p3_qwenvid.csv', judge='DiD', subset='pre_onset - straight_yaw'),
                    rj('p1-judge-ladder/rejudge_p2c.csv', judge='DiD', subset='pre_onset - straight_yaw'))}
    train = {'pre': rj('p3-backbone-ladder/rejudge_p3train.csv'),
             'str': rj('p3-backbone-ladder/rejudge_p3train.csv', subset='straight_yaw'),
             'did': rj('p3-backbone-ladder/rejudge_p3train.csv', judge='DiD', subset='pre_onset - straight_yaw')}
    arms = [('A\n1 frame', 'A ridge_late pooled (qwen4b L18)', 0, C['A'], 'A ridge_late pooled (qwen4b L18)'),
            ('b\nconcat', 'b temporal (subset)', 1, C['concat'], None),
            ('V-JEPA 2\nViT-L', 'd vjepa2 mean', 0, C['vjepa'], 'd vjepa2 (train fit)'),
            ("d''\nL18 last", 'd2 qwenvid L18_last', 0, C['video'], None),
            ("d''\nL18 mean", 'd2 qwenvid L18_mean', 0, C['video'], None)]
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.35), sharey=False)
    for ax, key, yl, lab in [(axs[0], 'pre', r'Pre-onset $\Delta$ADE (m)', '(a)'),
                             (axs[1], 'str', r'Straight $\Delta$ADE (m)', '(b)'),
                             (axs[2], 'did', 'DiD, pre-onset $-$ straight (m)', '(c)')]:
        for i, (name, arm, src, col, tarm) in enumerate(arms):
            tab = tabs[key][src]
            for dirn, mk, off in [(0, 'o', -0.2), (1, 's', 0.0)]:
                m, lo, hi, _ = pick(tab, arm, dirn)
                ax.errorbar(i + off, m, yerr=[[m - lo], [hi - m]], fmt=mk, ms=3.2, color=col,
                            mfc=col if dirn == 0 else 'white', mec=col, mew=.8, capsize=1.2, elinewidth=.75)
            if tarm is not None:
                m, lo, hi, _ = pick(train[key], tarm, 0)
                ax.errorbar(i + 0.2, m, yerr=[[m - lo], [hi - m]], fmt='^', ms=3.4, color=col, mfc=col,
                            mec='#222222', mew=.4, capsize=1.2, elinewidth=.75)
        ps.zero_line(ax)
        if key == 'pre':
            ax.axhline(THRESH, color='#444444', lw=.7, ls='--')
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], fontsize=6.4, rotation=0)
        ax.set_ylabel(yl)
        ps.bars(ax)
        ps.panel(ax, lab)
    axs[0].set_ylim(-0.2, 0.08)
    axs[1].set_ylim(-0.15, 0.03)
    axs[2].set_ylim(-0.17, 0.18)
    handles = [mpl.lines.Line2D([], [], ls='', marker='o', color='#555555', ms=3.2, label='Half-val dir. 0'),
               mpl.lines.Line2D([], [], ls='', marker='s', color='#555555', mfc='white', ms=3.2, label='Half-val dir. 1'),
               mpl.lines.Line2D([], [], ls='', marker='^', color='#555555', mec='#222222', ms=3.4,
                                label='Train fit, full val')]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.075, right=0.972, top=0.86, bottom=0.2, wspace=0.42)
    save(fig, 'fig5-where-time-enters')


def copy_existing():
    """Frame montage of the top decile cannot be regenerated locally (needs camera frames): copy it."""
    for name in ['top-decile-rater-frames.png']:
        shutil.copy2(ROOT / 'research' / 'figs' / name, OUT / name)
        print(f'copied {name}')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ps.apply()
    fig_subsets()
    fig_decile()
    fig_judges()
    fig_ladder()
    fig_time()
    copy_existing()
