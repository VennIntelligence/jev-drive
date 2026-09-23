"""Figures for the lateral v2 report from the small analysis outputs (run on the Mac, repo .venv).

    .venv/bin/python make_figures.py --results results/analysis-v1 --out figs/lateral-v2
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'research'))
import plot_style  # noqa: E402

P = plot_style.PALETTE
ARMS = [('prod-k0', 'Production, k = 0', P['orange'], '-'),
        ('prod-kfix', 'Production, fixed k', P['vermillion'], '-'),
        ('slipack-k0', 'Slip + Ackermann, k = 0', P['sky_blue'], '-'),
        ('slipack-kfix', 'Slip + Ackermann, fixed k', P['blue'], '-'),
        ('prod-truth', 'Production, truth pose', P['vermillion'], ':'),
        ('slipack-truth', 'Slip + Ackermann, truth pose', P['blue'], ':')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import matplotlib.pyplot as plt
    plot_style.apply()
    summary = json.loads((args.results / 'summary.json').read_text())
    windows = summary['windows']
    profile = {}
    for row in csv.DictReader((args.results / 'station-profiles.csv').open()):
        profile.setdefault((row['window'], row['arm']), []).append(
            (float(row['station_m']) + .5, float(row['cte_median']), float(row['cte_q25']), float(row['cte_q75'])))
    fig = plt.figure(figsize=(plot_style.DOUBLE_COLUMN_IN, 4.3))
    grid = fig.add_gridspec(2, 2, width_ratios=[1., 1.25], hspace=.55, wspace=.28)
    # (a) 26966 CTE along the turn, median and IQR over the 10 perturbations.
    ax = fig.add_subplot(grid[0, 0])
    w = next(x for x in windows if x['name'] == '26966')
    ax.axvspan(*w['core_m'], color='#DDDDDD', alpha=.5, lw=0)
    for arm, label, color, style in ARMS:
        rows = np.array(sorted(profile[('26966', arm)]))
        ax.plot(rows[:, 0], rows[:, 1], color=color, ls=style, label=label)
        if style == '-' and arm.endswith('kfix'):
            ax.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], color=color, alpha=.18, lw=0)
    plot_style.zero_line(ax)
    ax.set_xlim(w['window_m'][0] - 8, w['window_m'][1] + 10)
    ax.set_xlabel('Station along route 26966 (m)')
    ax.set_ylabel('CTE (m, left +)')
    plot_style.panel(ax, '(a) Sharp right 26966, median and IQR')
    # (b) Per-window median CTE RMS of the four sensor arms and two ceilings.
    ax = fig.add_subplot(grid[0, 1])
    names = [x['name'] for x in windows]
    x = np.arange(len(names))
    for k, (arm, label, color, style) in enumerate(ARMS):
        med = np.array([summary['per_arm'][n + '|' + arm]['cte_rms']['median'] for n in names], float)
        marker = 'o' if style == '-' else 'x'
        ax.plot(x + (k - 2.5) * .11, med, marker, color=color, ms=3.2, mfc=color if 'kfix' in arm or 'truth' in arm else 'white',
                label=label)
    ax.axvline(3.5, color='#999999', lw=.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right')
    ax.set_ylabel('Window CTE RMS (m)')
    ax.text(1.5, 1.02, 'development', transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
    ax.text(7.5, 1.02, 'held-out', transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
    plot_style.panel(ax, '(b)')
    ax.legend(loc='upper right', ncol=2, fontsize=6.3, handlelength=1.2, columnspacing=.8)
    plot_style.bars(ax)
    # (c) Paired candidate - production difference with the registered guard.
    ax = fig.add_subplot(grid[1, 1])
    for k, (metric, limit, color) in enumerate([('cte_rms', .03, P['blue'])]):
        stats = [summary['paired']['%s|slipack-kfix-vs-prod-kfix' % n][metric] for n in names]
        mean = np.array([s['mean'] for s in stats], float)
        lo = np.array([s['ci'][0] for s in stats], float)
        hi = np.array([s['ci'][1] for s in stats], float)
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt='o', color=color, ms=3.2, capsize=1.8, lw=.8,
                    label='Slip + Ackermann $-$ production (fixed k)')
        ax.axhline(limit, color=P['vermillion'], ls='--', lw=.7, label='Guard +0.03 m')
    plot_style.zero_line(ax)
    ax.axvline(3.5, color='#999999', lw=.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right')
    ax.set_ylabel(r'$\Delta$ window CTE RMS (m)')
    plot_style.panel(ax, '(c) Paired difference, mean and 95% CI')
    ax.legend(loc='lower right', fontsize=6.5)
    plot_style.bars(ax)
    # (d) Heading on 26966: body vs course, and the rear sideslip that separates them.
    ax = fig.add_subplot(grid[1, 0])
    metrics = [('body_p95_deg', 'Body'), ('course_p95_deg', 'Course'), ('beta_p95_deg', r'$\beta$ (truth)'),
               ('body_minus_beta_hat_p95_deg', r'Body $-\hat\beta$')]
    width = .19
    for k, (arm, label, color, style) in enumerate([a for a in ARMS if a[0] in ('prod-kfix', 'slipack-kfix', 'slipack-truth')]):
        vals = [summary['per_arm']['26966|' + arm][m]['median'] for m, _ in metrics]
        ax.bar(np.arange(len(metrics)) + (k - 1) * width, vals, width, color=color,
               alpha=1. if style == '-' else .45, label=label)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([t for _, t in metrics])
    ax.set_ylabel('Window |angle| P95 (deg)')
    plot_style.panel(ax, '(d) Heading metrics on 26966')
    ax.legend(loc='upper right', fontsize=6.3)
    plot_style.bars(ax)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=.075, right=.99, top=.93, bottom=.13)
    print(plot_style.save(fig, args.out))


if __name__ == '__main__':
    main()
