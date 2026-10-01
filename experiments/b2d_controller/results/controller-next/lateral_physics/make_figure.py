"""Figure for lateral-physics.md: rear slip vs PhysX prediction, 26966 error split, closed-loop screen.

Style mirrors the lateral-followup plot_style.py (CVPR double column, STIX serif, Okabe-Ito).

    python make_figure.py --logs logs.json --screen closed_loop.json --out figs/lateral-physics
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OKABE = {'orange': '#E69F00', 'sky_blue': '#56B4E9', 'green': '#009E73', 'blue': '#0072B2',
         'vermillion': '#D55E00', 'purple': '#CC79A7', 'black': '#000000', 'grey': '#777777'}
G = 9.81


def style():
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'font.size': 8.5,
        'mathtext.fontset': 'stix', 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.5, 'axes.linewidth': .55, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#BBBBBB', 'grid.alpha': .28,
        'grid.linewidth': .4, 'lines.linewidth': 1.05, 'legend.frameon': False, 'pdf.fonttype': 42,
        'savefig.dpi': 300, 'savefig.facecolor': 'white'})


def arrays(fr):
    return {k: np.array([np.nan if v is None else v for v in fr[k]], float) for k in fr}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs', required=True)
    p.add_argument('--screen', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    logs, screen = json.load(open(args.logs)), json.load(open(args.screen))
    style()
    fig, axes = plt.subplots(1, 3, figsize=(6.875, 2.45), gridspec_kw=dict(width_ratios=[1, 1.25, 1.05]))

    ax = axes[0]
    for key, fr in logs['frames'].items():
        if not key.startswith('pose-g2-v1/'):
            continue
        a = arrays(fr)
        x, y = a['vx'] ** 2 * a['w'], -a['vy']
        fast = '17563' not in key
        ax.plot(x, y, 'o', ms=1.6, alpha=.55, mec='none', color=OKABE['blue'] if fast else OKABE['orange'],
                label=None)
    xs = np.linspace(-45, 75, 2)
    ax.plot(xs, .010659832 * xs, color=OKABE['black'], lw=.9, label='frozen fit $k$=.0107')
    ax.plot(xs, 1.125 / (14.074 * G) * xs, color=OKABE['vermillion'], lw=.9, ls='--', label='PhysX nominal, 8 m/s')
    ax.plot(xs, 1.125 / (11.0 * G) * xs, color=OKABE['green'], lw=.9, ls=':', label='$c$=11 fit, 8 m/s')
    ax.plot([], [], 'o', ms=3, color=OKABE['blue'], label='8 m/s frames')
    ax.plot([], [], 'o', ms=3, color=OKABE['orange'], label='6 m/s frames')
    ax.set_xlabel(r'$v^2\omega$ (m$^2$/s$^3$, right turn +)')
    ax.set_ylabel(r'rear-axle slide $-v_{y,r}$ (m/s)')
    ax.set_ylim(-.55, 1.5)
    ax.legend(loc='upper left', handlelength=1.6)
    ax.text(0, 1.03, '(a)', transform=ax.transAxes)

    ax = axes[1]
    a = arrays(logs['frames']['pose-g2-v1/26966/candidate-fixed-k/right-sharp'])
    o = np.argsort(a['s'])
    ax.axvspan(29, 46, color='#DDDDDD', alpha=.5, lw=0)
    ax.axhline(0, color='#999999', lw=.5)
    ax.plot(a['s'][o], a['cte'][o], color=OKABE['black'], label='truth CTE')
    ax.plot(a['s'][o], a['cte_est'][o], color=OKABE['blue'], label='controller-visible')
    ax.plot(a['s'][o], a['loc'][o], color=OKABE['orange'], label='localization')
    b = arrays(logs['frames']['pose-g2-v1/26966/baseline-zero/right-sharp'])
    ob = np.argsort(b['s'])
    ax.plot(b['s'][ob], b['cte'][ob], color=OKABE['grey'], lw=.8, ls='--', label='truth, $k$=0 arm')
    ax.set_xlim(20, 61)
    ax.set_ylim(-.3, 1.35)
    ax.set_yticks([-.25, 0, .25, .5, .75, 1.])
    ax.set_xlabel('station on 26966 (m); core shaded')
    ax.set_ylabel('lateral error (m, left/outside +)')
    ax.legend(loc='upper center', ncol=2, handlelength=1.4, columnspacing=.8)
    ax.text(0, 1.03, '(b)', transform=ax.transAxes)

    ax = axes[2]
    variants = [('production', 'production'), ('ackermann', 'Ackermann map'), ('slip_frame', 'slip frame'),
                ('ackermann+slip', 'both'), ('ackermann+slip+odometry', 'both + odometry')]
    y = np.arange(len(variants))[::-1]
    for dy, pose, color, label in ((.17, 'truth', OKABE['sky_blue'], 'truth pose'),
                                  (-.17, 'loc:candidate-fixed-k', OKABE['blue'], 'fixed-$k$ pose err.')):
        vals = [screen['runs']['26966|right-sharp|%s|%s' % (pose, v)]['window']['cte_rms'] for v, _ in variants]
        ax.barh(y + dy, vals, height=.32, color=color, label=label)
    ax.axvline(.429, color=OKABE['black'], lw=.8, ls='--', label='CARLA fixed-$k$')
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, .2, .4, .6])
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in variants])
    ax.set_xlabel('26966 window CTE RMS (m)', x=.3)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', bbox_to_anchor=(1.08, 0), handlelength=1.0)
    ax.text(0, 1.03, '(c)', transform=ax.transAxes)

    fig.tight_layout(pad=.4, w_pad=.8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + '.pdf')
    fig.savefig(str(out) + '.png', dpi=300)


if __name__ == '__main__':
    main()
