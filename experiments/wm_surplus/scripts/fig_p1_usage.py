"""Paper figure: how many feature directions each grafted foundation model's planner needs (P1 usage probe, PD5/PD6/PD10).

Reads the committed readout CSVs; writes research/paper-outline/figs/p1_usage.{pdf,png}.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'research'))
import plot_style as S  # noqa: E402

R = REPO / 'experiments/wm_surplus/results'
OUT = REPO / 'research/paper-outline/figs/p1_usage'

C = {'pca': S.PALETTE['blue'], 'norm': S.PALETTE['orange'], 'wpca': S.PALETTE['green'],
     'L16': S.PALETTE['vermillion'], 'L31': S.PALETTE['purple'], 'rand': S.BASELINE}


def load(path, off):
    """Rows with k, retained % and its paired 95 % CI (from the EPDMS delta CI)."""
    d = pd.read_csv(path)
    d = d[d.k.notna()].copy()
    d['group'] = d.group.fillna('pca') if 'group' in d else 'pca'
    d['k'] = d.k.astype(float)
    for c in ('retained', 'lo', 'hi'):
        d[c] = 100 * (d.retained if c == 'retained' else (off + d[c]) / off)
    return d


def line(ax, d, g, label, ls='-', marker='o', lab_at=None, dy=0, lw=None):
    s = d[d.group == g].sort_values('k')
    ax.fill_between(s.k, s.lo, s.hi, color=C[g], alpha=.15, linewidth=0)
    ax.plot(s.k, s.retained, ls=ls, marker=marker, ms=3.2, color=C[g], label=label, lw=lw or 1.05,
            markeredgecolor='white', markeredgewidth=.4)


def frame(ax, width, title, note):
    ax.set_xscale('log', base=2)
    ax.axhline(95, color='#999999', lw=.6, ls=(0, (3, 2)), zorder=0)
    ax.axvline(width, color='#999999', lw=.5, zorder=0)
    ax.set_ylim(48, 103)
    ax.set_xlabel('directions kept at the read site, $k$')
    ax.set_title(title, loc='left', fontsize=8.5, pad=13)
    ax.text(0, 1.015, f'{note} · width {width}', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color='#555555')
    ax.legend(loc='lower right', fontsize=6.8, handlelength=1.8, borderaxespad=.3)


def main():
    S.apply()
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE_COLUMN_IN, 2.35), sharey=True,
                            gridspec_kw={'width_ratios': [1.15, 1.15, .9]})

    rc = load(R / 'p0_subjects/p1_recogdrive/variants/p1_readout.csv', 82.65)
    ax = axs[0]
    line(ax, rc, 'rand', 'random subspace', ls=(0, (2, 1.5)), marker=None, lab_at=128, dy=-6)
    line(ax, rc, 'pca', 'PCA of LM state', lab_at=128, dy=-8)
    line(ax, rc, 'norm', '+ norm restored', lab_at=32, dy=-7)
    line(ax, rc, 'wpca', "PCA of planner's read-in", lab_at=24, dy=9)
    frame(ax, 1536, '(a) ReCogDrive-2B (LLM, Qwen2.5)', 'navtest 12,146 · off 82.65')
    ax.set_ylabel('EPDMS kept (%)')

    w0 = load(R / 'p1_usage/DriveVLA-W0/controls/p1_readout.csv', 87.79)
    ax = axs[1]
    line(ax, w0, 'pca', 'all 32 layers cut', lab_at=512, dy=-9)
    line(ax, w0, 'norm', '+ norm restored', lab_at=512, dy=7)
    line(ax, w0, 'L16', 'one layer cut (16)', lw=3.2, marker='o')
    line(ax, w0, 'L31', 'one layer cut (31)', marker='s', ls=(0, (2, 1.2)))
    frame(ax, 4096, '(b) DriveVLA-W0 (Emu3 world model, 9B)', 'navtest subset 3,000 · off 87.79')
    ax.set_xlim(40, 5200)

    wj = load(R / 'p1_usage/results/p1_readout.csv', 91.71)
    ax = axs[2]
    line(ax, wj, 'pca', 'PCA of planner input', lab_at=16, dy=-9)
    frame(ax, 512, '(c) WA-JEPA (V-JEPA 2)', 'navtest · off 91.71 · navtrain PCA')

    for a in axs:
        a.set_yticks([50, 60, 70, 80, 90, 95, 100])
    fig.subplots_adjust(left=.07, right=.99, top=.85, bottom=.17, wspace=.1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    S.save(fig, OUT)


if __name__ == '__main__':
    main()
