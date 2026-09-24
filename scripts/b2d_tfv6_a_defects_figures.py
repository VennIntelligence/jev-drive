"""Figures for the read-only Task 9 audit of canonical TFv6 cases."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUT = Path('todos/2026-09-23-tfv6-controller/results/a-defects')
ROOT = Path('/data/runs/b2d/tfv6-w2/formal')


def frames(route, seed):
    paths = list(ROOT.glob(f'level[12]/cases/[12]/route-{route}/seed-{seed}/A/done.json'))
    assert len(paths) == 1, paths
    done = json.loads(paths[0].read_text())
    path = paths[0].parent / f"attempt-{done['attempt']}" / 'frames.jsonl'
    return [json.loads(line) for line in path.open()]


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    rows = list(csv.DictReader((OUT / 'case-metrics.csv').open()))
    metrics = ['moving_full_pedal_share', 'full_pedal_flips_per_min']
    labels = ['Moving ticks at full throttle or brake', 'Full-pedal reversals per sim minute']
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, metric, label in zip(axes, metrics, labels):
        values = [[float(r[metric]) for r in rows if r['arm'] == arm] for arm in 'ABCD']
        bp = ax.boxplot(values, tick_labels=list('ABCD'), patch_artist=True, showfliers=False)
        for box, color in zip(bp['boxes'], ['#d1495b', '#607c9f', '#70a67f', '#a77ab5']):
            box.set_facecolor(color)
            box.set_alpha(.65)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.25)
        ax.set_title(f'{label}\n48 paired canonical cases')
    fig.tight_layout()
    save(fig, 'pedal-saturation')

    data = frames('25381', '0')
    trace = [r for r in data if 4.5 <= r['sim_time'] <= 7.5 and r.get('target_speed') is not None]
    t = [r['sim_time'] for r in trace]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    ax1.plot(t, [r['target_speed'] for r in trace], color='#d1495b', lw=1.6, label='target speed')
    ax1.plot(t, [r['truth']['forward_speed_mps'] for r in trace], color='#304f70', lw=1.6, label='actual speed')
    ax1.set_ylabel('Speed (m/s)')
    ax1.legend(ncol=2, loc='upper right')
    ax2.step(t, [r['executed_control']['throttle'] for r in trace], where='post', label='throttle', color='#339966')
    ax2.step(t, [-r['executed_control']['brake'] for r in trace], where='post', label='negative brake', color='#d1495b')
    ax2.set_ylabel('Pedal command')
    ax2.set_xlabel('Simulation time (s)')
    ax2.legend(ncol=2, loc='upper right')
    for ax in (ax1, ax2): ax.grid(alpha=.25)
    fig.suptitle('A, route 25381 seed 0: DS 100 despite rapid stop-go commands')
    fig.tight_layout()
    save(fig, 'stop-go-trace')

    data = frames('17569', '1')
    trace = [r for r in data if r.get('target_speed') is not None]
    t = np.array([r['sim_time'] for r in trace])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(t, [r['target_speed'] for r in trace], color='#d1495b', lw=1, label='target speed')
    ax.plot(t, [r['truth']['forward_speed_mps'] for r in trace], color='#304f70', lw=1, label='actual speed')
    ax.set_xlabel('Simulation time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('A, route 17569 seed 1: long standstill with DS 100')
    ax.legend(ncol=2)
    ax.grid(alpha=.25)
    fig.tight_layout()
    save(fig, 'long-standstill')


if __name__ == '__main__':
    main()
