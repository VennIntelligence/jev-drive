#!/usr/bin/env python
"""Publication figures from preserved offline traces; no simulation or metric recomputation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--out', required=True, help='new figure directory; existing paths are rejected')
    args = parser.parse_args()
    source, out = Path(args.trace_dir).resolve(), Path(args.out).resolve()
    if out.exists():
        parser.error('refusing to overwrite figure directory: %s' % out)
    events = [json.loads(line) for line in (source / 'cases.jsonl').read_text().splitlines()]
    starts = {e['case_id']: e for e in events if e['event'] == 'start'}
    cases = []
    for event in events:
        if event['event'] != 'completed':
            continue
        params = starts[event['case_id']]['parameters']
        result = event['result']
        wanted_stop = result['kind'] == 'stop' and result['preset'] == 'pursuit'
        wanted_bearing = (result['kind'] == 'circle' and params.get('preset') == 'pursuit'
                          and params.get('lookahead') == 'max' and params.get('speed_window') == 'reference')
        if wanted_stop or wanted_bearing:
            path = source / (event['case_id'] + '.jsonl')
            if hashlib.sha256(path.read_bytes()).hexdigest() != event['trace_sha256']:
                raise ValueError('trace hash mismatch: %s' % path)
            cases.append((params, event, [json.loads(line) for line in path.read_text().splitlines()]))
    if len(cases) != 6:
        raise ValueError('expected exactly two pursuit stop and four bearing ablation traces, got %d' % len(cases))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    stops = sorted([case for case in cases if case[1]['result']['kind'] == 'stop'], key=lambda c: c[0]['speed_window'])
    for params, event, ticks in stops:
        near = params['speed_window'] == 'near'
        label = 'Near window [age, age+0.25]' if near else 'Future window [age+0.25, age+1.0]'
        color = '#0072B2' if near else '#D55E00'
        t = [row['sim_time'] for row in ticks]
        axes[0, 0].plot(t, [row['state_before'][3] for row in ticks], color=color, label=label)
        axes[0, 1].plot(t, [row['state_before'][0] - row['independent_reference_position_m'] for row in ticks],
                        color=color, label=label)
    reference = stops[0][2]
    axes[0, 0].plot([r['sim_time'] for r in reference], [r['independent_reference_speed_mps'] for r in reference],
                    color='black', linestyle='--', linewidth=1.2, label='Independent analytic speed')
    axes[0, 0].set(title='A  Stop speed: pursuit with two speed windows', xlabel='Simulation time (s)', ylabel='Speed (m/s)')
    axes[0, 0].legend(fontsize=8, loc='upper right')
    axes[0, 1].axhline(0, color='black', linewidth=.8)
    axes[0, 1].axhline(1, color='gray', linestyle=':', linewidth=.8)
    axes[0, 1].axhline(-1, color='gray', linestyle=':', linewidth=.8)
    axes[0, 1].set(title='B  Position error against analytic stop trajectory', xlabel='Simulation time (s)', ylabel='Longitudinal position error (m)')
    bearings = [case for case in cases if case[1]['result']['kind'] == 'circle']
    labels, rms, colors = [], [], []
    for params, event, ticks in bearings:
        summed = params['bearing_pid']
        left = params['sign'] > 0
        label = ('Pursuit + full bearing PID' if summed else 'Pursuit') + (' / left' if left else ' / right')
        color = '#D55E00' if summed else '#0072B2'
        axes[1, 0].plot([r['sim_time'] for r in ticks], [r['independent_lateral_error_m'] for r in ticks],
                       color=color, linestyle='-' if left else '--', linewidth=1.2, label=label)
        labels.append(('PP + PID' if summed else 'PP') + ('\nleft' if left else '\nright'))
        rms.append(event['result']['lateral_rms_m'])
        colors.append(color)
    axes[1, 0].axvspan(0, 2, color='gray', alpha=.12, label='Predeclared 2 s transient')
    axes[1, 0].set(title='C  Analytic circle error: full trace retained', xlabel='Simulation time (s)', ylabel='Signed radial error (m)')
    axes[1, 0].legend(fontsize=8, loc='best')
    bars = axes[1, 1].bar(range(len(rms)), rms, color=colors, width=.65)
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels)
    for bar, value in zip(bars, rms):
        axes[1, 1].annotate('%.3f' % value, (bar.get_x() + bar.get_width()/2, value),
                          ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    axes[1, 1].axhline(.2, color='gray', linestyle=':', linewidth=1, label='0.2 m synthetic accuracy gate')
    axes[1, 1].set(title='D  Circle RMS after predeclared 2 s transient', ylabel='Independent lateral RMS (m)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(0, max(.23, max(rms) * 1.2))
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Synthetic offline controller ablations — not CARLA driving results', fontsize=14)
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out / 'offline-ablations.png', dpi=250)
    fig.savefig(out / 'offline-ablations.pdf')
    plt.close(fig)
    provenance = {'trace_directory': str(source), 'trace_case_ids': [e['case_id'] for _, e, _ in cases],
                  'input_manifest_sha256': hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'circle_protocol': '6m/s, R20m, left/right, max lookahead, future speed window; only full bearing PID sum differs',
                  'stop_protocol': 'same pursuit plant, near vs future window; analytic8m/s then3s deceleration then hold',
                  'source': 'Complete previously recorded raw traces; summary RMS copied unchanged from case results.'}
    with (out/'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
    print(out)


if __name__ == '__main__':
    main()
