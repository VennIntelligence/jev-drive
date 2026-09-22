#!/usr/bin/env python
"""Reproducible publication figures for the B2D controller diagnostic experiments.

Dependencies: Python >=3.8, NumPy, Matplotlib. No simulator, pandas, or network.
All reported attempts, including failures and retries, are exported. Selected
attempts form the connected campaign series; other attempts remain visible as x.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'carla': '#0072B2', 'tcp': '#D55E00', 'pursuit': '#009E73'}
PRESETS = ('carla', 'tcp', 'pursuit')
LABEL = 'Diagnostic route oracle; policy = none'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def sort_id(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def metric(row, section, name='truth_cross_track_m', statistic='rms'):
    return ((row.get(section) or {}).get(name) or {}).get(statistic)


class Archive:
    def __init__(self, out, args):
        self.out = Path(out).resolve()
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError('Output directory is not empty; choose a new versioned --out to preserve earlier figures: %s' % self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'sources').mkdir(exist_ok=True)
        self.sources, self.figures, self.csvs, self.warnings = {}, [], [], []
        self.arguments = vars(args)

    def source(self, path, snapshot=False):
        path = Path(path).resolve()
        if str(path) not in self.sources:
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            entry = {'path': str(path), 'sha256': digest, 'bytes': len(content)}
            if snapshot:
                destination = self.out / 'sources' / (digest[:12] + '-' + path.name)
                destination.write_bytes(content)
                entry['snapshot'] = str(destination.relative_to(self.out))
            self.sources[str(path)] = entry
        return path

    def read(self, path):
        return json.loads(self.source(path, snapshot=True).read_text())

    def jsonl(self, path):
        path = self.source(path)
        rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                self.warnings.append('%s:%s invalid JSON retained in original source' % (path, number))
        return rows

    def csv(self, name, rows, fields):
        path = self.out / (name + '.csv')
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        self.csvs.append({'path': path.name, 'rows': len(rows), 'columns': fields,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return path.name

    def save(self, fig, name, data, note):
        fig.text(.01, .012, LABEL + ' | ' + note, fontsize=8, color='#444444')
        outputs = []
        for extension in ('png', 'pdf'):
            path = self.out / (name + '.' + extension)
            metadata = {'Creator': 'b2d_controller_plot.py', 'CreationDate': None,
                        'ModDate': None} if extension == 'pdf' else {'Software': 'b2d_controller_plot.py'}
            fig.savefig(path, dpi=300, bbox_inches='tight', metadata=metadata)
            outputs.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)
        self.figures.append({'name': name, 'outputs': outputs, 'data': data, 'note': note})

    def finish(self):
        self.source(Path(__file__), snapshot=True)
        manifest = {'schema_version': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                    'arguments': self.arguments, 'dependencies': {'numpy': np.__version__,
                    'matplotlib': matplotlib.__version__}, 'protocol': LABEL,
                    'inclusion': 'Every supplied G2 case and campaign attempt is exported. Failures are not excluded. '
                    'Campaign selected_attempt determines the connected series; retries remain x markers. '
                    'Missing metrics stay empty/NaN, never become zero. No confidence intervals from unreplicated runs.',
                    'sources': list(self.sources.values()), 'figures': self.figures,
                    'csv_data': self.csvs, 'warnings': self.warnings}
        (self.out / 'plot-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        readme = ['# Controller diagnostic figures', '', LABEL + '.', '',
                  'PNG: 300 dpi. PDF: vector output with embedded TrueType fonts.',
                  'Exact plotted observations are in adjacent CSV files; blank values mean unavailable.',
                  'The manifest records source paths, SHA256 hashes, snapshots of JSON summaries, versions, and warnings.',
                  'This output is immutable to the plot CLI: use a new versioned --out directory for subsequent campaigns.',
                  'Raw trajectories/telemetry remain at the paths in the manifest. All raw samples used by the trajectory and stop plots are also exported to CSV.', '',
                  'Run:', '', '```bash', '/data/envs/carla/bin/python scripts/b2d_controller_plot.py ' +
                  '--development ' + str(self.arguments['development']) + ' --campaign ' + str(self.arguments['campaign']) +
                  ' --out ' + str(self.out) + ' --route-id ' + self.arguments['route_id'], '```', '']
        (self.out / 'README.md').write_text('\n'.join(readme))
        print(json.dumps({'out': str(self.out), 'figures': len(self.figures), 'csvs': len(self.csvs),
                          'warnings': self.warnings}, indent=2))


def g2_plot(archive, development):
    source = development / 'summary.json'
    cases = archive.read(source)
    rows = []
    for case in cases:
        rows.append({'route_id': case['route_id'], 'preset': case['preset'], 'town': case.get('town'),
                     'status': case.get('status'), 'gate_pass': case.get('gate_pass'),
                     'failed_gates': ';'.join(key for key, ok in case.get('gates', {}).items() if not ok),
                     'cross_track_rms_m': case.get('cross_track_rms_m'),
                     'cross_track_p95_m': case.get('cross_track_p95_m'),
                     'pose_p90_m': case.get('pose_p90_m'), 'raw_pose_p90_m': case.get('raw_pose', {}).get('p90'),
                     'ticks': case.get('ticks'), 'source': str(source.resolve())})
    fields = list(rows[0]) if rows else ['route_id', 'preset']
    data = archive.csv('g2-route-metrics', rows, fields)
    routes = sorted({row['route_id'] for row in rows}, key=sort_id)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    x = np.arange(len(routes)); width = .24
    for index, preset in enumerate(PRESETS):
        selected = {row['route_id']: row for row in rows if row['preset'] == preset}
        for ax, field in zip(axes, ('cross_track_rms_m', 'pose_p90_m')):
            values = [selected.get(route, {}).get(field) for route in routes]
            values = [value if finite(value) else np.nan for value in values]
            bars = ax.bar(x + (index - 1) * width, values, width, color=COLORS[preset], label=preset.upper())
            for bar, route in zip(bars, routes):
                if not selected.get(route, {}).get('gate_pass', False):
                    bar.set_hatch('///'); bar.set_edgecolor('#222222'); bar.set_linewidth(.7)
    for ax in axes:
        ax.axhline(.5, color='#555555', ls='--', lw=1, label='Gate = 0.5 m')
        ax.set_xticks(x); ax.set_xticklabels(routes); ax.set_xlabel('Development route ID')
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Full-route truth cross-track RMS (m)')
    axes[1].set_ylabel('Position error, 90th percentile (m)')
    axes[0].set_title('Tracking, including failed cases'); axes[1].set_title('GNSS + IMU pose estimate')
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('G2: no-background controller validation', fontsize=13)
    fig.tight_layout(rect=(0, .06, 1, .93))
    archive.save(fig, 'g2-tracking-localization', [data], 'Hatching: any G2 gate failed; all %d cases shown' % len(rows))
    return cases


def campaign_plot(archive, campaign):
    reports = sorted(campaign.rglob('controller-report.json')) if campaign.is_dir() else [campaign]
    rows = []
    for report in reports:
        content = archive.read(report)
        for attempt in content.get('attempts', []):
            route = str(attempt.get('route_id', 'unknown'))
            number = attempt.get('attempt', 1)
            raw = report.parent / 'attempts' / route / str(number) / 'control.jsonl'
            if raw.exists():
                archive.source(raw)
            else:
                archive.warnings.append('Campaign telemetry unavailable: %s' % raw)
            rows.append({'route_id': route, 'preset': attempt.get('preset', 'unknown'),
                         'seed': attempt.get('seed'), 'attempt': number,
                         'selected_attempt': attempt.get('selected_attempt', True),
                         'official_status': attempt.get('official_status'), 'process_status': attempt.get('status'),
                         'completion_pct': attempt.get('completion'),
                         'full_cte_rms_m': metric(attempt, 'tracking'),
                         'precollision_cte_rms_m': metric(attempt, 'before_collision'),
                         'precollision_state': attempt.get('before_collision_state'),
                         'full_cte_samples': metric(attempt, 'tracking', statistic='n'),
                         'precollision_cte_samples': metric(attempt, 'before_collision', statistic='n'),
                         'driving_completed': attempt.get('driving_completed'), 'capped': attempt.get('capped'),
                         'report_source': str(report.resolve()), 'telemetry_source': str(raw.resolve())})
    fields = list(rows[0]) if rows else ['route_id', 'preset', 'completion_pct']
    data = archive.csv('dev10-all-attempts', rows, fields)
    if not rows:
        archive.warnings.append('No campaign attempts found; campaign figures not generated')
        return
    routes = sorted({row['route_id'] for row in rows}, key=sort_id)
    groups = sorted({(row['preset'], row['seed']) for row in rows}, key=lambda group: (str(group[0]), str(group[1])))
    seeds = sorted({row['seed'] for row in rows}, key=str)
    fig, axes = plt.subplots(3, 1, figsize=(12., 8.2), sharex=True)
    x = np.arange(len(routes)); lookup = {route: index for index, route in enumerate(routes)}
    for group_index, (preset, seed) in enumerate(groups):
        group_rows = [row for row in rows if (row['preset'], row['seed']) == (preset, seed)]
        selected = {row['route_id']: row for row in group_rows if row['selected_attempt']}
        if len(selected) != sum(bool(row['selected_attempt']) for row in group_rows):
            raise ValueError('Duplicate selected attempt for preset/seed/route; use a single campaign root')
        offset = (group_index - (len(groups) - 1) / 2) * min(.16, .7 / max(1, len(groups)))
        color = COLORS.get(preset, '#777777')
        seed_style = seeds.index(seed)
        for ax, field in zip(axes, ('completion_pct', 'full_cte_rms_m', 'precollision_cte_rms_m')):
            values = [selected.get(route, {}).get(field) for route in routes]
            ys = [value if finite(value) else np.nan for value in values]
            ax.plot(x + offset, ys, marker=('o', 's', '^', 'D')[seed_style % 4],
                    ls=('-', '--', ':', '-.')[seed_style % 4], lw=1.1, ms=4, color=color,
                    label='%s / seed %s' % (preset.upper(), seed))
            for row in group_rows:
                value = row.get(field)
                if finite(value) and not row['selected_attempt']:
                    ax.scatter(lookup[row['route_id']] + offset, value, marker='x', color=color, alpha=.6, s=35)
                if row['selected_attempt'] and not finite(value):
                    ax.annotate('NA', (lookup[row['route_id']] + offset, .025), xycoords=('data', 'axes fraction'),
                                fontsize=7, rotation=90, color=color)
            for route in routes:
                if route not in selected:
                    ax.annotate('NA', (lookup[route] + offset, .025), xycoords=('data', 'axes fraction'),
                                fontsize=7, rotation=90, color=color)
    axes[0].set_ylabel('Official completion (%)'); axes[0].set_ylim(-3, 105)
    axes[1].set_ylabel('Full-route truth CTE RMS (m)')
    axes[2].set_ylabel('Pre-collision truth CTE RMS (m)')
    axes[2].set_xticks(x); axes[2].set_xticklabels(routes, rotation=35, ha='right'); axes[2].set_xlabel('Dev10 route ID')
    for ax in axes:
        ax.grid(alpha=.2); ax.set_axisbelow(True)
    for ax in axes[1:]:
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, ncol=min(4, len(groups)))
    fig.suptitle('Dev10 diagnostic campaign: completion and route tracking', fontsize=13)
    fig.tight_layout(rect=(0, .055, 1, .96))
    archive.save(fig, 'dev10-completion-tracking', [data],
                 'All %d attempts; x = retry; NA = unavailable; pre-collision uses recorded event frames' % len(rows))


def trace_plots(archive, development, route_id, cases):
    trajectories, stops, references = [], [], []
    selected = [case for case in cases if str(case['route_id']) == route_id]
    if not selected:
        archive.warnings.append('Predeclared illustrative G2 route %s is absent' % route_id)
        return
    for case in selected:
        preset = case['preset']; directory = development / route_id / preset
        trace_path = directory / 'validation_trace.json'
        reference_path = directory / 'route_reference.json'
        control_path = directory / 'control.jsonl'
        missing = [str(path) for path in (trace_path, reference_path, control_path) if not path.exists()]
        if missing:
            archive.warnings.append('Illustrative trace sources missing: ' + ', '.join(missing))
            continue
        trace = archive.read(trace_path)
        reference = archive.read(reference_path)
        controls = {row['frame']: row for row in archive.jsonl(control_path)}
        for index, point in enumerate(reference['world_xy']):
            references.append({'preset': preset, 'point_index': index, 'world_x_m': point[0], 'world_y_m': point[1]})
        # Plot the entire time series. Mark true low-speed arrival, rather than
        # treating a noisy planner terminal flag as an independent stop metric.
        arrival = next((row['elapsed_s'] for row in trace
                        if row.get('endpoint_error_m', float('inf')) < 1 and abs(row['speed']) < .1), None)
        for row in trace:
            control = controls.get(row['frame'], {})
            xy = row['truth_xy']
            common = {'route_id': route_id, 'preset': preset, 'frame': row['frame'],
                      'elapsed_s': row['elapsed_s'], 'gate_pass': case.get('gate_pass'), 'status': case.get('status')}
            trajectories.append(dict(common, world_x_m=xy[0], world_y_m=xy[1],
                                     truth_cte_m=row.get('cross_track_m')))
            stops.append(dict(common, time_from_arrival_s=None if arrival is None else row['elapsed_s'] - arrival,
                              speed_mps=row['speed'], endpoint_error_m=row.get('endpoint_error_m'),
                              target_speed_mps=control.get('target_speed_mps'),
                              throttle=control.get('throttle'), brake=control.get('brake'),
                              terminal_hold=control.get('route_terminal_hold')))
    if not trajectories:
        return
    trace_data = archive.csv('g2-route-%s-trajectory' % route_id, trajectories, list(trajectories[0]))
    reference_data = archive.csv('g2-route-%s-reference' % route_id, references, list(references[0]))
    stop_data = archive.csv('g2-route-%s-stop' % route_id, stops, list(stops[0]))
    origin = np.array([references[0]['world_x_m'], references[0]['world_y_m']])
    fig, ax = plt.subplots(figsize=(6.8, 6.))
    first_preset = references[0]['preset']
    path = np.array([[row['world_x_m'], row['world_y_m']] for row in references if row['preset'] == first_preset])
    ax.plot(path[:, 0] - origin[0], -(path[:, 1] - origin[1]), '--', c='#222222', lw=1.8, label='Dense route reference')
    for preset in PRESETS:
        rows = [row for row in trajectories if row['preset'] == preset]
        if not rows:
            continue
        xy = np.array([[row['world_x_m'], row['world_y_m']] for row in rows])
        failed = not rows[0]['gate_pass']
        ax.plot(xy[:, 0] - origin[0], -(xy[:, 1] - origin[1]), color=COLORS[preset], lw=1.2,
                label=preset.upper() + (' (G2 failed)' if failed else ''))
        ax.scatter(xy[-1, 0] - origin[0], -(xy[-1, 1] - origin[1]), c=COLORS[preset], s=25, marker='s')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    ax.set_xlabel('East from reference origin (m)'); ax.set_ylabel('North from reference origin (m)')
    ax.set_title('G2 route %s: complete rear-axle trajectories' % route_id)
    ax.legend(fontsize=8); fig.tight_layout(rect=(0, .055, 1, 1))
    archive.save(fig, 'g2-route-%s-trajectory' % route_id, [trace_data, reference_data],
                 'Predeclared route; all presets and full traces; square = final rear axle')
    fig, axes = plt.subplots(3, 1, figsize=(9., 7.3), sharex=True)
    for preset in PRESETS:
        rows = [row for row in stops if row['preset'] == preset]
        if not rows:
            continue
        times = np.array([row['elapsed_s'] for row in rows])
        for ax, field in zip(axes, ('speed_mps', 'endpoint_error_m', 'throttle')):
            values = [row[field] if finite(row[field]) else np.nan for row in rows]
            ax.plot(times, values, color=COLORS[preset], lw=1.15, label=preset.upper())
        hold = [row['elapsed_s'] for row in rows if row['time_from_arrival_s'] is not None and row['time_from_arrival_s'] >= 0]
        if hold:
            axes[0].axvline(hold[0], color=COLORS[preset], alpha=.45, ls=':', lw=1)
    axes[0].set_ylabel('True speed (m/s)'); axes[1].set_ylabel('True endpoint distance (m)'); axes[2].set_ylabel('Throttle command')
    axes[2].set_xlabel('Elapsed simulation time (s)')
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_title('G2 route %s: braking and parking hold, full run' % route_id)
    for ax in axes:
        ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    axes[2].set_ylim(0, .85)
    fig.tight_layout(rect=(0, .055, 1, 1))
    archive.save(fig, 'g2-route-%s-stop-hold' % route_id, [stop_data],
                 'Dotted lines: first true speed <0.1 m/s within 1 m; full data, no time cropping')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True, help='Directory containing G2 summary.json and route/preset traces')
    parser.add_argument('--campaign', type=Path, required=True, help='Campaign root searched for controller-report.json, or one report file')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--route-id', default='26966', help='Predeclared illustrative route; default26966 includes all3presets')
    args = parser.parse_args()
    for key in ('development', 'campaign', 'out'):
        setattr(args, key, str(getattr(args, key).resolve()))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                         'axes.labelsize': 9, 'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.facecolor': 'white'})
    try:
        archive = Archive(args.out, args)
    except FileExistsError as exc:
        parser.error(str(exc))
    cases = g2_plot(archive, Path(args.development))
    campaign_plot(archive, Path(args.campaign))
    trace_plots(archive, Path(args.development), args.route_id, cases)
    archive.finish()


if __name__ == '__main__':
    main()
