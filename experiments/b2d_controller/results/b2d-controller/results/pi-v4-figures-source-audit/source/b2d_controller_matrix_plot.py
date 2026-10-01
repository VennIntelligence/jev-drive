#!/usr/bin/env python
"""Frozen five-route development comparison; no simulator or live partial claims.

Requires all 20 declared v2 cases plus the 15 fixed v1 cases. Full-route gates
remain authoritative; moving CTE is a supplementary, explicitly masked metric.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROUTES = ('1773', '17563', '24240', '26966', '25854')
PRESETS = ('carla', 'tcp', 'pursuit')
LABELS = {'v1': 'v1 adapter / preset default', 'v2': 'v2 adapter / preset default',
          'v2max': 'v2 adapter / max'}
COLORS = {'v1': '#777777', 'v2': '#0072B2', 'v2max': '#D55E00'}
STYLES = {'v1': '--', 'v2': '-', 'v2max': ':'}


def stats(values):
    a = np.asarray([v for v in values if isinstance(v, (float, int)) and math.isfinite(v)], dtype=float)
    if not len(a):
        return dict(count=0, rms=None, median=None, p95=None, max=None)
    return dict(count=len(a), rms=float(np.sqrt(np.mean(a*a))),
                median=float(np.median(np.abs(a))), p95=float(np.percentile(np.abs(a), 95)),
                max=float(np.max(np.abs(a))))


def require_complete(root):
    """Never select successful cases or render a currently partial matrix."""
    summary = json.loads((root / 'summary.json').read_text())
    expected = {(route, variant, preset) for route in ROUTES
                for variant, preset in [('baseline', 'carla'), ('baseline', 'tcp'),
                                        ('baseline', 'pursuit'), ('max', 'pursuit')]}
    keys = [(str(s['route_id']), s['variant'], s['preset']) for s in summary]
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line.strip()]
    ends = [event for event in events if event.get('kind') == 'end']
    if (len(keys) != 20 or set(keys) != expected or not ends
            or ends[-1].get('status') != 'completed' or ends[-1].get('cases') != 20):
        raise RuntimeError('Matrix is not complete: require exactly 20 fixed cases and completed end event')
    return summary


class Sources:
    def __init__(self):
        self.files = {}

    def read(self, path, jsonl=False):
        path = Path(path).resolve()
        content = path.read_bytes()
        self.files[str(path)] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        if jsonl:
            return [json.loads(line) for line in content.splitlines() if line.strip()]
        return json.loads(content)

    def register(self, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        self.files[str(path)] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))


def extract_case(path, version, route, preset, sources):
    summary = sources.read(path / 'validation.json')
    trace = sources.read(path / 'validation_trace.json')
    control = sources.read(path / 'control.jsonl', jsonl=True)
    plans = sources.read(path / 'trajectories.jsonl', jsonl=True)
    reference = sources.read(path / 'route_reference.json')
    config = sources.read(path / 'agent_config.json')
    if str(summary['route_id']) != route or summary['preset'] != preset:
        raise ValueError('Case identity mismatch: %s' % path)
    recomputed = stats(r['cross_track_m'] for r in trace)
    for metric, key in [('rms', 'cross_track_rms_m'), ('p95', 'cross_track_p95_m')]:
        reported = summary.get(key)
        if reported is not None and (recomputed[metric] is None or abs(recomputed[metric] - reported) > 1e-9):
            raise ValueError('Stored metric does not reproduce: %s %s' % (path, key))
    moving = [r for r in trace if abs(r['signed_speed']) >= .5]
    timing = stats(r.get('controller_step_ms') for r in control)
    reasons = Counter(r.get('reason', 'missing') for r in control)
    rejoin = [r['route_rejoin'] for r in plans if isinstance(r.get('route_rejoin'), dict)]
    concerns = [r for r in rejoin if r.get('curvature_bound_satisfied') is False]
    first_speeds = stats(float(np.linalg.norm(np.asarray(r['trajectory_xy'])[0])) / .25 for r in plans)
    row = dict(version=version, route_id=route, preset=preset, cruise_mps=config['cruise_mps'],
               status=summary['status'], gate_pass=summary['gate_pass'], ticks=summary['ticks'],
               source_dir=str(path.resolve()), full_cte_rms_m=recomputed['rms'],
               full_cte_p95_m=recomputed['p95'], moving_ticks=len(moving),
               moving_cte_rms_m=stats(r['cross_track_m'] for r in moving)['rms'],
               moving_cte_p95_m=stats(r['cross_track_m'] for r in moving)['p95'],
               cruise_speed_rms_mps=summary.get('speed_rms_mps'),
               all_cruise_speed_rms_mps=summary.get('cruise_speed_all', {}).get('rms'),
               endpoint_error_m=summary.get('endpoint_error_m'),
               stop_hold_s=summary.get('stop_hold_s'), stop_hold_displacement_m=summary.get('stop_hold_displacement_m'),
               stop_hold_max_speed_mps=summary.get('stop_hold_max_speed_mps'),
               controller_step_count=timing['count'], controller_step_median_ms=timing['median'],
               controller_step_p95_ms=timing['p95'], controller_step_max_ms=timing['max'],
               plan_count=len(plans), rejoin_plan_count=len(rejoin), rejoin_concern_plans=len(concerns),
               rejoin_concern_fraction=len(concerns)/len(rejoin) if rejoin else None,
               rejoin_max_curvature_inv_m=stats(r.get('max_rejoin_curvature_inv_m') for r in rejoin)['max'],
               first_point_speed_p95_mps=first_speeds['p95'], first_point_speed_max_mps=first_speeds['max'],
               controller_reasons=json.dumps(dict(reasons), sort_keys=True),
               rejoin_reasons=json.dumps(dict(Counter(r.get('reason', 'missing') for r in rejoin)), sort_keys=True),
               invalid_motion_ticks=reasons['invalid_motion'], trajectory_behind_ticks=reasons['trajectory_behind'],
               pose_error_max_m=stats(r.get('pose_error_m') for r in control)['max'],
               failed_gates=','.join(k for k, v in summary['gates'].items() if not v),
               validation_sha256=sources.files[str((path/'validation.json').resolve())]['sha256'])
    row.update({'gate_' + k: v for k, v in summary['gates'].items()})
    return dict(row=row, summary=summary, trace=trace, reference=reference)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def comparisons(cases):
    lookup = {(c['row']['version'], c['row']['route_id'], c['row']['preset']): c['row'] for c in cases}
    rows = []
    metrics = ('full_cte_rms_m', 'full_cte_p95_m', 'moving_cte_rms_m', 'cruise_speed_rms_mps',
               'endpoint_error_m', 'controller_step_median_ms', 'controller_step_p95_ms')
    for route in ROUTES:
        for preset in PRESETS:
            for before, after, comparison in [('v1', 'v2', 'adapter_revision')]+(
                    [('v2', 'v2max', 'matched_lookahead_ablation')] if preset == 'pursuit' else []):
                a, b = lookup[before, route, preset], lookup[after, route, preset]
                if a['cruise_mps'] != b['cruise_mps']:
                    raise ValueError('Unmatched cruise comparison')
                row = dict(comparison=comparison, route_id=route, preset=preset, before=before, after=after,
                           cruise_mps=a['cruise_mps'], before_pass=a['gate_pass'], after_pass=b['gate_pass'],
                           before_status=a['status'], after_status=b['status'],
                           before_failed_gates=a['failed_gates'], after_failed_gates=b['failed_gates'])
                for metric in metrics:
                    row[metric + '_before'], row[metric + '_after'] = a[metric], b[metric]
                    row[metric + '_delta_after_minus_before'] = (b[metric]-a[metric]
                                                               if a[metric] is not None and b[metric] is not None else None)
                rows.append(row)
    return rows


def figures(cases, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    lookup = {(c['row']['version'], c['row']['route_id'], c['row']['preset']): c for c in cases}
    outputs = []
    def save(fig, name, note):
        fig.text(.01, .01, 'Privileged route diagnostic; no scenarios. ' + note, fontsize=8)
        for extension in ('png', 'pdf'):
            path = out / (name + '.' + extension)
            fig.savefig(path, dpi=300, bbox_inches='tight')
            outputs.append(path)
        plt.close(fig)
    legend = [Line2D([0], [0], color=COLORS[v], linestyle=STYLES[v], label=LABELS[v])
              for v in ('v1', 'v2', 'v2max')]
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for i, preset in enumerate(PRESETS):
        for j, route in enumerate(ROUTES):
            ax = axes[i, j]
            ref = np.asarray(lookup['v2', route, preset]['reference']['world_xy'])
            # Rigid transform only: origin at first reference, x along first segment;
            # display y left-positive, with equal metric scales and no path warping.
            origin = ref[0]
            forward = ref[1]-ref[0]
            forward /= np.linalg.norm(forward)
            basis = np.array([forward, [forward[1], -forward[0]]]).T
            q = (ref-origin) @ basis
            ax.plot(q[:, 0], q[:, 1], color='#222222', lw=2.4, alpha=.35)
            for version in ('v1', 'v2', 'v2max') if preset == 'pursuit' else ('v1', 'v2'):
                case = lookup[version, route, preset]
                truth = np.asarray([r['truth_xy'] for r in case['trace']])
                q = (truth-origin) @ basis
                ax.plot(q[:, 0], q[:, 1], STYLES[version], color=COLORS[version], lw=1.2)
                ax.scatter(q[-1, 0], q[-1, 1], marker='o' if case['row']['gate_pass'] else 'x',
                           color=COLORS[version], s=23, zorder=5)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=.18)
            ax.set_title('%s | %s | %g m/s' % (route, preset, lookup['v2', route, preset]['row']['cruise_mps']))
            ax.set_xlabel('Initial-forward displacement (m)')
            if j == 0:
                ax.set_ylabel('Initial-left displacement (m)')
    fig.legend(handles=legend+[Line2D([0], [0], color='#444444', label='Original reference')],
               loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, .035, 1, .965])
    save(fig, 'five-route-trajectories', 'All cases retained. Endpoint o = all gates pass; x = any gate fails.')

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    metrics = [('full_cte_rms_m', 'Full-route CTE RMS (m)', .5),
               ('moving_cte_rms_m', 'Moving CTE RMS (m)', None),
               ('cruise_speed_rms_mps', 'Cruise speed RMS error (m/s)', .5)]
    for i, preset in enumerate(PRESETS):
        for j, (metric, label, gate) in enumerate(metrics):
            ax = axes[i, j]
            versions = ('v1', 'v2', 'v2max') if preset == 'pursuit' else ('v1', 'v2')
            for k, version in enumerate(versions):
                x = np.arange(5)+(k-(len(versions)-1)/2)*.23
                y = [lookup[version, route, preset]['row'][metric] for route in ROUTES]
                ax.bar(x, [v if v is not None else np.nan for v in y], width=.21, color=COLORS[version], alpha=.85)
                for xx, yy, route in zip(x, y, ROUTES):
                    if yy is not None and not lookup[version, route, preset]['row']['gate_pass']:
                        ax.scatter(xx, yy, marker='x', color='black', s=22, zorder=4)
            if gate is not None:
                ax.axhline(gate, color='#999999', linestyle=':', lw=1)
            ax.set_xticks(np.arange(5), ROUTES)
            ax.set_title(preset)
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.18)
    fig.legend(handles=legend, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .04, 1, .965])
    save(fig, 'errors-speed-comparison', 'Moving mask: |true signed speed| >= 0.5 m/s; no startup trim. Cruise uses fixed 5 s allowance. x = any gate fails.')

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for j, route in enumerate(ROUTES):
        for version in ('v1', 'v2', 'v2max'):
            trace = lookup[version, route, 'pursuit']['trace']
            t = [r['elapsed_s'] for r in trace]
            axes[0, j].plot(t, [r['cross_track_m'] for r in trace], STYLES[version], color=COLORS[version], lw=1)
            axes[1, j].plot(t, [r['signed_speed'] for r in trace], STYLES[version], color=COLORS[version], lw=1)
        axes[0, j].set_title(route + ' | pursuit')
        axes[1, j].axhline(lookup['v2', route, 'pursuit']['row']['cruise_mps'], color='#777777', linestyle=':', lw=.7)
        for i in range(2):
            axes[i, j].grid(alpha=.18)
            axes[i, j].set_xlabel('Simulation elapsed time (s)')
        if j == 0:
            axes[0, j].set_ylabel('True CTE (m, left positive)')
            axes[1, j].set_ylabel('True signed speed (m/s)')
    fig.legend(handles=legend, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .04, 1, .955])
    save(fig, 'pursuit-error-speed-traces', 'Complete traces, including startup, stopping and failures; clocks start at each case\'s first controlled tick.')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2', type=Path, default=Path('/data/runs/b2d/controller/development-v2'))
    parser.add_argument('--v1', type=Path, default=Path('/data/runs/b2d/controller/development3'))
    parser.add_argument('--v1-s', type=Path, default=Path('/data/runs/b2d/controller/development-s-v1'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_complete(args.v2)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose a fresh output directory: ' + str(args.out))
    sources = Sources()
    sources.read(args.v2 / 'summary.json')
    sources.read(args.v2 / 'events.jsonl', jsonl=True)
    matrix = sources.read(args.v2 / 'inputs/matrix.json')
    source_routes = ET.parse(str(args.v2 / 'inputs/routes.xml')).getroot()
    if set(r.get('id') for r in source_routes.findall('route')) != set(ROUTES):
        raise ValueError('The matrix route set differs from the fixed five-route selection')
    for root in (args.v2, args.v1, args.v1_s):
        for path in list((root/'inputs').glob('*')) + [root/'provenance/manifest.json']:
            if path.is_file():
                sources.register(path)
    sources.register(Path(__file__))
    cases = []
    for route in ROUTES:
        for preset in PRESETS:
            cases.append(extract_case((args.v1_s if route == '17563' else args.v1)/route/preset,
                                      'v1', route, preset, sources))
            cases.append(extract_case(args.v2/route/'baseline'/preset, 'v2', route, preset, sources))
        cases.append(extract_case(args.v2/route/'max'/'pursuit', 'v2max', route, 'pursuit', sources))
    lookup = {(c['row']['version'], c['row']['route_id'], c['row']['preset']): c for c in cases}
    for route in ROUTES:
        reference = np.asarray(lookup['v1', route, 'pursuit']['reference']['world_xy'])
        for case in [c for c in cases if c['row']['route_id'] == route]:
            candidate = np.asarray(case['reference']['world_xy'])
            if reference.shape != candidate.shape or not np.allclose(reference, candidate, rtol=0., atol=1e-6):
                raise ValueError('Reference geometry is not matched: %s' % case['row']['source_dir'])
    matched = comparisons(cases)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'cases.csv', [c['row'] for c in cases])
    write_csv(args.out/'matched-comparisons.csv', matched)
    output_files = figures(cases, args.out)
    gate_totals = {version: dict(total=sum(c['row']['version'] == version for c in cases),
                                passed=sum(c['row']['version'] == version and c['row']['gate_pass'] for c in cases))
                   for version in ('v1', 'v2', 'v2max')}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=35, v2_required_cases=20,
                  selection=dict(routes=ROUTES, presets=PRESETS, cruise_mps={'default': 8, '17563': 6},
                                 selection='fixed before v2 results; all cases and failures retained',
                                 baseline='development3 plus scenario-cleared supplemental S v1',
                                 paired_comparisons=['v1 vs v2 base: adapter/path generation revision',
                                                     'v2 pursuit additive vs max: matched controller lookahead ablation']),
                  metric_rules=dict(authoritative_gates='unaltered stored validation.json gates',
                                    cte='independent true rear-axle projection; full RMS/p95 reproduced within 1e-9',
                                    moving_mask='absolute true signed speed >=0.5 m/s; no startup exclusion; supplementary only',
                                    cruise='stored independent truth cruise RMS after fixed 5s startup allowance',
                                    rejoin='count each plan once; False curvature_bound_satisfied is concern; absent in v1 is unassessed',
                                    reference_matching='all presets/versions original world_xy equal within 1e-6 m on each route',
                                    timing='controller_step_ms only, excludes adapter trajectory generation and simulator',
                                    coordinates='rigid route-origin transform, initial-forward x and initial-left y; equal axis scale'),
                  limits=['Diagnostic map routes with privileged references, not leaderboard scores.',
                          'One deterministic run per case; no uncertainty estimates or generalization claims.',
                          'Moving-mask sample populations can differ; failed short runs are not comparable exposure.',
                          'v1 to v2 changes path generation and is not a controller-only causal effect.'],
                  gate_totals=gate_totals, matched=matched, matrix=matrix,
                  sources=list(sources.files.values()), outputs=[])
    for path in [args.out/'cases.csv', args.out/'matched-comparisons.csv']+output_files:
        report['outputs'].append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.out/'README.md').write_text(
        '# Frozen five-route controller development comparison\n\n'
        'All 20 v2 cases and 15 fixed v1 cases are included, including failures. '
        'See cases.csv for all original gates and matched-comparisons.csv for paired differences.\n\n'
        'The v1/v2 comparison changes the adapter and generated path. The v2 pursuit additive/max '
        'comparison holds that adapter fixed. These are privileged route diagnostics, not leaderboard scores.\n\n'
        'Moving CTE uses |true signed speed| >= 0.5 m/s with no startup exclusion, and supplements '
        'the unchanged full-route gates. Timing covers controller.step only. '
        'report.json records exact raw paths, SHA-256 hashes, metric definitions and limitations.\n')
    print(json.dumps(dict(out=str(args.out.resolve()), gate_totals=gate_totals), indent=2))


if __name__ == '__main__':
    main()
