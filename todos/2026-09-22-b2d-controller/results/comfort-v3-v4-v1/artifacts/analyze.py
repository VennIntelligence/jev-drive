#!/usr/bin/env python3
"""Recompute whole-episode physical diagnostics; NumPy + Matplotlib, no simulator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METRICS = ('a_long_mps2', 'a_right_mps2', 'j_long_mps3', 'j_right_mps3')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box5(values):
    # Centered offline sensitivity, five samples inside episode, truncated edges.
    return np.asarray([values[max(0, i - 2):min(len(values), i + 3)].mean(axis=0)
                       for i in range(len(values))])


def project(acceleration, forward, right, times):
    jerk = np.gradient(acceleration, times, axis=0, edge_order=1)
    return dict(a_long_mps2=np.einsum('ij,ij->i', acceleration, forward),
                a_right_mps2=np.einsum('ij,ij->i', acceleration, right),
                j_long_mps3=np.einsum('ij,ij->i', jerk, forward),
                j_right_mps3=np.einsum('ij,ij->i', jerk, right))


def stats(values):
    values = np.asarray(values)
    return dict(n=len(values), rms=float(np.sqrt(np.mean(values ** 2))),
                p50_abs=float(np.percentile(abs(values), 50)),
                p95_abs=float(np.percentile(abs(values), 95)),
                p99_abs=float(np.percentile(abs(values), 99)),
                max_abs=float(np.max(abs(values))),
                min_signed=float(np.min(values)), max_signed=float(np.max(values)))


def selftest():
    t = np.array([0., .04, .1, .15, .22, .27, .31])
    forward = np.tile([1., 0., 0.], (len(t), 1))
    right = np.tile([0., 1., 0.], (len(t), 1))
    a = np.column_stack((3 * t + 2, -2 * t, np.zeros(len(t))))
    got = project(a, forward, right, t)
    np.testing.assert_allclose(got['j_long_mps3'], 3.)
    np.testing.assert_allclose(got['j_right_mps3'], -2.)
    # A rotating body basis does not turn constant inertial acceleration into jerk.
    rotated = np.column_stack((np.cos(t), np.sin(t), np.zeros(len(t))))
    got = project(np.tile([2., 0., 0.], (len(t), 1)), rotated, right, t)
    np.testing.assert_allclose(got['j_long_mps3'], 0., atol=1e-12)
    np.testing.assert_allclose(box5(np.ones((7, 3))), 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3', type=Path, required=True)
    parser.add_argument('--v4', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selftest()
    if args.out.exists():
        parser.error('output already exists; preserve it and choose a new version')
    roots = {'v3': args.v3.resolve(), 'v4': args.v4.resolve()}
    traces = {version: {str(p.relative_to(root).parent): p for p in root.rglob('validation_trace.json')}
              for version, root in roots.items()}
    if set(traces['v3']) != set(traces['v4']) or len(traces['v3']) != 18:
        parser.error('expected exactly 18 identical matched case keys in both runs')
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, str(args.out / 'analyze.py'))
    sources, per_case, samples, data = [], [], [], {}
    for version, root in roots.items():
        summary = root / 'summary.json'
        sources.append(dict(path=str(summary), sha256=sha(summary), bytes=summary.stat().st_size))
        for key, path in sorted(traces[version].items()):
            rows = json.loads(path.read_text())
            validation_path = path.parent / 'validation.json'
            validation = json.loads(validation_path.read_text())
            for source in (path, validation_path):
                sources.append(dict(path=str(source), sha256=sha(source), bytes=source.stat().st_size))
            config = Path(validation['archived_controller_config'])
            config_data = json.loads(config.read_text())
            sources.append(dict(path=str(config), sha256=sha(config), bytes=config.stat().st_size))
            times = np.array([r['sim_time'] for r in rows])
            frames = np.array([r['frame'] for r in rows])
            arrays = {name: np.array([r['plant_kinematics'][name] for r in rows], dtype=float)
                      for name in ('acceleration_mps2', 'forward_vector', 'right_vector', 'angular_velocity_deg_s')}
            if not all(np.isfinite(x).all() for x in [times] + list(arrays.values())):
                raise ValueError('nonfinite kinematics: ' + str(path))
            if not (np.all(np.diff(frames) == 1) and np.allclose(np.diff(times), .05, atol=1e-6)):
                raise ValueError('unexpected frame/time gap: ' + str(path))
            forward, right = arrays['forward_vector'], arrays['right_vector']
            if not (np.allclose(np.linalg.norm(forward, axis=1), 1., atol=1e-5)
                    and np.allclose(np.linalg.norm(right, axis=1), 1., atol=1e-5)):
                raise ValueError('nonunit basis: ' + str(path))
            a = arrays['acceleration_mps2']
            derived = {mode: project(values, forward, right, times)
                       for mode, values in [('raw', a), ('box5', box5(a))]}
            for mode, metrics in derived.items():
                for metric, values in metrics.items():
                    result = dict(version=version, case=key, route_id=validation['route_id'],
                                  preset=validation['preset'], variant=validation['variant'],
                                  cruise_mps=validation['cruise_mps'], status=validation['status'],
                                  existing_gate_pass=validation['gate_pass'],
                                  pi_kp=config_data.get('pi_kp', 1.), pi_ki=config_data.get('pi_ki', .25),
                                  duration_s=float(times[-1] - times[0]), dt_min_s=float(np.diff(times).min()),
                                  dt_max_s=float(np.diff(times).max()), mode=mode, metric=metric, **stats(values))
                    per_case.append(result)
                    data[version, key, mode, metric] = values
            for i, row in enumerate(rows):
                sample = dict(version=version, case=key, tick=row['tick'], frame=row['frame'],
                              sim_time=times[i], elapsed_s=times[i] - times[0])
                for name, values in arrays.items():
                    for axis, value in zip('xyz', values[i]):
                        sample[name + '_' + axis] = value
                for mode, metrics in derived.items():
                    for metric, values in metrics.items():
                        sample[mode + '_' + metric] = values[i]
                sample['yaw_rate_rad_s'] = np.deg2rad(arrays['angular_velocity_deg_s'][i, 2])
                samples.append(sample)
    for name, rows in [('per-case.csv', per_case), ('samples.csv', samples)]:
        with (args.out / name).open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for mode in ('raw', 'box5'):
        for metric in METRICS:
            versions = {}
            for version in roots:
                selected = [r for r in per_case if r['version'] == version and r['mode'] == mode and r['metric'] == metric]
                versions[version] = dict(equal_case_mean_rms=float(np.mean([r['rms'] for r in selected])),
                                         equal_case_mean_p95_abs=float(np.mean([r['p95_abs'] for r in selected])),
                                         pooled_sample_stats=stats(np.concatenate([data[version, key, mode, metric] for key in sorted(traces[version])])))
            decreases = sum(float(stats(data['v4', key, mode, metric])['rms']) < float(stats(data['v3', key, mode, metric])['rms']) for key in traces['v3'])
            aggregate.append(dict(mode=mode, metric=metric, versions=versions, rms_decreased_cases=decreases,
                                  equal_case_mean_rms_change_pct=100 * (versions['v4']['equal_case_mean_rms'] / versions['v3']['equal_case_mean_rms'] - 1)))
    summary = dict(matched_cases=18, total_samples={v: sum(len(data[v, key, 'raw', METRICS[0]]) for key in traces[v]) for v in roots},
                   aggregation='Equal weight per case for mean RMS/p95; pooled statistics separately duration-weighted',
                   included='All recorded ticks, including startup and parking; all statuses and prior gate outcomes', aggregate=aggregate)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    labels = {'a_long_mps2': 'Longitudinal acceleration (m/s²)', 'a_right_mps2': 'Lateral acceleration (m/s²)',
              'j_long_mps3': 'Longitudinal jerk (m/s³)', 'j_right_mps3': 'Lateral jerk (m/s³)'}
    colors = {'v3': '#426B9C', 'v4': '#C66B32'}
    keys = sorted(traces['v3'])
    for mode in ('raw', 'box5'):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ax, metric in zip(axes.flat, METRICS):
            for version in roots:
                y = [stats(data[version, key, mode, metric])['rms'] for key in keys]
                ax.plot(np.arange(18), y, 'o-', color=colors[version], ms=4, lw=1.2,
                        label=version + (' (Kp 1.0)' if version == 'v3' else ' (Kp 0.5)'))
            ax.set_ylabel('Whole-episode RMS\n' + labels[metric])
            ax.set_xticks(np.arange(18)); ax.set_xticklabels([k.replace('/pi-max/pursuit', '/PP-max').replace('/pi/pursuit', '/PP-add').replace('/pi/carla', '/CARLA-lat') for k in keys], rotation=65, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=.25); ax.spines[['top', 'right']].set_visible(False)
        axes[0, 0].legend(frameon=False)
        fig.suptitle('Matched G2 physical comfort diagnostics — ' + ('unfiltered 20 Hz' if mode == 'raw' else 'five-sample smoothing sensitivity') + '\nRoute oracle / policy none · all startup and parking samples · no new acceptance gate')
        for extension in ('png', 'pdf'):
            fig.savefig(args.out / ('paired-rms-' + mode + '.' + extension), dpi=300)
        plt.close(fig)
    # ECDF gives each case equal total weight, despite differing run durations.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        for version in roots:
            values = np.concatenate([abs(data[version, key, 'raw', metric]) for key in keys])
            weights = np.concatenate([np.full(len(data[version, key, 'raw', metric]), 1. / (18 * len(data[version, key, 'raw', metric]))) for key in keys])
            order = np.argsort(values)
            ax.plot(values[order], np.cumsum(weights[order]), color=colors[version], label=version)
        ax.set_xscale('symlog', linthresh=.1); ax.set_ylim(0, 1)
        ax.set_xlabel('Absolute ' + labels[metric].lower()); ax.set_ylabel('Equal-case cumulative probability')
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Raw 20 Hz distributions · all 18 matched cases, including startup / parking\nRoute oracle / policy none · physical diagnostic, not official Smoothness or DS')
    for extension in ('png', 'pdf'):
        fig.savefig(args.out / ('distributions-raw.' + extension), dpi=300)
    plt.close(fig)
    files = [p for p in args.out.iterdir() if p.is_file()]
    manifest = dict(command=sys.argv, python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__,
                    inputs=list({r['path']: r for r in sources}.values()),
                    outputs=[dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)])
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
