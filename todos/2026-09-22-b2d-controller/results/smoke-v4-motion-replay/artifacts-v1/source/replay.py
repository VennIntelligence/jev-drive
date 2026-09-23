#!/usr/bin/env python3
"""Replay captured sensor bytes through frozen/new adapters; no CARLA or git."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def nonfinite(value, prefix=''):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(nonfinite(child, prefix + '.' + key if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(nonfinite(child, prefix + '[%d]' % index))
    elif isinstance(value, (float, int)) or value in ('nan', 'inf', '-inf'):
        if not np.isfinite(float(value)):
            result.append(dict(field=prefix, encoded_value=value))
    return result


def stopped_inventory(root):
    events = lines(root / 'events.jsonl')
    if not events or events[-1].get('kind') != 'end':
        raise ValueError('campaign has no final end event: ' + str(root))
    pids = []
    for path in sorted(root.rglob('*.pid')):
        pid = int(path.read_text().strip())
        if Path('/proc/%d' % pid).exists():
            raise ValueError('recorded PID still exists; refuse live inventory: %d' % pid)
        pids.append(dict(path=str(path), pid=pid, proc_exists=False))
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('source changed during inventory: ' + str(path))
            files.append(dict(path=str(path), bytes=after.st_size, sha256=digest))
    return dict(root=str(root), captured_at_unix=time.time(), last_event=events[-1],
                stopped_evidence=pids, files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--failed', type=Path, required=True)
    parser.add_argument('--adapter-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output exists; use a new versioned directory')
    inventories = [stopped_inventory(root.resolve()) for root in (args.failed, args.capture)]
    args.out.mkdir(parents=True)
    source_dir = args.out / 'source'; source_dir.mkdir()
    input_dir = args.out / 'inputs'; input_dir.mkdir()
    root = args.capture.resolve()
    attempt = root / 'pursuit-seed0/attempts/2390/1'
    snapshots = []

    def copy(source, dest):
        before = sha(source); shutil.copyfile(str(source), str(dest))
        if sha(dest) != before or sha(source) != before:
            raise ValueError('source changed while snapshotting: ' + str(source))
        snapshots.append(dict(original=str(source.resolve()), archived=str(dest.relative_to(args.out)), sha256=before))
        return dest

    copy(Path(__file__).resolve(), source_dir / 'replay.py')
    frozen_dir = root / 'provenance/source/scripts'
    frozen = load(copy(frozen_dir / 'b2d_controller_adapter.py', source_dir / 'frozen_adapter.py'), 'frozen_adapter')
    candidate = load(copy(args.adapter_source.resolve(), source_dir / 'candidate_adapter.py'), 'candidate_adapter')
    controller = load(copy(frozen_dir / 'b2d_controller.py', source_dir / 'frozen_controller.py'), 'frozen_controller')
    paths = {}
    for name in ('motion.jsonl', 'control.jsonl', 'trajectories.jsonl', 'route_reference.json', 'agent_config.json'):
        paths[name] = copy(attempt / name, input_dir / name)
    agent_config = json.loads(paths['agent_config.json'].read_text())
    config_path = copy(Path(agent_config['controller_config']), input_dir / 'controller_config.json')
    config = json.loads(config_path.read_text())
    reference = json.loads(paths['route_reference.json'].read_text())
    motion = lines(paths['motion.jsonl'])
    recorded = {r['frame']: r for r in lines(paths['control.jsonl'])}
    recorded_traj = {r['frame']: r for r in lines(paths['trajectories.jsonl'])}
    bad_rows = [dict(row=index, frame=row['frame'], sim_time=row['sim_time'], fields=nonfinite(row['sensors'], 'sensors'))
                for index, row in enumerate(motion) if nonfinite(row['sensors'], 'sensors')]
    if any(any(sensor['frame'] != row['frame'] for sensor in row['sensors'].values()) for row in motion):
        raise ValueError('sensor frame mismatch')
    if not np.all(np.diff([r['frame'] for r in motion]) == 1):
        raise ValueError('motion frame gap')
    comparison = {}
    for version, adapter in [('frozen_a1b50ed', frozen), ('candidate_compass_prediction', candidate)]:
        settings = reference['adapter']
        projector = adapter.GPSProjector(reference['gps_lat_lon'], reference['world_xy'])
        pose = adapter.PoseFilter(projector, reference['rear_axle_offset_m'], settings.get('gnss_x_m', -1.4),
                                  settings.get('pose_gnss_gain', .05), settings.get('pose_heading_gain', .1))
        route = adapter.RouteAdapter(reference['world_xy'], agent_config['cruise_mps'],
                                     settings.get('route_end_extension_m', 3.), settings.get('route_stop_deceleration', 2.))
        params = {key: value for key, value in config.items() if key not in ('metadata', 'preset', 'adapter', 'rear_axle_offset_m',
                  'gnss_x_m', 'pose_gnss_gain', 'pose_heading_gain', 'route_end_extension_m', 'truth_logging', 'route_stop_deceleration')}
        ctrl = controller.Controller(preset=agent_config['controller_preset'], **params)
        output, pose_errors, control_errors, trajectory_errors = [], [], [], []
        for index, row in enumerate(motion):
            sensors = row['sensors']; imu = [float(v) for v in sensors['IMU']['data']]
            gps = [float(v) for v in sensors['GPS']['data']]
            speed = adapter.controller_speed(float(sensors['SPEED']['data']['speed']))
            stamp = row['sim_time']; frame = row['frame']
            result = dict(frame=frame, sim_time=stamp, recorded_control_available=frame in recorded)
            try:
                xy, yaw = pose.update(gps, imu[6], speed, imu[5], stamp)
                route.project(xy, yaw, speed, stamp)
                if index % agent_config['decimate'] == 0:
                    trajectory = route.trajectory(xy, yaw)
                    accepted = ctrl.update(trajectory, stamp)
                    result.update(trajectory_xy=trajectory.tolist(), trajectory_accepted=accepted)
                    if frame in recorded_traj:
                        error = float(np.max(abs(trajectory - np.asarray(recorded_traj[frame]['trajectory_xy']))))
                        trajectory_errors.append(error); result['trajectory_max_abs_error_m'] = error
                control = ctrl.step(stamp, speed, -imu[5])
                result.update(status='finite_output', pose_xy=xy.tolist(), pose_yaw=yaw,
                              throttle=control[0], steer=control[1], brake=control[2],
                              controller_diagnostics=ctrl.diagnostics,
                              pose_diagnostics=getattr(pose, 'diagnostics', None))
                if not np.isfinite(np.r_[xy, yaw, control]).all():
                    raise AssertionError('nonfinite output')
                if frame in recorded:
                    truth = recorded[frame]  # Recorded estimated pose/control only; no actor truth read.
                    p_error = float(np.max(abs(np.r_[xy, yaw] - np.r_[truth['pose_xy'], truth['pose_yaw']])))
                    c_error = float(np.max(abs(np.asarray(control) - [truth[k] for k in ('throttle', 'steer', 'brake')])))
                    pose_errors.append(p_error); control_errors.append(c_error)
                    result.update(pose_max_abs_error=p_error, control_max_abs_error=c_error)
            except ValueError as exc:
                result.update(status='exception', error=str(exc), pose_diagnostics=getattr(pose, 'diagnostics', None))
                output.append(result)
                break
            output.append(result)
        with (args.out / (version + '.jsonl')).open('w') as fh:
            for result in output:
                fh.write(json.dumps(result, allow_nan=False) + '\n')
        comparison[version] = dict(rows=len(output), compared_pose_controls=len(pose_errors), compared_trajectories=len(trajectory_errors),
                                   max_pose_abs_error=max(pose_errors, default=None),
                                   max_control_abs_error=max(control_errors, default=None),
                                   max_trajectory_abs_error_m=max(trajectory_errors, default=None), final=output[-1])
    first, last = motion[0], motion[-1]
    last_finite_compass = next(row for row in reversed(motion) if np.isfinite(float(row['sensors']['IMU']['data'][6])))
    summary = dict(capture_rows=len(motion), recorded_control_rows=len(recorded), first_frame=first['frame'], last_frame=last['frame'],
                   first_sim_time=first['sim_time'], last_sim_time=last['sim_time'],
                   first_to_last_span_s=last['sim_time'] - first['sim_time'],
                   observed_compass_age_at_final_s=last['sim_time'] - last_finite_compass['sim_time'],
                   actual_dropout_duration='unknown: capture ends on first invalid sample; no recovery recorded',
                   bad_rows=bad_rows, comparison=comparison,
                   scope='Open-loop same-input replay; only adapter differs, frozen controller retained; no future plant/official completion claim')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.out / 'failed-run-inventories.json').write_text(json.dumps(inventories, indent=2) + '\n')
    outputs = [p for p in args.out.rglob('*') if p.is_file()]
    manifest = dict(command=sys.argv, python=sys.version, numpy=np.__version__, snapshots=snapshots,
                    outputs=[dict(path=str(p.relative_to(args.out)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(outputs)])
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
