#!/usr/bin/env python3
"""Regression checks for X's speed gate, unchanged geometry, and frozen P7 defaults."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.nq4_x import XState, check
from b2d_controller import Controller
from nq4_x_controller import XController


def run():
    route = np.c_[np.arange(120.), np.zeros(120)]
    left, right = route + [0, -3.5], route + [0, 3.5]
    traj = np.c_[np.arange(1, 21) * .5, np.sin(np.arange(1, 21) * .08)]
    params = json.loads((Path(__file__).resolve().parents[1] /
                        'todos/2026-09-23-tfv6-controller/controller-eval/P7.json').read_text())
    for key in ('rear_axle_offset_m', 'pose_lateral_coefficient_s2_per_m'):
        params.pop(key)
    records = []
    state = XState(route, left, right)
    reference = XState(route, left, right)
    for i, (mode, speed) in enumerate([(1, 0.), (4, .999999), (1, 1.), (4, 1.000001),
                                     (2, 2.), (1, .5), (4, 2.), (0, 2.), (3, 2.)]):
        pose = [float(i), 0., 0.]
        path, info = state.step(mode, traj, pose[:2], 0., speed, 8.)
        expected, _ = reference.step(0 if mode in (1, 4) else mode, traj, pose[:2], 0.)
        assert np.array_equal(path, expected), 'stop gate changed head/shift geometry'
        target = (8. if speed < 1. else 0.) if mode in (1, 4) else None
        assert info['target_speed_mps'] == target
        assert info['raw_mode'] == mode
        records.append(dict(mode=mode, traj=traj.tolist(), pose=pose, path=path.tolist(), x=info,
                            speed_mps=speed, cruise_mps=8.))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.savez(d / 'x_route.npz', route=route, left=left, right=right)
        (d / 'x_dump.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        assert check(d) == dict(plans=9, differing=0, max_abs_diff_m=0.)
    # Defaults must be bit-identical to the frozen controller across changes of speed/plan.
    base, x = Controller(preset='pursuit', **params), XController(preset='pursuit', **params)
    for tick in range(100):
        t = tick * .05
        if tick % 4 == 0:
            assert base.update(traj, t) == x.update(traj, t)
        a, b = base.step(t, tick * .02, .01), x.step(t, tick * .02, .01)
        assert a == b
        diag = x.diagnostics.copy(); diag.pop('x_target_speed_mps')
        assert diag == base.diagnostics
    # Independent targets reach P7 feedback/feedforward without rescaling coordinates.
    for target, speed in [(8., 0.), (0., 4.)]:
        x = XController(preset='pursuit', **params)
        x.update(traj, 0., target_speed_mps=target)
        throttle, steer, brake = x.step(0., speed, 0.)
        assert x.diagnostics['target_speed_mps'] == target
        assert x.diagnostics['accel_feedforward_mps2'] == 0.
        assert np.array_equal(x._points[1:], traj)
        assert (throttle > 0 and brake == 0) if target else (throttle == 0 and brake > 0)
        # Duplicate update cannot alter the accepted longitudinal target.
        assert not x.update(traj, 0., target_speed_mps=3.)
        x.step(.05, speed, 0.)
        assert x.diagnostics['target_speed_mps'] == target
        x.update(traj, .1)
        x.step(.1, speed, 0.)
        assert x.diagnostics['x_target_speed_mps'] is None
    x = XController(preset='pursuit', **params)
    x.update(np.zeros((20, 2)), 0., target_speed_mps=8.)
    assert x.step(0., 0., 0.)[0] == 0
    assert x.diagnostics['reason'] == 'stationary_trajectory'
    return dict(pass_=True, gate_and_geometry_cases=9, frozen_default_ticks=100,
                target_cases=2, replay_bitwise=True, no_geometry_safe=True)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
