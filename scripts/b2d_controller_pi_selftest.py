#!/usr/bin/env python
"""Fixed PI candidate synthetic G1 evidence; does not model CARLA gear changes."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from b2d_controller import Controller
from b2d_controller_selftest import _longitudinal, _TraceSuite, circle, stop, s_curve


def speed_steps(actuator_delay=0., trace=None):
    controller = Controller('pursuit', longitudinal_mode='pi')
    speed, position = 0., 0.
    queue, samples = deque(), []
    delay_ticks = int(round(actuator_delay / .05))
    for tick in range(920):
        now = tick * .05
        target = 6. if now < 12 else 8. if now < 24 else 0. if now < 34 else 6.
        if tick % 4 == 0:
            path = np.column_stack((target * np.arange(1, 21) * .25, np.zeros(20)))
            controller.update(path, now)
        commanded = controller.step(now, max(0., speed), 0.)
        queue.append(commanded)
        applied = queue.popleft() if len(queue) > delay_ticks else (0., 0., 0.)
        sample = {'tick': tick, 'sim_time': now, 'speed_mps': speed, 'position_m': position,
                  'reference_speed_mps': target, 'commanded_control': commanded,
                  'applied_control': applied, 'diagnostics': controller.diagnostics}
        samples.append(sample)
        if trace is not None:
            trace.tick(**sample)
        speed, distance = _longitudinal(speed, applied[0], applied[2], .05)
        position += distance
    steady = [row['speed_mps'] - row['reference_speed_mps'] for row in samples
              if 8 <= row['sim_time'] < 12 or 20 <= row['sim_time'] < 24 or row['sim_time'] >= 42]
    parked = [row for row in samples if 29 <= row['sim_time'] < 34]
    rms = float(np.sqrt(np.mean(np.square(steady))))
    drift = float(max(row['position_m'] for row in parked) - min(row['position_m'] for row in parked))
    peak_parked_speed = max(abs(row['speed_mps']) for row in parked)
    return {'kind': 'speed_steps', 'longitudinal_mode': 'pi', 'actuator_delay_s': actuator_delay,
            'steady_speed_rms_mps': rms, 'parked_displacement_m': drift,
            'parked_max_speed_mps': peak_parked_speed, 'ticks': len(samples),
            'integral_max_abs': max(abs(row['diagnostics']['longitudinal_integral_effort']) for row in samples),
            'main_pass': rms < .5 and drift <= .1 and peak_parked_speed < .1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output exists; use a new versioned path')
    sink = _TraceSuite(args.out)
    cases = []
    try:
        for preset in ('carla', 'tcp', 'pursuit'):
            cases.append(sink.wrap(stop)(preset, longitudinal_mode='pi'))
        for sign in (-1., 1.):
            for delay in (0., .1, .3):
                cases.append(sink.wrap(circle)('pursuit', sign=sign, delay=delay, longitudinal_mode='pi'))
        cases.append(sink.wrap(circle)('pursuit', lateral_offset=.5, heading_offset=np.deg2rad(5), longitudinal_mode='pi'))
        cases.append(sink.wrap(s_curve)('pursuit', longitudinal_mode='pi'))
        for delay in (0., .1, .2):
            cases.append(sink.wrap(speed_steps)(actuator_delay=delay))
    finally:
        sink.events.close()
    source_dir = args.out / 'source'; source_dir.mkdir()
    sources = []
    for name in ('b2d_controller.py', 'b2d_controller_selftest.py', 'b2d_controller_pi_selftest.py', 'test_b2d_controller_pi.py'):
        source = Path(__file__).with_name(name); destination = source_dir / name
        shutil.copyfile(str(source), str(destination))
        sources.append({'path': name, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    result = {'longitudinal_mode': 'pi', 'kp': 1., 'ki': .25,
              'plant': 'synthetic 3*throttle - .08*speed - 8*brake with static friction; no gears',
              'cases': cases, 'sources': sources,
              'pass': all(row.get('main_pass', row.get('pass_accuracy', False)) for row in cases)}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
