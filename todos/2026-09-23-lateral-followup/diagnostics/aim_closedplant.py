"""Paired existing synthetic closed-plant G1; changes only lateral aim mode."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
from b2d_controller import Controller
import b2d_controller_selftest as analytic
import b2d_controller_pi_selftest as pi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Fresh output required')
    sink = analytic._TraceSuite(args.out)
    config = json.loads((REPO / 'todos/2026-09-23-tcp-controller/turns/configs/baseline-max.json').read_text())
    cases, constructors = [], []
    gains = dict(longitudinal_mode='pi', pi_kp=.5, pi_ki=.25)
    original_a, original_p = analytic.Controller, pi.Controller
    mode = 'linear'

    def factory(*pos, **kwargs):
        assert pos == ('pursuit',) and kwargs['longitudinal_mode'] == 'pi'
        assert kwargs['pi_kp'] == .5 and kwargs['pi_ki'] == .25
        assert kwargs.get('lookahead') in (None, 'max')
        kwargs.update(lookahead='max', max_lookahead_time_s=.5, aim_interpolation=mode,
                      wheelbase=config['wheelbase'], max_steer_deg=config['max_steer_deg'],
                      steering_curve=config['steering_curve'])
        c = Controller(*pos, **kwargs)
        constructors.append(dict(mode=mode, arguments=kwargs))
        return c

    def run(function, *pos, **kwargs):
        result = sink.wrap(function)(*pos, **kwargs)
        result['aim_interpolation'] = mode
        cases.append(result)

    analytic.Controller = pi.Controller = factory
    try:
        for mode in ('linear', 'hermite'):
            for speed in (6., 8.):
                for sign in (-1., 1.):
                    for delay in (0., .3):
                        run(analytic.circle, 'pursuit', speed=speed, sign=sign, delay=delay, lookahead='max', **gains)
            for offset in (-.5, .5):
                for heading in (-5., 5.):
                    run(analytic.circle, 'pursuit', speed=8., lateral_offset=offset,
                        heading_offset=np.deg2rad(heading), lookahead='max', **gains)
            for delay in (0., .3):
                run(analytic.s_curve, 'pursuit', delay=delay, **gains)
            run(analytic.stop, 'pursuit', **gains)
            for delay in (0., .1, .2):
                run(pi.speed_steps, actuator_delay=delay, pi_kp=.5, pi_ki=.25)
    finally:
        sink.events.close()
        analytic.Controller, pi.Controller = original_a, original_p
    assert len(cases) == len(constructors) == 36
    paired = []
    for baseline, candidate in zip(cases[:18], cases[18:]):
        assert baseline['kind'] == candidate['kind']
        paired.append(dict(kind=baseline['kind'],
                           baseline=baseline, candidate=candidate,
                           metric_deltas={key: candidate[key] - value for key, value in baseline.items()
                                          if key.endswith(('_m', '_mps')) and isinstance(value, (int, float))}))
    source = args.out / 'source'; source.mkdir()
    paths = [REPO / 'scripts' / name for name in ('b2d_controller.py', 'b2d_controller_selftest.py',
             'b2d_controller_pi_selftest.py', 'test_b2d_controller_aim.py')]
    paths += [Path(__file__), REPO / 'todos/2026-09-23-tcp-controller/turns/configs/baseline-max.json']
    for path in paths:
        shutil.copyfile(str(path), str(source / path.name))
    passed = lambda row: bool(row.get('main_pass', row.get('pass_accuracy', False)))
    summary = dict(cases=cases, paired=paired, constructors=constructors,
                   groups={name:dict(passed=sum(passed(r) for r in cases if r['aim_interpolation']==name),total=18)
                           for name in ('linear','hermite')},
                   scope='Paired analytic closed-plant trajectories, synthetic tire/pedal dynamics; no CARLA or neural model; measured controller geometry, existing rounded analytic plant constants.',
                   frozen='pursuit max .5, PI .5/.25, same projection/time/longitudinal/steer limits; only aim interpolation changes')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    hashes = {str(path.relative_to(args.out)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.out.rglob('*')) if path.is_file()}
    (args.out / 'hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(summary['groups'], indent=2))
    return 0 if all(passed(r) for r in cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
