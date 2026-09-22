"""Supplementary synthetic G1 for the one-parameter turning candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from b2d_controller import Controller
import b2d_controller_selftest as analytic
import b2d_controller_pi_selftest as longitudinal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-lookahead-time-s', type=float, default=.375)
    args = p.parse_args()
    Controller(max_lookahead_time_s=args.max_lookahead_time_s)
    if args.out.exists():
        p.error('Choose a fresh evidence directory')
    sink = analytic._TraceSuite(args.out)
    constructed = []
    def factory(*pos, **kwargs):
        assert pos == ('pursuit',)
        assert kwargs['longitudinal_mode'] == 'pi'
        assert kwargs['pi_kp'] == .5 and kwargs['pi_ki'] == .25
        assert kwargs.get('lookahead') in (None, 'max')
        kwargs.update(lookahead='max', max_lookahead_time_s=args.max_lookahead_time_s)
        c = Controller(*pos, **kwargs)
        assert c.max_lookahead_time_s == args.max_lookahead_time_s and c.lookahead == 'max'
        constructed.append(dict(preset=c.preset, lookahead=c.lookahead,
                                max_lookahead_time_s=c.max_lookahead_time_s,
                                longitudinal_mode=c.longitudinal_mode, kp=c.longitudinal_pi.kp,
                                ki=c.longitudinal_pi.ki, steer_rate=c.steer_rate, max_steer=c.max_steer))
        return c
    original_analytic, original_longitudinal = analytic.Controller, longitudinal.Controller
    analytic.Controller = longitudinal.Controller = factory
    gains = dict(longitudinal_mode='pi', pi_kp=.5, pi_ki=.25)
    cases = []
    try:
        for speed in (6., 8.):
            for sign in (-1., 1.):
                for delay in (0., .3):
                    cases.append(sink.wrap(analytic.circle)('pursuit', speed=speed, sign=sign,
                                                          delay=delay, lookahead='max', **gains))
        for offset in (-.5, .5):
            for heading in (-5., 5.):
                cases.append(sink.wrap(analytic.circle)('pursuit', speed=8., lateral_offset=offset,
                                                      heading_offset=np.deg2rad(heading), lookahead='max', **gains))
        for delay in (0., .3):
            cases.append(sink.wrap(analytic.s_curve)('pursuit', delay=delay, **gains))
        cases.append(sink.wrap(analytic.stop)('pursuit', **gains))
        for delay in (0., .1, .2):
            cases.append(sink.wrap(longitudinal.speed_steps)(actuator_delay=delay, pi_kp=.5, pi_ki=.25))
    finally:
        sink.events.close()
        analytic.Controller, longitudinal.Controller = original_analytic, original_longitudinal
    assert len(cases) == len(constructed) == 18
    source_dir = args.out / 'source'
    source_dir.mkdir()
    for name in ('b2d_controller.py', 'b2d_controller_selftest.py', 'b2d_controller_pi_selftest.py',
                 'b2d_controller_turn_selftest.py', 'test_b2d_controller_turns.py'):
        shutil.copyfile(str(Path(__file__).with_name(name)), str(source_dir / name))
    result = dict(configuration=constructed[0], constructed_controllers=constructed,
                  scope='new turning-development candidate; after prior G4, before new turning G2',
                  plant='synthetic bicycle and pedal dynamics, measured default dimensions; no CARLA tire/actuator calibration',
                  limitations=['Analytic native 20-point paths, not padded TCP outputs.',
                               'No neural planner, traffic interactions or default qualification.',
                               'At actual speeds <=6m/s both .5/.375 coefficients retain3m floor; S reference speed can slightly exceed6.'],
                  cases=cases, pass_count=sum(bool(x.get('main_pass', x.get('pass_accuracy', False))) for x in cases))
    result['pass'] = result['pass_count'] == len(cases)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    hashes = {str(x.relative_to(args.out)): hashlib.sha256(x.read_bytes()).hexdigest()
              for x in sorted(args.out.rglob('*')) if x.is_file()}
    (args.out / 'hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(pass_count=result['pass_count'], count=len(cases), passed=result['pass'],
                          failed=[x for x in cases if not x.get('main_pass', x.get('pass_accuracy', False))]), indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
