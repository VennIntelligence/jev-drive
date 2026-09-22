#!/usr/bin/env python
"""Supplementary exact frozen candidate coverage, collected after formal start."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = out / 'source'
    if source.exists() or (out / 'summary.json').exists():
        parser.error('existing evidence; choose fresh output directory')
    source.mkdir()
    repo = Path('/data/worktrees/jev-drive-controller-v2')
    frozen = Path('/data/runs/b2d/controller/formal-v4/provenance/source/scripts/b2d_controller.py')
    current = repo / 'scripts/b2d_controller.py'
    assert sha(frozen) == sha(current), 'runtime differs from formal freeze'
    inputs = [frozen, repo / 'scripts/b2d_controller_selftest.py',
              repo / 'scripts/b2d_controller_pi_selftest.py',
              repo / 'todos/2026-09-22-b2d-controller/results/v4-freeze/candidate-pursuit.json']
    provenance = []
    for path in inputs:
        dest = source / path.name
        shutil.copyfile(str(path), str(dest))
        provenance.append(dict(original=str(path), archived=str(dest.relative_to(out)), sha256=sha(dest)))
    shutil.copyfile(__file__, str(source / 'run.py'))
    sys.path.insert(0, str(source))
    import numpy as np
    import b2d_controller as runtime
    import b2d_controller_selftest as analytic
    import b2d_controller_pi_selftest as pi
    config = json.loads((source / 'candidate-pursuit.json').read_text())
    signature = inspect.signature(runtime.Controller)
    candidate = {key: value for key, value in config.items() if key in signature.parameters}
    candidate['preset'] = 'pursuit'
    assert candidate['lookahead'] == 'max'
    assert candidate['longitudinal_mode'] == 'pi'
    assert candidate['pi_kp'] == .5 and candidate['pi_ki'] == .25
    instances = []

    def factory(*args, **kwargs):
        requested = signature.bind_partial(*args, **kwargs).arguments
        for key, value in candidate.items():
            if key in requested and requested[key] is not None:
                assert requested[key] == value, (key, requested[key], value)
        requested.update(candidate)
        ctrl = runtime.Controller(**requested)
        assert ctrl.preset == 'pursuit' and ctrl.lookahead == 'max'
        assert ctrl.longitudinal_mode == 'pi' and ctrl.speed_window == 'near'
        assert ctrl.longitudinal_pi.kp == .5 and ctrl.longitudinal_pi.ki == .25
        assert ctrl.wheelbase == config['wheelbase']
        assert np.array_equal(ctrl.steering_curve, np.asarray(config['steering_curve']))
        resolved = signature.bind_partial(**requested)
        resolved.apply_defaults()
        instances.append(dict(resolved.arguments))
        return ctrl

    # Scope is this helper process only. No runtime/helper source is modified.
    analytic.Controller = factory
    pi.Controller = factory
    sink = analytic._TraceSuite(out / 'traces')
    cases = []
    gains = dict(longitudinal_mode='pi', pi_kp=.5, pi_ki=.25)
    try:
        for sign in (-1., 1.):
            for delay in (0., .1, .3):
                cases.append(sink.wrap(analytic.circle)('pursuit', sign=sign, delay=delay,
                                                       lookahead='max', **gains))
        for offset in (-.5, .5):
            for heading in (-5., 5.):
                cases.append(sink.wrap(analytic.circle)('pursuit', lateral_offset=offset,
                                                       heading_offset=np.deg2rad(heading),
                                                       lookahead='max', **gains))
        for delay in (0., .3):
            cases.append(sink.wrap(analytic.s_curve)('pursuit', delay=delay, **gains))
        cases.append(sink.wrap(analytic.stop)('pursuit', **gains))
        for delay in (0., .1, .2):
            cases.append(sink.wrap(pi.speed_steps)(actuator_delay=delay, pi_kp=.5, pi_ki=.25))
    finally:
        sink.events.close()
    assert len(cases) == len(instances) == 16
    for row in cases:
        row['effective_candidate'] = candidate
    result = dict(schema_version=1, collected_at_utc=datetime.now(timezone.utc).isoformat(),
                  evidence_stage='supplementary configuration coverage AFTER formal-v4 started; not retrospective pre-run evidence',
                  configuration=candidate, constructor_assertions_passed=len(instances),
                  constructed_controllers=instances, sources=provenance,
                  plant='Independent synthetic kinematic bicycle; 3*throttle - .08*speed - 8*brake with static friction; no gears or CARLA tire dynamics',
                  limitations=['Original helper analytic 20-point future paths are generated from analytic functions, not padded native TCP outputs.',
                               'Plant steering curve uses original helper rounded measured constants; controller uses exact archived candidate constants.',
                               'No CARLA, sensors, real TCP model, scenario interactions, or real slope validation.',
                               'This configuration supplement does not override failed formal G4 or qualify a default.'],
                  cases=cases, pass_count=sum(bool(row.get('main_pass', row.get('pass_accuracy', False))) for row in cases))
    result['pass'] = result['pass_count'] == len(cases)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    inventory = [{'path': str(path.relative_to(out)), 'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'hashes.json']
    (out / 'hashes.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(pass_count=result['pass_count'], cases=len(cases), passed=result['pass'],
                          summary_sha256=sha(out / 'summary.json'),
                          failures=[row for row in cases if not row.get('main_pass', row.get('pass_accuracy', False))]), indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
