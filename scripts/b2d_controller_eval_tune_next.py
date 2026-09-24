"""Pre-registered round 2 of the dev L1 tuning: one-at-a-time moves around the round 1 winner
(accel_ki 0.05/0.2, jerk_limit_brake 4/16, brake_hysteresis 0/0.6), plus the winner itself and D.
With --freeze, copy the named round's winner config to controller-eval/P-final.json instead."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT / 'todos/2026-09-23-tfv6-controller/controller-eval/tune'
MOVES = [('accel_ki', .05), ('accel_ki', .2), ('jerk_limit_brake', 4.), ('jerk_limit_brake', 16.),
         ('brake_hysteresis', 0.), ('brake_hysteresis', .6)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from-round', required=True)
    p.add_argument('--to-round')
    p.add_argument('--freeze', action='store_true')
    a = p.parse_args()
    result = json.loads((TUNE / f'{a.from_round}-result.json').read_text())
    winner = result['winner']
    if winner is None:
        raise SystemExit(f'{a.from_round}: no eligible candidate')
    variants = json.loads((TUNE / f'{a.from_round}-variants.json').read_text())
    source = Path(variants[winner]['controller_config'].replace('/home/ujs/mycode/jev-drive-w2', str(ROOT)))
    config = json.loads(source.read_text())
    if a.freeze:
        target = TUNE.parent / 'P-final.json'
        target.write_text(json.dumps(config, indent=2) + '\n')
        print(f'froze {a.from_round}/{winner} -> {target}')
        return
    remote = '/home/ujs/mycode/jev-drive-w2/'
    out = {'W': {'controller_config': remote + str((TUNE / f'{a.to_round}-W.json').relative_to(ROOT)),
                 'presets': ['pursuit']},
           'D': variants['D']}
    (TUNE / f'{a.to_round}-W.json').write_text(json.dumps(config, indent=2) + '\n')
    for index, (key, value) in enumerate(MOVES, 1):
        if config.get(key) == value:
            continue
        label = f'M{index}'
        path = TUNE / f'{a.to_round}-{label}.json'
        path.write_text(json.dumps(dict(config, **{key: value}), indent=2) + '\n')
        out[label] = {'controller_config': remote + str(path.relative_to(ROOT)), 'presets': ['pursuit']}
    (TUNE / f'{a.to_round}-variants.json').write_text(json.dumps(out, indent=2) + '\n')
    # The scorer's candidate prefix is E; relabel so W and M* are eligible.
    print(','.join(out))


if __name__ == '__main__':
    main()
