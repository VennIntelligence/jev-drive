"""Pin valid truth-pose CARLA trajectories as L1 expert references."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(a.runs.glob('*/pursuit/validation.json'))
    if len(paths) != a.expected:
        raise ValueError(f'Expected exactly {a.expected} privileged expert cases; found {len(paths)}')
    manifest = {}
    for path in paths:
        summary = json.loads(path.read_text())
        route = path.parent.parent.name
        if summary['route_id'] != route or summary['status'] != 'completed' or summary['collisions'] or not summary['gate_pass']:
            raise ValueError(f'Invalid privileged expert reference {path}')
        rows = json.loads((path.parent / 'validation_trace.json').read_text())
        if len(rows) != summary['ticks'] or [row['tick'] for row in rows] != list(range(len(rows))):
            raise ValueError(f'Invalid privileged expert trace {path}')
        elapsed = [float(row['elapsed_s']) for row in rows]
        if abs(elapsed[0]) > 1e-7 or any(b <= c for c,b in zip(elapsed, elapsed[1:])):
            raise ValueError(f'Invalid expert time grid {path}')
        payload = {'route': route, 'source': str(path.parent), 'elapsed_s': elapsed,
                   'world_xy': [row['truth_xy'] for row in rows],
                   'speed_mps': [row['speed'] for row in rows]}
        target = a.out / f'expert-{route}.json'
        target.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
        manifest[route] = str(target.resolve())
    (a.out / 'reference-traces.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (a.out / 'sha256.json').write_text(json.dumps({route: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                                for route,path in manifest.items()}, indent=2) + '\n')
    print(f'Pinned {len(manifest)} privileged expert references')


if __name__ == '__main__':
    main()
