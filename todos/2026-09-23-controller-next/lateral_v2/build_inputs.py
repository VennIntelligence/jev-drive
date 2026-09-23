"""Build the frozen lateral v2 campaign inputs from the development routes and the held-out selection.

Writes inputs/: routes.xml (3 development + 8 held-out routes, grouped by town), windows.json,
routes-g1..g4.xml (one CARLA worker each), route-cruises.json,
perturbations.json (p00 nominal anchor + p01..p10), configs/<arm>.json and
variants.json (absolute paths for the box checkout given by --repo-root). Plain Python, deterministic.

    python3 build_inputs.py --repo-root /root/autodl-tmp/ujs/jev-drive   # the box checkout, resolved
"""
import argparse
import copy
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
POSE = REPO / 'todos/2026-09-23-lateral-followup/pose-followup'
K_FIXED = 0.010659832
C_MKZ = 11.0             # PhysX lateral stiffness per unit load, open-loop plant fit over 8 windows (lateral-physics.md)
TRACK_MKZ = 1.5929       # front wheel spacing, calibration-physics.json wheels[0..1] (159.29 cm)
# Development windows (stations on route_reference world_xy), frozen since turns-v1.
DEVELOPMENT = [dict(name='24240', route_id='24240', town='Town10HD', cls='left', cruise_mps=8., core_m=[23., 59.],
                    window_m=[18., 64.]),
               dict(name='26966', route_id='26966', town='Town05', cls='right', cruise_mps=8., core_m=[29., 46.],
                    window_m=[24., 51.], post_m=[51., 61.]),
               dict(name='17563-S1', route_id='17563', town='Town12', cls='S', cruise_mps=6., core_m=[32.5, 43.5],
                    window_m=[27.5, 48.5]),
               dict(name='17563-S2', route_id='17563', town='Town12', cls='S', cruise_mps=6., core_m=[78.5, 90.],
                    window_m=[73.5, 95.])]
ORDER = ['24240', '26966', '27506', '17563', '2084', '2881', '17569', '23695', '25358', '27494', '27515']
# One CARLA worker per group (maps load once per worker); roughly equal simulated time.
GROUPS = {'g1': ['24240', '26966', '27506'], 'g2': ['17563', '2084', '2881'], 'g3': ['17569', '23695'],
          'g4': ['25358', '27494', '27515']}
PERTURBATION_SEED, N_PERTURBATIONS = 20260923, 10
SPAWN_LATERAL_M, SPAWN_YAW_DEG = .25, 1.5


def arms():
    base = json.loads((POSE / 'configs/baseline-zero.json').read_text())
    slip = dict(pursuit_frame='rear_slip', rear_slip_c_per_rad=C_MKZ, steer_inverse='ackermann', track_width_m=TRACK_MKZ)
    return {'prod-k0': dict(base),
            'prod-kfix': dict(base, pose_lateral_coefficient_s2_per_m=K_FIXED),
            'slipack-k0': dict(base, **slip),
            'slipack-kfix': dict(base, **slip, pose_lateral_coefficient_s2_per_m=K_FIXED),
            'prod-truth': dict(base, diagnostic_truth_pose_ceiling=True),
            'slipack-truth': dict(base, **slip, diagnostic_truth_pose_ceiling=True)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', required=True, help='absolute checkout path on the box that will run')
    args = ap.parse_args()
    out = HERE / 'inputs'
    (out / 'configs').mkdir(parents=True, exist_ok=True)
    held = json.loads((out / 'heldout-windows.json').read_text())
    held_routes = {r.get('id'): r for r in ET.parse(str(out / 'heldout-routes.xml')).getroot().findall('route')}
    dev_routes = {r.get('id'): r for r in ET.parse(str(POSE / 'routes.xml')).getroot().findall('route')}
    root = ET.Element('routes')
    for rid in ORDER:
        root.append(copy.deepcopy(dev_routes.get(rid, held_routes.get(rid))))
    ET.ElementTree(root).write(str(out / 'routes.xml'), encoding='utf-8', xml_declaration=True)
    assert sorted(sum(GROUPS.values(), [])) == sorted(ORDER)
    for name, ids in GROUPS.items():
        group = ET.Element('routes')
        for element in root.findall('route'):
            if element.get('id') in ids:
                group.append(copy.deepcopy(element))
        ET.ElementTree(group).write(str(out / ('routes-%s.xml' % name)), encoding='utf-8', xml_declaration=True)
    windows = [dict(w, heldout=False) for w in DEVELOPMENT]
    for w in held:
        windows.append(dict(name=w['route_id'], route_id=w['route_id'], town=w['town'], cls=w['cls'],
                            cruise_mps=w['cruise_mps'], core_m=w['core_m'], window_m=w['window_m'],
                            post_m=[w['window_m'][1], w['window_m'][1] + 10.], heldout=True,
                            turn_deg=w['turn_deg'], r_min_m=w['r_min_m'], reference_xy=w['reference_xy']))
    assert sorted({w['route_id'] for w in windows}) == sorted(ORDER)
    (out / 'windows.json').write_text(json.dumps(windows, indent=1))
    cruises = {'default': 8}
    cruises.update({w['route_id']: w['cruise_mps'] for w in windows if w['cruise_mps'] != 8.})
    (out / 'route-cruises.json').write_text(json.dumps(cruises, indent=1))
    rng = random.Random(PERTURBATION_SEED)
    table = {'p00': None}
    for i in range(1, N_PERTURBATIONS + 1):
        table['p%02d' % i] = dict(gnss_noise_seed=100 + i, imu_noise_seed=200 + i,
                                  spawn_lateral_m=round(rng.uniform(-SPAWN_LATERAL_M, SPAWN_LATERAL_M), 3),
                                  spawn_yaw_deg=round(rng.uniform(-SPAWN_YAW_DEG, SPAWN_YAW_DEG), 3))
    (out / 'perturbations.json').write_text(json.dumps(table, indent=1))
    variants = {}
    for name, config in arms().items():
        path = out / 'configs' / (name + '.json')
        path.write_text(json.dumps(config, indent=2))
        variants[name] = dict(controller_config=str(Path(args.repo_root) / path.relative_to(REPO)), presets=['pursuit'])
    (out / 'variants.json').write_text(json.dumps(variants, indent=1))
    print('\n'.join('%s %s' % (k, v) for k, v in table.items()))


if __name__ == '__main__':
    main()
