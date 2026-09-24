"""Create the frozen Task 10 route and controller inputs without reading outcomes."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'todos/2026-09-23-tfv6-controller/controller-eval'
SOURCE = Path('/data/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml')
CONFIGS = ROOT / 'todos/2026-09-23-controller-next/lateral_v2/inputs/configs'
# Selected by scenario family/town, after excluding all W2/W2b cases and the
# old lateral-v2 development windows. No Task 10 controller results were used.
HELDOUT = ['24816', '25845', '24252', '26990', '26944', '25863', '25928', '3364']
DEV = ['24240', '26966', '17563']


def route_file(source, ids, target):
    by_id = {r.get('id'): r for r in ET.parse(source).getroot().findall('route')}
    if set(ids) - set(by_id):
        raise ValueError('Unknown route ID')
    root = ET.Element('routes')
    for rid in ids:
        r = ET.fromstring(ET.tostring(by_id[rid]))
        # L1 runs use the route geometry with no background/scenario actors;
        # L2/L3 uses the untouched source XML with scenarios.
        scenarios = r.find('scenarios')
        if scenarios is not None:
            r.remove(scenarios)
        root.append(r)
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    route_file(SOURCE, HELDOUT, OUT / 'l1-heldout.xml')
    route_file(SOURCE, DEV, OUT / 'l1-dev.xml')
    root = ET.Element('routes')
    source = {r.get('id'): r for r in ET.parse(SOURCE).getroot().findall('route')}
    for rid in HELDOUT:
        root.append(ET.fromstring(ET.tostring(source[rid])))
    ET.ElementTree(root).write(OUT / 'l23-heldout.xml', encoding='utf-8', xml_declaration=True)
    adapter = {
        'rear_axle_offset_m': -1.388633220199954,
        'pose_lateral_coefficient_s2_per_m': .010659832,
        'route_stop_deceleration': 2.0,
    }
    (OUT / 'author.json').write_text(json.dumps({'adapter': adapter}, indent=2) + '\n')
    configs = {
        'A': OUT / 'author.json', 'B': OUT / 'author.json',
        'C': CONFIGS / 'prod-kfix.json', 'D': CONFIGS / 'slipack-kfix.json',
    }
    variants = {arm: {'controller_config': str(path.resolve()),
                      'presets': [preset]} for arm, path, preset in [
                          ('A', configs['A'], 'author_route'),
                          ('B', configs['B'], 'author_waypoint'),
                          ('C', configs['C'], 'pursuit'),
                          ('D', configs['D'], 'pursuit')]}
    (OUT / 'l1-variants.json').write_text(json.dumps(variants, indent=2) + '\n')
    perturb = {'p00': None}
    for i, lateral, yaw in [(1, -.077, -.621), (2, .143, -.763), (3, -.003, -1.418),
                            (4, .072, .056), (5, .152, .047)]:
        perturb[f'p{i:02d}'] = {'gnss_noise_seed': 100+i, 'imu_noise_seed': 200+i,
                               'spawn_lateral_m': lateral, 'spawn_yaw_deg': yaw}
    (OUT / 'perturbations.json').write_text(json.dumps(perturb, indent=2) + '\n')
    (OUT / 'l1-cruises.json').write_text(json.dumps({'default': 8., '17563': 6.,
                                                    '25863': 6., '3364': 6.}, indent=2) + '\n')
    (OUT / 'selection.json').write_text(json.dumps({'dev': DEV, 'heldout': HELDOUT,
        'selection': 'Scenario-family and town coverage, no Task 10 outcome viewed',
        'source_xml': str(SOURCE)}, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
