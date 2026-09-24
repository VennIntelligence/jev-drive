"""Task 10 amendment v2 inputs: larger held-out route sets, drawn before any v2 outcome.

Excluded: every route driven in controller development (W2/W2b, D2/D3, search, A-defects,
lateral-v2, TCP controller work) and the Task 10 dev routes. The eight v1 held-out routes stay.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'todos/2026-09-23-tfv6-controller/controller-eval'
SOURCE = Path('/data/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml')
DEVELOPMENT = {'1773', '2050', '2084', '2091', '2390', '2881', '3072', '3255', '3514', '17563',
               '17569', '23695', '24240', '25318', '25358', '25378', '25381', '25424', '25854',
               '26405', '26966', '27494', '27506', '27515', '27529', '28154', '28198'}
V1 = ['24816', '25845', '24252', '26990', '26944', '25863', '25928', '3364']
BIG = ('Town12', 'Town13')
# (L1 draws, L2 draws) per stratum
QUOTA = {'small': (20, 4), 'Town12': (8, 3), 'Town13': (4, 1)}


def write(routes, target, strip):
    root = ET.Element('routes')
    for route in routes:
        route = ET.fromstring(ET.tostring(route))
        if strip and route.find('scenarios') is not None:
            route.remove(route.find('scenarios'))
        root.append(route)
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)


def main():
    source = {r.get('id'): r for r in ET.parse(SOURCE).getroot().findall('route')}
    rng = np.random.default_rng(20260925)
    l1, l2 = list(V1), list(V1)
    for stratum, (n1, n2) in QUOTA.items():
        pool = sorted((rid for rid, r in source.items() if rid not in DEVELOPMENT and rid not in V1
                       and (r.get('town') == stratum if stratum in BIG else r.get('town') not in BIG)),
                      key=int)
        if stratum == 'small':
            # Spread small-town draws across towns before taking a second route from any town.
            by_town = {}
            for rid in rng.permutation(pool):
                by_town.setdefault(source[rid].get('town'), []).append(rid)
            order = [rid for depth in range(max(map(len, by_town.values())))
                     for town in sorted(by_town) for rid in by_town[town][depth:depth+1]]
        else:
            order = list(rng.permutation(pool))
        l1 += order[:n1]
        l2 += order[n1:n1+n2]
    assert len(set(l1)) == 40 and len(set(l2)) == 16 and not (set(l1) - set(V1)) & set(l2)
    write([source[r] for r in l1], OUT / 'l1-v2-heldout.xml', strip=True)
    write([source[r] for r in l2], OUT / 'l23-v2-heldout.xml', strip=False)
    (OUT / 'selection-v2.json').write_text(json.dumps({
        'l1_heldout': l1, 'l23_heldout': l2, 'excluded_development': sorted(DEVELOPMENT, key=int),
        'rng_seed': 20260925, 'quota': QUOTA, 'source_xml': str(SOURCE),
        'selection': 'town-stratified random draw; no v2 outcome viewed'}, indent=2) + '\n')
    for name, ids in (('L1', l1), ('L2', l2)):
        towns = [source[r].get('town') for r in ids]
        print(name, len(ids), {t: towns.count(t) for t in sorted(set(towns))})


if __name__ == '__main__':
    main()
