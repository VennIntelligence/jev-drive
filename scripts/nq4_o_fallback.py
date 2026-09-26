#!/usr/bin/env python3
"""Count same-town/type clips; trigger distance is explicitly not checked."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    files = {
        'b2d_base': 'bench2drive_base_1000.json',
        'b2d_full': 'bench2drive_full+sup_13638.json',
    }
    inventories = {}
    pattern = re.compile(r'^(?P<scenario>.+)_(?P<town>Town\d+(?:HD)?)_Route\d+_[^/]*\.tar\.gz$')
    provenance = {}
    for dataset, filename in files.items():
        path = args.metadata_dir / filename
        manifest = json.loads(path.read_text())
        counter = collections.Counter()
        for clip in manifest:
            match = pattern.fullmatch(clip)
            if match is None:
                raise ValueError(f'Unrecognized clip name: {clip}')
            counter[match['town'], match['scenario']] += 1
        assert sum(counter.values()) == len(manifest)
        inventories[dataset] = counter
        provenance[dataset] = {'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'clips': len(manifest), 'unparsed_clips': 0}
    xml = args.metadata_dir / 'bench2drive220.xml'
    routes = ET.parse(xml).getroot().findall('route')
    if len(routes) != 220 or len({r.attrib['id'] for r in routes}) != 220:
        raise ValueError('Expected 220 unique evaluation routes')
    rows = []
    for route in routes:
        town = route.attrib['town']
        types = sorted({s.attrib['type'] for s in route.findall('.//scenario')})
        if not types:
            raise ValueError(f'Missing scenario type: {route.attrib}')
        for dataset in [*files, 'tfv6_lead', 'bridgedrive_lead', 'simlingo', 'blue']:
            count = sum(inventories[dataset][town, s] for s in types) if dataset in inventories else 'not determinable'
            rows.append({'route_id': route.attrib['id'], 'town': town, 'scenario_types': '|'.join(types),
                         'dataset': dataset, 'same_town_scenario_clip_count': count,
                         'trigger_distance_checked': 'false',
                         'status': 'counted' if dataset in inventories else 'not determinable'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'overlap_fallback.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance['evaluation'] = {'file': xml.name, 'sha256': hashlib.sha256(xml.read_bytes()).hexdigest(), 'routes': len(routes)}
    (args.output_dir / 'fallback_sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'routes': len(routes), 'rows': len(rows), 'sources': provenance}))


if __name__ == '__main__':
    main()
