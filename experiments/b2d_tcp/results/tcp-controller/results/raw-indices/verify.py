"""Recheck retained raw files against exclusive post-run inventories."""
import hashlib
import json
from pathlib import Path


def verify(directory):
    rows = []
    for path in sorted(directory.glob('*-inventory.json')):
        index = json.loads(path.read_text())
        mismatches = []
        for record in index['files']:
            source = Path(index['root']) / record['path']
            if (not source.is_file() or source.stat().st_size != record['bytes']
                    or hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']):
                mismatches.append(record['path'])
        rows.append(dict(index=path.name, root=index['root'], files=index['file_count'],
                         bytes=index['total_bytes'], mismatches=mismatches,
                         index_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


if __name__ == '__main__':
    result = verify(Path(__file__).resolve().parent)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if any(row['mismatches'] for row in result) else 0)
