#!/usr/bin/env python3
"""Move preserved raw editions to a fresh /data archive, hash-verify, retain alias."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def inventory(directory):
    records = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            before = path.stat()
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('evidence changed during hashing: ' + str(path))
            records[str(path.relative_to(directory))] = dict(bytes=after.st_size, sha256=value)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--final-csv', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    args = parser.parse_args()
    if args.snapshots.is_symlink() or args.archive.exists() or args.mapping.exists():
        parser.error('source already relocated or destination/mapping exists; refuse overwrite')
    manifests = sorted(args.snapshots.glob('edition-*/manifest.json'))
    if not manifests or not json.loads(manifests[-1].with_name('summary.json').read_text())['campaign_ended'] or not args.final_csv.is_file():
        parser.error('final audit and final CSV must exist before relocation')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if len(argv) > 1 and Path(argv[1].decode(errors='replace')).name == 'audit.py':
            parser.error('audit.py process still present; stop the writer before relocation')
    original = args.snapshots.absolute()
    before = inventory(original)
    args.archive.mkdir(parents=True)
    archived = args.archive.resolve() / 'snapshots'
    shutil.move(str(original), str(archived))
    after = inventory(archived)
    if before != after:
        raise RuntimeError('before/after inventory differs; evidence retained in archive for inspection')
    original.symlink_to(archived, target_is_directory=True)
    raw_inventory = args.archive / 'inventory.json'
    raw_inventory.write_text(json.dumps(dict(verified=True, files=after), indent=2) + '\n')
    mapping = dict(relocated_at_unix=time.time(), original_path=str(original), archived_path=str(archived),
                   original_alias=str(original), alias_type='directory symlink; keep untracked',
                   historical_manifests='unchanged; original raw subfolder remains resolvable through alias',
                   files=len(after), bytes=sum(row['bytes'] for row in after.values()),
                   every_file_hash_verified_before_after=True, inventory_path=str(raw_inventory.resolve()),
                   inventory_sha256=hashlib.sha256(raw_inventory.read_bytes()).hexdigest(),
                   final_csv=str(args.final_csv.resolve()))
    args.mapping.write_text(json.dumps(mapping, indent=2) + '\n')
    print(json.dumps(mapping, indent=2))


if __name__ == '__main__':
    main()
