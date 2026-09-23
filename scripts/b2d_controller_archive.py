#!/usr/bin/env python
"""Preserve controller experiment inputs and inventory raw evidence without rewriting it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
# Importing b2d_run would require DATA_DIR; this module must not. /data is the Tokyo layout.
DATA_DIR = Path(os.environ.get('DATA_DIR', '/data'))
RUNTIME_NAMES = {'b2d_run.py', 'b2d_route.py', 'b2d_agent.py', 'b2d_hooks.py',
                 'b2d_controller.py', 'b2d_controller_adapter.py'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def git(*args, root=REPO):
    return subprocess.check_output(['git', '-C', str(root)] + list(args), text=True).strip()


def source_files():
    paths = set((REPO / 'scripts').glob('b2d*.py'))
    paths.update((REPO / 'scripts').glob('test_b2d*.py'))
    for directory in ('scripts/drive_runtime', 'drive_runtime', 'jevdrive/drive_runtime'):
        paths.update((REPO / directory).rglob('*.py'))
    return sorted(paths)


def snapshot(out, inputs, timing='before_run'):
    """Create an exclusive snapshot. Captured bytes, not HEAD alone, identify code."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for source in source_files():
        relative = source.relative_to(REPO)
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(dict(path=str(relative), sha256=digest(destination), bytes=destination.stat().st_size))
    preserved_inputs = []
    for index, source in enumerate(inputs):
        source = Path(source).resolve()
        destination = out / 'inputs' / ('%02d-%s' % (index, source.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        preserved_inputs.append(dict(original=str(source), archived=str(destination.relative_to(out)),
                                     sha256=digest(destination), bytes=destination.stat().st_size))
    repositories = {}
    for name, path in [('project', REPO), ('bench2drive', Path(os.environ.get('BENCH2DRIVE_ROOT', DATA_DIR / 'third_party/Bench2Drive'))),
                       ('tcp', Path(os.environ.get('B2D_ZOO_ROOT', DATA_DIR / 'third_party/Bench2DriveZoo')))]:
        try:
            repositories[name] = dict(path=str(path), commit=git('rev-parse', 'HEAD', root=path),
                                      status=git('status', '--porcelain', root=path))
        except (OSError, subprocess.CalledProcessError) as error:
            repositories[name] = dict(path=str(path), unavailable=str(error))
    relative_paths = [str(path.relative_to(REPO)) for path in source_files()]
    (out / 'source.patch').write_text(git('diff', 'HEAD', '--', *relative_paths))
    manifest = dict(schema_version=1, captured_at=time.time(), capture_timing=timing,
                    repositories=repositories, python=sys.version, executable=sys.executable,
                    platform=platform.platform(), argv=sys.argv, sources=records, inputs=preserved_inputs,
                    environment={key: os.environ.get(key) for key in
                                 ('DATA_DIR', 'CARLA_ROOT', 'BENCH2DRIVE_ROOT', 'CUDA_VISIBLE_DEVICES', 'DISPLAY')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def assert_sources_unchanged(manifest):
    changed = [row['path'] for row in manifest['sources']
               if (Path(row['path']).name in RUNTIME_NAMES or 'drive_runtime/' in row['path'])
               if not (REPO / row['path']).is_file() or digest(REPO / row['path']) != row['sha256']]
    if changed:
        raise RuntimeError('Source changed during frozen experiment: ' + ', '.join(changed))


def inventory(root, destination):
    """Hash every existing raw file; never treat a still-running run as immutable."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == destination:
            continue
        before = path.stat()
        checksum = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('File changed during inventory: ' + str(path))
        files.append(dict(path=str(path.relative_to(root)), bytes=after.st_size, sha256=checksum))
    result = dict(schema_version=1, root=str(root), captured_at=time.time(), files=files,
                  file_count=len(files), total_bytes=sum(row['bytes'] for row in files))
    destination.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('snapshot')
    capture.add_argument('--out', required=True)
    capture.add_argument('--input', action='append', default=[])
    capture.add_argument('--timing', default='before_run')
    index = commands.add_parser('inventory')
    index.add_argument('--root', required=True)
    index.add_argument('--out', required=True)
    args = parser.parse_args()
    result = snapshot(args.out, args.input, args.timing) if args.command == 'snapshot' else inventory(args.root, args.out)
    print(json.dumps({key: value for key, value in result.items() if key not in ('files', 'sources')}, indent=2))


if __name__ == '__main__':
    main()
