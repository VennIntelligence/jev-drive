#!/usr/bin/env python3
"""Track and stop exact Linux process identities, never process groups."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import time

from cx_controller import atomic, identity, process_snapshot, same


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)  # Linux x86_64/aarch64 pidfd_open, also on older Python.
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, sig, 0, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def refresh(path, pid=None):
    rows = process_snapshot()
    old = json.loads(path.read_text()) if path.exists() else {'members': []}
    if pid is not None:
        if pid not in rows:
            raise RuntimeError(f'Cannot capture absent PID {pid}')
        old = {'root': identity(rows[pid]), 'members': [identity(rows[pid])]}
    owned = {r['pid'] for r in old['members'] if same(r, rows)}
    while True:
        new = {p for p, r in rows.items() if r['ppid'] in owned}
        if new <= owned:
            break
        owned |= new
    known = {(r['pid'], r['start_ticks']): r for r in old['members']}
    known.update({(p, rows[p]['start_ticks']): identity(rows[p]) for p in owned})
    old['members'] = list(known.values())
    atomic(path, old)
    return old, rows


def stop(path, grace=10):
    if not path.exists():
        raise RuntimeError(f'Missing owned identity record: {path}')
    record, _ = refresh(path)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        rows = process_snapshot()
        for member in reversed(record['members']):
            if same(member, rows):
                try:
                    # pidfd closes the final PID-reuse race between validation and signal.
                    fd = pidfd_open(member['pid'])
                    try:
                        if same(member, process_snapshot()):
                            pidfd_signal(fd, sig)
                    finally:
                        os.close(fd)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + grace
        while time.monotonic() < end:
            rows = process_snapshot()
            if not any(same(m, rows) for m in record['members']):
                return
            time.sleep(.2)
    raise RuntimeError(f'Owned processes did not exit: {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['capture', 'alive', 'stop'])
    ap.add_argument('record', type=Path)
    ap.add_argument('pid', nargs='?', type=int)
    a = ap.parse_args()
    if a.mode == 'stop':
        stop(a.record)
    else:
        if a.mode == 'alive' and not a.record.exists():
            return 1
        record, rows = refresh(a.record, a.pid if a.mode == 'capture' else None)
        return 0 if a.mode == 'capture' or same(record['root'], rows) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
