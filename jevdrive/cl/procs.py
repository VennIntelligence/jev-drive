"""Owned process trees, identified by (pid, start time), never by pattern (no pkill -f / pgrep -f, docs/long-runs.md).

A record (JSON) holds the root process and every descendant seen so far. `refresh` adds the descendants of members that
are still alive, so a server re-parented to init after its runner died is still ours; `stop` signals only members whose
pid still carries the recorded start time, through a pidfd, which closes the race between checking and signalling.
Extracted from scripts/cx_owned_process.py and cx_controller.py.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import tempfile
import time
from pathlib import Path

from .box import processes


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, str(path))


def identity(row: dict) -> dict:
    return {k: row[k] for k in ("pid", "start_ticks", "pgid", "sid")}


def same(rec: dict, rows: dict) -> bool:
    """Is the recorded process still the one running under its pid?"""
    return bool(rec) and rec["pid"] in rows and rows[rec["pid"]]["start_ticks"] == rec["start_ticks"]


def capture(path: Path, pid: int, rows: dict = None) -> dict:
    rows = processes() if rows is None else rows
    if pid not in rows:
        raise RuntimeError("cannot capture absent pid %d" % pid)
    rec = {"root": identity(rows[pid]), "members": [identity(rows[pid])]}
    atomic_json(path, rec)
    return rec


def refresh(path: Path, rows: dict = None) -> tuple:
    """Add the live descendants of live members; return (record, rows)."""
    rows = processes() if rows is None else rows
    rec = json.loads(Path(path).read_text())
    owned = {m["pid"] for m in rec["members"] if same(m, rows)}
    while True:
        new = {p for p, r in rows.items() if r["ppid"] in owned}
        if new <= owned:
            break
        owned |= new
    known = {(m["pid"], m["start_ticks"]): m for m in rec["members"]}
    added = {(p, rows[p]["start_ticks"]) for p in owned} - set(known)
    if added:
        known.update({k: identity(rows[k[0]]) for k in added})
        rec["members"] = list(known.values())
        atomic_json(path, rec)
    return rec, rows


def live_members(rec: dict, rows: dict) -> list:
    return [m["pid"] for m in rec["members"] if same(m, rows)]


def _pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)            # pidfd_open on x86_64 / aarch64
    if fd < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return fd


def _pidfd_signal(fd: int, sig: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, sig)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, sig, 0, 0) < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))


def signal_member(member: dict, sig: int) -> bool:
    try:
        fd = _pidfd_open(member["pid"])
    except OSError:
        return False
    try:
        if not same(member, processes()):
            return False
        _pidfd_signal(fd, sig)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def stop(path: Path, grace: float = 15.0) -> list:
    """SIGTERM, then SIGKILL, every live recorded member (children first). Returns the pids that were still alive."""
    rec, rows = refresh(path)
    hit = live_members(rec, rows)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for m in reversed(rec["members"]):
            if same(m, rows):
                signal_member(m, sig)
        end = time.monotonic() + grace
        while time.monotonic() < end:
            rows = processes()
            if not live_members(rec, rows):
                return hit
            time.sleep(0.3)
    if live_members(rec, processes()):
        raise RuntimeError("owned processes did not exit: %s" % path)
    return hit


def tree_threads(rec: dict, rows: dict) -> dict:
    """Threads in a live record by role: CARLA servers, route clients, the rest (runner, wrappers, agent servers)."""
    out = {"servers": 0, "server_threads": 0, "clients": 0, "client_threads": 0, "other_threads": 0}
    for pid in live_members(rec, rows):
        r = rows[pid]
        a = r["argv"]
        if a and "CarlaUE4-Linux-Shipping" in a[0]:
            out["servers"] += 1
            out["server_threads"] += r["threads"]
        elif any(x.endswith("b2d_route.py") for x in a):
            out["clients"] += 1
            out["client_threads"] += r["threads"]
        else:
            out["other_threads"] += r["threads"]
    return out
