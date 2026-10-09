"""Page-cache trimmer: keeps the container's working set away from the platform's kill line.

The platform SIGKILLs the largest process (by RSS) of the container whenever
    memory.current - inactive_file  >=  0.98 x memory.max
(measured 2026-10-10, docs/remote-box.md "Host memory"). Page cache that was read more than once sits on the active
file list and counts in full; the kernel only moves it back to the inactive list when that list is nearly empty, and
only reclaims at memory.high, which lies above the kill line. So a container whose cache is warm has no room for a
job that allocates quickly, whatever its quota. Nothing in the container can change the cgroup (no memory.reclaim on
this kernel, /sys/fs/cgroup is read-only, vm.drop_caches needs root), but any user can drop the cached pages of a
file it can read: posix_fadvise(POSIX_FADV_DONTNEED) frees clean pages that no process has mapped, active or not
(2.4 GiB in 0.15 s).

    margin()                     GiB between the working set and the kill line
    trim(need_gb)                drop cached files until margin() >= need_gb: files nobody has open first, oldest
                                 access first, then files that are open (their mapped pages stay)
    python -m jevdrive.cl trim --margin 120 [--dry]

The file list (regular files >= 16 MiB under the roots, default $DATA_DIR) is walked once and cached in
<pool>/cachefiles.json for LIST_MAX_AGE_S; 949 such files held 276 of the 286 GiB cached on 2026-10-10.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

MIN_BYTES = 16 * 2 ** 20
LIST_MAX_AGE_S = 2 * 3600.0
KILL_FRAC = 0.98                      # the platform's kill line as a share of memory.max
CG = Path("/sys/fs/cgroup")
G = 2 ** 30


def mem(cg: Path = CG) -> dict:
    """{max, current, inactive_file, active_file, anon, ws} in bytes; ws = current - inactive_file. Empty without a
    cgroup v2 memory controller or without a limit."""
    try:
        mx = (cg / "memory.max").read_text().strip()
        cur = int((cg / "memory.current").read_text())
        stat = dict(l.split() for l in (cg / "memory.stat").read_text().splitlines() if l.count(" ") == 1)
    except (OSError, ValueError):
        return {}
    if mx == "max":
        return {}
    d = {k: int(stat.get(k, 0)) for k in ("inactive_file", "active_file", "anon")}
    d.update(max=int(mx), current=cur, ws=cur - d["inactive_file"])
    return d


def margin(cg: Path = CG, frac: float = KILL_FRAC) -> float:
    """GiB the working set may still grow before the kill line; inf when the container has no memory limit."""
    m = mem(cg)
    return (frac * m["max"] - m["ws"]) / G if m else float("inf")


def in_use(proc: Path = Path("/proc")) -> set:
    """(major, minor, inode) of every file some readable process has open or mapped."""
    out = set()
    for p in proc.glob("[0-9]*"):
        try:
            for fd in os.scandir(p / "fd"):
                try:
                    st = os.stat(fd.path)
                    out.add((os.major(st.st_dev), os.minor(st.st_dev), st.st_ino))
                except OSError:
                    pass
            for line in (p / "maps").read_text().splitlines():
                f = line.split(None, 5)
                if len(f) == 6 and f[5].startswith("/") and f[4] != "0":
                    mj, mn = f[3].split(":")
                    out.add((int(mj, 16), int(mn, 16), int(f[4])))
        except OSError:
            continue
    return out


def big_files(roots, min_bytes: int = MIN_BYTES) -> list:
    """[path, size, atime, major, minor, inode] of regular files >= min_bytes under the roots (one filesystem each)."""
    out = []
    for root in roots:
        try:
            dev = os.stat(root).st_dev
        except OSError:
            continue
        stack = [str(root)]
        while stack:
            try:
                it = list(os.scandir(stack.pop()))
            except OSError:
                continue
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.stat(follow_symlinks=False).st_dev == dev:
                            stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                        if st.st_size >= min_bytes:
                            out.append([e.path, st.st_size, st.st_atime, os.major(st.st_dev), os.minor(st.st_dev), st.st_ino])
                except OSError:
                    continue
    return out


def file_list(roots, cache_file: Path = None, max_age: float = LIST_MAX_AGE_S) -> list:
    roots = [str(r) for r in roots]
    try:
        d = json.loads(Path(cache_file).read_text())
        if d["roots"] == roots and time.time() - d["t"] < max_age:
            return d["files"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    files = big_files(roots)
    if cache_file:
        tmp = Path(str(cache_file) + ".tmp")
        tmp.write_text(json.dumps(dict(t=time.time(), roots=roots, files=files)))
        os.replace(tmp, cache_file)
    return files


def drop(path: str) -> bool:
    """Drop the clean, unmapped cached pages of one file."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def trim(need_gb: float, roots, cache_file: Path = None, cg: Path = CG, frac: float = KILL_FRAC, dry: bool = False,
         lock: Path = None, margin_fn=None, drop_fn=drop, busy: set = None) -> dict:
    """Drop cached files until the margin to the kill line is >= need_gb. Returns {before, after (GiB margin), files
    (dropped), busy (of them open somewhere), seconds, skipped (another trim holds the lock)}."""
    margin_fn = margin_fn or (lambda: margin(cg, frac))
    t0, before = time.time(), margin_fn()
    out = dict(before=round(before, 1), after=round(before, 1), need=round(need_gb, 1), files=0, busy=0, seconds=0.0)
    if before >= need_gb:
        return out
    lk = None
    if lock:
        lk = open(lock, "a")
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lk.close()
            return dict(out, skipped=True)
    try:
        files = file_list(roots, cache_file)
        busy = in_use() if busy is None else busy
        files = sorted(files, key=lambda f: ((f[3], f[4], f[5]) in busy, f[2]))    # idle files first, oldest access first
        now = before
        for f in files:
            if now >= need_gb:
                break
            if dry:
                out["files"] += 1
                continue
            if drop_fn(f[0]):
                out["files"] += 1
                out["busy"] += (f[3], f[4], f[5]) in busy
                now = margin_fn()                              # two small reads; a drop is an open + a syscall
        out.update(after=round(margin_fn(), 1), seconds=round(time.time() - t0, 2))
    finally:
        if lk:
            lk.close()
    return out
