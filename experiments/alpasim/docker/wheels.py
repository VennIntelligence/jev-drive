#!/usr/bin/env python3
"""Fetch the wheels of requirements.lock into a directory, each file in parallel byte ranges, and check every sha256 against the lock.
For hosts where one connection is slow but many are not (the Tokyo box: ~0.25 MB/s per stream to the wheel CDNs through its proxy node;
a serial installer needs hours for torch + the CUDA libraries). The files are the official ones (PyPI's JSON API gives the URL of each
locked hash, torch comes from the URL in the lock), so an offline install from the directory equals an install from the indexes.
Restartable: finished files are kept, a part that stalls is retried. stdlib only.

  wheels.py <requirements.lock> <dir> [--streams 16] [--python cp311] [--part-mb 8]
then: WHEELS=<dir> experiments/alpasim/docker/build.sh
New pins: `wheels.py requirements.in <dir>` fetches the pinned files themselves (no hashes to check yet), then
  uv pip compile requirements.in --find-links <dir> --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock
resolves torch from the local file instead of streaming it, and `wheels.py requirements.lock <dir>` fetches the rest and checks all.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "jev-wheels/1"}


def get(url: str, rng=None, timeout=30) -> bytes:
    h = dict(UA, **({"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}))
    for k in range(40):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                b = r.read()
            if rng is None or len(b) == rng[1] - rng[0] + 1:
                return b
        except Exception as e:  # noqa: BLE001  (stalls, resets: retry the part)
            err = e
        time.sleep(min(2 + k, 15))
    raise RuntimeError(f"{url} {rng}: {err}")


def size(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0")), timeout=30) as r:
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


def parse(lock: str):
    """requirements.txt with hashes -> [(name, version or None, url or None, {sha256})]."""
    out = []
    for blk in re.split(r"\n(?=\S)", re.sub(r"\\\n", " ", lock)):
        line = blk.split("\n")[0]
        if not line.strip() or line.startswith("#"):
            continue
        hs = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
        if m := re.match(r"(\S+) @ (\S+)", line):
            out.append((m[1], None, m[2], hs))
        elif m := re.match(r"([A-Za-z0-9_.\-]+)==(\S+)", line):
            out.append((m[1], m[2], None, hs))
    return out


def usable(fn: str, py: str) -> bool:
    if not fn.endswith(".whl"):
        return False
    _, _, pyt, abi, plat = fn[:-4].rsplit("-", 4)
    return (py in pyt.split(".") or "py3" in pyt.split(".") or (abi == "abi3" and int(pyt[2:] or 0) <= int(py[2:]))) \
        and (plat == "any" or ("x86_64" in plat and "manylinux" in plat))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lock")
    ap.add_argument("dir")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--python", default="cp311")
    ap.add_argument("--part-mb", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    files = []                                                        # (file name, url, sha256 or None when the lock has no hash for it)
    for name, ver, url, hs in parse(Path(a.lock).read_text()):
        if url:
            files.append((urllib.request.unquote(url.rsplit("/", 1)[1]), url, next(iter(hs)) if len(hs) == 1 else None))
            continue
        if "+cu" in ver:                                              # a PyTorch-index build: not on PyPI
            fn = f"{name}-{ver}-{a.python}-{a.python}-manylinux_2_28_x86_64.whl"
            files.append((fn, f"https://download.pytorch.org/whl/{ver.split('+')[1]}/{urllib.request.quote(fn)}", next(iter(hs)) if len(hs) == 1 else None))
            continue
        meta = json.loads(get(f"https://pypi.org/pypi/{name}/{ver}/json"))
        ok = [f for f in meta["urls"] if usable(f["filename"], a.python) and (not hs or f["digests"]["sha256"] in hs)]
        assert ok, f"{name}=={ver}: no {a.python} manylinux x86_64 wheel among the locked hashes"
        files += [(f["filename"], f["url"], f["digests"]["sha256"]) for f in ok]
    pool, part, t0, total = ThreadPoolExecutor(a.streams), a.part_mb << 20, time.time(), 0
    for fn, url, sha in sorted(files):
        f = d / fn
        if not f.exists():
            n = size(url)
            parts = list(pool.map(lambda s: get(url, (s, min(s + part, n) - 1), 120), range(0, n, part)))
            f.with_suffix(".part").write_bytes(b"".join(parts))
            f.with_suffix(".part").rename(f)
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if sha and got != sha:
            f.unlink()
            sys.exit(f"{fn}: sha256 {got} != locked {sha} (file removed)")
        total += f.stat().st_size
        print(f"{time.time() - t0:7.0f} s  {f.stat().st_size / 2**20:8.1f} MiB  {got[:16]}{'' if sha else ' (no hash in the lock)'}  {fn}", flush=True)
    (d / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((d / fn).read_bytes()).hexdigest()}  {fn}\n" for fn, _, _ in sorted(files)))
    print(f"{len(files)} wheels, {total / 2**30:.2f} GiB in {time.time() - t0:.0f} s -> {d}")


if __name__ == "__main__":
    main()
