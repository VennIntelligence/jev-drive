"""Parallel ranged HTTP download with sha256 check (the box's link is shared per TCP stream)."""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HF_MIRROR = "https://hf-mirror.com"  # domestic HF mirror, goes direct (no proxy) on the box


def hf_url(repo: str, path: str, dataset: bool = False, mirror: bool = True) -> str:
    base = HF_MIRROR if mirror else "https://huggingface.co"
    return f"{base}/{'datasets/' if dataset else ''}{repo}/resolve/main/{path}"


def download(url: str, dst: Path, size: int | None = None, sha256: str | None = None, streams: int = 8,
             chunk: int = 4 << 20) -> Path:
    """Fetch url to dst using `streams` concurrent range requests; skip if dst already has the right size."""
    dst = Path(dst)
    if size is None:
        size = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    if dst.exists() and dst.stat().st_size == size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as f:
        f.truncate(size)

    def get(a):
        b = min(a + chunk, size) - 1
        for attempt in range(8):
            try:
                buf, t0 = bytearray(), time.monotonic()
                with requests.get(url, headers={"Range": f"bytes={a}-{b}"}, stream=True, timeout=(10, 20)) as r:
                    r.raise_for_status()
                    for part in r.iter_content(1 << 20):
                        buf += part
                        if time.monotonic() - t0 > 120:  # trickling connection: drop it and retry
                            raise TimeoutError("slow chunk")
                assert len(buf) == b - a + 1, "short read"
                with open(tmp, "r+b") as f:
                    f.seek(a)
                    f.write(buf)
                return
            except Exception:
                if attempt == 7:
                    raise

    with ThreadPoolExecutor(streams) as ex:
        list(ex.map(get, range(0, size, chunk)))
    if sha256:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            while b := f.read(1 << 24):
                h.update(b)
        if h.hexdigest() != sha256:
            raise RuntimeError(f"{dst.name}: sha256 mismatch")
    tmp.rename(dst)
    return dst
