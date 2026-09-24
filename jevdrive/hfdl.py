"""Fast HuggingFace downloads for the box: parallel range requests through hf-mirror.com (direct, no proxy).

Generalizes jevdrive/openpilot/dl.py with auth (gated repos), tree listing and HF-cache snapshots, so code that
calls `from_pretrained("<repo>")` finds the files without a slow single-stream hub download.
The token is read from $HF_TOKEN or $HF_HOME/token and never logged.
"""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

import requests

HF_MIRROR = "https://hf-mirror.com"


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", HF_MIRROR).rstrip("/")


def token() -> str | None:
    if t := os.environ.get("HF_TOKEN"):
        return t
    p = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return p.read_text().strip() if p.exists() else None


def auth() -> dict:
    return {"Authorization": f"Bearer {t}"} if (t := token()) else {}


def _kind(dataset: bool) -> str:
    return "datasets/" if dataset else ""


def hf_url(repo: str, path: str, dataset: bool = False, rev: str = "main") -> str:
    return f"{endpoint()}/{_kind(dataset)}{repo}/resolve/{rev}/{path}"


def repo_info(repo: str, dataset: bool = False) -> dict:
    r = requests.get(f"{endpoint()}/api/{'datasets' if dataset else 'models'}/{repo}", headers=auth(), timeout=60)
    r.raise_for_status()
    return r.json()


def tree(repo: str, path: str = "", dataset: bool = False, recursive: bool = False, rev: str = "main") -> list[dict]:
    """File entries {path, size, oid, lfs?} under `path`, following the API's Link pagination."""
    url = f"{endpoint()}/api/{'datasets' if dataset else 'models'}/{repo}/tree/{rev}/{path}"
    out, params = [], {"recursive": str(recursive).lower()}
    while url:
        r = requests.get(url, headers=auth(), params=params, timeout=120)
        r.raise_for_status()
        out += [e for e in r.json() if e["type"] == "file"]
        url, params = r.links.get("next", {}).get("url"), None
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, size: int | None = None, sha256: str | None = None, streams: int = 8,
             chunk: int = 4 << 20, headers: dict | None = None) -> Path:
    """Fetch url to dst using `streams` concurrent range requests; skip if dst already has the right size."""
    dst, headers = Path(dst), headers if headers is not None else auth()
    if size is None:
        size = int(requests.head(url, headers=headers, allow_redirects=True, timeout=60).headers["Content-Length"])
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
                with requests.get(url, headers={**headers, "Range": f"bytes={a}-{b}"}, stream=True,
                                  timeout=(10, 20)) as r:
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
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(streams) as ex:
        list(ex.map(get, range(0, size, chunk)))
    if sha256 and sha256_file(tmp) != sha256:
        raise RuntimeError(f"{dst.name}: sha256 mismatch")
    tmp.rename(dst)
    return dst


def snapshot(repo: str, patterns: tuple[str, ...] = ("*",), dataset: bool = False, streams: int = 16,
             log=print) -> Path:
    """Download matching files of `repo` into the standard HF hub cache layout (blobs + snapshots/<sha>),
    sha256-checking every LFS file, and return the snapshot dir. Offline `from_pretrained(repo)` then works."""
    from huggingface_hub.constants import HF_HUB_CACHE
    info = repo_info(repo, dataset)
    sha = info["sha"]
    root = Path(HF_HUB_CACHE) / f"{'datasets' if dataset else 'models'}--{repo.replace('/', '--')}"
    snap = root / "snapshots" / sha
    files = [e for e in tree(repo, dataset=dataset, recursive=True, rev=sha)
             if any(fnmatch(e["path"], p) for p in patterns)]
    total, t0 = sum(e["size"] for e in files), time.monotonic()
    for e in sorted(files, key=lambda e: e["size"]):
        lfs = e.get("lfs")
        blob = root / "blobs" / (lfs["oid"] if lfs else e["oid"])
        if not (blob.exists() and blob.stat().st_size == e["size"]):
            t1 = time.monotonic()
            download(hf_url(repo, e["path"], dataset, sha), blob, e["size"], lfs and lfs["oid"], streams)
            dt = time.monotonic() - t1
            log(f"{e['path']}: {e['size'] / 1e6:.0f} MB in {dt:.0f} s = {e['size'] / 1e6 / dt:.1f} MB/s"
                + (" (sha256 ok)" if lfs else ""))
        link = snap / e["path"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(blob, link.parent))
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(sha)
    dt = time.monotonic() - t0
    log(f"{repo}@{sha[:10]}: {len(files)} files, {total / 1e9:.2f} GB, {dt:.0f} s wall")
    return snap
