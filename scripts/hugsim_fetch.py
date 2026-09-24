"""Fetch the HUGSIM closed-loop benchmark data into a plain directory tree (not the HF cache), so
`docs/hugsim.md` paths are stable and match how the eval scripts expect to find the data.

  python scripts/hugsim_fetch.py sample_data   # hyzhou404/HUGSIM sample_data/data.zip, ~2.4 GB, smoke test
  python scripts/hugsim_fetch.py benchmark     # XDimLab/HUGSIM full closed-loop benchmark, ~61 GB
  python scripts/hugsim_fetch.py sample_data benchmark --workers 8

Both repos serve LFS through HF's Xet CAS bridge, whose Range (byte-serving) path stalls and times out
under load regardless of route (hf-mirror.com direct or proxy_on + huggingface.co) -- a plain whole-file
GET on the same object is reliable. So each file is fetched with ONE plain GET (`streams=1`, see
`jevdrive.hfdl.download`), and parallelism comes from fetching many files at once (`--workers`) instead of
many ranges of one file. Routed through `proxy_on` + huggingface.co: measured faster and far more reliable
than hf-mirror.com direct for this repo (see docs/hugsim.md).

Every LFS file is sha256-checked against the HF API's reported oid; every non-LFS file against the git
blob sha1. Resumable at file granularity: a file already at the right size is skipped. A plain GET has no
mid-file resume, so a dropped connection re-fetches the whole file -- cheap at this file size (files run
up to ~1 GB; the file itself, not a chunk of it, is the retry unit).
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.hfdl import download, git_blob_sha1, hf_url, tree

DEST = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "datasets" / "hugsim"
CLASH_PROXY = "http://127.0.0.1:7890"

# repo, dataset-repo?, path prefixes to include (a leading dir is a prefix match; a full path is exact)
JOBS = {
    "sample_data": ("hyzhou404/HUGSIM", ("sample_data/",)),
    "benchmark": ("XDimLab/HUGSIM", ("3DRealCar/", "scenes/", "nusc_map_cache.zip", "scenarios.zip")),
}


def wanted(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p) for p in prefixes)


def use_proxy() -> None:
    """Both HUGSIM repos serve LFS through the Xet CAS bridge; proxy_on + huggingface.co measured
    faster and far more reliable than hf-mirror.com direct for it (docs/hugsim.md). Scripts set their
    own proxy rather than relying on the caller's shell (docs/network-proxy.md)."""
    subprocess.run(["bash", "-ic", "proxy_on"], check=True)
    os.environ["http_proxy"] = os.environ["https_proxy"] = CLASH_PROXY
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"


def fetch(repo: str, prefixes: tuple[str, ...], workers: int, log=print) -> None:
    entries = [e for e in tree(repo, dataset=True, recursive=True) if wanted(e["path"], prefixes)]
    total = sum(e["size"] for e in entries)
    log(f"{repo}: {len(entries)} files, {total / 1e9:.2f} GB")
    done, lock = 0, Lock()

    def one(e: dict) -> tuple[str, int, bool]:
        dst = DEST / e["path"]
        lfs = e.get("lfs")
        url = hf_url(repo, e["path"], dataset=True)
        already = dst.exists() and dst.stat().st_size == e["size"]
        download(url, dst, e["size"], lfs and lfs["oid"], streams=1)
        if not lfs and git_blob_sha1(dst) != e["oid"]:
            dst.unlink()
            raise RuntimeError(f"{e['path']}: git oid mismatch against {repo}")
        return e["path"], e["size"], already

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, e) for e in sorted(entries, key=lambda e: e["size"])]
        for fut in as_completed(futs):
            path, size, already = fut.result()
            with lock:
                done += size
                snapshot = done
            tag = "(cached)" if already else "(sha256/oid ok)"
            log(f"[{snapshot / 1e9:.2f}/{total / 1e9:.2f} GB] {path}: {size / 1e6:.1f} MB {tag}")
    log(f"{repo}: done, {total / 1e9:.2f} GB in {DEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=list(JOBS))
    ap.add_argument("--workers", type=int, default=8, help="files fetched concurrently (measured best on the box)")
    a = ap.parse_args()
    use_proxy()
    for w in a.what:
        repo, prefixes = JOBS[w]
        fetch(repo, prefixes, a.workers)
