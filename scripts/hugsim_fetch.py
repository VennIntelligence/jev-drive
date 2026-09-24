"""Fetch the HUGSIM closed-loop benchmark data into a plain directory tree (not the HF cache), so
`docs/hugsim.md` paths are stable and match how the eval scripts expect to find the data.

  python scripts/hugsim_fetch.py sample_data   # hyzhou404/HUGSIM sample_data/data.zip, ~2.4 GB, smoke test
  python scripts/hugsim_fetch.py benchmark     # XDimLab/HUGSIM full closed-loop benchmark, ~61 GB
  python scripts/hugsim_fetch.py sample_data benchmark --streams 16

Every LFS file is sha256-checked against the HF API's reported oid; every non-LFS file against the git
blob sha1. Resumable: a file already at the right size is skipped, and `hfdl.download` retries a stalled
chunk instead of failing the whole file.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.hfdl import download, git_blob_sha1, hf_url, tree

DEST = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "datasets" / "hugsim"

# repo, dataset-repo?, path prefixes to include (a leading dir is a prefix match; a full path is exact)
JOBS = {
    "sample_data": ("hyzhou404/HUGSIM", ("sample_data/",)),
    "benchmark": ("XDimLab/HUGSIM", ("3DRealCar/", "scenes/", "nusc_map_cache.zip", "scenarios.zip")),
}


def wanted(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p) for p in prefixes)


def fetch(repo: str, prefixes: tuple[str, ...], streams: int, log=print) -> None:
    entries = [e for e in tree(repo, dataset=True, recursive=True) if wanted(e["path"], prefixes)]
    total = sum(e["size"] for e in entries)
    log(f"{repo}: {len(entries)} files, {total / 1e9:.2f} GB")
    done = 0
    for e in sorted(entries, key=lambda e: e["size"]):
        dst = DEST / e["path"]
        lfs = e.get("lfs")
        url = hf_url(repo, e["path"], dataset=True)
        already = dst.exists() and dst.stat().st_size == e["size"]
        download(url, dst, e["size"], lfs and lfs["oid"], streams=streams)
        if not lfs and git_blob_sha1(dst) != e["oid"]:
            dst.unlink()
            raise RuntimeError(f"{e['path']}: git oid mismatch against {repo}")
        done += e["size"]
        tag = "(cached)" if already else "(sha256 ok)" if lfs else "(git oid ok)"
        log(f"[{done / 1e9:.2f}/{total / 1e9:.2f} GB] {e['path']}: {e['size'] / 1e6:.1f} MB {tag}")
    log(f"{repo}: done, {total / 1e9:.2f} GB in {DEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=list(JOBS))
    ap.add_argument("--streams", type=int, default=16)
    a = ap.parse_args()
    for w in a.what:
        repo, prefixes = JOBS[w]
        fetch(repo, prefixes, a.streams)
