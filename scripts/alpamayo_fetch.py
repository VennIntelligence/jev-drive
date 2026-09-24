"""Fetch Alpamayo 1.5 weights (+ the Qwen3-VL processor files it borrows) into the HF hub cache.

  python scripts/alpamayo_fetch.py weights            # nvidia/Alpamayo-1.5-10B, ~22 GB, sha256-checked
  python scripts/alpamayo_fetch.py processor          # Qwen/Qwen3-VL-2B-Instruct tokenizer/processor configs only

Clips are not fetched here: jevdrive.alpamayo.data streams single clips out of the dataset's zip chunks.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.hfdl import snapshot

REPOS = {
    # HF metadata + sha256 from hf-mirror; the shards themselves from ModelScope's nv-community copy, because the
    # mirror redirects LFS to the HF US CDN (~0.5 MB/s per stream from the box) and ModelScope is ~6 MB/s per stream
    "weights": ("nvidia/Alpamayo-1.5-10B", ("*",), "nv-community/Alpamayo-1.5-10B"),
    "processor": ("Qwen/Qwen3-VL-2B-Instruct", ("*.json", "*.txt", "*.jinja", "merges.txt"), None),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=list(REPOS))
    ap.add_argument("--streams", type=int, default=16)
    a = ap.parse_args()
    for w in a.what:
        repo, pats, ms = REPOS[w]
        print(snapshot(repo, pats, streams=a.streams, ms_repo=ms), flush=True)
