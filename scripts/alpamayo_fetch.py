"""Fetch Alpamayo 1.5 weights (+ the Qwen3-VL processor files it borrows) into the HF hub cache via hf-mirror.

  python scripts/alpamayo_fetch.py weights            # nvidia/Alpamayo-1.5-10B, ~22 GB, sha256-checked
  python scripts/alpamayo_fetch.py processor          # Qwen/Qwen3-VL-2B-Instruct tokenizer/processor configs only

Clips are not fetched here: jevdrive.alpamayo.data streams single clips out of the dataset's zip chunks.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.hfdl import snapshot

REPOS = {
    "weights": ("nvidia/Alpamayo-1.5-10B", ("*",)),
    "processor": ("Qwen/Qwen3-VL-2B-Instruct", ("*.json", "*.txt", "*.jinja", "merges.txt")),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=list(REPOS))
    ap.add_argument("--streams", type=int, default=16)
    a = ap.parse_args()
    for w in a.what:
        repo, pats = REPOS[w]
        print(snapshot(repo, pats, streams=a.streams), flush=True)
