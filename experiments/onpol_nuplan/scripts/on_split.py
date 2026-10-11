#!/usr/bin/env python3
"""ONPOL-NUPLAN split (plans/2026-10-11-onpol-prereg.md, Data): the 1491 public nuPlan AlpaSim scenes -> `train` (on-policy rows) and `held`
(the lane's only closed-loop read), by log.

Rule, as registered: scenes = c0b chunk0-2 (700) + cf1fresh (391) + ot3new (400), scene id `<log>-<token>`; unit = log (c0b_report.log_of);
the sorted logs are permuted with numpy.random.default_rng(0) and taken in that order into `train` until it holds at least 1000 scenes; the
rest is `held`. Nothing else is read (no score, no rollout).

  python experiments/onpol_nuplan/scripts/on_split.py --lists <dir with chunk0.txt chunk1.txt chunk2.txt cf1fresh.txt ot3new.txt>
      (the box: $DATA_DIR/runs/alpasim/c0b/lists; plain python + numpy; --check only compares with the files already written)
Writes experiments/onpol_nuplan/results/split/{train,held}.txt (sorted scene ids), split.json (counts, source hashes), and registers
`alpasim_nuplan/onpol-train` / `alpasim_nuplan/onpol-held` (jevdrive.data.splits, unit scene).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/alpasim/scripts")]
OUT = REPO / "experiments/onpol_nuplan/results/split"
SOURCES = ("chunk0", "chunk1", "chunk2", "cf1fresh", "ot3new")
N_TRAIN = 1000


def split(scenes) -> tuple:
    """Sorted scene ids -> (train, held) scene lists by the registered rule."""
    from c0b_report import log_of
    logs = sorted({log_of(s) for s in scenes})
    by = {}
    for s in scenes:
        by.setdefault(log_of(s), []).append(s)
    train = []
    for i in np.random.default_rng(0).permutation(len(logs)):
        if len(train) >= N_TRAIN:
            break
        train += by[logs[i]]
    tr = set(train)
    return sorted(train), sorted(s for s in scenes if s not in tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True)
    ap.add_argument("--check", action="store_true", help="compare with the written files, write nothing")
    a = ap.parse_args()
    from c0b_report import log_of
    from jevdrive.data import splits
    src = {n: (Path(a.lists) / f"{n}.txt").read_text().split() for n in SOURCES}
    scenes = sorted(s for v in src.values() for s in v)
    assert len(scenes) == len(set(scenes)) == 1491, f"{len(scenes)} scenes, {len(set(scenes))} distinct: the source lists changed"
    train, held = split(scenes)
    assert not {log_of(s) for s in train} & {log_of(s) for s in held}
    if a.check:
        for n, v in (("train", train), ("held", held)):
            assert (OUT / f"{n}.txt").read_text().split() == v, f"{n}.txt differs from the rule"
        print("split files match the rule")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    info = dict(rule="sorted logs permuted with numpy.random.default_rng(0), taken into train until >= 1000 scenes; rest held", n_scenes=len(scenes),
                n_logs=len({log_of(s) for s in scenes}), sources={n: dict(n=len(v), sha256=hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()) for n, v in src.items()})
    for n, v in (("train", train), ("held", held)):
        (OUT / f"{n}.txt").write_text("\n".join(v) + "\n")
        sp = splits.define("alpasim_nuplan", f"onpol-{n}", v, "scene",
                           "experiments/onpol_nuplan/scripts/on_split.py: the 1491 public nuPlan AlpaSim scenes (c0b chunk0-2, cf1fresh, ot3new) split by log "
                           f"(c0b_report.log_of), rng(0) permutation of the sorted logs, train until >= {N_TRAIN} scenes; frozen in "
                           f"experiments/onpol_nuplan/results/split/{n}.txt",
                           used_by=["experiments/onpol_nuplan"], status="frozen",
                           notes=("on-policy rollouts and rows come from these scenes only" if n == "train" else
                                  "held out from the on-policy rows; the lane's only closed-loop read; never rolled out for rows"))
        info[n] = dict(scenes=len(v), logs=len({log_of(s) for s in v}), split_id=sp.id,
                       by_source={k: len(set(v) & set(x)) for k, x in src.items()})
    (OUT / "split.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("n_scenes", "n_logs", "train", "held")}, indent=1))


if __name__ == "__main__":
    main()
