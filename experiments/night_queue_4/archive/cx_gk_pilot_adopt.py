#!/usr/bin/env python3
"""Adopt complete ten-route staged pilots into the registered chain's ARMS=arms tree.

The GPU1 pilot owner (experiments/night_queue_4/archive/cx_gk_pilots.py, ARMS=cx_pilot_20260927) and the earlier
smoke runs wrote their pilot evidence under their own ARMS roots. The registered chain
(experiments/night_queue_4/archive/nq4_gk.sh, ARMS=arms) reads only runs/nq4/gk/arms_pilot/<cand>.<variant>/PASS.
This tool verifies that a source pilot is a complete staged ten-route pass and copies its
PASS and stage verdicts there, so the chain does not re-run the same pilot. Route outputs
stay where they are; the chain's step then runs its full registered route list itself.
No threshold, route list, candidate or reading is changed.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
G = DATA / "runs/nq4/gk"
DST = G / "arms_pilot"
SOURCES = (G / "smoke_pilot", G / "cx_pilot_20260927_pilot")
KEEP = ("PASS", "stage1.json", "stage2.json", "check.err")


def event(kind, **detail):
    row = dict(time=time.time(), kind=kind, **detail)
    with (G / "pilot_adopt.log").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def verify(d: Path):
    """A complete staged ten-route pilot, or None."""
    if not (d / "PASS").is_file():
        return None
    try:
        s1 = json.loads((d / "stage1.json").read_text())
        s2 = json.loads((d / "stage2.json").read_text())
    except (OSError, ValueError):
        return None
    if s1.get("pass") is not True or s2.get("pass") is not True:
        return None
    facts = s2.get("facts", {})
    if facts.get("routes") != 10 or facts.get("finished") != 10:
        return None
    return s2


def adopt(name: str, source: Path, dry: bool) -> str:
    cand, variant = name.rsplit(".", 1)
    if cand.startswith("k"):
        return "skip-k"                      # K unseen is owned by cx_gk_batch_v2, not the chain
    dst = DST / name
    if dst.exists():
        return "exists"
    verdict = verify(source / name)
    if verdict is None:
        return "not-a-complete-ten-route-pilot"
    if dry:
        return "would-adopt"
    dst.mkdir(parents=True)
    for f in KEEP:
        if (source / name / f).is_file():
            shutil.copy2(source / name / f, dst / f)
    event("ADOPTED", candidate=name, source=str(source / name), ds_ref=verdict["facts"].get("ds_ref"))
    return "adopted"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        if not source.is_dir():
            continue
        for d in sorted(source.iterdir()):
            if not d.is_dir() or "." not in d.name:
                continue
            if (DST / d.name).exists():
                continue
            result = adopt(d.name, source, args.dry_run)
            if result not in ("exists", "not-a-complete-ten-route-pilot", "skip-k"):
                print(f"{d.name}: {result}")


if __name__ == "__main__":
    main()
