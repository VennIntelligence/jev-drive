"""Validate the gate's segment holdout and create nested curve manifests without reading validation outcomes.

Preregistration: todos/2026-10-02-op-adapt-L-gate-and-curve.md.
Run on the box through tmux with CUDA_VISIBLE_DEVICES empty and an explicit CPU affinity.
All outputs are confined to $DATA_DIR/runs/op_adapt_L/gate_curve/prep/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive.common import data_dir  # noqa: E402
from op_adapt_l_train import Log  # noqa: E402

SLICES = ("start", "stop", "turn_onset")
COUNTS = (25, 110, 300, "all")


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def core_set(spec: str) -> set[int]:
    result = set()
    for item in spec.split(","):
        if "-" in item:
            start, end = map(int, item.split("-"))
            result.update(range(start, end + 1))
        else:
            result.add(int(item))
    return result


def resource_check(root: Path):
    grant = dict(line.strip().split("=", 1) for line in (root / "GO").read_text().splitlines()
                 if line.strip() and not line.startswith("#") and "=" in line)
    allowed = core_set(grant["GC_CPUS"])
    actual = set(os.sched_getaffinity(0))
    assert actual <= allowed, f"CPU affinity outside grant: {sorted(actual - allowed)}"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation must hide every GPU"
    assert os.environ.get("OPENBLAS_CORETYPE") == "Haswell", "OpenBLAS compatibility setting missing"
    return {"cpu_affinity": sorted(actual), "visible_gpus": [], "openblas_coretype": "Haswell"}


def events(tab, rows):
    names, seq = tab["name"][rows].astype(str), tab["seq"][rows].astype(str)
    times = np.asarray([int(n.rsplit("-", 1)[1]) * 0.1 for n in names])
    order = np.lexsort((times, seq))
    seq, times = seq[order], times[order]
    return int(1 + ((seq[1:] != seq[:-1]) | (np.diff(times) > 1.0 + 1e-8)).sum()) if len(rows) else 0


def run(root: Path, out: Path, log: Log):
    log.event("start", git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              resources=resource_check(root))
    source = root.parent / "prep"
    tab = dict(np.load(source / "wod.npz", allow_pickle=True))
    val_seq = np.load(source / "wodval.npz", allow_pickle=True)["seq"].astype(str)
    samples = pd.read_parquet(data_dir() / "runs/op_adapt_r2/t/samples/wod.parquet",
                              columns=[f"ctx{k}" for k in range(9)])
    full = (samples.to_numpy() >= 0).all(1)
    assert len(full) == len(tab["seq"]), "Sample and kinematic table lengths differ"
    split, seq = tab["split"].astype(str), tab["seq"].astype(str)
    original_train = np.unique(seq[split == "train"])
    dev_seq = np.random.default_rng(20261002).permutation(original_train)[:round(0.1 * len(original_train))]
    is_gate_dev = (split == "train") & np.isin(seq, dev_seq)
    is_gate_train = (split == "train") & ~np.isin(seq, dev_seq)
    identities = {"train": set(seq[is_gate_train]), "dev": set(seq[is_gate_dev]), "val": set(val_seq)}
    overlaps = {f"gate_{a}_vs_{b}": sorted(identities[a] & identities[b])
                for a, b in (("train", "dev"), ("train", "val"), ("dev", "val"))}
    check = {"overlap_counts": {k: len(v) for k, v in overlaps.items()},
             "overlap_examples": {k: v[:5] for k, v in overlaps.items()},
             "split_revision": "v2: fresh gate holdout carved from original L train; main previously trained on these segments"}
    stationary = np.maximum.reduce([tab[k] for k in ("v0", "vm05", "vm1")]) <= 0.5
    gate_rows = {}
    for label, mask in (("train", is_gate_train), ("dev", is_gate_dev)):
        rows = np.flatnonzero(mask & full & tab["has"] & stationary)
        y = tab["s_start"][rows].astype(np.int8)
        gate_rows[label] = rows
        check[label] = {"rows": len(rows), "segments": len(set(seq[rows])),
                        "positive": int(y.sum()), "negative": int((1 - y).sum()),
                        "creeping": int((~(tab["s_start"][rows] | tab["s_stay"][rows])).sum())}
    dump(out / "split_check.json", check)
    log.event("split_check", **check)
    log.info(f"Gate split check: {json.dumps(check, ensure_ascii=True)}")
    assert not any(overlaps.values()), "Whole-segment holdout overlaps a training or validation split"
    assert check["dev"]["segments"] >= 20, "Gate dev has fewer than 20 independent segments"
    for label in gate_rows:
        assert min(check[label]["positive"], check[label]["negative"]) > 0, f"Gate {label} lacks a label class"
    np.savez(out / "gate_rows.npz", **gate_rows)
    log.scalar("prep/gate_dev_segments", check["dev"]["segments"], 0)
    manifest, counts = {}, []
    for si, s in enumerate(tqdm(SLICES, desc="Curve manifests")):
        pool = np.flatnonzero((split == "train") & full & tab["has"] & tab[f"s_{s}"])
        ordered = np.random.default_rng(20261002 + si).permutation(np.unique(seq[pool]))
        assert len(ordered) >= 300, f"Insufficient segments for {s}"
        manifest[s] = {}
        for n in COUNTS:
            chosen = ordered if n == "all" else ordered[:n]
            rows = pool[np.isin(seq[pool], chosen)]
            manifest[s][str(n)] = {"segments": chosen.tolist(), "rows": rows.tolist()}
            counts.append({"slice": s, "size": n, "segments": len(chosen), "frames": len(rows),
                           "events": events(tab, rows)})
        assert set(manifest[s]["25"]["segments"]) <= set(manifest[s]["110"]["segments"])
        assert set(manifest[s]["110"]["segments"]) <= set(manifest[s]["300"]["segments"])
        assert set(manifest[s]["300"]["rows"]) <= set(manifest[s]["all"]["rows"])
    dump(out / "manifest.json", manifest)
    digest = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    with (out / "counts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    dump(out / "manifest_checksum.json", {"sha256": digest, "path": "manifest.json"})
    log.event("end", status="passed", manifest_sha256=digest)
    log.info("Preparation passed; no validation outcomes or feature values were read")
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def main():
    root = data_dir() / "runs/op_adapt_L/gate_curve"
    out = root / "prep-v2"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists() or (out / "ERROR").exists():
        raise SystemExit("Preparation already has a sentinel; inspect it before creating a new attempt")
    log = Log(out)
    assert log.tb is not None, "TensorBoard writer is required"
    try:
        run(root, out, log)
    except BaseException as error:
        (out / "ERROR").write_text(traceback.format_exc())
        log.event("error", error=str(error))
        log.info(f"Preparation failed: {error}")
        raise
    finally:
        log.tb.close()


if __name__ == "__main__":
    main()
