"""WOMD interactive TFRecords -> connectivity-enriched GPUDrive JSONs + per-ego manifest (statepol exam).

Runs in the conversion env (ScenarioMax + TF 2.11). Reuses BehaviorBench's extraction helpers
(data_utils/womd/extract_interactive_benchmark.py, steps 1-3); the .bin files are written later by
statepol_build.py in the policy env. A shim maps `waymo_open_dataset.protos.scenario_pb2` onto
ScenarioMax's vendored proto so the helpers run without the waymo-open-dataset wheel.
Usage: python statepol_convert.py --tfrecords DIR --json DIR --enriched DIR --workers 64
(--json must already hold ScenarioMax output: scenariomax-convert --target_format gpudrive)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

BB = Path(os.environ.get("BB_ROOT", Path.home() / "data/third_party/statepol/behavior-bench"))


def make_shim(root: Path):
    pkg = root / "waymo_open_dataset" / "protos"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "waymo_open_dataset" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "scenario_pb2.py").write_text(
        "from scenariomax.raw_to_unified.datasets.waymo.waymo_protos.scenario_pb2 import *  # noqa\n")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecords", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    shim = make_shim(Path(a.enriched).parent / "_shim")
    os.environ["PYTHONPATH"] = f"{shim}:{os.environ.get('PYTHONPATH', '')}"  # for helper subprocesses
    sys.path[:0] = [str(shim), str(BB)]
    from data_utils.womd.extract_interactive_benchmark import (
        extract_connectivity_parallel, filter_interactive_jsons, enrich_jsons)

    conn = extract_connectivity_parallel(a.tfrecords, num_workers=a.workers)
    kept = filter_interactive_jsons(a.json)
    results = enrich_jsons(kept, conn, a.enriched, a.workers)
    with open(Path(a.enriched) / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_filename", "ego_agent_idx", "scenario_id"])
        for (jf, sid, agents), (ok, _) in zip(kept, results):
            if ok:
                for track_index, _ in agents:
                    w.writerow([jf.name, track_index, sid])
    print("manifest rows written", flush=True)


if __name__ == "__main__":
    main()
