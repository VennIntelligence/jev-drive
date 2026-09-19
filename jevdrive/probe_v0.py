"""Probe v0 end to end: index -> labels -> features -> probes -> table (todos/2026-09-19-probe-v0.md).

  uv run python -m jevdrive.probe_v0                        # whole chain on nuScenes mini
  uv run python -m jevdrive.probe_v0 --version v1.0-trainval
  uv run python -m jevdrive.probe_v0 --steps bench          # extraction speed / VRAM sweep only
  uv run python -m jevdrive.probe_v0 --version v1.0-trainval --qwen-layers 36 --qwen-width 800,native
                                                            # probe v1: all layers, two input sizes
  uv run python -m jevdrive.probe_v0 --steps probe --probe-sets 'qwen/L3.*'   # probe a subset of the stored sets
Outputs go to $DATA_DIR/runs/probe_v0/<version>/<timestamp>/: log.txt, events.jsonl, tb/ (docs/long-runs.md),
results.{csv,md}, timings.json, bench.json.
Features are cached under processed/ and reused unless --force.
"""
import argparse
import json
import shutil
import time

import torch

from . import common, features, labels, nuscenes_index, probe
from .runlog import RunLog

STEPS = ("index", "labels", "features", "probe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1.0-mini", choices=list(nuscenes_index.SPLITS))
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma list of {STEPS} and/or bench")
    ap.add_argument("--backbones", default="dinov2,qwen")
    ap.add_argument("--qwen-batch-size", type=int, default=8)
    ap.add_argument("--dino-batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=common.n_cpus(), help="DataLoader workers (default: CPU quota)")
    ap.add_argument("--qwen-width", default="native",
                    help="comma list of input widths, one feature set each; 'native' keeps 1600 px (set qwen)")
    ap.add_argument("--qwen-layers", type=int, default=4, help="evenly spaced LLM layers to probe (36 = all)")
    ap.add_argument("--min-free-gb", type=float, default=20, help="refuse to start with less free space on $DATA_DIR")
    ap.add_argument("--probe-sets", default=".*", help="regex over '<set>/<array>' names to probe (baselines always run)")
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    a = ap.parse_args()
    widths = [None if w in ("native", "0") else int(w) for w in a.qwen_width.split(",")]
    torch.set_num_threads(common.n_cpus())

    free_gb = shutil.disk_usage(features.processed_dir(a.version)).free / 2**30
    if free_gb < a.min_free_gb:
        raise SystemExit(f"only {free_gb:.0f} GB free on $DATA_DIR, need {a.min_free_gb:.0f} (--min-free-gb)")

    rl = RunLog("probe_v0", a.version)
    run_dir, log = rl.dir, rl.log
    log.info("args %s -> %s", vars(a), run_dir)
    rl.event("start", args=vars(a))

    steps, timings, t_all = a.steps.split(","), {}, time.perf_counter()
    for step in steps:
        t0 = time.perf_counter()
        rl.event("step_start", step=step)
        if step == "index":
            nuscenes_index.run(a.version)
        elif step == "labels":
            labels.run(a.version)
        elif step == "features":
            for b in a.backbones.split(","):
                for w in widths if b == "qwen" else [None]:
                    kw = {"width": w, "n_layer_probes": a.qwen_layers} if b == "qwen" else {}
                    bs = a.qwen_batch_size if b == "qwen" else a.dino_batch_size
                    meta = features.run(a.version, b, bs, a.workers, a.force, rl, **kw)
                    timings[f"features_{features.set_name(b, **kw)}_meta"] = meta
        elif step == "probe":
            probe.run(a.version, run_dir, rl=rl, pattern=a.probe_sets)
        elif step == "bench":
            rows = features.bench(a.version, "qwen", [(1, 0), (1, 8), (8, 16), (16, 16)], rl=rl,
                                  width=widths[0])
            rows += features.bench(a.version, "dinov2", [(1, 0), (64, 16)], rl=rl)
            (run_dir / "bench.json").write_text(json.dumps(rows, indent=2))
        else:
            raise ValueError(f"unknown step {step}")
        timings[f"{step}_s"] = time.perf_counter() - t0
        log.info("step %s done in %.1fs", step, timings[f"{step}_s"])
        rl.event("step_end", step=step, seconds=timings[f"{step}_s"])
    timings["total_s"] = time.perf_counter() - t_all
    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    log.info("all steps done in %.1fs -> %s", timings["total_s"], run_dir)
    rl.event("end", timings=timings)
    rl.close()


if __name__ == "__main__":
    main()
