"""Probe v0 end to end: index -> labels -> features -> probes -> table (todos/2026-09-19-probe-v0.md).

  uv run python -m jevdrive.probe_v0                        # whole chain on nuScenes mini
  uv run python -m jevdrive.probe_v0 --version v1.0-trainval
  uv run python -m jevdrive.probe_v0 --steps bench          # extraction speed / VRAM sweep only
Outputs go to $DATA_DIR/runs/probe_v0/<version>/<timestamp>/ (log, results.{csv,md}, timings.json, bench.json).
Features are cached under processed/ and reused unless --force.
"""
import argparse
import json
import logging
import time

from . import features, labels, nuscenes_index, probe
from .common import data_dir, get_logger

STEPS = ("index", "labels", "features", "probe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1.0-mini", choices=list(nuscenes_index.SPLITS))
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma list of {STEPS} and/or bench")
    ap.add_argument("--backbones", default="dinov2,qwen")
    ap.add_argument("--qwen-batch-size", type=int, default=8)
    ap.add_argument("--dino-batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--qwen-width", type=int, default=None, help="resize frames to this width first (default native)")
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    a = ap.parse_args()

    run_dir = data_dir() / "runs" / "probe_v0" / a.version / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log = get_logger("probe_v0")
    logging.getLogger().addHandler(fh := logging.FileHandler(run_dir / "log.txt"))
    fh.setFormatter(logging.getLogger().handlers[0].formatter)
    log.info("args %s -> %s", vars(a), run_dir)

    steps, timings, t_all = a.steps.split(","), {}, time.perf_counter()
    for step in steps:
        t0 = time.perf_counter()
        if step == "index":
            nuscenes_index.run(a.version)
        elif step == "labels":
            labels.run(a.version)
        elif step == "features":
            for b in a.backbones.split(","):
                kw = {"width": a.qwen_width} if b == "qwen" else {}
                bs = a.qwen_batch_size if b == "qwen" else a.dino_batch_size
                timings[f"features_{b}_meta"] = features.run(a.version, b, bs, a.workers, a.force, **kw)
        elif step == "probe":
            probe.run(a.version, run_dir)
        elif step == "bench":
            rows = features.bench(a.version, "qwen", [(1, 0), (1, 8), (4, 8), (8, 16), (16, 16)], width=a.qwen_width)
            rows += features.bench(a.version, "dinov2", [(1, 0), (64, 16)])
            (run_dir / "bench.json").write_text(json.dumps(rows, indent=2))
        else:
            raise ValueError(f"unknown step {step}")
        timings[f"{step}_s"] = time.perf_counter() - t0
        log.info("step %s done in %.1fs", step, timings[f"{step}_s"])
    timings["total_s"] = time.perf_counter() - t_all
    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    log.info("all steps done in %.1fs -> %s", timings["total_s"], run_dir)


if __name__ == "__main__":
    main()
