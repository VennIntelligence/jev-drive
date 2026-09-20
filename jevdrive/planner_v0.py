"""Planner v0 end to end: trajectory targets -> vocabulary -> heads -> tables (todos/2026-09-20-planner-v0.md).

  uv run python -m jevdrive.planner_v0 --version v1.0-trainval       # the whole thing on the cached features
  uv run python -m jevdrive.planner_v0 --steps plan --layers 22      # one layer only, for a quick check
  uv run python -m jevdrive.planner_v0 --steps bench --exit-layer 22 # head latency + truncated extraction
  uv run python -m jevdrive.planner_v0 --horizon 5 --rate 4          # the Waymo E2E target: 5 s at 4 Hz

Features must already be cached by jevdrive.probe_v0 (--steps features); nothing here touches the images
except the extraction benchmark. Outputs go to $DATA_DIR/runs/planner_v0/<version>/<timestamp>/:
log.txt, events.jsonl, tb/ (docs/long-runs.md), results.{csv,md}, per_sample.npz, vocab_K*.npy,
latency.json, bench.json.
"""
import argparse
import json
import time

import torch

from . import common, features, planner, traj
from .runlog import RunLog

STEPS = ("plan", "bench")
LAYERS = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36)  # sparse sweep over the 36 Qwen decoder layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--steps", default="plan", help=f"comma list of {STEPS}")
    ap.add_argument("--horizon", type=float, default=traj.HORIZON, help="seconds of future to predict")
    ap.add_argument("--rate", type=float, default=traj.RATE, help="waypoints per second")
    ap.add_argument("--ks", default="64,256,1024,4096,8192", help="vocabulary sizes to sweep")
    ap.add_argument("--k-ref", type=int, default=1024, help="K used for the layer curve and the head comparison")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)), help="Qwen decoder layers in the sweep")
    ap.add_argument("--qwen-set", default="qwen_w800", help="cached Qwen feature set (input width) to sweep")
    ap.add_argument("--sets", default="", help="regex over '<set>/<array>' overriding --layers / --qwen-set")
    ap.add_argument("--ref-set", default=None, help="skip picking the reference set on the inner split")
    ap.add_argument("--exit-layer", type=int, default=22, help="layer the extraction benchmark truncates at")
    ap.add_argument("--threads", type=int, default=common.n_cpus(), help="torch CPU threads (default: CPU quota)")
    ap.add_argument("--bench-reps", type=int, default=1000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ks = tuple(int(k) for k in a.ks.split(","))
    layers = "|".join(f"L{int(l):02d}" for l in a.layers.split(","))
    pattern = a.sets or rf"({a.qwen_set}/(({layers})_(mean|last)|vit_mean|vis_mean)|dinov2/(cls|patch_mean))"

    rl = RunLog("planner_v0", a.version)
    run_dir, log = rl.dir, rl.log
    log.info("args %s -> %s", vars(a), run_dir)
    rl.event("start", args=vars(a))

    timings, t_all = {}, time.perf_counter()
    for step in a.steps.split(","):
        t0 = time.perf_counter()
        rl.event("step_start", step=step)
        if step == "plan":
            planner.run(a.version, run_dir, rl=rl, horizon=a.horizon, rate=a.rate, ks=ks, k_ref=a.k_ref,
                        pattern=pattern, ref_set=a.ref_set)
        elif step == "bench":
            lat = planner.bench_heads(2560, traj.n_waypoints(a.horizon, a.rate), ks, reps=a.bench_reps)
            (run_dir / "latency.json").write_text(json.dumps(lat, indent=2))
            rows = []
            for n_layers in (36, a.exit_layer):  # full depth vs stopping the decoder at the layer we read
                rows += [{"layers": n_layers, **r} for r in features.bench(
                    a.version, "qwen", [(1, 0), (8, 16)], rl=rl, width=800, layers=[n_layers])]
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
