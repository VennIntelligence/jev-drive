"""AutoVLA on WOD-E2E val: how often does the adaptive model actually reason, and what does it cost?

The single-scene row in docs/baselines.md only measures the fast path. This runs the released checkpoint over
frames sampled by our own subset definitions (`autovla_waymo_sample.py`: straight / turning / pre-onset, 50 each,
one per sequence, stated seed) and reports latency as a distribution, split by whether the model reasoned.

  $DATA_DIR/envs/autovla/bin/python scripts/bench_baselines/autovla_waymo.py [--warmup 20]

Everything about the model is as released: their prompt (the adaptive CoT one), their generation settings,
batch 1, one timed call per frame, `torch.cuda.synchronize()` around each. Frames differ, so the spread here is
input-dependence, not run-to-run noise; the single-scene rows in docs/baselines.md give the latter.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
import types
from pathlib import Path

import torch
import yaml

DATA = Path(os.environ["DATA_DIR"])
SRC = DATA / "third_party" / "autovla"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))
from _bench import pct, run_dir  # noqa: E402

stub = types.ModuleType("models.utils.score")  # navsim + nuplan, only the GRPO reward uses it
stub.PDM_Reward = stub.TrajectorySampling = stub.Trajectory = object
sys.modules["models.utils.score"] = stub
from models.autovla import AutoVLA  # noqa: E402

ckpt = sorted((DATA / "cache/huggingface/hub/models--Zewei-Zhou--AutoVLA/snapshots").glob("*/AutoVLA_PDMS_89.ckpt"))
p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, default=DATA / "processed/autovla_waymo/manifest.json")
p.add_argument("--checkpoint", default=str(ckpt[0]) if ckpt else None, required=not ckpt)
p.add_argument("--base-model", default=str(DATA / "models/Qwen2.5-VL-3B-Instruct"))
p.add_argument("--config", default=str(SRC / "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--think-tokens", type=int, default=32,
               help="a think block longer than this counts as the model choosing to reason")
args = p.parse_args()

frames = json.loads(args.manifest.read_text())
config = yaml.safe_load(open(args.config))
config["model"]["pretrained_model_path"] = args.base_model
config["model"]["codebook_cache_path"] = str(SRC / config["model"]["codebook_cache_path"])

model = AutoVLA(config, device="cuda")
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict({k.replace("autovla.", ""): v for k, v in state.items()}, strict=False)
del state
model.eval()
model.use_cot = True  # the released adaptive prompt: the model decides whether to reason
tok = model.processor.tokenizer


def features(f):
    return {"images": {k: v for k, v in f["images"].items()},
            "vehicle_velocity": f["velocity"], "vehicle_acceleration": f["acceleration"],
            "driving_command": f["command"], "sensor_data_path": None}


def think_block(text):
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    return m.group(1).strip() if m else ""


with torch.no_grad():
    for f in frames[:args.warmup]:
        model.predict(features(f))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i, f in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        traj, text = model.predict(features(f))
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        think = think_block(text)
        rows.append({"name": f["name"], "stratum": f["stratum"], "intent": f["intent"], "speed": f["speed"],
                     "yaw_rate_deg": f["yaw_rate_deg"], "ms": ms, "think": think,
                     "think_tokens": len(tok.encode(think)) if think else 0,
                     "output_tokens": len(tok.encode(text)), "traj_points": int(traj.shape[0])})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(frames)}  {ms:7.1f} ms  think {rows[-1]['think_tokens']:4d} tok  {f['stratum']}",
                  flush=True)


def stats(sel):
    t = [r["ms"] for r in sel]
    if not t:
        return {"n": 0}
    return {"n": len(t), "mean_ms": st.fmean(t), "p50_ms": pct(t, .5), "p95_ms": pct(t, .95),
            "p99_ms": pct(t, .99), "min_ms": min(t), "max_ms": max(t),
            "think_rate": st.fmean(r["think_tokens"] > args.think_tokens for r in sel),
            "think_tokens_mean": st.fmean(r["think_tokens"] for r in sel),
            "think_tokens_p95": pct([r["think_tokens"] for r in sel], .95),
            "output_tokens_mean": st.fmean(r["output_tokens"] for r in sel)}


thinks = [r for r in rows if r["think_tokens"] > args.think_tokens]
summary = {
    "model": "autovla", "tag": "waymo-val-strata", "frames": len(rows), "warmup": args.warmup,
    "source": "ucla-mobility/AutoVLA@ba34eed, AutoVLA_PDMS_89.ckpt, released adaptive-CoT prompt",
    "sampling": "jevdrive.waymo.subsets on WOD-E2E val, 50 frames per stratum, one per sequence, seed 0",
    "think_threshold_tokens": args.think_tokens,
    "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    "all": stats(rows), "think": stats(thinks), "no_think": stats([r for r in rows if r not in thinks]),
    "by_stratum": {s: stats([r for r in rows if r["stratum"] == s]) for s in ("straight_yaw", "turn_yaw", "pre_onset")},
    "by_intent": {i: stats([r for r in rows if r["intent"] == i]) for i in {r["intent"] for r in rows}},
    "distinct_short_blocks": sorted({r["think"] for r in rows if r["think_tokens"] <= args.think_tokens}),
}
out = run_dir("autovla", "waymo-val-strata")
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "frames.json").write_text(json.dumps(rows, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "distinct_short_blocks"}, indent=2))
print(f"wrote {out}")
