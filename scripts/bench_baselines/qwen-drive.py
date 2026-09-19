"""Latency of Qwen-Drive-1.0-4B as released (QwenLM/Qwen-Drive-1.0 @ 28091c1), batch 1.

Run on the box from the qwen-drive venv (see setup_qwen_drive.sh):
  $DATA_DIR/envs/qwen-drive/bin/python scripts/bench_baselines/qwen_drive.py [--iters 200]
Input: WOD-E2E demo scene 1 shipped with the repo (3 cameras x 4 frames, JPEG on disk, ego history, route).
Timed span = model.run(mode, scene): JPEG decode + resize + tokenize (CPU), vision encoder, prefill,
reasoning decode (reasoning mode only), flow-matching Planning Expert steps, copy of the trajectory to CPU.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _bench import bench  # noqa: E402

from qwen_drive import InferenceMode, QwenDriveForPlanning  # noqa: E402
from qwen_drive.benchmarks import read_scene_file  # noqa: E402

SRC = Path(os.environ["DATA_DIR"]) / "third_party" / "qwen-drive"

p = argparse.ArgumentParser()
p.add_argument("--scene-index", type=int, default=1)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--iters", type=int, default=200)
p.add_argument("--attn", default="flash_attention_2")
p.add_argument("--only", default=None, help="run one tag only")
args = p.parse_args()

root = Path(os.environ["DATA_DIR"]) / "models" / "Qwen-Drive-1.0-4B"  # ModelScope copy, see download_models.sh
model = QwenDriveForPlanning.from_pretrained(
    str(root), planner=str(root / "planner-sft"), dtype=torch.bfloat16, attn_implementation=args.attn
).to("cuda").eval()
samples = list(read_scene_file(str(SRC / "data/demo/planning_scenes.jsonl"), image_root=str(SRC / "data/demo"),
                               num_history_points=model.config.num_history_points, limit=args.scene_index + 1))
scene = samples[args.scene_index].scene
tok = model.processor.tokenizer
n_prompt = {m: int(model.processor(scene, with_reasoning=m == "reasoning", device="cpu")["input_ids"].shape[1])
            for m in ("direct", "reasoning")}
print(f"scene {samples[args.scene_index].token}: {scene.num_camera_frames} frames x {len(scene.views)} views, "
      f"prompt tokens {n_prompt}")
cfg = model.config
meta = {"source": "QwenLM/Qwen-Drive-1.0@28091c1, Qwen/Qwen-Drive-1.0-4B (ModelScope copy of HF rev 2848408)",
        "dtype": "bfloat16", "attn": args.attn, "scene": samples[args.scene_index].token,
        "flow_steps": cfg.num_inference_steps, "max_reasoning_tokens": cfg.max_reasoning_tokens}

# (tag, planner head, mode, num_samples). planner-rl is meant for reasoning mode only (model card).
runs = [
    ("direct-sft-n1", "planner-sft", InferenceMode.DIRECT_PLANNING, 1),
    ("direct-sft-n6", "planner-sft", InferenceMode.DIRECT_PLANNING, 6),
    ("reasoning-rl-n1", "planner-rl", InferenceMode.REASONING_PLANNING, 1),
    ("reasoning-rl-n6", "planner-rl", InferenceMode.REASONING_PLANNING, 6),
    ("reasoning-sft-n1", "planner-sft", InferenceMode.REASONING_PLANNING, 1),
]
loaded = "planner-sft"
for tag, planner, mode, n in runs:
    if args.only and tag != args.only:
        continue
    if planner != loaded:
        model.load_planner(str(root / planner))
        loaded = planner
    res = model.run(mode, scene=scene, num_samples=n)
    print(f"[{tag}] trajectories {res.trajectories.shape}, reasoning: {res.reasoning!r}")
    bench("qwen-drive-1.0-4b", tag, lambda: model.run(mode, scene=scene, num_samples=n),
          warmup=args.warmup, iters=args.iters,
          meta={**meta, "planner": planner, "mode": mode.value, "num_samples": n,
                "prompt_tokens": n_prompt["reasoning" if mode is InferenceMode.REASONING_PLANNING else "direct"],
                "example_reasoning": res.reasoning},
          extra=lambda r: {"reasoning_tokens": len(tok.encode(r.reasoning)) if r.reasoning else 0})
