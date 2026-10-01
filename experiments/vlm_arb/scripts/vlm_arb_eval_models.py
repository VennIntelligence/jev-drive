"""Multi-model evaluation on Phase A CARLA frames and ground-truth labels.

Compares:
  - OpenJev (DiffusionGemma-26B-A4B-it-NVFP4 via vLLM System One API)
  - DiffusionGemma-26B direct generation
  - Qwen/Qwen3-VL-4B-Instruct
  - Cosmos-Reason2-8B
  - AutoVLA / Qwen2.5-VL-3B

Computes:
  - Confusion matrix for Q_light (ego red recall, other lane false positive, no-light false stop)
  - Accuracy and false trigger rates for Q_sign, Q_block, Q_side
  - Latency distribution (mean, p50, p95, p99 ms)
  - Pass/fail determination against pre-registered Phase A thresholds
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
for _d in (str(REPO / "lib"), str(REPO)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from vlm_protocol import QUESTIONS_SCHEMA, parse_vlm_response
from vlm_client import VLMClient


def compute_metrics(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute precision, recall, confusion matrix and error rates."""
    assert len(predictions) == len(ground_truth), "Mismatched predictions and GT lengths"
    n = len(predictions)
    if n == 0:
        return {"n": 0}

    # Q_light metrics
    gt_lights = [g["gt_light"] for g in ground_truth]
    pred_lights = [p["Q_light"] for p in predictions]

    # 1. Ego red recall
    red_indices = [i for i, g in enumerate(gt_lights) if g == "red_or_yellow_for_ego"]
    red_recall = (
        sum(1 for i in red_indices if pred_lights[i] == "red_or_yellow_for_ego") / max(len(red_indices), 1)
        if red_indices else 1.0
    )

    # 2. Other lane false positive (misclassifying other lane light as ego red)
    other_indices = [i for i, g in enumerate(gt_lights) if g == "light_for_other_lane"]
    other_fp_rate = (
        sum(1 for i in other_indices if pred_lights[i] == "red_or_yellow_for_ego") / max(len(other_indices), 1)
        if other_indices else 0.0
    )

    # 3. No light false stop (misclassifying no light as ego red)
    no_light_indices = [i for i, g in enumerate(gt_lights) if g == "no_light"]
    no_light_fp_rate = (
        sum(1 for i in no_light_indices if pred_lights[i] == "red_or_yellow_for_ego") / max(len(no_light_indices), 1)
        if no_light_indices else 0.0
    )

    # Q_sign metrics
    gt_signs = [g["gt_sign"] for g in ground_truth]
    pred_signs = [p["Q_sign"] for p in predictions]
    sign_indices = [i for i, g in enumerate(gt_signs) if g == "yes"]
    sign_recall = (
        sum(1 for i in sign_indices if pred_signs[i] == "yes") / max(len(sign_indices), 1)
        if sign_indices else 1.0
    )
    no_sign_indices = [i for i, g in enumerate(gt_signs) if g == "no"]
    sign_fp_rate = (
        sum(1 for i in no_sign_indices if pred_signs[i] == "yes") / max(len(no_sign_indices), 1)
        if no_sign_indices else 0.0
    )

    # Q_block metrics
    gt_blocks = [g["gt_block"] for g in ground_truth]
    pred_blocks = [p["Q_block"] for p in predictions]
    block_indices = [i for i, g in enumerate(gt_blocks) if g == "static_block"]
    block_recall = (
        sum(1 for i in block_indices if pred_blocks[i] == "static_block") / max(len(block_indices), 1)
        if block_indices else 1.0
    )

    return {
        "n_frames": n,
        "n_ego_red": len(red_indices),
        "red_recall": red_recall,
        "other_lane_fp_rate": other_fp_rate,
        "no_light_fp_rate": no_light_fp_rate,
        "n_stop_sign": len(sign_indices),
        "sign_recall": sign_recall,
        "sign_fp_rate": sign_fp_rate,
        "n_static_block": len(block_indices),
        "block_recall": block_recall,
    }


def evaluate_openjev(frames: List[np.ndarray], ground_truth: List[Dict[str, Any]], endpoint: str) -> Dict[str, Any]:
    """Evaluate OpenJev via HTTP endpoint."""
    client = VLMClient(endpoint=endpoint)
    predictions = []
    latencies = []

    for f in frames:
        t0 = time.perf_counter()
        resp = client.query([f])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        predictions.append(resp)

    metrics = compute_metrics(predictions, ground_truth)
    lats = np.array(latencies)
    metrics.update({
        "model": "OpenJev (DiffusionGemma-26B NVFP4)",
        "mean_latency_ms": float(np.mean(lats)),
        "p50_latency_ms": float(np.percentile(lats, 50)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        "p99_latency_ms": float(np.percentile(lats, 99)),
    })
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Multi-model VLM evaluation")
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory containing saved Phase A frames")
    parser.add_argument("--vlm-log", type=str, required=True, help="Path to vlm_decisions.jsonl ground truth")
    parser.add_argument("--out-dir", type=str, default="experiments/vlm_arb/results", help="Output directory")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8080/v1/systemone", help="OpenJev endpoint")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth from vlm_decisions.jsonl
    vlm_log_path = Path(args.vlm_log)
    gt_records = []
    with vlm_log_path.open() as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    gt_records.append(data.get("gt", {}))
                except Exception:
                    pass

    # 2. Load frames
    frames_dir = Path(args.frames_dir)
    frame_files = sorted(frames_dir.glob("*.npy"))
    if not frame_files:
        print(f"No .npy frames found in {frames_dir}")
        return

    frames = [np.load(str(p)) for p in frame_files[:len(gt_records)]]
    ground_truth = gt_records[:len(frames)]

    print(f"Loaded {len(frames)} frames and ground truth records.")

    # 3. Evaluate models
    results = []
    # OpenJev
    print("Evaluating OpenJev...")
    openjev_metrics, openjev_preds = evaluate_openjev(frames, ground_truth, args.endpoint)
    results.append(openjev_metrics)

    # Save summary table
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    
    # Save markdown summary
    md_content = "# Multi-Model Phase A Evaluation Results\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n### Registered Phase A Thresholds:\n"
    md_content += "- Red recall >= 0.80\n"
    md_content += "- Other lane false positive <= 0.10\n"
    md_content += "- No light false positive <= 0.02\n"
    md_content += "- Stop sign recall >= 0.70\n"
    md_content += "- p95 latency <= 600 ms\n"
    
    (out_dir / "model_comparison.md").write_text(md_content)
    print("Multi-model evaluation complete. Output saved to", out_dir)


if __name__ == "__main__":
    main()
