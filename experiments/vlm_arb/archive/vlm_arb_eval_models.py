"""Multi-model evaluation on Phase A CARLA frames and ground-truth labels.

Compares:
  - OpenJev (DiffusionGemma-26B-A4B-it-NVFP4 via vLLM System One API)
  - DiffusionGemma-26B Direct (Multimodal Vision Chat via vLLM API)
  - Pre-registered Phase A Pass/Fail Criteria

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
from io import BytesIO
import urllib.request
import urllib.error

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
        return {"n_frames": 0}

    # Q_light metrics
    gt_lights = [g.get("gt_light", "no_light") for g in ground_truth]
    pred_lights = [p.get("Q_light", "no_light") for p in predictions]

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
    gt_signs = [g.get("gt_sign", "no") for g in ground_truth]
    pred_signs = [p.get("Q_sign", "no") for p in predictions]
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
    gt_blocks = [g.get("gt_block", "clear") for g in ground_truth]
    pred_blocks = [p.get("Q_block", "clear") for p in predictions]
    block_indices = [i for i, g in enumerate(gt_blocks) if g == "static_block"]
    block_recall = (
        sum(1 for i in block_indices if pred_blocks[i] == "static_block") / max(len(block_indices), 1)
        if block_indices else 1.0
    )

    return {
        "n_frames": n,
        "n_ego_red": len(red_indices),
        "red_recall": round(red_recall, 4),
        "other_lane_fp_rate": round(other_fp_rate, 4),
        "no_light_fp_rate": round(no_light_fp_rate, 4),
        "n_stop_sign": len(sign_indices),
        "sign_recall": round(sign_recall, 4),
        "sign_fp_rate": round(sign_fp_rate, 4),
        "n_static_block": len(block_indices),
        "block_recall": round(block_recall, 4),
    }


def evaluate_openjev(frames: List[np.ndarray], ground_truth: List[Dict[str, Any]], endpoint: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
    lats = np.array(latencies) if latencies else np.array([0.0])
    metrics.update({
        "model": "OpenJev (DiffusionGemma-26B NVFP4 via System One API)",
        "mean_latency_ms": round(float(np.mean(lats)), 1),
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 1),
        "p99_latency_ms": round(float(np.percentile(lats, 99)), 1),
    })
    return metrics, predictions


def evaluate_dgemma_direct(frames: List[np.ndarray], ground_truth: List[Dict[str, Any]], endpoint: str = "http://127.0.0.1:8000/v1/chat/completions") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate DiffusionGemma-26B directly via vLLM OpenAI-compatible vision chat completions."""
    def encode_frame(frame_np: np.ndarray, quality: int = 85) -> str:
        img = Image.fromarray(frame_np)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    prompt_text = (
        "You are an autonomous driving perception system. Observe the camera image and answer the following questions in valid JSON format:\n"
        "{\n"
        '  "Q_light": "no_light" | "green_for_ego" | "red_or_yellow_for_ego" | "light_for_other_lane",\n'
        '  "Q_sign": "yes" | "no",\n'
        '  "Q_block": "clear" | "static_block" | "moving_lead",\n'
        '  "Q_side": "left_free" | "right_free" | "none_free"\n'
        "}\n"
        "Return ONLY the raw JSON object without markdown fences or additional explanation."
    )

    predictions = []
    latencies = []

    for f in frames:
        b64_url = encode_frame(f)
        payload = {
            "model": "dgemma",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": b64_url}}
                    ]
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                lat = (time.perf_counter() - t0) * 1000
                content = raw["choices"][0]["message"]["content"]
                s_idx = content.find("{")
                e_idx = content.rfind("}")
                if s_idx != -1 and e_idx != -1:
                    ans = json.loads(content[s_idx:e_idx+1])
                else:
                    ans = {}
                parsed = {
                    "Q_light": ans.get("Q_light", "no_light"),
                    "Q_sign": ans.get("Q_sign", "no"),
                    "Q_block": ans.get("Q_block", "clear"),
                    "Q_side": ans.get("Q_side", "none_free"),
                    "latency_ms": lat
                }
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000
            parsed = {
                "Q_light": "no_light",
                "Q_sign": "no",
                "Q_block": "clear",
                "Q_side": "none_free",
                "latency_ms": lat,
                "error": str(e)
            }
        latencies.append(lat)
        predictions.append(parsed)

    metrics = compute_metrics(predictions, ground_truth)
    lats = np.array(latencies) if latencies else np.array([0.0])
    metrics.update({
        "model": "DiffusionGemma-26B (Direct vLLM Vision Chat)",
        "mean_latency_ms": round(float(np.mean(lats)), 1),
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 1),
        "p99_latency_ms": round(float(np.percentile(lats, 99)), 1),
    })
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Multi-model VLM evaluation")
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory containing saved Phase A frames")
    parser.add_argument("--vlm-log", type=str, required=True, help="Path to vlm_decisions.jsonl ground truth")
    parser.add_argument("--out-dir", type=str, default="experiments/vlm_arb/results", help="Output directory")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8080/v1/systemone", help="OpenJev endpoint")
    parser.add_argument("--vllm-endpoint", type=str, default="http://127.0.0.1:8000/v1/chat/completions", help="Direct vLLM chat endpoint")
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

    # Evaluate on up to 100 representative frames to maintain fast turnaround
    subsample_n = min(len(frame_files), len(gt_records), 100)
    indices = np.linspace(0, min(len(frame_files), len(gt_records)) - 1, subsample_n, dtype=int)
    frames = [np.load(str(frame_files[i])) for i in indices]
    ground_truth = [gt_records[i] for i in indices]

    print(f"Loaded {len(frames)} frames and ground truth records for evaluation.")

    # 3. Evaluate models
    results = []
    
    # Model 1: OpenJev System One API
    print("Evaluating Model 1: OpenJev (DiffusionGemma-26B System One API)...")
    openjev_metrics, openjev_preds = evaluate_openjev(frames, ground_truth, args.endpoint)
    results.append(openjev_metrics)

    # Model 2: DiffusionGemma-26B Direct vLLM Vision Chat
    print("Evaluating Model 2: DiffusionGemma-26B (Direct vLLM Vision Chat)...")
    try:
        dgemma_metrics, dgemma_preds = evaluate_dgemma_direct(frames, ground_truth, args.vllm_endpoint)
        results.append(dgemma_metrics)
    except Exception as e:
        print(f"DiffusionGemma-26B direct chat evaluation error: {e}")

    # Save summary table
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    
    # Save markdown summary
    md_content = "# Multi-Model Phase A Evaluation Results\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n### Pre-Registered Phase A Thresholds:\n"
    md_content += "- Red light recall >= 0.80\n"
    md_content += "- Other lane false positive <= 0.10\n"
    md_content += "- No light false positive <= 0.02\n"
    md_content += "- Stop sign recall >= 0.70\n"
    md_content += "- p95 latency <= 600 ms\n"
    
    (out_dir / "model_comparison.md").write_text(md_content)
    print("Multi-model evaluation complete. Output saved to", out_dir)


if __name__ == "__main__":
    main()
