"""Latency client for openjev (razorback16/openjev @ 91d5005) on DiffusionGemma-26B-A4B NVFP4. Started by openjev.sh.

Serial requests (concurrency 1) to POST /v1/systemone on localhost; timed span = one HTTP round trip:
JSON + base64 image decode, Gemma image preprocessing, vision tower, prefill of the canvas, one read-only
denoise step (plus up to 3 automatic re-reads when an answer is uncertain), answer post-processing.
Image payloads are prepared before timing. "fresh" variants change one pixel per request so vLLM's
prefix/multimodal caches cannot hit (a new camera frame each request, as in driving); "cached" repeats one request.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _bench import pct, run_dir  # noqa: E402

FRAMES = Path(os.environ["DATA_DIR"]) / "third_party/qwen-drive/data/demo/frames"
CAMS = ["scene1_3.jpg", "scene1_7.jpg", "scene1_11.jpg"]  # WOD-E2E demo scene 1 at t=0: FRONT, FRONT_LEFT, FRONT_RIGHT
DRIVE_Q = {
    "lateral": {"type": "choice", "instructions": "Which lateral action should the ego vehicle take next?",
                "criteria": {"keep_lane": "follow the current lane", "turn_left": "turn left", "turn_right": "turn right",
                             "change_left": "change to the left lane", "change_right": "change to the right lane"}},
    "longitudinal": {"type": "choice", "instructions": "Which longitudinal action should the ego vehicle take next?",
                     "criteria": {"accelerate": "speed up", "keep": "keep speed", "decelerate": "slow down", "stop": "stop"}},
    "yield": {"type": "noul", "instructions": "Another road user requires the ego vehicle to yield or stop"},
}
STATE = "Cameras: front, front-left, front-right, current frame. Ego speed 8.2 m/s. Route command: go straight."
README_Q = {  # the openjev README benchmark: 3 text questions
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
}


def jpeg_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_sets(n: int, cams: list[str], fresh: bool) -> list[list[str]]:
    base = [Image.open(FRAMES / c).convert("RGB") for c in cams]
    if not fresh:
        urls = [jpeg_url(im) for im in base]
        return [urls] * n
    out = []
    for i in range(n):
        ims = [im.copy() for im in base]
        for im in ims:
            im.putpixel((i % im.width, i // im.width), (i * 37 % 256, i * 91 % 256, i * 53 % 256))
        out.append([jpeg_url(im) for im in ims])
    return out


def gpu_mem_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) / 1024


def run(client, tag, bodies, warmup, meta):
    import json
    times, usage = [], []
    for i, body in enumerate(bodies):
        t0 = time.perf_counter()
        r = client.post("/v1/systemone", json=body)
        dt = (time.perf_counter() - t0) * 1e3
        r.raise_for_status()
        if i >= warmup:
            times.append(dt)
            usage.append(r.json()["usage"])
    last = r.json()
    s = {"model": "openjev-diffusiongemma-26b-a4b-nvfp4", "tag": tag, "warmup": warmup, "iters": len(times),
         "mean_ms": statistics.fmean(times), "p50_ms": pct(times, .5), "p95_ms": pct(times, .95),
         "p99_ms": pct(times, .99), "min_ms": min(times), "max_ms": max(times),
         "input_tokens_mean": statistics.fmean(u["input_tokens"] for u in usage),
         "gpu_mem_used_gb_nvidia_smi": gpu_mem_gb(), "example_answers": last["answers"], **meta}
    d = run_dir("openjev", tag)
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    (d / "times_ms.json").write_text(json.dumps({"times_ms": times, "usage": usage}))
    print(json.dumps(s, indent=2), f"\nwrote {d}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8080")
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--iters", type=int, default=200)
a = p.parse_args()
n = a.warmup + a.iters
meta = {"source": "razorback16/openjev@91d5005, razorback16/vllm@9bbf741, nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "precision": "NVFP4 weights", "concurrency": 1}
with httpx.Client(base_url=a.url, timeout=120) as c:
    run(c, "text-readme-3q", [{"model": "openjev-latest", "state": f"[{i}] Everything is down and we have a demo "
        "with our biggest client at noon.", "questions": README_Q} for i in range(n)], a.warmup, meta)
    for cams, name in ((CAMS[:1], "1cam"), (CAMS, "3cam")):
        for fresh in (True, False):
            tag = f"img-{name}-drive-3q-{'fresh' if fresh else 'cached'}"
            bodies = [{"model": "openjev-latest", "state": STATE, "questions": DRIVE_Q, "images": imgs}
                      for imgs in image_sets(n, cams, fresh)]
            run(c, tag, bodies, a.warmup, meta)
