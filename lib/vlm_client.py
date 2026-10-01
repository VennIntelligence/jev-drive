"""Client for OpenJev System One server and VLM endpoints.

Supports synchronous and non-blocking calls, base64 encoding of CARLA native camera frames,
and fallbacks.
"""
import base64
import json
import time
import urllib.request
import urllib.error
from io import BytesIO
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

from vlm_protocol import QUESTIONS_SCHEMA, parse_vlm_response


class VLMClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:8080/v1/systemone", timeout_s: float = 2.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def encode_frame(self, frame_np: np.ndarray, quality: int = 85) -> str:
        """Encode an RGB numpy array (H, W, 3) to a base64 data URI."""
        img = Image.fromarray(frame_np)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def query(self, frames: List[np.ndarray], state_desc: str = "Driving in CARLA autonomous mode") -> Dict[str, Any]:
        """Query the System One decision endpoint with the given camera frames."""
        images = [self.encode_frame(f) for f in frames]
        payload = {
            "state": state_desc,
            "images": images,
            "questions": QUESTIONS_SCHEMA,
            "steps": 1,
            "samples": 1
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                latency_ms = (time.perf_counter() - t0) * 1000
                parsed = parse_vlm_response(raw)
                parsed["latency_ms"] = latency_ms
                parsed["ok"] = True
                parsed["raw"] = raw
                return parsed
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "ok": False,
                "error": str(e),
                "latency_ms": latency_ms,
                "Q_light": "no_light",
                "Q_light_conf": 0.0,
                "Q_sign": "no",
                "Q_sign_val": 0.0,
                "Q_sign_conf": 0.0,
                "Q_block": "clear",
                "Q_block_conf": 0.0,
                "Q_side": "none_free",
                "Q_side_conf": 0.0
            }


class DummyVLMClient(VLMClient):
    """Fallback client for unit tests and local dry-runs."""
    def query(self, frames: List[np.ndarray], state_desc: str = "") -> Dict[str, Any]:
        return {
            "ok": True,
            "latency_ms": 50.0,
            "Q_light": "no_light",
            "Q_light_conf": 1.0,
            "Q_sign": "no",
            "Q_sign_val": 0.0,
            "Q_sign_conf": 1.0,
            "Q_block": "clear",
            "Q_block_conf": 1.0,
            "Q_side": "none_free",
            "Q_side_conf": 1.0
        }
