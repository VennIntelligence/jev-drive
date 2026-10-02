"""Client of the OpenJev System One decision endpoint (Python 3.8, route process env or analysis env).

One call = one HTTP request with JPEG frames and a question schema. A failed request returns {"ok": False, ...} with
no answers: the caller must not treat it as an answer (the arbitration layer lets the previous answer age out).
"""
import base64
import json
import time
import urllib.request
from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image

try:
    from .vlm_protocol import CROP_BOX, QUESTIONS_SCHEMA, VARIANTS, parse_vlm_response
except (ImportError, ValueError):
    from vlm_protocol import CROP_BOX, QUESTIONS_SCHEMA, VARIANTS, parse_vlm_response

ENDPOINT = "http://127.0.0.1:8080/v1/systemone"


def jpeg(frame, quality: int = 85) -> bytes:
    """RGB array or PIL image -> JPEG bytes."""
    img = frame if isinstance(frame, Image.Image) else Image.fromarray(np.ascontiguousarray(frame))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def crop_wide(jpg: bytes) -> bytes:
    """Upper-centre crop of the wide frame (CROP_BOX), upsampled 2x: where a light facing the ego lane sits."""
    img = Image.open(BytesIO(jpg)).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = CROP_BOX
    c = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return jpeg(c.resize((c.width * 2, c.height * 2), Image.BICUBIC))


class VLMClient:
    def __init__(self, endpoint: str = ENDPOINT, model: str = "openjev-latest", timeout_s: float = 5.0):
        self.endpoint, self.model, self.timeout_s = endpoint, model, timeout_s

    def ask(self, jpgs: Dict[str, bytes], variant: str = "base", only_light: bool = False, state: str = "") -> Dict[str, Any]:
        """`jpgs`: {"wide": jpeg bytes, "road": jpeg bytes}. `only_light`: send the Q_light question alone."""
        v = VARIANTS[variant]
        images = [jpgs[c] for c in v["cams"] if c in jpgs]
        if v["crop"] and "wide" in jpgs:
            images.append(crop_wide(jpgs["wide"]))
        questions = {"Q_light": v["light"]} if only_light else dict(QUESTIONS_SCHEMA, Q_light=v["light"])
        payload = {"model": self.model, "state": state or "Driving in CARLA autonomous mode", "questions": questions,
                   "images": ["data:image/jpeg;base64," + base64.b64encode(b).decode("ascii") for b in images],
                   "steps": 1, "samples": 1}
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = parse_vlm_response(json.loads(resp.read().decode("utf-8")))
            out.update(ok=True)
        except Exception as e:  # noqa: BLE001 - a failed request is data, not a crash of the route
            out = {"ok": False, "error": str(e)[:200]}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.endpoint.rsplit("/v1/", 1)[0] + "/health", timeout=2.0) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False
