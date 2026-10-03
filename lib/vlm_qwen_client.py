"""Client of experiments/vlm_arb/scripts/vlm_qwen_server.py (Python 3.8, route process env or analysis env).

One call = one HTTP request with the JPEG bytes of the wide and the road frame. A failed request returns
{"ok": False, ...} with no answer: the caller must not treat it as one (the arbitration layer lets the previous
answer age out). The server of card g listens on PORT0 + g.
"""
import json
import time
import urllib.request
from typing import Any, Dict

PORT0 = 8200
LIGHTS = ["no_light", "red_or_yellow_for_ego", "green_for_ego", "light_for_other_lane"]
SIGNS = ["stop_sign_for_ego", "no_stop_sign_for_ego"]          # POST /sign (vmerge)
CROSS = ["vehicle_crossing", "path_clear"]                       # POST /cross (vmerge3 release check; first option = positive)
KEYS = {"light": ("Q_light", LIGHTS), "sign": ("Q_sign", SIGNS), "cross": ("Q_cross", CROSS)}


class QwenClient:
    def __init__(self, port: int, timeout_s: float = 5.0):
        self.url, self.timeout_s = "http://127.0.0.1:%d" % port, timeout_s

    def ask(self, jpgs: Dict[str, bytes], q: str = "light") -> Dict[str, Any]:
        """`jpgs`: {"wide": bytes, "road": bytes}. Returns Q_light, Q_light_p (q = "sign" / "cross": Q_sign / Q_cross and its _p), the round trip in
        `latency_ms` and the server's own `srv_queue_ms` / `srv_svc_ms` / `srv_depth`."""
        w, r = jpgs["wide"], jpgs["road"]
        req = urllib.request.Request(self.url + "/" + q, data=w + r, headers={"X-Sizes": "%d,%d" % (len(w), len(r)),
                                                                               "Content-Type": "application/octet-stream"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            key, opts = KEYS[q]
            out = {key: d["ans"], key + "_p": dict(zip(opts, d["p"]))}
            out.update(ok=True, srv_queue_ms=d["queue_ms"],
                       srv_svc_ms=d["svc_ms"], srv_depth=d["depth"], srv_batch=d["batch"])
        except Exception as e:  # noqa: BLE001 - a failed request is data, not a crash of the route
            out = {"ok": False, "error": str(e)[:200]}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2.0) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False
