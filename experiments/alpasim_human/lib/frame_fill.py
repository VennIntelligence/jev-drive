"""The model drivers' slot frames for the human on the AlpaSim nuPlan track (display only; never touches control).

nuPlan cameras come at 2 Hz (one CAM_F0 keyframe per 0.5 s drive call). The served model (experiments/alpasim/lib/sh30_core.py)
sees 8 policy slots on a 0.2 s clock ending at its drive time t (sh30_core.SLOT_T): a keyframe where the slot is one, otherwise the
nearest keyframe (op_interp.synth_cpu "warp": the older one when the slot is at most half way) re-projected to the ego pose at the
slot through the road plane at the camera height (op_interp.warp_map, rays missing the road within 60 m at 60 m). Between two drive
calls t0 and t1 = t0 + 0.5 s the slots t1 - 0.4 and t1 - 0.2 fall in the gap; this module shows exactly those frames to the human,
each at the wall time of its slot (sim time mapped from the last drive's return), then the keyframe t1 itself: 5 Hz on the slot clock.

Causal rule: only keyframes the driver has received by the display moment (the next keyframe normally arrives during the previous
drive's pacing sleep, so the nearest one is usually there; else the newest older one, src_rule "newest"). Ego pose at the slot:
the driven pose (the sim's egomotion samples, linear interpolation) when the sim has reported that time, else the human's last 4 s
plan (human_driver.rollout, 0.1 s poses). The front view is the raw CAM_F0 keyframe warped at reduced size (HUMAN_FILL_W, 960 px
wide, pinhole intrinsics and rig_to_camera rotation of camera.json, distortion ignored); <live>/front.jpg (atomic rename) and
state.json "fill" {slot_t, src_t, src_rule, pose (driven | plan | hold), age_s, synth, ms}. HUMAN_MODEL_VIEW=1 also writes
<live>/model.jpg: the openpilot road + wide model frame the policy consumes at the slot (sh30_core.pack_fast + op_interp.warp_frame,
the served synth "cpu" path), stacked vertically, RGB.

Env: HUMAN_FILL=slots (default on nuPlan) | 0. Before the start (and while held): the newest raw keyframe; paused mid-scene: the view stops at the next drive time.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(p) for p in (_R, _R / "experiments/alpasim/lib") if str(p) not in sys.path]
from jevdrive import op_interp as I  # noqa: E402

LOG = logging.getLogger("human.fill")
E = os.environ.get
SLOT_T = np.round(np.arange(-7, 1) * 0.2, 3)          # sh30_core.SLOT_T (not imported there: its imports pull the model stack)
FILL_W, TICK = int(E("HUMAN_FILL_W", "960")), 0.01


def enabled() -> bool:
    return E("HUMAN_FILL", "slots" if E("HUMAN_TRACK") == "nuplan" else "0") == "slots"


def decode(jpg: bytes, w: int) -> np.ndarray:
    """JPEG -> BGR resized to w px wide."""
    import cv2
    im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(im, (w, round(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)


def interp_pose(ts, P, t):
    """ts (n,) us ascending, P (n, 3) x, y, yaw -> pose at t (linear, yaw unwrapped); None outside [ts[0], ts[-1]]."""
    if len(ts) == 0 or t < ts[0] - 1e3 or t > ts[-1] + 1e3:
        return None
    yaw = np.unwrap(P[:, 2])
    return np.array([np.interp(t, ts, P[:, 0]), np.interp(t, ts, P[:, 1]), np.interp(t, ts, yaw)])


def model_rgb(packed) -> np.ndarray:
    """(2, 6, 128, 256) packed YUV420 model frame -> (512, 512, 3) BGR: road over wide (JPEG YCbCr, as op_interp.to_rgb)."""
    import cv2
    out = []
    for p in packed:
        Y, U, V = I.unpack(p)
        U, V = (cv2.resize(c, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_NEAREST) for c in (U, V))
        out.append(cv2.cvtColor(np.dstack([Y, V, U]), cv2.COLOR_YCrCb2BGR))
    return np.vstack(out)


class Fill:
    """One background thread for the driver; the newest session that drove owns the front view."""

    def __init__(self, live: Path, put, state_put):
        import cv2
        cv2.setNumThreads(2)
        self.live, self.put, self.state_put = live, put, state_put
        self.model_view = E("HUMAN_MODEL_VIEW", "0") == "1"
        self.s, self.cal, self.keys, self.mkeys = None, None, {}, {}       # keys: ts -> small BGR; mkeys: ts -> packed
        self.d = None                     # last drive: (wall, now us, period us, plan ts (n,), plan (n, 3), started, paused)
        self.shown, self.info, self.lock = None, {}, threading.Lock()
        self.ms = []
        threading.Thread(target=self._run, daemon=True, name="frame-fill").start()

    # ---- hooks from human_driver (cheap, never raise)
    def calib(self, cal: dict | None):
        ok = bool(cal) and cal.get("spec", {}).get("model") == "pinhole"
        with self.lock:
            self.s, self.cal, self.keys, self.mkeys, self.d, self.shown = None, (cal if ok else None), {}, {}, None, None
        if not ok:
            LOG.warning("fill off for this scene: no pinhole front camera")

    def drive(self, s, now: int, ta: int, plan: np.ndarray, started: bool, paused: bool = False):
        with self.lock:
            per = now - self.d[1] if self.d and self.d[0] is not None and self.s is s and 0 < now - self.d[1] < 2_000_000 else 500_000
            self.s = s
            self.d = (time.time(), now, per, ta + np.arange(len(plan)) * 100_000, plan[:, :3].copy(), started, paused)

    def state(self) -> dict:
        return {"fill": dict(self.info)} if self.info else {}

    # ---- thread
    def _run(self):
        while True:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001  (display only)
                LOG.warning("fill: %r", e)
                time.sleep(0.5)
            time.sleep(TICK)

    def _ingest(self, s):
        with s.lock:
            f = s.frames.get("front")
        if f and f[0] not in self.keys:
            self.keys[f[0]] = decode(f[1], FILL_W)
            if self.model_view:
                import sh30_core as C
                c = self.cal
                K = [[c["spec"]["f"][0], 0, c["spec"]["c"][0]], [0, c["spec"]["f"][1], c["spec"]["c"][1]], [0, 0, 1]]
                self.mkeys[f[0]] = C.pack_fast(f[1], C.camera(K, c["R"], c["cam_t"]))
            for k in sorted(self.keys)[:-3]:
                self.keys.pop(k), self.mkeys.pop(k, None)

    def _pose(self, s, t: int, d):
        with s.lock:
            ts = np.array(sorted(s.poses), float)
            P = np.array([s.poses[k][:2] + (s.poses[k][3],) for k in sorted(s.poses)], float).reshape(-1, 3)
        p = interp_pose(ts, P, t)
        if p is not None:
            return p, "driven"
        if d is not None:
            p = interp_pose(d[3].astype(float), d[4], t)
            if p is not None:
                return p, "plan"
        return (P[-1] if len(P) else np.zeros(3)), "hold"

    def _tick(self):
        with self.lock:
            s, d, cal = self.s, self.d, self.cal
        if s is None or cal is None:
            return
        self._ingest(s)
        if not self.keys:
            return
        kt = sorted(self.keys)
        if d is None or not d[5] or d[6]:                          # not driving: the newest raw keyframe
            return self._show(s, kt[-1], kt[-1], "keyframe", d)
        wall, now, per = d[:3]
        t1 = now + per
        t_sim = min(now + (time.time() - wall) * 1e6 * max(float(E("HUMAN_REALTIME", "1.0")), 1e-3), t1)
        slots = [now] + [int(t) for t in t1 + np.round(SLOT_T * 1e6).astype(np.int64) if now < t <= t_sim]
        t = slots[-1]
        exact = [k for k in kt if abs(k - t) <= 1e3]
        older, newer = [k for k in kt if k < t - 1e3], [k for k in kt if k > t + 1e3]
        if exact:
            src, rule = exact[0], "keyframe"
        elif older and newer:                                      # op_interp.synth_cpu: the older one when at most half way
            src, rule = (older[-1] if (t - older[-1]) / (newer[0] - older[-1]) <= 0.5 else newer[0]), "nearest"
        else:
            src, rule = (older[-1], "newest") if older else (newer[0], "nearest")
        self._show(s, t, src, rule, d)

    def _show(self, s, t: int, src: int, rule: str, d):
        if self.shown == (t, src):
            return
        import cv2
        t_in = time.time()
        small, mod = self.keys[src], self.mkeys.get(src)
        pose = "keyframe"
        if src != t:
            pd, pose = self._pose(s, t, d)
            ps, _ = self._pose(s, src, d)
            c, (h, w) = self.cal, small.shape[:2]
            k, cam = w / c["spec"]["native"][0], np.asarray(c["cam_t"], float)
            K = np.array([[c["spec"]["f"][0] * k, 0, c["spec"]["c"][0] * k], [0, c["spec"]["f"][1] * k, c["spec"]["c"][1] * k], [0, 0, 1]])
            mx, my = I.warp_map("road", cam, pd, ps, K=K, R=np.asarray(c["R"], float), size=(w, h))
            small = cv2.remap(small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if mod is not None:
                mod = I.warp_frame(mod, cam, pd, ps)
        self.put(self.live / "front.jpg", cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        if mod is not None:
            self.put(self.live / "model.jpg", cv2.imencode(".jpg", model_rgb(mod), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        ms = 1e3 * (time.time() - t_in)
        self.ms = (self.ms + [ms])[-50:]
        self.shown = (t, src)
        self.info = {"synth": src != t, "slot_t": round(t / 1e6, 2), "src_t": round(src / 1e6, 2), "age_s": round((t - src) / 1e6, 2),
                     "src_rule": rule, "pose": pose, "ms": round(ms, 1), "ms_mean": round(float(np.mean(self.ms)), 1),
                     "model_view": mod is not None, "wall": round(time.time(), 3)}
        self.state_put(self.state())
