"""Non-blocking reader for the phone wheel daemon (stdlib only, Python 3.8+).

    from wheel_client import WheelClient
    wheel = WheelClient()                        # once
    steer, throttle, brake, age_s = wheel.latest()   # every tick; never blocks

steer in [-1, 1] (+ = right), throttle and brake in [0, 1]. If no frame arrived for STALE_S, latest() returns
throttle 0 and brake ramping to 1 over WHEEL_BRAKE_RAMP_S (steer held at its last value meanwhile; handbrake once the ramp is
full); `wheel.stale` and `wheel.state()` (adds reverse / handbrake, age, receive rate) tell the caller.
Env: WHEEL_STALE_S (1.0; was 0.3 until 2026-10-13, when Chrome's ~1 Hz timer throttling between routes tripped it),
WHEEL_BRAKE_RAMP_S (1.0).
Run as a script for a 10 Hz console readout: python experiments/phone_wheel/lib/wheel_client.py
"""
import collections
import json
import os
import socket
import time

STALE_S = float(os.environ.get("WHEEL_STALE_S", 1.0))
BRAKE_RAMP_S = float(os.environ.get("WHEEL_BRAKE_RAMP_S", 1.0))
RENEW_S = 0.5  # resubscribe period; the daemon drops a subscriber after 3 s of silence


class WheelClient:
    def __init__(self, host="127.0.0.1", port=None):
        self.addr = (host, int(port or os.environ.get("WHEEL_PORT", 47800)))  # WHEEL_PORT: tests point it at a dead port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))
        self.sock.setblocking(False)
        self.frame, self.t_rx, self.t_sub = None, -1e9, -1e9
        self.rx = collections.deque()  # receive times over the last second

    def _poll(self):
        now = time.monotonic()
        if now - self.t_sub > RENEW_S:
            self.t_sub = now
            try:
                self.sock.sendto(b"sub", self.addr)
            except OSError:
                pass
        data = None
        while True:  # drain; keep only the newest datagram
            try:
                data = self.sock.recv(4096)
                self.rx.append(now)
            except (BlockingIOError, ConnectionRefusedError, OSError):
                break
        while self.rx and now - self.rx[0] > 1.0:
            self.rx.popleft()
        if data is not None:
            try:
                self.frame, self.t_rx = json.loads(data.decode()), now
            except ValueError:
                pass
        return now - self.t_rx

    @property
    def stale(self):
        return self.frame is None or time.monotonic() - self.t_rx > STALE_S

    def state(self):
        """Dict with steer, throttle (slew-limited), raw_throttle (thumb), brake, reverse, handbrake, age_s, stale, rx_hz, reason."""
        age = self._poll()
        f = self.frame or {}
        stale = self.frame is None or age > STALE_S
        ramp = 1.0 if self.frame is None else min(1.0, (age - STALE_S) / max(BRAKE_RAMP_S, 1e-3)) if stale else 0.0
        reason = "" if not stale else "no phone frame yet" if self.frame is None else "no phone frame for %.1f s" % age
        return {"steer": (0.0 if self.frame is None else float(f.get("steer", 0.0))) if stale else float(f.get("steer", 0.0)),
                "throttle": 0.0 if stale else float(f.get("throttle", 0.0)),
                "brake": max(ramp, float(f.get("brake", 0.0))) if stale else float(f.get("brake", 0.0)),
                "raw_throttle": 0.0 if stale else float(f.get("raw_throttle", f.get("throttle", 0.0))),
                "reverse": bool(f.get("reverse")) and not stale,
                "handbrake": bool(f.get("handbrake")) or ramp >= 1.0,
                "age_s": age, "stale": stale, "rx_hz": len(self.rx), "reason": reason}

    def latest(self):
        """(steer, throttle, brake, age_s); stale -> throttle 0, brake 1, steer 0."""
        s = self.state()
        return s["steer"], s["throttle"], s["brake"], s["age_s"]


_default = None


def latest():
    """Module-level convenience over one shared WheelClient."""
    global _default
    if _default is None:
        _default = WheelClient()
    return _default.latest()


if __name__ == "__main__":
    w = WheelClient()
    try:
        while True:
            s = w.state()
            age = "never" if s["age_s"] > 1e8 else "%6.3f s" % s["age_s"]
            print("steer %+5.2f  throttle %4.2f  brake %4.2f  %s%s  age %s %s" % (
                s["steer"], s["throttle"], s["brake"], "R" if s["reverse"] else "-",
                "H" if s["handbrake"] else "-", age, "STALE" if s["stale"] else ""), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
