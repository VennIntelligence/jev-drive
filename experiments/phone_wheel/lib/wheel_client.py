"""Non-blocking reader for the phone wheel daemon (stdlib only, Python 3.8+).

    from wheel_client import WheelClient
    wheel = WheelClient()                        # once
    steer, throttle, brake, age_s = wheel.latest()   # every tick; never blocks

steer in [-1, 1] (+ = right), throttle and brake in [0, 1]. If no frame arrived for STALE_S, latest() returns
throttle 0 and brake 1; `wheel.stale` and `wheel.state()` (adds reverse / handbrake) tell the caller.
Run as a script for a 10 Hz console readout: python experiments/phone_wheel/lib/wheel_client.py
"""
import json
import socket
import time

STALE_S = 0.3
RENEW_S = 0.5  # resubscribe period; the daemon drops a subscriber after 3 s of silence


class WheelClient:
    def __init__(self, host="127.0.0.1", port=47800):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))
        self.sock.setblocking(False)
        self.frame, self.t_rx, self.t_sub = None, -1e9, -1e9

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
            except (BlockingIOError, ConnectionRefusedError, OSError):
                break
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
        """Dict with steer, throttle (slew-limited), raw_throttle (thumb), brake, reverse, handbrake, age_s, stale."""
        age = self._poll()
        f = self.frame or {}
        stale = self.frame is None or age > STALE_S
        return {"steer": 0.0 if stale else float(f.get("steer", 0.0)),
                "throttle": 0.0 if stale else float(f.get("throttle", 0.0)),
                "brake": 1.0 if stale else float(f.get("brake", 0.0)),
                "raw_throttle": 0.0 if stale else float(f.get("raw_throttle", f.get("throttle", 0.0))),
                "reverse": bool(f.get("reverse")) and not stale,
                "handbrake": bool(f.get("handbrake")) or stale,
                "age_s": age, "stale": stale}

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
