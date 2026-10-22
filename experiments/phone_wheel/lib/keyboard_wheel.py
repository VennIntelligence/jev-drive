"""Keyboard as a second wheel input (stdlib only, Python 3.8+): the pygame viewers call it every frame while their window has focus.

    from keyboard_wheel import KeyboardWheel, pygame_keys
    kb = KeyboardWheel()                                         # once
    kb.update(pygame_keys(pygame))                               # every frame, after the event pump; returns the frame sent

Keys: Left / A and Right / D steer (ramps toward full lock at STEER_RATE per s while held, recenters at CENTER_RATE when released,
as CARLA's manual_control), Up / W throttle (ramps at THROTTLE_RATE; raw_throttle = 1 while held, so "start on throttle" holds
release), Down / S brake (immediate, full), Space handbrake. Every update sends one JSON datagram to the wheel daemon's keyboard
port (127.0.0.1:WHEEL_KEY_PORT, 47802), also with no key held (a heartbeat: the daemon counts the keyboard while it is fresh and
merges it with the phone; scripts/wheel_server.py). Without focus nothing is sent and the keys reset.
"""
import json
import os
import socket
import time

STEER_RATE, CENTER_RATE, THROTTLE_RATE = 1.2, 4.0, 2.0   # per s
NAMES = {"left": ("K_LEFT", "K_a"), "right": ("K_RIGHT", "K_d"), "up": ("K_UP", "K_w"), "down": ("K_DOWN", "K_s"), "space": ("K_SPACE",)}


def pygame_keys(pygame):
    """Held key names (left / right / up / down / space) from pygame's key state; None while the window has no focus."""
    if not pygame.key.get_focused():
        return None
    k = pygame.key.get_pressed()
    return {n for n, ks in NAMES.items() if any(k[getattr(pygame, c)] for c in ks)}


def _toward(v, target, step):
    return min(target, v + step) if v < target else max(target, v - step)


class KeyboardWheel(object):
    def __init__(self, host="127.0.0.1", port=None):
        self.addr = (host, int(port or os.environ.get("WHEEL_KEY_PORT", 47802)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.steer, self.throttle, self.t, self.frame = 0.0, 0.0, None, None

    def update(self, keys, now=None):
        """keys: set of held names, or None (no focus: reset, send nothing). Returns the frame sent, or None."""
        now = time.monotonic() if now is None else now
        dt, self.t = (0.0 if self.t is None else min(0.1, now - self.t)), now
        if keys is None:
            self.steer, self.throttle, self.frame = 0.0, 0.0, None
            return None
        lr = ("right" in keys) - ("left" in keys)
        self.steer = _toward(self.steer, float(lr), STEER_RATE * dt) if lr else _toward(self.steer, 0.0, CENTER_RATE * dt)
        self.throttle = min(1.0, self.throttle + THROTTLE_RATE * dt) if "up" in keys else 0.0
        if lr and self.steer * lr < 0:                                          # reversing direction: through centre at once
            self.steer = 0.0
        f = {"steer": round(self.steer, 3), "throttle": round(self.throttle, 3), "brake": 1.0 if "down" in keys else 0.0,
             "raw_throttle": 1.0 if "up" in keys else 0.0, "handbrake": "space" in keys, "keys": sorted(keys), "t": time.time()}
        try:
            self.sock.sendto(json.dumps(f, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass
        self.frame = f
        return f

    def text(self):
        """Short status for the viewers' input panel."""
        f = self.frame
        return "keyboard: no focus" if f is None else "keyboard: %s" % (" ".join(f["keys"]) or "focused, no key")
