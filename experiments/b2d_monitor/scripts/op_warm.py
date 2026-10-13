#!/usr/bin/env python
"""Warm a freshly started op_arb_server: one client session, reset + N plans on noise frames, so the first real route does not
pay the first-inference cost (8.8 s on Tokyo, 2026-10-13) inside the synchronous CARLA loop. usage: op_warm.py <socket> [n]"""
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(__file__).rsplit(os.sep, 4)[0], "scripts"))   # <repo>/scripts
import zeroshot_wire as wire  # noqa: E402

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])
wire.send(s, {"cmd": "reset"})
wire.recv(s)
rng = np.random.default_rng(0)
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
    img = {k: rng.integers(0, 255, (1208, 1928, 4), dtype=np.uint8) for k in ("OP_ROAD", "OP_WIDE")}
    t0 = time.perf_counter()
    wire.send(s, {"cmd": "plan", "seed": 0, "dump": "", "speed": 0.0, "t": 0.05 * i, "desire": 0, "twin": False,
                  "intent": 0, "yaw": 0.0}, img)
    wire.recv(s)
    print("warm plan %d: %.0f ms" % (i, 1e3 * (time.perf_counter() - t0)), flush=True)
s.close()
