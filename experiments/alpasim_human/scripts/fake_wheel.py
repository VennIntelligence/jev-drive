#!/usr/bin/env python3
"""Stand-in for the phone wheel daemon's UDP fan-out (experiments/phone_wheel/scripts/wheel_server.py) on its own port, so the real
daemon keeps running: answers subscriptions on 127.0.0.1:<port> (default 47801) and sends the daemon's frame format at 60 Hz.
Point the driver at it with HUMAN_WHEEL_PORT=47801. Stdlib only.

Policy (wall time from start): throttle --thr (0.3) and steer 0 for --straight s (20), then steer --steer (0.15) for --turn s (10),
then steer 0 again; repeats. --silent-after s stops sending (the driver must brake: stale).
    python3 experiments/alpasim_human/scripts/fake_wheel.py [--port 47801] [--straight 20] [--steer 0.15]
"""
import argparse
import json
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=47801)
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--straight", type=float, default=20.0)
ap.add_argument("--steer", type=float, default=0.15)
ap.add_argument("--turn", type=float, default=10.0)
ap.add_argument("--silent-after", type=float, default=0.0)
a = ap.parse_args()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", a.port))
sock.setblocking(False)
subs, t0, seq = {}, time.time(), 0
print("fake wheel on 127.0.0.1:%d" % a.port, flush=True)
while True:
    now = time.time()
    while True:
        try:
            _, addr = sock.recvfrom(64)
            if addr not in subs:
                print("subscriber %s:%d" % addr, flush=True)
            subs[addr] = now
        except (BlockingIOError, OSError):
            break
    t = (now - t0) % (a.straight + 2 * a.turn)
    steer = a.steer if a.straight <= t < a.straight + a.turn else 0.0
    if not (a.silent_after and now - t0 > a.silent_after):
        seq += 1
        msg = json.dumps({"steer": steer, "throttle": a.thr, "brake": 0.0, "raw_throttle": a.thr, "reverse": False, "handbrake": False,
                          "ts": now * 1e3, "seq": seq, "rtt": 0.5, "fake": True}).encode()
        for addr in [k for k, v in subs.items() if now - v > 3.0]:
            del subs[addr]
        for addr in subs:
            try:
                sock.sendto(msg, addr)
            except OSError:
                pass
    time.sleep(1 / 60)
