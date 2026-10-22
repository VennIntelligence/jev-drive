#!/usr/bin/env python3
"""Stand-in for the phone wheel daemon's UDP fan-out (experiments/phone_wheel/scripts/wheel_server.py) on its own port, so the real
daemon keeps running: answers subscriptions on 127.0.0.1:<port> (default 47801) and sends the daemon's frame format at 60 Hz.
Point the driver at it with HUMAN_WHEEL_PORT=47801. Stdlib only.

Policy (wall time from start): throttle --thr (0.3) and steer 0 for --straight s (20), then steer --steer (0.15) for --turn s (10),
then steer 0 again; repeats. --silent-after s stops sending (the driver must brake: stale).
--brake-at V (brake probe, results/brake_spin.md): steer 0, full throttle until the driver's speed (<live>/state.json, --live) reaches
V m/s, then brake --brake (1.0) for --brake-s s (6), then throttle again; repeats.
    python3 experiments/alpasim_human/scripts/fake_wheel.py [--port 47801] [--straight 20] [--steer 0.15]
"""
import argparse
import json
import socket
from pathlib import Path
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=47801)
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--straight", type=float, default=20.0)
ap.add_argument("--steer", type=float, default=0.15)
ap.add_argument("--turn", type=float, default=10.0)
ap.add_argument("--silent-after", type=float, default=0.0)
ap.add_argument("--brake-at", type=float, default=0.0)
ap.add_argument("--brake", type=float, default=1.0)
ap.add_argument("--brake-s", type=float, default=6.0)
ap.add_argument("--live", default="/data/runs/alpasim_human/live")
a = ap.parse_args()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", a.port))
sock.setblocking(False)
subs, t0, seq, t_brake = {}, time.time(), 0, None
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
    thr, brk = a.thr, 0.0
    if a.brake_at > 0:
        steer, thr = 0.0, 1.0
        if t_brake is None:
            try:
                st = json.loads((Path(a.live) / "state.json").read_text())
                if st.get("started") and st.get("speed", 0) >= a.brake_at:
                    t_brake = now
                    print("brake at %.1f m/s, sim %.1f s" % (st["speed"], st.get("sim_t", -1)), flush=True)
            except (OSError, ValueError):
                pass
        if t_brake is not None:
            thr, brk = 0.0, a.brake
            if now - t_brake > a.brake_s:
                t_brake = None
    if not (a.silent_after and now - t0 > a.silent_after):
        seq += 1
        msg = json.dumps({"steer": steer, "throttle": thr, "brake": brk, "raw_throttle": thr, "reverse": False, "handbrake": False,
                          "ts": now * 1e3, "seq": seq, "rtt": 0.5, "fake": True}).encode()
        for addr in [k for k, v in subs.items() if now - v > 3.0]:
            del subs[addr]
        for addr in subs:
            try:
                sock.sendto(msg, addr)
            except OSError:
                pass
    time.sleep(1 / 60)
