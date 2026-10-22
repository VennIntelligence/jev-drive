"""Phone steering wheel daemon.

Serves the phone page (web/) and a WebSocket at /ws. The phone sends JSON frames
{steer, throttle, brake, raw_throttle, reverse, handbrake, ts, seq, rtt} at 50-100 Hz; the daemon keeps only the latest,
echoes {ts} for round-trip timing. Keyboard frames {steer, throttle, brake, raw_throttle, handbrake, keys} arrive as UDP datagrams on
127.0.0.1:<key-port> (lib/keyboard_wheel.py, sent by the focused viewer window every frame, a heartbeat even with no key held).
The two are merged and sent as one JSON datagram to each local UDP subscriber on every phone frame and at MERGE_HZ: steer =
clip(phone + keyboard), throttle / brake / raw_throttle = max, handbrake = or, the phone's other fields as they came, plus
source = phone | keyboard | both. A source counts while its last frame is younger than FRESH_S; with neither fresh nothing is
sent (clients go stale and brake). Status (phone / keyboard age, source) goes to <status> about once a second for the desktop panel.

Local consumers subscribe by sending any datagram to 127.0.0.1:<udp-port> at least every few seconds
(lib/wheel_client.py does this). Publish over HTTPS with `tailscale serve --bg http://127.0.0.1:<port>`.

    /data/envs/wheel/bin/python experiments/phone_wheel/scripts/wheel_server.py [--port 8765] [--lan-port 8766] [--key-port 47802]
"""
import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from aiohttp import WSMsgType, web

WEB = Path(__file__).resolve().parents[1] / "web"
SUB_TTL = 3.0      # s a UDP subscriber stays registered without renewing
STATS_EVERY = 5.0  # s between log lines and stats pushes
FRESH_S = 0.3      # s a phone / keyboard frame keeps counting in the merge
MERGE_HZ = 60.0    # fixed emit rate (phone frames are also sent as they come)
STATUS = os.environ.get("WHEEL_STATUS", os.path.join(os.environ.get("DATA_DIR", "/data"), "tmp/wheel/status.json"))

log = logging.getLogger("wheel")


class Keys(asyncio.DatagramProtocol):
    """Keyboard frames from the viewers (lib/keyboard_wheel.py); keeps the latest."""

    def __init__(self, hub):
        self.hub = hub

    def datagram_received(self, data, addr):
        try:
            f = json.loads(data.decode())
            k = {n: float(f.get(n, 0.0)) for n in ("steer", "throttle", "brake", "raw_throttle")}
            k.update(handbrake=bool(f.get("handbrake")), keys=list(f.get("keys") or [])[:8])
        except (ValueError, TypeError, AttributeError):
            return
        if self.hub.key_t < time.monotonic() - FRESH_S:
            log.info("keyboard active (%s:%d)", *addr)
        self.hub.key, self.hub.key_t = k, time.monotonic()


class Hub(asyncio.DatagramProtocol):
    """Latest phone and keyboard frames, their merge, UDP fan-out to local subscribers, counters."""

    def __init__(self):
        self.latest, self.latest_t = None, 0.0
        self.key, self.key_t, self.sent_t, self.source = None, -1e9, -1e9, None
        self.subs = {}  # addr -> last renew time
        self.transport = None
        self.n, self.drops, self.last_seq, self.rtt = 0, 0, None, None
        self.clients = set()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if addr not in self.subs:
            log.info("udp subscriber %s:%d", *addr)
        self.subs[addr] = time.monotonic()

    def publish(self, frame):
        now = time.monotonic()
        seq = frame.get("seq")
        if isinstance(seq, int) and isinstance(self.last_seq, int) and seq > self.last_seq + 1:
            self.drops += seq - self.last_seq - 1
        self.last_seq = seq
        if frame.get("rtt") is not None:
            self.rtt = frame["rtt"]
        self.n += 1
        frame["rx"] = time.time()
        self.latest, self.latest_t = frame, now
        self.emit()

    def merged(self):
        """Phone + keyboard frame by the merge rules, or None when neither is fresh."""
        now = time.monotonic()
        ph = self.latest if self.latest and now - self.latest_t < FRESH_S else None
        kb = self.key if self.key and now - self.key_t < FRESH_S else None
        if not (ph or kb):
            return None
        f = dict(ph) if ph else {"steer": 0.0, "throttle": 0.0, "brake": 0.0, "raw_throttle": 0.0, "reverse": False, "handbrake": False,
                                 "ts": None, "seq": None, "rtt": None, "rx": time.time()}
        if kb:
            f.update(steer=max(-1.0, min(1.0, f["steer"] + kb["steer"])), handbrake=f["handbrake"] or kb["handbrake"], keys=kb["keys"],
                     **{n: max(f[n], kb[n]) for n in ("throttle", "brake", "raw_throttle")})
        f["source"] = "both" if ph and kb else "phone" if ph else "keyboard"
        return f

    def emit(self):
        f = self.merged()
        self.source = f and f["source"]
        if f is None:
            return
        now = self.sent_t = time.monotonic()
        data = json.dumps(f, separators=(",", ":")).encode()
        for addr, t in list(self.subs.items()):
            if now - t > SUB_TTL:
                del self.subs[addr]
                log.info("udp subscriber %s:%d expired", *addr)
            else:
                self.transport.sendto(data, addr)


async def ws_handler(request):
    hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=5.0, max_msg_size=4096)
    await ws.prepare(request)
    peer = request.headers.get("X-Forwarded-For") or request.remote
    hub.clients.add(ws)
    log.info("phone connected from %s (%d open)", peer, len(hub.clients))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                f = json.loads(msg.data)
                frame = {k: float(f.get(k, 0.0)) for k in ("steer", "throttle", "brake")}
                frame["raw_throttle"] = float(f.get("raw_throttle", frame["throttle"]))
                frame.update(reverse=bool(f.get("reverse")), handbrake=bool(f.get("handbrake")),
                             ts=f.get("ts"), seq=f.get("seq"), rtt=f.get("rtt"))
            except (ValueError, TypeError, AttributeError):
                log.warning("bad frame from %s: %.80s", peer, msg.data)
                continue
            hub.publish(frame)
            await ws.send_str(json.dumps({"ts": frame["ts"]}))
    finally:
        hub.clients.discard(ws)
        log.info("phone disconnected from %s (%d open)", peer, len(hub.clients))
    return ws


async def merge_loop(hub):
    """Emit the merge at MERGE_HZ (skipped right after a phone frame went out)."""
    while True:
        await asyncio.sleep(1.0 / MERGE_HZ)
        if time.monotonic() - hub.sent_t > 0.8 / MERGE_HZ:
            hub.emit()


async def status_loop(hub):
    """<status> once a second for the desktop panel (lib/exam_panel.py): phone / keyboard freshness and the driving source."""
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    while True:
        now = time.monotonic()
        st = {"t": time.time(), "pid": os.getpid(), "source": hub.source, "subs": len(hub.subs),
              "phone": {"clients": len(hub.clients), "age_s": round(now - hub.latest_t, 2) if hub.latest else None, "rtt": hub.rtt},
              "keyboard": {"age_s": round(now - hub.key_t, 2) if hub.key else None, "keys": (hub.key or {}).get("keys", [])}}
        try:
            with open(STATUS + ".tmp", "w") as fh:
                json.dump(st, fh)
            os.replace(STATUS + ".tmp", STATUS)
        except OSError as e:
            log.warning("status write failed: %r", e)
        await asyncio.sleep(1.0)


async def stats_loop(hub):
    n0, d0 = 0, 0
    while True:
        await asyncio.sleep(STATS_EVERY)
        rate, drops = (hub.n - n0) / STATS_EVERY, hub.drops - d0
        n0, d0 = hub.n, hub.drops
        age = time.monotonic() - hub.latest_t if hub.latest else None
        if hub.clients or rate:
            log.info("rate %.1f Hz, drops %d, rtt %s ms, phones %d, udp subs %d, last %s",
                     rate, drops, None if hub.rtt is None else round(hub.rtt, 1), len(hub.clients),
                     len(hub.subs), hub.latest and {k: round(hub.latest[k], 2) for k in ("steer", "throttle", "brake")})
        msg = json.dumps({"stats": {"rate": rate, "drops": drops, "subs": len(hub.subs),
                                    "age": None if age is None else round(age, 3)}})
        for ws in list(hub.clients):
            if not ws.closed:
                await ws.send_str(msg)


STATIC = {"/": "index.html", **{"/" + n: n for n in ("manifest.webmanifest", "sw.js", "icon-192.png", "icon-512.png")}}


async def static(request):
    resp = web.FileResponse(WEB / STATIC[request.path])
    resp.headers["Cache-Control"] = "no-cache"
    return resp


async def main(args):
    hub = Hub()
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: hub, local_addr=("127.0.0.1", args.udp_port))
    await loop.create_datagram_endpoint(lambda: Keys(hub), local_addr=("127.0.0.1", args.key_port))
    app = web.Application()
    app["hub"] = hub
    app.router.add_get("/ws", ws_handler)
    for path in STATIC:
        app.router.add_get(path, static)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    log.info("http://127.0.0.1:%d (publish with: tailscale serve --bg http://127.0.0.1:%d)", args.port, args.port)
    if args.lan_port:
        await web.TCPSite(runner, "0.0.0.0", args.lan_port).start()
        log.info("plain-http fallback on 0.0.0.0:%d", args.lan_port)
    log.info("udp fan-out on 127.0.0.1:%d, keyboard in on 127.0.0.1:%d, status %s", args.udp_port, args.key_port, STATUS)
    asyncio.ensure_future(merge_loop(hub))
    asyncio.ensure_future(status_loop(hub))
    await stats_loop(hub)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lan-port", type=int, default=0, help="also serve plain http on 0.0.0.0 (0 = off)")
    ap.add_argument("--udp-port", type=int, default=47800)
    ap.add_argument("--key-port", type=int, default=int(os.environ.get("WHEEL_KEY_PORT", 47802)), help="keyboard frames in (viewers)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
