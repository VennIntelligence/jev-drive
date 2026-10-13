"""Phone steering wheel daemon.

Serves the phone page (web/) and a WebSocket at /ws. The phone sends JSON frames
{steer, throttle, brake, raw_throttle, reverse, handbrake, ts, seq, rtt} at 50-100 Hz; the daemon keeps only the latest,
echoes {ts} for round-trip timing, and forwards every frame as one JSON datagram to each local UDP subscriber.

Local consumers subscribe by sending any datagram to 127.0.0.1:<udp-port> at least every few seconds
(lib/wheel_client.py does this). Publish over HTTPS with `tailscale serve --bg http://127.0.0.1:<port>`.

    /data/envs/wheel/bin/python experiments/phone_wheel/scripts/wheel_server.py [--port 8765] [--lan-port 8766]
"""
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import WSMsgType, web

WEB = Path(__file__).resolve().parents[1] / "web"
SUB_TTL = 3.0      # s a UDP subscriber stays registered without renewing
STATS_EVERY = 5.0  # s between log lines and stats pushes

log = logging.getLogger("wheel")


class Hub(asyncio.DatagramProtocol):
    """Latest frame, UDP fan-out to local subscribers, counters."""

    def __init__(self):
        self.latest, self.latest_t = None, 0.0
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
        data = json.dumps(frame, separators=(",", ":")).encode()
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
    log.info("udp fan-out on 127.0.0.1:%d", args.udp_port)
    await stats_loop(hub)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lan-port", type=int, default=0, help="also serve plain http on 0.0.0.0 (0 = off)")
    ap.add_argument("--udp-port", type=int, default=47800)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
