#!/usr/bin/env python3
"""Feeder of a resident human stack (human_stack.sh HUMAN_SERVE=1): one-scene simulate requests to AlpaSim's runtime daemon.

    python human_feed.py <stack dir> <runtime host:port> <live dir>        (in the driver image: grpc + alpasim_grpc)

The runtime runs in its shipped daemon mode (wizard.run_mode=SERVER, `alpasim_runtime.simulate --serve`, as queue_read.py on the
box): renderer, physics, controller, runtime and the human driver stay up, and every scene is one `simulate` request
(RolloutSpec(scene, nr_rollouts=1); no driver in the request, so the runtime uses the configured driver pool = the human driver).
Protocol, all files in <stack dir>/feed/ (written atomically by both sides):
  req/<name>.json   {"scene", "owner", "t"}       a client (scripts/human_loop.py) asks for one scene; served in name order, at most
                                                  HUMAN_PIPE (1; 2 with HUMAN_PREFETCH: the runtime then runs two rollouts) in flight
  lease-<owner>.json {"t", "run"}                 the client's heartbeat; a request whose owner's lease is older than LEASE_S is
                                                  dropped, and a running one is aborted (<live>/abort.json, as the panel's Next)
  busy.json         {"name", "scene", "t_sent", "items": [{"name", "scene", "t_sent"}]}   the requests in flight (name = the oldest)
  res/<name>.json   {"scene", "t_sent", "t_done", "rollouts": [{"uuid", "success", "error"}], "error"}
  ready.json, alive.json {"t"}                   the runtime answered / the feeder's heartbeat (1 s)
  stop                                            ask the feeder to exit (the stack script then takes the stack down)
Exits 0 after HUMAN_IDLE_S (1200) with no fresh lease and nothing in flight, 2 when the runtime is unreachable.
"""
import json
import os
import sys
import time
from pathlib import Path

import grpc
from alpasim_grpc.v0 import runtime_pb2, runtime_pb2_grpc

out, addr, live = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
F = out / "feed"
IDLE, LEASE_S = float(os.environ.get("HUMAN_IDLE_S", "1200")), 20.0
PIPE = max(1, int(os.environ.get("HUMAN_PIPE", "1")))
for d in ("req", "res"):
    (F / d).mkdir(parents=True, exist_ok=True)


def put(p, d):
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.write_text(json.dumps(d))
    os.replace(str(tmp), str(p))


def get(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return d


def lease_age(owner=None):
    """Seconds since the freshest lease (of `owner`, else of anyone)."""
    ts = [(get(p, {}) or {}).get("t", 0) for p in F.glob("lease-%s.json" % (owner or "*"))]
    return time.time() - max(ts or [0])


def log(msg):
    print("%s feed: %s" % (time.strftime("%F %T"), msg), flush=True)


ch = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 256 << 20)])
grpc.channel_ready_future(ch).result(timeout=120)
stub = runtime_pb2_grpc.RuntimeServiceStub(ch)
put(F / "ready.json", {"t": time.time(), "addr": addr})
log("runtime %s ready" % addr)
last = time.time()
fly = {}                                                         # name -> (future, rec, owner, aborted at)


def busy_put():
    items = [{"name": n, "scene": r["scene"], "t_sent": r["t_sent"]} for n, (_, r, _, _) in sorted(fly.items())]
    if items:
        put(F / "busy.json", {**items[0], "items": items})
    else:
        try:
            (F / "busy.json").unlink()
        except OSError:
            pass


while not (F / "stop").exists():
    put(F / "alive.json", {"t": time.time(), **({"busy": sorted(fly)} if fly else {})})
    for name in sorted(fly):                                     # returned requests
        fut, rec, owner, aborted = fly[name]
        if not fut.done():
            if lease_age(owner) > LEASE_S and time.time() - aborted > 5:   # the client is gone: end its scene (held or driven)
                put(live / "abort.json", {"scene": rec["scene"], "t": time.time(), "by": "feed"})
                fly[name] = (fut, rec, owner, time.time())
            continue
        del fly[name]
        try:
            ret = fut.result()
            rec["rollouts"] = [{"uuid": x.rollout_uuid, "success": x.success, "error": x.error[:2000]} for x in ret.rollout_returns]
        except grpc.RpcError as e:
            rec["error"] = "%s: %s" % (e.code().name, (e.details() or "")[:2000])
        rec["t_done"] = time.time()
        put(F / "res" / name, rec)
        busy_put()
        log("done %s in %.1f s %s" % (rec["scene"], rec["t_done"] - rec["t_sent"], rec["error"] or [x["success"] for x in rec["rollouts"]]))
        last = time.time()
        if rec["error"].startswith("UNAVAILABLE"):               # the runtime is gone; other errors are per request
            sys.exit(2)
    todo = sorted(p for p in (F / "req").glob("*.json") if not (F / "res" / p.name).exists() and p.name not in fly)
    if not todo or len(fly) >= PIPE:
        if fly or lease_age() < LEASE_S:
            last = time.time()
        elif time.time() - last > IDLE:
            log("idle for %.0f s with no client: exiting" % IDLE)
            sys.exit(0)
        time.sleep(0.2 if not fly else 0.1)
        continue
    p = todo[0]
    r = get(p, {}) or {}
    scene, owner = r.get("scene"), r.get("owner")
    rec = {"scene": scene, "owner": owner, "t_sent": time.time(), "rollouts": [], "error": ""}
    if not scene or lease_age(owner) > LEASE_S:
        rec.update(error="dropped: no scene or the client is gone", t_done=time.time())
        put(F / "res" / p.name, rec)
        continue
    log("simulate %s (%s)%s" % (scene, p.name, ", %d in flight" % len(fly) if fly else ""))
    fly[p.name] = (stub.simulate.future(runtime_pb2.SimulationRequest(rollout_specs=[runtime_pb2.RolloutSpec(scenario_id=scene, nr_rollouts=1)])),
                   rec, owner, 0.0)
    busy_put()
log("stop file: exiting")
