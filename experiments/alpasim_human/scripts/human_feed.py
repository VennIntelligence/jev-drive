#!/usr/bin/env python3
"""Feeder of a resident human stack (human_stack.sh HUMAN_SERVE=1): one-scene simulate requests to AlpaSim's runtime daemon.

    python human_feed.py <stack dir> <runtime host:port> <live dir>        (in the driver image: grpc + alpasim_grpc)

The runtime runs in its shipped daemon mode (wizard.run_mode=SERVER, `alpasim_runtime.simulate --serve`, as queue_read.py on the
box): renderer, physics, controller, runtime and the human driver stay up, and every scene is one `simulate` request
(RolloutSpec(scene, nr_rollouts=1); no driver in the request, so the runtime uses the configured driver pool = the human driver).
Protocol, all files in <stack dir>/feed/ (written atomically by both sides):
  req/<name>.json   {"scene", "owner", "t"}       a client (scripts/human_loop.py) asks for one scene; served in name order, one at a time
  lease-<owner>.json {"t", "run"}                 the client's heartbeat; a request whose owner's lease is older than LEASE_S is
                                                  dropped, and a running one is aborted (<live>/abort.json, as the panel's Next)
  busy.json         {"name", "scene", "t_sent"}   the request in flight
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
while not (F / "stop").exists():
    put(F / "alive.json", {"t": time.time()})
    todo = sorted(p for p in (F / "req").glob("*.json") if not (F / "res" / p.name).exists())
    if not todo:
        if lease_age() < LEASE_S:
            last = time.time()
        elif time.time() - last > IDLE:
            log("idle for %.0f s with no client: exiting" % IDLE)
            sys.exit(0)
        time.sleep(0.2)
        continue
    p = todo[0]
    r = get(p, {}) or {}
    scene, owner = r.get("scene"), r.get("owner")
    rec = {"scene": scene, "owner": owner, "t_sent": time.time(), "rollouts": [], "error": ""}
    if not scene or lease_age(owner) > LEASE_S:
        rec.update(error="dropped: no scene or the client is gone", t_done=time.time())
        put(F / "res" / p.name, rec)
        continue
    put(F / "busy.json", {"name": p.name, "scene": scene, "t_sent": rec["t_sent"]})
    log("simulate %s (%s)" % (scene, p.name))
    fut = stub.simulate.future(runtime_pb2.SimulationRequest(rollout_specs=[runtime_pb2.RolloutSpec(scenario_id=scene, nr_rollouts=1)]))
    aborted = 0.0
    while not fut.done():
        put(F / "alive.json", {"t": time.time(), "busy": p.name})
        if lease_age(owner) > LEASE_S and time.time() - aborted > 5:   # the client is gone: end its scene (held or driven)
            put(live / "abort.json", {"scene": scene, "t": time.time(), "by": "feed"})
            aborted = time.time()
        time.sleep(0.5)
    try:
        ret = fut.result()
        rec["rollouts"] = [{"uuid": x.rollout_uuid, "success": x.success, "error": x.error[:2000]} for x in ret.rollout_returns]
    except grpc.RpcError as e:
        rec["error"] = "%s: %s" % (e.code().name, (e.details() or "")[:2000])
    rec["t_done"] = time.time()
    put(F / "res" / p.name, rec)
    (F / "busy.json").unlink()
    log("done %s in %.1f s %s" % (scene, rec["t_done"] - rec["t_sent"], rec["error"] or [x["success"] for x in rec["rollouts"]]))
    last = time.time()
    if rec["error"].startswith("UNAVAILABLE"):                   # the runtime is gone; other errors are per request
        sys.exit(2)
log("stop file: exiting")
