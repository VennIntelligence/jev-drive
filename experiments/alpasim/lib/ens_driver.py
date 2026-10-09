"""Adapter ensemble as an AlpaSim nuPlan-track driver (lane OT2 piece C, plans/2026-10-09-ot2-ensemble-prereg.md; decision 211).

The Cinque vision encoder is frozen and identical in every op_parity / AP2 checkpoint (only the plan pathway and the ego adapter are
trained and stored), so N checkpoints of one input standard share ONE encoder pass per decision; each then runs its own policy on the same
tokens and the plan outputs (33 x 15, before the rear-axle export) are averaged with equal weights. Everything else is the single driver,
imported unchanged: sh30_driver.Driver around sh30_core.Core when the checkpoints are NAVSIM-standard tags (SH30, OT30), ap2_driver.Driver
around ap2_core.Core when they are AlpaSim-standard tags (AP2, APO). Mixed standards are refused (the members would need different inputs).

Environment: ENS_TAGS=tag1,tag2[,...] or tag1+tag2 (required); ENS_COLD (unset = the members' own rule, which must agree); the rest as sh30_driver.py /
ap2_driver.py (SH30_DUMP, SH30_LHT, SH30_MOTION, SH30_MOTION_GATE, ALPASIM_DRIVER_*). Besides drive.jsonl the log dir gets members.jsonl:
per decision the 8 exported poses of every member on the shared state (for the disagreement read-out).

  ENS_TAGS=OT30-F-s0,OT30-F-s1 python experiments/alpasim/lib/ens_driver.py            serve
  python experiments/alpasim/lib/ens_driver.py bench OT30-F-s0,OT30-F-s1 [n]           single-step latency: first member alone vs the ensemble
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOG = logging.getLogger("ens")


class Mean:
    """N PModels behind the call interface the cores use: `net` (the shared encoder) of the first member, the mean of the members' outputs."""

    def __init__(self, models):
        import torch
        self.torch, self.models = torch, list(models)
        m = self.models[0]
        self.net, self.arm, self.adapter = m.net, m.arm, m.adapter
        self.tl = threading.local()
        for o in self.models[1:]:                              # the shared part must be the same weights: everything that is not trained
            for k, p in m.net.params.items():
                if not p.requires_grad:
                    assert torch.equal(p, o.net.params[k]), f"frozen weight {k} differs between ensemble members"

    def __call__(self, H, ego, tc, **kw):
        outs = self.torch.stack([m(H, ego, tc, **kw).float() for m in self.models])
        self.tl.members = outs
        return outs.mean(0)


def make_core(tags, dev: str = "cuda", cold: str = "", motion: float = 1.0):
    """-> (core with the averaged model, "ap2" | "sh30": which single driver wraps it)."""
    import torch
    import ap2_core as AC
    import pp_train as T
    import sh30_core as C
    std = []
    for t in tags:
        ck = torch.load(T.proot("runs", t) / "ckpt-final.pt", map_location="cpu", weights_only=False)
        ap = ck.get("ap2") or {}
        assert not ap.get("route"), f"{t}: route arms are not supported in the ensemble"
        std.append(("ap2", ap.get("cold", "backwarp")) if ap.get("std", "navsim") == "alpasim" else ("sh30", "backwarp"))
    assert len(set(std)) == 1, f"ensemble members must share an input standard and cold-start rule: {dict(zip(tags, std))}"
    kind = std[0][0]
    core = AC.Core(tags[0], dev, cold, motion) if kind == "ap2" else C.Core(tags[0], dev, cold or "backwarp", motion)
    if len(tags) > 1:
        core.model = Mean([core.model] + [T.load_pmodel(t, core.dev) for t in tags[1:]])
        core.tag = "ENS-" + "+".join(tags)
    return core, kind


def driver_class(kind: str):
    import ap2_driver as AD
    import sh30_driver as D
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    base = AD.Driver if kind == "ap2" else D.Driver

    class Driver(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.mem = open(self.dir / "members.jsonl", "a", buffering=1)

        def drive(self, req, ctx):
            s = self.sessions.get(req.session_uuid)
            out = super().drive(req, ctx)
            mm = getattr(getattr(self.core.model, "tl", None), "members", None)
            if mm is not None and s is not None:               # the members' own exports on the state this decision was taken in
                mu = mm[:, 0, self.core.pi].reshape(len(mm), 33, 15).cpu().numpy()
                cam = np.asarray(s.cam["t"], np.float64)[:2]
                poses = [I.to_rear(m[:, 0:3], m[:, 11], I.T_IDXS, cam, Z.T_OUT, "lever").round(4).tolist() for m in mu]
                self.mem.write(json.dumps({"session": req.session_uuid, "scene": s.scene, "now": int(req.time_now_us), "k": s.count["drive"] - 1,
                                           "poses": poses}) + "\n")
            return out
    return Driver, D


def bench(tags, n: int = 60) -> dict:
    """Single-step latency of the first member alone and of the ensemble, full-history decisions on random frames, same process and card."""
    import sh30_core as C
    rng = np.random.default_rng(0)
    out = {}
    for name, tt in (("single", tags[:1]), ("ensemble", tags)):
        core, _ = make_core(tt)
        fr = [rng.integers(0, 255, C.FRAME, dtype=np.uint8) for _ in range(4)]
        pose = np.c_[np.array([-12.0, -8.0, -4.0, 0.0]), np.zeros(4), np.zeros(4)]
        ms = []
        for i in range(n + 10):
            o = core.plan(fr, pose, np.tile([8.0, 0.0], (4, 1)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
            if i >= 10:
                ms.append(o["ms"])
        out[name] = {k: float(np.median([m[k] for m in ms])) for k in ms[0]} | {"total": float(np.median([sum(m.values()) for m in ms])),
                                                                             "total_p90": float(np.percentile([sum(m.values()) for m in ms], 90)), "members": len(tt)}
        del core
    out["extra_ms"] = out["ensemble"]["total"] - out["single"]["total"]
    return out


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        print(json.dumps(bench(sys.argv[2].replace("+", ",").split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 60), indent=1))
        return
    import torch
    host, port = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0"), int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    log_dir = Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver"))
    t0 = time.time()
    tags = [t for t in os.environ["ENS_TAGS"].replace("+", ",").split(",") if t]
    core, kind = make_core(tags, os.environ.get("SH30_DEVICE", "cuda"), os.environ.get("ENS_COLD", ""), float(os.environ.get("SH30_MOTION", "1")))
    Driver, D = driver_class(kind)
    z = np.zeros((2, 6, 128, 256), np.uint8)
    for m in (1, 2, 3, 4, 4):                               # warm-up: every slot count compiled before the port opens
        core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
    LOG.info("%s (%s driver, cold %s, %d members) ready in %.1f s, VRAM %.2f GiB", core.tag, kind, core.cold, len(tags), time.time() - t0,
             torch.cuda.max_memory_allocated() / 2**30)
    drv = Driver(core, log_dir, int(os.environ.get("SH30_DUMP", "0")), os.environ.get("SH30_LHT", "0") == "1", os.environ.get("SH30_MOTION_GATE", ""))
    server = D.grpc.server(D.warm_workers(lambda: core.plan([z] * 4, np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])))
    D.egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(drv, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    server.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: server.stop(grace=0.0))
    LOG.info("listening on %s:%d", host, port)
    try:
        server.wait_for_termination()
    finally:
        (log_dir / "vram.json").write_text(json.dumps({"max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                       "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}))


if __name__ == "__main__":
    main()
