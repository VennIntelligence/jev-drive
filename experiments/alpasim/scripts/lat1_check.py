"""LAT1: the GPU slot warp (sh30_core.lattice_gpu / op_interp.warp_gpu) against the CPU reference on logged driver inputs, and the
latency of both. The logged driver-side messages of a finished run (c1_extract.py msgs) go back through the real driver class, no
simulator (c1_replay.py's harness). Results: experiments/alpasim/results/lat1_frame_synthesis.md.

  lat1_check.py equiv --msgs <dir> --out <json> [--n N]              (box, envs/op-train, one GPU; SH30_TAG as served)
      every decision is planned by the CPU core as run, then again by the CPU core (repeatability floor) and by the GPU core on the
      same inputs: differing pixels of the slot frames, 4 s endpoint distance, plan yaw at 0.5 s, stage times of a lone stream
  lat1_check.py remap --msgs <dir> --out <json> [--n N]
      the resampler alone: cv2.remap vs op_interp._remap_gpu on the CPU path's own maps (must be 0 differing pixels)
  lat1_check.py load --msgs <dir> --out <json> --synth cpu|gpu [--streams 8] [--n N]
      N scenes replayed as `streams` concurrent sessions through one driver (one inference lock, as served): stage times per `drive`
      and per image, from the driver's own drive.jsonl / images.jsonl
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
TAG = os.environ.get("SH30_TAG", "P2H10-F-s0")
Q = (50, 95, 100)


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def feed(drv, pb, msgs, ctx, on_drive=None):
    """One scene's messages through the driver in arrival order."""
    uuid = None
    for kind, raw in msgs:
        if kind == "driver_session_request":
            req = pb.DriveSessionRequest.FromString(raw)
            uuid = req.session_uuid
            drv.start_session(req, ctx)
        elif kind == "driver_camera_image":
            drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
        elif kind == "driver_ego_trajectory":
            drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
        elif kind == "route_request":
            drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
        elif kind == "driver_request":
            drv.drive(pb.DriveRequest.FromString(raw), ctx)
            if on_drive:
                on_drive()
    drv.sessions.pop(uuid, None)


def scenes(a):
    return sorted(Path(a.msgs).glob("*.pkl"))[:a.n or None]


def pct(x):
    return [round(float(v), 3) for v in np.percentile(np.asarray(x, float), Q)] if len(x) else []


def cmd_equiv(a):
    import sh30_driver as D
    cpu, gpu = D.C.Core(TAG, "cuda", synth="cpu"), D.C.Core(TAG, "cuda", synth="gpu")
    drv, last, rows = D.Driver(cpu, Path(tempfile.mkdtemp())), {}, []
    orig = cpu.plan

    def plan(*x, **k):
        last["x"], last["k"] = x, k
        last["o"] = orig(*x, **k)
        return last["o"]
    cpu.plan = plan
    z = np.zeros(D.C.FRAME, np.uint8)
    for c in (cpu, gpu):
        for m in (1, 2, 3, 4, 4):
            c.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])

    def on_drive():
        o, o2, g = last["o"], orig(*last["x"], **last["k"]), gpu.plan(*last["x"], **last["k"])
        d = o["cur"].astype(np.int16) - g["cur"].astype(np.int16)
        assert (o["valid"] == g["valid"]).all() and (o["cur"] == o2["cur"]).all()
        end = lambda p: float(np.hypot(*(p["poses"][-1, :2] - o["poses"][-1, :2])))        # noqa: E731
        yaw = lambda p: float(abs(np.degrees(p["poses"][0, 2] - o["poses"][0, 2])))        # noqa: E731
        rows.append(dict(scene=scene, k=len([r for r in rows if r["scene"] == scene]), n_keys=len(last["x"][0]),
                         px=int(o["cur"][o["valid"]].size), px_diff=int((d != 0).sum()), px_max=int(np.abs(d).max()),
                         frames_diff=int((d != 0).any((1, 2, 3, 4)).sum()), end_gpu=end(g), yaw_gpu=yaw(g), end_rep=end(o2), yaw_rep=yaw(o2),
                         mu_gpu=float(np.abs(g["mu"] - o["mu"]).max()), mu_rep=float(np.abs(o2["mu"] - o["mu"]).max()),
                         ms_cpu=o2["ms"], ms_gpu=g["ms"]))
    for i, f in enumerate(scenes(a)):
        scene = f.stem
        feed(drv, D.egodriver_pb2, pickle.load(open(f, "rb")), Ctx(), on_drive)
        if i % 20 == 0:
            print(i, scene, len(rows), sum(r["px_diff"] for r in rows), flush=True)
    R = lambda k: np.array([r[k] for r in rows])                                           # noqa: E731
    import torch
    s = dict(tag=TAG, scenes=len({r["scene"] for r in rows}), decisions=len(rows), cold=int((R("n_keys") < 4).sum()),
             px=int(R("px").sum()), px_diff=int(R("px_diff").sum()), px_max=int(R("px_max").max()), decisions_with_diff=int((R("px_diff") > 0).sum()),
             frames_diff=int(R("frames_diff").sum()),
             **{k: dict(max=float(R(k).max()), mean=float(R(k).mean()), nonzero=int((R(k) > 0).sum())) for k in
                ("end_gpu", "yaw_gpu", "mu_gpu", "end_rep", "yaw_rep", "mu_rep")},
             ms={c: {st: pct([r[f"ms_{c}"][st] for r in rows]) for st in ("frames", "encode", "policy", "export")} for c in ("cpu", "gpu")},
             ms_total={c: pct([sum(r[f"ms_{c}"].values()) for r in rows]) for c in ("cpu", "gpu")},
             vram_gib_two_cores=torch.cuda.max_memory_allocated() / 2**30)
    Path(a.out).write_text(json.dumps(dict(summary=s, rows=rows)))
    print(json.dumps(s, indent=1), flush=True)


def cmd_remap(a):
    import cv2
    import torch
    import sh30_driver as D
    I, pb, n, bad, worst = D.C.I, D.egodriver_pb2, 0, 0, 0
    rng = np.random.default_rng(0)
    for f in scenes(a):
        ims = [pb.RolloutCameraImage.FromString(raw) for kind, raw in pickle.load(open(f, "rb")) if kind == "driver_camera_image"]
        req = next(pb.DriveSessionRequest.FromString(raw) for kind, raw in pickle.load(open(f, "rb")) if kind == "driver_session_request")
        s = D.Session(req, 0)
        fr = D.C.pack(ims[len(ims) // 2].camera_image.image_bytes, s.cam)
        dst, src = np.r_[rng.uniform(-8, 8), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)], np.r_[rng.uniform(-1, 1, 2), rng.uniform(-0.1, 0.1)]
        for k, view in enumerate(("road", "wide")):
            mx, my = I.warp_map(view, s.cam["t"].astype(float), dst, src)
            Y, U, V = I.unpack(fr[k])
            hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
            for img, x, y in ((Y, mx, my), (np.ascontiguousarray(U), hx, hy), (np.ascontiguousarray(V), hx, hy)):
                ref = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                got = I._remap_gpu(*(torch.from_numpy(v[None]).cuda() for v in (img, x, y)))[0].cpu().numpy()
                d = np.abs(ref.astype(int) - got)
                n, bad, worst = n + d.size, bad + int((d > 0).sum()), max(worst, int(d.max()))
    s = dict(scenes=len(scenes(a)), px=n, px_diff=bad, px_max=worst, cv2=cv2.__version__)
    Path(a.out).write_text(json.dumps(s))
    print(json.dumps(s), flush=True)


def cmd_load(a):
    import torch
    import sh30_driver as D
    core = D.C.Core(TAG, "cuda", synth=a.synth)
    z = np.zeros(D.C.FRAME, np.uint8)
    for m in (1, 2, 3, 4, 4):
        core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
    log = Path(tempfile.mkdtemp())
    drv, todo, lock = D.Driver(core, log), [pickle.load(open(f, "rb")) for f in scenes(a)], threading.Lock()
    torch.cuda.reset_peak_memory_stats()

    def stream(_):
        core.plan([z] * 4, np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])      # per-thread CUDA init
        while True:
            with lock:
                if not todo:
                    return
                m = todo.pop()
            feed(drv, D.egodriver_pb2, m, Ctx())
    t0 = time.time()
    with ThreadPoolExecutor(a.streams) as ex:
        list(ex.map(stream, range(a.streams)))
    wall = time.time() - t0
    dr = [json.loads(x) for x in open(log / "drive.jsonl")]
    dr = [r["ms"] for r in dr if r["kind"] == "drive"]
    im = [json.loads(x)["pack_ms"] for x in open(log / "images.jsonl")]
    s = dict(tag=TAG, synth=a.synth, streams=a.streams, scenes=len(scenes(a)), drives=len(dr), wall_s=round(wall, 1), cpus=len(os.sched_getaffinity(0)),
             ms={k: pct([r[k] for r in dr]) for k in ("frames", "encode", "policy", "wait", "total")}, pack_ms=pct(im),
             vram_gib=torch.cuda.max_memory_allocated() / 2**30, vram_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    Path(a.out).write_text(json.dumps(s))
    print(json.dumps(s, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["equiv", "remap", "load"]), ap.add_argument("--msgs", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0), ap.add_argument("--synth", default="gpu"), ap.add_argument("--streams", type=int, default=8)
    a = ap.parse_args()
    {"equiv": cmd_equiv, "remap": cmd_remap, "load": cmd_load}[a.cmd](a)


if __name__ == "__main__":
    main()
