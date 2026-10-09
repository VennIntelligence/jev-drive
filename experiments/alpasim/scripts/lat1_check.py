"""LAT1: the GPU slot warp (sh30_core.lattice_gpu / op_interp.warp_gpu) against the CPU reference on logged driver inputs, and the
latency of both. The logged driver-side messages of a finished run (c1_extract.py msgs) go back through the real driver class, no
simulator (c1_replay.py's harness). Results: experiments/alpasim/results/lat1_frame_synthesis.md.

  lat1_check.py equiv --msgs <dir> --out <json> [--n N] [--driver ap2]    (box, envs/op-train, one GPU; SH30_TAG / AP2_TAG as served)
      every decision is planned by the CPU core as run, then again by the CPU core (repeatability floor) and by the GPU core on the
      same inputs: differing pixels of the slot frames, 4 s endpoint distance, plan yaw at 0.5 s, stage times of a lone stream
  lat1_check.py remap --msgs <dir> --out <json> [--n N]
      the resampler alone: cv2.remap vs op_interp._remap_gpu on the CPU path's own maps, and sh30_core.pack vs pack_fast and pack_gpu
      on every logged JPEG (both must be 0 differing pixels); for comparison the float sampler torch.nn.functional.grid_sample on the same maps
  lat1_check.py prof --msgs <dir> --out <json>
      where a decision's input work goes, alone on the job's cores: track, CPU lattice, GPU lattice, JPEG decode, model-frame packing
  lat1_check.py load --msgs <dir> --out <json> --synth cpu|gpu [--streams 8] [--n N] [--nosync] [--pack gpu]
      N scenes replayed as `streams` concurrent sessions through one driver (one inference lock, as served): stage times per `drive`
      and per image, from the driver's own drive.jsonl / images.jsonl, and every plan. --nosync: Core.sync off; --pack gpu: pack_gpu
      in place of pack_fast
  lat1_check.py nvjpeg --msgs <dir> --out <json> [--n N]
      a measurement, not a served path: the CAM_F0 JPEG decoded by nvJPEG on the card (torchvision.io.decode_jpeg, RGB -> YCbCr)
      instead of libjpeg: decode time, pixel difference of the model frames, plan difference on the logged inputs
  lat1_check.py runs --msgs <closed-loop run dir A> --out <run dir B>                    (no GPU)
      two closed-loop runs of one scene list: scene scores, and every decision's ego pose and plan (driver-logs/drive.jsonl) one
      against the other; the stage latency of both, and the runtime's own mean time per driver RPC (telemetry/metrics.prom)
  lat1_check.py model --msgs <dir> --out <json>
      where the encoder and policy time goes (jevdrive.op_torch interprets the ONNX graph node by node under vmap): graph nodes and
      aten calls per pass, host time to enqueue a pass against the time until the card has finished it, and the same for 2 and 4
      decisions in one pass (what cross-session batching would cost)
  lat1_check.py same --msgs <load json of one run> --out <load json of another>
      the plans of two load runs, decision by decision
"""
import argparse
import io
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
    if a.driver == "ap2":
        import ap2_driver as AD
        tag = os.environ.get("AP2_TAG", "AP2-AB-s0")
        cpu, gpu = AD.AC.Core(tag, "cuda", "", synth="cpu"), AD.AC.Core(tag, "cuda", "", synth="gpu")
        drv = AD.Driver(cpu, Path(tempfile.mkdtemp()))
    else:
        tag = TAG
        cpu, gpu = D.C.Core(TAG, "cuda", synth="cpu"), D.C.Core(TAG, "cuda", synth="gpu")
        drv = D.Driver(cpu, Path(tempfile.mkdtemp()))
    last, rows = {}, []
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
    s = dict(tag=tag, scenes=len({r["scene"] for r in rows}), decisions=len(rows), cold=int((R("n_keys") < 4).sum()),
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
    I, pb, n, bad, worst, pk, gs = D.C.I, D.egodriver_pb2, 0, 0, 0, [0, 0, 0], [0, 0]
    rng = np.random.default_rng(0)
    for f in scenes(a):
        ims = [pb.RolloutCameraImage.FromString(raw) for kind, raw in pickle.load(open(f, "rb")) if kind == "driver_camera_image"]
        req = next(pb.DriveSessionRequest.FromString(raw) for kind, raw in pickle.load(open(f, "rb")) if kind == "driver_session_request")
        s = D.Session(req, 0)
        fr = D.C.pack(ims[len(ims) // 2].camera_image.image_bytes, s.cam)
        for im in ims:
            j, ref = im.camera_image.image_bytes, D.C.pack(im.camera_image.image_bytes, s.cam).astype(int)
            for got in (D.C.pack_fast(j, s.cam), D.C.pack_gpu(j, s.cam, torch.device("cuda")).cpu().numpy()):
                d = np.abs(ref - got)
                pk = [pk[0] + d.size, pk[1] + int((d > 0).sum()), max(pk[2], int(d.max()))]
        dst, src = np.r_[rng.uniform(-8, 8), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)], np.r_[rng.uniform(-1, 1, 2), rng.uniform(-0.1, 0.1)]
        for k, view in enumerate(("road", "wide")):
            mx, my = I.warp_map(view, s.cam["t"].astype(float), dst, src)
            Y, U, V = I.unpack(fr[k])
            hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
            for img, x, y in ((Y, mx, my), (np.ascontiguousarray(U), hx, hy), (np.ascontiguousarray(V), hx, hy)):
                ref = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                got = I._remap_gpu(*(torch.from_numpy(v[None]).cuda() for v in (img, x, y)))[0].cpu().numpy()
                d = np.abs(ref.astype(int) - got)
                h, w = img.shape
                grid = torch.stack([torch.from_numpy(x).cuda() * (2 / (w - 1)) - 1, torch.from_numpy(y).cuda() * (2 / (h - 1)) - 1], -1)[None]
                f = torch.nn.functional.grid_sample(torch.from_numpy(img)[None, None].cuda().float(), grid, mode="bilinear", padding_mode="border",
                                                    align_corners=True)[0, 0].round().cpu().numpy()
                gs = [gs[0] + int((f != ref).sum()), max(gs[1], int(np.abs(f - ref).max()))]
                n, bad, worst = n + d.size, bad + int((d > 0).sum()), max(worst, int(d.max()))
    s = dict(scenes=len(scenes(a)), px=n, px_diff=bad, px_max=worst, pack_px=pk[0], pack_px_diff=pk[1], pack_px_max=pk[2],
             grid_sample_px_diff=gs[0], grid_sample_px_max=gs[1], cv2=cv2.__version__)
    Path(a.out).write_text(json.dumps(s))
    print(json.dumps(s), flush=True)


def cmd_prof(a):
    import torch
    import sh30_driver as D
    C, I, Z, pb = D.C, D.C.I, D.C.Z, D.egodriver_pb2
    msgs = pickle.load(open(scenes(a)[0], "rb"))
    s = D.Session(next(pb.DriveSessionRequest.FromString(raw) for kind, raw in msgs if kind == "driver_session_request"), 0)
    jpg = [pb.RolloutCameraImage.FromString(raw).camera_image.image_bytes for kind, raw in msgs if kind == "driver_camera_image"][:4]
    keys, cam_t, dev = np.stack([C.pack(j, s.cam) for j in jpg]), s.cam["t"].astype(np.float64), torch.device("cuda")
    pose, vel = np.array([[-12.0, 0.9, -0.12], [-8.1, 0.4, -0.08], [-4.0, 0.1, -0.03], [0, 0, 0]]), np.tile([8.0, 0.0], (4, 1))
    maps, out = C._MAPS[Z.calib_key({"CAM_F0": s.cam})], {}

    def clock(name, fn, n=100):
        fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t.append(1e3 * (time.perf_counter() - t0))
        out[name] = pct(t)
    track = I.track_navsim(pose, vel)
    clock("track: build", lambda: I.track_navsim(pose, vel))
    clock("track: 12 poses", lambda: [track(t) for t in C.SLOT_T[:6]] + [track(t) for t in I.T_KEY[[0, 1, 1, 2, 2, 3]]])
    clock("lattice cpu (8 threads)", lambda: C.lattice(keys, 0, track, cam_t, "backwarp"), 30)
    clock("lattice gpu + sync", lambda: (C.lattice_gpu(keys, 0, track, cam_t, "backwarp", dev), torch.cuda.synchronize()))
    K, pd, ps = torch.from_numpy(keys).to(dev), np.stack([track(t) for t in C.SLOT_T[:6]]), np.stack([track(t) for t in I.T_KEY[[0, 1, 1, 2, 2, 3]]])
    clock("warp_gpu + sync (6 frames)", lambda: (I.warp_gpu(K[[0, 1, 1, 2, 2, 3]], cam_t, pd, ps), torch.cuda.synchronize()))
    clock("keys -> device", lambda: (torch.from_numpy(keys).to(dev), torch.cuda.synchronize()))
    ycc = maps.decode(io.BytesIO(jpg[0]))
    clock("jpeg decode (PIL, YCbCr)", lambda: maps.decode(io.BytesIO(jpg[0])), 50)
    clock("model-frame packing (numpy)", lambda: maps(ycc), 50)
    clock("jpeg decode + packing: pack", lambda: C.pack(jpg[0], s.cam), 50)
    clock("jpeg decode + packing: pack_fast", lambda: C.pack_fast(jpg[0], s.cam), 50)
    clock("jpeg decode + packing: pack_gpu + sync", lambda: (C.pack_gpu(jpg[0], s.cam, dev), torch.cuda.synchronize()), 50)
    out = dict(cpus=len(os.sched_getaffinity(0)), jpeg_bytes=len(jpg[0]), ms_median_p95_max=out)
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(out, indent=1), flush=True)


def cmd_nvjpeg(a):
    import torch
    import torchvision.io as tio
    import sh30_driver as D
    dev = torch.device("cuda")
    core = D.C.Core(TAG, "cuda", synth="gpu")
    drv, last, pack, t_nv, t_cpu, px = D.Driver(core, Path(tempfile.mkdtemp())), {}, D.C.pack_gpu, [], [], [0, 0, 0.0, 0]
    orig = core.plan
    core.plan = lambda *x, **k: last.update(o=orig(*x, **k)) or last["o"]
    M = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], device=dev)

    def nv(jpeg):
        rgb = tio.decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8), device=dev).permute(1, 2, 0).float()
        return (rgb @ M.T + torch.tensor([0.0, 128.0, 128.0], device=dev)).round().clamp(0, 255).to(torch.uint8)

    def timed(jpeg, cam, dev_):
        t0 = time.perf_counter()
        ref = pack(jpeg, cam, dev_)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        got = pack(jpeg, cam, dev_, nv)
        torch.cuda.synchronize()
        t_cpu.append(1e3 * (t1 - t0)), t_nv.append(1e3 * (time.perf_counter() - t1))
        d = (ref.int() - got.int()).abs()
        px[:] = [px[0] + d.numel(), px[1] + int((d > 0).sum()), px[2] + float(d.sum()), max(px[3], int(d.max()))]
        return got if mode["nv"] else ref
    D.C.pack_gpu, mode, plans = timed, {"nv": False}, {False: [], True: []}
    for f in scenes(a):
        for nvm in (False, True):
            mode["nv"] = nvm
            feed(drv, D.egodriver_pb2, pickle.load(open(f, "rb")), Ctx(), lambda: plans[mode["nv"]].append(last["o"]["poses"].copy()))
    A, B = np.array(plans[False]), np.array(plans[True])
    end, yaw = np.hypot(*(A[:, -1, :2] - B[:, -1, :2]).T), np.abs(np.degrees(A[:, 0, 2] - B[:, 0, 2]))
    s = dict(tag=TAG, scenes=len(scenes(a)), decisions=len(A), images=len(t_nv) // 2, pack_ms_libjpeg=pct(t_cpu[2:]), pack_ms_nvjpeg=pct(t_nv[2:]),
             px_differing_share=px[1] / px[0], px_mean_abs=px[2] / px[0], px_max=px[3], end_4s_m=pct(end) + [float(end.mean())],
             yaw_05s_deg=pct(yaw) + [float(yaw.mean())], note="percentiles 50 / 95 / 100, then the mean")
    Path(a.out).write_text(json.dumps(s))
    print(json.dumps(s, indent=1), flush=True)


def cmd_model(a):
    import torch
    import sh30_driver as D
    core = D.C.Core(TAG, "cuda", synth="gpu")
    A, net, dev, out = core.A, core.model.net, core.dev, {}
    z = np.zeros(D.C.FRAME, np.uint8)
    for m in (1, 4, 4):
        o = core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
    ego = torch.from_numpy(o["ego"][None]).to(dev)

    def enc(b):
        x = torch.randint(0, 255, (8 * b,) + D.C.FRAME, dtype=torch.uint8, device=dev)
        return lambda: net.run_batched(A.vision_feeds(x, x), ["view_39"])["view_39"]

    def pol(b):
        H, e, tc = o["tokens"][None].repeat(b, 1, 1, 1), ego.repeat(b, 1), torch.tensor([[1.0, 0.0]], device=dev).repeat(b, 1)
        return lambda: core.model(H, e, tc)
    with torch.no_grad():
        for name, mk in (("encode", enc), ("policy", pol)):
            for b in (1, 2, 4):
                fn, enq, tot = mk(b), [], []
                for i in range(33):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn()
                    t1 = time.perf_counter()
                    torch.cuda.synchronize()
                    enq.append(1e3 * (t1 - t0)), tot.append(1e3 * (time.perf_counter() - t0))
                out[f"{name} x{b}"] = dict(host_enqueue_ms=pct(enq[3:]), until_done_ms=pct(tot[3:]))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                mk(1)()
            ev = prof.key_averages()
            out[f"{name} x1"].update(aten_calls=int(sum(e.count for e in ev if e.key.startswith("aten::"))),
                                     top=[[e.key, e.count] for e in sorted(ev, key=lambda e: -e.count)[:8]])
    out["graph_nodes"] = dict(encode=len(net.plan(tuple(sorted(A.VISION_IN)), ("view_39",))), total=len(net.nodes), dtype=str(net.dtype))
    out["cpus"] = len(os.sched_getaffinity(0))
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(out, indent=1), flush=True)


def cmd_load(a):
    import torch
    import sh30_driver as D
    core = D.C.Core(TAG, "cuda", synth=a.synth)
    core.sync = not a.nosync
    if a.pack == "gpu":
        D.C.pack_fast = lambda jpeg, cam: D.C.pack_gpu(jpeg, cam, core.dev)
    z = np.zeros(D.C.FRAME, np.uint8)
    for m in (1, 2, 3, 4, 4):
        core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
    log = Path(tempfile.mkdtemp())
    drv, todo, lock = D.Driver(core, log), [pickle.load(open(f, "rb")) for f in scenes(a)], threading.Lock()
    torch.cuda.reset_peak_memory_stats()

    def stream(_):
        with drv.gpu:                                    # per-thread CUDA init, one thread at a time as every later inference
            core.plan([z] * 4, np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
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
    plans = {f"{r['scene']}/{r['k']}": r["poses"] for r in dr if r["kind"] == "drive"}
    dr = [r["ms"] for r in dr if r["kind"] == "drive"]
    im = [json.loads(x)["pack_ms"] for x in open(log / "images.jsonl")]
    s = dict(tag=TAG, synth=a.synth, sync=core.sync, pack=a.pack if a.synth == "gpu" else "pack", streams=a.streams, scenes=len(scenes(a)), drives=len(dr), wall_s=round(wall, 1), cpus=len(os.sched_getaffinity(0)),
             ms={k: pct([r[k] for r in dr]) for k in ("frames", "encode", "policy", "wait", "total")}, pack_ms=pct(im),
             vram_gib=torch.cuda.max_memory_allocated() / 2**30, vram_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    print(json.dumps(s, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(dict(s, plans=plans)))


def rpc_means(run: Path) -> dict:
    """Runtime-side mean duration (ms) of every driver RPC, summed over the runtime workers."""
    import re
    acc = {}
    for m in re.finditer(r'^rpc_duration_seconds_(count|sum)\{method="(\w+)",service="driver"[^}]*\} (\S+)$', (run / "telemetry/metrics.prom").read_text(), re.M):
        acc.setdefault(m[2], {"count": 0.0, "sum": 0.0})[m[1]] += float(m[3])
    return {k: round(1e3 * v["sum"] / v["count"], 2) for k, v in acc.items() if v["count"]}


def cmd_runs(a):
    out = {}
    for name, run in (("a", Path(a.msgs)), ("b", Path(a.out))):
        rows = [json.loads(x) for x in open(run / "driver-logs/drive.jsonl")]
        dr = {(r["scene"], r["k"]): r for r in rows if r["kind"] == "drive"}
        sc = {r["clipgt_id"]: r["score"] for r in json.loads((run / "aggregate/results-summary.json").read_text())["rollouts"]}
        ns, im = json.loads((run / "native_summary.json").read_text()), [json.loads(x)["pack_ms"] for x in open(run / "driver-logs/images.jsonl")]
        out[name] = dict(run=str(run), scenes=len(sc), mean_score=float(np.mean(list(sc.values()))), zeros=int(sum(v == 0 for v in sc.values())),
                         decisions=len(dr), runtime_s=ns["times"]["runtime_s"], driver_gpu_mib=ns["peak"]["driver"]["gpu_mib"],
                         driver_cpu_s=ns["peak"]["driver"]["cpu_s"], driver_rss_gib=ns["peak"]["driver"]["rss_gib"],
                         ms={k: pct([r["ms"][k] for r in dr.values()]) for k in ("frames", "encode", "policy", "export", "prep", "wait", "total")},
                         pack_ms=pct(im), rpc_mean_ms=rpc_means(run))
        out[name + "_"] = dr, sc
    (da, sa), (db, sb) = out.pop("a_"), out.pop("b_")
    common = sorted(set(da) & set(db))
    dev = np.array([max(np.abs(np.array(da[k]["anchor"]) - np.array(db[k]["anchor"])).max(), np.abs(np.array(da[k]["poses"]) - np.array(db[k]["poses"])).max())
                    for k in common])
    ss = sorted(set(sa) & set(sb))
    out["cmp"] = dict(scenes_common=len(ss), scores_differing=int(sum(sa[x] != sb[x] for x in ss)), score_max_abs=float(max(abs(sa[x] - sb[x]) for x in ss)),
                      decisions_common=len(common), decisions_only_a=len(set(da) - set(db)), decisions_only_b=len(set(db) - set(da)),
                      decisions_differing=int((dev > 0).sum()), max_abs=float(dev.max()),
                      scenes_with_a_differing_decision=len({k[0] for k, d in zip(common, dev) if d > 0}))
    print(json.dumps(out, indent=1), flush=True)


def cmd_same(a):
    A, B = (json.loads(Path(f).read_text())["plans"] for f in (a.msgs, a.out))
    d = np.array([np.abs(np.array(A[k]) - np.array(B[k])).max() for k in A if k in B])
    print(json.dumps(dict(decisions_a=len(A), decisions_b=len(B), common=len(d), differing=int((d > 0).sum()), max_abs=float(d.max()))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["equiv", "remap", "prof", "load", "same", "nvjpeg", "runs", "model"]), ap.add_argument("--msgs", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0), ap.add_argument("--synth", default="gpu"), ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--nosync", action="store_true"), ap.add_argument("--pack", default="fast"), ap.add_argument("--driver", default="sh30")
    a = ap.parse_args()
    {"equiv": cmd_equiv, "remap": cmd_remap, "prof": cmd_prof, "load": cmd_load, "same": cmd_same, "nvjpeg": cmd_nvjpeg, "runs": cmd_runs, "model": cmd_model}[a.cmd](a)


if __name__ == "__main__":
    main()
