#!/usr/bin/env python
"""One resident zero-shot driving model serving any number of CARLA workers over a unix socket.
Pre-registration: todos/2026-09-24-zeroshot-exam/bench2drive.md. Agent side: scripts/b2d_zeroshot_agent.py.

  alpamayo   NVIDIA Alpamayo 1.5 (runs in third_party/alpamayo1.5/.venv): 4 cameras x 4 frames at 10 Hz, rendered
             as pinhole in CARLA and resampled here on the GPU into the native f-theta images at 576x320, 16-step
             ego history, optional nav text. Returns 64 rig-frame waypoints at 10 Hz.
  lebowski   comma openpilot model (runs in envs/openpilot): road + wide cameras at the 5 Hz context rate, warped
  (etc.)     into the model frames exactly as modeld does, desire from route commands. Returns the 33-point plan.

One process per model; each connection is one CARLA worker with its own recurrent state; the forwards are
serialised on the GPU by one lock. Protocol: scripts/zeroshot_wire.py. Requests carry {"cmd": "reset"|"plan"}.

    $DATA_DIR/third_party/alpamayo1.5/.venv/bin/python scripts/zeroshot_policy_server.py alpamayo \
        --socket $DATA_DIR/runs/zeroshot-exam/alpamayo.sock
    $DATA_DIR/envs/openpilot/bin/python scripts/zeroshot_policy_server.py lebowski \
        --socket $DATA_DIR/runs/zeroshot-exam/lebowski.sock
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import zeroshot_rigs as rigs  # noqa: E402
import zeroshot_wire as wire  # noqa: E402


class Alpamayo:
    def __init__(self, a):
        import torch
        from jevdrive.alpamayo import infer
        self.torch, self.infer = torch, infer
        self.cfg = infer.Config(name="b2d", attn=a.attn, flow_steps=a.flow_steps,
                                compile=tuple(x for x in a.compile.split(",") if x))
        self.model, self.processor = infer.load(self.cfg.attn)
        infer.apply(self.model, self.cfg)
        wh = np.array(rigs.ALPAMAYO_MODEL_WH, np.float32)
        self.grids, self.render_wh = [], []
        for cam in rigs.ALPAMAYO_CAMERAS:
            w, h, _, _ = rigs.alpamayo_render_spec(cam)
            g = rigs.alpamayo_sample_grid(cam)                      # render pixels, (320, 576, 2)
            g = (g + .5) / np.array([w, h], np.float32) * 2 - 1     # grid_sample, align_corners=False
            self.grids.append(torch.from_numpy(g)[None].cuda())
            self.render_wh.append((w, h))
        self.cam_idx = torch.tensor([c[1] for c in rigs.ALPAMAYO_CAMERAS])
        self.meta = {"model": "alpamayo", "config": vars(self.cfg), "versions": infer.versions()}
        # Warm-up with the real shapes: torch.compile and the first generate calls happen here, not in a route.
        dummy = {c[0]: np.zeros((4, h, w, 4), np.uint8) for c, (w, h) in zip(rigs.ALPAMAYO_CAMERAS, self.render_wh)}
        hist = np.zeros((16, 3), np.float32)
        hist[:, 0] = np.linspace(-7.5, 0, 16)
        for nav in (None, "Turn left in 30m", None):
            m = {"nav_text": nav, "seed": 0}
            self.plan({}, m, self.prepare(m, dict(dummy, hist=hist)))

    def new_state(self, state=None):
        return {}

    def images(self, arrays):
        """BGRA renders -> (N_cam, 4, 3, 320, 576) uint8 model images, on the GPU."""
        F, torch = self.torch.nn.functional, self.torch
        out = []
        for cam, grid in zip(rigs.ALPAMAYO_CAMERAS, self.grids):
            x = torch.from_numpy(np.ascontiguousarray(arrays[cam[0]])).cuda(non_blocking=True)
            x = x[..., [2, 1, 0]].permute(0, 3, 1, 2).float()          # (4, 3, H, W) RGB
            y = F.grid_sample(x, grid.expand(len(x), -1, -1, -1), mode="bilinear", align_corners=False)
            out.append(y.round_().clamp_(0, 255).to(torch.uint8))
        return torch.stack(out)

    def prepare(self, meta, arrays):
        """Everything before the forward; runs outside the GPU lock so it overlaps another worker's forward."""
        torch = self.torch
        t0 = time.perf_counter()
        frames = self.images(arrays).cpu()
        h = arrays["hist"].astype(np.float64)                          # (16, 3) x, y, yaw in the t0 rig frame
        xyz = np.c_[h[:, :2], np.zeros(len(h))]
        c, s = np.cos(h[:, 2]), np.sin(h[:, 2])
        rot = np.zeros((len(h), 3, 3))
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
        data = {"image_frames": frames, "camera_indices": self.cam_idx,
                "ego_history_xyz": torch.from_numpy(xyz).float()[None, None],
                "ego_history_rot": torch.from_numpy(rot).float()[None, None]}
        inputs = self.infer.build_inputs(data, self.processor, nav_text=meta.get("nav_text"))
        return {"inputs": inputs, "frames": frames, "prep_ms": 1e3 * (time.perf_counter() - t0)}

    def plan(self, state, meta, prep):
        r = self.infer.run(self.model, prep["inputs"], self.cfg, timer=None, seed=int(meta.get("seed", 0)))
        xy = r["xyz"][0, :, :2].astype(np.float32)
        info = {"cot": r["cot"][0], "n_prompt": r["n_prompt"], "prep_ms": prep["prep_ms"],
                "infer_ms": 1e3 * r["wall"]}
        return info, {"xy": xy, "t": (np.arange(1, 65) * 0.1).astype(np.float32)}

    def finish(self, meta, prep, info, out):
        if meta.get("dump"):
            self.dump(meta["dump"], prep["frames"], out["xy"], meta, info)

    def dump(self, path, frames, xy, meta, info):
        """Model-input images of the newest frame (2x2: cross-left, front-wide / cross-right, tele), with the
        predicted path projected into front-wide, and the reasoning text: the visual check of the adapter."""
        from PIL import Image, ImageDraw
        mw, mh = rigs.ALPAMAYO_MODEL_WH
        canvas = Image.new("RGB", (2 * mw, 2 * mh + 40))
        for k, (col, row) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            canvas.paste(Image.fromarray(frames[k, -1].permute(1, 2, 0).numpy()), (col * mw, row * mh))
        uv, ok = rigs.alpamayo_project(rigs.ALPAMAYO_CAMERAS[1], np.c_[xy, np.zeros(len(xy))])
        d = ImageDraw.Draw(canvas)
        pts = [(float(u) + mw, float(v)) for (u, v), k in zip(uv, ok) if k]
        if len(pts) > 1:
            d.line(pts, fill=(255, 40, 40), width=3)
        d.text((4, 2 * mh + 4), "nav: %s | %s" % (meta.get("nav_text"), info["cot"][:150]), fill=(255, 255, 255))
        canvas.save(path, quality=90)


class OpenpilotModel:
    STATE = ("img_q", "desire_q", "feat_q", "prev_feat", "prev_desire", "n")

    def __init__(self, a):
        from jevdrive.openpilot import frames as opf
        from jevdrive.openpilot.model import OPModel, T_IDXS, decode
        self.opf, self.decode, self.t_idxs = opf, decode, T_IDXS.astype(np.float32)
        # Lebowski keeps its queues on the host: one step per 5 Hz context step, exact at that phase, and one model
        # serves every connection by swapping the host queues. small / Cinque keep their queues inside the ONNX on the
        # GPU and step at 20 Hz; each connection gets its own session (TensorRT engine from the cache, ~2.3 GB).
        self.context_rate = a.model == "lebowski"
        self.make = lambda: OPModel(a.model, a.backend, context_rate=self.context_rate)  # noqa: E731
        self.model = self.make()
        # small / Cinque: sessions are created once, up front (--pool, one per CARLA worker), and handed out per
        # connection; building one takes ~30 s and doing it per route was where the server twice died (2026-09-25)
        self.free = [] if self.context_rate else [self.make() for _ in range(max(0, a.pool - 1))] + [self.model]
        w, h = rigs.OP_CAMERA_WH
        self.idx = {}
        for name, f in rigs.OP_FOCAL.items():
            M = opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, f), name == "wide")
            y = opf._nn_index(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
            uv = opf._nn_index(M * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], np.float32),
                               (opf.MODEL_W // 2, opf.MODEL_H // 2), (w // 2, h // 2))
            r, c = np.divmod(uv, w // 2)
            quad = np.stack([(2 * r + i) * w + 2 * c + j for i in (0, 1) for j in (0, 1)])   # 2x2 block per chroma
            self.idx[name] = (y, quad)
        self.meta = {"model": a.model, "backend": a.backend, "step_hz": 5 if self.context_rate else 20}
        st = self.new_state()
        img = {"OP_ROAD": np.zeros((h, w, 4), np.uint8), "OP_WIDE": np.zeros((h, w, 4), np.uint8)}
        for _ in range(3):
            m = {"desire": 0, "speed": 0.0}
            self.plan(st, m, self.prepare(m, img))

    def new_state(self, state=None):
        if not self.context_rate:
            if state and "model" in state:
                m = state["model"]
            elif state is None:
                m = self.model
            else:
                m = self.free.pop() if self.free else self.make()
            m.reset()
            return {"model": m}
        self.model.reset()
        return {k: getattr(self.model, k).copy() if hasattr(getattr(self.model, k), "copy") else getattr(self.model, k)
                for k in self.STATE}

    def release(self, state):
        if not self.context_rate and state and "model" in state:
            self.free.append(state["model"])

    def pack(self, bgra, name):
        """CARLA BGRA -> 6x128x256 model-frame planes: BT.601 limited-range YUV (what the comma ISP delivers,
        checked on comma1M: Y spans 16..245) sampled at modeld's nearest-neighbour warp indices; chroma is the
        2x2 average, as in NV12."""
        opf = self.opf
        y_idx, quad = self.idx[name]
        px = bgra.reshape(-1, 4)
        b, g, r = (px[y_idx, k].astype(np.float32) for k in range(3))
        Y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).reshape(opf.MODEL_H, opf.MODEL_W)
        q = px[quad].astype(np.float32).mean(0)                       # (N, 4) BGRA block means
        b, g, r = q[:, 0], q[:, 1], q[:, 2]
        U = 128 - 0.148 * r - 0.291 * g + 0.439 * b
        V = 128 + 0.439 * r - 0.368 * g - 0.071 * b
        Y, U, V = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (Y, U, V))
        out = np.empty((6, opf.MODEL_H // 2, opf.MODEL_W // 2), np.uint8)
        out[0], out[1], out[2], out[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
        out[4] = U.reshape(opf.MODEL_H // 2, opf.MODEL_W // 2)
        out[5] = V.reshape(opf.MODEL_H // 2, opf.MODEL_W // 2)
        return out

    def prepare(self, meta, arrays):
        t0 = time.perf_counter()
        img2 = np.stack([self.pack(arrays["OP_ROAD"], "road"), self.pack(arrays["OP_WIDE"], "wide")])
        return {"img2": img2, "prep_ms": 1e3 * (time.perf_counter() - t0)}

    def plan(self, state, meta, prep):
        img2 = prep["img2"]
        desire = np.zeros(8, np.float32)
        desire[int(meta.get("desire", 0))] = 1
        t1 = time.perf_counter()
        if not self.context_rate:
            m = state["model"]
            raw = m.step(img2, desire=desire, traffic=(1, 0))
        else:
            m = self.model
            for k in self.STATE:
                setattr(m, k, state[k])
            raw = m.step(img2, desire=desire, traffic=(1, 0))
            for k in self.STATE:
                state[k] = getattr(m, k)
        d = self.decode(raw, m.slices, float(meta.get("speed", 0.0)))
        info = {"prep_ms": prep["prep_ms"], "infer_ms": 1e3 * (time.perf_counter() - t1),
                "curvature": d["curvature"], "accel": d["accel"], "engaged": d["engaged"]}
        return info, {"pos": d["plan_pos"].astype(np.float32), "vel": d["plan_vel"][:, 0].astype(np.float32),
                      "yaw": d["plan_yaw"].astype(np.float32), "t": self.t_idxs}

    def finish(self, meta, prep, info, out):
        if meta.get("dump"):
            self.dump(meta["dump"], prep["img2"], out["pos"], meta)

    def dump(self, path, img2, plan_pos, meta):
        """Road and wide model frames (Y) as the network sees them, with the plan projected into each."""
        from PIL import Image, ImageDraw
        opf = self.opf
        tiles = []
        for k, K in enumerate((opf.MEDMODEL_K, opf.SBIGMODEL_K)):
            im = Image.fromarray(opf.unpack_luma(img2[k])).convert("RGB")
            p = plan_pos[plan_pos[:, 0] > 1] + [0, 0, rigs.OP_MOUNT_RIG[2]]   # plan z is height above the road
            uvw = (K @ opf.VIEW_FROM_DEVICE @ p.T).T
            pts = [tuple(map(float, q[:2] / q[2])) for q in uvw]
            if len(pts) > 1:
                ImageDraw.Draw(im).line(pts, fill=(255, 40, 40), width=2)
            tiles.append(im)
        canvas = Image.new("RGB", (2 * opf.MODEL_W, opf.MODEL_H + 20))
        for k, im in enumerate(tiles):
            canvas.paste(im, (k * opf.MODEL_W, 0))
        ImageDraw.Draw(canvas).text((4, opf.MODEL_H + 4), "desire %s  v %.1f" % (meta.get("desire"), meta.get("speed", 0)),
                                    fill=(255, 255, 255))
        canvas.save(path, quality=90)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["alpamayo", "lebowski", "cinque", "small"])
    p.add_argument("--socket", required=True)
    p.add_argument("--ready-file", default="")
    p.add_argument("--attn", default="sdpa", help="alpamayo: sdpa decodes 29%% faster than FA2 on this card")
    p.add_argument("--compile", default="visual,expert", help="alpamayo: submodules to torch.compile")
    p.add_argument("--flow-steps", type=int, default=5, help="alpamayo: 5 moves the path 0.17 m, far below the "
                   "1.28 m seed noise floor (todos/2026-09-24-alpamayo-smoke)")
    p.add_argument("--pool", type=int, default=1, help="openpilot small / Cinque: sessions built at start-up")
    p.add_argument("--backend", default="trt", help="openpilot: onnxruntime backend (jevdrive/openpilot/model.py)")
    return p.parse_args()


def log_signals():
    """Block the catchable termination signals in every thread and log who sent one (sender pid and its command line)
    before exiting, so an external kill leaves a trace. SIGKILL cannot be caught; the launcher logs that case."""
    import signal
    sigs = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT}
    signal.pthread_sigmask(signal.SIG_BLOCK, sigs)

    def wait():
        info = signal.sigwaitinfo(sigs)
        try:
            cmd = Path("/proc/%d/cmdline" % info.si_pid).read_bytes().replace(b"\0", b" ").decode()[:300]
        except OSError:
            cmd = "?"
        print("%s received signal %d from pid %d uid %d: %s" % (time.strftime("%F %T"), info.si_signo, info.si_pid,
                                                              info.si_uid, cmd), flush=True)
        os._exit(128 + info.si_signo)
    threading.Thread(target=wait, daemon=True).start()


def main():
    log_signals()
    a = parse_args()
    t0 = time.time()
    policy = Alpamayo(a) if a.model == "alpamayo" else OpenpilotModel(a)
    print("%s ready after %.0f s" % (a.model, time.time() - t0), flush=True)
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    Path(a.socket).parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(64)
    if a.ready_file:
        Path(a.ready_file).write_text(json.dumps(policy.meta, default=str))
    gpu = threading.Lock()
    stats = {"calls": 0, "busy_s": 0.0, "t0": time.time()}

    def serve(conn):
        state = None
        try:
            while True:
                meta, arrays = wire.recv(conn)
                if meta["cmd"] == "reset":
                    with gpu:
                        state = policy.new_state(state if state is not None else {})
                    wire.send(conn, {"ok": True, "server": policy.meta}, {})
                    continue
                t = time.perf_counter()
                prep = policy.prepare(meta, arrays)
                t_prep = time.perf_counter()
                with gpu:
                    t_lock = time.perf_counter()
                    info, out = policy.plan(state, meta, prep)
                    busy = time.perf_counter() - t_lock
                stats["calls"] += 1
                stats["busy_s"] += busy
                info.update(queue_ms=1e3 * (t_lock - t_prep), server_ms=1e3 * (time.perf_counter() - t))
                wire.send(conn, info, out)
                policy.finish(meta, prep, info, out)
                if stats["calls"] % 200 == 0:
                    up = time.time() - stats["t0"]
                    print("calls %d, GPU-lock busy %.0f%% of %.0f s" % (stats["calls"], 100 * stats["busy_s"] / up, up),
                          flush=True)
        except ConnectionError:
            pass
        finally:
            conn.close()
            if hasattr(policy, "release"):
                with gpu:
                    policy.release(state)

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
