#!/usr/bin/env python
"""Serve real frozen features to the closed-loop agent over a unix socket.

The CARLA client wheel only exists for Python 3.7/3.8 (docs/carla.md) and torch for this card only
exists in the project's 3.11 env, so a policy in the loop is two processes whether we like it or
not. This is that second process, and it is also what makes the measurement honest: a `time.sleep`
stand-in leaves the GPU idle, while the real thing competes with CARLA's renderer for the same
card. The difference between the two is exactly the question "does closed-loop take hours or days".

Backbones, matching the ladder in research/carla-efficiency.md:
  dinov2  DINOv2 ViT-B/14 on the front camera. A cheap backbone in the loop.
  qwen    Qwen3-VL-4B, the decoder truncated at `--layers`. What we would actually run.

    $DATA_DIR/envs/jevdrive/bin/python scripts/b2d_policy_server.py \
        --socket /tmp/b2d-policy-0.sock --backbone qwen --width 1200 --n-images 3 --layers 18

Protocol, deliberately minimal: the client sends
    <uint32 n_images><uint32 width><uint32 height><uint64 nbytes><nbytes of BGR planes>
and gets back one float32, the server-side latency in milliseconds. One frame of three 1600x900
cameras is 12.9 MiB on the wire; `--bench` reports how much of the latency that transfer is.
"""
import argparse
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--backbone", default="qwen", choices=["qwen", "dinov2"])
    p.add_argument("--width", type=int, default=1200,
                   help="qwen: model input width. 1200 on a 16:9 frame gives ~1020 tokens per "
                        "camera, matching the 3060 tokens/frame of the Waymo front3 default "
                        "(docs/waymo-e2e.md), so the latency is comparable to the 129 ms we quote")
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--layers", default="18", help="qwen: decoder layers to read; the deepest one "
                                                  "is where the forward stops (decisions 5)")
    p.add_argument("--compile", action="store_true", help="torch.compile; costs minutes at startup")
    p.add_argument("--bench", type=int, default=0, help="run N local forwards and exit")
    p.add_argument("--bench-size", default="1600x900",
                   help="size of the frames the bench feeds in. CARLA's per-camera cost is "
                        "independent of resolution, so rendering straight at the model's input "
                        "size is free on the simulator side and removes the resize here")
    p.add_argument("--ready-file", default="")
    return p.parse_args()


def recv_exactly(conn, n):
    parts, got = [], 0
    while got < n:
        chunk = conn.recv(min(1 << 20, n - got))
        if not chunk:
            return None
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def build(a):
    import torch
    from PIL import Image
    from jevdrive.features import DinoFeatures, QwenFeatures

    split = {"transform_ms": [], "forward_ms": []}

    def timed(make_batch):
        def infer(arrays):
            t0 = time.perf_counter()
            batch = make_batch(arrays)
            t1 = time.perf_counter()
            fx(batch)
            torch.cuda.synchronize()
            split["transform_ms"].append(1e3 * (t1 - t0))
            split["forward_ms"].append(1e3 * (time.perf_counter() - t1))
        return infer

    if a.backbone == "dinov2":
        fx = DinoFeatures()
        infer = timed(lambda arrays: fx.collate(
            [fx.transform(Image.fromarray(arrays[0][:, :, ::-1]))]))
    else:
        fx = QwenFeatures(width=a.width, n_images=a.n_images, compile=a.compile,
                          layers=[int(x) for x in a.layers.split(",")])
        # CARLA hands out BGRA; the [:, :, ::-1] view is what the processor resizes, so the
        # channel flip never materialises an array of its own.
        infer = timed(lambda arrays: QwenFeatures.collate(
            [fx.transform([Image.fromarray(x[:, :, ::-1]) for x in arrays[:a.n_images]])]))

    return fx, infer, split


def main():
    a = parse_args()
    import numpy as np
    fx, infer, split = build(a)

    # Warm up: the first calls build the batch-shape constants and, with compile, the graphs.
    bw, bh = (int(x) for x in a.bench_size.split("x"))
    dummy = [np.zeros((bh, bw, 3), dtype=np.uint8) for _ in range(max(1, a.n_images))]
    for _ in range(3):
        infer(dummy)
    if a.backbone == "qwen":
        print("qwen image tokens per frame: %d" % fx.n_image_tokens, flush=True)

    if a.bench:
        ts = []
        for _ in range(a.bench):
            t0 = time.perf_counter()
            infer(dummy)
            ts.append(1e3 * (time.perf_counter() - t0))
        ts.sort()
        med = lambda xs: sorted(xs)[len(xs) // 2]
        n = len(split["forward_ms"])
        print("local: total median %.1f ms (min %.1f max %.1f), transform %.1f, forward %.1f, "
              "over %d" % (ts[len(ts) // 2], ts[0], ts[-1],
                           med(split["transform_ms"][-a.bench:]),
                           med(split["forward_ms"][-a.bench:]), len(ts)), flush=True)
        return 0

    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(8)
    if a.ready_file:
        Path(a.ready_file).write_text("ready")
    print("policy server ready on %s" % a.socket, flush=True)

    # One connection per worker, so the server has to be able to hold several at once. Serving
    # them from a single accept loop looks fine with one worker and silently starves workers 2..N
    # with four - their requests sit in the backlog, their inference never returns, and the agent
    # drives on a stale control while reporting nothing wrong.
    gpu = threading.Lock()

    def serve(conn):
        try:
            while True:
                head = recv_exactly(conn, 20)
                if head is None:
                    return
                n, w, h, nbytes = struct.unpack("<IIIQ", head)
                payload = recv_exactly(conn, nbytes)
                if payload is None:
                    return
                t0 = time.perf_counter()
                per = h * w * 3
                arrays = [np.frombuffer(payload, dtype=np.uint8, count=per,
                                        offset=i * per).reshape(h, w, 3) for i in range(n)]
                with gpu:  # one model, one card: the forwards serialise anyway, do it explicitly
                    infer(arrays)
                conn.sendall(struct.pack("<f", 1e3 * (time.perf_counter() - t0)))
        except (OSError, struct.error):
            return
        finally:
            conn.close()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
