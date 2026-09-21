#!/usr/bin/env python
"""Serve frozen Qwen3-VL features to the closed-loop agent over a unix socket.

The CARLA client wheel only exists for Python 3.7/3.8 (docs/carla.md) and torch for this card only
exists in the project's 3.11 env, so a policy in the loop is two processes whether we like it or
not. This is that second process, and it is also what makes the measurement honest: a `time.sleep`
stand-in leaves the GPU idle, while the real thing competes with CARLA's renderer for the same
card. The difference between the two is exactly the question "does closed-loop take hours or days".

    $DATA_DIR/envs/jevdrive/bin/python scripts/b2d_policy_server.py \
        --socket /tmp/b2d-policy-0.sock --width 800 --n-images 3

Protocol, deliberately minimal: the client sends
    <uint32 n_images><uint32 width><uint32 height><uint64 nbytes><nbytes of BGRA planes>
and gets back one float32, the server-side latency in milliseconds. Payload size at 3x1600x900 is
about 17 MiB per frame; `--recv-width` downsamples on this side only to show what the transfer
costs, never to change what the model sees.
"""
import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--width", type=int, default=800, help="model input width (decisions 4)")
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--layers", default="18", help="comma-separated decoder layers to read")
    p.add_argument("--compile", action="store_true", help="torch.compile; costs minutes at startup")
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


def main():
    a = parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from jevdrive.features import QwenFeatures

    feats = QwenFeatures(width=a.width, n_images=a.n_images, compile=a.compile,
                         layers=[int(x) for x in a.layers.split(",")])

    def infer(arrays):
        imgs = [Image.fromarray(arr[:, :, ::-1]) for arr in arrays]  # CARLA hands out BGRA
        batch = QwenFeatures.collate([feats.transform(imgs)])
        out = feats(batch)
        torch.cuda.synchronize()
        return out

    # Warm up: the first call builds the batch-shape constants and, with compile, the graphs.
    dummy = [np.zeros((900, 1600, 3), dtype=np.uint8) for _ in range(a.n_images)]
    for _ in range(3):
        infer(dummy)
    print("policy server ready on %s" % a.socket, flush=True)

    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(8)
    if a.ready_file:
        Path(a.ready_file).write_text("ready")

    while True:
        conn, _ = srv.accept()
        try:
            while True:
                head = recv_exactly(conn, 20)
                if head is None:
                    break
                n, w, h, nbytes = struct.unpack("<IIIQ", head)
                payload = recv_exactly(conn, nbytes)
                if payload is None:
                    break
                t0 = time.perf_counter()
                arrays = []
                per = h * w * 3
                for i in range(n):
                    arrays.append(np.frombuffer(payload, dtype=np.uint8, count=per,
                                                offset=i * per).reshape(h, w, 3))
                infer(arrays[:a.n_images])
                conn.sendall(struct.pack("<f", 1e3 * (time.perf_counter() - t0)))
        finally:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
