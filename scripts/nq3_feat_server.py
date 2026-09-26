#!/usr/bin/env python
"""Feature servers behind the closed-loop head server (night queue 3, lane B; scripts/nq3_cl_server.py is the client).

  qwen   (repo .venv)        12 JPEGs (3 Waymo cameras x 4 frames at 5 Hz, camera-major, oldest first) -> Qwen3-VL
                             `L18_last`, the P5 extractor (waymo_qwenvid.make_fx, eager) on one clip item, decoded as
                             p4_carla.ClipFiles does, rounded to float16 as the stored P5 features are
  yolo   (envs/ultralytics)  3 JPEGs (current frame of each camera) -> the 672-d image-plane token row of night2_n4
                             (Detector on the three images in one call, build_tokens with the P5 v1 BA PCA)

One process per model and card, any number of connections, forwards serialised by one lock, batch 1 per request (the
offline equivalence check calls `features` below on the same bytes). Protocol: scripts/zeroshot_wire.py.

    .venv/bin/python scripts/nq3_feat_server.py qwen --socket $DATA_DIR/runs/nq3/b/sock/qwen-g0.sock
"""
import argparse
import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import zeroshot_wire as wire  # noqa: E402


class Qwen:
    """make_fx's transform split in two: the per-frame decode + resize (cached per connection by JPEG bytes: the 4-frame
    clip slides by one frame per plan, so 9 of its 12 frames were resized one plan earlier) and the processor call on
    the 12 resized frames; then the forward. `features` = the uncached path, what the offline check runs."""

    def __init__(self):
        import torch
        from jevdrive import waymo_qwenvid as qv
        self.torch, self.qv = torch, qv
        self.fx = qv.make_fx(compile=False)

    def _frame(self, b, cache):
        import hashlib
        from PIL import Image
        k = hashlib.sha1(memoryview(b)).digest()
        if k not in cache:
            if len(cache) > 32:
                cache.pop(next(iter(cache)))
            cache[k] = self.fx._resize(Image.open(io.BytesIO(bytes(b))).convert("RGB"))
        return cache[k]

    def prepare(self, blobs, cache):
        fr = [self._frame(b, cache) for b in blobs]
        n, F = len(self.qv.waymo.CAMS), self.qv.FRAMES
        o = self.fx.proc(text=[self.fx.prompt], videos=[fr[i * F:(i + 1) * F] for i in range(n)], return_tensors="pt")
        return self.fx.collate([(o["input_ids"][0], o["mm_token_type_ids"][0],
                                 o["pixel_values_videos"].to(self.torch.bfloat16), o["video_grid_thw"])])

    def forward(self, batch):
        with self.torch.inference_mode():
            f = self.fx(batch)["L18_last"]
        return {"q": f.to(self.torch.float16).float().cpu().numpy()[0]}

    def features(self, blobs):
        from PIL import Image
        imgs = [Image.open(io.BytesIO(bytes(b))).convert("RGB") for b in blobs]
        return self.forward(self.fx.collate([self.fx.transform(imgs)]))


class Yolo:
    def __init__(self):
        from jevdrive import night2_n4 as N4
        from jevdrive.nq3_cl import head_dir
        self.N4 = N4
        self.det = N4.Detector()
        z = np.load(head_dir() / "heads.npz")
        self.mu, self.V = z["pca_mu"], z["pca_V"]

    def prepare(self, blobs, cache=None):
        from jevdrive.sam_detect import decode
        return [np.ascontiguousarray(decode(bytes(b)).permute(1, 2, 0).numpy()[:, :, ::-1]) for b in blobs]

    def forward(self, ims):
        res = self.det(ims)
        return {"tok": self.N4.build_tokens(res, self.mu, self.V).astype(np.float32),
                "n_det": np.array([len(r[0]) for r in res], np.int32)}

    def features(self, blobs):
        return self.forward(self.prepare(blobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=("qwen", "yolo"))
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ready-file", default="")
    a = ap.parse_args()
    t0 = time.time()
    m = Qwen() if a.model == "qwen" else Yolo()
    n_in = 12 if a.model == "qwen" else 3
    for _ in range(2):          # warm-up at the real shapes (cuDNN / cuBLAS selection happens here)
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (972, 1079), (90, 90, 90)).save(buf, "JPEG", quality=95)
        m.features([np.frombuffer(buf.getvalue(), np.uint8)] * n_in)
    print("%s ready after %.0f s" % (a.model, time.time() - t0), flush=True)
    import socket
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    Path(a.socket).parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(64)
    if a.ready_file:
        Path(a.ready_file).write_text(a.model)
    lock = threading.Lock()
    stats = {"calls": 0, "busy": 0.0, "t0": time.time()}

    def serve(conn):
        cache = {}
        try:
            while True:
                meta, arrays = wire.recv(conn)
                blobs = [arrays["jpg%d" % i] for i in range(n_in)]
                t0_ = time.perf_counter()
                item = m.prepare(blobs, cache)          # CPU, outside the lock: overlaps other connections' forwards
                prep = time.perf_counter() - t0_
                with lock:
                    t = time.perf_counter()
                    out = m.forward(item)
                    busy = time.perf_counter() - t
                stats["calls"] += 1
                stats["busy"] += busy
                wire.send(conn, {"ms": round(1e3 * busy, 2), "prep_ms": round(1e3 * prep, 2)}, out)
                if stats["calls"] % 500 == 0:
                    up = time.time() - stats["t0"]
                    print("calls %d, busy %.0f%% of %.0f s" % (stats["calls"], 100 * stats["busy"] / up, up), flush=True)
        except ConnectionError:
            pass
        finally:
            conn.close()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
