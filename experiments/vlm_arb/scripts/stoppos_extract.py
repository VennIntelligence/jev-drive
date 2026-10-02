"""Frozen Cinque features for the stop-position probe (openpilot venv; plan: plans/2026-10-03-stoppos-probe.md).

Every stream (a closed-loop attempt: saved 2 Hz road + wide JPEGs held for 10 steps of the 20 Hz clock; a P4 run: the
P5 recipe, 5 Hz three-camera frames held for 4 steps) starts from a zero state, no desire pulse, and is read at the last
held step. Output per stream: <out>/feats/<key>.npz with temporal (512), vision (512), hidden (16384, fp16), native
(2066: every output head but the hidden state, float32). Resumable (a finished stream is skipped).

  CUDA_VISIBLE_DEVICES=1 python stoppos_extract.py --out DIR --per-src 3        # staged: evenly spaced streams per source
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO") or Path(__file__).resolve().parents[3])
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
TAPS = ("select_4", "mean")
N_NATIVE = 2066
_IDX = {}


def cl_index():
    """Model-frame sampling indices of the closed-loop road / wide cameras (zeroshot_policy_server.OpenpilotModel.__init__)."""
    if not _IDX:
        import zeroshot_rigs as rigs
        from jevdrive.openpilot import frames as opf
        w, h = rigs.OP_CAMERA_WH
        for name, f in rigs.OP_FOCAL.items():
            M = opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, f), name == "wide")
            y = opf._nn_index(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
            uv = opf._nn_index(M * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], np.float32), (opf.MODEL_W // 2, opf.MODEL_H // 2), (w // 2, h // 2))
            r, c = np.divmod(uv, w // 2)
            quad = np.stack([(2 * r + i) * w + 2 * c + j for i in (0, 1) for j in (0, 1)])
            _IDX[name] = (y, quad)
        _IDX["hw"] = (opf.MODEL_H, opf.MODEL_W)
    return _IDX


def pack_bgr(img: np.ndarray, name: str) -> np.ndarray:
    """BGR image (1208, 1928, 3) -> (6, 128, 256) model planes; BT.601 limited range, chroma 2 x 2 mean (the server's `pack`)."""
    idx = cl_index()
    mh, mw = idx["hw"]
    y_idx, quad = idx[name]
    px = img.reshape(-1, 3)
    b, g, r = (px[y_idx, k].astype(np.float32) for k in range(3))
    Y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).reshape(mh, mw)
    q = px[quad].astype(np.float32).mean(0)
    b, g, r = q[:, 0], q[:, 1], q[:, 2]
    U = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    V = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    Y, U, V = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (Y, U, V))
    out = np.empty((6, mh // 2, mw // 2), np.uint8)
    out[0], out[1], out[2], out[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
    out[4] = U.reshape(mh // 2, mw // 2)
    out[5] = V.reshape(mh // 2, mw // 2)
    return out


def render_cl(files) -> np.ndarray:
    import cv2
    cv2.setNumThreads(1)
    out = np.empty((len(files), 2, 6, 128, 256), np.uint8)
    for j, (road, wide) in enumerate(files):
        for m, (p, name) in enumerate(((road, "road"), (wide, "wide"))):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            assert img is not None and img.shape == (1208, 1928, 3), (p, None if img is None else img.shape)
            out[j, m] = pack_bgr(img, name)
    return out


def init_worker():
    import p5_openpilot as P5S
    import wod_zeroshot_openpilot as WZ
    from jevdrive import p5_openpilot as P
    WZ._init({}, {P5S.SEQ: P.carla_calib()}, ".")


def job(st):
    if st["src"] == "cl":
        return st, render_cl(st["files"])
    import p5_openpilot as P5S
    return st, P5S.render(st["files"])


def run_stream(m, frames, hold):
    n = len(frames)
    temporal, vision = np.empty((n, 512), np.float32), np.empty((n, 512), np.float32)
    hidden, native = np.empty((n, 16384), np.float16), np.empty((n, N_NATIVE), np.float32)
    m.reset()
    for j in range(n):
        for _ in range(hold):
            o = m.step(frames[j])
        temporal[j], vision[j] = m.tap_values[TAPS[0]], m.tap_values[TAPS[1]]
        hidden[j], native[j] = o[N_NATIVE:N_NATIVE + 16384], o[:N_NATIVE]
    return dict(temporal=temporal, vision=vision, hidden=hidden, native=native)


def bounded_map(ex, fn, items, depth):
    from collections import deque
    it, q = iter(items), deque()
    for x in it:
        q.append(ex.submit(fn, x))
        if len(q) >= depth:
            break
    while q:
        yield q.popleft().result()
        x = next(it, None)
        if x is not None:
            q.append(ex.submit(fn, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--per-src", type=int, default=0, help="only this many evenly spaced streams per source (staged launch)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "feats").mkdir(parents=True, exist_ok=True)
    streams = json.load(open(a.streams))
    if a.per_src:
        sel = []
        for src in ("cl", "p4"):
            ss = [s for s in streams if s["src"] == src]
            sel += [ss[i] for i in np.unique(np.linspace(0, len(ss) - 1, a.per_src).astype(int))]
        streams = sel
    fname = lambda s: out / "feats" / (s["key"].replace("/", "__") + ".npz")  # noqa: E731
    todo = [s for s in streams if not fname(s).exists()]
    n_frames = sum(s["n"] for s in todo)

    def status(msg):
        (out / "STATUS").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " stoppos_extract: " + msg + "\n")
        print(msg, flush=True)
    status(f"{len(todo)} of {len(streams)} streams to do, {n_frames} frames")
    try:
        from jevdrive.openpilot.model import OPModel
        with ProcessPoolExecutor(a.workers, initializer=init_worker) as ex:
            list(ex.map(int, range(a.workers)))                 # fork the workers before the TensorRT session exists
            cache = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "cache/stoppos_trt"
            m = OPModel("cinque", "trt", cache=cache, taps=list(TAPS))
            status("model ready")
            t0, nf, ns, tg = time.time(), 0, 0, 0.0
            for st, frames in bounded_map(ex, job, todo, 2 * a.workers):
                t = time.perf_counter()
                arr = run_stream(m, frames, st["hold"])
                tg += time.perf_counter() - t
                f = fname(st)
                tmp = f.with_suffix(".tmp.npz")
                np.savez(tmp, **arr)
                tmp.replace(f)
                nf, ns = nf + st["n"], ns + 1
                el = time.time() - t0
                status(f"[{ns}/{len(todo)}] {nf}/{n_frames} frames, {el / nf * 1e3:.1f} ms/frame wall, model {tg / nf * 1e3:.1f} ms/frame, "
                       f"ETA {(n_frames - nf) * el / nf / 60:.0f} min")
        (out / "DONE").write_text(json.dumps(dict(streams=ns, frames=nf, wall_s=time.time() - t0)) + "\n")
    except BaseException:
        (out / "ERROR").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
