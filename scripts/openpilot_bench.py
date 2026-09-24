"""Numerics and latency of openpilot driving models across onnxruntime backends.

Numerics: every backend replays the same input sequence (random frames, and real comma1M frames when
--seg is given) from a zero state and is compared with the reference backend on the decoded outputs
(plan positions, plan speed, desired curvature and acceleration), not only the raw vector.
Latency: one 20 Hz step at batch 1, as modeld does it (host inputs in, output vector back on the host).
Throughput: K independent streams (one session each, own thread) since the ONNX graphs are batch-1.

  python scripts/openpilot_bench.py --models small cinque lebowski --seg <comma1M dir>
"""
import argparse, json, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.openpilot.model import OPModel, decode
from jevdrive.runlog import RunLog

ort.preload_dlls()
KEYS = ("plan_x", "plan_y", "plan_v", "curvature", "accel")


def gpu_mem_mb():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return float(q.split()[0])


def replay(m, frames, v_ego=15.0):
    m.reset()
    raws = np.stack([m.step(f) for f in frames])
    dec = [decode(r, m.slices, v_ego) for r in raws]
    return raws, {"plan_x": np.stack([d["plan_pos"][:, 0] for d in dec]),
                  "plan_y": np.stack([d["plan_pos"][:, 1] for d in dec]),
                  "plan_v": np.stack([d["plan_vel"][:, 0] for d in dec]),
                  "curvature": np.array([d["curvature"] for d in dec]),
                  "accel": np.array([d["accel"] for d in dec])}


def latency(m, frames, warmup, n):
    for i in range(warmup):
        m.step(frames[i % len(frames)])
    ts = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        m.step(frames[i % len(frames)])
        ts[i] = time.perf_counter() - t
    return ts * 1e3


def throughput(name, backend, frames, k, n):
    models = [OPModel(name, backend) for _ in range(k)]
    for m in models:
        latency(m, frames, 20, 1)
    barrier = threading.Barrier(k + 1)

    def run(m):
        barrier.wait()
        for i in range(n):
            m.step(frames[i % len(frames)])

    th = [threading.Thread(target=run, args=(m,)) for m in models]
    [t.start() for t in th]
    barrier.wait()
    t0 = time.perf_counter()
    [t.join() for t in th]
    return k * n / (time.perf_counter() - t0), gpu_mem_mb()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small", "cinque", "lebowski"])
    ap.add_argument("--backends", nargs="+", default=["cuda", "cuda-iob", "cuda-graph", "trt", "trt-graph"])
    ap.add_argument("--ref", default="cpu")
    ap.add_argument("--seg", type=Path, help="comma1M segment dir with a packed frames cache (openpilot_replay)")
    ap.add_argument("--steps", type=int, default=100, help="sequence length for the numerics replay")
    ap.add_argument("--lat-n", type=int, default=1000)
    ap.add_argument("--streams", nargs="+", type=int, default=[1, 2, 4, 8])
    a = ap.parse_args()
    log = RunLog("openpilot_bench", "_".join(a.models))
    rng = np.random.default_rng(0)
    seqs = {"random": rng.integers(0, 256, (a.steps, 2, 6, 128, 256), dtype=np.uint8)}
    if a.seg:
        seqs["real"] = np.load(a.seg / "model_frames.npy", mmap_mode="r")[:a.steps]
    rows, lat_rows, thr_rows = [], [], []
    for name in a.models:
        ref = {}
        try:
            m = OPModel(name, a.ref, threads=0)
            ref = {s: replay(m, x) for s, x in seqs.items()}
            ref_name = a.ref
        except Exception as e:  # noqa: BLE001 - fp16-only graphs may lack CPU kernels
            log.info(f"{name}: reference {a.ref} failed ({str(e)[:120]}), using cuda")
            m = OPModel(name, "cuda")
            ref = {s: replay(m, x) for s, x in seqs.items()}
            ref_name = "cuda"
        del m
        for b in a.backends:
            t0 = time.perf_counter()
            try:
                m = OPModel(name, b)
                for s, x in seqs.items():
                    raw, dec = replay(m, x)
                    rr, rd = ref[s]
                    row = dict(model=name, backend=b, ref=ref_name, seq=s, n=len(x),
                               raw_max=float(np.abs(raw - rr).max()), raw_mean=float(np.abs(raw - rr).mean()))
                    for k in KEYS:
                        d = np.abs(dec[k] - rd[k])
                        row[k + "_max"], row[k + "_mean"] = float(d.max()), float(d.mean())
                    rows.append(row)
                    log.event("numerics", **row)
                x = seqs.get("real", seqs["random"])
                ts = latency(m, x, 100, a.lat_n)
                lr = dict(model=name, backend=b, n=a.lat_n, init_s=round(time.perf_counter() - t0, 1),
                          p50=float(np.percentile(ts, 50)), p99=float(np.percentile(ts, 99)),
                          mean=float(ts.mean()), max=float(ts.max()), gpu_mem_mb=gpu_mem_mb())
                lat_rows.append(lr)
                log.event("latency", **lr)
                log.info(f"{name:9s} {b:10s} p50 {lr['p50']:.2f} ms p99 {lr['p99']:.2f} ms "
                         f"curv max {rows[-1]['curvature_max']:.2e} accel max {rows[-1]['accel_max']:.2e}")
                del m
            except Exception as e:  # noqa: BLE001 - record and continue with the other backends
                log.info(f"{name} {b}: FAILED {type(e).__name__}: {str(e)[:300]}")
                lat_rows.append(dict(model=name, backend=b, error=str(e)[:300]))
        best = min((r for r in lat_rows if r["model"] == name and "p50" in r), key=lambda r: r["p50"])["backend"]
        for k in a.streams:
            sps, mem = throughput(name, best, seqs.get("real", seqs["random"]), k, 300)
            thr_rows.append(dict(model=name, backend=best, streams=k, steps_per_s=sps, gpu_mem_mb=mem))
            log.event("throughput", **thr_rows[-1])
            log.info(f"{name} {best} streams {k}: {sps:.0f} steps/s")
    info = dict(onnxruntime=ort.__version__, providers=ort.get_available_providers())
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    (log.dir / "results.json").write_text(json.dumps(dict(info=info, numerics=rows, latency=lat_rows,
                                                         throughput=thr_rows), indent=1))
    log.info(f"results -> {log.dir / 'results.json'}")
    log.close()
