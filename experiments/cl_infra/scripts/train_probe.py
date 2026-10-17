"""Step-time probe of an rm_train run (docs/speed.md): wraps pp_train.Prefetch.get, which the training loop calls once per step.

  python train_probe.py --out OUT.json [--prof] -- <tree>/experiments/route_mem/scripts/rm_train.py train --tag T ...

Per step it records the wall time between consecutive get() calls (the whole step) and the time spent inside get() (waiting for the
prefetched batch: > 0 means the step is data-bound). With --prof, torch.profiler records steps 80-99 and the summary gets the GPU kernel
time against the wall time of those steps and the CPU time spent in host-device synchronisation. Nothing in the run changes: the
patched get() returns the same batch in the same order; the profiled run is not used for timing.
"""
import argparse, atexit, json, os, pathlib, runpy, sys, time  # noqa: E401

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prof", action="store_true")
ap.add_argument("--skip", type=int, default=20, help="first steps left out of the statistics (warm-up)")
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
script = pathlib.Path(rest[0]).resolve()
tree = script.parents[3]
sys.path[:0] = [str(tree / "experiments/op_parity/scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pp_train as T  # noqa: E402

P0, P1 = 80, 100                                           # profiled steps [P0, P1)
rec = dict(step=[], wait=[])
st = dict(t=None, n=0, prof=None, t_p0=None, t_p1=None)
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpyAsync", "cudaMemcpy",
        "aten::item", "aten::_local_scalar_dense")


def _prof_start():
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    st["prof"] = torch.profiler.profile(activities=acts)
    st["prof"].__enter__()


def _prof_stop():
    torch.cuda.synchronize()
    st["t_p1"] = time.perf_counter()
    p, st["prof"] = st["prof"], None
    p.__exit__(None, None, None)
    ka = p.key_averages()
    dev_us = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka
                 if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type))
    if not dev_us:                                         # older event attribute layout: kernels carry their own device time
        dev_us = sum(getattr(e, "self_device_time_total", 0) for e in ka)
    sync = {e.key: dict(n=e.count, cpu_ms=round(e.cpu_time_total / 1e3, 1)) for e in ka if e.key in SYNC}
    top = sorted(ka, key=lambda e: -getattr(e, "self_device_time_total", 0))[:15]
    wall = st["t_p1"] - st["t_p0"]
    rec["prof"] = dict(steps=P1 - P0, wall_ms=round(wall * 1e3, 1), gpu_kernel_ms=round(dev_us / 1e3, 1),
                       gpu_busy=round(dev_us / 1e6 / wall, 3), sync=sync,
                       top_gpu=[(e.key[:60], round(getattr(e, "self_device_time_total", 0) / 1e3, 1), e.count) for e in top],
                       top_cpu=[(e.key[:60], round(e.self_cpu_time_total / 1e3, 1), e.count)
                                for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:15]])


_get = T.Prefetch.get


def get(self):
    t = time.perf_counter()
    if st["t"] is not None:
        rec["step"].append(t - st["t"])
    if a.prof and st["n"] == P0:
        torch.cuda.synchronize()
        st["t_p0"] = time.perf_counter()
        _prof_start()
    if a.prof and st["n"] == P1 and st["prof"] is not None:
        _prof_stop()
    t = time.perf_counter()
    b = _get(self)
    t2 = time.perf_counter()
    rec["wait"].append(t2 - t)
    st["t"], st["n"] = t2, st["n"] + 1
    return b


T.Prefetch.get = get


@atexit.register
def _dump():
    s, w = np.array(rec["step"][a.skip:]), np.array(rec["wait"][a.skip + 1:])
    out = dict(argv=rest[1:], steps=len(rec["step"]) + 1, card=os.environ.get("CUDA_VISIBLE_DEVICES"), t=time.time())
    if len(s):
        out |= dict(step_ms_median=round(1e3 * float(np.median(s)), 2), step_ms_mean=round(1e3 * float(s.mean()), 2),
                    step_ms_p90=round(1e3 * float(np.percentile(s, 90)), 2), it_s=round(1 / float(np.median(s)), 3),
                    wait_ms_median=round(1e3 * float(np.median(w)), 3), wait_ms_mean=round(1e3 * float(w.mean()), 3),
                    wait_share=round(float(w.sum() / s.sum()), 4))
    if torch.cuda.is_available():
        out["max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    if "prof" in rec:
        out["prof"] = rec["prof"]
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


sys.argv = [str(script)] + rest[1:]
runpy.run_path(str(script), run_name="__main__")
