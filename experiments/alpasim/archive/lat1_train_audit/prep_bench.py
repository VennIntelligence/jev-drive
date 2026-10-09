"""Training-side audit, prep: per-token cost breakdown of the row / token cache builders (pp_prep, ap2_prep) on a small subset.
Measurement only; reads the existing op_parity tab of one navtrain shard, writes JSON next to itself.

  python prep_bench.py cpu  --out cpu.json  [--n 24]          single-process CPU microbenchmarks (run under the OMP env to compare)
  python prep_bench.py gpu  --out gpu.json  [--n 96]          encoder batch scaling, GPU lattice for the prep jobs + byte equality, end to end
"""
import sys, pathlib, os, time, json, argparse, io  # noqa: E401
_R = pathlib.Path(os.environ.get("JEV_REPO", pathlib.Path.home() / "data/jev-drive")).resolve()
sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts")]
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

DATA = "navtrain_full.s2of12"


def stat(x):
    x = np.asarray(x, float)
    return dict(median=round(float(np.median(x)), 3), p95=round(float(np.percentile(x, 95)), 3), mean=round(float(x.mean()), 3), n=len(x))


def rows(n, seed=0):
    """n tokens spread over the shard (so over many logs / calibrations), their entries, poses, velocities, yaw rates, cameras."""
    import ap2_prep as AP
    import ap2_inputs as AI
    from jevdrive.common import data_dir
    tab = np.load(data_dir() / "runs/op_parity/cache" / DATA / "tab.npz")
    N = len(tab["names"])
    sel = np.sort(np.random.default_rng(seed).choice(N, n, replace=False))
    t0 = time.perf_counter()
    ents = AP.entries(tab["names"][sel].tolist(), tab["log"][sel].tolist(), None)
    t_ent = time.perf_counter() - t0
    W = AI.yaw_rates(tab["pose"], tab["fut"])[sel]
    return sel, ents, tab["pose"][sel].astype(float), tab["vel"][sel].astype(float), W, tab["cam"][sel].astype(float), t_ent


def cmd_cpu(a):
    import cv2
    import ap2_prep as AP
    import pp_prep as PP
    import sh30_core as C
    import navsim_zs_openpilot as NZ
    from jevdrive import navsim_zs as Z, op_interp as I
    cv2.setNumThreads(1)
    sel, ents, pose, vel, W, cam, t_ent = rows(a.n)
    out = dict(cpus=len(os.sched_getaffinity(0)), omp=os.environ.get("OMP_NUM_THREADS"), n=a.n, entries_s=round(t_ent, 2), load1=os.getloadavg()[0])
    T = {k: [] for k in ("read", "decode", "pack_numpy", "pack_fast_total", "render_token", "warp_frame", "warp_map_x2", "remap_x2", "track",
                         "ap2_job_bw", "ap2_job_nobw", "pp_full_job")}
    eq_pack = 0
    for i, e in enumerate(ents):
        c = e["cams"][-1]["CAM_F0"]
        p = c["path"]
        t0 = time.perf_counter(); raw = open(p, "rb").read(); T["read"].append(time.perf_counter() - t0)                      # noqa: E702
        k = Z.calib_key({"CAM_F0": c})
        m = NZ._maps.get(k) or NZ._maps.setdefault(k, Z.OpenpilotMaps(c))
        t0 = time.perf_counter(); ycc = m.decode(io.BytesIO(raw)); T["decode"].append(time.perf_counter() - t0)                # noqa: E702
        t0 = time.perf_counter(); ref = m(ycc); T["pack_numpy"].append(time.perf_counter() - t0)                               # noqa: E702
        C.pack_fast(raw, c)
        t0 = time.perf_counter(); got = C.pack_fast(raw, c); T["pack_fast_total"].append(time.perf_counter() - t0)            # noqa: E702
        eq_pack += int(not np.array_equal(ref, got))
        t0 = time.perf_counter(); kf = NZ.render_token(e); T["render_token"].append(time.perf_counter() - t0)                 # noqa: E702
        t0 = time.perf_counter(); tr = I.track_navsim(pose[i], vel[i]); [tr(t) for t in C.SLOT_T]; T["track"].append(time.perf_counter() - t0)  # noqa: E702
        pd, ps = tr(-0.2), tr(0.0)
        t0 = time.perf_counter(); I.warp_frame(kf[3], cam[i], pd, ps); T["warp_frame"].append(time.perf_counter() - t0)       # noqa: E702
        t0 = time.perf_counter(); mm = [I.warp_map(v, cam[i], pd, ps) for v in ("road", "wide")]; T["warp_map_x2"].append(time.perf_counter() - t0)  # noqa: E702
        Y, U, V = I.unpack(kf[3][0])
        t0 = time.perf_counter()
        for mx, my in mm:
            cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
            cv2.remap(np.ascontiguousarray(U), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.remap(np.ascontiguousarray(V), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        T["remap_x2"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); AP._job((e, pose[i], vel[i], W[i], cam[i], True)); T["ap2_job_bw"].append(time.perf_counter() - t0)    # noqa: E702
        t0 = time.perf_counter(); AP._job((e, pose[i], vel[i], W[i], cam[i], False)); T["ap2_job_nobw"].append(time.perf_counter() - t0)  # noqa: E702
        t0 = time.perf_counter(); PP._full_job((e, pose[i], vel[i], cam[i], np.asarray(SYN_T))); T["pp_full_job"].append(time.perf_counter() - t0)  # noqa: E702
    out["ms"] = {k: stat(1e3 * np.asarray(v)) for k, v in T.items()}
    out["pack_fast_mismatch_frames"] = eq_pack
    out["calibrations"] = len(NZ._maps)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def _job_keys(args):
    """ap2_prep._job's key rendering only (what would stay on the CPU with a GPU lattice), with pack_fast or the shipped renderer."""
    import cv2
    cv2.setNumThreads(1)
    ent, fast = args
    t0 = time.perf_counter()
    if fast:
        import sh30_core as C
        c = ent["cams"][-1]["CAM_F0"]
        kf = np.stack([C.pack_fast(ent["cams"][f]["CAM_F0"]["path"], c) for f in range(4)])
    else:
        import navsim_zs_openpilot as NZ
        kf = NZ.render_token(ent)
    return kf, time.perf_counter() - t0


def _job_timed(args):
    import ap2_prep as AP
    t0 = time.perf_counter()
    r = AP._job(args)
    return r, time.perf_counter() - t0


def gpu_lattices(C, I, AP, torch, K, pose, vel, w, cam, dev, bw=True):
    """ap2_prep._job's frame path with sh30_core.lattice_gpu: K (4, 2, 6, 128, 256) uint8 tensor on dev -> cold (10, ...), back (3, 8, ...)."""
    cold = torch.zeros((AP.N_COLD,) + C.FRAME, dtype=torch.uint8, device=dev)
    back = torch.zeros((3, 8) + C.FRAME, dtype=torch.uint8, device=dev) if bw else None
    cam = np.asarray(cam, np.float64)
    for m in (1, 2, 3):
        e = 4 - m
        P, V = C.fill_history(pose[e:], vel[e:], float(w[e]))
        Kc = torch.zeros_like(K)
        Kc[e:] = K[e:]
        tr = I.track_navsim(P, V)
        if bw:                                           # one lattice per m: the zero-rule slots are the real slots of the backwarp lattice
            cur, _ = C.lattice_gpu(Kc, e, tr, cam, "backwarp", dev)
            back[m - 1] = cur
            cold[AP.COLD_AT[m]] = cur[8 - AP.AI.N_SLOT[m]:]
        else:
            cur, valid = C.lattice_gpu(Kc, e, tr, cam, "zero", dev)
            cold[AP.COLD_AT[m]] = cur[torch.as_tensor(valid, device=dev)]
    return cold, back


def cmd_gpu(a):
    import torch
    import ap2_prep as AP
    import pp_prep as PP
    import sh30_core as C
    from jevdrive import op_adapt as A, op_interp as I
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.vram / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30))
    sel, ents, pose, vel, W, cam, t_ent = rows(a.n)
    ncpu = len(os.sched_getaffinity(0))
    out = dict(cpus=ncpu, n=a.n, entries_s=round(t_ent, 2), load1=os.getloadavg()[0], gpu=torch.cuda.get_device_name(0))
    net, enc = PP.encoder(dev)

    # ---- 1. end to end as ap2_prep runs today (--bw), `workers` processes; where the wall time goes
    workers = max(1, ncpu - 1)
    t_enc, t_job, t_main = [], [], []
    with ProcessPoolExecutor(workers) as pool, ThreadPoolExecutor(6) as ex:
        list(pool.map(_job_keys, [(ents[0], False)] * workers))                    # start the workers
        chunks = [np.arange(i, min(i + 32, a.n)) for i in range(0, a.n, 32)]

        def load(rws):
            res = list(pool.map(_job_timed, [(ents[i], pose[i], vel[i], W[i], cam[i], True) for i in rws]))
            t_job.extend(r[1] for r in res)
            t0 = time.perf_counter()
            o = rws, np.stack([r[0][0] for r in res]), np.stack([r[0][1] for r in res])
            t_main.append(time.perf_counter() - t0)
            return o
        cold_ref, back_ref = np.zeros((a.n, 10) + C.FRAME, np.uint8), np.zeros((a.n, 3, 8) + C.FRAME, np.uint8)
        tok_ref = np.zeros((a.n, 3, 8) + A.H_SHAPE, np.float16)
        tokc_ref = np.zeros((a.n, 10) + A.H_SHAPE, np.float16)
        T0 = time.perf_counter()
        for rws, cf, bf in PP._bounded(ex, load, chunks, 12):
            b = len(rws)
            torch.cuda.synchronize(); t0 = time.perf_counter()                                                                  # noqa: E702
            for m, sl in AP.COLD_AT.items():
                tokc_ref[rws, sl] = enc(*AP.pairs(cf[:, sl])).reshape(b, AP.AI.N_SLOT[m], *A.H_SHAPE)
            tok_ref[rws] = enc(*AP.pairs(bf.reshape(b * 3, 8, *C.FRAME))).reshape(b, 3, 8, *A.H_SHAPE)
            torch.cuda.synchronize(); t_enc.append(time.perf_counter() - t0)                                                    # noqa: E702
            cold_ref[rws], back_ref[rws] = cf, bf
        wall = time.perf_counter() - T0
    out["e2e_today"] = dict(workers=workers, tokens_per_s=round(a.n / wall, 3), wall_s=round(wall, 2), worker_s_per_token=stat(t_job),
                            worker_busy_share=round(sum(t_job) / (wall * workers), 3), enc_s=round(sum(t_enc), 2), enc_share_of_wall=round(sum(t_enc) / wall, 3),
                            stack_s=round(sum(t_main), 2), pairs_per_token=34, enc_pairs_per_s_while_running=round(34 * a.n / sum(t_enc), 1))

    pathlib.Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)                 # noqa: E702

    # ---- 2. encoder: batch scaling on real frames (host enqueue vs until done), bs 128 = today
    pr, cu = AP.pairs(back_ref[: min(a.n, 48)].reshape(-1, 8, *C.FRAME))
    pr, cu = pr[:1024], cu[:1024]
    P_, C_ = torch.from_numpy(pr).to(dev), torch.from_numpy(cu).to(dev)
    encs = {}
    with torch.no_grad():
        for bs in (8, 32, 64, 128, 192):
            if bs > len(cu):
                continue
            try:
                torch.cuda.reset_peak_memory_stats()
                enq, tot = [], []
                for rep in range(5):
                    torch.cuda.synchronize(); t0 = time.perf_counter()                                                          # noqa: E702
                    o = net.run_batched(A.vision_feeds(P_[:bs], C_[:bs]), ["view_39"])["view_39"]
                    t1 = time.perf_counter(); torch.cuda.synchronize()                                                          # noqa: E702
                    enq.append(t1 - t0), tot.append(time.perf_counter() - t0)
                encs[bs] = dict(host_enqueue_ms=round(1e3 * np.median(enq[1:]), 2), until_done_ms=round(1e3 * np.median(tot[1:]), 2),
                                pairs_per_s=round(bs / np.median(tot[1:]), 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
                del o
            except torch.OutOfMemoryError:
                encs[bs] = "oom"
                torch.cuda.empty_cache()
        # the enc() wrapper as called (numpy in, numpy out): upload + download + astype
        t0 = time.perf_counter(); r128 = enc(pr[:512], cu[:512], bs=128); t_w = time.perf_counter() - t0                       # noqa: E702
        encs["wrapper_bs128_pairs_per_s"] = round(512 / t_w, 1)
        r512 = enc(pr[:512], cu[:512], bs=64)
        encs["bs64_vs_bs128_max_abs_diff"] = float(np.abs(r128.astype(np.float32) - r512.astype(np.float32)).max())
        encs["bs64_vs_bs128_identical"] = bool(np.array_equal(r128, r512))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            net.run_batched(A.vision_feeds(P_[:128], C_[:128]), ["view_39"])["view_39"]
            torch.cuda.synchronize()
        ev = prof.key_averages()
        cu_attr = "self_device_time_total" if hasattr(ev[0], "self_device_time_total") else "self_cuda_time_total"
        encs["bs128_profile"] = dict(aten_calls=int(sum(e.count for e in ev if e.key.startswith("aten::"))),
                                     cuda_kernel_ms=round(sum(getattr(e, cu_attr) for e in ev) / 1e3, 1),
                                     top_cuda=[[e.key, round(getattr(e, cu_attr) / 1e3, 1)] for e in sorted(ev, key=lambda e: -getattr(e, cu_attr))[:6]])
    out["encoder"] = encs

    # ---- 2b. the interpreter keeps every intermediate value of the graph alive until it returns (env dict): free each value after its
    #          last use (same ops, same order) and compare peak memory, time and bytes
    def run_free(feeds, want):
        names = sorted(feeds)
        ks = net.plan(tuple(names), tuple(want))
        last = {}
        for k in ks:
            for i in net.nodes[k][1]:
                last[i] = k
        keep = set(want)

        def f(*xs):
            env = dict(zip(names, xs))
            for k in ks:
                op, ins, outs, at = net.nodes[k]
                x = [env[i] if i in env else (net._w(i) if i else None) for i in ins]
                y = getattr(net, "op_" + op)(x, at, ins)
                env.update(zip(outs, y if isinstance(y, (list, tuple)) else [y]))
                del x, y
                for i in ins:
                    if last.get(i) == k and i in env and i not in keep:
                        del env[i]
            return {w: env[w] for w in want}
        return torch.func.vmap(f)(*[feeds[n] for n in names])
    fr = {}
    with torch.no_grad():
        for name, fn in (("today", lambda f: net.run_batched(f, ["view_39"])), ("free_after_last_use", lambda f: run_free(f, ["view_39"]))):
            for bs in (32, 128):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()                # noqa: E702
                ts = []
                for rep in range(4):
                    torch.cuda.synchronize(); t0 = time.perf_counter()                                                          # noqa: E702
                    o = fn(A.vision_feeds(P_[:bs], C_[:bs]))["view_39"]
                    torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)                                               # noqa: E702
                fr[f"{name}_bs{bs}"] = dict(ms=round(1e3 * np.median(ts[1:]), 1), peak_gb_above_inputs=round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2))
                fr[f"{name}_bs{bs}_out"] = o.cpu()
                del o
        for bs in (32, 128):
            fr[f"identical_bs{bs}"] = bool(torch.equal(fr.pop(f"today_bs{bs}_out"), fr.pop(f"free_after_last_use_bs{bs}_out")))
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); ts = []                                                 # noqa: E702
            for rep in range(3):
                torch.cuda.synchronize(); t0 = time.perf_counter()                                                              # noqa: E702
                o = run_free(A.vision_feeds(P_[:512], C_[:512]), ["view_39"])["view_39"]
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)                                                   # noqa: E702
            fr["free_after_last_use_bs512"] = dict(ms=round(1e3 * np.median(ts[1:]), 1), pairs_per_s=round(512 / np.median(ts[1:]), 1),
                                                   peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            del o
        except torch.OutOfMemoryError:
            fr["free_after_last_use_bs512"] = "oom"
    out["encoder_free_values"] = fr
    del P_, C_
    torch.cuda.empty_cache()
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)                 # noqa: E702

    # ---- 3. GPU lattice for the prep job: bytes against the CPU job, time per token, tokens of both
    n_eq = min(a.n, a.n_eq)
    diff_px, tot_px, t_gpu, t_up = 0, 0, [], []
    tok_gpu = np.zeros((n_eq, 3, 8) + A.H_SHAPE, np.float16)
    with torch.no_grad():
        for i in range(n_eq):
            kf = _job_keys((ents[i], False))[0]
            torch.cuda.synchronize(); t0 = time.perf_counter()                                                                  # noqa: E702
            K = torch.from_numpy(kf).to(dev)
            t1 = time.perf_counter()
            cold, back = gpu_lattices(C, I, AP, torch, K, pose[i], vel[i], W[i], cam[i], dev)
            torch.cuda.synchronize(); t_gpu.append(time.perf_counter() - t1); t_up.append(t1 - t0)                              # noqa: E702
            cn, bn = cold.cpu().numpy(), back.cpu().numpy()
            diff_px += int((cn != cold_ref[i]).sum()) + int((bn != back_ref[i]).sum())
            tot_px += cn.size + bn.size
            pv = torch.cat([torch.zeros_like(back[:, :1]), back[:, :-1]], 1).reshape(-1, *C.FRAME)
            tok_gpu[i] = net.run_batched(A.vision_feeds(pv, back.reshape(-1, *C.FRAME)), ["view_39"])["view_39"].reshape(3, 8, *A.H_SHAPE).cpu().numpy()
    out["gpu_lattice"] = dict(tokens=n_eq, pixels=tot_px, differing_pixels=diff_px, ms_per_token_3_lattices=stat(1e3 * np.asarray(t_gpu[2:])),
                              upload_ms=stat(1e3 * np.asarray(t_up[2:])),
                              tokens_identical_to_cpu_path=bool(np.array_equal(tok_gpu, tok_ref[:n_eq])),
                              tokens_max_abs_diff=float(np.abs(tok_gpu.astype(np.float32) - tok_ref[:n_eq].astype(np.float32)).max()))

    pathlib.Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)                 # noqa: E702
    torch.cuda.empty_cache()

    # ---- 4. end to end with keys on the CPU (shipped renderer / pack_fast), lattice per token + encoder per G tokens on the GPU;
    #         cold tokens deduplicated: a zero-rule pair differs from the backwarp pair only in the oldest real slot (27 pairs per token, not 34)
    G = 4
    for fast in (False, True):
        t_key, t_g = [], []
        with ProcessPoolExecutor(workers) as pool, torch.no_grad():
            list(pool.map(_job_keys, [(ents[0], fast)] * workers))
            T0 = time.perf_counter()
            mism = mism_c = 0
            buf = []

            def flush():
                nonlocal mism, mism_c
                back = torch.stack([b for _, _, b in buf])                                   # (g, 3, 8, ...)
                first = torch.stack([torch.stack([c[AP.COLD_AT[m]][0] for m in (1, 2, 3)]) for _, c, _ in buf])   # (g, 3, ...)
                pv = torch.cat([torch.zeros_like(back[:, :, :1]), back[:, :, :-1]], 2)
                cur = torch.cat([back.reshape(-1, *C.FRAME), first.reshape(-1, *C.FRAME)])
                prv = torch.cat([pv.reshape(-1, *C.FRAME), torch.zeros_like(first).reshape(-1, *C.FRAME)])
                h = net.run_batched(A.vision_feeds(prv, cur), ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                g = len(buf)
                hb, hf = h[: g * 24].reshape(g, 3, 8, *A.H_SHAPE).cpu().numpy(), h[g * 24:].reshape(g, 3, *A.H_SHAPE).cpu().numpy()
                for k, (i, _, _) in enumerate(buf):
                    cold_tok = np.concatenate([np.concatenate([hf[k, m - 1][None], hb[k, m - 1, 8 - AP.AI.N_SLOT[m] + 1:]]) for m in (1, 2, 3)])
                    mism += int(not np.array_equal(hb[k], tok_ref[i]))
                    mism_c += int(not np.array_equal(cold_tok, tokc_ref[i]))
                buf.clear()
            for i, (kf, dt) in enumerate(pool.map(_job_keys, [(e, fast) for e in ents], chunksize=4)):
                t_key.append(dt)
                t0 = time.perf_counter()
                K = torch.from_numpy(kf).to(dev)
                cold, back = gpu_lattices(C, I, AP, torch, K, pose[i], vel[i], W[i], cam[i], dev)
                buf.append((i, cold, back))
                if len(buf) == G or i == a.n - 1:
                    flush()
                t_g.append(time.perf_counter() - t0)
            wall = time.perf_counter() - T0
        out["e2e_gpu_lattice" + ("_pack_fast" if fast else "")] = dict(
            workers=workers, tokens_per_s=round(a.n / wall, 3), wall_s=round(wall, 2), key_s_per_token=stat(t_key),
            main_gpu_s_per_token_mean=round(sum(t_g) / a.n, 4), main_thread_share=round(sum(t_g) / wall, 3),
            bw_tokens_differing_from_today=mism, cold_tokens_differing_from_today=mism_c, encoder_pairs_per_token=27)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


SYN_T = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu", "gpu"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-eq", type=int, default=48)
    ap.add_argument("--vram", type=float, default=44.0)
    a = ap.parse_args()
    import op_lb as OL
    SYN_T = OL.SYN_T
    {"cpu": cmd_cpu, "gpu": cmd_gpu}[a.cmd](a)
