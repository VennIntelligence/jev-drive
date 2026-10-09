"""Training-side audit, trainer: where a step of ap2_train.py (pp_train.PModel on the cached tokens, the AP2-AB recipe) spends its time.
Measurement only: builds the trainer's objects exactly as ap2_train.main (same Store / PModel / Losses / optimizer), then times phases.
Nothing is saved but the JSON next to this file.

  python train_bench.py --out train.json [--sizes 16 32 64 128] [--compile inductor aot_eager] [--shards 12]
"""
import sys, pathlib, os, time, json, argparse  # noqa: E401
_R = pathlib.Path(os.environ.get("JEV_REPO", pathlib.Path.home() / "data/jev-drive")).resolve()
sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts")]
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

pc = time.perf_counter
OUT = {}


def dump(a):
    pathlib.Path(a.out).write_text(json.dumps(OUT, indent=1))


def med(x):
    return round(1e3 * float(np.median(x)), 2)


def sync():
    torch.cuda.synchronize()
    return pc()


def main(a):
    import ap2_core as AC
    import ap2_inputs as AI
    import ap2_train as AT
    import pp_train as T
    from jevdrive import op_adapt as A
    from jevdrive.common import data_dir
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.vram / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30))
    OUT.update(cpus=len(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(), omp=os.environ.get("OMP_NUM_THREADS"), load1=os.getloadavg()[0],
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    datas = tuple(f"navtrain_full.s{i}of12" for i in range(a.shards))
    cfg = T.Cfg(arm="P2", seed=0, steps=10000, batch=128, data=datas, split="navsim/op-parity-full", frames="warp", host=True, warmup=300,
                eval_every=1000, hinge_lam=30.0, hinge_margin=0.5)
    torch.manual_seed(0)
    rng, mrng = np.random.default_rng([0, 0]), np.random.default_rng([0, 0, 23])
    st = {}
    t0 = pc(); S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True); st["store_s"] = pc() - t0                       # noqa: E702
    tabs = dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d, is_wod=S.is_wod)
    t0 = pc(); tr_rows, dv_rows, sp = T.split_rows(tabs, cfg.split); st["split_s"] = pc() - t0                                         # noqa: E702
    t0 = pc(); wp, ok = AT.routes(cfg.data, S.tab["names"]); EGO = torch.from_numpy(AC.ego_table(S.tab, wp, False)).to(dev); st["routes_ego_s"] = pc() - t0  # noqa: E702
    cold = T.Tokens([AT.AROOT / "cache" / d / "bw.npy" for d in cfg.data], dev, host=True)
    tr_rows, dv_rows = tr_rows[ok[tr_rows, 3]], dv_rows[ok[dv_rows, 3]]
    S.ego = EGO[:, 3]
    mix = np.asarray(AI.MIX, float) / np.sum(AI.MIX)
    t0 = pc(); model = AC.widen(T.PModel("P2"), False).to(dev); st["model_s"] = pc() - t0                                              # noqa: E702
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    from drivable_hinge import Hinge
    t0 = pc(); hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], S.tab["names"], dev, cfg.hinge_margin, list(cfg.hinge_footprint)); st["hinge_s"] = pc() - t0  # noqa: E702
    LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    OUT["startup_s"] = {k: round(v, 2) for k, v in st.items()}
    OUT["rows"] = dict(train=len(tr_rows), dev=len(dv_rows), base_params_M=round(sum(p.numel() for p in base) / 1e6, 2),
                       adapter_params_M=round(sum(p.numel() for p in new) / 1e6, 3), n_base_tensors=len(base), dtype=str(model.net.dtype))
    dump(a)

    def draw(nB):
        r = rng.choice(tr_rows, nB, replace=False)
        an = rng.random(nB) < cfg.d_frac
        m = mrng.choice(4, nB, p=mix) + 1
        m[an | ~ok[r, m - 1]] = 4
        return r, an, m

    def fetch(dr):
        return dr, AT.slots(S, cold, dr[0], dr[2], "backwarp")

    def step(batch, mdl=model, do_opt=True, timed=True):
        (r, an, m), (front, nv) = batch
        rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
        ego = EGO[rows, torch.as_tensor(m - 1, device=dev)] * (~anchor)[:, None].float()
        tc = S.tc[rows]
        t0 = sync() if timed else pc()
        out = mdl(front, ego, tc, nv=nv)
        t1 = pc(); t2 = sync() if timed else t1                                                                                        # noqa: E702
        total, Ls = LS(out, S, rows, anchor)
        t3 = sync() if timed else pc()
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        t4 = pc(); t5 = sync() if timed else t4                                                                                        # noqa: E702
        if do_opt:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
        {k: float(v) for k, v in Ls.items()}
        t6 = sync() if timed else pc()
        return dict(fwd_enqueue=t1 - t0, fwd=t2 - t0, loss=t3 - t2, bwd_enqueue=t4 - t3, bwd=t5 - t3, opt=t6 - t5, total=t6 - t0), float(total)

    # ---- fetch (host gather from the page cache), batch 128 as the runs
    tf, tfw, tfc = [], [], []
    for i in range(a.fetch_reps):
        dr = draw(128)
        t0 = pc(); f = S.front[dr[0]]; t1 = pc(); fetch(dr); t2 = sync()                                                               # noqa: E702
        tfw.append(t1 - t0), tf.append(t2 - t1)
        sel = dr[0][dr[2] < 4]
        t0 = pc(); cold[sel] if len(sel) else None; tfc.append(pc() - t0)                                                             # noqa: E702
    OUT["fetch_b128_ms"] = dict(front_gather=dict(first=round(1e3 * tfw[0], 1), median=med(tfw[1:]), p95=round(1e3 * float(np.percentile(tfw[1:], 95)), 1)),
                                slots_total=dict(median=med(tf[1:]), p95=round(1e3 * float(np.percentile(tf[1:], 95)), 1)), cold_gather_median=med(tfc[1:]),
                                bytes_per_batch_MB=round(128 * 8 * 32 * 512 * 2 / 2 ** 20, 1))
    dump(a)

    # ---- step phases by batch size (CUDA sync at every phase boundary)
    OUT["step_ms"] = {}
    for nB in a.sizes:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()                                                             # noqa: E702
            rec = [step(fetch(draw(nB)))[0] for _ in range(a.steps + 3)][3:]
            OUT["step_ms"][nB] = {k: med([x[k] for x in rec]) for k in rec[0]} | {"peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                                                                                   "rows_per_s": round(nB / float(np.median([x["total"] for x in rec])), 1)}
        except torch.OutOfMemoryError:
            OUT["step_ms"][nB] = "oom"
            opt.zero_grad(set_to_none=True)
        dump(a)
        print(nB, OUT["step_ms"][nB], flush=True)

    # ---- the loop as it runs (prefetch thread, no extra syncs), batch 128
    from concurrent.futures import ThreadPoolExecutor
    pre = ThreadPoolExecutor(2)
    nxt = pre.submit(fetch, draw(128))
    wait = []
    for i in range(a.steps + 3):
        if i == 3:
            T0 = sync()
        t0 = pc(); b = nxt.result(); wait.append(pc() - t0)                                                                            # noqa: E702
        nxt = pre.submit(fetch, draw(128))
        step(b, timed=False)
    el = sync() - T0
    OUT["loop_b128"] = dict(steps_per_s=round(a.steps / el, 2), ms_per_step=round(1e3 * el / a.steps, 1), wait_for_fetch_ms_median=med(wait[3:]))
    dump(a)
    print("loop", OUT["loop_b128"], flush=True)

    # ---- one profiled step, batch 128: aten calls, CUDA kernel time against the wall time
    b = fetch(draw(128))
    step(b)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        t0 = sync(); step(b); wall = sync() - t0                                                                                       # noqa: E702
    ev = prof.key_averages()
    cu = "self_device_time_total" if hasattr(ev[0], "self_device_time_total") else "self_cuda_time_total"
    OUT["profile_b128"] = dict(wall_ms_under_profiler=round(1e3 * wall, 1), aten_calls=int(sum(e.count for e in ev if e.key.startswith("aten::"))),
                               cuda_kernel_ms=round(sum(getattr(e, cu) for e in ev) / 1e3, 1),
                               cuda_ms_aten_ops=round(sum(getattr(e, cu) for e in ev if e.key.startswith("aten::")) / 1e3, 1),
                               cuda_ms_casts_copy=round(sum(getattr(e, cu) for e in ev if e.key in ("aten::copy_", "aten::to", "aten::_to_copy")) / 1e3, 1),
                               self_cpu_ms_total=round(sum(e.self_cpu_time_total for e in ev if "Synchronize" not in e.key) / 1e3, 1),
                               top_by_count=[[e.key, e.count] for e in sorted(ev, key=lambda e: -e.count)[:12]],
                               top_by_cuda_ms=[[e.key, round(getattr(e, cu) / 1e3, 1), e.count] for e in sorted(ev, key=lambda e: -getattr(e, cu))[:12]],
                               top_by_self_cpu_ms=[[e.key, round(e.self_cpu_time_total / 1e3, 1), e.count] for e in sorted(ev, key=lambda e: -e.self_cpu_time_total)[:12]])
    dump(a)

    # ---- static: the policy sub-graph the trainer runs; which nodes never see the hidden tokens (constant across steps)
    net = model.net
    H0 = torch.zeros(1, A.CONTEXT, *A.H_SHAPE, device=dev, dtype=net.dtype)
    feeds = A.policy_feeds(net, H0, T.AT, torch.tensor([[1.0, 0.0]], device=dev))
    ks = net.plan(tuple(sorted(feeds)), ("outputs",))
    trainable = {k for k, p in net.params.items() if p.requires_grad}
    from jevdrive.op_torch import _key
    dyn, wdep = {"view_39", "_to_copy_1", "_to_copy_2"}, set()                  # tokens, past tokens, traffic convention: vary per row
    n_const = n_wonly = 0
    ops, const_ops = Counter(), Counter()
    for k in ks:
        op, ins, outs, at = net.nodes[k]
        ops[op] += 1
        d = any(i in dyn for i in ins)
        w = any((i in net.fnames and _key(i) in trainable) or i in wdep for i in ins)
        if d:
            dyn.update(outs)
        elif w:
            wdep.update(outs); n_wonly += 1                                                                                            # noqa: E702
        else:
            n_const += 1; const_ops[op] += 1                                                                                           # noqa: E702
    OUT["policy_graph"] = dict(nodes_total=len(net.nodes), nodes_run=len(ks), by_op=dict(ops.most_common()), constant_nodes=n_const,
                               constant_by_op=dict(const_ops.most_common()), weight_only_nodes=n_wonly,
                               casts=ops["Cast"], trainable_matmul_gemm_conv=sum(1 for k in ks if net.nodes[k][0] in ("MatMul", "Gemm", "Conv")
                                                                                    and any(i in net.fnames and _key(i) in trainable for i in net.nodes[k][1])))
    dump(a)

    # ---- no-grad forward (dev eval): batch scaling, and the eval calls of the loop on 512 dev rows
    ev_ms = {}
    model.eval()
    with torch.no_grad():
        for nB in (1, 16, 128, 256):
            try:
                (r, an, m), (front, nv) = fetch(draw(nB))
                rows = torch.as_tensor(r, device=dev)
                ts, te = [], []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(6):
                    t0 = sync(); model(front, S.ego[rows], S.tc[rows], nv=nv); t1 = pc(); te.append(t1 - t0); ts.append(sync() - t0)   # noqa: E702
                ev_ms[nB] = dict(host_enqueue=med(te[1:]), until_done=med(ts[1:]), rows_per_s=round(nB / float(np.median(ts[1:])), 1),
                                 peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
            except torch.OutOfMemoryError:
                ev_ms[nB] = "oom"
        # vmap against the plain interpreter on one sample
        Hc = torch.cat([front.new_zeros(1, 1, *A.H_SHAPE), front[:1]], 1).to(net.dtype)
        fb = A.policy_feeds(net, Hc, T.AT, S.tc[rows[:1]].to(net.dtype))
        f1 = {k: v[0] for k, v in fb.items()}
        tv, tp = [], []
        for _ in range(12):
            t0 = sync(); o1 = net.run_batched(fb, ["outputs"])["outputs"]; tv.append(pc() - t0)                                         # noqa: E702
            t0 = sync(); o2 = net.run(f1, ["outputs"])["outputs"]; tp.append(pc() - t0)                                                 # noqa: E702
        ev_ms["one_sample_host_ms"] = dict(vmap=med(tv[2:]), plain_run=med(tp[2:]), max_abs_diff=float((o1.float().reshape(-1) - o2.float().reshape(-1)).abs().max()))
    d512 = dv_rows[:512]
    for bs in (128, 256):
        try:
            torch.cuda.empty_cache()
            t0 = sync(); T.dev_eval(model, S, d512, LS.W, bs=bs); t1 = sync()                                                          # noqa: E702
            AT.dev_by_m(model, S, cold, EGO, ok, d512, LS.W, "backwarp", bs=bs); t2 = sync()
            ev_ms[f"dev_eval_512rows_bs{bs}_s"] = round(t1 - t0, 2)
            ev_ms[f"dev_by_m_512rows_bs{bs}_s"] = round(t2 - t1, 2)
        except torch.OutOfMemoryError:
            ev_ms[f"dev_eval_512rows_bs{bs}_s"] = "oom"
    torch.cuda.empty_cache()
    model.train()
    OUT["eval"] = ev_ms
    dump(a)
    print("eval", ev_ms, flush=True)

    # ---- torch.compile of the model inside the training step
    def grads(mdl, batch):
        (r, an, m), (front, nv) = batch
        rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
        ego = EGO[rows, torch.as_tensor(m - 1, device=dev)] * (~anchor)[:, None].float()
        opt.zero_grad(set_to_none=True)
        out = mdl(front, ego, S.tc[rows], nv=nv)
        total, _ = LS(out, S, rows, anchor)
        total.backward()
        g = torch.cat([p.grad.flatten().float() for p in base + new if p.grad is not None])
        return float(total), g.clone(), out.detach().float().clone()
    OUT["compile"] = {}
    fixed = [fetch(draw(128)) for _ in range(3)]
    l0, g0, o0 = grads(model, fixed[0])
    l0b, g0b, o0b = grads(model, fixed[0])
    OUT["compile"]["eager_repeat"] = dict(loss_diff=abs(l0 - l0b), out_max_abs=float((o0 - o0b).abs().max()), grad_rel=float((g0 - g0b).norm() / g0.norm()))
    for backend in a.compile:
        res = {}
        try:
            import torch._dynamo as _dyn
            _dyn.reset()
            cm = torch.compile(model, backend=backend)
            t0 = sync(); l1, g1, o1 = grads(cm, fixed[0]); res["first_call_s"] = round(sync() - t0, 1)                                   # noqa: E702
            t0 = sync(); grads(cm, fixed[1]); res["second_call_s"] = round(sync() - t0, 2)                                               # noqa: E702
            l1, g1, o1 = grads(cm, fixed[0])
            res.update(loss_eager=l0, loss_compiled=l1, out_max_abs=float((o0 - o1).abs().max()), out_identical=bool(torch.equal(o0, o1)),
                       grad_rel=float((g0 - g1).norm() / g0.norm()), grad_cos=float(torch.nn.functional.cosine_similarity(g0, g1, 0)))
            opt.zero_grad(set_to_none=True)
            rec = [step(fixed[i % 3], mdl=cm, do_opt=False)[0] for i in range(a.steps + 3)][3:]
            res["step_ms"] = {k: med([x[k] for x in rec]) for k in rec[0]}
            rec = [step(fixed[i % 3], mdl=model, do_opt=False)[0] for i in range(a.steps + 3)][3:]
            res["eager_step_ms_same_batches"] = {k: med([x[k] for x in rec]) for k in rec[0]}
            t0 = sync(); step(fetch(draw(64)), mdl=cm, do_opt=False); res["other_batch_size_first_call_s"] = round(sync() - t0, 1)      # noqa: E702
            with torch.no_grad():
                cm.eval()
                (r, an, m), (front, nv) = fixed[0]
                rows = torch.as_tensor(r, device=dev)
                t0 = sync(); cm(front, S.ego[rows], S.tc[rows], nv=nv); res["nograd_first_call_s"] = round(sync() - t0, 1)              # noqa: E702
                ts = []
                for _ in range(6):
                    t0 = sync(); cm(front, S.ego[rows], S.tc[rows], nv=nv); ts.append(sync() - t0)                                      # noqa: E702
                res["nograd_b128_ms"] = med(ts[1:])
                cm.train()
        except Exception as e:  # noqa: BLE001
            import traceback
            res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
            res["trace_tail"] = traceback.format_exc()[-1500:]
        OUT["compile"][backend] = res
        dump(a)
        print("compile", backend, res, flush=True)
    OUT["load1_end"] = os.getloadavg()[0]
    dump(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--fetch-reps", type=int, default=25)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--compile", nargs="*", default=["inductor"])
    ap.add_argument("--vram", type=float, default=40.0)
    main(ap.parse_args())
