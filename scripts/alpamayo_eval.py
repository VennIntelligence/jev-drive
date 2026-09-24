"""Alpamayo 1.5 smoke run: fetch clips, latency/config sweep, open-loop quality on a small clip sample.

Runs in the alpamayo1.5 venv (~/data/third_party/alpamayo1.5/.venv) with HF_ENDPOINT=https://hf-mirror.com.
  fetch  --n 40          example clip + n clips drawn (seed 0) from the repo's notebooks/clip_ids.parquet
  bench  --clips 4       latency rows (stage breakdown, p50/p99, peak VRAM) + output drift vs the default row
  eval                   minADE_6@6.4s vs ground truth on every fetched clip (default config, 6 samples)
Outputs go to $DATA_DIR/runs/alpamayo/<cmd>/<stamp>/ (log.txt, events.jsonl, results.csv, *.json, *.npz).
"""
import argparse, json, subprocess, sys, time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.alpamayo import data as D
from jevdrive.runlog import RunLog

ALP = Path.home() / "data/third_party/alpamayo1.5"
EXAMPLE = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # the clip test_inference.py uses (t0 = 5.1 s)


def clip_list() -> list[str]:
    return json.loads((D.cache_dir() / "clips.json").read_text())


def gpu_state() -> dict:
    """Utilization and the compute apps on the GPU that are not us (shared box)."""
    import os
    q = lambda a: subprocess.run(["nvidia-smi", f"--query-{a}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
    apps = [l.split(", ") for l in q("compute-apps=pid,used_memory").splitlines() if l]
    return {"util_pct": int(q("gpu=utilization.gpu").split()[0]),
            "others": [(int(p), int(m)) for p, m in apps if int(p) != os.getpid()]}


def cmd_fetch(a, log):
    ids = pd.read_parquet(ALP / "notebooks/clip_ids.parquet")["clip_id"].tolist()
    rng = np.random.default_rng(0)
    clips = [EXAMPLE] + [ids[i] for i in rng.choice(len(ids), a.n, replace=False)]
    info = D.fetch_clips(clips, log=log.info)
    (D.cache_dir() / "clips.json").write_text(json.dumps(clips))
    log.event("fetch", n=len(clips), **info)


def load_all(avdi, clips, log):
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        out = dict(zip(clips, ex.map(lambda c: D.load_clip(c, avdi), clips)))
    log.info(f"decoded {len(clips)} clips in {time.time() - t0:.0f} s")
    return out


def rows(I):
    """FA2 rows. torch.compile of the vision tower is not possible under FA2 (dynamo cannot trace flash_attn's
    varlen op), so it is tried on the SDPA load (sdpa_rows). The expert always runs SDPA and compiles here."""
    C = I.Config
    r = []
    for b in (C("default", n_samples=1), C("default", n_samples=6)):
        r += [b, replace(b, name="expert-cuda-graph", expert_graph=True),
              replace(b, name="compile-expert", compile=("expert",)),
              replace(b, name="flow-5", flow_steps=5), replace(b, name="flow-2", flow_steps=2),
              replace(b, name="reason-cap-32", max_gen=32), replace(b, name="reason-cap-16", max_gen=16),
              replace(b, name="no-reasoning", no_reasoning=True),
              replace(b, name="stack: compile-expert+flow-5", compile=("expert",), flow_steps=5),
              replace(b, name="stack + no-reasoning", compile=("expert",), flow_steps=5, no_reasoning=True)]
        if b.n_samples == 1:
            r += [replace(b, name="compile-lm", compile=("lm",))]
        else:  # one reasoning rollout, 6 flow samples on its KV cache (public num_traj_sets argument)
            r += [replace(b, name="1-rollout x 6 flow samples", n_samples=1, n_sets=6),
                  replace(b, name="stack: 1-rollout x 6 + compile-expert+flow-5", n_samples=1, n_sets=6,
                          flow_steps=5, compile=("expert",))]
    return r


def sdpa_rows(I):
    """SDPA load. static-kv-cache goes last: HF generate compiles and CUDA-graphs the decoder for a static cache,
    and when that fails it leaves the CUDA RNG in capture state, which breaks every later sampling call."""
    C = I.Config
    r = []
    for n in (1, 6):
        b = C("sdpa", attn="sdpa", n_samples=n)
        r += [b, replace(b, name="sdpa + compile-visual+expert", compile=("visual", "expert"))]
    return r + [C("sdpa + static-kv-cache", attn="sdpa", n_samples=1, static_cache=True)]


def bench_rows(model, proc, I, clips, cfgs, a, log, out):
    timer = I.StageTimer(model)
    inputs = {nr: {c: I.build_inputs(d, proc, no_reasoning=nr) for c, d in clips.items()} for nr in (False, True)}
    seeds = list(range(42, 42 + a.seeds))
    for cfg in cfgs:
        key = f"{cfg.name}/n{cfg.n_samples * cfg.n_sets}"
        rec = {"row": cfg.name, "key": key, **{k: v for k, v in asdict(cfg).items() if k != "name"}}
        try:
            I.apply(model, cfg)
            torch.cuda.reset_peak_memory_stats()
            g0 = gpu_state()
            t0 = time.time()
            for c in list(clips)[:2]:  # warmup: compile / graph capture land here
                I.run(model, inputs[cfg.no_reasoning][c], cfg, timer, seed=0)
            rec["warmup_s"] = time.time() - t0
            runs = []
            for s in seeds:
                for c in clips:
                    o = I.run(model, inputs[cfg.no_reasoning][c], cfg, timer, seed=s)
                    runs.append({"clip": c, "seed": s, **{k: v for k, v in o.items() if k not in ("xyz", "cot")}})
                    out.setdefault(key, {})[(c, s)] = (o["xyz"], o["cot"])
            nohook = [I.run(model, inputs[cfg.no_reasoning][c], cfg, None, seed=seeds[0])["wall"] for c in clips]
            g1 = gpu_state()
        except Exception as e:  # e.g. static cache is incompatible with the expert's prompt-cache crop
            log.info(f"{key}: FAILED {type(e).__name__}: {str(e)[:300]}")
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            I.reset(model)
            yield rec
            continue
        df = pd.DataFrame(runs)
        ms = lambda x: 1e3 * x
        rec.update(n_runs=len(df), p50_ms=ms(df.wall.median()), p99_ms=ms(df.wall.quantile(.99)),
                   mean_ms=ms(df.wall.mean()), nohook_p50_ms=ms(np.median(nohook)), prompt_tokens=df.n_prompt.mean(),
                   **{f"{k}_ms": ms(df[k].mean()) for k in
                                                   ("vision", "prefill", "decode", "flow", "other")},
                   decode_tokens=df.n_decode.mean(), ms_per_token=ms((df.decode / df.n_decode.clip(1)).mean()),
                   peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                   gpu_util_before=g0["util_pct"], gpu_others_before=g0["others"], gpu_others_after=g1["others"])
        if cfg.expert_graph and model.diffusion_expert_cuda_graph_stats:
            rec["graph_stats"] = model.diffusion_expert_cuda_graph_stats
        log.info(f"{key:40s} p50 {rec['p50_ms']:7.1f} ms p99 {rec['p99_ms']:7.1f}  "
                 f"vis {rec['vision_ms']:6.1f} pre {rec['prefill_ms']:6.1f} dec {rec['decode_ms']:6.1f} "
                 f"({rec['decode_tokens']:.0f} tok) flow {rec['flow_ms']:6.1f} other {rec['other_ms']:6.1f}  "
                 f"peak {rec['peak_alloc_gb']:.1f} GB  gpu-others {g0['others']}")
        I.reset(model)
        yield rec


def drift(out, clips, gt):
    """Per row: ADE of sample 0 vs the default row's sample 0 (same clip, same seed, same sample count), CoT
    identical rate, minADE vs GT. noise-floor rows compare default runs across seeds."""
    res = {}
    for key, o in out.items():
        ref = out.get("default/n" + key.rsplit("/n", 1)[1])
        d, same, mde = [], [], []
        for (c, s), (xyz, cot) in o.items():
            rx, rc = ref[(c, s)] if ref else (None, None)
            if ref:
                d.append(np.linalg.norm(xyz[0, :, :2] - rx[0, :, :2], axis=-1).mean())
                same.append(cot[0] == rc[0])
            mde.append(np.linalg.norm(xyz[:, :, :2] - gt[c][None], axis=-1).mean(-1).min())
        res[key] = {"drift_ade_m": float(np.mean(d)) if d else None,
                    "cot_identical": float(np.mean(same)) if same else None, "minade_vs_gt_m": float(np.mean(mde))}
    for n in (1, 6):
        o = out.get(f"default/n{n}")
        if o:
            seeds = sorted({s for _, s in o})
            pairs = [(o[(c, seeds[0])], o[(c, s)]) for c in clips for s in seeds[1:]]
            res[f"noise-floor/n{n}"] = {
                "drift_ade_m": float(np.mean([np.linalg.norm(x[0][0, :, :2] - y[0][0, :, :2], axis=-1).mean()
                                              for x, y in pairs])),
                "cot_identical": float(np.mean([x[1][0] == y[1][0] for x, y in pairs]))}
    return res


def cmd_bench(a, log):
    from jevdrive.alpamayo import infer as I
    log.event("versions", **I.versions())
    log.info(json.dumps(I.versions()))
    avdi = D.interface()
    clips = load_all(avdi, clip_list()[:a.clips], log)
    gt = {c: d["ego_future_xyz"][0, 0, :, :2].numpy() for c, d in clips.items()}
    out, recs = {}, []
    model, proc = I.load("flash_attention_2")
    cfgs = rows(I) if not a.only else [r for r in rows(I) if r.name in a.only.split(",")]
    for rec in bench_rows(model, proc, I, clips, cfgs, a, log, out):
        recs.append(rec)
        log.event("row", **{k: v for k, v in rec.items() if k != "compile"})
    del model
    torch.cuda.empty_cache()
    sd = [r for r in sdpa_rows(I) if not a.only or r.name in a.only.split(",")]
    if sd:
        model, proc = I.load("sdpa")
        for rec in bench_rows(model, proc, I, clips, sd, a, log, out):
            recs.append(rec)
            log.event("row", **{k: v for k, v in rec.items() if k != "compile"})
    dr = drift(out, list(clips), gt)
    df = pd.DataFrame(recs)
    df = df.merge(pd.DataFrame(dr).T.rename_axis("key").reset_index(), on="key", how="outer")
    df.to_csv(log.dir / "results.csv", index=False)
    (log.dir / "drift.json").write_text(json.dumps(dr, indent=1))
    np.savez_compressed(log.dir / "bench_preds.npz", **{f"{k}|{c}|{s}": v[0] for k, o in out.items()
                                                         for (c, s), v in o.items()})
    json.dump({f"{k}|{c}|{s}": v[1] for k, o in out.items() for (c, s), v in o.items()},
              open(log.dir / "bench_cot.json", "w"), indent=0)
    log.info("\n" + df.drop(columns=["gpu_others_before", "gpu_others_after"], errors="ignore").to_string())


def cmd_profile(a, log):
    """torch.profiler over one default n=1 inference: top CUDA kernels overall and inside the expert."""
    from torch.profiler import ProfilerActivity, profile, record_function
    from jevdrive.alpamayo import infer as I
    c = clip_list()[0]
    d = D.load_clip(c, D.interface())
    model, proc = I.load(a.attn)
    inp, cfg = I.build_inputs(d, proc), I.Config(n_samples=a.n)
    I.run(model, inp, cfg, None)
    f = model.expert.forward
    def tagged(*x, **k):
        with record_function("expert_forward"):
            return f(*x, **k)
    model.expert.forward = tagged
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        I.run(model, inp, cfg, None)
    t = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70)
    (log.dir / "profile.txt").write_text(t)
    prof.export_chrome_trace(str(log.dir / "trace.json"))
    log.info("\n" + t)


def cmd_eval(a, log):
    from jevdrive.alpamayo import infer as I
    avdi = D.interface()
    ids = clip_list()
    clips = load_all(avdi, ids, log)
    model, proc = I.load(a.attn)
    timer, cfg, rows_ = I.StageTimer(model), I.Config("default", n_samples=6), []
    preds, cots = {}, {}
    for c in ids:
        d = clips[c]
        o = I.run(model, I.build_inputs(d, proc), cfg, timer, seed=42)
        gt = d["ego_future_xyz"][0, 0, :, :2].numpy()
        ades = I.ade(o["xyz"][:, :, :2], gt)
        rows_.append({"clip": c, "minade6_m": ades.min(), "ade_sample0_m": ades[0], "ade_mean6_m": ades.mean(),
                      "fde_min_m": np.linalg.norm(o["xyz"][:, -1, :2] - gt[-1], axis=-1).min(),
                      "gt_len_m": np.linalg.norm(np.diff(gt, axis=0), axis=-1).sum(), "wall_ms": 1e3 * o["wall"],
                      "decode_tokens": o["n_decode"]})
        preds[c] = {"pred": o["xyz"], "gt": d["ego_future_xyz"][0, 0].numpy(),
                    "hist": d["ego_history_xyz"][0, 0].numpy()}
        cots[c] = o["cot"]
        log.info(f"{c} minADE6 {ades.min():.3f} m  ade0 {ades[0]:.3f}  cot: {o['cot'][0][:120]}")
    df = pd.DataFrame(rows_)
    df.to_csv(log.dir / "results.csv", index=False)
    np.savez_compressed(log.dir / "preds.npz", **{f"{c}|{k}": v for c, p in preds.items() for k, v in p.items()})
    json.dump(cots, open(log.dir / "cot.json", "w"), indent=1)
    s = {"n": len(df), "minade6_mean_m": float(df.minade6_m.mean()), "minade6_median_m": float(df.minade6_m.median()),
         "ade_sample0_mean_m": float(df.ade_sample0_m.mean())}
    log.event("summary", **s)
    log.info(json.dumps(s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").add_argument("--n", type=int, default=40)
    b = sub.add_parser("bench")
    b.add_argument("--clips", type=int, default=4)
    b.add_argument("--seeds", type=int, default=4)
    b.add_argument("--only", default="")
    sub.add_parser("eval").add_argument("--attn", default="flash_attention_2")
    p = sub.add_parser("profile")
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    if a.cmd != "fetch":
        import torch
    log = RunLog("alpamayo", a.cmd)
    log.info(f"args {vars(a)} -> {log.dir}")
    {"fetch": cmd_fetch, "bench": cmd_bench, "eval": cmd_eval, "profile": cmd_profile}[a.cmd](a, log)
    log.event("end")
    log.close()
