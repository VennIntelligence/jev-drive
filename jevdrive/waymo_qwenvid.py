"""Qwen3-VL-4B native-video features over the Waymo train split: P3(d'')'s recipe, profiled and sharded.

The recipe is `waymo_ladder.extract_qwenvid` unchanged -- a 4-frame clip at stride 2 frames per camera, three
cameras, Qwen3-VL's own video path, decoder stopped at layer 18, `L18_mean` / `L18_last` / `vis_mean` /
`vit_mean` pooled -- so the heads trained on it read the representation P3(d'') measured. What changes is only
how it runs:

  profile   ms/frame for a grid of (batch size, torch.compile, loader workers, spatial summary on/off) on the
            first rows of the `qwenvid_p3` item list, a kernel breakdown from torch.profiler, the CPU cost of
            decoding one clip, and the numerical difference of every configuration from the stored rows.
  run       the sharded, resumable extraction: one directory per Waymo shard of the target frame under
            `features/<set>/<shard>/`, meta.json written last as the done-marker, so an interruption costs one
            shard and a re-run skips what is built. The P3 val subset goes first, in the same configuration,
            so the evaluation side is never a different numerical recipe from the training side.

`--grid H,W` also stores `L18_grid`: the last temporal slot's tokens of each camera pooled to H x W, the small
spatial summary a DiffusionDrive-style head can attend to (a pooled 2560-vector is a weak condition for it).
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from . import waymo, waymo_ladder as lad
from .common import data_dir, get_logger

log = get_logger(__name__)
FRAMES, STRIDE, LAYER = 4, 2, 18
SET = "qwenvid_train"
REF = lad.QWENVID_SET


def make_fx(compile: bool = False, grid_hw=None, model_id: str | None = None, layers=(LAYER,)):
    """The P3(d'') feature extractor, with the decoder layers it never runs taken off the card."""
    from . import features as F
    if compile:
        # Every shard ends on a partial batch. With automatic dynamic shapes the second batch size would swap
        # the static batch-8 graphs for dynamic ones for the rest of the run; static graphs per size instead
        # cost one extra compile per distinct remainder (at most batch - 1 of them) and keep batch 8 static.
        import torch._dynamo as dynamo     # `import torch._dynamo` would make `torch` local to make_fx
        dynamo.config.automatic_dynamic_shapes = False
        dynamo.config.cache_size_limit = 32
    fx = F.QwenVideoFeatures(frames=FRAMES, n_videos=len(waymo.CAMS), layers=list(layers), compile=compile,
                             grid_hw=grid_hw, **({"model_id": model_id} if model_id else {}))
    lm = fx.model.language_model
    lm.layers = lm.layers[:max(layers)]             # the deeper layers are never called: VRAM back
    torch.cuda.empty_cache()
    n, t = len(waymo.CAMS), fx.transform

    def transform(imgs):
        # One clip item's pixel_values_videos is 24 480 patches x 1536 float32 = 150 MB, and the loader keeps
        # workers x prefetch x batch of them in pinned RAM: batch 12 with 8 workers is ~58 GB and got the
        # profiler OOM-killed. The patch embedding casts to bf16 as its first op, so casting here is exact
        # (checked bit for bit against qwenvid_p3) and halves both the RAM and the H2D copy.
        ids, mm, pv, grid = t([imgs[i * FRAMES:(i + 1) * FRAMES] for i in range(n)])
        return ids, mm, pv.to(torch.bfloat16), grid
    fx.transform = transform
    return fx


def clip_items(df, rows):
    return waymo.multicam_clip_items(df, rows, FRAMES - 1, STRIDE, waymo.CAMS)


def ref_items(n: int | None = None):
    """The `qwenvid_p3` item list in extraction order (its first `n`), so a prefix pairs up batch for batch."""
    ctx = lad.base_context()
    items, idx, _ = clip_items(ctx["df"], ctx["rows"][lad.load_subset(ctx)])
    return (items[:n], idx.iloc[:n].reset_index(drop=True)) if n else (items, idx)


def train_rows(df, thin: int | None = None, seed_subsets=None) -> np.ndarray:
    """Every train frame with a future; with `thin`, the stratified subsample: every pre-onset and turning
    frame, and of the rest only those whose frame index is a multiple of `thin` (0.1 s * thin apart)."""
    past, future = waymo.load_ego()
    rows = np.flatnonzero((df.split == "train").to_numpy() & df.has_future.to_numpy())
    if thin:
        sub = seed_subsets or waymo.subsets(df, past, future)
        keep = sub["pre_onset"][rows] | sub["turn_yaw"][rows] | (df.frame.to_numpy()[rows] % thin == 0)
        rows = rows[keep]
    return rows


def compare(out_dir, idx: pd.DataFrame, arrays=("L18_mean", "L18_last", "vis_mean", "vit_mean")) -> dict:
    """Max |diff|, relative L2 diff and bit-identical rows of freshly extracted arrays against `qwenvid_p3`."""
    ref_idx, ref = waymo.load_flat_features(REF, list(arrays))
    pos = pd.Series(np.arange(len(ref_idx)), index=ref_idx.frame_name.to_numpy()).reindex(idx.frame_name).to_numpy()
    out = {}
    for a in arrays:
        x = np.load(out_dir / f"{a}.npy").astype(np.float32)
        y = np.asarray(ref[a])[pos.astype(int)].astype(np.float32)
        d = x - y
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        out[a] = {"identical_rows": int((d == 0).all(1).sum()), "rows": len(x), "max_abs": float(np.abs(d).max()),
                  "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(y)), "min_cos": float(cos.min())}
    return out


def cpu_decode_ms(fx, items, n: int = 16) -> float:
    """Single-core cost of reading, decoding and preprocessing one clip item (what a loader worker does)."""
    ds = waymo.Shards(items[:n], fx.transform)
    ds[0]
    t = time.perf_counter()
    for i in range(1, n):
        ds[i]
    return 1e3 * (time.perf_counter() - t) / (n - 1)


def kernel_table(fx, items, batch: int, n_batches: int = 4, top: int = 15) -> pd.DataFrame:
    """CUDA time by operator over a few steady-state batches, to name the bottleneck rather than guess it."""
    from torch.profiler import ProfilerActivity, profile as tprof
    from torch.utils.data import DataLoader
    dl = DataLoader(waymo.Shards(items[:batch * (n_batches + 1)], fx.transform), batch_size=batch, num_workers=4,
                    collate_fn=fx.collate)
    it = iter(dl)
    fx(next(it))                                    # warm-up outside the trace
    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CUDA]) as p:
        for b in it:
            fx(b)
        torch.cuda.synchronize()
    ev = p.key_averages()
    tot = sum(e.device_time_total for e in ev if e.device_time_total > 0 and not e.key.startswith("aten::"))
    rows = [{"op": e.key[:90], "cuda_ms_per_frame": e.device_time_total / 1e3 / (batch * n_batches),
             "share": e.device_time_total / tot} for e in ev if not e.key.startswith("aten::")]
    return pd.DataFrame(rows).sort_values("cuda_ms_per_frame", ascending=False).head(top)


def profile(rl, n: int = 240, batches=(2, 8), workers=(6,), grid_hw=(4, 4)):
    """The configuration grid, each on the same first `n` rows of the `qwenvid_p3` item list."""
    from . import features as F
    items, idx = ref_items(n)
    scratch = data_dir() / "scratch" / "qwenvid_profile" / rl.dir.name
    rows = []
    for comp in (False, True):
        fx = make_fx(compile=comp)
        if not comp:
            ms = cpu_decode_ms(fx, items)
            rl.event("cpu_decode", ms_per_item=ms)
            rl.log.info("CPU: %.0f ms to read + decode + preprocess one 3-camera 4-frame clip item", ms)
        for b in batches:
            for w in (workers if b == batches[-1] else workers[:1]):
                for g in ((None, grid_hw) if b == batches[-1] else (None,)):
                    fx.grid_hw = g
                    tag = f"b{b}-{'compile' if comp else 'eager'}-w{w}" + (f"-grid{g[0]}x{g[1]}" if g else "")
                    dst = scratch / tag
                    dst.mkdir(parents=True, exist_ok=True)
                    torch.cuda.reset_peak_memory_stats()
                    st = F.extract(fx, items, b, w, dst, rl, f"profile/{tag}", dataset=waymo.Shards)
                    eq = compare(dst, idx)
                    r = {"config": tag, "batch": b, "compile": comp, "workers": w, "grid": str(g),
                         "ms_per_frame": st["ms_per_frame"], "peak_vram_gb": st["peak_vram_gb"],
                         "bytes_per_frame": st["bytes_per_sample"],
                         **{f"{a}_{k}": v for a, e in eq.items() for k, v in e.items() if k != "rows"}}
                    rows.append(r)
                    rl.event("profile", **r)
                    rl.log.info("%-26s %.1f ms/frame, peak %.1f GB, %d B/frame | L18_mean identical %d/%d, "
                                "max|d| %.3g, rel %.2e", tag, r["ms_per_frame"], r["peak_vram_gb"],
                                r["bytes_per_frame"], eq["L18_mean"]["identical_rows"], n,
                                eq["L18_mean"]["max_abs"], eq["L18_mean"]["rel_l2"])
        fx.grid_hw = None
        kt = kernel_table(fx, items, batches[-1])
        kt.to_csv(rl.dir / f"kernels_{'compile' if comp else 'eager'}_b{batches[-1]}.csv", index=False)
        rl.log.info("kernels (%s, batch %d)\n%s", "compile" if comp else "eager", batches[-1],
                    kt.to_markdown(index=False, floatfmt=".3f"))
        del fx
        F.free_gpu()
    t = pd.DataFrame(rows)
    t.to_csv(rl.dir / "profile.csv", index=False)
    rl.log.info("profile\n%s", t.to_markdown(index=False, floatfmt=".4g"))
    return t


def run(rl, batch: int, compile: bool, workers: int, grid_hw=None, thin: int | None = None, name: str = SET,
        with_val_subset: bool = True):
    """The sharded extraction: the P3 val subset's shards first, then every train shard. A shard with
    meta.json is skipped, so re-running resumes where the last run stopped."""
    from . import features as F
    df = waymo.load_index()
    parts = []
    if with_val_subset:
        ctx = lad.base_context()
        parts.append(ctx["rows"][lad.load_subset(ctx)])
    parts.append(train_rows(df, thin))
    rows = np.concatenate(parts)
    items, idx, _ = clip_items(df, rows)
    shard = df.shard.astype(str).to_numpy()[idx.row.to_numpy()]
    split = df.split.to_numpy()[idx.row.to_numpy()]
    val_first = sorted(set(shard[split == "val"]))
    order = val_first + sorted(set(shard[split != "val"]))
    root = waymo.out_dir("features", name)
    todo = [s for s in order if not (root / s / "meta.json").exists()]
    left = sum(int((shard == s).sum()) for s in todo)
    log.info("%s: %d rows over %d shards (%d val-subset shards first); %d shards to do, batch %d, compile %s, "
             "workers %d, grid %s, thin %s", name, len(items), len(order), len(val_first), len(todo), batch,
             compile, workers, grid_hw, thin)
    rl.event("plan", rows=len(items), shards=len(order), todo=len(todo))
    fx = make_fx(compile, grid_hw)
    t0, done_rows = time.perf_counter(), 0
    for i, s in enumerate(todo):
        m = np.flatnonzero(shard == s)
        dst = root / s
        dst.mkdir(parents=True, exist_ok=True)
        st = F.extract(fx, [items[j] for j in m], batch, workers, dst, None, f"{name}/{s}", dataset=waymo.Shards)
        idx.iloc[m].to_parquet(dst / "index.parquet", index=False)
        (dst / "meta.json").write_text(json.dumps({"set": name, "shard": s, "recipe": "P3(d'') qwenvid",
                                                   "frames_per_clip": FRAMES, "clip_stride": STRIDE,
                                                   "layer": LAYER, "batch_size": batch, "compile": compile,
                                                   "grid_hw": grid_hw, "thin": thin, **st}, indent=2, default=float))
        done_rows += len(m)
        el = time.perf_counter() - t0
        eta = el / done_rows * (left - done_rows)
        log.info("%s/%s: %d rows, %.1f ms/frame; %d/%d shards this run, %.1f h elapsed, ETA %.1f h", name, s,
                 len(m), st["ms_per_frame"], i + 1, len(todo), el / 3600, eta / 3600)
        rl.event("shard_done", shard=s, rows=len(m), ms_per_frame=st["ms_per_frame"], done=i + 1, of=len(todo),
                 elapsed_h=el / 3600, eta_h=eta / 3600)
        rl.scalar("extract/ms_per_frame", st["ms_per_frame"], i)
    del fx
    F.free_gpu()


def subset(rl, model_id: str, layers, name: str, batch: int = 2, compile: bool = False):
    """Another backbone over exactly the `qwenvid_p3` rows, for the P3 ladder. The item list is the same, in
    the same order, so the two sets pair up row for row. d'' ran eager at batch 2; compile at batch 8 is 0.68x
    the time and moves the pooled features by ~1e-2 relative (cos >= 0.9997, see the profile)."""
    from . import features as F
    items, idx = ref_items()
    fx = make_fx(compile, None, model_id, layers)
    meta = waymo.extract_items(name, fx, items, idx, batch, rl=rl, cams=list(waymo.CAMS), frames_per_clip=FRAMES,
                               clip_stride=STRIDE, complete=len(items), layers=list(layers), compile=compile)
    del fx
    F.free_gpu()
    return meta


def check(rl, name: str, taps=("L18_last", "L18_mean")):
    """The downstream equivalence test of a new extraction against `qwenvid_p3` on the P3 val subset.

    Two numbers per tap: how far the features moved (relative L2, worst row cosine), and whether that moves
    the thing we read off them -- P3(d'')'s ridge_late row, refitted on each version under decision 22's judge.
    The new set is shard-keyed and may still be growing; only the rows both sets cover take part.
    """
    ctx = lad.base_context()
    keep = lad.load_subset(ctx)
    a, b = lad.align(ctx, REF, list(taps)), lad.align(ctx, name, list(taps), flat=False)
    keep &= a["covered"] & b["covered"]
    sel = np.flatnonzero(keep)
    for t in taps:
        x, y = b[t][sel].astype(np.float32), a[t][sel].astype(np.float32)
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        rl.event("feature_diff", tap=t, rows=len(sel), rel_l2=float(np.linalg.norm(x - y) / np.linalg.norm(y)),
                 min_cos=float(cos.min()), median_cos=float(np.median(cos)))
        rl.log.info("%s %s vs %s on %d rows: rel L2 %.3e, cos min %.6f median %.6f", t, name, REF, len(sel),
                    np.linalg.norm(x - y) / np.linalg.norm(y), cos.min(), np.median(cos))
    arms = {**{f"{REF} {t}": lad.ridge_arm(a[t]) for t in taps}, **{f"{name} {t}": lad.ridge_arm(b[t]) for t in taps}}
    lad.run_ladder(lambda k: arms, "check", keep, ctx, rl=rl)
    for k, t in lad.rejudge(rl.dir, "check").items():
        t.to_csv(rl.dir / f"{k}.csv", index=False)
    head = pd.read_csv(rl.dir / "pre_onset_dec19.csv")
    rl.log.info("pre-onset (deciles 1-9)\n%s", head[["direction", "arm", "n", "delta", "lo", "hi"]].to_markdown(
        index=False, floatfmt=".4f"))


P0_RUN = "runs/waymo_p0/train_split/20260922-175708"     # whose s_ego the decile axis reuses, as P3(d''') did


def trainfit(rl, name: str, taps=("L18_last", "L18_mean"), thin: int | None = 4):
    """The decisive test of P3(d''): fit ridge_late on the train split, evaluate on val, decision 22's judge.

    P3(d''') did this for V-JEPA and the half-val effect shrank to "unmeasurable"; this is the same protocol
    for the Qwen video features. Rows are whatever the (possibly still growing) shard-keyed set covers: the
    train frames of the extraction and the P3 val subset, which carries every pre-onset and rater frame of
    val, so the headline pre-onset delta is over all of them. Arm A (the single-frame pooled Qwen feature) is
    refitted on exactly the same rows beside it. The fit rows are uniform over what was extracted, i.e. over
    the stratified subsample when `thin` was used -- the composition the todo states.
    """
    ctx = lad.train_context(p0_run=data_dir() / P0_RUN)
    b = lad.align(ctx, name, list(taps), flat=False)
    keep = b["covered"]
    rl.log.info("trainfit: %d train rows, %d val rows covered", int((keep & (ctx["half"] == 0)).sum()),
                int((keep & (ctx["half"] == 1)).sum()))
    arms = {"A ridge_late pooled (qwen4b L18 image)": lad.ridge_arm(ctx["pooled"]),
            **{f"d'' ridge_late {t}": lad.ridge_arm(b[t]) for t in taps}}
    lad.run_ladder(lambda k: arms, "trainfit", keep, ctx, directions=(0,), rl=rl)
    for k, t in lad.rejudge(rl.dir, "trainfit").items():
        t.to_csv(rl.dir / f"{k}.csv", index=False)
        rl.log.info("%s\n%s", k, t.to_markdown(index=False, floatfmt=".4f"))
    if thin:
        trainfit_weighted(rl, ctx, keep, {"A ridge_late pooled (qwen4b L18 image)": ctx["pooled"],
                                          **{f"d'' ridge_late {t}": b[t] for t in taps}}, thin)


def trainfit_weighted(rl, ctx: dict, keep: np.ndarray, feats: dict, thin: int):
    """The same fit with the stratified subsample weighted back to the full train composition.

    Every thinned frame (neither pre-onset nor turning) stands for `thin` frames of the split, so it gets
    weight `thin` and the kept-whole strata weight 1: the weighted loss is then the uniform loss over all of
    train, up to the sampling of which frame of each run of `thin` was kept. The base `ridge ego` is weighted
    the same way, so every arm is still a correction to the prior the full split would give.
    """
    from . import planner, waymo_l0 as l0, waymo_stage_a as sa
    sel = np.flatnonzero(keep)
    seq, h, fut = ctx["seq"][sel], ctx["half"][sel], ctx["fut"][sel]
    sp = sa.Halves(ctx["df"], seq, h == 0, h == 1, ctx["seed"])
    sub = {k: ctx["sub"][k][sel] for k in lad.SUBSETS}
    w = torch.as_tensor(np.where(sub["pre_onset"] | sub["turn_yaw"], 1.0, float(thin)), device=lad.DEV).float()
    n, T = len(sel), fut.shape[1]
    F = torch.as_tensor(fut.reshape(n, -1), device=lad.DEV)
    Xe = planner.standardize(torch.as_tensor(ctx["ego"][sel], device=lad.DEV), sp.train)
    p_ego, st_ego, W_ego = l0.wridge_cv(Xe, F, sp, fut, w, sub["pre_onset"])
    base = planner.linear_apply(W_ego, Xe, np.arange(n))[0]
    R = F - base
    res_fut, off = R.reshape(-1, T, 2).cpu().numpy(), base[sp.val].reshape(-1, T, 2).cpu().numpy()
    preds = {lad.BASE: p_ego[:, 0]}
    rl.log.info("weighted (thin %d): %s lambda %.3g", thin, lad.BASE, st_ego["lam"])
    for name, X in feats.items():
        Xi = lad.standardize_np(X[sel], sp.train)
        p, st, _ = l0.wridge_cv(Xi, R, sp, res_fut, w, sub["pre_onset"])
        preds[name] = p[:, 0] + off
        rl.log.info("weighted: %s lambda %.3g", name, st["lam"])
        del Xi
        torch.cuda.empty_cache()
    sctx = {"sp": sp, "fname": ctx["fname"][sel], "seq": seq, "s": lad.s_ego_from_p0(ctx)[sel], "fut": fut,
            "direction": 0, "past": ctx["past"], "rows": ctx["rows"][sel]}
    lad.save_preds(rl, preds, sctx, "trainfitw")
    for k, t in lad.rejudge(rl.dir, "trainfitw").items():
        t.to_csv(rl.dir / f"{k}_weighted.csv", index=False)
        rl.log.info("weighted %s\n%s", k, t.to_markdown(index=False, floatfmt=".4f"))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("profile", "run", "subset", "check", "trainfit"))
    ap.add_argument("--model", default=None, help="subset: HF model id")
    ap.add_argument("--layers", default="18", help="subset: comma list of decoder layers to tap")
    ap.add_argument("--n", type=int, default=240, help="profile: rows of the qwenvid_p3 item list")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grid", default=None, help="H,W of the per-camera spatial summary, e.g. 4,4")
    ap.add_argument("--thin", type=int, default=None, help="run: stratified subsample, keep every n-th other frame")
    ap.add_argument("--name", default=SET)
    ap.add_argument("--vram-gb", type=float, default=30.0)
    a = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / total))
    grid = tuple(int(x) for x in a.grid.split(",")) if a.grid else None
    rl = RunLog("waymo_qwenvid", a.step)
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    if a.step == "profile":
        profile(rl, a.n, grid_hw=grid or (4, 4))
    elif a.step == "subset":
        rl.event("extract", **subset(rl, a.model, [int(x) for x in a.layers.split(",")], a.name, a.batch_size,
                                         a.compile))
    elif a.step == "check":
        check(rl, a.name)
    elif a.step == "trainfit":
        trainfit(rl, a.name, thin=a.thin)
    else:
        run(rl, a.batch_size, a.compile, a.workers, grid, a.thin, a.name)
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
