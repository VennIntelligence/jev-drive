"""Night queue 2, N6 (todos/2026-09-26-night-queue-2.md): the backbone rows of the P5 v1 BehaviorAgent exam.
Frozen V-JEPA 2 / DINOv2 / SigLIP2 (and openpilot small, scripts/n6_op_small.sh) features, `ridge_late` and pair-Delta
heads, the p5_exam flips. The preprocessing is `jevdrive/features.py`'s classes unchanged; the heads are
`p5_exam.heads` and `reactivity_mc.fit_fold` unchanged; the exam is `p5_exam.exam` and `reactivity_mc.criteria`.

  plan      every (index row, camera) -> a unit = its 4-frame clip; units deduplicated on the JPEGs' real paths
  extract   one process per GPU shard: JPEG decode + the three models' transforms in DataLoader workers, the three
            models on one card (bf16), batches of units; per-shard npz of every tap (float16)
  check     batch-vs-single equivalence of the extractor on a few units
  finalize  per-row features (front | front_left | front_right concatenated) -> processed/<set>/bb_<backbone>/
  fit       per route-fold seed: `ridge_late <bb>` and pair-Delta (single stream on the Cinque / Lebowski prior, plus the
            dual stream bb + openpilot) for every backbone tap; exam, criteria -> runs/night2/n6/fit-seed<s>/
  report    3-seed table + judgement -> research/results/night2/N6/

Backbone taps (primary first): vjepa2 mean | last_mean; dinov2 patch_mean | cls; siglip2 patch_mean | pooled;
op small temporal; qwen L18_last (the reference row, reproduces the stored M-C "pair qwen").
"""
import json
import os
import time
from pathlib import Path

import numpy as np

CAMS = ("front", "front_left", "front_right")
TAPS = {"vjepa2": ("mean", "last_mean"), "dinov2": ("patch_mean", "cls"), "siglip2": ("patch_mean", "pooled")}
PRIMARY = {"vjepa2": "mean", "dinov2": "patch_mean", "siglip2": "patch_mean", "opsmall": "temporal", "qwen": "L18_last"}
DINO_SIZE = (350, 322)          # 25 x 23 patches of 14 px: DINOv2's nuScenes recipe (576 patches) at P5's aspect
SET = "carla_p5v1_ba"


def _root(*p) -> Path:
    os.environ.setdefault("P5_SET", SET)
    from . import p5_pairs as P
    return P.processed(*p)


# ================================================================ plan

def plan() -> dict:
    import pandas as pd
    t = pd.read_parquet(_root() / "index.parquet")
    real = {}

    def rp(f):
        d, b = f.rsplit("/", 1)
        if d not in real:
            real[d] = os.path.realpath(d)
        return real[d] + "/" + b
    keys, files = {}, []
    unit = np.empty((len(t), len(CAMS)), np.int64)
    for i, fs in enumerate(t.files):
        for c in range(len(CAMS)):
            clip = [rp(f) for f in fs[4 * c: 4 * c + 4]]
            k = "|".join(clip)
            if k not in keys:
                keys[k] = len(files)
                files.append(clip)
            unit[i, c] = keys[k]
    d = _root("bb")
    pd.DataFrame({"unit": np.arange(len(files)), "files": files}).to_parquet(d / "units.parquet", index=False)
    np.save(d / "row_units.npy", unit)
    t[["frame_name"]].to_parquet(d / "rows.parquet", index=False)
    return {"rows": len(t), "units": len(files), "row_cam_pairs": int(unit.size)}


# ================================================================ extract (GPU)

class Units:
    """torch Dataset: unit id, V-JEPA clip (4, 3, 256, 256), DINOv2 (3, 350, 322) and SigLIP2 (3, 384, 384) of its
    last frame -- each model's own `transform` on PIL images, run in the loader workers."""

    def __init__(self, files, fx):
        self.files, self.fx = files, fx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        ims = [Image.open(f).convert("RGB") for f in self.files[i]]
        return i, self.fx["vjepa2"].transform(ims), self.fx["dinov2"].transform(ims[-1]), self.fx["siglip2"].transform(ims[-1])


def _models():
    from . import features as F
    return {"vjepa2": F.VJepaFeatures(frames=4), "dinov2": F.DinoFeatures(size=DINO_SIZE), "siglip2": F.SiglipFeatures()}


def _collate(b):
    import torch
    return [x[0] for x in b], *(torch.stack([x[j] for x in b]) for j in (1, 2, 3))


def _forward(fx, v, d, s) -> dict:
    out = {}
    for name, x in (("vjepa2", v), ("dinov2", d), ("siglip2", s)):
        for tap, y in fx[name](x).items():
            out[f"{name} {tap}"] = y.half()
    return out


def extract(shard: str = "0/1", batch: int = 64, workers: int = 24, limit: int = 0, rl=None) -> dict:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    si, sn = map(int, shard.split("/"))
    u = pd.read_parquet(_root("bb") / "units.parquet")
    ids = u.unit.to_numpy()[si::sn][: limit or None]
    out = _root("bb", "shards") / f"shard{si}of{sn}{'-limit' if limit else ''}.npz"
    if out.exists():
        return {"skipped": str(out)}
    fx = _models()
    dl = DataLoader(Units([u.files.iloc[i] for i in ids], fx), batch_size=batch, num_workers=workers,
                    collate_fn=_collate, prefetch_factor=4, pin_memory=True)
    res, t0, n = {}, time.time(), 0
    pending = None
    for idx, v, d, s in tqdm(dl, desc=f"shard {si}/{sn}", mininterval=30):
        o = _forward(fx, v, d, s)
        o = {k: x.to("cpu", non_blocking=True) for k, x in o.items()}
        if pending is not None:                 # the previous batch's copies are done by now
            for k, x in pending.items():
                res.setdefault(k, []).append(x.numpy())
        pending, n = o, n + len(idx)
        if rl and (n // batch) % 50 == 0:
            el = time.time() - t0
            rl.event("progress", units=n, of=len(ids), wall_s=el, units_per_s=n / el)
            rl.info(f"{n}/{len(ids)} units, {n / el:.1f} units/s, ETA {(len(ids) - n) / (n / el) / 60:.1f} min")
    torch.cuda.synchronize()
    for k, x in pending.items():
        res.setdefault(k, []).append(x.numpy())
    arr = {k.replace(" ", "__"): np.concatenate(v) for k, v in res.items()}
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, unit=ids, **arr)
    tmp.replace(out)
    el = time.time() - t0
    info = {"shard": shard, "units": len(ids), "wall_s": el, "units_per_s": len(ids) / el,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    if rl:
        rl.event("end", **info)
    return info


def check(n: int = 8) -> dict:
    """Batch-of-n against one-at-a-time on the same units (bf16 batch-shape noise only)."""
    import pandas as pd
    import torch
    u = pd.read_parquet(_root("bb") / "units.parquet")
    fx = _models()
    ds = Units([u.files.iloc[i] for i in np.linspace(0, len(u) - 1, n).astype(int)], fx)
    items = [ds[i] for i in range(n)]
    b = _forward(fx, *(torch.stack([x[j] for x in items]) for j in (1, 2, 3)))
    rows = {}
    for i in range(n):
        s = _forward(fx, *(items[i][j][None] for j in (1, 2, 3)))
        for k in b:
            d = float((b[k][i].float() - s[k][0].float()).abs().max() / s[k][0].float().abs().max())
            rows[k] = max(rows.get(k, 0.0), d)
    return {"max_rel_diff_batch_vs_single": rows}


def finalize() -> dict:
    import pandas as pd
    d = _root("bb")
    shards = sorted((d / "shards").glob("shard*of*.npz"))
    shards = [s for s in shards if "limit" not in s.name]
    parts = [dict(np.load(s)) for s in shards]
    unit = np.concatenate([p["unit"] for p in parts])
    order = np.argsort(unit)
    assert (unit[order] == np.arange(len(unit))).all(), "units missing or duplicated across shards"
    ru = np.load(d / "row_units.npy")
    rows = pd.read_parquet(d / "rows.parquet")
    info = {}
    for bb, taps in TAPS.items():
        o = _root(f"bb_{bb}")
        rows.to_parquet(o / "index.parquet", index=False)
        for tap in taps:
            a = np.concatenate([p[f"{bb}__{tap}"] for p in parts])[order]
            np.save(o / f"{tap}.npy", a[ru].reshape(len(ru), -1))
            info[f"{bb} {tap}"] = [len(ru), int(a.shape[1] * len(CAMS))]
    return info


# ================================================================ heads and exam

def load_backbones(t, which) -> dict:
    """{"<bb> <tap>": (n, d) float32} aligned to the index `t`."""
    import pandas as pd
    from . import p5_openpilot, p5_pairs as P
    out = {}
    for bb in which:
        if bb == "qwen":
            out["qwen L18_last"] = P.load_features(t, ("L18_last",))["L18_last"]
        elif bb == "opsmall":
            out["opsmall temporal"] = p5_openpilot.load(t, ("small",), sub="op_streams_vis")["op-small temporal"]
        else:
            d = _root(f"bb_{bb}")
            names = pd.read_parquet(d / "index.parquet").frame_name
            at = t.frame_name.map(pd.Series(np.arange(len(names)), index=names))
            assert at.notna().all()
            for tap in TAPS[bb]:
                out[f"{bb} {tap}"] = np.load(d / f"{tap}.npy", mmap_mode="r")[at.astype(int).to_numpy()].astype(np.float32)
    return out


def fit(seed: int, which, rl, priors=("cinque", "lebowski"), eigh: str = "cpu") -> dict:
    import pandas as pd
    import torch
    from . import p5_exam as E, p5_openpilot, reactivity_mc as M
    os.environ.setdefault("P5_SET", SET)
    M.EIGH_DEVICE = eigh
    t, past, fut, obs, null, pairs = E.load()
    n = len(t)
    X = load_backbones(t, which)
    op = p5_openpilot.load(t, priors, sub="op_streams_vis")
    fold = E.folds(t, pairs, seed)
    rl.info(f"seed {seed}: {n} rows, taps {[(k, v.shape[1]) for k, v in X.items()]}")
    preds = E.heads(t, past, fut, X, fold, rl)                     # ridge ego + ridge_late <tap>
    preds = {k: v for k, v in preds.items() if k != "ridge ego"}
    F = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    Ego = torch.as_tensor(E.ego_input(t, past), device="cuda")
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    keep = {"prior": "prior", "M-C pair qwen": "pair-Δ", "M-C pair": "pair-Δ dual"}
    primary = {f"{bb} {tp}" for bb, tp in PRIMARY.items()}
    for m in priors:
        Xop = torch.as_tensor(op[f"op-{m} temporal"], device="cuda")
        for tap, Xa in X.items():
            if m != priors[0] and tap not in primary:      # side priors: primary taps only
                continue
            Q = torch.as_tensor(Xa, device="cuda")
            for f in range(E.K_FOLDS):
                ev = obs_rows[fold[obs_rows] == f]
                if not len(ev):
                    continue
                o = M.fit_fold(f, fold, t, F, Ego, Xop, Q, pr_ip, pr_im, pr_group, rl, f"{m}/{tap}",
                               arms=("pair", "pair qwen"))
                for arm, name in keep.items():
                    key = f"prior [{m}]" if arm == "prior" else f"{name} {tap} [{m}]"
                    preds.setdefault(key, np.full((n, 20, 2), np.nan, np.float32))[ev] = o[arm][ev].reshape(-1, 20, 2).cpu().numpy()
            del Q
            torch.cuda.empty_cache()
        del Xop
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(preds))
    crit = pd.concat([M.criteria(res, [k for k in preds if k.endswith(f"[{m}]") or k.startswith("ridge_late")],
                                 f"prior [{m}]").assign(prior=m) for m in priors])
    d = rl.dir
    res["flips"].to_csv(d / "flip_rates.csv", index=False)
    crit.to_csv(d / "criteria.csv", index=False)
    np.savez_compressed(d / "preds_obs.npz", rows=obs_rows, **{k: v[obs_rows] for k, v in preds.items()})
    rl.info("criteria\n" + crit.to_markdown(index=False, floatfmt=".3f"))
    return {"seed": seed, "examinees": len(preds)}


MC_STORED = "reactivity/mc-carla_p5v1_ba/20260925-233126"     # the stored M-C run on this set (seed 0)
RESULTS = Path(__file__).resolve().parents[1] / "research/results/night2/N6"


def _latest(pattern: str) -> Path:
    import glob
    from .common import data_dir
    fs = sorted(glob.glob(str(data_dir() / "runs" / pattern)))
    assert fs, pattern
    return Path(fs[-1])


def repro() -> "pd.DataFrame":
    """The Qwen reference row against the stored M-C run: CPU-eigh seed 0 must be bit-identical; GPU-eigh seed 0 is
    compared to the CPU one (float64 eigh on another device)."""
    import pandas as pd
    from .common import data_dir
    st = data_dir() / "runs" / MC_STORED
    cpu, gpu = _latest("night2/n6/fit-seed0/*"), _latest("night2/n6/fit-seed0-cuda/*")
    zs, zc, zg = (np.load(d / "preds_obs.npz") for d in (st, cpu, gpu))
    rows = []
    for m in ("cinque", "lebowski"):
        a, b = f"M-C pair qwen [{m}]", f"pair-Δ qwen L18_last [{m}]"
        if b not in zc.files:
            continue
        cs = pd.read_csv(st / "criteria.csv").set_index("arm").loc[a]
        cc = pd.read_csv(cpu / "criteria.csv").set_index("arm").loc[b]
        rows.append({"prior": m, "stored_vs_cpu_max_abs_m": float(np.nanmax(np.abs(zs[a] - zc[b]))),
                     "cpu_vs_gpu_max_abs_m": float(np.nanmax(np.abs(zc[b] - zg[b]))) if b in zg.files else np.nan,
                     "ped_flip_stored": cs.ped_flip, "ped_flip_cpu": cc.ped_flip,
                     "prior_stored_vs_cpu_max_abs_m": float(np.nanmax(np.abs(zs[f"prior [{m}]"] - zc[f"prior [{m}]"])))})
    return pd.DataFrame(rows)


def report() -> "pd.DataFrame":
    """3-seed table from the GPU-eigh fits: per examinee and seed, pedestrian flip [CI], null false flip (oos), the
    N6 criterion (CI low > null + 10 pp), cut-in flip and its delta against the prior."""
    import pandas as pd
    rows = []
    for s in (0, 1, 2):
        c = pd.read_csv(_latest(f"night2/n6/fit-seed{s}-cuda/*") / "criteria.csv")
        rows.append(c.assign(seed=s))
    c = pd.concat(rows, ignore_index=True)
    c["e_layer"] = c.ped_lo > c.null_ff_oos + 0.10
    long = c[["seed", "prior", "arm", "ped_reactive", "ped_flip", "ped_lo", "ped_hi", "null_ff_oos", "e_layer", "cutin_flip",
              "cutin_delta_vs_prior", "cutin_lo", "cutin_hi"]]
    g = long.groupby(["prior", "arm"], sort=False)
    summ = g.agg(ped_s0=("ped_flip", "first"), ped_mean=("ped_flip", "mean"), ped_min=("ped_flip", "min"),
                 ped_max=("ped_flip", "max"), ped_lo_min=("ped_lo", "min"), null_mean=("null_ff_oos", "mean"),
                 null_max=("null_ff_oos", "max"), cutin_mean=("cutin_flip", "mean"),
                 cutin_delta_mean=("cutin_delta_vs_prior", "mean"), seeds_pass=("e_layer", "sum"), n_seeds=("seed", "nunique")
                 ).reset_index()
    summ["verdict"] = np.where(summ.seeds_pass == summ.n_seeds, "E-layer signal",
                               np.where(summ.seeds_pass == 0, "none", "unstable"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    long.to_csv(RESULTS / "criteria_seeds.csv", index=False, float_format="%.4f")
    summ.to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f")
    return summ


def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("plan", "extract", "check", "finalize", "fit", "report"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbones", default="qwen,vjepa2,dinov2,siglip2,opsmall")
    ap.add_argument("--priors", default="cinque,lebowski")
    ap.add_argument("--eigh", default="cpu", help="fit: float64 eigh device of the pair-Delta solves (cuda on a busy box)")
    a = ap.parse_args()
    os.environ.setdefault("P5_SET", SET)
    tag = {"extract": f"extract-{a.shard.replace('/', 'of')}", "fit": f"fit-seed{a.seed}-{a.eigh}"}.get(a.step, a.step)
    rl = RunLog("night2", "n6", tag)
    rl.event("start", args=vars(a), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    rl.info(f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}, args {vars(a)}")
    if a.step == "plan":
        r = plan()
    elif a.step == "extract":
        r = extract(a.shard, a.batch, a.workers, a.limit, rl)
    elif a.step == "check":
        r = check()
    elif a.step == "finalize":
        r = finalize()
    elif a.step == "report":
        rp = repro()
        RESULTS.mkdir(parents=True, exist_ok=True)
        rp.to_csv(RESULTS / "qwen_reproduction.csv", index=False)
        rl.info("reproduction\n" + rp.to_markdown(index=False))
        sm = report()
        rl.info("summary\n" + sm.to_markdown(index=False, floatfmt=".3f"))
        r = {"rows": len(sm)}
    else:
        r = fit(a.seed, a.backbones.split(","), rl, tuple(a.priors.split(",")), a.eigh)
    rl.info(json.dumps(r, default=float))
    rl.event("result", **{"r": r})
    rl.close()


if __name__ == "__main__":
    main()
