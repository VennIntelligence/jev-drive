"""V-JEPA 2 spatial tokens for the WL world model's token side arm (fc65452:todos/2026-09-29-wl-spatial-tokens.md).

Per index row of processed/<set>/index.parquet (the same rows, in the same order, as that set's z.npy): the three
cameras' 4-frame clips (WL `vjepa` step's extractor, so the same preprocessing) through V-JEPA 2 ViT-L; the last
temporal slice's 16 x 16 patch tokens are block-pooled to a 2 x 4 grid (rows x cols, row-major) per camera.
  extract  GPU, resumable, shardable (--shard i/n takes chunks with id % n == i): tok/c<id>.npy (n, 24, 1024) float16
           and tok/m<id>.npy (n, 3072), the full `mean` recomputed from the same forward, only for the equivalence check
  assemble every chunk -> <set>/tok.npy (N, 24, 1024) float16; token = camera * 8 + row * 4 + col, cameras
           front, front_left, front_right
  check    the recomputed `mean` against z.npy[:, D_OP:] (alignment + preprocessing) and the grid mean against it
Usage: python -m experiments.world_model.archive.wl_tokens extract|assemble|check [--set wl_gen] [--shard 0/2] [--limit N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.world_model.archive import nq4_w as W
from jevdrive.common import data_dir, get_logger

log = get_logger(__name__)
GRID = (2, 4)
CHUNK = 3000


def _dir(s: str) -> Path:
    return data_dir() / "processed" / s


def extract(s: str, shard: tuple[int, int], batch: int = 64, workers: int = 12, limit: int | None = None, rl=None):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from jevdrive import features as F
    d = _dir(s)
    t = pd.read_parquet(d / "index.parquet", columns=["frame_name", "files"])
    (d / "tok").mkdir(exist_ok=True)
    nchunk = -(-len(t) // CHUNK)
    todo = [c for c in range(nchunk) if c % shard[1] == shard[0] and not (d / "tok" / f"c{c:04d}.npy").exists()]
    if limit:
        todo = todo[:limit]
    fx = F.VJepaFeatures(frames=4, grid=GRID) if todo else None
    for c in todo:
        part = t.iloc[c * CHUNK:(c + 1) * CHUNK]
        files = [list(fs)[4 * k: 4 * k + 4] for fs in part.files for k in range(len(W.CAMS))]
        dl = DataLoader(W._Clips(files, fx.transform), batch_size=batch, num_workers=workers, prefetch_factor=4,
                        pin_memory=True, collate_fn=lambda b: torch.stack([x[1] for x in b]))
        tok, mean, pend, t0 = [], [], None, time.time()
        for v in tqdm(dl, desc=f"chunk {c}/{nchunk}", unit="batch", dynamic_ncols=True, mininterval=30):
            o = fx(v)
            cur = (o["grid"].half().to("cpu", non_blocking=True), o["mean"].half().to("cpu", non_blocking=True))
            if pend is not None:
                tok.append(pend[0].numpy()), mean.append(pend[1].numpy())
            pend = cur
        torch.cuda.synchronize()
        tok.append(pend[0].numpy()), mean.append(pend[1].numpy())
        n = len(part)
        tok = np.concatenate(tok).reshape(n, len(W.CAMS) * GRID[0] * GRID[1], -1)
        mean = np.concatenate(mean).reshape(n, -1)
        np.save(d / "tok" / f"m{c:04d}.npy", mean)
        np.save(d / "tok" / f"c{c:04d}.tmp.npy", tok)
        (d / "tok" / f"c{c:04d}.tmp.npy").replace(d / "tok" / f"c{c:04d}.npy")
        msg = f"chunk {c} rows {n} in {time.time() - t0:.0f} s ({3 * n / (time.time() - t0):.0f} clips/s)"
        log.info(msg)
        if rl:
            rl.event("chunk", chunk=c, rows=n, s=time.time() - t0)


def assemble(s: str) -> dict:
    d = _dir(s)
    n = len(pd.read_parquet(d / "index.parquet", columns=["frame_name"]))
    nchunk = -(-n // CHUNK)
    miss = [c for c in range(nchunk) if not (d / "tok" / f"c{c:04d}.npy").exists()]
    if miss:
        raise SystemExit(f"missing chunks {miss}")
    out = np.lib.format.open_memmap(d / "tok.tmp.npy", "w+", np.float16, (n, 24, 1024))
    for c in range(nchunk):
        out[c * CHUNK:(c + 1) * CHUNK] = np.load(d / "tok" / f"c{c:04d}.npy")
    out.flush()
    (d / "tok.tmp.npy").replace(d / "tok.npy")
    return {"rows": n, "shape": [n, 24, 1024], "gb": round(n * 24 * 1024 * 2 / 2**30, 2)}


def _cos(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def check(s: str, sample: int | None = None) -> dict:
    """cos(recomputed mean, stored mean) per row and camera; cos(mean of the grid tokens, stored mean)."""
    d = _dir(s)
    z = np.load(d / "z.npy", mmap_mode="r")
    n = len(z)
    res = {"mean": [], "grid_mean": []}
    for c in range(-(-n // CHUNK)):
        f = d / "tok" / f"m{c:04d}.npy"
        if not f.exists() or not (d / "tok" / f"c{c:04d}.npy").exists():
            continue
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        ref = np.asarray(z[sl, W.D_OP:]).astype(np.float32).reshape(-1, len(W.CAMS), 1024)
        m = np.load(f).astype(np.float32).reshape(-1, len(W.CAMS), 1024)
        g = np.load(d / "tok" / f"c{c:04d}.npy").astype(np.float32).reshape(-1, len(W.CAMS), 8, 1024).mean(2)
        for k, x in (("mean", m), ("grid_mean", g)):
            res[k].append(_cos(x.reshape(-1, 1024), ref.reshape(-1, 1024)))
    out = {}
    for k, v in res.items():
        v = np.concatenate(v)
        out[k] = {"clips": len(v), "min": float(v.min()), "p1": float(np.percentile(v, 1)),
                  "median": float(np.median(v)), "frac_ge_0.999": float((v >= 0.999).mean())}
    return out


def main():
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("extract", "assemble", "check"))
    ap.add_argument("--set", default="wl_gen")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    if a.step == "extract":
        rl = RunLog("wl", f"tokens_{i}of{n}")
        extract(a.set, (i, n), a.batch, a.workers, a.limit, rl)
        rl.close()
        r = {"done": True}
    else:
        r = {"assemble": assemble, "check": check}[a.step](a.set)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
