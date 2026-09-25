"""Qwen `L18_*` features on NAVSIM tokens: P3(d'')'s extractor unchanged, fed the clip a NAVSIM agent may see
(elicitation program E1 NAVSIM column, deviation [E1] (5): todos/2026-09-26-elicitation-program.md).

Clip = the 4 history frames of the token (-1.5, -1.0, -0.5, 0 s at 2 Hz; P5 / WOD clips are 0.2 s apart -- a
recorded input difference, not a change of recipe), cameras CAM_F0 / CAM_L0 / CAM_R0 as front / front_left /
front_right, 12 JPEG paths camera-major, oldest first (p4_carla.ClipFiles' layout). Eager, batch 2, the same
`waymo_qwenvid.make_fx` as qwenvid_p3 and P5.

  index <split>   processed/navsim_qwen/<split>/index.parquet (token, files) from runs/navsim_zs/index/<split>.pkl
  work <split>    claim chunks with a lock file (p5_qwen's), extract, meta.json last; start one per card
  check <split>   every chunk done -> exit 0
  load            {tap: (n, d)} aligned to a token list
"""
import os
import pickle

import numpy as np
import pandas as pd

from . import p5_qwen
from .common import data_dir, get_logger

log = get_logger(__name__)
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0")
CHUNK = 1500


def root(split: str, *parts):
    d = data_dir() / "processed" / "navsim_qwen" / split
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*parts)


def index(split: str) -> pd.DataFrame:
    with open(data_dir() / "runs/navsim_zs/index" / f"{split}.pkl", "rb") as f:
        idx = pickle.load(f)
    t = pd.DataFrame({"token": [e["token"] for e in idx],
                      "files": [[fr[c]["path"] for c in CAMS for fr in e["cams"]] for e in idx]})
    assert t.files.map(len).eq(12).all() and t.token.is_unique
    t["chunk"] = np.arange(len(t)) // CHUNK
    t.to_parquet(root(split, "index.parquet"), index=False)
    log.info("%s: %d tokens, %d chunks", split, len(t), t.chunk.nunique())
    return t


def work(rl, split: str, batch: int = 2, workers: int = 4):
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    t = pd.read_parquet(root(split, "index.parquet"))
    fx, n_done = None, 0
    for i, g in t.groupby("chunk"):
        dst = root(split, f"c{i:03d}")
        if p5_qwen._done(dst, g.token.rename("frame_name")):
            continue
        lock = root(split, f"c{i:03d}.lock")
        if not p5_qwen._claim(lock):
            continue
        try:
            if fx is None:
                fx = qv.make_fx(compile=False)
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "meta.json").unlink(missing_ok=True)
            st = F.extract(fx, g.files.map(list).tolist(), batch, workers, dst, rl, f"{split}/c{i:03d}", dataset=p4.ClipFiles)
            g[["token"]].rename(columns={"token": "frame_name"}).to_parquet(dst / "index.parquet", index=False)
            (dst / "meta.json").write_text(pd.Series({"recipe": "P3(d'') qwenvid, NAVSIM 2 Hz clip", "batch_size": batch,
                                                      "host_pid": os.getpid(), **st}).to_json())
            n_done += 1
            rl.log.info("%s chunk %d: %d clips, %.1f ms/frame", split, i, len(g), st["ms_per_frame"])
            rl.event("chunk_done", split=split, chunk=int(i), n=len(g), ms_per_frame=st["ms_per_frame"])
        finally:
            lock.unlink(missing_ok=True)
    rl.log.info("no chunk left to claim (%d done by this process)", n_done)


def check(split: str) -> bool:
    t = pd.read_parquet(root(split, "index.parquet"))
    bad = [i for i, g in t.groupby("chunk")
           if not p5_qwen._done(root(split, f"c{i:03d}"), g.token.rename("frame_name"))]
    log.info("%s: %d / %d chunks missing", split, len(bad), t.chunk.nunique())
    return not bad


def load(split: str, tokens, taps=("L18_last",)) -> dict:
    t = pd.read_parquet(root(split, "index.parquet"))
    pos, arrs = {}, {k: [] for k in taps}
    n = 0
    for i in sorted(t.chunk.unique()):
        d = root(split, f"c{i:03d}")
        names = pd.read_parquet(d / "index.parquet").frame_name.tolist()
        for k in taps:
            arrs[k].append(np.load(d / f"{k}.npy", mmap_mode="r"))
        pos.update({f: n + j for j, f in enumerate(names)})
        n += len(names)
    at = pd.Series(list(tokens)).map(pos)
    assert at.notna().all(), f"{int(at.isna().sum())} tokens without features"
    at = at.astype(int).to_numpy()
    return {k: np.concatenate(v)[at].astype(np.float32) for k, v in arrs.items()}


def main():
    import argparse
    import sys
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("index", "work", "check"))
    ap.add_argument("split")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "index":
        index(a.split)
    elif a.cmd == "check":
        sys.exit(0 if check(a.split) else 1)
    else:
        rl = RunLog("elicitation", f"navsim-qwen-{a.split}")
        work(rl, a.split, workers=a.workers)
        rl.close()


if __name__ == "__main__":
    main()
