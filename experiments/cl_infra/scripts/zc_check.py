#!/usr/bin/env python3
"""Zero-copy pool merge check (jevdrive.tokens; rf_pait.py merge default since 2026-10-20): a zero-copy merged pool B against the copied pool
A of the same sources, bit for bit, through every reader path. Box, repo root, op-train env (zc_merge.sh job bodies).

  rows   --a on1_JTPH --b on1_JTPHZ    CPU. Every array of tab / agent / road / drivable / route / feat / teacher equal (values, dtype, shape);
         every token row: A's front.npy (np.load mmap) vs B through jevdrive.tokens.open_front, contiguous chunks over all rows in parallel
         processes; random fancy gathers (unsorted, repeated, across part boundaries), a slot index, pp_train.open_front, pp_train.Tokens
         host mode (pageable gather) and RowCache on the CPU -> <out>/rows.json
  pinned --a --b                       GPU. pp_train.Tokens host mode with the page-locked gather (Tokens._pinned -> CatMM.take, the path
         rm_train uses with --prefetch > 1) and the on-card RowCache, random batches -> <out>/pinned.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts")]
from jevdrive.common import n_cpus  # noqa: E402
from jevdrive.tokens import CatMM, open_front  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, FEAT = DATA / "runs/op_parity/cache", DATA / "runs/route_mem/feat"
CHUNK = 1024


def same(x, y) -> bool:
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def files_equal(a, b) -> dict:
    """{file: [keys that differ]} over the per-row files, the feat file and the teacher."""
    F = [(CR / a / f, CR / b / f) for f in ("tab.npz", "agent.npz", "road.npz", "drivable.npz", "route.npz")]
    F += [(FEAT / f"{a}.npz", FEAT / f"{b}.npz"), (CR / f"{a}@warp/teacher.npz", CR / f"{b}@warp/teacher.npz")]
    out = {}
    for x, y in F:
        if not x.exists():
            assert not y.exists(), f"{y} exists, {x} does not"
            continue
        zx, zy = np.load(x), np.load(y)
        out[str(y.relative_to(DATA))] = sorted(set(zx.files) ^ set(zy.files)) + [k for k in zx.files if k in zy.files and not same(zx[k], zy[k])]
    return out


def chunk(job):
    """Rows [i, j) of A and B equal; then B's source pages of those rows leave the page cache (the sources are not what running jobs read,
    on1_JTPH is, so its pages stay)."""
    a, b, i, j = job
    fb = open_front(CR / f"{b}@warp/front.npy")
    ok = same(np.asarray(np.load(CR / f"{a}@warp/front.npy", mmap_mode="r")[i:j]), fb[i:j])
    rb = int(np.prod(fb.shape[1:])) * fb.dtype.itemsize
    for s, m in enumerate(fb.mms):
        lo, hi = max(i, fb.off[s]), min(j, fb.off[s + 1])
        if lo < hi:
            fd = os.open(fb.files[s], os.O_RDONLY)
            os.posix_fadvise(fd, int(m.offset + (lo - fb.off[s] + fb.ab[s, 0]) * rb), int((hi - lo) * rb), os.POSIX_FADV_DONTNEED)
            os.close(fd)
    return i, j, ok


def batches(fb: CatMM, n_batch: int, bs: int, rng) -> list:
    """Random row batches: uniform, sorted, with repeats, and straddling every part boundary."""
    n, B = len(fb), []
    for _ in range(n_batch):
        B.append(rng.integers(0, n, bs))
    B.append(np.sort(rng.integers(0, n, bs)))
    for o in fb.off[1:-1]:
        B.append(np.clip(o + np.arange(-bs // 2, bs // 2), 0, n - 1)[rng.permutation(bs)])
    return B


def cmd_rows(a):
    import torch
    import pp_train as T
    t0, rec = time.time(), dict(a=a.a, b=a.b)
    rec["files_diff"] = files_equal(a.a, a.b)
    fa, fb = np.load(CR / f"{a.a}@warp/front.npy", mmap_mode="r"), open_front(CR / f"{a.b}@warp/front.npy")
    assert isinstance(fb, CatMM), f"{a.b}: not a zero-copy pool"
    rec |= dict(rows=len(fa), shape_equal=fa.shape == fb.shape and fa.dtype == fb.dtype, parts=[str(f.relative_to(CR)) for f in fb.files],
                part_rows=np.diff(fb.off).tolist())
    jobs = [(a.a, a.b, i, min(i + CHUNK, len(fa))) for i in range(0, len(fa), CHUNK)]
    with ProcessPoolExecutor(a.workers or n_cpus()) as ex:
        bad = [(i, j) for i, j, ok in ex.map(chunk, jobs, chunksize=8) if not ok]
    rec["sweep"] = dict(chunks=len(jobs), rows=int(sum(j - i for _, _, i, j in jobs)), bad_chunks=bad, s=round(time.time() - t0, 1))
    rng = np.random.default_rng(0)
    B = batches(fb, a.batches, a.bs, rng)
    pf = T.open_front(CR / f"{a.b}@warp/front.npy")
    tok = T.Tokens([CR / f"{a.b}@warp/front.npy"], torch.device("cpu"), host=True)
    rc = T.RowCache(tok, np.concatenate(B[:4]))
    rec["gather"] = dict(
        batches=len(B), rows_per_batch=a.bs,
        catmm=all(same(np.asarray(fa[r]), fb[r]) for r in B),
        catmm_slot7=all(same(np.asarray(fa[r, 7]), fb[r, 7]) for r in B[:8]),
        pp_train_open_front=isinstance(pf, CatMM) and all(same(np.asarray(fa[r]), pf[r]) for r in B[:8]),
        tokens_host_pageable=all(same(np.asarray(fa[r]), tok[torch.as_tensor(r)].numpy()) for r in B),
        rowcache=all(same(np.asarray(fa[r]), rc[r].numpy()) for r in B[:4]))
    rec["pass_"] = bool(not any(rec["files_diff"].values()) and rec["shape_equal"] and not bad and all(v for k, v in rec["gather"].items()
                                                                                                        if isinstance(v, bool)))
    rec["s"] = round(time.time() - t0, 1)
    dump(Path(a.out) / "rows.json", rec)


def cmd_pinned(a):
    import torch
    import pp_train as T
    dev = torch.device("cuda")
    fa = np.load(CR / f"{a.a}@warp/front.npy", mmap_mode="r")
    tok = T.Tokens([CR / f"{a.b}@warp/front.npy"], dev, host=True)
    tok.pin = True
    B = batches(open_front(CR / f"{a.b}@warp/front.npy"), a.batches, a.bs, np.random.default_rng(1))
    pin = all(same(np.asarray(fa[r]), tok[torch.as_tensor(r)].cpu().numpy()) for r in B)
    rc = T.RowCache(tok, np.concatenate(B[:4]))
    rec = dict(a=a.a, b=a.b, batches=len(B), rows_per_batch=a.bs, tokens_host_pinned=pin,
               rowcache_card=all(same(np.asarray(fa[r]), rc[r].cpu().numpy()) for r in B[:4]))
    rec["pass_"] = bool(rec["tokens_host_pinned"] and rec["rowcache_card"])
    dump(Path(a.out) / "pinned.json", rec)


def dump(p: Path, rec: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec, default=str))
    if not rec["pass_"]:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("rows", "pinned"):
        p = sub.add_parser(c)
        p.add_argument("--a", default="on1_JTPH"), p.add_argument("--b", default="on1_JTPHZ"), p.add_argument("--out", required=True)
        p.add_argument("--batches", type=int, default=64), p.add_argument("--bs", type=int, default=96)
        p.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    dict(rows=cmd_rows, pinned=cmd_pinned)[a.cmd](a)
