"""SCH GPU helper: lane D's NAVSIM Qwen chunks (jevdrive.navsim_qwen work) on an idle card, staged.

The per-chunk body is navsim_qwen.work's, unchanged (index.parquet chunking, make_fx, features.extract with
p4_carla.ClipFiles, index.parquet, meta.json last). A chunk that lane D's own `work` process already holds is staged
under processed/navsim_qwen/<split>_sch/ and moved into place by `install` only once that process is gone.

  verify <split> --n 24          first n clips of chunk 0 into a scratch dir, compared bit for bit with the stored c000
  run <split> --chunks 2         extract those chunks into the staging root (skips chunks done in either root)
  install <split> --owner-pid P  once P is gone: every chunk missing in the real root must be staged, finished and hold
                                 exactly the index's tokens; rename them into place (a partial target is replaced)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive import navsim_qwen as NQ, p5_qwen  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

TAPS = ("L18_last", "L18_mean", "vis_mean", "vit_mean")


def stage(split: str, *parts) -> Path:
    d = data_dir() / "processed" / "navsim_qwen" / f"{split}_sch"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*parts)


def extract_chunk(fx, rl, split, i, g, dst: Path, batch=2, workers=4):
    """navsim_qwen.work's loop body."""
    from jevdrive import features as F, p4_carla as p4
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").unlink(missing_ok=True)
    st = F.extract(fx, g.files.map(list).tolist(), batch, workers, dst, rl, f"{split}/c{i:03d}", dataset=p4.ClipFiles)
    g[["token"]].rename(columns={"token": "frame_name"}).to_parquet(dst / "index.parquet", index=False)
    (dst / "meta.json").write_text(pd.Series({"recipe": "P3(d'') qwenvid, NAVSIM 2 Hz clip", "batch_size": batch,
                                              "host_pid": os.getpid(), **st}).to_json())
    return st


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    from jevdrive import waymo_qwenvid as qv
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "run", "install"))
    ap.add_argument("split")
    ap.add_argument("--chunks", default="")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--owner-pid", type=int, default=0)
    a = ap.parse_args()
    t = pd.read_parquet(NQ.root(a.split, "index.parquet"))
    gs = dict(list(t.groupby("chunk")))
    if a.cmd == "verify":
        rl = RunLog("sch_gpu", f"navsim-qwen-verify-{a.split}")
        dst = stage(a.split, f"verify_{os.getpid()}")
        st = extract_chunk(qv.make_fx(compile=False), rl, a.split, 0, gs[0].iloc[:a.n], dst, workers=a.workers)
        ref = NQ.root(a.split, "c000")
        assert pd.read_parquet(ref / "index.parquet").frame_name.iloc[:a.n].tolist() == gs[0].token.iloc[:a.n].tolist()
        same = {k: bool(np.array_equal(np.load(dst / f"{k}.npy"), np.load(ref / f"{k}.npy", mmap_mode="r")[:a.n])) for k in TAPS}
        rl.log.info("verify %d clips: bit-identical %s, %.1f ms/frame", a.n, same, st["ms_per_frame"])
        rl.close()
        shutil.rmtree(dst)
        print(same, st["ms_per_frame"])
        sys.exit(0 if all(same.values()) else 1)
    if a.cmd == "run":
        rl = RunLog("sch_gpu", f"navsim-qwen-{a.split}-{os.getpid()}")
        fx = None
        for i in map(int, a.chunks.split(",")):
            name = f"c{i:03d}"
            if p5_qwen._done(NQ.root(a.split, name), gs[i].token) or p5_qwen._done(stage(a.split, name), gs[i].token):
                continue
            lock = stage(a.split, f"{name}.lock")
            if not p5_qwen._claim(lock):
                continue
            try:
                fx = fx or qv.make_fx(compile=False)
                st = extract_chunk(fx, rl, a.split, i, gs[i], stage(a.split, name), workers=a.workers)
                rl.log.info("%s chunk %d: %d clips, %.1f ms/frame", a.split, i, len(gs[i]), st["ms_per_frame"])
            finally:
                lock.unlink(missing_ok=True)
        rl.close()
        return
    if _alive(a.owner_pid):
        sys.exit(f"lane D's work process {a.owner_pid} is alive: not installing")
    todo = [i for i in gs if not p5_qwen._done(NQ.root(a.split, f"c{i:03d}"), gs[i].token)]
    for i in todo:
        d = stage(a.split, f"c{i:03d}")
        assert p5_qwen._done(d, gs[i].token), f"{d} not finished"
        assert pd.read_parquet(d / "index.parquet").frame_name.tolist() == gs[i].token.tolist(), f"{d}: rows differ"
        for k in TAPS:
            assert np.load(d / f"{k}.npy", mmap_mode="r").shape[0] == len(gs[i]), f"{d}/{k}.npy: row count"
    for i in todo:
        d, tgt = stage(a.split, f"c{i:03d}"), NQ.root(a.split, f"c{i:03d}")
        if tgt.exists():
            shutil.rmtree(tgt)
        d.rename(tgt)
        print(f"installed {d} -> {tgt}")


if __name__ == "__main__":
    main()
