"""Night-gap audit step 1: front-camera luminance of WOD-E2E val frames (context.stats has no time_of_day: the field is empty in the E2ED shards).
For every val sequence, frames 100..230 every 10 (the openpilot-eligible range) are read from the slim shard, decoded at 1/4 scale, and
mean luma plus the 99th-percentile luma (sky / lamps) are stored. Output $DATA_DIR/runs/night_gap/lum.parquet.
  $DATA_DIR/envs/jevdrive/bin/python experiments/leaderboard_audit/scripts/ng_lum.py"""
import io, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

D = Path(os.environ["DATA_DIR"])
SH = D / "datasets/waymo_e2e/front3"


def work(args):
    shard, rows = args
    out = []
    with open(SH / shard, "rb") as f:
        for seq, fr, off, ln in rows:
            f.seek(off); b = f.read(ln)
            im = Image.open(io.BytesIO(b)); im.draft("L", (im.width // 4, im.height // 4)); a = np.asarray(im.convert("L"), np.float32)
            top = a[: a.shape[0] // 2]
            out.append((seq, fr, a.mean(), np.percentile(a, 99), top.mean()))
    return out


if __name__ == "__main__":
    ix = pd.read_parquet(D / "processed/waymo_e2e/index.parquet", columns=["sequence", "frame", "split", "shard", "front_off", "front_len", "has_future"])
    v = ix[(ix.split == "val") & (ix.frame >= 100) & (ix.frame % 10 == 0)]
    jobs = [(str(s), list(zip(g.sequence.astype(str), g.frame, g.front_off, g.front_len))) for s, g in v.groupby("shard", observed=True)]
    n = int(os.environ.get("NPROC", 32))
    with ProcessPoolExecutor(n) as ex:
        res = [r for part in ex.map(work, jobs) for r in part]
    o = D / "runs/night_gap"; o.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res, columns=["sequence", "frame", "luma", "luma_p99", "luma_top"]).to_parquet(o / "lum.parquet")
    print(len(res))
