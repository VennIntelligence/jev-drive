"""Night queue 3, lane C: frozen features on the P6 exam frames (todos/2026-09-26-night-queue-3.md, [C] entries).
The extractors are the P5 v1 ones unchanged; only the frame source differs.

  rows     the unique exam frames (jevdrive.nq3_p6.exam_frames) in P6 index order
  qwen     Qwen3-VL `L18_last` (P3(d'') qwenvid, p5_pairs.extract's recipe) in chunks c<NNN> under
           processed/carla_p6/features, sharded over processes (--shard i/n); read back with p5_pairs.load_features
  vjepa    V-JEPA 2 ViT-L `mean` / `last_mean` (n6_backbones' model and transform, 4-frame clip per camera) ->
           processed/carla_p6/bb_vjepa2/{index.parquet, mean.npy, last_mean.npy} (front | front_left | front_right)
  images   the current JPEG of every camera of every row -> processed/carla_p6/nq3_images.parquet (key, path) for
           `night2_n4 detect --images ... --out ...` (envs/ultralytics)
  tokens   the detections -> image-plane tokens with N4's rule and N4's PCA fitted on P5 v1 BA (so P5-trained students
           apply unchanged) -> processed/carla_p6/nq3_tokB.npy + nq3_tokB_index.parquet
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
SET = "carla_p6"
CAMS = ("front", "front_left", "front_right")
CHUNK = 1500


def proc(*p) -> Path:
    d = data_dir() / "processed" / SET
    d.joinpath(*p[:-1]).mkdir(parents=True, exist_ok=True) if len(p) > 1 else None
    return d.joinpath(*p)


def rows() -> pd.DataFrame:
    t = pd.read_parquet(proc("index.parquet"))
    need = set(pd.read_parquet(proc("nq3_exam_frames.parquet")).frame_name)
    return t[t.frame_name.isin(need)].reset_index(drop=True)


# ---------------------------------------------------------------- Qwen

def qwen(shard: str = "0/1", batch: int = 2, workers: int = 4):
    os.environ["P5_SET"] = SET
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    from .runlog import RunLog
    si, sn = map(int, shard.split("/"))
    t = rows()
    root = proc("features", "x")
    root = root.parent
    chunks = [t.iloc[i:i + CHUNK] for i in range(0, len(t), CHUNK)]
    left = [(i, c) for i, c in enumerate(chunks) if i % sn == si and not (root / f"c{i:03d}" / "meta.json").exists()]
    rl = RunLog("nq3_c", f"qwen-{si}of{sn}")
    rl.log.info("%d rows, %d chunks, %d left on shard %s", len(t), len(chunks), len(left), shard)
    if not left:
        rl.close()
        return
    fx = qv.make_fx(compile=False)
    for i, c in left:
        dst = root / f"c{i:03d}"
        dst.mkdir(parents=True, exist_ok=True)
        st = F.extract(fx, c.files.map(list).tolist(), batch, workers, dst, rl, f"p6/c{i:03d}", dataset=p4.ClipFiles)
        c[["frame_name"]].to_parquet(dst / "index.parquet", index=False)
        (dst / "meta.json").write_text(pd.Series({"recipe": "P3(d'') qwenvid", "batch_size": batch, **st}).to_json())
        rl.log.info("chunk %d: %d clips, %.1f ms/frame", i, len(c), st["ms_per_frame"])
        rl.event("chunk_done", chunk=i, n=len(c), ms_per_frame=st["ms_per_frame"])
    rl.close()


# ---------------------------------------------------------------- V-JEPA 2

class _Clips:
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        return i, self.tf([Image.open(f).convert("RGB") for f in self.files[i]])


def vjepa(batch: int = 64, workers: int = 12, limit: int = 0) -> dict:
    """One V-JEPA 2 unit per (row, camera) = its 4-frame clip; float16 taps, rows = cameras concatenated."""
    import torch
    from torch.utils.data import DataLoader
    from . import features as F
    from .runlog import RunLog
    t = rows()
    if limit:
        t = t.iloc[:: max(1, len(t) // limit)].iloc[:limit].reset_index(drop=True)
    files = [fs[4 * c: 4 * c + 4] for fs in t.files for c in range(len(CAMS))]
    rl = RunLog("nq3_c", "vjepa" + ("-limit" if limit else ""))
    fx = F.VJepaFeatures(frames=4)
    dl = DataLoader(_Clips(files, fx.transform), batch_size=batch, num_workers=workers, prefetch_factor=4,
                    pin_memory=True, collate_fn=lambda b: ([x[0] for x in b], torch.stack([x[1] for x in b])))
    res, t0, n, pending = {}, time.time(), 0, None
    for idx, v in dl:
        o = {k: y.half().to("cpu", non_blocking=True) for k, y in fx(v).items()}
        if pending is not None:
            for k, x in pending.items():
                res.setdefault(k, []).append(x.numpy())
        pending, n = o, n + len(idx)
        if (n // batch) % 50 == 0:
            el = time.time() - t0
            rl.log.info(f"{n}/{len(files)} units, {n / el:.1f} units/s, ETA {(len(files) - n) / (n / el) / 60:.1f} min")
            rl.event("progress", units=n, of=len(files), wall_s=el)
    torch.cuda.synchronize()
    for k, x in pending.items():
        res.setdefault(k, []).append(x.numpy())
    el = time.time() - t0
    info = {"rows": len(t), "units": len(files), "wall_s": el, "units_per_s": len(files) / el,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    if not limit:
        o = proc("bb_vjepa2", "x").parent
        t[["frame_name"]].to_parquet(o / "index.parquet", index=False)
        for k, v in res.items():
            a = np.concatenate(v)
            np.save(o / f"{k}.npy", a.reshape(len(t), -1))
    rl.event("end", **info)
    rl.log.info(json.dumps(info))
    rl.close()
    return info


# ---------------------------------------------------------------- YOLO image-plane tokens

def images() -> int:
    t = rows()
    recs = [{"key": f"{f}|{c}", "path": fs[4 * j + 3]} for f, fs in zip(t.frame_name, t.files) for j, c in enumerate(CAMS)]
    assert all(r["path"].rsplit("/", 2)[-2] == r["key"].split("|")[1] for r in recs)
    pd.DataFrame(recs).to_parquet(proc("nq3_images.parquet"), index=False)
    return len(recs)


def tokens(dets: str = "") -> dict:
    """night2_n4.tokens' row rule with N4's stored PCA (fitted on P5 v1 BA train-role detections)."""
    import glob
    from . import night2_n4 as N4
    t = rows()
    dd = Path(dets) if dets else proc("nq3_dets", "x").parent
    parts = sorted(glob.glob(str(dd / "part-*.parquet")))
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    Fe = np.concatenate([np.load(p.replace("part-", "feat-").replace(".parquet", ".npy")) for p in parts]).astype(np.float32)
    assert len(Fe) == len(d)
    pca = np.load(N4.out("pca.npz"))
    Z = (Fe - pca["mu"]) @ pca["V"]
    fn, cam = d.key.str.split("|").str[0].to_numpy(), d.key.str.split("|").str[1].to_numpy()
    d = d.assign(_z=np.arange(len(d)), fn=fn, cam=cam).sort_values(["key", "y1"], ascending=[True, False], kind="stable")
    d["rank"] = d.groupby("key").cumcount()
    d = d[d["rank"] < N4.K_TOK]
    ri = pd.Series(np.arange(len(t)), index=t.frame_name).reindex(d.fn).to_numpy()
    ci = d.cam.map({c: i for i, c in enumerate(CAMS)}).to_numpy()
    tok = np.zeros((len(t), len(CAMS), N4.K_TOK, N4.TOK_D), np.float32)
    W, H = d.W.to_numpy(np.float32), d.H.to_numpy(np.float32)
    v = np.zeros((len(d), N4.TOK_D), np.float32)
    v[np.arange(len(d)), ci] = 1
    v[:, 3], v[:, 4] = (d.x0 + d.x1).to_numpy() / 2 / W, (d.y0 + d.y1).to_numpy() / 2 / H
    v[:, 5], v[:, 6] = (d.x1 - d.x0).to_numpy() / W, (d.y1 - d.y0).to_numpy() / H
    v[np.arange(len(d)), 7 + d.prompt.map({c: i for i, c in enumerate(N4.CLASSES)}).to_numpy()] = 1
    v[:, 10] = d.score.to_numpy()
    v[:, 11:11 + N4.PCA_D] = Z[d._z.to_numpy()]
    v[:, -1] = 1
    tok[ri, ci, d["rank"].to_numpy()] = v
    np.save(proc("nq3_tokB.npy"), tok.reshape(len(t), -1))
    t[["frame_name"]].to_parquet(proc("nq3_tokB_index.parquet"), index=False)
    info = {"rows": len(t), "detections": len(parts) and int(len(Fe)), "rows_with_detection": float((tok[..., -1].sum((1, 2)) > 0).mean())}
    log.info(json.dumps(info))
    return info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qwen", "vjepa", "images", "tokens"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "qwen":
        qwen(a.shard, a.batch or 2, a.workers)
    elif a.cmd == "vjepa":
        print(vjepa(a.batch or 64, a.workers, a.limit))
    elif a.cmd == "images":
        print(images())
    else:
        print(tokens())


if __name__ == "__main__":
    main()
