"""WL-2 feature extraction, pipelined with the fork generation (todos/2026-09-30-wl2-feat.md).

Same features as WL-1 (`wl_data` z = openpilot `temporal` | V-JEPA `mean`, `wl_tokens` 2 x 4 spatial grid), but keyed by
frame_name instead of by index-row chunk, so it can run on the runs finished so far and be re-scanned while the generation
goes on (the index, and with it the row order, only settles at the end).
  vjepa    GPU. Loop: finished runs not yet featurised -> chunks of ~3 000 rows -> the three cameras' 4-frame clips through
           V-JEPA 2 ViT-L (one forward gives `mean` and the last-slice grid tokens) -> processed/wl2_gen/vjepa/c1<seq>.npz
           (frame_name, mean; the format `wl_data.vjepa` / `zmat` read) and processed/wl2_gen/tok/x<seq>.npz (frame_name,
           grid (n, 24, 1024) fp16, ey (n, 2): e_y / e_psi of the ego pose against the run's route.json polyline, the
           WL-1 dry-run ego channel). Frames are decoded once per worker (LRU), not once per clip: the output is bit-identical
           to `VJepaFeatures.transform`, 4x fewer JPEG decodes.
  op       Loop: openpilot `temporal` streams for finished runs without one (`scripts/p5_openpilot.py --keys`), plan =
           `wl_data.opspec`'s stream layout for those runs, output into processed/wl2_gen/op_streams_vis/cinque (through a
           symlink of the scratch set wl2_gen_feat, so wl_data's own op_plan.json is never touched).
  assemble after `wl_data index` + `z`: tok.npy (N, 24, 1024) fp16 and ey.npy (N, 2), rows aligned with index.parquet.
  check    the checklist numbers (shapes, NaN, norms, alignment) -> json
Both loops stop after the pass that starts once runs/wl2/pipe/STATUS.md has the "generation end" line (or on pipe/ERROR).
Usage (CUDA_VISIBLE_DEVICES set outside): python -m jevdrive.wl2_feat vjepa|op|assemble|check [--limit-rows N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("WL_NAME", "wl2")
from . import wl as WL  # noqa: E402  (WL_NAME must be set before the import)
from . import wl_data as D  # noqa: E402
from .common import data_dir, get_logger  # noqa: E402

log = get_logger(__name__)
CAMS = D.CAMS
CHUNK, MIN_CHUNK, SEQ0 = 3000, 1500, 1000
OUT = data_dir() / "runs" / WL.NAME / "gen"
PIPE = data_dir() / "runs" / WL.NAME / "pipe"


def pdir(*p) -> Path:
    return D.pdir(*p)


def gen_finished() -> bool:
    if (PIPE / "ERROR").exists():
        raise SystemExit("pipe/ERROR present, stopping")
    s = PIPE / "STATUS.md"
    return s.exists() and "generation end" in s.read_text()


# ================================================================ rows of a run (the layout of wl_data.index)

def run_rows(rid: str, adir: str) -> pd.DataFrame:
    """frame_name, tick, frame and the 12 clip files (3 cameras x the frame and the 3 before it, the earliest repeated
    where the run starts) of every camera frame of a run: exactly wl_data.index's `files` column."""
    a = Path(adir)
    fr = pd.read_json(a / "frames.jsonl", lines=True)
    fr = fr[fr.files.map(bool)].sort_values("tick").reset_index(drop=True)
    cf = [[f"{adir}/{fr.files[i][c]}" for c in CAMS] for i in range(len(fr))]
    files = [sum(([cf[i - min(i, 3 - q)][c] for q in range(4)] for c in range(3)), []) for i in range(len(fr))]
    return pd.DataFrame({"route_id": rid, "adir": adir, "tick": fr.tick.astype(int), "frame": fr.frame.astype(int),
                         "frame_name": [f"{rid}-{f:07d}" for f in fr.frame], "files": files})


def _rr(a):
    return run_rows(*a)


# ================================================================ V-JEPA mean + grid

class Clips:
    """The clip dataset of wl_tokens / nq4_w with a per-worker LRU of transformed frames (consecutive clips of a camera
    share 3 of their 4 frames); each frame goes through fx.tf alone, as VJepaFeatures.transform does."""

    def __init__(self, files, tf, cache=48):
        self.files, self.tf, self.cap, self.c = files, tf, cache, OrderedDict()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        out = []
        for f in self.files[i]:
            x = self.c.get(f)
            if x is None:
                x = self.tf(Image.open(f).convert("RGB"))
                self.c[f] = x
                if len(self.c) > self.cap:
                    self.c.popitem(last=False)
            else:
                self.c.move_to_end(f)
            out.append(x)
        return i, torch.stack(out)


def extract_rows(files12: list, fx, batch: int = 64, workers: int = 8):
    """(n, 24, 1024) grid and (n, 3072) mean, float16, for n rows of 12 clip files (camera-major, 4 frames each)."""
    import torch
    from torch.utils.data import DataLoader
    n = len(files12)
    clips = [list(fs)[4 * c: 4 * c + 4] for fs in files12 for c in range(len(CAMS))]
    dl = DataLoader(Clips(clips, fx.tf), batch_size=batch, num_workers=workers, prefetch_factor=4, pin_memory=True,
                    collate_fn=lambda b: torch.stack([x[1] for x in b]))
    grid, mean, pend = [], [], None
    for v in dl:
        o = fx(v)
        cur = (o["grid"].half().to("cpu", non_blocking=True), o["mean"].half().to("cpu", non_blocking=True))
        if pend is not None:
            grid.append(pend[0].numpy()), mean.append(pend[1].numpy())
        pend = cur
    torch.cuda.synchronize()
    grid.append(pend[0].numpy()), mean.append(pend[1].numpy())
    return np.concatenate(grid).reshape(n, 24, 1024), np.concatenate(mean).reshape(n, -1)


def done_runs() -> set:
    s = set()
    for f in pdir("tok").glob("x*.npz"):
        with np.load(f) as z:
            s |= set(map(str, z["route_id"]))
    return s


def ego_of(rows: pd.DataFrame, workers: int) -> np.ndarray:
    from multiprocessing import Pool
    from .wl_dryrun import _ego_group
    g = [(a, x.frame.to_numpy()) for a, x in rows.groupby("adir", sort=False)]
    with Pool(workers) as p:
        res = p.map(_ego_group, g)
    out = np.full((len(rows), 2), np.nan, np.float32)
    pos = {a: x.index.to_numpy() for a, x in rows.groupby("adir", sort=False)}
    for (a, _), o in zip(g, res):
        out[pos[a]] = o
    return out


def vjepa_loop(batch: int, workers: int, limit_rows: int | None, rl=None):
    from . import features as F
    pdir("tok").mkdir(exist_ok=True), pdir("vjepa").mkdir(exist_ok=True)
    fx, t_start, tot = None, time.time(), 0
    final = False
    while True:
        end = gen_finished()
        r = D.finished(OUT)
        r = r[~r.route_id.isin(done_runs())].sort_values("route_id").reset_index(drop=True)
        if len(r) == 0 and end:
            break
        # rows of the not yet featurised runs, taken chunk by chunk
        take, n = [], 0
        for rid, adir in zip(r.route_id, r.adir):
            take.append((rid, adir))
            n += 27
            if n >= CHUNK or (limit_rows and n >= limit_rows):
                break
        if len(take) == 0 or (n < MIN_CHUNK and not end and not limit_rows):
            log.info(f"{len(r)} runs waiting, sleeping (generation end: {end})")
            time.sleep(120)
            continue
        from multiprocessing import Pool
        with Pool(min(workers, 8)) as p:
            rows = pd.concat(p.map(_rr, take), ignore_index=True)
        ok = np.array([all(os.path.exists(f) for f in fs) for fs in rows.files])
        if not ok.all():
            bad = sorted(set(rows.route_id[~ok]))
            with open(pdir("tok", "missing.jsonl"), "a") as f:
                f.write(json.dumps({"runs": bad, "time": time.strftime("%F %T")}) + "\n")
            log.warning(f"{len(bad)} runs with missing frames skipped: {bad[:5]}")
            rows = rows[rows.route_id.isin(set(rows.route_id[ok]) - set(bad))].reset_index(drop=True)
        t0 = time.time()
        fx = fx or F.VJepaFeatures(frames=4, grid=(2, 4))
        grid, mean = extract_rows(rows.files.tolist(), fx, batch, workers)
        ey = ego_of(rows, workers)
        seq = SEQ0 + len(list(pdir("tok").glob("x*.npz")))
        names = rows.frame_name.to_numpy().astype(str)
        np.savez(pdir("vjepa", f"c{seq:04d}.tmp.npz"), frame_name=names, mean=mean)
        pdir("vjepa", f"c{seq:04d}.tmp.npz").replace(pdir("vjepa", f"c{seq:04d}.npz"))
        np.savez(pdir("tok", f"x{seq:04d}.tmp.npz"), frame_name=names, route_id=np.array(sorted(set(rows.route_id))).astype(str),
                 grid=grid, ey=ey)
        pdir("tok", f"x{seq:04d}.tmp.npz").replace(pdir("tok", f"x{seq:04d}.npz"))
        tot += len(rows)
        el = time.time() - t0
        msg = (f"chunk {seq}: {rows.route_id.nunique()} runs, {len(rows)} rows in {el:.0f} s ({3 * len(rows) / el:.0f} clips/s); "
               f"total {tot} rows in {(time.time() - t_start) / 60:.1f} min; {len(r) - len(take)} runs waiting; end={end}")
        log.info(msg)
        if rl:
            rl.event("chunk", seq=seq, rows=len(rows), s=el, total=tot)
        if limit_rows:
            break


# ================================================================ openpilot temporal

def op_stream(g: pd.DataFrame, src: str | None) -> dict:
    """wl_data.opspec's stream of one run: the source run's frames before the first saved tick, then the run's own."""
    names, files = [], []
    if src:
        sf = pd.read_json(Path(src) / "frames.jsonl", lines=True)
        sf = sf[sf.tick < g.tick.min()]
        sroute = Path(src).parent.name
        names += [f"{sroute}-{f:07d}" for f in sf.frame]
        files += [[f"{src}/{r[c]}" for c in CAMS] for r in sf.files]
    n0 = len(names)
    names += g.frame_name.tolist()
    files += [[f[4 * c + 3] for c in range(3)] for f in g.files]
    return {"key": f"wl_{g.route_id.iloc[0]}", "names": names, "targets": list(range(n0, len(names))), "files": files, "gaps": 0}


def _os(a):
    rid, adir, src = a
    return op_stream(run_rows(rid, adir), src)


def op_loop(workers: int, per_cycle: int, limit: int | None, shard: tuple[int, int] = (0, 1)):
    from multiprocessing import Pool
    from .p5_openpilot import carla_calib
    scratch = data_dir() / "processed" / f"{WL.NAME}_gen_feat" / f"shard{shard[0]}of{shard[1]}"
    scratch.mkdir(parents=True, exist_ok=True)
    link = scratch / "op_streams_vis"
    if not link.exists():
        link.symlink_to(Path("..") / ".." / f"{WL.NAME}_gen" / "op_streams_vis")
    outd = pdir("op_streams_vis", "cinque")
    outd.mkdir(parents=True, exist_ok=True)
    opy = data_dir() / "envs" / "openpilot" / "bin" / "python"
    calib, t_start, tot = carla_calib(), time.time(), 0
    while True:
        end = gen_finished()
        r = D.finished(OUT)
        have = {f.stem for f in outd.glob("wl_*.npz") if not f.name.endswith(".tmp.npz")}
        r = r[~("wl_" + r.route_id).isin(have)].sort_values("route_id")
        r = r[r.route_id.astype(np.int64) % shard[1] == shard[0]].reset_index(drop=True)
        if len(r) == 0 and end:
            break
        if len(r) == 0 or (len(r) < 100 and not end):
            log.info(f"{len(r)} streams waiting, sleeping (generation end: {end})")
            time.sleep(120)
            continue
        r = r.iloc[: limit or per_cycle]
        src = r.src_dir if "src_dir" in r else pd.Series([None] * len(r))
        with Pool(min(workers, 8)) as p:
            streams = p.map(_os, [(a, b, c if isinstance(c, str) else None) for a, b, c in zip(r.route_id, r.adir, src)])
        (scratch / "op_plan.json").write_text(json.dumps({"calib": calib, "streams": streams}))
        (scratch / "keys.txt").write_text("\n".join(s["key"] for s in streams))
        t0 = time.time()
        env = {**os.environ, "P5_SET": f"{WL.NAME}_gen_feat/shard{shard[0]}of{shard[1]}", "OMP_NUM_THREADS": "2"}
        rc = subprocess.call([str(opy), "scripts/p5_openpilot.py", "--models", "cinque", "--arrays", "temporal", "--out-sub",
                              "op_streams_vis", "--workers", str(workers), "--keys", f"@{scratch / 'keys.txt'}"],
                             env=env, cwd=Path(__file__).resolve().parents[1])
        if rc:
            raise SystemExit(f"p5_openpilot exit {rc}")
        tot += len(streams)
        nf = sum(len(s["names"]) for s in streams)
        log.info(f"op cycle: {len(streams)} streams, {nf} frames in {time.time() - t0:.0f} s; total {tot} streams in "
                 f"{(time.time() - t_start) / 60:.1f} min; {len(r)} was the batch; end={end}")
        if limit:
            break


# ================================================================ assemble / check

def assemble() -> dict:
    t = pd.read_parquet(pdir("index.parquet"), columns=["frame_name"])
    n = len(t)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    tok = np.lib.format.open_memmap(pdir("tok.tmp.npy"), "w+", np.float16, (n, 24, 1024))
    ey = np.full((n, 2), np.nan, np.float32)
    got = np.zeros(n, bool)
    for f in sorted(pdir("tok").glob("x*.npz")):
        with np.load(f) as z:
            i = pos.reindex(z["frame_name"].astype(str)).to_numpy()
            ok = ~np.isnan(i)
            ii = i[ok].astype(int)
            o = np.argsort(ii)
            tok[ii[o]] = z["grid"][ok][o]
            ey[ii] = z["ey"][ok]
            got[ii] = True
    tok.flush()
    pdir("tok.tmp.npy").replace(pdir("tok.npy"))
    np.save(pdir("ey.npy"), ey)
    np.save(pdir("tok_ok.npy"), got)
    return {"rows": n, "with_tokens": int(got.sum()), "tok_gb": round(n * 24 * 1024 * 2 / 2**30, 2)}


def check() -> dict:
    t = pd.read_parquet(pdir("index.parquet"), columns=["route_id", "frame_name"])
    z, zok = np.load(pdir("z.npy"), mmap_mode="r"), np.load(pdir("z_ok.npy"))
    tok, tok_ok, ey = np.load(pdir("tok.npy"), mmap_mode="r"), np.load(pdir("tok_ok.npy")), np.load(pdir("ey.npy"))
    out = {"rows": len(t), "runs": int(t.route_id.nunique()), "z_shape": list(z.shape), "tok_shape": list(tok.shape),
           "z_ok": int(zok.sum()), "tok_ok": int(tok_ok.sum()), "ey_nan": int(np.isnan(ey[:, 0]).sum())}
    idx = np.random.RandomState(0).choice(len(t), 20000, replace=False)
    idx.sort()
    zz, tt = np.asarray(z[idx]).astype(np.float32), np.asarray(tok[idx]).astype(np.float32)
    out["z_nan_frac_sample"] = float(np.isnan(zz).any(1).mean())
    out["tok_nan_frac_sample"] = float(np.isnan(tt).any((1, 2)).mean())
    nz = np.linalg.norm(zz[:, :512], axis=1), np.linalg.norm(zz[:, 512:], axis=1)
    out["z_norm_op_pct_1_50_99"] = np.percentile(nz[0], [1, 50, 99]).round(3).tolist()
    out["z_norm_vjepa_pct_1_50_99"] = np.percentile(nz[1], [1, 50, 99]).round(3).tolist()
    out["tok_norm_pct_1_50_99"] = np.percentile(np.linalg.norm(tt, axis=2).ravel(), [1, 50, 99]).round(3).tolist()
    out["ey_abs_median_p95"] = [float(np.nanmedian(np.abs(ey[:, 0]))), float(np.nanpercentile(np.abs(ey[:, 0]), 95))]
    return out


def main():
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("vjepa", "op", "assemble", "check"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-cycle", type=int, default=1500)
    ap.add_argument("--shard", default="0/1", help="op: runs with route_id %% n == i")
    ap.add_argument("--limit-rows", type=int, help="vjepa: one chunk of about this many rows, then stop (staged launch)")
    ap.add_argument("--limit-streams", type=int, help="op: one cycle of this many streams, then stop")
    a = ap.parse_args()
    if a.step == "vjepa":
        rl = RunLog("wl2", "feat_vjepa")
        vjepa_loop(a.batch, a.workers, a.limit_rows, rl)
        rl.close()
    elif a.step == "op":
        op_loop(a.workers, a.per_cycle, a.limit_streams, tuple(map(int, a.shard.split('/'))))
    else:
        print(json.dumps({"assemble": assemble, "check": check}[a.step](), indent=1))


if __name__ == "__main__":
    main()
