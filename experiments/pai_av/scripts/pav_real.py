#!/usr/bin/env python3
"""REAL-GAP: real PAI-AV footage against the NuRec render of the same clip and time, before any camera download at scale.
Measurement only: the exam clips' real frames (test281) never enter a training row or a training cache; the probes fitted here are
read-outs, not models. Box, repo root.

  pick     (jevdrive env, CPU) 50 test281 clips with a usdz on disk + 50 non-exam PAI-AV clips (outside curated-val, the 441 = test281
           + train160), stratified by country group (US / DE / other) x day (hour_of_day in [6, 18)) / night, seed 0
           -> results/real_sample.csv (the split defs physical_ai_av/realgap-exam50 / realgap-nonexam50 are made from it)
  fetch    (jevdrive env, network) camera_front_wide_120fov mp4 + timestamps and egomotion of the sample, ONE ZIP MEMBER AT A TIME by
           range request (never a chunk zip), and the two calibration parquets of each chunk -> $PAV/real_sample/; aborts above MAX_GB
  render   the 50 exam scenes along the human log (onp_log_driver.py) through pai_box.sh, NTV0-L18-s0 fed, every session dumped once
           per second (8 slot frames, ego row, source JPEG) -> $RUN/render (pai_real.sh render)
  replay   (op-train env, one pool GPU) at every dumped decision with 8 rendered slots: real slot frames from the PAI-AV video at the same
           clip times through the driver's own conversion (pai_core.Maps) with (a) the PAI-AV calibration and (b) the session's NuRec
           CameraSpec; NTV0-L18-s0 / RSG-L18-s0 plans on render / real with the dumped ego row and the logged route memory; t0-slot
           tokens. Non-exam clips: real tokens at clip t0 = 2 .. 19 s -> $RUN/replay/
  report   (CPU) conversion check, plan / speed vs the human log, token cosine / CKA, probe transfer -> results/realgap.{md,json}
  fig      side-by-side real vs render frames -> figs/realgap_frames.png
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

R_ = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(R_))

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
PAV = D / "datasets/physical_ai_av"                    # as scripts/pav.py (constants copied: this file runs from a git checkout alone)
REV = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
HUB = PAV / "hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots" / REV
USDZ = D / "datasets/nurec/all-usdzs"
CLIP_MAP = D / "runs/pai_av/results/clip_map.csv"     # pav.py map: curated scene -> clip_id, group test281 / val160 / train1761
TOPIC = R_ / "experiments/pai_av"
SAMPLE = PAV / "real_sample"
RUN = D / "runs/pai_av/realgap"
CAM = "camera_front_wide_120fov"
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MAX_GB = float(os.environ.get("REALGAP_MAX_GB", "3.0"))
MODEL, YARD = "NTV0-L18-s0", "RSG-L18-s0"           # the PAI model; the yardstick checkpoint (a model change on the same input)


# ---------------------------------------------------------------- pick
def strata(df: pd.DataFrame) -> pd.Series:
    cg = np.where(df.country == "United States", "US", np.where(df.country == "Germany", "DE", "other"))
    dn = np.where((df.hour_of_day >= 6) & (df.hour_of_day < 18), "day", "night")
    return pd.Series([f"{a}/{b}" for a, b in zip(cg, dn)], index=df.index)


def strat_sample(df: pd.DataFrame, n: int, seed: int = 0, floor: int = 5) -> pd.DataFrame:
    """n rows: every stratum min(floor, its size), the rest by largest remainder of the stratum sizes."""
    s = strata(df)
    size = s.value_counts()
    take = np.minimum(size, floor)
    free = size - take
    share = free / free.sum() * (n - take.sum())
    take = take + np.floor(share).astype(int)
    for k in (share - np.floor(share)).sort_values(ascending=False).index[: n - int(take.sum())]:
        take[k] += 1
    rng = np.random.default_rng(seed)
    out = [df[s == k].iloc[rng.permutation(int((s == k).sum()))[:int(t)]] for k, t in take.items()]
    d = pd.concat(out)
    return d.assign(stratum=strata(d))


def cmd_pick(a):
    cm = pd.read_csv(CLIP_MAP)
    ex = cm[(cm.group == "test281") & cm.in_pai_av & cm.uuid.map(lambda u: (USDZ / f"{u}.usdz").exists())].copy()
    ex["chunk"] = ex.chunk.astype(int)
    e50 = strat_sample(ex, a.n, seed=0).assign(group="test281")
    M = pd.read_parquet(PAV / "meta/clip_index.parquet").join(pd.read_parquet(PAV / "meta/data_collection.parquet"))
    val = set(cm[cm.group.isin(["test281", "val160"])].clip_id)          # curated-val (441) is the exam side; nothing of it is sampled
    pool = M[M.clip_is_valid & (M.split == "train") & ~M.index.isin(val)].reset_index().rename(columns={"index": "clip_id"})
    pool = pool.dropna(subset=["hour_of_day", "country"])
    n50 = strat_sample(pool, a.n, seed=0).assign(group="nonexam", scene_id="", uuid="")
    cols = ["clip_id", "group", "scene_id", "uuid", "chunk", "split", "country", "hour_of_day", "platform_class", "stratum"]
    out = pd.concat([e50[cols], n50[cols]], ignore_index=True)
    out["chunk"] = out.chunk.astype(int)
    p = TOPIC / "results/real_sample.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(out.groupby(["group", "stratum"]).size().unstack(0).to_string())


# ---------------------------------------------------------------- fetch
class Budget:
    def __init__(self, gb):
        import threading
        self.max, self.n, self.lock = gb * 1e9, 0, threading.Lock()

    def add(self, b):
        with self.lock:
            self.n += b
            if self.n > self.max:
                raise RuntimeError(f"download budget exceeded: {self.n / 1e9:.2f} GB > {self.max / 1e9:.1f} GB")


def fetch_zip_members(path: str, members: list[str], dst: Path, budget: Budget) -> int:
    """Members of one remote chunk zip by range request (central directory + each member's local header and data only)."""
    from jevdrive.alpamayo.data import RemoteFile
    from jevdrive.hfdl import hf_url
    todo = [m for m in members if not (dst / m).exists()]
    if not todo:
        return 0
    rf = RemoteFile(hf_url(REPO_ID, path, dataset=True, rev=REV), streams=4)
    zf = zipfile.ZipFile(rf)
    dst.mkdir(parents=True, exist_ok=True)
    for m in todo:
        if m not in zf.NameToInfo:
            continue
        i = zf.getinfo(m)
        budget.add(30 + len(i.filename) + 1024 + i.compress_size)
        rf.prefetch(i.header_offset, 30 + len(i.filename) + 1024 + i.compress_size)
        tmp = dst / (m + ".part")
        tmp.write_bytes(zf.read(m))
        tmp.rename(dst / m)
    return rf.nbytes


def calib_file(kind: str, chunk: int, budget: Budget) -> Path:
    """calibration/<kind>/<kind>.chunk_NNNN.parquet: the downloader's copy or the hub cache if present, else fetched (small)."""
    from jevdrive.hfdl import download, hf_url
    rel = f"calibration/{kind}/{kind}.chunk_{chunk:04d}.parquet"
    for root in (PAV, HUB, PAV / "calib/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots" / REV):
        if (root / rel).exists():
            return root / rel
    p = SAMPLE / rel
    if not p.exists():
        download(hf_url(REPO_ID, rel, dataset=True, rev=REV), p, streams=1)
        budget.add(p.stat().st_size)
    return p


def cmd_fetch(a):
    import time
    s = pd.read_csv(TOPIC / "results/real_sample.csv")
    budget, t0 = Budget(MAX_GB), time.time()
    jobs = []
    for ch, g in s.groupby("chunk"):
        cl = list(g.clip_id)
        jobs.append((f"camera/{CAM}/{CAM}.chunk_{ch:04d}.zip", [f"{c}.{CAM}.{x}" for c in cl for x in ("mp4", "timestamps.parquet")], SAMPLE / "camera"))
        jobs.append((f"labels/egomotion/egomotion.chunk_{ch:04d}.zip", [f"{c}.egomotion.parquet" for c in cl], SAMPLE / "egomotion"))
    with ThreadPoolExecutor(a.workers) as ex:
        nb = sum(ex.map(lambda j: fetch_zip_members(*j, budget), jobs))
        cal = list(ex.map(lambda k: calib_file(*k, budget), [(k, int(c)) for c in s.chunk.unique() for k in ("camera_intrinsics", "sensor_extrinsics")]))
    miss = [c for c in s.clip_id if not (SAMPLE / "camera" / f"{c}.{CAM}.mp4").exists()]
    rep = dict(clips=len(s), chunks=int(s.chunk.nunique()), bytes_transferred=int(nb), budget_counted=int(budget.n), calib_files=len(cal),
               missing_video=miss, disk_bytes=int(sum(f.stat().st_size for f in SAMPLE.rglob("*") if f.is_file())), seconds=round(time.time() - t0))
    (SAMPLE / "fetch.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pick")
    p.add_argument("--n", type=int, default=50)
    p = sp.add_parser("fetch")
    p.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
