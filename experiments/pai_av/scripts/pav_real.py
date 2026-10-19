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


# ---------------------------------------------------------------- replay
def quat_R(qx, qy, qz, qw) -> np.ndarray:
    """Unit quaternion (scalar last) -> rotation matrix (sh30_driver.quat_R)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pav_calib(clip: str, chunk: int) -> dict:
    """The PAI-AV calibration of the front-wide camera -> pai_core.project's spec, R (camera -> rig), translation (rig frame)."""
    b = Budget(0.01)
    i = pd.read_parquet(calib_file("camera_intrinsics", chunk, b)).loc[(clip, CAM)]
    e = pd.read_parquet(calib_file("sensor_extrinsics", chunk, b)).loc[(clip, CAM)]
    spec = dict(model="ftheta", c=[float(i.cx), float(i.cy)], native=[int(i.width), int(i.height)], windshield="",
                fw=[float(i[f"fw_poly_{k}"]) for k in range(5)], bw=[float(i[f"bw_poly_{k}"]) for k in range(5)], max_angle=0.0)
    return dict(spec=spec, R=quat_R(e.qx, e.qy, e.qz, e.qw), t=np.array([e.x, e.y, e.z], float))


def usdz_clock(uuid: str) -> dict:
    """NuRec clip start (the rollout clock is the NuRec absolute clock) and its first front-wide frame time."""
    with zipfile.ZipFile(USDZ / f"{uuid}.usdz") as zf:
        di = json.loads(zf.read("data_info.json"))
    sh = di["shards"][0]
    return dict(start_us=int(di["pose-range"]["start-timestamp_us"]), cam0_us=int(sh["sensors"][CAM]["frame-range"]["start-timestamp_us"]),
                n_frames=int(sh["sensors"][CAM]["frame-range"]["num-frames"]))


class Video:
    """One PAI-AV front-wide clip: frame timestamps (clip clock, us) and decoding of selected frames to YCbCr (as pai_core.decode)."""

    def __init__(self, clip: str):
        self.mp4 = SAMPLE / "camera" / f"{clip}.{CAM}.mp4"
        self.ts = pd.read_parquet(SAMPLE / "camera" / f"{clip}.{CAM}.timestamps.parquet").timestamp.to_numpy(np.int64)

    def nearest(self, t_us: np.ndarray):
        j = np.clip(np.searchsorted(self.ts, t_us), 1, len(self.ts) - 1)
        j = np.where(np.abs(self.ts[j - 1] - t_us) <= np.abs(self.ts[j] - t_us), j - 1, j)
        return j, self.ts[j] - t_us

    def decode(self, idx) -> dict:
        import cv2
        from PIL import Image
        want, out = set(int(i) for i in idx), {}
        cap = cv2.VideoCapture(str(self.mp4))
        for i in range(max(want) + 1):
            if not cap.grab():
                break
            if i in want:
                ok, bgr = cap.retrieve()
                if ok:
                    out[i] = np.asarray(Image.fromarray(bgr[:, :, ::-1]).convert("YCbCr"))
        cap.release()
        return out


def ego_targets(clip: str, t_clip_s: np.ndarray) -> dict:
    """Probe targets from the PAI-AV egomotion at clip times t (s): speed now, speed change and heading change over the next 3 s."""
    e = pd.read_parquet(SAMPLE / "egomotion" / f"{clip}.egomotion.parquet")
    t = e.timestamp.to_numpy() * 1e-6
    v = np.hypot(e.vx.to_numpy(), e.vy.to_numpy())
    q = e[["qw", "qx", "qy", "qz"]].to_numpy()
    yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    f = lambda x, tt: np.where((tt >= t[0]) & (tt <= t[-1]), np.interp(tt, t, x), np.nan)  # noqa: E731
    return dict(v0=f(v, t_clip_s), dv3=f(v, t_clip_s + 3) - f(v, t_clip_s), dyaw3=np.degrees(f(yaw, t_clip_s + 3) - f(yaw, t_clip_s)))


def load_render():
    """drive.jsonl inference records with the log fields {(scene, k): rec}, the session start records {scene: rec}, dump files."""
    recs, starts, dumps = {}, {}, []
    for f in sorted((RUN / "render/runs").glob("*/driver/drive.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "drive" and r.get("infer") and "log_poses" in r:
                recs[(r["scene"], r["k"])] = r
        for line in open(f.parent / "start.jsonl"):
            r = json.loads(line)
            starts[r["scene"]] = r
        dumps += sorted((f.parent / "dump").glob("*.npz"))
    return recs, starts, dumps


def relative(a, b):
    """Poses b (..., 3) in the frame of pose a (both in one frame)."""
    b = np.asarray(b, np.float64)
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[..., 0] - a[0], b[..., 1] - a[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, b[..., 2] - a[2]], -1)


def shift(a: np.ndarray, b: np.ndarray):
    """Phase-correlation shift (dx, dy) in px of luma image b against a, and its response."""
    import cv2
    w = cv2.createHanningWindow(a.shape[::-1], cv2.CV_64F)
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), w)
    return float(dx), float(dy), float(resp)


def cmd_replay(a):
    import torch
    from jevdrive.data import splits
    from jevdrive.openpilot.frames import unpack_luma
    from jevdrive.run import Run
    for d in ("experiments/alpasim/lib", "experiments/route_mem/lib", "experiments/op_parity/scripts", "experiments/onpol_pai/lib", "lib",
              "experiments/body1/lib"):
        sys.path.insert(0, str(R_ / d))
    import pai_core as PC
    import sh30_core as C
    S = pd.read_csv(TOPIC / "results/real_sample.csv")
    out = RUN / a.out
    out.mkdir(parents=True, exist_ok=True)
    with Run("pai_av", "realgap-replay", config=vars(a)) as run:
        ex, ne = splits.load("physical_ai_av/realgap-exam50"), splits.load("physical_ai_av/realgap-nonexam50")
        run.use_split(ex), run.use_split(ne)
        assert set(S[S.group == "test281"].scene_id) == set(ex) and set(S[S.group == "nonexam"].clip_id) == set(ne)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cores = {t: C.Core(t, dev) for t in (MODEL, YARD)}
        recs, starts, dumps = load_render()
        byscene = {}
        for f in dumps:
            with np.load(f) as z:
                sc, k = str(z["scene"]), int(f.stem.split("_k")[1])
                r = recs.get((sc, k))
                if r is None or int(z["now"]) != r["now"] or not (z["real"].all() and z["valid"].all()) or r.get("log_valid", 0) < 8:
                    continue
                byscene.setdefault(sc, []).append(dict(k=k, t0=int(z["t0"]), now=int(z["now"]), cur=z["cur"], ego=z["ego"].astype(np.float32),
                                                       cam_t=z["cam_t"], jpeg=z["jpeg"].tobytes(), rec=r))
        have = lambda c: (SAMPLE / "camera" / f"{c}.{CAM}.mp4").exists()  # noqa: E731
        if a.limit:                                         # smoke: the first scenes whose video is on disk
            byscene = {k: byscene[k] for k in [k for k in sorted(byscene) if have(k.removeprefix("clipgt-"))][:a.limit]}
        run.info(f"{len(dumps)} dumps, {sum(map(len, byscene.values()))} usable decisions in {len(byscene)} scenes")
        meta = S.set_index("scene_id")
        rows, tok, calrows, figs = [], {"render": [], "real": [], "realN": []}, [], {}
        dt_slot = np.round(C.SLOT_T * 1e6).astype(np.int64)

        def prep(sc):
            """Decode and convert one scene's real frames (the conversion of pai_core.Maps, two calibrations) -> packed slot frames."""
            m, st = meta.loc[sc], starts[sc]
            clk, cal, vid = usdz_clock(m.uuid), pav_calib(m.clip_id, int(m.chunk)), Video(m.clip_id)
            ds = byscene[sc]
            want = np.stack([d["t0"] + dt_slot for d in ds]) - clk["start_us"]          # (n, 8) clip clock
            j, err = vid.nearest(want.ravel())
            j, err = j.reshape(want.shape), err.reshape(want.shape)
            lagi = [i for i, d in enumerate(ds) if d["k"] % 50 == 0]
            jl = np.clip(j[lagi, -1][:, None] + np.arange(-3, 4), 0, len(vid.ts) - 1)
            ims = vid.decode(np.r_[j.ravel(), jl.ravel()])
            wh = ims[j[0, -1]].shape[1::-1]
            mapP, mapN = PC.Maps(cal["spec"], cal["R"], wh), PC.Maps(st["spec"], np.array(st["R"]), wh)
            pk = {jj: mapP(im) for jj, im in ims.items()}
            curP = np.stack([np.stack([pk[jj] for jj in j[i]]) for i in range(len(ds))])
            curN = np.stack([np.stack([mapN(ims[jj]) for jj in j[i]]) for i in range(len(ds))])
            lag = {i: [pk[jj][0] if jj in pk else None for jj in jl[n]] for n, i in enumerate(lagi)}
            thumbs = {i: ims[j[i, -1]][::4, ::4, 0].copy() for i in lagi}
            return m, st, clk, cal, vid, ds, err, curP, curN, lag, thumbs, wh, (mapP.coverage, mapN.coverage)

        def batches(xs, n):
            for b in range(0, len(xs), n):
                yield from pool.map(prep, xs[b:b + n])

        with ThreadPoolExecutor(6) as pool:
            for m, st, clk, cal, vid, ds, err, CP, CN, lag, thumbs, wh, cov in run.tqdm(batches(sorted(byscene), 6), total=len(byscene), desc="exam"):
                ang = np.degrees(np.arccos(np.clip((np.trace(cal["R"].T @ np.array(st["R"])) - 1) / 2, -1, 1)))
                th = np.radians([20, 40, 60])
                rP = sum(c * th ** i for i, c in enumerate(cal["spec"]["fw"]))
                rN = sum(c * th ** i for i, c in enumerate(st["spec"]["fw"]))
                ci = vid.nearest(np.array([clk["cam0_us"] - clk["start_us"]]))[1][0]
                calrows.append(dict(scene=m.name, clip=m.clip_id, stratum=m.stratum, rot_deg=float(ang),
                                    dt_m=float(np.linalg.norm(cal["t"] - np.array(st["cam_t"]))), dc_px=float(np.hypot(*np.subtract(cal["spec"]["c"], st["spec"]["c"]))),
                                    dcx=cal["spec"]["c"][0] - st["spec"]["c"][0], dcy=cal["spec"]["c"][1] - st["spec"]["c"][1],
                                    dr20=float(rP[0] - rN[0]), dr40=float(rP[1] - rN[1]), dr60=float(rP[2] - rN[2]),
                                    native_pav=str(cal["spec"]["native"]), native_nurec=str(st["spec"]["native"]), real_wh=str(wh),
                                    cov_pav=cov[0], cov_nurec=cov[1], nurec_cam0_vs_pav_us=int(-ci), slot_dt_max_us=int(np.abs(err).max())))
                tg = ego_targets(m.clip_id, np.array([(d["t0"] - clk["start_us"]) * 1e-6 for d in ds]))
                for i, d in enumerate(ds):
                    curP, curN, r = CP[i], CN[i], d["rec"]
                    row = dict(scene=m.name, clip=m.clip_id, stratum=m.stratum, group="test281", k=d["k"], t_clip=(d["t0"] - clk["start_us"]) * 1e-6,
                               human=relative(np.array(r["log_now"]), np.array(r["log_poses"])).tolist(), served=np.array(r["poses_model"]).tolist(),
                               **{key: float(tg[key][i]) for key in tg})
                    for tag, core in cores.items():
                        rf = np.asarray(r["rmem"], np.float32) if core.rm and "rmem" in r else None
                        for side, cur in (("render", d["cur"]), ("real", curP), ("realN", curN)):
                            o = PC.plan(core, cur, np.ones(8, bool), np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.eye(4, dtype=np.float32)[r["cmd"]],
                                        d["cam_t"], False, route=rf, ego_fn=lambda _, e=d["ego"]: e)
                            row[f"{tag}|{side}"] = np.asarray(o["poses"], np.float64).tolist()
                            if tag == MODEL:
                                tok[side].append(o["tokens"][-1].float().cpu().numpy().astype(np.float16))
                    LR, LP, LN = (unpack_luma(c[-1, 0]) for c in (d["cur"], curP, curN))
                    row.update(zip(("sh_dx", "sh_dy", "sh_resp"), shift(LR, LP)))
                    row.update(zip(("shN_dx", "shN_dy", "shN_resp"), shift(LN, LP)))
                    row["mae_render_real"] = float(np.abs(LR.astype(int) - LP).mean())
                    row["mae_realN_real"] = float(np.abs(LN.astype(int) - LP).mean())
                    row["mae_realN_real_wide"] = float(np.abs(unpack_luma(curN[-1, 1]).astype(int) - unpack_luma(curP[-1, 1])).mean())
                    if i in lag:                             # lag scan: render road luma vs real at -3 .. +3 frames around the matched one
                        row["lag_mae"] = [float(np.abs(LR.astype(int) - unpack_luma(f)).mean()) if f is not None else np.nan for f in lag[i]]
                        figs[f"{m.name}|{d['k']}"] = dict(render_jpeg=d["jpeg"], real=thumbs[i], lumaR=LR, lumaP=LP, lumaN=LN,
                                                         wideR=unpack_luma(d["cur"][-1, 1]), wideP=unpack_luma(curP[-1, 1]))
                    rows.append(row)
            run.info(f"exam: {len(rows)} decisions")
            # non-exam: real tokens only (PAI-AV calibration), clip t0 = 2 .. 19 s
            ne_s = S[S.group == "nonexam"]
            if a.limit:
                ne_s = ne_s[ne_s.clip_id.map(have)].iloc[:a.limit]

            def prep_ne(m):
                vid, cal = Video(m.clip_id), pav_calib(m.clip_id, int(m.chunk))
                t0 = np.arange(2.0, 19.01, 1.0)
                t0 = t0[(t0 - 1.4 >= vid.ts[0] * 1e-6) & (t0 <= vid.ts[-1] * 1e-6)]
                j, err = vid.nearest(np.round((t0[:, None] + C.SLOT_T[None]) * 1e6).astype(np.int64).ravel())
                ims = vid.decode(j)
                mp = PC.Maps(cal["spec"], cal["R"], next(iter(ims.values())).shape[1::-1]) if ims else None
                return m, cal, t0, j.reshape(len(t0), 8), {jj: mp(im) for jj, im in ims.items()}

            n_ne = 0
            for m, cal, t0, j, pk in run.tqdm(pool.map(prep_ne, [r for _, r in ne_s.iterrows()]), total=len(ne_s), desc="nonexam"):
                if not pk:
                    run.info(f"{m.clip_id}: no frames decoded")
                    continue
                tg = ego_targets(m.clip_id, t0)
                for i in range(len(t0)):
                    if not all(jj in pk for jj in j[i]):
                        continue
                    cur = np.stack([pk[jj] for jj in j[i]])
                    o = PC.plan(cores[MODEL], cur, np.ones(8, bool), np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.eye(4, dtype=np.float32)[1],
                                cal["t"], False, route=None, ego_fn=lambda _: np.zeros(20, np.float32))
                    tok.setdefault("nonexam", []).append(o["tokens"][-1].float().cpu().numpy().astype(np.float16))
                    rows.append(dict(scene="", clip=m.clip_id, stratum=m.stratum, group="nonexam", k=-1, t_clip=float(t0[i]),
                                     **{key: float(tg[key][i]) for key in tg}))
                    n_ne += 1
        run.info(f"non-exam: {n_ne} decisions")
        df = pd.DataFrame(rows)
        df.to_pickle(out / "rows.pkl")
        pd.DataFrame(calrows).to_csv(out / "calib.csv", index=False)
        np.savez(out / "tokens.npz", **{k: np.stack(v) for k, v in tok.items() if v})
        with open(out / "figs.pkl", "wb") as fh:
            import pickle
            pickle.dump(figs, fh)
        run.summary.update(exam_decisions=int((df.group == "test281").sum()), nonexam_decisions=n_ne, scenes=len(byscene))


# ---------------------------------------------------------------- report
def plan_metrics(P: np.ndarray, H: np.ndarray) -> dict:
    """Plans P (n, 8, 3) and human futures H (n, 8, 3), poses at 0.5 .. 4 s, each in its own start frame."""
    arc = lambda X: np.cumsum(np.hypot(*np.diff(np.concatenate([np.zeros((len(X), 1, 2)), X[..., :2]], 1), axis=1).transpose(2, 0, 1)), -1)  # noqa: E731
    aP, aH = arc(P), arc(H)
    return dict(ade=np.linalg.norm(P[..., :2] - H[..., :2], axis=-1).mean(-1), fde=np.linalg.norm(P[:, -1, :2] - H[:, -1, :2], axis=-1),
                v2=aP[:, 3] / 2, v4=aP[:, 7] / 4, v2_h=aH[:, 3] / 2, v4_h=aH[:, 7] / 4, lat4=P[:, -1, 1] - H[:, -1, 1])


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA of two (n, d) representations of the same n samples."""
    X, Y = X - X.mean(0), Y - Y.mean(0)
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))


def cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cmd_report(a):
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from jevdrive import stats
    rp = RUN / "replay"
    df, cal = pd.read_pickle(rp / "rows.pkl"), pd.read_csv(rp / "calib.csv")
    T = np.load(rp / "tokens.npz")
    E = df[df.group == "test281"].reset_index(drop=True)
    N = df[df.group == "nonexam"].reset_index(drop=True)
    g = E.scene.to_numpy()
    night = E.stratum.str.endswith("night").to_numpy()
    out, md = {}, ["# REAL-GAP: real PAI-AV footage vs the NuRec render, same clips and times", "",
                   f"{E.scene.nunique()} test281 clips, {len(E)} decisions (log replay, 1 Hz dumps with 8 rendered slots, k >= 20); "
                   f"{N.clip.nunique()} non-exam clips, {len(N)} real decisions. CIs: scene-cluster bootstrap (jevdrive.stats). "
                   "Measurement only: no test281 real frame is in any training row or cache.", ""]
    q = lambda x: f"{np.nanmedian(x):.3g} [{np.nanquantile(x, 0.1):.3g}, {np.nanquantile(x, 0.9):.3g}]"  # noqa: E731
    fmt = lambda r, f=".3f": f"{stats.fmt(r, f)}"  # noqa: E731

    # A. conversion
    lag = np.array([r for r in E.lag_mae.dropna()])
    best = np.nanargmin(lag, 1) - 3 if len(lag) else np.array([], int)
    deg = lambda px: np.degrees(np.asarray(px) / 910.0)  # noqa: E731   road frame: focal 910 px on 512 x 256
    out["conversion"] = dict(rot_deg=cal.rot_deg.describe().to_dict(), dt_m=cal.dt_m.describe().to_dict(), dc_px=cal.dc_px.describe().to_dict(),
                             dr60=cal.dr60.abs().describe().to_dict(), clock_us=cal.nurec_cam0_vs_pav_us.describe().to_dict(),
                             slot_dt_max_us=int(cal.slot_dt_max_us.max()), lag_best=np.bincount(best + 3, minlength=7).tolist(),
                             shift_dx_deg=q(deg(E.sh_dx)), shift_dy_deg=q(deg(E.sh_dy)), shiftN_dx_deg=q(deg(E.shN_dx)), shiftN_dy_deg=q(deg(E.shN_dy)),
                             mae_render_real=q(E.mae_render_real), mae_calib=q(E.mae_realN_real), mae_calib_wide=q(E.mae_realN_real_wide),
                             native=sorted(set(cal.native_pav) | set(cal.native_nurec)), real_wh=sorted(set(cal.real_wh)))
    md += ["## A. Conversion check (the driver's f-theta -> openpilot path on real frames)", "",
           "| Check | median [p10, p90] or range |", "|---|---|",
           f"| camera rotation, PAI-AV extrinsics vs the session's NuRec rig_to_camera (deg) | {q(cal.rot_deg)} (max {cal.rot_deg.max():.3f}) |",
           f"| camera position difference (m) | {q(cal.dt_m)} |",
           f"| principal point difference (px, native) | {q(cal.dc_px)}; dcy {q(cal.dcy)} |",
           f"| f-theta radius difference at 20 / 40 / 60 deg (px) | {q(cal.dr20)} / {q(cal.dr40)} / {q(cal.dr60)} |",
           f"| native size PAI-AV / NuRec / decoded video | {sorted(set(cal.native_pav))} / {sorted(set(cal.native_nurec))} / {sorted(set(cal.real_wh))} |",
           f"| NuRec first front-wide frame minus nearest PAI-AV frame time (us) | {q(cal.nurec_cam0_vs_pav_us)} |",
           f"| real frame vs slot time, worst (us) | {int(cal.slot_dt_max_us.max())} |",
           f"| image lag scan: best real frame offset (frames -3 .. +3) | {np.bincount(best + 3, minlength=7).tolist()} |",
           f"| road-frame shift render -> real (deg, x / y) | {q(deg(E.sh_dx))} / {q(deg(E.sh_dy))} |",
           f"| road-frame shift real via NuRec spec -> via PAI-AV calib (deg, x / y) | {q(deg(E.shN_dx))} / {q(deg(E.shN_dy))} |",
           f"| road luma MAE render vs real / real NuRec-spec vs PAI-AV-calib (0-255) | {q(E.mae_render_real)} / {q(E.mae_realN_real)} |", ""]

    # B. plans
    H = np.stack(E.human.map(np.array))
    md += ["## B. Open-loop plans on render vs real (same ego row, route memory and command; only the frames differ)", ""]
    res = {}
    for tag in (MODEL, YARD):
        for side in ("render", "real", "realN"):
            res[(tag, side)] = plan_metrics(np.stack(E[f"{tag}|{side}"].map(np.array)), H)
    served = np.stack(E.served.map(np.array))
    eq = np.abs(np.stack(E[f"{MODEL}|render"].map(np.array))[..., :2] - served[..., :2]).max((1, 2))
    out["served_equivalence_m"] = dict(median=float(np.median(eq)), p99=float(np.quantile(eq, 0.99)), max=float(eq.max()))
    md += [f"Replay check: {MODEL} on the dumped render slots vs the plan the driver logged: median {np.median(eq):.4f} m, p99 "
           f"{np.quantile(eq, 0.99):.4f} m, max {eq.max():.4f} m.", "",
           "| Model | Input | ADE (m) | FDE 4 s (m) | v4 plan - human (m/s) | abs v4 err (m/s) | lat 4 s (m) |", "|---|---|---|---|---|---|---|"]
    for (tag, side), m in res.items():
        md.append(f"| {tag} | {side} | {fmt(stats.bootstrap(m['ade'], g))} | {fmt(stats.bootstrap(m['fde'], g))} | "
                  f"{fmt(stats.bootstrap(m['v4'] - m['v4_h'], g))} | {fmt(stats.bootstrap(np.abs(m['v4'] - m['v4_h']), g))} | {fmt(stats.bootstrap(m['lat4'], g))} |")
    P = {k: np.stack(E[f"{k[0]}|{k[1]}"].map(np.array)) for k in res}
    dd = lambda a_, b_: np.linalg.norm(P[a_][..., :2] - P[b_][..., :2], axis=-1).mean(-1)  # noqa: E731
    pr = {"real - render ADE": stats.paired(res[(MODEL, "real")]["ade"], res[(MODEL, "render")]["ade"], g),
          "real - render v4 bias": stats.paired(res[(MODEL, "real")]["v4"], res[(MODEL, "render")]["v4"], g),
          "real - render abs v4 err": stats.paired(np.abs(res[(MODEL, "real")]["v4"] - res[(MODEL, "real")]["v4_h"]),
                                                   np.abs(res[(MODEL, "render")]["v4"] - res[(MODEL, "render")]["v4_h"]), g),
          "plan distance real vs render": stats.bootstrap(dd((MODEL, "real"), (MODEL, "render")), g),
          "plan distance NTV0 vs RSG on render (yardstick)": stats.bootstrap(dd((MODEL, "render"), (YARD, "render")), g),
          "plan distance real PAI-AV calib vs NuRec spec": stats.bootstrap(dd((MODEL, "real"), (MODEL, "realN")), g),
          "plan distance real vs render, day": stats.bootstrap(dd((MODEL, "real"), (MODEL, "render"))[~night], g[~night]),
          "plan distance real vs render, night": stats.bootstrap(dd((MODEL, "real"), (MODEL, "render"))[night], g[night]),
          "real - render ADE, day": stats.paired(res[(MODEL, "real")]["ade"][~night], res[(MODEL, "render")]["ade"][~night], g[~night]),
          "real - render ADE, night": stats.paired(res[(MODEL, "real")]["ade"][night], res[(MODEL, "render")]["ade"][night], g[night])}
    md += ["", f"Paired ({MODEL}):", "", "| Quantity | mean [95 % CI] |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in pr.items()]
    out["plans"] = {f"{t}|{s_}": {k: float(np.nanmean(v)) for k, v in m.items()} for (t, s_), m in res.items()}
    out["paired"] = pr

    # C. tokens
    Rn, Re, RN = (T[k].astype(np.float32).reshape(len(E), -1) for k in ("render", "real", "realN"))
    nxt = {(sc, k): i for i, (sc, k) in enumerate(zip(E.scene, E.k))}
    pairs = np.array([(i, nxt[(sc, k + 10)]) for i, (sc, k) in enumerate(zip(E.scene, E.k)) if (sc, k + 10) in nxt])
    rng = np.random.default_rng(0)
    other = np.array([rng.choice(np.flatnonzero(g != g[i])) for i in range(len(E))])
    cs = {"render vs real (same clip, time)": cos_rows(Rn, Re), "real NuRec-spec vs real PAI-AV-calib": cos_rows(RN, Re),
          "real t vs real t + 1 s (yardstick)": cos_rows(Re[pairs[:, 0]], Re[pairs[:, 1]]),
          "render t vs render t + 1 s": cos_rows(Rn[pairs[:, 0]], Rn[pairs[:, 1]]),
          "real vs real of another clip (null)": cos_rows(Re, Re[other])}
    mp = lambda X: X.reshape(len(X), 32, 512).mean(1)  # noqa: E731
    ck = {"render vs real (paired)": cka(mp(Rn), mp(Re)), "render vs real (pairing permuted, null)": cka(mp(Rn), mp(Re)[rng.permutation(len(E))]),
          "render vs real, flattened tokens": cka(Rn, Re), "real NuRec-spec vs PAI-AV-calib": cka(mp(RN), mp(Re))}
    X = np.r_[Rn, Re]
    yd, gd = np.r_[np.zeros(len(E)), np.ones(len(E))], np.r_[g, g]
    acc = []
    for tr, te in GroupKFold(5).split(X, yd, gd):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[tr], yd[tr])
        acc.append(clf.score(X[te], yd[te]))
    md += ["", "## C. Frozen Cinque tokens (t0 slot, 32 x 512)", "", "| Cosine (flattened) | median [p10, p90] |", "|---|---|"]
    md += [f"| {k} | {q(v)} |" for k, v in cs.items()]
    md += ["", "| Linear CKA (mean-pooled 512 unless said) | value |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in ck.items()]
    md += ["", f"Domain classifier (logistic, scene-grouped 5-fold) render vs real: accuracy {np.mean(acc):.3f} (0.5 = indistinguishable)."]
    out["tokens"] = dict(cosine={k: q(v) for k, v in cs.items()}, cka=ck, domain_acc=float(np.mean(acc)))

    # D. probes
    md += ["", "## D. Probe transfer (ridge on the flattened t0 tokens; targets from the PAI-AV egomotion)", "",
           "| Target | Train | Test real R2 | Test render R2 | render - real squared error [95 % CI] |", "|---|---|---|---|---|"]
    out["probes"] = {}
    NT = T["nonexam"].astype(np.float32).reshape(len(N), -1)
    for tgt in ("v0", "dv3", "dyaw3"):
        ok_n, ok_e = np.isfinite(N[tgt].to_numpy()), np.isfinite(E[tgt].to_numpy())
        y, ye = N[tgt].to_numpy()[ok_n], E[tgt].to_numpy()[ok_e]
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13))).fit(NT[ok_n], y)
        pR, pr_ = mdl.predict(Re[ok_e]), mdl.predict(Rn[ok_e])
        d = stats.paired((pr_ - ye) ** 2, (pR - ye) ** 2, g[ok_e])
        out["probes"][f"{tgt}|nonexam-real"] = dict(real=r2(ye, pR), render=r2(ye, pr_), dse=d)
        md.append(f"| {tgt} | real non-exam ({ok_n.sum()}) | {r2(ye, pR):.3f} | {r2(ye, pr_):.3f} | {fmt(d)} |")
        for src, Xs, Xo in (("render", Rn, Re), ("real", Re, Rn)):
            ps, po = np.full(len(E), np.nan), np.full(len(E), np.nan)
            idx = np.flatnonzero(ok_e)
            for tr, te in GroupKFold(5).split(idx, groups=g[idx]):
                m_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13))).fit(Xs[idx[tr]], E[tgt].to_numpy()[idx[tr]])
                ps[idx[te]], po[idx[te]] = m_.predict(Xs[idx[te]]), m_.predict(Xo[idx[te]])
            yy = E[tgt].to_numpy()
            r_real, r_rend = (r2(ye, ps[ok_e]), r2(ye, po[ok_e])) if src == "real" else (r2(ye, po[ok_e]), r2(ye, ps[ok_e]))
            se_rend = (ps - yy) ** 2 if src == "render" else (po - yy) ** 2
            se_real = (po - yy) ** 2 if src == "render" else (ps - yy) ** 2
            d = stats.paired(se_rend[ok_e], se_real[ok_e], g[ok_e])
            out["probes"][f"{tgt}|exam-{src}-cv"] = dict(real=r_real, render=r_rend, dse=d)
            md.append(f"| {tgt} | exam {src}, scene 5-fold | {r_real:.3f} | {r_rend:.3f} | {fmt(d)} |")
    (TOPIC / "results").mkdir(exist_ok=True)
    (TOPIC / "results/realgap.md").write_text("\n".join(md) + "\n")
    (TOPIC / "results/realgap.json").write_text(json.dumps(out, indent=1, default=float))
    cal.to_csv(TOPIC / "results/realgap_calib.csv", index=False)
    print("\n".join(md))


def cmd_fig(a):
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    F = pickle.load(open(RUN / "replay/figs.pkl", "rb"))
    S = pd.read_csv(TOPIC / "results/real_sample.csv").set_index("scene_id")
    keys = sorted(F)
    pick, seen = [], set()
    for want in ("US/day", "DE/day", "other/day", "US/night", "DE/night", "other/night"):
        for k in keys:
            sc = k.split("|")[0]
            if S.at[sc, "stratum"] == want and sc not in seen and int(k.split("|")[1]) >= 100:
                pick.append(k), seen.add(sc)
                break
    cols = ("render (source JPEG)", "real (PAI-AV video)", "render: openpilot road frame", "real: road frame, PAI-AV calibration",
            "real: road frame, NuRec CameraSpec", "|render - real| road frame")
    fig, ax = plt.subplots(len(pick), len(cols), figsize=(3.2 * len(cols), 1.9 * len(pick)))
    for r, k in enumerate(pick):
        f = F[k]
        src = np.asarray(Image.open(io.BytesIO(f["render_jpeg"])).convert("L"))[::4, ::4] if len(f["render_jpeg"]) else np.zeros_like(f["real"])
        ims = (src, f["real"], f["lumaR"], f["lumaP"], f["lumaN"], np.abs(f["lumaR"].astype(int) - f["lumaP"]))
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, cmap="gray" if c < 5 else "magma", vmin=0, vmax=255 if c < 5 else 120)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=8)
        sc, kk = k.split("|")
        ax[r, 0].set_ylabel(f"{S.at[sc, 'stratum']}\n{sc[7:15]} k={kk}", fontsize=7)
    fig.tight_layout()
    p = TOPIC / "figs/realgap_frames.png"
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=110)
    print(p)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pick")
    p.add_argument("--n", type=int, default=50)
    p = sp.add_parser("fetch")
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("replay")
    p.add_argument("--limit", type=int, default=0), p.add_argument("--out", default="replay")
    for k in ("report", "fig"):
        sp.add_parser(k)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
