"""Waymo Open Dataset E2E (WOD-E2E) data path: frame index, image history, targets, ego baselines,
frozen features and the challenge submission writer (docs/waymo-e2e.md).

Input is the slim front-3-camera shards written by scripts/download_waymo_e2e.sh
($DATA_DIR/datasets/waymo_e2e/front3/*.tfrecord-*), outputs go to $DATA_DIR/processed/waymo_e2e/.

  index      scan new shards -> shards/<shard>.parquet, then merge to index.parquet + past.npy + future.npy
  report     sequence / gap / history-completeness stats and the ego-only baseline table
  check      assertions that run on whatever shards are present
  features   frozen Qwen3-VL features per frame, same layout as processed/nuscenes/<v>/features/

A record is one E2EDFrame; shards hold shuffled frames, not video. `frame.context.name` is
"<sequence>-<frame>" with the frame index at the source ~10 Hz, so index step 1 = 0.1 s.
past_states are 16 steps at 4 Hz over (-4 s, 0], future_states 20 steps over (0, 5 s], both in the
current rear-axle ego frame (+x forward, +y left) -- which is exactly the submission convention,
so future_states.pos_x/pos_y is the regression target with no transform.
"""
import importlib
import io
import json
import os
import struct
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger, n_cpus

log = get_logger(__name__)

DT = 0.25                      # ego state and trajectory step, s (4 Hz)
N_PAST, N_FUTURE = 16, 20      # past (-4 s, 0], future (0, 5 s]
FRAME_DT = 0.1                 # one context.name index step, s (source ~10 Hz)
INTENTS = ("UNKNOWN", "GO_STRAIGHT", "GO_LEFT", "GO_RIGHT")
CAMS = ("front", "front_left", "front_right")   # CameraName 1, 2, 3 -- the three kept by the slim step
CAM_IDS = dict(zip((1, 2, 3), CAMS))
PAST_FIELDS = ("pos_x", "pos_y", "vel_x", "vel_y", "accel_x", "accel_y")
SPAN_COLS = [f"{c}_{s}" for c in CAMS for s in ("off", "len")]
SCALAR_COLS = (["sequence", "frame", "split", "shard", "rec_off", "rec_len"] + SPAN_COLS
               + ["intent", "has_future", "n_pref"])


# ---------------------------------------------------------------- paths and protos

def shard_dir() -> Path:
    return data_dir() / "datasets" / "waymo_e2e" / "front3"


def out_dir(*parts: str) -> Path:
    d = data_dir() / "processed" / "waymo_e2e" / Path(*parts) if parts else data_dir() / "processed" / "waymo_e2e"
    d.mkdir(parents=True, exist_ok=True)
    return d


def proto(module: str):
    """A compiled WOD proto module: "end_to_end_driving_data_pb2" for frames, "..._submission_pb2" for a
    submission. scripts/waymo_prepare.sh builds them into $DATA_DIR/envs/waymo/gen."""
    gen = os.environ.get("WAYMO_PROTO_GEN") or str(data_dir() / "envs" / "waymo" / "gen")
    if gen not in sys.path:
        sys.path.insert(0, gen)
    return importlib.import_module(f"waymo_open_dataset.protos.{module}")


def e2ed_frame():
    return proto("end_to_end_driving_data_pb2").E2EDFrame


def split_of(name: str) -> str:
    return {"val_": "val", "trai": "train", "test": "test"}[name[:4]]


def _records(f):
    """(payload offset, payload) per TFRecord. Both CRCs were checked when the shard was slimmed, and a
    wrong length or a corrupt payload fails the proto parse right after, so we do not pay for them again."""
    while head := f.read(12):
        (n,) = struct.unpack("<Q", head[:8])
        off = f.tell()
        rec = f.read(n)
        f.seek(4, os.SEEK_CUR)
        if len(rec) != n:
            raise ValueError(f"truncated record at offset {off}")
        yield off, rec


# ---------------------------------------------------------------- index

def scan_shard(path: str, cache: str) -> dict:
    """One shard -> one parquet: a row per frame with the byte spans of the record and of its three JPEGs,
    plus the flattened past and future states. Runs in its own process (one shard per core)."""
    E2EDFrame = e2ed_frame()
    name, t0 = Path(path).name, time.perf_counter()
    rows, ego, nbytes = [], [], 0
    with open(path, "rb", buffering=1 << 20) as f:
        for off, rec in _records(f):
            fr = E2EDFrame.FromString(rec)
            mv, nbytes = memoryview(rec), nbytes + len(rec)
            spans, hint = dict.fromkeys(SPAN_COLS, 0), 0
            for im in fr.frame.images:
                jpeg, n = im.image, len(im.image)
                o = rec.find(jpeg[:256], hint)          # images are stored in field order: start where the last ended
                while o >= 0 and mv[o:o + n] != jpeg:
                    o = rec.find(jpeg[:256], o + 1)
                if o < 0:
                    raise ValueError(f"{name}: camera {im.name} JPEG not found in its own record")
                c = CAM_IDS[im.name]
                spans[f"{c}_off"], spans[f"{c}_len"], hint = off + o, n, o + n
            seq, _, frame = fr.frame.context.name.rpartition("-")
            ps, fs = fr.past_states, fr.future_states
            past = np.zeros((N_PAST, len(PAST_FIELDS)), np.float32)
            got = np.array([getattr(ps, k) for k in PAST_FIELDS], np.float32).T
            past[:len(got)] = got[:N_PAST]
            fut = np.zeros((N_FUTURE, 3), np.float32)
            gotf = np.array([fs.pos_x, fs.pos_y, fs.pos_z], np.float32).T.reshape(-1, 3)
            fut[:len(gotf)] = gotf[:N_FUTURE]
            rows.append((seq, int(frame), split_of(name), name, off, len(rec), *spans.values(),
                         fr.intent, len(gotf) == N_FUTURE,
                         sum(t.preference_score >= 0 for t in fr.preference_trajectories)))
            ego.append(np.concatenate([past.ravel(), fut.ravel()]))
    df = pd.DataFrame(rows, columns=SCALAR_COLS)
    df["ego"] = ego
    tmp = Path(cache).with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(cache)
    return {"shard": name, "frames": len(df), "bytes": nbytes, "seconds": round(time.perf_counter() - t0, 2)}


def build_index(workers: int | None = None, force: bool = False) -> pd.DataFrame:
    """Scan every slim shard that has no up-to-date cache, then merge all caches into the index.
    Incremental: re-running after new shards land only reads the new ones."""
    cache_dir, root = out_dir("shards"), out_dir()
    shards = sorted(p for p in shard_dir().glob("*.tfrecord-*") if p.is_file())
    if not shards:
        raise SystemExit(f"no slim shards in {shard_dir()}")
    todo = [p for p in shards if force or not (cache_dir / f"{p.name}.parquet").exists()]
    log.info("%d slim shards, %d already indexed, %d to scan", len(shards), len(shards) - len(todo), len(todo))
    if todo:
        w = min(workers or max(1, n_cpus() // 2), len(todo))
        t0, done = time.perf_counter(), []
        with ProcessPoolExecutor(w) as ex:
            futs = [ex.submit(scan_shard, str(p), str(cache_dir / f"{p.name}.parquet")) for p in todo]
            for i, f in enumerate(futs):
                done.append(r := f.result())
                log.info("[%d/%d] %s: %d frames, %.2f GB in %.1f s", i + 1, len(todo), r["shard"], r["frames"],
                         r["bytes"] / 1e9, r["seconds"])
        dt, gb = time.perf_counter() - t0, sum(r["bytes"] for r in done) / 1e9
        log.info("scanned %d shards, %d frames, %.1f GB in %.1f s with %d workers (%.2f GB/s, %.0f frames/s)",
                 len(done), sum(r["frames"] for r in done), gb, dt, w, gb / dt, sum(r["frames"] for r in done) / dt)

    parts = [pd.read_parquet(cache_dir / f"{p.name}.parquet") for p in shards]
    df = pd.concat(parts, ignore_index=True)
    ego = np.stack(df.pop("ego").to_numpy()).astype(np.float32)
    df = df.sort_values(["split", "sequence", "frame"], kind="stable", ignore_index=False)
    ego, df = ego[df.index.to_numpy()], df.reset_index(drop=True)
    cluster = shard_dir() / "val_sequence_name_to_scenario_cluster.json"
    if cluster.exists():
        c = {k: v["scenario_cluster"] for k, v in json.loads(cluster.read_bytes()).items()}
        df["cluster"] = df.sequence.map(c).fillna("")
    for c in ("split", "shard", "cluster"):
        if c in df:
            df[c] = df[c].astype("category")
    df.to_parquet(root / "index.parquet", index=False)
    np.save(root / "past.npy", ego[:, :N_PAST * len(PAST_FIELDS)].reshape(-1, N_PAST, len(PAST_FIELDS)))
    np.save(root / "future.npy", ego[:, N_PAST * len(PAST_FIELDS):].reshape(-1, N_FUTURE, 3))
    log.info("index: %d frames, %d sequences, splits %s -> %s", len(df), df.sequence.nunique(),
             df.split.value_counts().to_dict(), root)
    return df


def load_index() -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "index.parquet")


def load_ego() -> tuple[np.ndarray, np.ndarray]:
    """(past (n, 16, 6), future (n, 20, 3)), row-aligned with the index."""
    return np.load(out_dir() / "past.npy"), np.load(out_dir() / "future.npy")


# ---------------------------------------------------------------- sequences and image history

def sequence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Frames per sequence and index gaps, per split. `gap` is the step between consecutive indexed frames of
    one sequence: 1 means the source 10 Hz is complete, more means shards holding those frames are missing."""
    rows = []
    for split, g in df.groupby("split", observed=True):
        n = g.groupby("sequence", observed=True).frame.agg(["size", "min", "max"])
        gaps = g.sort_values(["sequence", "frame"]).groupby("sequence", observed=True).frame.diff().dropna()
        shards = set(g.shard.astype(str))
        rows.append({"split": split, "shards": len(shards), "of": int(next(iter(shards)).rsplit("-of-", 1)[1]),
                     "frames": len(g), "sequences": len(n),
                     "frames_per_seq_mean": round(n["size"].mean(), 1),
                     "frames_per_seq_p10": int(n["size"].quantile(.1)), "frames_per_seq_p90": int(n["size"].quantile(.9)),
                     "frame_min": int(n["min"].min()), "frame_max": int(n["max"].max()),
                     "gap_1": round((gaps == 1).mean(), 4) if len(gaps) else float("nan"),
                     "gap_median": float(gaps.median()) if len(gaps) else float("nan"),
                     "coverage": round(len(g) / (n["max"] - n["min"] + 1).sum(), 4)})
    # `coverage` is the share of the source 10 Hz grid that the downloaded shards hold. Frames are spread over
    # the split's shards at random, so it tracks shards/of and goes to ~1 once the whole split is on disk.
    return pd.DataFrame(rows)


def history_rows(df: pd.DataFrame, n_back: int, stride: int, tol: int | None = None,
                 targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions of an image-history window for each target frame.

    The window asks for the target frame plus `n_back` earlier frames of the same sequence, `stride` source
    indices (0.1 s each) apart. The shards are a shuffled subset, so a requested index may be absent. Policy:
      1. exact index if it is indexed;
      2. else the nearest indexed frame of the same sequence within +/- `tol` indices (default stride // 2),
         never at or after the target frame, older preferred on a tie;
      3. else carry the previous (newer) slot forward -- the usual "repeat the last frame" padding, which for
         slot 1 means repeating the target frame itself.
    Returns (rows (m, n_back+1) int64 into `df`, dt (m, n_back+1) float32 actual time offset in seconds,
    exact (m, n_back+1) bool: resolved by rule 1). Slot 0 is the target and is always exact.
    """
    tol = stride // 2 if tol is None else tol
    seq = df.sequence.astype("category").cat.codes.to_numpy(np.int64)
    frame = df.frame.to_numpy(np.int64)
    key = seq * (1 << 20) + frame
    order = np.argsort(key, kind="stable")
    skey, srow = key[order], order.astype(np.int64)
    tgt = np.arange(len(df)) if targets is None else np.asarray(targets, np.int64)

    rows = np.empty((len(tgt), n_back + 1), np.int64)
    got = np.zeros_like(rows, bool)
    dt = np.zeros(rows.shape, np.float32)
    rows[:, 0], got[:, 0] = tgt, True
    for k in range(1, n_back + 1):
        want = frame[tgt] - k * stride
        wkey = seq[tgt] * (1 << 20) + want
        j = np.searchsorted(skey, wkey)
        hit = (j < len(skey)) & (skey[np.minimum(j, len(skey) - 1)] == wkey)
        rows[:, k], got[:, k] = srow[np.minimum(j, len(skey) - 1)], hit
        dt[:, k] = -k * stride * FRAME_DT
        if tol:                                    # nearest indexed frame of the same sequence within tol
            miss = ~hit
            for cand in (j - 1, j):                # the two neighbours of the insertion point
                c = np.clip(cand, 0, len(skey) - 1)[miss]
                ok = (seq[srow[c]] == seq[tgt][miss]) & (frame[srow[c]] < frame[tgt][miss])
                d = frame[srow[c]] - want[miss]
                ok &= np.abs(d) <= tol
                better = ok & (~got[miss, k] | (np.abs(d) < np.abs(dt[miss, k] / FRAME_DT + k * stride)))
                idx = np.flatnonzero(miss)[better]
                rows[idx, k] = srow[c][better]
                dt[idx, k] = (frame[srow[c][better]] - frame[tgt][idx]) * FRAME_DT
                got[idx, k] = True
        fill = ~got[:, k]                          # rule 3: repeat the previous slot
        rows[fill, k], dt[fill, k] = rows[fill, k - 1], dt[fill, k - 1]
    return rows, dt, got


def history_report(df: pd.DataFrame, windows=((1, 1), (1, 5), (3, 5), (3, 10), (3, 20), (7, 5))) -> pd.DataFrame:
    """Share of target frames whose window is exactly complete, and complete once the tolerance rule is used."""
    rows = []
    for n_back, stride in windows:
        _, _, got = history_rows(df, n_back, stride, tol=0)
        _, _, got_tol = history_rows(df, n_back, stride)
        rows.append({"n_back": n_back, "stride": stride, "span_s": round(n_back * stride * FRAME_DT, 2),
                     "exact": round(got.all(1).mean(), 4), "within_tol": round(got_tol.all(1).mean(), 4),
                     "slots_exact": round(got[:, 1:].mean(), 4), "slots_within_tol": round(got_tol[:, 1:].mean(), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- targets, inputs, ego-only baselines

def future_xy(future: np.ndarray) -> np.ndarray:
    """(n, 20, 2) submission-convention waypoints: current rear-axle ego frame, +x forward, +y left,
    first point at t + 0.25 s. future_states is already in that frame, so this only drops z."""
    return future[..., :2]


def ego_state(past: np.ndarray) -> np.ndarray:
    """(n, 96) flat past-state input: 16 steps x (pos x/y, vel x/y, accel x/y), oldest first."""
    return past.reshape(len(past), -1)


def intent_onehot(df: pd.DataFrame) -> np.ndarray:
    return np.eye(len(INTENTS), dtype=np.float32)[df.intent.to_numpy()]


def past_kinematics(past: np.ndarray, k: int = 4) -> dict[str, np.ndarray]:
    """Speed, longitudinal acceleration and yaw rate at t=0 from the last k+1 past positions.

    Positions are the rear axle in the current ego frame, so the heading at t=0 is +x by construction and only
    the yaw rate has to be estimated: fit a line to the interval headings and take its slope. The provided
    vel_*/accel_* are also returned (`v_vec`); their last entry repeats the previous one, and during turns their
    direction disagrees with the position differences by a few degrees, so both variants are worth reporting.
    """
    d = np.diff(past[:, -(k + 1):, :2], axis=1) / DT            # (n, k, 2) mean velocity per 0.25 s interval
    v = np.linalg.norm(d, axis=-1)
    th = np.unwrap(np.arctan2(d[..., 1], d[..., 0]), axis=1)
    t = (np.arange(k) - k + 0.5) * DT                           # interval midpoints, last one at -DT/2
    w = ((t - t.mean()) * (th - th.mean(1, keepdims=True))).sum(1) / ((t - t.mean()) ** 2).sum()
    a = np.gradient(v, DT, axis=1)[:, -1] if k > 1 else np.zeros(len(v))
    return {"v": v[:, -1] + a * DT / 2, "a": a, "w": w, "v_vec": past[:, -1, 2:4], "a_vec": past[:, -1, 4:6]}


def _arc(v: np.ndarray, a: np.ndarray, w: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Positions on a constant-turn-rate arc from the origin heading +x, speed v + a t. For |w| -> 0 this is
    the straight constant-acceleration line; the arc form ignores a (closed form would need Fresnel integrals)."""
    s = v[:, None] * t + 0.5 * a[:, None] * t ** 2
    wt = w[:, None] * t
    small = np.abs(w)[:, None] * t[-1] < 1e-3
    r = np.where(small, 1.0, v[:, None] / np.where(np.abs(w)[:, None] < 1e-9, 1e-9, w[:, None]))
    x = np.where(small, s, r * np.sin(wt))
    y = np.where(small, 0.0, r * (1 - np.cos(wt)))
    return np.stack([x, y], -1)


def baselines(past: np.ndarray, k: int = 4) -> dict[str, np.ndarray]:
    """Ego-only trajectory baselines, each (n, 20, 2) in the submission frame."""
    t = np.arange(1, N_FUTURE + 1) * DT
    z, kin = np.zeros(len(past)), past_kinematics(past, k)
    return {"zero": np.zeros((len(past), N_FUTURE, 2), np.float32),
            "cv_vel": kin["v_vec"][:, None, :] * t[None, :, None],
            "cv": _arc(kin["v"], z, z, t),
            "ca": _arc(kin["v"], kin["a"], z, t),
            "ctrv": _arc(kin["v"], z, kin["w"], t),
            "ctra": _arc(kin["v"], kin["a"], kin["w"], t)}


def ade_fde(pred: np.ndarray, gt: np.ndarray, horizons=(3.0, 5.0)) -> dict[str, float]:
    d = np.linalg.norm(pred - gt, axis=-1)
    out = {}
    for h in horizons:
        n = int(round(h / DT))
        out[f"ade{h:g}"], out[f"fde{h:g}"] = float(d[:, :n].mean()), float(d[:, n - 1].mean())
    return out


def baseline_table(df: pd.DataFrame, past: np.ndarray, future: np.ndarray, split: str = "val") -> pd.DataFrame:
    """ADE/FDE of every ego-only baseline on the frames of `split` that have a future, overall and per intent."""
    m = (df.split == split).to_numpy() & df.has_future.to_numpy()
    gt, preds = future_xy(future[m]), baselines(past[m])
    intent = df.intent.to_numpy()[m]
    rows = []
    for name, p in preds.items():
        r = {"baseline": name, "n": int(m.sum()), **ade_fde(p, gt)}
        for i, label in enumerate(INTENTS):
            if (s := intent == i).any():
                r[f"ade5_{label.removeprefix('GO_').lower()}"] = round(ade_fde(p[s], gt[s])["ade5"], 3)
        rows.append(r)
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- frozen features (reuses jevdrive.features)

class Shards:
    """Dataset over slim shards: pread the JPEG byte spans the index recorded, decode, hand to fx.transform.
    No protobuf and no full-record read in the DataLoader workers."""

    def __init__(self, items, transform):
        from PIL import Image
        self.items, self.transform, self.fds, self._open = items, transform, {}, Image.open

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, spans = self.items[i]
        fd = self.fds.get(path)
        if fd is None:
            fd = self.fds[path] = os.open(path, os.O_RDONLY)
        return self.transform([self._open(io.BytesIO(os.pread(fd, n, o))).convert("RGB") for o, n in spans])


def feature_items(df: pd.DataFrame, cams=CAMS, separate: bool = False) -> tuple[list, pd.DataFrame]:
    """(items for `Shards`, row index). One item per frame, or -- with `separate` -- one per (frame, camera),
    so that each camera gets its own forward pass and its own feature row."""
    shard = (shard_dir().as_posix() + "/" + df.shard.astype(str)).to_numpy()
    spans = {c: df[[f"{c}_off", f"{c}_len"]].to_numpy() for c in cams}
    if separate:
        items = [(shard[i], [tuple(spans[c][i])]) for i in range(len(df)) for c in cams]
        idx = pd.DataFrame({"row": np.repeat(np.arange(len(df)), len(cams)), "cam": list(cams) * len(df)})
    else:
        items = [(shard[i], [tuple(spans[c][i]) for c in cams]) for i in range(len(df))]
        idx = pd.DataFrame({"row": np.arange(len(df)), "cam": ",".join(cams)})
    name = (df.sequence.astype(str) + "-" + df.frame.map("{:03d}".format)).to_numpy()
    idx.insert(0, "frame_name", name[idx.row])
    return items, idx


def set_name(cams, separate: bool, long_side: int | None) -> str:
    return "qwen_" + ("front" if tuple(cams) == ("front",) else "front3") + ("sep" if separate else "") \
        + (f"_l{long_side}" if long_side else "")


def extract_features(cams=CAMS, separate: bool = False, long_side: int | None = None, batch_size: int = 4,
                     workers: int | None = None, limit: int | None = None, splits=("val",), save: bool = True,
                     n_layer_probes: int = 4, rl=None, compile: bool = True) -> dict:
    """Frozen Qwen3-VL features for every indexed frame, with jevdrive.features doing the model work.
    Layout matches processed/nuscenes/<version>/features/<set>/: <name>.npy float16 + index.parquet + meta.json."""
    from . import features as F
    df = load_index()
    df = df[df.split.isin(splits)].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit]
    items, idx = feature_items(df, cams, separate)
    name = set_name(cams, separate, long_side)
    dst = out_dir("features", name) if save else None
    t0 = time.perf_counter()
    fx = F.QwenFeatures(n_layer_probes=n_layer_probes, long_side=long_side, n_images=1 if separate else len(cams),
                        compile=compile)
    load_s = time.perf_counter() - t0
    stats = F.extract(fx, items, batch_size, max(1, n_cpus() // 2) if workers is None else workers, dst, rl,
                      f"waymo/{name}", dataset=Shards)
    tokens = int(fx.n_image_tokens)
    meta = {"set": name, "model": F.QWEN, "cams": list(cams), "separate": separate, "long_side": long_side,
            "splits": list(splits), "frames": len(df), "rows": len(items), "tokens_per_forward": tokens,
            "tokens_per_frame": tokens * (len(cams) if separate else 1), "load_s": round(load_s, 1), **stats}
    if dst is not None:
        idx.to_parquet(dst / "index.parquet", index=False)
        meta["features"] = sorted(p.stem for p in dst.glob("*.npy"))
        meta["bytes_per_frame"] = meta["bytes_per_sample"] * (len(cams) if separate else 1)
        (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: %d rows, %.1f ms/frame, %d tokens/frame, peak VRAM %.2f GB, %.1f KB/frame", name, stats["n"],
             stats["ms_per_frame"] * (len(cams) if separate else 1), meta["tokens_per_frame"], stats["peak_vram_gb"],
             stats["bytes_per_sample"] * (len(cams) if separate else 1) / 1024)
    del fx
    F.free_gpu()
    return meta


# ---------------------------------------------------------------- challenge submission

SUBMISSION_META = {"account_name": "", "unique_method_name": "", "authors": [], "affiliation": "",
                   "description": "", "method_link": "", "uses_public_model_pretraining": True,
                   "public_model_names": [], "num_model_parameters": ""}


def submission_frames() -> list[str]:
    """The frame names the official manifest asks for (one per test clip, at the 12 s mark)."""
    j = json.loads((shard_dir() / "test_sequence_frames_for_submission.json").read_bytes())
    return [f"{seq}-{int(k):03d}" for seq, k in j.items()]


def write_submission(frame_names, trajectories: np.ndarray, path, meta: dict, per_file: int = 10000) -> Path:
    """Write an E2EDChallengeSubmission tar.gz: 20 XY waypoints per frame, current rear-axle ego frame,
    +x forward, +y left, first at t+0.25 s. `meta` fills the required identity/pretraining fields."""
    pb = proto("end_to_end_driving_submission_pb2")
    names, traj, path = list(frame_names), np.asarray(trajectories, np.float32), Path(path)
    if traj.shape != (len(names), N_FUTURE, 2):
        raise ValueError(f"trajectories must be ({len(names)}, {N_FUTURE}, 2), got {traj.shape}")
    if not np.isfinite(traj).all():
        raise ValueError("trajectories contain non-finite values")
    if len(set(names)) != len(names):
        raise ValueError("duplicate frame names")
    missing = set(submission_frames()) - set(names)
    if missing:
        log.warning("submission is missing %d of the %d frames the manifest asks for", len(missing),
                    len(submission_frames()))
    fields = {k: v for k, v in {**SUBMISSION_META, **meta}.items()}
    if unknown := set(fields) - set(SUBMISSION_META):
        raise ValueError(f"unknown submission fields: {sorted(unknown)}")
    for k in ("account_name", "unique_method_name", "num_model_parameters"):
        if not fields[k]:
            raise ValueError(f"{k} is required for a valid submission")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for start in range(0, len(names), per_file):
            sub = pb.E2EDChallengeSubmission(
                submission_type=pb.E2EDChallengeSubmission.E2ED_SUBMISSION,
                **{k: v for k, v in fields.items() if not isinstance(v, list)},
                authors=fields["authors"], public_model_names=fields["public_model_names"])
            for n, xy in zip(names[start:start + per_file], traj[start:start + per_file]):
                sub.predictions.add(frame_name=n,
                                    trajectory=pb.TrajectoryPrediction(pos_x=xy[:, 0], pos_y=xy[:, 1]))
            blob = sub.SerializeToString()
            info = tarfile.TarInfo(f"{fields['unique_method_name']}_{start // per_file:03d}.bin")
            info.size, info.mtime = len(blob), 0
            tar.addfile(info, io.BytesIO(blob))
    log.info("submission: %d frames, %.2f MB -> %s", len(names), path.stat().st_size / 2**20, path)
    return path


def read_submission(path) -> tuple[list[str], np.ndarray, dict]:
    """Parse a submission tar.gz back: (frame names, (n, 20, 2) trajectories, the metadata fields)."""
    pb = proto("end_to_end_driving_submission_pb2")
    names, traj, meta = [], [], {}
    with tarfile.open(path, "r:gz") as tar:
        for m in sorted(tar.getmembers(), key=lambda m: m.name):
            sub = pb.E2EDChallengeSubmission.FromString(tar.extractfile(m).read())
            assert sub.submission_type == pb.E2EDChallengeSubmission.E2ED_SUBMISSION, "wrong submission type"
            meta = {f.name: (list(v) if f.is_repeated else v) for f, v in sub.ListFields()
                    if f.name not in ("predictions", "submission_type")}
            for p in sub.predictions:
                names.append(p.frame_name)
                traj.append(np.stack([p.trajectory.pos_x, p.trajectory.pos_y], -1))
    return names, np.asarray(traj, np.float32), meta


# ---------------------------------------------------------------- reports and checks

def report(df=None, past=None, future=None) -> dict:
    df = load_index() if df is None else df
    past, future = load_ego() if past is None else (past, future)
    seq, hist = sequence_stats(df), history_report(df)
    base = baseline_table(df, past, future)
    log.info("sequences:\n%s", seq.to_string(index=False))
    log.info("intent x split:\n%s", pd.crosstab(df.split, df.intent.map(dict(enumerate(INTENTS))), margins=True))
    log.info("image-history completeness (share of frames whose whole window resolves):\n%s",
             hist.to_string(index=False))
    log.info("ego-only baselines on val (m):\n%s", base.to_string(index=False))
    return {"sequences": seq, "history": hist, "baselines": base}


def check(n: int = 64) -> None:
    """Assertions on whatever shards are present: byte spans, ego frame conventions, history policy,
    baseline sanity and a submission round-trip. Raises on the first failure."""
    from PIL import Image
    E2EDFrame = e2ed_frame()
    df, (past, future) = load_index(), load_ego()
    assert len(df) == len(past) == len(future) and len(df) > 0
    rng = np.random.default_rng(0)
    rows = rng.choice(len(df), min(n, len(df)), replace=False)

    for i in rows:                                              # byte spans point at the right record and JPEGs
        r = df.iloc[i]
        with open(shard_dir() / str(r.shard), "rb") as f:
            f.seek(r.rec_off)
            fr = E2EDFrame.FromString(f.read(r.rec_len))
            assert fr.frame.context.name == f"{r.sequence}-{r.frame:03d}", fr.frame.context.name
            by_cam = {CAM_IDS[im.name]: im.image for im in fr.frame.images}
            for c in CAMS:
                f.seek(r[f"{c}_off"])
                blob = f.read(r[f"{c}_len"])
                assert blob == by_cam[c], f"{c} bytes differ at row {i}"
                im = Image.open(io.BytesIO(blob))
                assert im.size == (972, 1079) and im.format == "JPEG", (c, im.size, im.format)
            assert np.allclose(np.array(fr.past_states.pos_x, np.float32), past[i, :, 0], atol=1e-6)
            assert (len(fr.future_states.pos_x) == N_FUTURE) == bool(r.has_future)
    log.info("check: %d records re-read from their byte spans, JPEGs and ego states match", len(rows))

    assert (np.abs(past[:, -1, :2]) < 1e-3).all(), "past position at t=0 must be the origin"
    assert df.intent.between(0, 3).all() and df.frame.between(0, 1 << 20 - 1).all()
    assert (df[df.split == "test"].has_future == False).all(), "test futures must be hidden"  # noqa: E712
    assert df[df.split != "test"].has_future.all(), "train/val frames must all have a future"
    log.info("check: ego frame conventions hold on all %d frames", len(df))

    rows_h, dt, got = history_rows(df, 3, 5)
    seq = df.sequence.to_numpy()
    assert (seq[rows_h] == seq[:, None]).all(), "history left the sequence"
    fr = df.frame.to_numpy()[rows_h]
    assert (np.diff(fr, axis=1) <= 0).all() and (fr <= fr[:, :1]).all(), "history is not in the past"
    assert got[:, 0].all() and (dt[:, 0] == 0).all()
    assert np.isclose(dt, (fr - fr[:, :1]) * FRAME_DT).all()
    log.info("check: history windows stay in the sequence, in the past, and their dt matches the frame indices")

    m = (df.split == "val").to_numpy() & df.has_future.to_numpy()
    if m.any():
        gt, b = future_xy(future[m]), baselines(past[m])
        straight = np.abs(past_kinematics(past[m])["w"]) < 0.02
        assert ade_fde(b["ctrv"], gt)["ade5"] < ade_fde(b["zero"], gt)["ade5"], "CTRV must beat standing still"
        assert ade_fde(b["cv"][straight], gt[straight])["ade5"] < ade_fde(b["cv"], gt)["ade5"], \
            "CV must be better on straight frames"
        log.info("check: baselines ordered as expected (%d val frames, %d near-straight)", m.sum(), straight.sum())

    names = submission_frames()
    traj = np.tile(np.stack([np.arange(1, N_FUTURE + 1) * DT * 10, np.zeros(N_FUTURE)], -1), (len(names), 1, 1))
    meta = {"account_name": "a@b.c", "unique_method_name": "roundtrip_check", "authors": ["x", "y"],
            "affiliation": "VennIntelligence", "description": "d", "method_link": "https://example.org",
            "uses_public_model_pretraining": True, "public_model_names": ["Qwen/Qwen3-VL-4B-Instruct"],
            "num_model_parameters": "4B"}
    p = write_submission(names, traj, out_dir("submissions") / "roundtrip_check.tar.gz", meta)
    got_names, got_traj, got_meta = read_submission(p)
    assert got_names == names and np.allclose(got_traj, traj, atol=1e-5)
    assert got_meta == meta, {k: (meta[k], got_meta.get(k)) for k in meta if got_meta.get(k) != meta[k]}
    p.unlink()
    log.info("check: submission with %d frames round-trips with every field intact", len(names))
    log.info("all checks passed")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("index", "report", "check", "features", "bench"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="index: rescan every shard")
    ap.add_argument("--cams", default="front3", choices=("front3", "front"))
    ap.add_argument("--separate", action="store_true", help="one forward per camera instead of one per frame")
    ap.add_argument("--long-side", type=int, default=0, help="resize so the long side is this many px (0: native)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N frames only")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    cams = CAMS if a.cams == "front3" else ("front",)
    kw = dict(cams=cams, separate=a.separate, long_side=a.long_side or None, batch_size=a.batch_size,
              workers=a.workers, splits=tuple(a.splits.split(",")), compile=not a.no_compile)

    if a.cmd == "index":
        build_index(a.workers, a.force)
    elif a.cmd == "report":
        report()
    elif a.cmd == "check":
        check()
    elif a.cmd == "features":
        extract_features(limit=a.limit, **kw)
    elif a.cmd == "bench":
        from .runlog import RunLog
        rl = RunLog("waymo", "bench")
        rows = []
        for cs, sep in ((CAMS, False), (CAMS, True), (("front",), False)):
            for ls in (None, 800):
                m = extract_features(cams=cs, separate=sep, long_side=ls, batch_size=a.batch_size,
                                     workers=a.workers, limit=a.limit or 96, splits=tuple(a.splits.split(",")),
                                     save=False, rl=rl, compile=not a.no_compile)
                per_frame = len(cs) if sep else 1        # a "sample" is one forward, a frame may need three
                rows.append({**m, "ms_per_frame": m["ms_per_frame"] * per_frame,
                             "frames_per_s": m["frames_per_s"] / per_frame,
                             "bytes_per_frame": m["bytes_per_sample"] * per_frame})
                rl.event("bench", **{k: v for k, v in rows[-1].items() if not isinstance(v, list)})
        (rl.dir / "bench.json").write_text(json.dumps(rows, indent=2, default=float))
        log.info("feature-extraction bench (per frame, batch %d):\n%s", a.batch_size,
                 pd.DataFrame(rows)[["set", "tokens_per_frame", "ms_per_frame", "frames_per_s", "peak_vram_gb",
                                     "bytes_per_frame"]].round(2).to_string(index=False))
        rl.close()


if __name__ == "__main__":
    main()
