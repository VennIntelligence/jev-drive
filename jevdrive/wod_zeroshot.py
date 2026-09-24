"""WOD-E2E zero-shot exam of open driving models (todos/2026-09-24-zeroshot-exam/wod-e2e.md).

  sets        the pre-registered frame sets (479 rater frames, <= 958 ADE-extra frames, 8 adapter-check frames),
              frozen with everything scoring needs (past, future, rater trajectories, cluster, intent) and the
              slim-shard JPEG spans of the openpilot history frames, so later steps never read the moving index
  fetch       full 8-camera records of frames f-3..f for Alpamayo, pulled from the raw GCS val shards by
              walking their TFRecord framing (the slim shards only keep the front three cameras)
  converters  WOD past states -> Alpamayo egomotion history; Alpamayo / openpilot outputs -> WOD 20 x 0.25 s
  scoring     RFS / ADE with bootstrap CIs, reusing jevdrive.waymo's bit-exact RFS

This module is imported by three venvs (project, alpamayo, openpilot): only numpy/scipy at import time.
"""
import json
import os
from pathlib import Path

import numpy as np

from .common import data_dir, n_cpus

N_FUT, DT_FUT = 20, 0.25
T_FUT = np.arange(1, N_FUT + 1) * DT_FUT                   # 0.25 ... 5.0 s
T_PAST4 = (np.arange(16) - 15) * 0.25                      # WOD past_states: -3.75 ... 0 s
T_HIST10 = (np.arange(16) - 15) * 0.1                      # Alpamayo egomotion history: -1.5 ... 0 s
ALP_T = np.arange(1, 65) * 0.1                             # Alpamayo output: 0.1 ... 6.4 s
WOD_CAMS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT", 6: "REAR_LEFT",
            7: "REAR", 8: "REAR_RIGHT"}
ALP_SRC = (1, 2, 3, 4, 5, 6, 8)                            # every camera but REAR can reach a virtual view
OP_SRC = (1, 2, 3)                                         # the openpilot frames are inside the front three
NAV_TEXT = {1: "Continue straight", 2: "Turn left", 3: "Turn right"}   # pre-registered; UNKNOWN -> no nav
OP_WARMUP = 100                                            # WOD frames (10 s) fed before the target frame


def root(*parts) -> Path:
    d = data_dir() / "processed" / "wod_zeroshot" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- frame sets

def build_sets(seed: int = 0) -> dict:
    """Pick and freeze the pre-registered frame sets. Writes sets.npz + sets.json under root()."""
    import pandas as pd
    from . import waymo as W
    for _ in range(5):  # the train watcher renames new index files in; retry until index and ego agree
        df = W.load_index()
        past, fut = W.load_ego()
        if len(past) == len(fut) == len(df):
            break
    else:
        raise RuntimeError("index and past/future arrays disagree; pin a snapshot")
    names = W.frame_names(df)
    rows_r, traj, scores = W.load_rater(df)
    val = df.split.astype(str).to_numpy() == "val"
    seq = df.sequence.astype(str).to_numpy()
    frame = df.frame.to_numpy()
    rater_frame = dict(zip(seq[rows_r], frame[rows_r]))
    rng = np.random.default_rng(seed)
    extra, check_pool = [], []
    by_seq = pd.Series(np.flatnonzero(val)).groupby(seq[val]).apply(list)
    for s in sorted(by_seq.index):
        idx = np.array(by_seq[s])
        rf = rater_frame.get(s, -999)
        ok = idx[(frame[idx] >= 100) & df.has_future.to_numpy()[idx] & (np.abs(frame[idx] - rf) >= 10)]
        extra += list(rng.choice(ok, min(2, len(ok)), replace=False)) if len(ok) else []
    extra = np.sort(np.array(extra))
    # adapter-check frames: outside both sets, chosen by kinematics (images are looked at, never scored)
    taken = set(rows_r) | set(extra)
    kin = W.past_kinematics(past)
    cand = np.array([i for i in np.flatnonzero(val & df.has_future.to_numpy() & (frame >= 100)) if i not in taken])
    intent, v, w = df.intent.to_numpy()[cand], kin["v"][cand], np.degrees(kin["w"][cand])
    picks = []
    for m, k in (((intent == 1) & (v > 10) & (np.abs(w) < 1), 2), ((intent == 2) & (v > 3), 2),
                 ((intent == 3) & (v > 3), 2), ((v < 0.2), 1), ((intent == 1) & (v > 5), 1)):
        pool = [c for c in cand[m] if seq[c] not in {seq[p] for p in picks}]
        picks += list(rng.choice(pool, k, replace=False))
    check = np.array(picks)

    def pack(rows):
        u = lambda x: np.asarray(x, dtype=str)  # noqa: E731  (no object arrays: loadable without pickle)
        return dict(name=u(names[rows]), sequence=u(seq[rows]), frame=frame[rows], intent=df.intent.to_numpy()[rows],
                    cluster=u(df.cluster.astype(str).to_numpy()[rows]), shard=u(df.shard.astype(str).to_numpy()[rows]),
                    past=past[rows], future=fut[rows])
    out = {"rater": pack(rows_r) | dict(traj=traj, scores=scores), "extra": pack(extra), "check": pack(check)}
    # slim-shard JPEG spans of every frame in each sequence we touch (openpilot history), keyed by frame name
    need = set(out["rater"]["sequence"]) | set(out["extra"]["sequence"]) | set(out["check"]["sequence"])
    m = val & np.isin(seq, list(need))
    spans = {n: [str(df.shard.iloc[i])] + [int(df[f"{c}_{s}"].iloc[i]) for c in W.CAMS for s in ("off", "len")]
             for n, i in zip(names[m], np.flatnonzero(m))}
    # the slim shard's record ordinal of every frame, to locate it in the raw GCS shard (same order)
    ordn = df[val].groupby("shard", observed=True).rec_off.rank(method="first").astype(int) - 1
    ordinal = dict(zip(names[val], ordn.reindex(df.index[val]).to_numpy()))
    np.savez(root() / "sets.npz", **{f"{k}/{f}": v for k, d in out.items() for f, v in d.items()})
    (root() / "sets.json").write_text(json.dumps({"seed": seed, "spans": spans,
                                                  "ordinal": {k: int(v) for k, v in ordinal.items()}}))
    return {k: len(v["name"]) for k, v in out.items()}


def load_sets() -> dict:
    z = np.load(root() / "sets.npz", allow_pickle=False)
    out = {}
    for k in z.files:
        s, f = k.split("/")
        out.setdefault(s, {})[f] = z[k]
    return out


def load_spans() -> tuple[dict, dict]:
    js = json.loads((root() / "sets.json").read_text())
    return js["spans"], js["ordinal"]


def history_names(name: str, n_back: int, first: int = 0) -> list[str]:
    seq, f = name.rsplit("-", 1)
    return [f"{seq}-{k:03d}" for k in range(max(first, int(f) - n_back), int(f) + 1)]


# ---------------------------------------------------------------- Alpamayo input conversion

def alpamayo_history(past: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WOD past_states (16, 6) @4 Hz -> Alpamayo egomotion history xyz (16, 3), rot (16, 3, 3) @10 Hz, t0 frame.
    Positions: cubic spline of the 16 x/y samples; z = 0. Yaw from the velocity direction, shifted so yaw(t0) = 0;
    steps slower than 1 m/s take the yaw of the nearest moving step, 0 if the car never moved. Roll = pitch = 0."""
    from scipy.interpolate import CubicSpline
    past = np.asarray(past, np.float64)
    xy = CubicSpline(T_PAST4, past[:, :2], axis=0)(T_HIST10)
    v = past[:, 2:4]
    moving = np.linalg.norm(v, axis=1) >= 1.0
    if moving.any():
        yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        mi = np.flatnonzero(moving)
        near = mi[np.abs(np.arange(16)[:, None] - mi[None]).argmin(1)]
        yaw = yaw[near]
        yaw = yaw - yaw[-1]
        yaw10 = np.interp(T_HIST10, T_PAST4, yaw)
    else:
        yaw10 = np.zeros(16)
    c, s = np.cos(yaw10), np.sin(yaw10)
    rot = np.zeros((16, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    xyz = np.concatenate([xy, np.zeros((16, 1))], 1)
    xyz[-1] = 0  # the spline passes through the t0 sample, which is the origin
    return xyz.astype(np.float32), rot.astype(np.float32)


def resample(t_src: np.ndarray, xy: np.ndarray, t_dst: np.ndarray = T_FUT) -> np.ndarray:
    """Linear interpolation in time of (..., T, 2) trajectories, with the origin prepended at t = 0."""
    xy = np.asarray(xy, np.float64)
    xy0 = np.concatenate([np.zeros(xy.shape[:-2] + (1, 2)), xy], -2)
    t0 = np.concatenate([[0.0], t_src])
    flat = xy0.reshape(-1, len(t0), 2)
    out = np.stack([np.stack([np.interp(t_dst, t0, f[:, k]) for k in range(2)], -1) for f in flat])
    return out.reshape(xy.shape[:-2] + (len(t_dst), 2)).astype(np.float32)


def openpilot_to_wod(plan_pos: np.ndarray, plan_yaw: np.ndarray, t_idx: np.ndarray, dev_xy) -> np.ndarray:
    """openpilot plan (33 points, device frame x fwd / y right, origin at the camera) -> WOD 20 x 0.25 s
    rear-axle waypoints (+y left): rear(t) = d + p(t) - R(psi_t) d with d the camera's (x, y) on the vehicle."""
    p = np.stack([plan_pos[:, 0], -plan_pos[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(plan_yaw, np.float64)
    d = np.asarray(dev_xy, np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    rear = d + p - Rd
    return np.stack([np.interp(T_FUT, t_idx, rear[:, k]) for k in range(2)], -1).astype(np.float32)


def medoid(trajs: np.ndarray) -> int:
    """Index of the trajectory with the smallest mean ADE to the others. trajs (K, T, 2)."""
    d = np.linalg.norm(trajs[:, None] - trajs[None], axis=-1).mean(-1)
    return int(d.sum(1).argmin())


# ---------------------------------------------------------------- GCS fetch (project venv)

def _gcs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("waymo_e2e", Path(__file__).resolve().parents[1] / "scripts/waymo_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_records(want: dict, route: str = "proxy", proxy: str = "http://127.0.0.1:7890", workers: int = 48,
                  log=print) -> dict:
    """want: {slim shard name: {ordinal: frame name}}. Walk each raw shard's TFRecord framing up to the largest
    wanted ordinal (one 12-byte range GET per record), then GET the wanted records whole and write them to
    root('records')/<frame name>.pb. Verifies context.name. Idempotent. Returns counters."""
    import struct
    import time
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor
    mod = _gcs()
    gcs = mod.Gcs(route, proxy)
    out = root("records")
    E2ED = None

    def get(name, a, b):
        path = f"/{mod.BUCKET}/{urllib.parse.quote(name)}"
        return gcs._retry(lambda: gcs._get(path, {"Range": f"bytes={a}-{b}"}).read(), f"{name}[{a}]")

    def one(item):
        nonlocal E2ED
        shard, ords = item
        todo = {o: n for o, n in ords.items() if not (out / f"{n}.pb").exists()}
        if not todo:
            return 0, 0
        # record framing walked so far, cached per shard: [(payload offset, length)] by ordinal
        wf = root("walk") / f"{shard}.json"
        walked = json.loads(wf.read_text()) if wf.exists() else []
        nb, top = 0, max(todo)
        off = walked[-1][0] + walked[-1][1] + 4 if walked else 0
        while len(walked) <= top:
            head = get(shard, off, off + 11)
            (n,) = struct.unpack("<Q", head[:8])
            walked.append((off + 12, n))
            off += 12 + n + 4
            if len(walked) % 100 == 0 or len(walked) > top:
                wf.write_text(json.dumps(walked))
        locs = {k: walked[k] for k in todo}
        for o, (a, n) in locs.items():
            rec = get(shard, a, a + n - 1)
            nb += len(rec)
            if E2ED is None:
                E2ED = mod.e2ed_frame()
            got = E2ED.FromString(rec).frame.context.name
            if got != todo[o]:
                raise RuntimeError(f"{shard} record {o} is {got}, expected {todo[o]} (slim/raw order differs?)")
            tmp = out / f".{todo[o]}.tmp"
            tmp.write_bytes(rec)
            os.replace(tmp, out / f"{todo[o]}.pb")
        return len(locs), nb

    t0, recs, nbytes = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        for i, (r, b) in enumerate(ex.map(one, sorted(want.items()))):
            recs, nbytes = recs + r, nbytes + b
            log(f"[{i + 1}/{len(want)}] {recs} records, {nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s")
    return {"records": recs, "bytes": nbytes, "seconds": time.time() - t0}


def calib_dict(frame_proto, cams=ALP_SRC) -> dict:
    """{camera id: calib} from an E2EDFrame.frame (intrinsic 9, extrinsic 4x4, width, height)."""
    return {c.name: {"intrinsic": np.array(c.intrinsic, np.float64), "extrinsic": np.array(c.extrinsic.transform),
                     "width": c.width, "height": c.height}
            for c in frame_proto.context.camera_calibrations if c.name in cams}


def write_package(name: str, n_back: int = 3) -> Path:
    """Alpamayo input package for one target frame: JPEG bytes of ALP_SRC x frames f-3..f (from the fetched
    records) + the target frame's camera calibration, as a flat npz readable without protobuf."""
    from . import waymo as W
    E2ED = W.e2ed_frame()
    dst = root("packages") / f"{name}.npz"
    if dst.exists():
        return dst
    arrs, cal = {}, None
    for k, hn in enumerate(history_names(name, n_back)):
        fr = E2ED.FromString((root("records") / f"{hn}.pb").read_bytes()).frame
        imgs = {im.name: im.image for im in fr.images}
        for c in ALP_SRC:
            arrs[f"jpg_{k}_{c}"] = np.frombuffer(imgs[c], np.uint8)
        if hn == name:
            cal = calib_dict(fr)
    for c, d in cal.items():
        arrs[f"cal_{c}_intrinsic"], arrs[f"cal_{c}_extrinsic"] = d["intrinsic"], d["extrinsic"]
        arrs[f"cal_{c}_wh"] = np.array([d["width"], d["height"]])
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, dst)
    return dst


def read_calib(z, cams) -> dict:
    return {c: {"intrinsic": z[f"cal_{c}_intrinsic"], "extrinsic": z[f"cal_{c}_extrinsic"],
                "width": int(z[f"cal_{c}_wh"][0]), "height": int(z[f"cal_{c}_wh"][1])} for c in cams}


def write_op_calib(names) -> Path:
    """Front-three calibration per sequence (openpilot runner reads it without protobuf)."""
    from . import waymo as W
    E2ED, out = W.e2ed_frame(), {}
    df = W.load_index()
    key = dict(zip(W.frame_names(df), range(len(df))))
    for n in names:
        seq = n.rsplit("-", 1)[0]
        if seq in out:
            continue
        r = df.iloc[key[n]]
        with open(W.shard_dir() / str(r.shard), "rb") as f:
            f.seek(int(r.rec_off))
            fr = E2ED.FromString(f.read(int(r.rec_len))).frame
        out[seq] = {str(c): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
                    for c, d in calib_dict(fr, OP_SRC).items()}
    p = root() / "op_calib.json"
    p.write_text(json.dumps(out))
    return p


# ---------------------------------------------------------------- scoring (project venv)

def bootstrap(fn, n: int, B: int = 10000, strata: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """B bootstrap replicates of fn(idx). With strata, resample within each stratum (every stratum kept)."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    return np.array([fn(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(B)])


def ci(x: np.ndarray, q=(2.5, 97.5)) -> tuple[float, float]:
    lo, hi = np.percentile(x, q)
    return float(lo), float(hi)
