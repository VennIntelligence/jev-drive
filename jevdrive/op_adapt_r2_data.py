"""op-adapt round 2, package C: data, indexes, caches and teacher (todos/2026-09-29-op-adapt-r2-prereg.md §2.1 v3, §3,
§5 items 4-5; the format contract is section C of tmp/2026-09-30-op-adapt-r2-build.md).

  splits   sim: Town12 instances of the Cosmos G4 full set, seed-0 permutation, 75 / 10 / 15 (all variants of an instance
           together); nuScenes: round 1's scene split; WOD train sequences and navtrain logs: seed-0 permutation, 5 % dev
  sim      per pair, from the pass-1 CARLA worlds (runs/cosmos_full/gen): 5 Hz slots 9..23 of the 93-tick window
           (tick = k0 + 4 slot), round-1 corridor labels on the CARLA walker hazards (rear-axle frame, logged 3 s future
           path extended to 30 m), px_eq = the gt.npz mask pixels (the Cosmos camera is the px_eq camera), trigger tick,
           BehaviorAgent speed difference dv*(1, 2, 3 s) for A-bhv, and a compact world file for the scorer
  index    one parquet per domain (simC simK nus wod nav off p5), one row per slot, joined on `uid`
  offset   the navtrain offset-start sample table (§2.1 v3, all numbers fixed there) and the plane-induced homography
           model-frame maps
  loader   TrunkStore: cached stage-3 trunks -> (B, 9, 1024, 8, 16) context batches for op_adapt.stage4_policy
  teacher  the original Cinque's plan MDN (mean, std) without desire and with a laneChangeLeft / Right pulse
"""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir

DOMAINS = ("simC", "simK", "nus", "wod", "nav", "off", "p5")
UID_BASE = {d: (i + 1) * 10 ** 9 for i, d in enumerate(DOMAINS)}
CTX = 9
NSLOT, STEP, MIN_SLOT = 24, 4, 9             # Cosmos window: 93 ticks at 20 Hz -> 24 slots at 5 Hz; slots >= 9 in training
STREAMS = ("C+", "C-", "K+", "K-")           # row block order of cache-sim/<pair>.npy
VIS_PX, BIG_PX = 68.0, 500.0                 # P5 visibility threshold, Q1 main bin (px_eq)
REAR_AXLE_X = -1.388633220                   # hero rear axle relative to the CARLA actor centre (p4_carla, M0)
FUT_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)       # corridor path: logged future, as the nuScenes labels (N_FUT = 6)
TICK = 0.05
AT = (0.275, 0.525)
DESIRES = (0, 3, 4)                           # op (none), op_L (laneChangeLeft), op_R (laneChangeRight)
F_COSMOS = 640.0 / math.tan(math.radians(32.0))
F_P5_SEG = 1113.5 / 2


def root(*parts) -> Path:
    d = data_dir() / "runs" / "op_adapt_r2" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(p: Path) -> str:
    return str(Path(p).relative_to(data_dir()))


def load_index(domain: str) -> pd.DataFrame:
    return pd.read_parquet(root("index") / f"{domain}.parquet")


def npz_shape(f: Path, key: str = "trunk") -> tuple:
    """Shape of one member of an .npz without reading it."""
    with zipfile.ZipFile(f) as z, z.open(f"{key}.npy") as fh:
        v = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
        return rd(fh)[0]


def stride_ctx(slots: np.ndarray, stride: int) -> np.ndarray:
    """(n, 9) context rows j - stride * (8 .. 0); -1 before the stream start."""
    c = np.asarray(slots)[:, None] + stride * np.arange(-(CTX - 1), 1)[None]
    return np.where(c >= 0, c, -1).astype(np.int32)


# ---------------------------------------------------------------- splits
def _perm_split(items, fracs: dict, seed: int = 0) -> dict:
    items = sorted(items)
    perm = [str(x) for x in np.random.default_rng(seed).permutation(np.array(items, dtype=object))]
    out, a = {}, 0
    names = list(fracs)
    for i, k in enumerate(names):
        b = len(perm) if i == len(names) - 1 else a + int(round(fracs[k] * len(perm)))
        out.update({x: k for x in perm[a:b]})
        a = b
    return {"seed": seed, "fracs": fracs, "perm": perm, "split": out}


def sim_pairs() -> pd.DataFrame:
    """Every finished Cosmos G4 pair with its spec and pass-1 route ids."""
    from .cosmos_full import main_dir
    pdir = main_dir() / "pairs"
    var = pd.read_csv(main_dir() / "variants.csv", dtype={"id1": str, "id2": str}).set_index("pair")
    rows = []
    for p in sorted(pdir.iterdir()):
        if not (p / "done.json").exists():
            continue
        s = json.loads((p / "spec.json").read_text())
        rows.append({"pair": s["pair"], "inst": int(s["inst"]), "variant": int(s["v"]), "family": s["family"],
                     "k0": int(s["k0"]), "k1": int(s["k1"]), "id1": var.at[s["pair"], "id1"], "id2": var.at[s["pair"], "id2"]})
    return pd.DataFrame(rows)


def make_splits() -> dict:
    """splits/*.json (deterministic; rewriting them gives the same files)."""
    out = {}
    sp = sim_pairs()
    s = _perm_split(sp.inst.unique().tolist(), {"train": 0.75, "dev": 0.10, "test": 0.15})
    s["split"] = {str(k): v for k, v in s["split"].items()}
    out["sim_instances"] = s
    from . import op_adapt_data as D
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
    tr = np.array(sorted(lab[lab.split == "train"].scene.unique()))
    dev = set(np.random.default_rng(0).permutation(tr)[:50])          # scripts/op_adapt_train.split_scenes
    out["nus_scenes"] = {"seed": 0, "rule": "op_adapt_train.split_scenes", "split": {
        **{x: ("dev" if x in dev else "train") for x in tr}, **{x: "val" for x in sorted(lab[lab.split == "val"].scene.unique())}}}
    wl = pd.read_parquet(D.root() / "wod_train_labels.parquet", columns=["sequence"])
    out["wod_train_seqs"] = _perm_split(wl.sequence.unique().tolist(), {"dev": 0.05, "train": 0.95})
    logs = [p.stem for p in D.root("navtrain").glob("*.npz") if not p.stem.endswith(".tmp")]
    out["navtrain_logs"] = _perm_split(logs, {"dev": 0.05, "train": 0.95})
    for k, v in out.items():
        (root("splits") / f"{k}.json").write_text(json.dumps(v))
    return out


def splits(name: str) -> dict:
    return json.loads((root("splits") / f"{name}.json").read_text())["split"]


# ---------------------------------------------------------------- sim pairs: worlds, labels, dv*
def _interp_actor(k_rec: np.ndarray, v_rec: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Linear interpolation of an actor's recorded (5 Hz) track to ticks k; NaN more than one camera period away."""
    out = np.stack([np.interp(k, k_rec, v_rec[:, i]) for i in range(v_rec.shape[1])], -1)
    far = (k < k_rec[0] - STEP) | (k > k_rec[-1] + STEP)
    out[far] = np.nan
    return out


def _rear(pose_xy: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    y = np.radians(yaw_deg)
    return pose_xy + REAR_AXLE_X * np.stack([np.cos(y), np.sin(y)], -1)


def _to_ego(xy: np.ndarray, origin: np.ndarray, yaw_deg: float) -> np.ndarray:
    """World (CARLA, left-handed) -> ego (forward, right); the corridor is symmetric in the lateral sign."""
    y = math.radians(yaw_deg)
    c, s = math.cos(y), math.sin(y)
    d = xy - origin
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def _world(adir: Path) -> dict:
    from . import p5_pairs as PP
    W = PP.load_world(adir)
    p = W["pose"]
    summ = json.loads((adir / "p5_summary.json").read_text())
    tt = summ.get("t_trigger")
    route = pd.read_json(adir / "route.json")[["x", "y", "z"]].to_numpy(np.float32)
    return {"W": W, "pose": np.c_[p.index.to_numpy(), p[["x", "y", "z", "yaw", "vx", "vy"]].to_numpy()].astype(np.float64),
            "k_trig": int(round(tt / TICK)) if tt is not None else -1, "route": route}


def sim_pair(pair: pd.Series) -> list[dict]:
    """World file R2/sim/world/<pair>.npz and the per-slot label rows of one pair (x+ geometry; x- rows get no walker)."""
    from . import op_adapt_data as D
    from . import p5_pairs as PP
    from .cosmos_full import main_dir
    gen = main_dir() / "gen"
    pdir = main_dir() / "pairs" / pair.pair
    wp, wm = _world(PP.attempt(gen, pair.id1)), _world(PP.attempt(gen, pair.id2))
    W = wp["W"]
    walk = [(h, t) for h, t in zip(W["hazards"], W["hazard_types"]) if str(t).startswith("walker.")]
    gtb = np.load(pdir / "gt_boxes.npz")
    gt = np.load(pdir / "gt.npz")
    assert len(gtb["hazards"]) == len(walk), (pair.pair, len(gtb["hazards"]), len(walk))
    act = W["act"]
    # world file for the scorer: both pass-1 worlds, raw CARLA frames (x, y in m, yaw in degrees, left-handed)
    wf = {"k0": pair.k0, "k1": pair.k1, "hz_ids": np.array([h for h, _ in walk], np.int64),
          "hz_px": gtb["px"], "hz_box": gtb["box"], "px": gt["px"], "route": wp["route"], "kinds": json.dumps(W["kinds"])}
    for m, w in (("plus", wp), ("minus", wm)):
        a = w["W"]["act"]
        wf |= {f"{m}_pose": w["pose"], f"{m}_k_trig": w["k_trig"], f"{m}_hazards": np.array(w["W"]["hazards"], np.int64),
               f"{m}_act_k": np.asarray(a["k"], np.int64), f"{m}_act_id": a["id"], f"{m}_act_xyz": a["xyz"],
               f"{m}_act_yaw": a["yaw"], f"{m}_act_v": a["v"]}
    tmp = root("sim", "world") / f"{pair.pair}.tmp.npz"
    np.savez_compressed(tmp, **wf)
    tmp.replace(root("sim", "world") / f"{pair.pair}.npz")
    # labels
    slots = np.arange(MIN_SLOT, NSLOT)
    ticks = pair.k0 + STEP * slots
    P = pd.DataFrame(wp["pose"][:, 1:], index=wp["pose"][:, 0].astype(int), columns=["x", "y", "z", "yaw", "vx", "vy"])
    M = pd.DataFrame(wm["pose"][:, 1:], index=wm["pose"][:, 0].astype(int), columns=["x", "y", "z", "yaw", "vx", "vy"])
    tracks = {}
    for h, _ in walk:
        m = act["id"] == h
        if m.sum():
            o = np.argsort(act["k"][m])
            tracks[h] = (act["k"][m][o].astype(float), np.c_[act["xyz"][m][o][:, :2], act["yaw"][m][o]].astype(float),
                         2 * np.array(W["kinds"][str(h)][2][3:5], float))
    speed = lambda F, k: float(np.hypot(F.at[k, "vx"], F.at[k, "vy"])) if k in F.index else np.nan  # noqa: E731
    rows = []
    for j, k in zip(slots, ticks):
        e = P.loc[k]
        o = _rear(np.array([e.x, e.y]), np.array(e.yaw))
        fk = [k + int(round(t / TICK)) for t in FUT_S]
        fk = [x for x in fk if x in P.index]
        fut = _to_ego(_rear(P.loc[fk, ["x", "y"]].to_numpy(), P.loc[fk, "yaw"].to_numpy()), o, e.yaw) if fk else np.zeros((0, 2))
        xy, yaw, lw = [], [], []
        for h, (kr, vr, l_w) in tracks.items():
            v = _interp_actor(kr, vr, np.array([float(k)]))[0]
            if np.isfinite(v).all():
                xy.append(_to_ego(v[:2], o, e.yaw))
                yaw.append(math.radians(v[2] - e.yaw))
                lw.append(l_w)
        xy = np.array(xy).reshape(-1, 2)
        f = D._flags(fut, xy, np.array(yaw), np.array(lw).reshape(-1, 2), np.ones(len(xy), bool))
        dv = {f"dv{t}": speed(P, k + 20 * t) - speed(M, k + 20 * t) for t in (1, 2, 3)}
        rows.append({"pair": pair.pair, "slot": int(j), "tick": int(k), "px_eq": float(gt["px"][STEP * j]),
                     "k_trig": wp["k_trig"], "n_fut": len(fk), "v0": speed(P, k), **f, **dv})
    return rows


def sim_labels(workers: int = 24) -> pd.DataFrame:
    from multiprocessing import Pool
    sp = sim_pairs()
    with Pool(workers) as p:
        rows = [r for part in p.imap(sim_pair, [r for _, r in sp.iterrows()], chunksize=4) for r in part]
    lab = pd.DataFrame(rows).merge(sp[["pair", "inst", "variant", "family"]], on="pair")
    lab.to_parquet(root("sim") / "labels.parquet", index=False)
    return lab


# ---------------------------------------------------------------- indexes
COMMON = ["uid", "domain", "key", "slot", "sign", "split", "file", "row", "ctx", "tc0", "tc1", "token", "labeled",
          "ped_corr", "ped_wide", "vru_wide", "ped_dist", "dist_bin", "uncertain", "normal"]


def _finish(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "uid", np.int64(UID_BASE[domain]) + np.arange(len(df), dtype=np.int64))
    df.insert(1, "domain", domain)
    for c, v in (("sign", 0), ("token", ""), ("labeled", False), ("ped_corr", False), ("ped_wide", False), ("vru_wide", False),
                 ("ped_dist", np.nan), ("dist_bin", -1), ("uncertain", False), ("normal", False)):
        if c not in df:
            df[c] = v
    df = df.astype({"slot": np.int32, "sign": np.int8, "row": np.int32, "tc0": np.float32, "tc1": np.float32,
                    "ped_dist": np.float32, "dist_bin": np.int8})
    df["ctx"] = [np.asarray(c, np.int32) for c in df.ctx]
    return df[COMMON + [c for c in df.columns if c not in COMMON]]


def index_sim() -> dict:
    from . import op_adapt_data as D
    lab = pd.read_parquet(root("sim") / "labels.parquet").sort_values(["pair", "slot"]).reset_index(drop=True)
    sp = splits("sim_instances")
    lab["split"] = lab.inst.astype(str).map(sp)
    assert lab.split.notna().all()
    lab["dist_bin"] = D.dist_bin(lab)
    out = {}
    for dom, s0 in (("simC", 0), ("simK", 2)):
        parts = []
        for sign, sb in ((1, s0), (-1, s0 + 1)):
            d = lab.copy()
            d["sign"] = sign
            d["row"] = sb * NSLOT + d.slot
            d["ctx"] = [sb * NSLOT + np.arange(j - CTX + 1, j + 1) for j in d.slot]
            if sign < 0:
                d[["ped_corr", "ped_wide", "vru_corr", "vru_wide"]] = False
                d[["ped_dist", "ped_dist_wide"]] = np.nan
                d["px_eq"], d["dist_bin"] = 0.0, -1
            d["normal"] = sign < 0
            parts.append(d)
        d = pd.concat(parts).sort_values(["pair", "slot", "sign"], ascending=[True, True, False]).rename(columns={"pair": "key"})
        d["file"] = [rel(root("cache-sim") / f"{k}.npy") for k in d.key]
        d["tc0"], d["tc1"], d["labeled"] = 1.0, 0.0, True
        d["vis"] = d.px_eq >= VIS_PX
        d["pre_trig"] = (d.k_trig >= 0) & (d.tick < d.k_trig)
        out[dom] = _finish(d, dom)
    c, k = out["simC"], out["simK"]
    for d in (c, k):
        d["twin_uid"] = d.uid + np.where(d.sign > 0, 1, -1)       # rows alternate +, - per (pair, slot)
    c["other_uid"], k["other_uid"] = k.uid.to_numpy(), c.uid.to_numpy()
    for dom, d in out.items():
        d.to_parquet(root("index") / f"{dom}.parquet", index=False)
    return out


def index_nus() -> pd.DataFrame:
    from . import op_adapt_data as D
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet").set_index("token")
    sp = splits("nus_scenes")
    parts = []
    for sc in sorted(sp):
        f = D.root("nusc") / f"{sc}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        n = npz_shape(f)[0]
        key = {int(k): str(t) for k, t in zip(z["key_slot"], z["tokens"])}
        kn = {k: not bool(lab.at[t, "vru_wide"]) for k, t in key.items() if t in lab.index}
        j = np.arange(n)
        k0 = 5 * (j // 5)
        norm = [kn.get(a, False) and (b % 5 == 0 or kn.get(a + 5, False)) for a, b in zip(k0, j)]   # round 1's rule
        d = pd.DataFrame({"key": sc, "slot": j, "split": sp[sc], "file": rel(f), "row": j, "ctx": list(stride_ctx(j, 2)),
                          "tc0": z["traffic"][0], "tc1": z["traffic"][1], "token": [key.get(x, "") for x in j], "normal": norm})
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    L = lab.reindex(d.token)
    d["labeled"] = (d.token != "").to_numpy()                    # keyframes (round 1 trained on every keyframe)
    for c in ("ped_corr", "ped_wide", "vru_corr", "vru_wide"):
        d[c] = L[c].fillna(False).to_numpy(bool)
    d["ped_dist"] = L.ped_dist.to_numpy(float)
    d["ped_dist_wide"] = L.ped_dist_wide.to_numpy(float)
    d["dist_bin"] = D.dist_bin(d)
    d = _finish(d, "nus")
    d.to_parquet(root("index") / "nus.parquet", index=False)
    return d


def labels_wod_val(workers: int = 16) -> pd.DataFrame:
    """R-wod (§4.1): the labels_wod rule (op_adapt_data.labels_wod) on the WOD val readout streams' frames that YOLO26x
    saw (real_transfer/yolo/wod_val) -> R2/labels/wod_val_labels.parquet."""
    from multiprocessing import Pool
    from . import op_adapt_data as D
    y = data_dir() / "processed/real_transfer/yolo/wod_val"
    fr = pd.read_parquet(y / "frames.parquet")
    names = set()
    for f in sorted(D.root("wod").glob("*.npz")):
        if not f.stem.endswith(".tmp"):
            names |= set(np.load(f)["names"].tolist())
    fr = fr[fr.frame_id.isin(names)]
    d = pd.read_parquet(y / "dets.parquet", columns=["frame_id", "prompt", "score", "gx", "gy", "lift_ok"])
    d = d[d.prompt.isin(["pedestrian", "cyclist"]) & d.frame_id.isin(set(fr.frame_id))]
    fut = np.load(data_dir() / "processed/waymo_e2e/future.npy", mmap_mode="r")
    by = d.groupby("frame_id")
    jobs, chunk = [], []
    for fid, row in zip(fr.frame_id, fr.row):
        g = by.get_group(fid) if fid in by.groups else d.iloc[:0]
        ok = g.lift_ok.to_numpy(bool)
        chunk.append((fid, np.asarray(fut[row][:, :2], np.float64), g[["gx", "gy"]].to_numpy(float)[ok],
                      (g.prompt.to_numpy() == "pedestrian")[ok], (~ok).any()))
        if len(chunk) == 512:
            jobs.append(chunk)
            chunk = []
    jobs.append(chunk)
    with Pool(workers) as p:
        rows = [r for part in p.imap(D._wod_worker, jobs) for r in part]
    lab = pd.DataFrame(rows).merge(fr[["frame_id", "sequence"]], on="frame_id")
    lab["dist_bin"] = D.dist_bin(lab)
    lab.to_parquet(root("labels") / "wod_val_labels.parquet", index=False)
    return lab


def index_wod() -> pd.DataFrame:
    from . import op_adapt_data as D
    sp = splits("wod_train_seqs")
    lt = pd.read_parquet(D.root() / "wod_train_labels.parquet").set_index("frame_id")
    lv = pd.read_parquet(root("labels") / "wod_val_labels.parquet").set_index("frame_id")
    parts = []
    for part, sub, lab in (("train", "wodtrain", lt), ("val", "wod", lv)):
        for f in sorted(D.root(sub).glob("*.npz")):
            if f.stem.endswith(".tmp"):
                continue
            z = np.load(f)
            names, st = z["names"].astype(str), int(z["stride"])
            j = np.arange(len(names))
            seq = f.stem.split("_", 1)[1]
            tg = np.zeros(len(j), bool)
            tg[z["targets"]] = True
            parts.append(pd.DataFrame({"key": f.stem, "slot": j, "part": part, "name": names, "target": tg,
                                       "split": sp.get(seq, "?") if part == "train" else "val", "file": rel(f), "row": j,
                                       "ctx": list(stride_ctx(j, 1 if st == 1 else 2)), "tc0": z["traffic"][0],
                                       "tc1": z["traffic"][1]}))
    d = pd.concat(parts, ignore_index=True)
    L = pd.concat([lt, lv]).reindex(d.name)                    # frame ids are unique across train and val
    d["labeled"] = L.ped_corr.notna().to_numpy()
    for c in ("ped_corr", "ped_wide", "vru_corr", "vru_wide", "uncertain"):
        d[c] = L[c].fillna(False).to_numpy(bool)
    d["ped_dist"] = L.ped_dist.to_numpy(float)
    d["ped_dist_wide"] = L.ped_dist_wide.to_numpy(float)
    d["dist_bin"] = np.where(d.labeled, L.dist_bin.fillna(-1).to_numpy(), -1)
    d["normal"] = d.labeled & ~d.vru_wide & ~d.uncertain
    d = _finish(d, "wod")
    d.to_parquet(root("index") / "wod.parquet", index=False)
    return d


def index_nav() -> pd.DataFrame:
    from . import op_adapt_data as D
    sp = splits("navtrain_logs")
    lab = pd.read_parquet(D.root() / "navtrain_labels.parquet").set_index("token")
    parts = []
    for log in sorted(sp):
        f = D.root("navtrain") / f"{log}.npz"
        z = np.load(f)
        ctx = z["ctx"]
        parts.append(pd.DataFrame({"key": log, "slot": np.arange(len(ctx)), "split": sp[log], "file": rel(f), "row": ctx[:, -1],
                                   "ctx": list(ctx), "tc0": z["traffic"][:, 0], "tc1": z["traffic"][:, 1],
                                   "token": z["tokens"].astype(str)}))
    d = pd.concat(parts, ignore_index=True)
    L = lab.reindex(d.token)
    for c in ("ped_corr", "ped_wide", "vru_corr", "vru_wide"):
        d[c] = L[c].fillna(False).to_numpy(bool)
    d["ped_dist"] = L.ped_dist.to_numpy(float)
    d["dist_bin"] = L.dist_bin.fillna(-1).to_numpy()
    d["normal"] = ~d.vru_wide                                   # distillation only (Q3); labels kept for description
    d = _finish(d, "nav")
    d.to_parquet(root("index") / "nav.parquet", index=False)
    return d


def index_p5() -> tuple[pd.DataFrame, pd.DataFrame]:
    import os
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from . import op_adapt_data as D
    from . import p5_exam as E
    parts = []
    for f in sorted(D.root("p5").glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        z = np.load(f)
        t = z["targets"]
        parts.append(pd.DataFrame({"key": f.stem, "slot": t, "name": z["names"].astype(str)[t], "split": "heldout",
                                   "file": rel(f), "row": t, "ctx": list(stride_ctx(t, int(z["stride"]))),
                                   "tc0": z["traffic"][0], "tc1": z["traffic"][1]}))
    d = _finish(pd.concat(parts, ignore_index=True), "p5")
    d.to_parquet(root("index") / "p5.parquet", index=False)
    t, past, fut, obs, null, pairs = E.load()
    o = obs[obs.family.isin(E.PED_FAMILIES)].copy()
    u = pd.Series(d.uid.to_numpy(), index=d.name)
    o["uid_plus"], o["uid_minus"] = u.reindex(o.fn_plus).to_numpy(), u.reindex(o.fn_minus).to_numpy()
    o["px_eq"] = o.factor_px * (F_COSMOS / F_P5_SEG) ** 2
    o["vis500"] = o.px_eq >= BIG_PX
    o.to_parquet(root("index") / "p5_obs.parquet", index=False)
    return d, o


# ---------------------------------------------------------------- loader
class TrunkStore:
    """Cached stage-3 trunks of the index rows `idx` (a DataFrame slice of one or more index parquets).
    batch(i) -> x (B, 9, 1024, 8, 16) fp16, valid (B, 9) bool, tc (B, 2) float32 for the positions i of `idx`."""

    def __init__(self, idx: pd.DataFrame, threads: int = 16):
        import torch
        from concurrent.futures import ThreadPoolExecutor
        files = list(dict.fromkeys(idx.file))

        def load(f):
            p = data_dir() / f
            return np.load(p) if p.suffix == ".npy" else np.load(p)["trunk"]
        with ThreadPoolExecutor(threads) as ex:
            arrs = list(ex.map(load, files))
        base = np.cumsum([0] + [len(a) for a in arrs])[:-1]
        self.T = torch.from_numpy(np.concatenate(arrs)) if len(arrs) > 1 else torch.from_numpy(np.ascontiguousarray(arrs[0]))
        b = pd.Series(base, index=files)[idx.file].to_numpy()[:, None]
        c = np.stack(idx.ctx.to_numpy()).astype(np.int64)
        self.valid = c >= 0
        self.ctx = np.where(self.valid, c + b, (idx.row.to_numpy()[:, None] + b))
        self.tc = idx[["tc0", "tc1"]].to_numpy(np.float32)

    def __len__(self):
        return len(self.ctx)

    def batch(self, i, device="cuda"):
        import torch
        i = np.asarray(i)
        x = self.T[torch.from_numpy(self.ctx[i].ravel())].view(len(i), CTX, *self.T.shape[1:])
        return (x.to(device, non_blocking=True), torch.from_numpy(self.valid[i]).to(device),
                torch.from_numpy(self.tc[i]).to(device))


# ---------------------------------------------------------------- teacher
def desire_block(B: int, d: int, dtype, device):
    """The policy's desire input `_to_copy` (B, 1, 33, 8): 33 blocks of 4 20 Hz steps (max-pooled pulses), the last block
    holding the current step. A rising-edge pulse of desire d at the current step sets block 32, channel d (decision 49:
    the plan of the step that carries the pulse)."""
    import torch
    x = torch.zeros(B, 1, 33, 8, dtype=dtype, device=device)
    if d:
        x[:, :, 32, d] = 1
    return x


def teacher_plans(net, H, valid, tc, action_t=AT):
    """H (B, 9, 32, 512) hidden states of the context, valid (B, 9) -> plan mean, std (B, 3, 33, 15) float32 for the
    desires DESIRES (op, op_L, op_R), numerically the round-1 path (op_adapt._policy) for op."""
    import torch
    from . import op_adapt as A
    H = H * valid[:, :, None, None].to(H.dtype)
    f = A.policy_feeds(net, H, action_t, tc)
    s = net.slices["plan"]
    mu, sd = [], []
    for d in DESIRES:
        f["_to_copy"] = desire_block(len(H), d, net.dtype, H.device)
        o = net.run_batched(f, ["outputs"])["outputs"].reshape(len(H), -1).float()
        mu.append(o[:, s.start:s.start + 495].view(-1, 33, 15))
        sd.append(torch.exp(o[:, s.start + 495:s.start + 990]).view(-1, 33, 15))
    return torch.stack(mu, 1), torch.stack(sd, 1)


def load_teacher(domain: str) -> dict:
    return dict(np.load(root("teacher") / f"{domain}.npz"))


# ---------------------------------------------------------------- offset-start slots (§2.1 v3)
N_OFF = {1: 12000, 0: 6000, 2: 6000}             # straight, left, right (NAVSIM cmd one-hot [left, straight, right, unknown])
SEED_TRAIN, SEED_DEV, N_DEV_TOK = 20260929, 20260930, 250
CORNERS = ((2.0, 0.3), (2.0, -0.3), (-2.0, 0.3), (-2.0, -0.3))
SHARD = 250


def offset_table() -> pd.DataFrame:
    """R2/offset/table.parquet: 24 000 train + 1 000 dev samples; draw order documented in the build doc."""
    from . import navsim_zs as Z
    sp = splits("navtrain_logs")
    nav = load_index("nav")[["uid", "key", "token", "tc0", "tc1"]].rename(columns={"key": "log", "uid": "twin_uid"})
    idx = Z.load_index("navtrain", slim=True)
    meta = pd.DataFrame({"token": [e["token"] for e in idx], "cmd": [int(np.argmax(e["cmd"][-1])) for e in idx],
                         "map": [e["map"] for e in idx]})
    m = nav.merge(meta, on="token")
    m["lsplit"] = m.log.map(sp)
    rng = np.random.default_rng(SEED_TRAIN)
    tr = m[(m.lsplit == "train") & (m.cmd < 3)].sort_values("token")
    pick = np.concatenate([rng.choice(tr.token[tr.cmd == c].to_numpy(), n, replace=False) for c, n in N_OFF.items()])
    t = tr.set_index("token").loc[np.sort(pick)].reset_index()
    n = len(t)
    u_e, u_p = rng.uniform(1.0, 2.0, n), rng.uniform(0.15, 0.30, n)
    s1, s2 = rng.choice([-1.0, 1.0], n), rng.choice([-1.0, 1.0], n)
    t = t.assign(split="train", e=s1 * u_e, psi=s2 * u_p, corner=-1)
    dv = m[m.lsplit == "dev"].sort_values("token")
    dt = np.sort(np.random.default_rng(SEED_DEV).choice(dv.token.to_numpy(), N_DEV_TOK, replace=False))
    dd = dv.set_index("token").loc[dt].reset_index()
    dd = pd.concat([dd.assign(split="dev", e=e, psi=p, corner=c) for c, (e, p) in enumerate(CORNERS)])
    dd = dd.sort_values(["token", "corner"])
    t = pd.concat([t, dd], ignore_index=True)[["token", "log", "split", "cmd", "e", "psi", "corner", "twin_uid", "map", "tc0", "tc1"]]
    t.insert(0, "sid", np.arange(len(t)))
    t.insert(1, "uid", np.int64(UID_BASE["off"]) + t.sid.to_numpy(np.int64))
    t["shard"] = t.sid // SHARD
    t.to_parquet(root("offset") / "table.parquet", index=False)
    (root("offset") / "seeds.json").write_text(json.dumps({"train": SEED_TRAIN, "dev": SEED_DEV, "n": {str(k): v for k, v in N_OFF.items()},
                                                          "dev_tokens": N_DEV_TOK, "corners": CORNERS, "draw_order":
                                                          "strata choice (straight, left, right) -> sort tokens -> u_e, u_psi, s1, s2"}))
    return t


class OffsetMaps:
    """openpilot road / wide model frames of CAM_F0 seen from an ego pose moved by (0, e) m (left positive) and turned by
    psi rad (counter-clockwise), through the road plane z = 0 of the NAVSIM ego frame (plane-induced homography,
    distortion kept). e = psi = 0 reproduces navsim_zs.OpenpilotMaps bit for bit. Same packing (decode / __call__)."""

    def __init__(self, cam: dict, e: float, psi: float):
        from .navsim_zs import NUPLAN_WH, OpenpilotMaps, project_nuplan
        from .openpilot.frames import MEDMODEL_K, SBIGMODEL_K, VIEW_FROM_DEVICE, MODEL_W, MODEL_H
        self.decode, self._pack = OpenpilotMaps.decode, OpenpilotMaps.__call__
        uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
        c, s = math.cos(psi), math.sin(psi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t = np.asarray(cam["t"], np.float64)
        dc = np.array([0.0, e, 0.0]) + Rz @ t - t                   # virtual camera centre minus the real one (old ego frame)
        oz = (Rz @ t)[2]                                             # camera height above the road plane
        self.idx, self.coverage = [], []
        w, h = NUPLAN_WH
        for Km in (MEDMODEL_K, SBIGMODEL_K):
            ray_dev = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(Km @ VIEW_FROM_DEVICE).T
            d = (ray_dev * np.array([1., -1., -1.])) @ Rz.T
            q = d - dc * (d[..., 2:3] / oz) if (e or psi) else d    # ground point seen from the real camera (projective)
            uv, ok, _ = project_nuplan(q, cam, 1)
            xi = np.clip(np.rint(uv[..., 0]), 0, w - 1).astype(np.int64)
            yi = np.clip(np.rint(uv[..., 1]), 0, h - 1).astype(np.int64)
            self.idx.append((yi * w + xi).ravel())
            self.coverage.append(float(ok.mean()))

    def __call__(self, ycc):
        return self._pack(self, ycc)


def offset_frames(entry: dict, e: float, psi: float):
    """One offset sample on the NAVSIM 2 Hz sample-and-hold protocol (scripts/op_adapt_cache.navtrain_job, one token):
    unique image pairs prev, cur (n, 2, 6, 128, 256) and ctx (9,) rows (-1 = zero hidden state)."""
    from . import navsim_zs as Z
    T = np.round(np.arange(-8, 1) * 0.2, 3)
    slot = lambda t: int(np.searchsorted(Z.T_HIST2, t + 1e-6) - 1) if t >= -1.5 - 1e-6 else -1  # noqa: E731
    m = OffsetMaps(entry["cams"][-1]["CAM_F0"], e, psi)
    frames, fi, pairs, ctx = [], {}, {}, []

    def fidx(k):
        if k not in fi:
            fi[k] = len(frames)
            frames.append(m(m.decode(entry["cams"][k]["CAM_F0"]["path"])))
        return fi[k]
    for t in T:
        c = slot(t)
        if c < 0:
            ctx.append(-1)
            continue
        pr = slot(round(t - 0.2, 3))
        pk = (fidx(pr) if pr >= 0 else -1, fidx(c))
        ctx.append(pairs.setdefault(pk, len(pairs)))
    F = np.concatenate([np.zeros((1, 2, 6, 128, 256), np.uint8), np.stack(frames)])
    pk = np.array(list(pairs), np.int64) + 1
    return F[pk[:, 0]], F[pk[:, 1]], np.array(ctx, np.int32), m.coverage


def main():
    """python -m jevdrive.op_adapt_r2_data <step> ... (CPU steps; GPU passes are scripts/op_adapt_r2_cache.py)."""
    import sys
    import time
    steps = {"splits": make_splits, "simlab": sim_labels, "sim": index_sim, "nus": index_nus, "wodval": labels_wod_val,
             "wod": index_wod, "nav": index_nav, "p5": index_p5, "offtab": offset_table}
    for s in sys.argv[1:]:
        t = time.time()
        r = steps[s]()
        d = r[0] if isinstance(r, tuple) else r
        if isinstance(d, pd.DataFrame):
            msg = f"{s}: {len(d)} rows"
            if "split" in d:
                msg += f", split {d.split.value_counts().to_dict()}"
            for c in ("labeled", "ped_corr", "vis", "normal"):
                if c in d:
                    msg += f", {c} {int(d[c].sum())}"
            print(msg, f"({time.time() - t:.0f} s)", flush=True)
        if isinstance(r, tuple) and len(r) > 1 and isinstance(r[1], pd.DataFrame):
            print(f"  second table: {len(r[1])} rows", flush=True)


if __name__ == "__main__":
    main()
