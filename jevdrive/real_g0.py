"""Real-data transfer G0 and the shared YOLO26x-seg detections / 64-d embeddings of this round
(todos/2026-09-26-real-data-transfer.md, G0 and deviation-log entries [G0], written before any G0 number).

  lists     image lists of every frame set (current frame x front / front_left / front_right), interleaved by set
            priority into disjoint slices, one per detector process
  (detect)  scripts/real_g0_detect.sh: jevdrive.fastperc detect on each slice, E5's detector config unchanged
  embed     detections -> flat-ground BEV with each frame's own calibration -> agent-legal corridor -> E5's k = 8
            embedding (64 dims) per frame set, plus READY markers; see HANDOFF.md in the output root
  students  E5's students refitted with the E5 code (checked against the stored run), weights and CARLA statistics
            kept, so the students can be applied to foreign rows
  wod / navsim / navsim-table / figs    G0 readouts, E1's functions unchanged

Run on the box (envs/jevdrive; the detector in envs/ultralytics): python -m jevdrive.real_g0 <step>
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
CAMS = ("front", "front_left", "front_right")
NAV_CAMS = {"front": "CAM_F0", "front_left": "CAM_L0", "front_right": "CAM_R0"}
SETS = ("i3", "wod_val", "navtest", "wod_train", "navtrain")       # detection priority order
YOLO = "yolo:yolo26x-seg.pt:640:half"


def root(*p) -> Path:
    d = data_dir() / "processed" / "real_transfer" / "yolo"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


# ---------------------------------------------------------------- frame sets and image lists

def _wod(names: np.ndarray) -> pd.DataFrame:
    from . import waymo as W
    df = W.load_index()
    at = pd.Series(np.arange(len(df)), index=W.frame_names(df)).reindex(names)
    assert at.notna().all(), f"{int(at.isna().sum())} WOD frames not in the index"
    r = df.iloc[at.astype(int).to_numpy()]
    sh = [str(W.shard_dir() / x) for x in r.shard]
    fr = pd.DataFrame({"frame_id": names, "sequence": r.sequence.astype(str).to_numpy(), "row": at.astype(int).to_numpy()})
    li = pd.concat([pd.DataFrame({"key": [f"{n}|{c}" for n in names], "path": "", "shard": sh,
                                  "off": r[f"{c}_off"].to_numpy(), "len": r[f"{c}_len"].to_numpy()}) for c in CAMS])
    return fr, li


def frame_sets() -> dict:
    """{set: (frames, image list)}; frames are one row per frame (frame_id first), lists one row per image."""
    from . import navsim_zs as Z
    from .waymo_ladder import subset_path
    D = data_dir()
    out = {}
    s = pd.read_parquet(subset_path())
    out["wod_val"] = _wod(s.frame_name.to_numpy())
    tr = D / "processed/waymo_e2e/features/qwenvid_train_t4"
    names = pd.concat([pd.read_parquet(sh / "index.parquet").frame_name for sh in sorted(tr.iterdir())
                       if sh.name.startswith("training_") and (sh / "index.parquet").exists()], ignore_index=True)
    out["wod_train"] = _wod(names.to_numpy())
    for split in ("navtest", "navtrain"):
        tok = np.load(D / "runs/navsim_zs/openpilot" / split / "cinque_temporal.npz")["tokens"]
        idx = {e["token"]: e["cams"][-1] for e in Z.load_index(split)}
        fr = pd.DataFrame({"frame_id": tok})
        li = pd.concat([pd.DataFrame({"key": [f"{t}|{c}" for t in tok], "path": [idx[t][NAV_CAMS[c]]["path"] for t in tok]})
                        for c in CAMS])
        out[split] = (fr, li)
    t = pd.read_parquet(D / "processed/hugsim_pairs/index.parquet")
    fr = t[["frame_name", "route_id", "base_id", "world", "role"]].rename(columns={"frame_name": "frame_id"})
    li = []
    for i, c in enumerate(CAMS):
        p = t.files.map(lambda f, i=i: f[4 * i + 3])
        assert p.str.contains(f"/{c}/").all()
        li.append(pd.DataFrame({"key": t.frame_name + f"|{c}", "path": p}))
    out["i3"] = (fr.reset_index(drop=True), pd.concat(li))
    return out


def lists(n_slices: int) -> dict:
    """Write frames.parquet per set and n_slices disjoint detector lists (keys prefixed '<set>:')."""
    fs = frame_sets()
    parts = []
    for name in SETS:
        fr, li = fs[name]
        d = root(name)
        d.mkdir(exist_ok=True)
        fr.to_parquet(d / "frames.parquet", index=False)
        li = li.assign(key=name + ":" + li.key).reset_index(drop=True)
        for c in ("shard", "off", "len"):
            if c not in li:
                li[c] = "" if c == "shard" else 0
        parts.append(li.sample(frac=1.0, random_state=0))          # mix sequences so every slice sees every set
        log.info("%s: %d frames, %d images", name, len(fr), len(li))
    allimg = pd.concat(parts, ignore_index=True)
    sl = root("slices")
    sl.mkdir(exist_ok=True)
    for k in range(n_slices):
        allimg.iloc[k::n_slices].to_parquet(sl / f"slice_{k:02d}.parquet", index=False)
    summ = {n: int((allimg.key.str.split(":").str[0] == n).sum()) for n in SETS}
    (sl / "summary.json").write_text(json.dumps({"slices": n_slices, "images": len(allimg), **summ}, indent=1))
    log.info("%d images in %d slices: %s", len(allimg), n_slices, summ)
    return summ



# ---------------------------------------------------------------- real-data embedding (deviation [G0], registered
# before any G0 number): E5's k = 8 corridor embedding with three operational substitutions --
#   corridor  E5's route centreline -> the ego-history arc: the circle tangent to the current heading through the
#             ego's own position 1 s ago (kappa = 2 y / (x^2 + y^2); straight below 2 m of travel, |kappa| <= 0.1),
#             60 m long, built exactly like E5's route path (vehicle-centre origin, points ahead of it)
#   geometry  each frame's own camera calibration and ground plane (WOD per sequence, NAVSIM per token, I3 per scene)
#   box size  box height / image height -> box height / focal length x (f / H of the P5 rig), so that the same
#             object at the same range gives the same number on every camera (identical to E5 on the P5 / WOD rig)

K_DET, HALF_W, REACH, ROUTE_LEN = 8, 4.0, 40.0, 60.0         # elicit_e5, unchanged
CLS3 = ("pedestrian", "cyclist", "vehicle")
SCORE = 0.25
ARC_T, ARC_MIN, KAPPA_MAX = 1.0, 2.0, 0.1
# NAVSIM's ego origin is the rear axle at wheel-centre height, not on the road: GT vehicle boxes 5-30 m away have their
# bottom at z = -0.36 m (navtrain median, 152k boxes; navtest -0.37), so the flat ground is lifted onto z = -0.36
NAV_GROUND_Z = -0.36
I3_CAM_FWD = 1.73        # HUGSIM ego (front camera) ahead of the nuScenes rear axle (hugsim_zs.rear_offset of the rig)


def p5_f_over_h() -> float:
    c = json.loads((data_dir() / "processed/carla_p5v1_ba/op_plan.json").read_text())["calib"]["1"]
    return float(c["intrinsic"][1]) / float(c["height"])


def arc_path(p1: np.ndarray, x_off: float) -> np.ndarray:
    """Ego-history arc through p1 (native frame, position ARC_T s ago), moved to the vehicle-centre frame (x += x_off)
    and cut like elicit_e5._route_path: origin, then the arc from its first point ahead of the origin, <= ROUTE_LEN."""
    r2 = float(p1 @ p1)
    k = 0.0 if r2 < ARC_MIN ** 2 else float(np.clip(2 * p1[1] / r2, -KAPPA_MAX, KAPPA_MAX))
    s = np.arange(0.0, ROUTE_LEN + 10.0 + 1e-9, 0.5)
    pts = np.c_[s, np.zeros_like(s)] if abs(k) < 1e-6 else np.c_[np.sin(k * s) / k, (1 - np.cos(k * s)) / k]
    pts[:, 0] += x_off
    fwd = pts[:, 0] > 0
    pts = pts[int(fwd.argmax()):] if fwd.any() else pts[-1:]
    path = np.r_[[[0.0, 0.0]], pts]
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path[: max(2, int(np.searchsorted(cum, ROUTE_LEN)) + 1)]


def arc_kappa(p1: np.ndarray) -> np.ndarray:
    r2 = (p1 ** 2).sum(1)
    return np.where(r2 < ARC_MIN ** 2, 0.0, np.clip(2 * p1[:, 1] / np.maximum(r2, 1e-9), -KAPPA_MAX, KAPPA_MAX))


def _embed_frames(args):
    """elicit_e5._embed_group with the path given: (frame index, path, dets (m, 7)) -> (index, 64-d, selected rows)."""
    from .fusion_diag import project
    out = []
    for i, path, d, rows in args:
        e = np.zeros((K_DET, 8), np.float32)
        sel_rows = np.zeros(0, np.int64)
        s_all = d_all = np.zeros(0)
        if len(d):
            s_all, d_all, _, _ = project(path, d[:, 3:5])
            keep = (np.abs(d_all) <= HALF_W) & (s_all > 0) & (s_all <= REACH)
            sel = np.flatnonzero(keep)[np.argsort(s_all[keep], kind="stable")][:K_DET]
            for j, q in enumerate(sel):
                e[j, int(d[q, 0])] = 1.0
                e[j, 3:7] = d[q, 3], d[q, 4], d[q, 5], d[q, 6]
                e[j, 7] = 1.0
            sel_rows = rows[sel]
        out.append((i, e.reshape(-1), rows, s_all, d_all, sel_rows))
    return out


# per-set geometry: calibration of every (frame, camera), ego position ARC_T s ago, native -> vehicle-centre x offset

def _lift_fmt(K, D, R_cv2ego, t) -> dict:
    from .fusion_q4 import OPENCV_TO_WOD
    E = np.eye(4)
    E[:3, :3] = np.asarray(R_cv2ego, np.float64) @ OPENCV_TO_WOD.T
    E[:3, 3] = t
    K = np.asarray(K, np.float64)
    return {"intrinsic": [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *[float(x) for x in D]], "extrinsic": E.ravel().tolist()}


def geometry(name: str, fr: pd.DataFrame) -> tuple:
    """(calibs {key: calib}, cal_key per (frame, cam) as a DataFrame, p1 (n, 2), x_off (n,))."""
    from .fusion_q4 import REAR_AXLE_X
    n = len(fr)
    if name.startswith("wod"):
        from . import waymo as W
        cal = {}
        for f in (data_dir() / "processed/wod_zeroshot/op_calib.json", data_dir() / "processed/drive_backbones/op_calib_trainval.json"):
            if f.exists():
                for seq, c in json.loads(f.read_text()).items():
                    for i, cam in enumerate(CAMS):
                        cal.setdefault(f"{seq}|{cam}", c[str(i + 1)])
        miss = set(fr.sequence) - {k.split("|")[0] for k in cal}
        assert not miss, f"{len(miss)} WOD sequences without calibration"
        keys = pd.DataFrame({cam: fr.sequence + f"|{cam}" for cam in CAMS})
        past, _ = W.load_ego()
        p1 = past[fr.row.to_numpy(), -int(ARC_T / 0.25) - 1, :2].astype(np.float64)
        return cal, keys, p1, np.full(n, REAR_AXLE_X)
    if name.startswith("nav"):
        from . import navsim_zs as Z
        idx = {e["token"]: e for e in Z.load_index(name)}
        cal, keys = {}, {cam: [] for cam in CAMS}
        p1 = np.zeros((n, 2))
        for i, t in enumerate(fr.frame_id):
            e = idx[t]
            p1[i] = e["pose"][1][:2]                     # 4 poses at -1.5, -1.0, -0.5, 0 s
            for cam in CAMS:
                c = e["cams"][-1][NAV_CAMS[cam]]
                k = hash((np.round(c["K"], 4).tobytes(), np.round(c["D"], 5).tobytes(), np.round(c["R"], 5).tobytes(),
                          np.round(c["t"], 4).tobytes()))
                if k not in cal:
                    cal[k] = _lift_fmt(c["K"], c["D"], c["R"], np.asarray(c["t"], np.float64) - [0.0, 0.0, NAV_GROUND_Z])
                keys[cam].append(k)
        assert np.allclose(np.stack([idx[t]["pose"][-1] for t in fr.frame_id[:100]]), 0)
        return cal, pd.DataFrame(keys), p1, np.full(n, REAR_AXLE_X)
    if name == "i3":
        import pickle
        from .hugsim_pairs import scenes_dir
        past = np.load(data_dir() / "processed/hugsim_pairs/past.npy")
        t = pd.read_parquet(data_dir() / "processed/hugsim_pairs/index.parquet")
        assert (t.frame_name.to_numpy() == fr.frame_id.to_numpy()).all()
        cal = {}
        for key in fr.base_id.unique():
            m = json.loads((data_dir() / "processed/hugsim_pairs/scenes" / key / "meta.json").read_text())
            with open(scenes_dir() / m["dataset"] / m["scene"] / "ground_param.pkl", "rb") as f:
                ch = float(pickle.load(f)[1])
            for cam in CAMS:
                c = m["cams"]["CAM_" + cam.upper()]
                c2f = np.asarray(c["c2front"], np.float64)
                from .hugsim_zs import CV2V
                R, tr = CV2V @ c2f[:3, :3], CV2V @ c2f[:3, 3]      # OpenCV camera -> ego M (x fwd, y left, z up)
                tr = tr + np.array([0.0, 0.0, ch])               # ground plane at z = 0: the ego is ch above it
                cal[f"{key}|{cam}"] = _lift_fmt(np.asarray(c["K"])[:3, :3], [0] * 5, R, tr)
        keys = pd.DataFrame({cam: fr.base_id + f"|{cam}" for cam in CAMS})
        p1 = past[:, -int(ARC_T / 0.25) - 1, :2].astype(np.float64)
        return cal, keys, p1, np.full(n, I3_CAM_FWD + REAR_AXLE_X)
    raise ValueError(name)


def load_set_dets(name: str) -> pd.DataFrame:
    """All slices' detections of one set, score > SCORE (fusion_q4.load_dets' rule), E5's three classes."""
    from .fusion_q4 import load_dets
    ds = [load_dets(p, SCORE) for p in sorted(root("dets").glob("s*")) if p.is_dir()]
    d = pd.concat(ds, ignore_index=True)
    d = d[d.key.str.startswith(name + ":") & d.prompt.isin(CLS3)].reset_index(drop=True)
    k = d.key.str.slice(len(name) + 1).str.split("|")
    d["frame_id"], d["cam"] = k.str[0], k.str[1]
    return d


def slices_done(name: str) -> tuple[int, int]:
    """(images of `name` covered by finished part files, images of `name` in all lists)."""
    got = tot = 0
    for f in sorted(root("slices").glob("slice_*.parquet")):
        keys = pd.read_parquet(f, columns=["key"]).key
        mine = keys.str.startswith(name + ":").to_numpy()
        d = root("dets") / f"s{f.stem.split('_')[1]}"
        n_parts = len(list(d.glob("part-*.parquet"))) if d.exists() else 0
        covered = np.zeros(len(keys), bool)
        covered[: n_parts * 2000] = True                          # fastperc.detect: shard = 2000 images, in order
        got += int((mine & covered).sum())
        tot += int(mine.sum())
    return got, tot


def embed_set(name: str, workers: int | None = None) -> dict:
    """<set>/embed.npy (n, 64) aligned with <set>/frames.parquet, <set>/dets.parquet (lifted, with corridor
    coordinates and the embedding rank), <set>/READY.json."""
    from multiprocessing import Pool
    from . import fusion_q4 as Q
    from .common import n_cpus
    got, tot = slices_done(name)
    assert got == tot, f"{name}: detections cover {got} / {tot} images"
    fr = pd.read_parquet(root(name, "frames.parquet"))
    cal, keys, p1, x_off = geometry(name, fr)
    at = pd.Series(np.arange(len(fr)), index=fr.frame_id.to_numpy())
    d = load_set_dets(name)
    fi = at.reindex(d.frame_id).to_numpy()
    assert not np.isnan(fi).any()
    d["fi"] = fi.astype(np.int64)
    ck = np.empty(len(d), object)
    for cam in CAMS:
        m = (d.cam == cam).to_numpy()
        ck[m] = keys[cam].to_numpy()[d.fi.to_numpy()[m]]
    d = Q.lift_dets(d, ck, cal)
    fv = np.array([cal[k]["intrinsic"][1] for k in ck])
    d["h_feat"] = (d.y1 - d.y0).to_numpy() / fv * p5_f_over_h()
    d["xc"], d["yc"] = d.gx.to_numpy() + x_off[d.fi.to_numpy()], d.gy.to_numpy()
    ok = d.lift_ok.to_numpy()
    arr = np.c_[d.prompt.map({c: i for i, c in enumerate(CLS3)}).to_numpy(), np.zeros((len(d), 2)), d.xc, d.yc,
                d.h_feat, d.score]
    rows_ok = np.flatnonzero(ok)
    by = pd.Series(rows_ok).groupby(d.fi.to_numpy()[rows_ok]).indices
    kap = arc_kappa(p1)
    jobs, chunk = [], []
    for i in range(len(fr)):
        r = rows_ok[by[i]] if i in by else np.zeros(0, np.int64)
        chunk.append((i, arc_path(p1[i], x_off[i]), arr[r], r))
        if len(chunk) == 256:
            jobs.append(chunk)
            chunk = []
    jobs.append(chunk)
    E = np.zeros((len(fr), K_DET * 8), np.float32)
    S, Dl, rank = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), -1, np.int64)
    with Pool(workers or min(64, n_cpus())) as p:
        for part in p.imap_unordered(_embed_frames, jobs):
            for i, e, r, s_, d_, sel in part:
                E[i] = e
                S[r], Dl[r] = s_, d_
                rank[sel] = np.arange(len(sel))
    d["s"], d["d"], d["rank"] = S, Dl, rank
    np.save(root(name, "embed.npy"), E)
    np.save(root(name, "kappa.npy"), kap.astype(np.float32))
    cols = ["frame_id", "cam", "prompt", "score", "x0", "y0", "x1", "y1", "area", "cu", "cv", "gx", "gy", "lift_ok",
            "h_feat", "xc", "yc", "s", "d", "rank"]
    d[cols].to_parquet(root(name, "dets.parquet"), index=False)
    m = E.reshape(len(E), K_DET, 8)
    summ = {"set": name, "frames": len(fr), "images": tot, "dets": len(d), "dets_lifted": int(ok.sum()),
            "rows_with_any": float((m[:, :, 7].sum(1) > 0).mean()), "mean_in_corridor": float(m[:, :, 7].sum(1).mean()),
            "rows_with_ped": float((m[:, :, 0].sum(1) > 0).mean()), "kappa_nonzero": float((kap != 0).mean())}
    root(name, "READY.json").write_text(json.dumps(summ, indent=1))
    log.info("%s: %s", name, summ)
    return summ


def load_embed(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """(frames, (n, 64) embedding) of a READY set; the reader G1 / G2 / G3 use."""
    assert root(name, "READY.json").exists(), f"{name} is not READY"
    return pd.read_parquet(root(name, "frames.parquet")), np.load(root(name, "embed.npy"))



# ---------------------------------------------------------------- geometry checks of the embedding (no Delta, no metric)

def _corridor_agreement(E_a: np.ndarray, E_b: np.ndarray) -> dict:
    a, b = E_a.reshape(len(E_a), K_DET, 8), E_b.reshape(len(E_b), K_DET, 8)
    na, nb = a[:, :, 7].sum(1), b[:, :, 7].sum(1)
    pa, pb = a[:, :, 0].sum(1) > 0, b[:, :, 0].sum(1) > 0
    return {"rows": len(a), "identical": float(np.isclose(a, b, atol=1e-5).all((1, 2)).mean()),
            "same_count": float((na == nb).mean()), "same_ped_flag": float((pa == pb).mean()),
            "ped_flag_a": float(pa.mean()), "ped_flag_b": float(pb.mean()), "ped_both_given_a": float((pa & pb).sum() / max(pa.sum(), 1))}


def geom_check(rl) -> dict:
    """(1) P5 v1 BA: E5's route corridor vs the ego-history arc on E5's own detections; (2) WOD val: the arc vs the
    logged 5 s path (extended to 40 m; an oracle, for scale only); (3) navtest: lifted pedestrians vs GT pedestrian boxes;
    (4) I3: lifted vehicles vs the inserted actor in the x+ worlds."""
    from multiprocessing import Pool
    from . import fusion_q4 as Q, p5_exam as E, waymo as W
    from .fusion_q2b import extend
    out = {}
    # (1) P5
    os.environ["P5_SET"] = "carla_p5v1_ba"
    t, past, *_ = E.load()
    sub = sorted(p for p in (data_dir() / "processed/elicit_e5/dets").iterdir() if p.is_dir())
    d = pd.concat([Q.load_dets(x, SCORE) for x in sub], ignore_index=True)
    d = d[d.prompt.isin(CLS3)]
    d = Q.lift_dets(d, d.key.str.split("|").str[1].to_numpy(), Q.p5_calib())
    d = d[d.lift_ok]
    H = pd.read_parquet(sorted(sub[0].glob("part-*.parquet"))[0], columns=["H"]).H.iloc[0]
    arr = np.c_[d.prompt.map({c: i for i, c in enumerate(CLS3)}).to_numpy(), np.zeros((len(d), 2)),
                d.gx.to_numpy() + Q.REAR_AXLE_X, d.gy.to_numpy(), ((d.y1 - d.y0) / H).to_numpy(), d.score.to_numpy()]
    by = pd.Series(np.arange(len(d))).groupby(d.key.str.split("|").str[0].to_numpy()).indices
    p1 = past[:, -int(ARC_T / 0.25) - 1, :2].astype(np.float64)
    jobs = [[(i, arc_path(p1[i], Q.REAR_AXLE_X), arr[by[fn]] if fn in by else np.zeros((0, 7)),
              by.get(fn, np.zeros(0, np.int64))) for i, fn in enumerate(t.frame_name[c0:c0 + 512], c0)]
            for c0 in range(0, len(t), 512)]
    Ea = np.zeros((len(t), 64), np.float32)
    with Pool(48) as pool:
        for part in pool.imap_unordered(_embed_frames, jobs):
            for i, e, *_ in part:
                Ea[i] = e
    Er = np.load(data_dir() / "processed/elicit_e5/embed.npy")
    out["p5_route_vs_arc"] = _corridor_agreement(Er, Ea)
    obs = t.role.to_numpy() == "obs"
    out["p5_route_vs_arc_obs"] = _corridor_agreement(Er[obs], Ea[obs])
    # (2) WOD val: arc vs logged path
    fr = pd.read_parquet(root("wod_val", "frames.parquet"))
    if root("wod_val", "READY.json").exists():
        dd = pd.read_parquet(root("wod_val", "dets.parquet"))
        dd = dd[dd.lift_ok]
        _, fut = W.load_ego()
        arr = np.c_[dd.prompt.map({c: i for i, c in enumerate(CLS3)}).to_numpy(), np.zeros((len(dd), 2)), dd.xc, dd.yc,
                    dd.h_feat, dd.score]
        at = pd.Series(np.arange(len(fr)), index=fr.frame_id)
        by = pd.Series(np.arange(len(dd))).groupby(at[dd.frame_id].to_numpy()).indices
        from .fusion_q4 import REAR_AXLE_X
        jobs, rows = [], fr.row.to_numpy()
        for c0 in range(0, len(fr), 512):
            ch = []
            for i in range(c0, min(c0 + 512, len(fr))):
                pth = extend(np.r_[[[0.0, 0.0]], fut[rows[i], :, :2]], ROUTE_LEN)
                pth = pth + [REAR_AXLE_X, 0.0]
                pth = np.r_[[[0.0, 0.0]], pth[int((pth[:, 0] > 0).argmax()):]]
                ch.append((i, pth, arr[by[i]] if i in by else np.zeros((0, 7)), by.get(i, np.zeros(0, np.int64))))
            jobs.append(ch)
        El = np.zeros((len(fr), 64), np.float32)
        with Pool(48) as pool:
            for part in pool.imap_unordered(_embed_frames, jobs):
                for i, e, *_ in part:
                    El[i] = e
        out["wod_val_logged_vs_arc"] = _corridor_agreement(El, np.load(root("wod_val", "embed.npy")))
    # (3) navtest pedestrians vs GT
    if root("navtest", "READY.json").exists():
        from . import elicit_e3 as E3
        ag = E3.extract("navtest")
        dd = pd.read_parquet(root("navtest", "dets.parquet"))
        dd = dd[dd.lift_ok & (dd.prompt == "pedestrian") & (np.hypot(dd.gx, dd.gy) <= 40)]
        err, rng = [], []
        for tok, g in dd.groupby("frame_id"):
            a = ag[tok]
            ped = a["boxes"][a["cls"] == E3.CLASSES.index("pedestrian")] if len(a["boxes"]) else np.zeros((0, 7))
            if not len(ped):
                continue
            dist = np.linalg.norm(g[["gx", "gy"]].to_numpy()[:, None] - ped[None, :, :2], axis=-1)
            err += list(dist.min(1))
            rng += list(np.hypot(g.gx, g.gy))
        err, rng = np.array(err), np.array(rng)
        out["navtest_ped_vs_gt"] = {f"{lo}-{hi} m": {"n": int(((rng >= lo) & (rng < hi)).sum()),
                                                    "median_err_m": float(np.median(err[(rng >= lo) & (rng < hi)])) if ((rng >= lo) & (rng < hi)).any() else None,
                                                    "share_within_2m": float((err[(rng >= lo) & (rng < hi)] <= 2).mean()) if ((rng >= lo) & (rng < hi)).any() else None}
                                    for lo, hi in ((0, 10), (10, 20), (20, 40))}
    # (4) I3: the inserted car
    from .hugsim_pairs import Logged, ego_frame, unpack
    fr = pd.read_parquet(root("i3", "frames.parquet"))
    dd = pd.read_parquet(root("i3", "dets.parquet"))
    dd = dd[dd.lift_ok & (dd.prompt == "vehicle")]
    t3 = pd.read_parquet(data_dir() / "processed/hugsim_pairs/index.parquet").set_index("frame_name")
    by = dd.groupby("frame_id")
    rows = []
    for key in fr.base_id.unique():
        m = json.loads((data_dir() / "processed/hugsim_pairs/scenes" / key / "meta.json").read_text())
        L = Logged(unpack(m["dataset"], m["scene"]), m["dataset"])
        tsim = np.array(m["t_sim"])
        for w in ("static", "cutin", "oncoming"):
            if not m["worlds"].get(w, {}).get("rendered"):
                continue
            tr = np.array(m["worlds"][w]["track"])
            for fn in fr.frame_id[(fr.base_id == key) & (fr.world == "plus") & (fr.role == "obs")]:
                if fn not in by.groups or f"/{w}/" not in t3.loc[fn].files[3]:
                    continue
                tt = float(t3.loc[fn].t)
                i = int(np.searchsorted(tsim, tt - 1e-6))
                c = ego_frame(L, tt, tr[i:i + 1, :2])[0]
                g = by.get_group(fn)
                dist = np.hypot(g.gx - c[0], g.gy - c[1]).min()
                rows.append({"world": w, "range": float(np.hypot(*c)), "err": float(dist)})
    r = pd.DataFrame(rows)
    if len(r):
        r["bin"] = pd.cut(r.range, [0, 10, 20, 40, 200])
        out["i3_actor_vs_nearest_vehicle"] = {str(k): {"n": len(g), "median_err_m": float(g.err.median()),
                                                       "share_within_3m": float((g.err <= 3).mean())}
                                              for k, g in r.groupby("bin", observed=True)}
    (rl.dir / "geom_check.json").write_text(json.dumps(out, indent=1, default=float))
    rl.info(json.dumps(out, indent=1, default=float))
    return out



# ---------------------------------------------------------------- G0: E5's students zero-shot on WOD and NAVSIM

E5_RUN = "runs/elicitation/e5-fit/20260926-021421"
MODELS, ARMS, SEEDS, K_FOLDS = ("cinque", "lebowski"), ("A", "B"), (0, 1, 2), 5
ACT_HARM = 0.07
NAV_HEADS = "runs/navsim_zs/heads/20260925-232810"


def g0_dir(*p) -> Path:
    d = data_dir() / "processed" / "real_transfer" / "g0"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def students(rl):
    """Refit E5 with elicit_e5.fit unchanged, keep weights and statistics, and check the refit against the stored run:
    identical early-stop steps per fit and max |pred - stored| <= 1e-2 m on the obs rows (deviation [G0] 08:35)."""
    import torch
    from . import elicit_e5 as E5
    os.environ["P5_SET"] = "carla_p5v1_ba"
    keep = {}
    E5.fit(rl, store=keep)
    run = data_dir() / E5_RUN
    ref, new = np.load(run / "preds_obs.npz"), np.load(rl.dir / "preds_obs.npz")
    ev = lambda f: {(e["tag"], e["seed"]): e["best_step"] for e in map(json.loads, open(f)) if e.get("kind") == "e5_fit"}  # noqa: E731
    a, b = ev(run / "events.jsonl"), ev(rl.dir / "events.jsonl")
    steps_same = a == b
    diffs = {k: float(np.nanmax(np.abs(ref[k] - new[k]))) for k in ref.files if k.startswith("E5 ")}
    rep = {"best_steps_identical": steps_same, "n_fits": len(b), "step_mismatch": [str(k) for k in a if a[k] != b.get(k)],
           "max_abs_diff_m": max(diffs.values()), "per_arm": diffs}
    rl.event("g0_student_check", **rep)
    rl.info(f"student refit check: {json.dumps(rep, default=str)}")
    (rl.dir / "student_check.json").write_text(json.dumps(rep, indent=1, default=str))
    assert steps_same and rep["max_abs_diff_m"] <= 1e-2, "E5 students do not reproduce the stored run"
    torch.save(keep, g0_dir("students.pt"))
    return rep


def _mlp_cpu():
    import torch
    from .elicit_e5 import HIDDEN
    return torch.nn.Sequential(torch.nn.Linear(576, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, HIDDEN),
                               torch.nn.GELU(), torch.nn.Linear(HIDDEN, 40))


_KEEP = {}


def student_delta(model: str, arm: str, seed: int, Xop: np.ndarray, Emb: np.ndarray, own: dict | None = None) -> np.ndarray:
    """Mean over the five route-fold students of Delta_s([z_op, e]), (n, 20, 2). Each fold standardises with its own
    CARLA training rows (E5); `own` = {"op_mu", "op_sd", "e_mu", "e_sd"} of another dataset replaces them (descriptive)."""
    import torch
    if not _KEEP:
        _KEEP.update(torch.load(g0_dir("students.pt"), weights_only=False))
    Xo, Ee = torch.as_tensor(Xop, dtype=torch.float32), torch.as_tensor(Emb, dtype=torch.float32)
    mask = torch.as_tensor(np.arange(64) % 8 == 7)
    out = 0
    with torch.no_grad():
        for f in range(K_FOLDS):
            st = own or _KEEP[model, f]
            zo = (Xo - st["op_mu"]) / st["op_sd"] / np.sqrt(Xo.shape[1])
            ze = torch.where(mask, Ee, (Ee - st["e_mu"]) / st["e_sd"]) / np.sqrt(Ee.shape[1])
            net = _mlp_cpu()
            net.load_state_dict(_KEEP[f"{model} f{f} {arm}", seed])
            net.eval()
            out = out + net(torch.cat([zo, ze], 1))
    return (out / K_FOLDS).numpy().reshape(-1, 20, 2)


def _tau(model: str, arm: str, seed: int) -> tuple[float, float]:
    fl = pd.read_csv(data_dir() / E5_RUN / "flip_rates.csv")
    cr = pd.read_csv(data_dir() / E5_RUN / "criteria.csv").set_index("arm")
    k = f"E5 {arm} s{seed} [{model}]"
    return float(fl[(fl.examinee == k) & (fl.scope == "pooled")].tau_model.iloc[0]), float(cr.loc[k, "null_ff_oos"])


def _wod_train_stats(model: str) -> dict:
    """Descriptive: op `temporal` and embedding statistics over the WOD train frames of the shared set."""
    import torch
    from .elicit_e1 import WOD_FEAT, _rows
    fr, Emb = load_embed("wod_train")
    oi = pd.read_parquet(data_dir() / WOD_FEAT / f"op_{model}_p3_trainval/index.parquet").frame_name
    O = np.load(data_dir() / WOD_FEAT / f"op_{model}_p3_trainval/temporal.npy", mmap_mode="r")[_rows(fr.frame_id, oi)]
    O, Emb = torch.as_tensor(np.asarray(O, np.float32)), torch.as_tensor(Emb)
    return {"op_mu": O.mean(0), "op_sd": O.std(0, correction=0).clamp_min(1e-6),
            "e_mu": Emb.mean(0), "e_sd": Emb.std(0, correction=0).clamp_min(1e-6)}


def run_wod(rl):
    """E1's readouts, unchanged, for every student (model x arm x seed) on E1's 19 663 frames."""
    from . import elicit_e1 as E1
    d = E1.wod_frames()
    fr, Emb = load_embed("wod_val")
    Ew = Emb[pd.Series(np.arange(len(fr)), index=fr.frame_id).loc[d["frame_name"]].to_numpy()]
    tabs, acts = [], []
    for m in MODELS:
        own = _wod_train_stats(m)
        for arm in ARMS:
            for seed in SEEDS:
                tau, null_ff = _tau(m, arm, seed)
                for stats in ("carla", "wod-train"):
                    delta = student_delta(m, arm, seed, d[f"op {m}"], Ew, own if stats == "wod-train" else None)
                    tab, act = E1.readouts(d, d[f"prior {m}"], delta, tau)
                    tag = {"model": m, "arm": arm, "seed": seed, "stats": stats}
                    tabs.append(tab.assign(**tag))
                    acts.append(act.assign(**tag, tau=tau, null_ff_oos=null_ff))
                    if stats == "carla":
                        np.savez_compressed(g0_dir(f"wod_val_delta_{m}_{arm}_s{seed}.npz"), frame_name=d["frame_name"], delta=delta)
                    rl.log.info("%s %s s%d %s\n%s", m, arm, seed, stats,
                                tab[tab.judge == "RFS (rater)"].to_markdown(index=False, floatfmt=".3f"))
    pd.concat(tabs).to_csv(rl.dir / "wod_deltas.csv", index=False)
    pd.concat(acts).to_csv(rl.dir / "wod_activation.csv", index=False)


def run_navsim(rl):
    """prior + Delta predictions on navtest for the devkit (E1's mapping onto 0.5 ... 4.0 s, heading kept) and the
    activation table (E1's straight / pedestrian-cyclist scopes)."""
    from . import elicit_e1 as E1, p5_pairs as P
    fr, Emb = load_embed("navtest")
    acts = []
    sc = None
    for m in MODELS:
        z = np.load(data_dir() / "runs/navsim_zs/openpilot/navtest" / f"{m}_temporal.npz")
        tok = z["tokens"]
        assert (tok == fr.frame_id.to_numpy()).all()
        p = np.load(data_dir() / NAV_HEADS / f"navtest_ridge_late_{m}_temporal.npz")
        assert (p["tokens"] == tok).all()
        if sc is None:
            sc = E1.nav_scopes(tok)
            sc.to_csv(rl.dir / "navtest_scopes.csv", index=False)
        for arm in ARMS:
            for seed in SEEDS:
                tau, _ = _tau(m, arm, seed)
                delta = student_delta(m, arm, seed, z["temporal"].astype(np.float32), Emb)
                np.savez_compressed(g0_dir(f"navtest_delta_{m}_{arm}_s{seed}.npz"), tokens=tok, delta=delta)
                arm_p = p["poses"].copy()
                arm_p[..., :2] += delta[:, 1:16:2]
                np.savez(rl.dir / f"navtest_g0_{arm}_s{seed}_{m}.npz", tokens=tok, poses=arm_p.astype(np.float32))
                act = (np.abs(P.v2(E1._grid20(arm_p)) - P.v2(E1._grid20(p["poses"]))) >= tau).astype(float)
                mag = np.linalg.norm(delta[:, 1:16:2], axis=-1).mean(-1)
                for name, msk in (("all", np.ones(len(tok), bool)), ("straight", sc.straight.to_numpy()),
                                  ("ped_cyc_corridor", sc.ped_cyc_corridor.to_numpy()), ("no ped_cyc", ~sc.ped_cyc_corridor.to_numpy())):
                    acts.append({"model": m, "arm": arm, "seed": seed, "scope": name, "n": int(msk.sum()), "tau": tau,
                                 "activation": float(act[msk].mean()), "delta_mag_median_m": float(np.median(mag[msk]))})
    a = pd.DataFrame(acts)
    a.to_csv(rl.dir / "navsim_activation.csv", index=False)
    rl.log.info("activation\n%s", a.to_markdown(index=False, floatfmt=".3f"))


def navsim_table(rl, run_dir):
    """PDMS (v1.1) / EPDMS (v2) of prior + Delta against the stored prior scores, paired, token bootstrap (E1's)."""
    from .elicit_e1 import _devkit
    run_dir = Path(run_dir)
    sc = pd.read_csv(run_dir / "navtest_scopes.csv").set_index("token")
    groups = {"all": None, "ped_cyc_corridor": sc.ped_cyc_corridor, "no ped_cyc": ~sc.ped_cyc_corridor, "straight": sc.straight}
    rows = []
    rng = np.random.default_rng(0)
    f = lambda df: df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)].set_index("token")["score"].astype(float)  # noqa: E731
    for m in MODELS:
        for arm in ARMS:
            for seed in SEEDS:
                for ver, metric in (("v1", "PDMS"), ("v2", "EPDMS")):
                    a = _devkit(ver, "navtest", f"g0_{arm}_s{seed}_{m}_plus_student")
                    b = _devkit(ver, "navtest", f"heads_ridge_late_{m}_temporal")
                    if a is None or b is None:
                        rl.log.warning("missing scores: %s %s %s s%d", ver, m, arm, seed)
                        continue
                    x, y = f(a).align(f(b), join="inner")
                    for g, msk in groups.items():
                        keep = np.ones(len(x), bool) if msk is None else msk.reindex(x.index).fillna(False).to_numpy(bool)
                        dd = (x - y).to_numpy()[keep]
                        bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
                        rows.append({"model": m, "arm": arm, "seed": seed, "metric": metric, "group": g, "n": len(dd),
                                     "prior_score": 100 * y.to_numpy()[keep].mean(), "plus_student": 100 * x.to_numpy()[keep].mean(),
                                     "delta": 100 * dd.mean(), "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
    t = pd.DataFrame(rows)
    t.to_csv(rl.dir / "navsim_paired.csv", index=False)
    rl.log.info("navtest\n%s", t.to_markdown(index=False, floatfmt=".2f"))
    return t


WOD_SCOPES = ("all", "Pedestrians", "Cyclists", "Cut_ins", "FOD", "Intersections")
NAV_GROUPS = ("all", "ped_cyc_corridor", "no ped_cyc", "straight")


def verdicts(wod_run, nav_run, nav_table_run, out: Path) -> pd.DataFrame:
    """The registered G0 cells per model x arm x seed (deviation [G0], written before any G0 number)."""
    wt = pd.read_csv(Path(wod_run) / "wod_deltas.csv")
    wa = pd.read_csv(Path(wod_run) / "wod_activation.csv")
    na = pd.read_csv(Path(nav_run) / "navsim_activation.csv")
    nt = pd.read_csv(Path(nav_table_run) / "navsim_paired.csv")
    rows = []
    for m in MODELS:
        for arm in ARMS:
            for seed in SEEDS:
                sel = lambda t: t[(t.model == m) & (t.arm == arm) & (t.seed == seed)]  # noqa: E731
                w = sel(wt)
                w = w[(w.stats == "carla") & (w.judge == "RFS (rater)")].set_index("scope").loc[list(WOD_SCOPES)]
                a_w = float(sel(wa)[(sel(wa).stats == "carla")].set_index("scope").loc["straight_yaw", "activation"])
                n = sel(nt)
                n = n[n.metric == "PDMS"].set_index("group").loc[list(NAV_GROUPS)]
                a_n = float(sel(na).set_index("scope").loc["straight", "activation"])
                harm = w.loc["all", "hi"] < 0 or n.loc["all", "hi"] < 0 or a_w > ACT_HARM or a_n > ACT_HARM
                useful = (not harm) and (w.loc["Pedestrians", "lo"] > 0 or n.loc["ped_cyc_corridor", "lo"] > 0)
                cross = ((w.lo <= 0) & (w.hi >= 0)).all() and ((n.lo <= 0) & (n.hi >= 0)).all()
                cell = "harmful" if harm else "useful" if useful else "harmless, not useful" if cross else "none of the three cells"
                rows.append({"model": m, "arm": arm, "seed": seed, "cell": cell,
                             "wod_rfs_all": w.loc["all", "delta"], "wod_rfs_all_lo": w.loc["all", "lo"], "wod_rfs_all_hi": w.loc["all", "hi"],
                             "wod_rfs_ped": w.loc["Pedestrians", "delta"], "wod_rfs_ped_lo": w.loc["Pedestrians", "lo"],
                             "wod_rfs_ped_hi": w.loc["Pedestrians", "hi"], "wod_act_straight": a_w,
                             "nav_pdms_all": n.loc["all", "delta"], "nav_pdms_all_lo": n.loc["all", "lo"], "nav_pdms_all_hi": n.loc["all", "hi"],
                             "nav_pdms_ped": n.loc["ped_cyc_corridor", "delta"], "nav_pdms_ped_lo": n.loc["ped_cyc_corridor", "lo"],
                             "nav_pdms_ped_hi": n.loc["ped_cyc_corridor", "hi"], "nav_act_straight": a_n})
    v = pd.DataFrame(rows)
    v.to_csv(out / "verdict.csv", index=False)
    return v




def run_i3(rl):
    """Student Delta on every I3 index row (openpilot `op_streams`, as elicit_i3), stored for G1 / G2; no readout."""
    from . import p5_openpilot
    fr, Emb = load_embed("i3")
    os.environ["P5_SET"] = "hugsim_pairs"
    t = pd.DataFrame({"frame_name": fr.frame_id})
    for m in MODELS:
        Xop = p5_openpilot.load(t, (m,), sub="op_streams")[f"op-{m} temporal"]
        for arm in ARMS:
            for seed in SEEDS:
                np.savez_compressed(g0_dir(f"i3_delta_{m}_{arm}_s{seed}.npz"), frame_name=fr.frame_id.to_numpy(),
                                    delta=student_delta(m, arm, seed, Xop, Emb))
    rl.info("I3 student deltas written")


def figs(res_dir="research/results/real-data-transfer/g0", out_dir="research/figs"):
    """WOD RFS delta per cluster, NAVSIM PDMS delta per group, straight / pedestrian activation; students A / B x models,
    seed 0 with CI and seeds 1-2 as crosses."""
    import matplotlib.pyplot as plt
    from . import plots
    res_dir, out_dir = Path(res_dir), Path(out_dir)
    wt, wa = pd.read_csv(res_dir / "wod_deltas.csv"), pd.read_csv(res_dir / "wod_activation.csv")
    nt, na = pd.read_csv(res_dir / "navsim_paired.csv"), pd.read_csv(res_dir / "navsim_activation.csv")
    wt, wa = wt[(wt.stats == "carla") & (wt.judge == "RFS (rater)")], wa[wa.stats == "carla"]
    nt = nt[nt.metric == "PDMS"]
    series = [(m, a, c, mk) for m, c in (("cinque", plots.OKABE_ITO[5]), ("lebowski", plots.OKABE_ITO[6]))
              for a, mk in (("A", "o"), ("B", "s"))]
    panels = [(wt, "scope", list(WOD_SCOPES), ["All", "Ped.", "Cyc.", "Cut-in", "FOD", "Inters."], r"WOD $\Delta$RFS"),
              (nt, "group", list(NAV_GROUPS), ["All", "Ped./cyc.", "Other", "Straight"], r"navtest $\Delta$PDMS")]
    with plots.mpl.rc_context(plots.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(plots.PAGE, 2.0), gridspec_kw={"width_ratios": [1.3, 1, 0.9]})
        for ax, (t, col, keys, labels, ylab) in zip(axes, panels):
            for k, (m, a, c, mk) in enumerate(series):
                x = np.arange(len(keys)) + (k - 1.5) * 0.17
                r = t[(t.model == m) & (t.arm == a) & (t.seed == 0)].set_index(col).loc[keys]
                ax.errorbar(x, r.delta, yerr=[r.delta - r.lo, r.hi - r.delta], fmt=mk, color=c, ms=3, lw=0.8, capsize=1.5,
                            mfc=c if a == "A" else "white", label=f"{m.capitalize()}, student {a}")
                for sd in (1, 2):
                    r = t[(t.model == m) & (t.arm == a) & (t.seed == sd)].set_index(col).loc[keys]
                    ax.plot(x + 0.06, r.delta, "x", color=c, ms=2.5, mew=0.6)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xticks(np.arange(len(keys)), labels)
            ax.set_ylabel(ylab)
        ax = axes[2]
        acts = [("WOD\nstraight", wa, "straight_yaw"), ("WOD\nped.", wa, "Pedestrians"), ("nav.\nstraight", na, "straight"),
                ("nav.\nped./cyc.", na, "ped_cyc_corridor")]
        for k, (m, a, c, mk) in enumerate(series):
            x = np.arange(len(acts)) + (k - 1.5) * 0.17
            for sd in SEEDS:
                y = [100 * float(t[(t.model == m) & (t.arm == a) & (t.seed == sd) & (t.scope == sc)].activation.iloc[0])
                     for _, t, sc in acts]
                ax.plot(x + (0.06 if sd else 0), y, mk if sd == 0 else "x", color=c, ms=3 if sd == 0 else 2.5,
                        mfc=(c if a == "A" else "white") if sd == 0 else None, mew=0.6, ls="none")
        ax.axhline(100 * ACT_HARM, color="0.3", ls="--", lw=0.7)
        ax.set_xticks(np.arange(len(acts)), [n for n, *_ in acts])
        ax.set_ylim(0, 8)
        ax.set_ylabel("Activation rate (%)")
        h, lab = axes[0].get_legend_handles_labels()       # above the panels: the two-line ticks sit where legend_below goes
        fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, columnspacing=1.4, handlelength=1.8)
        plots.save(fig, out_dir, "real-g0-student-transfer")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("lists", "embed", "status", "geom", "students", "wod", "navsim", "navsim-table", "verdict", "figs", "i3"))
    ap.add_argument("--slices", type=int, default=30)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--runs", nargs="*", default=[], help="navsim-table: the g0-navsim run; verdict: wod, navsim, navsim-table runs")
    a = ap.parse_args()
    if a.step == "lists":
        lists(a.slices)
    elif a.step == "status":
        for n in a.sets:
            log.info("%s: %d / %d images detected, READY %s", n, *slices_done(n), root(n, "READY.json").exists())
    elif a.step == "figs":
        figs()
    elif a.step in ("students", "wod", "navsim", "navsim-table", "verdict", "figs", "i3"):
        from .runlog import RunLog
        rl = RunLog("real-data-transfer", f"g0-{a.step}")
        if a.step == "navsim-table":
            navsim_table(rl, a.runs[0])
        elif a.step == "verdict":
            rl.info(verdicts(*a.runs, rl.dir).to_markdown(index=False, floatfmt=".3f"))
        else:
            {"students": students, "wod": run_wod, "navsim": run_navsim, "i3": run_i3}[a.step](rl)
        rl.close()
    elif a.step == "geom":
        from .runlog import RunLog
        rl = RunLog("real-data-transfer", "g0-geom")
        geom_check(rl)
        rl.close()
    else:
        for n in a.sets:
            embed_set(n)


if __name__ == "__main__":
    main()
