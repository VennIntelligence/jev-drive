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
                    cal[k] = _lift_fmt(c["K"], c["D"], c["R"], c["t"])
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("lists", "embed", "status", "geom"))
    ap.add_argument("--slices", type=int, default=30)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    a = ap.parse_args()
    if a.step == "lists":
        lists(a.slices)
    elif a.step == "status":
        for n in a.sets:
            log.info("%s: %d / %d images detected, READY %s", n, *slices_done(n), root(n, "READY.json").exists())
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
