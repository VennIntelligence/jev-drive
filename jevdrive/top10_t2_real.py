"""Top-10 intersection, executor T2, real-data exams: DrivoR and WA-JEPA zero-shot on WOD-E2E val and nuScenes
(todos/2026-09-26-top10-intersection.md 5.2 / 5.3 and the [T2] entries; WOD input rules registered in
$DATA_DIR/runs/top10_t2/registrations.md before any number).

  fetch-wod   8-camera records of f-15 / f-10 / f-5 for the rater + ADE-extra frames (raw GCS val shards, the
              Alpamayo exam's fetch path); f-3 .. f are already on disk
  req-wod     JPEGs of FRONT / FRONT_LEFT / FRONT_RIGHT / REAR of f-15, f-10, f-5, f -> files, one request file
  exam-wod    RFS (cluster mean, frame mean), ADE vs rater_best and vs the logged future, paired against cv, logged
              future, ours `cls ego`, Alpamayo 1.5 nav and openpilot Cinque; decision 22's ADE on s_ego deciles 1-9
  req-nusc    nuScenes main (4636): the four keyframes t0-1.5 ... t0 of CAM_FRONT_LEFT / FRONT / FRONT_RIGHT / BACK
  exam-nusc   the nuscenes-physicalai exam's metrics (VAD / ST-P3, BEV-Planner) with scene bootstrap, paired vs CV

Request format and runners: jevdrive/top10_t2.py, scripts/top10_t2/{drivor,wajepa}_run.py.
"""
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger
from . import top10_t2 as T

log = get_logger(__name__)
WOD_HIST = (-15, -10, -5, 0)            # 10 Hz frame offsets of the 2 Hz history
WOD_SLOTS = (2, 1, 3, 7)                # request cameras [l0, f0, r0, b0] <- FRONT_LEFT, FRONT, FRONT_RIGHT, REAR
NUSC_SLOTS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK")
NAV_OF_VAD = {"left": 0, "straight": 1, "right": 2}
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "top10-exams"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hist_pose(past: np.ndarray) -> np.ndarray:
    """(n, 4, 3) poses at -1.5 / -1.0 / -0.5 / 0 s from WOD past states (4 Hz, t0 last): x, y and the direction of
    each sample's velocity, held from the next later sample below 0.5 m/s, 0 at t0 (night2_n3.nav_ego's rule)."""
    P = past[:, (9, 11, 13, 15)].astype(np.float64)
    sp, hv = np.linalg.norm(P[..., 2:4], axis=-1), np.arctan2(P[..., 3], P[..., 2])
    h = np.zeros(sp.shape)
    for k in (2, 1, 0):
        h[:, k] = np.where(sp[:, k] >= 0.5, hv[:, k], h[:, k + 1])
    return np.concatenate([P[..., :2], h[..., None]], -1)


# ---------------------------------------------------------------- WOD

def wod_targets() -> dict:
    from . import wod_zeroshot as Z
    S = Z.load_sets()
    return {k: S[k] for k in ("rater", "extra")}


def fetch_wod(workers: int = 16):
    from . import wod_zeroshot as Z
    spans, ordinal = Z.load_spans()
    want, n = {}, 0
    for s in wod_targets().values():
        for name in s["name"]:
            seq, f = str(name).rsplit("-", 1)
            for d in WOD_HIST[:-1]:
                hn = f"{seq}-{int(f) + d:03d}"
                if hn in ordinal and not (Z.root("records") / f"{hn}.pb").exists():
                    want.setdefault(spans[hn][0], {})[ordinal[hn]] = hn
                    n += 1
    log.info("%d records to fetch from %d raw shards", n, len(want))
    import os
    proxy = os.environ.get("https_proxy") or "http://127.0.0.1:7890"
    r = Z.fetch_records(want, "proxy", proxy, workers, log=log.info)
    log.info("fetched %s", r)


def _extract(name: str) -> dict:
    """JPEGs of the four request cameras of one record -> files; {camera id: path}."""
    from . import waymo as W, wod_zeroshot as Z
    d = Z.root("t2_jpg") / name
    out = {c: d / f"{c}.jpg" for c in WOD_SLOTS}
    if all(p.exists() for p in out.values()):
        return {c: str(p) for c, p in out.items()}
    fr = W.e2ed_frame().FromString((Z.root("records") / f"{name}.pb").read_bytes()).frame
    imgs = {im.name: im.image for im in fr.images}
    d.mkdir(parents=True, exist_ok=True)
    for c, p in out.items():
        p.write_bytes(imgs[c])
    return {c: str(p) for c, p in out.items()}


def req_wod(workers: int = 6):
    from concurrent.futures import ProcessPoolExecutor
    from . import wod_zeroshot as Z
    rec = Z.root("records")
    keys, img, pasts, cmds, clamped = [], [], [], [], 0
    jobs = []
    for s in wod_targets().values():
        for name, past, it in zip(s["name"], s["past"], s["intent"]):
            seq, f = str(name).rsplit("-", 1)
            hs = [f"{seq}-{int(f) + d:03d}" for d in WOD_HIST]
            # registered clamp: a missing slot takes the nearest later slot that exists
            for k in range(len(hs) - 2, -1, -1):
                if not (rec / f"{hs[k]}.pb").exists():
                    hs[k] = hs[k + 1]
                    clamped += 1
            jobs.append(hs)
            keys.append(str(name))
            pasts.append(past)
            cmds.append(T.NAV_CMD[int(it)])
    uniq = sorted({h for hs in jobs for h in hs})
    with ProcessPoolExecutor(workers) as ex:
        paths = dict(zip(uniq, ex.map(_extract, uniq, chunksize=16)))
    for hs in jobs:
        img.append([paths[h][c] for h in hs for c in WOD_SLOTS])
    pasts = np.array(pasts)
    T.save_req("wod", keys, img, hist_pose(pasts), T.ego_rows(pasts[:, -1], "drivor"), cmds)
    log.info("WOD: %d frames, %d history slots clamped", len(keys), clamped)


def _decile(names: list) -> np.ndarray:
    """s_ego decile (0-9) with edges over the whole of val, from the file E1 used (elicit_e1.PRIOR_NPZ)."""
    from .elicit_e1 import PRIOR_NPZ
    z = np.load(data_dir() / PRIOR_NPZ)
    s = z["s_ego"]
    edges = np.quantile(s, np.linspace(0, 1, 11))[1:-1]
    at = pd.Series(np.arange(len(z["frame_name"])), index=z["frame_name"].astype(str))
    return np.clip(np.searchsorted(edges, s[at.loc[names].to_numpy()], "right"), 0, 9)


def exam_wod(rl, B: int = 10000):
    from . import waymo as W, wod_zeroshot as Z
    zs = _script("wod_zeroshot")
    S = wod_targets()
    preds = T.load_preds("wod")
    grids = {}
    for m, (keys, traj) in preds.items():
        grids[T.NAME[m]] = dict(zip(keys, T.grid(traj, extrap=True)))
    # ---- rater frames: RFS / ADE, paired
    r = S["rater"]
    names = [str(x) for x in r["name"]]
    past, fut, traj, sc, cl = r["past"], r["future"][..., :2], r["traj"], r["scores"], r["cluster"].astype(str)
    speed, best = W.init_speed(past), traj[np.arange(len(traj)), r["scores"].argmax(1)]
    base = W.baselines(past)
    cands = {"cv": base["cv"], "logged_future": fut, "ours cls ego": zs._ours(names)["ours cls ego"]}
    alp, _ = zs._alp(names, "nav")
    cands["Alpamayo 1.5 nav (E[1 sample])"] = alp
    cands["openpilot Cinque"] = zs._op(names, "cinque")
    for k, g in grids.items():
        cands[k] = np.stack([g[n] for n in names])
    per = {}
    for k, p in cands.items():
        p = p if p.ndim == 4 else p[:, None]
        rfs = np.stack([W.rater_feedback_score(p[:, j], traj, sc, speed) for j in range(p.shape[1])], 1).mean(1)
        per[k] = {"rfs": rfs, "ade5_rater": np.linalg.norm(p - best[:, None], axis=-1).mean(-1).mean(1),
                  "ade5_log": np.linalg.norm(p - fut[:, None], axis=-1).mean(-1).mean(1),
                  "lon5_bias": (p[..., -1, 0] - fut[:, None, -1, 0]).mean(1)}
    rng = np.random.default_rng(0)
    sidx = zs._strat_idx(cl, B, rng)
    fidx = rng.integers(0, len(names), (B, len(names)))
    cmean = lambda x: float(pd.Series(x).groupby(cl).mean().mean())  # noqa: E731
    cboot = lambda x: np.mean([x[g].mean(1) for g in sidx], 0)  # noqa: E731
    rows = []
    for k, q in per.items():
        rb = cboot(q["rfs"])
        row = {"row": k, "n": len(names), "rfs": cmean(q["rfs"]), "rfs_lo": np.percentile(rb, 2.5),
               "rfs_hi": np.percentile(rb, 97.5), "rfs_frame": q["rfs"].mean(),
               "ade5_rater": q["ade5_rater"].mean(), "ade5_log": q["ade5_log"].mean(), "lon5_bias": q["lon5_bias"].mean()}
        for ref in ("cv", "logged_future", "ours cls ego", "Alpamayo 1.5 nav (E[1 sample])", "openpilot Cinque"):
            d = rb - cboot(per[ref]["rfs"])
            row[f"d_rfs_{ref}"] = cmean(q["rfs"]) - cmean(per[ref]["rfs"])
            row[f"d_rfs_{ref}_lo"], row[f"d_rfs_{ref}_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(row)
    rater = pd.DataFrame(rows)
    rater.to_csv(rl.dir / "wod_rater.csv", index=False)
    # ---- decision 22: ADE@5s vs the logged future on s_ego deciles 1-9 (rater + extra pooled), top decile apart
    e = S["extra"]
    en = [str(x) for x in e["name"]]
    allnames = names + en
    fut_all = np.concatenate([fut, e["future"][..., :2]])
    past_all = np.concatenate([past, e["past"]])
    seq = np.array([n.rsplit("-", 1)[0] for n in allnames])
    dec = _decile(allnames)
    ours_all = zs._ours(allnames)["ours cls ego"]
    alp_e, _ = zs._alp(en, "nav")
    op_e = zs._op(en, "cinque")
    ade = {"cv": W.baselines(past_all)["cv"], "ours cls ego": ours_all,
           "openpilot Cinque": np.concatenate([cands["openpilot Cinque"], op_e]) if op_e is not None else None}
    ade["Alpamayo 1.5 nav (E[1 sample])"] = None  # per-sample expectation: ADE of each sample, averaged
    for k, g in grids.items():
        ade[k] = np.stack([g[n] for n in allnames])
    err = {k: np.linalg.norm(p - fut_all, axis=-1).mean(-1) for k, p in ade.items() if p is not None}
    if alp_e is not None:
        a4 = np.concatenate([alp, alp_e])
        err["Alpamayo 1.5 nav (E[1 sample])"] = np.linalg.norm(a4 - fut_all[:, None], axis=-1).mean(-1).mean(1)
    useq, sid = np.unique(seq, return_inverse=True)
    sb = rng.integers(0, len(useq), (B, len(useq)))
    out = []
    for scope, m in (("s_ego deciles 1-9", dec < 9), ("top decile", dec == 9), ("all", np.ones(len(dec), bool))):
        def boot(v):
            s, c = np.bincount(sid[m], v[m], len(useq)), np.bincount(sid[m], None, len(useq))
            return s[sb].sum(1) / np.maximum(c[sb].sum(1), 1)
        for k, v in err.items():
            row = {"scope": scope, "row": k, "n": int(m.sum()), "ade5_log": float(v[m].mean())}
            row["lo"], row["hi"] = np.percentile(boot(v), [2.5, 97.5])
            for ref in ("cv", "ours cls ego", "Alpamayo 1.5 nav (E[1 sample])", "openpilot Cinque"):
                if ref in err:
                    d = v - err[ref]
                    row[f"d_{ref}"] = float(d[m].mean())
                    row[f"d_{ref}_lo"], row[f"d_{ref}_hi"] = np.percentile(boot(d), [2.5, 97.5])
            out.append(row)
    dj = pd.DataFrame(out)
    dj.to_csv(rl.dir / "wod_ade_decile.csv", index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rater.round(4).to_csv(RESULTS / "wod_rater.csv", index=False)
    dj.round(4).to_csv(RESULTS / "wod_ade_sego.csv", index=False)
    pd.set_option("display.width", 250)
    rl.log.info("rater\n%s", rater[["row", "rfs", "rfs_lo", "rfs_hi", "d_rfs_cv", "d_rfs_cv_lo", "d_rfs_cv_hi",
                                   "ade5_rater", "ade5_log", "lon5_bias"]].round(3).to_string())
    rl.log.info("decision 22\n%s", dj[["scope", "row", "n", "ade5_log", "lo", "hi", "d_cv", "d_cv_lo", "d_cv_hi"]].round(3).to_string())


# ---------------------------------------------------------------- nuScenes

def nusc_main():
    from . import nuscenes_zs as Z
    idx = Z.load_index()
    sets = json.loads((Z.index_path().with_name("sets.json")).read_text())
    by = {e["token"]: e for e in idx["samples"]}
    return idx, [by[t] for t in sets["main"]]


def nusc_ego(scene: dict, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """Registered: velocity at t as the backward 0.2 s displacement of the 20 Hz ego (rear-axle) poses, in the t0
    frame; acceleration as (v(t0) - v(t0 - 0.5 s)) / 0.5. Returns [vx, vy, ax, ay] and nothing from the future."""
    from . import nuscenes_zs as Z
    xyz, R = Z.ego_at(scene, [t0 - 7e5, t0 - 5e5, t0 - 2e5, t0])
    R0t = R[-1].T
    p = (xyz - xyz[-1]) @ R0t.T
    v1, v0 = (p[3, :2] - p[2, :2]) / 0.2, (p[1, :2] - p[0, :2]) / 0.2
    return np.r_[v1, (v1 - v0) / 0.5]


def req_nusc():
    from . import nuscenes_zs as Z
    idx, samples = nusc_main()
    kt = {}
    for e in idx["samples"]:
        kt[(e["scene"], e["i"])] = e["t0"]
    root = Z.dataroot()
    keys, img, hist, ego, cmd, miss = [], [], [], [], [], 0
    for e in samples:
        sc = idx["scenes"][e["scene"]]
        ts = [kt[(e["scene"], e["i"] + d)] for d in (-3, -2, -1, 0)]
        row = []
        for t in ts:
            for c in NUSC_SLOTS:
                j = Z.frame_at(sc, c, t)
                p = root / sc["cams"][c]["path"][j]
                miss += not p.exists()
                row.append(str(p))
        xyz, R = Z.ego_at(sc, ts)
        R0t = R[-1].T
        xy = ((xyz - xyz[-1]) @ R0t.T)[:, :2]
        yaw = Z.yaw_of(R0t @ R)
        keys.append(e["token"])
        img.append(row)
        hist.append(np.c_[xy, yaw])
        ego.append(nusc_ego(sc, e["t0"]))
        cmd.append(NAV_OF_VAD[e["cmd"]])
    assert miss == 0, f"{miss} image files missing (CAM_BACK extraction finished?)"
    T.save_req("nusc", keys, img, hist, ego, cmd)


def exam_nusc(rl, B: int = 10000):
    from . import nuscenes_zs as Z
    nz = _script("nusc_zs")
    idx, samples = nusc_main()
    toks = [e["token"] for e in samples]
    src = nz.load_preds(idx)
    preds = {"cv": np.stack([src["cv"][t] for t in toks]), "logged future": np.stack([src["gt"][t] for t in toks])}
    for row, s in (("openpilot small", "op_small_none"), ("openpilot Cinque", "op_cinque_none"),
                   ("openpilot Lebowski", "op_lebowski_none")):
        if s in src and all(t in src[s] for t in toks):
            preds[row] = np.stack([src[s][t] for t in toks])
    t8 = 0.5 * np.arange(1, 9)
    for m, (keys, traj) in T.load_preds("nusc").items():
        assert (keys == np.array(toks)).all()
        preds[T.NAME[m]] = np.stack([Z.to_lidar_point(t8, tr[:, :2], tr[:, 2], idx["scenes"][e["scene"]]["lidar_xyz"],
                                                      e["fut_t"])[0] for tr, e in zip(traj, samples)])
    scene_ids = np.unique([e["scene"] for e in samples], return_inverse=True)[1]
    n_sc = scene_ids.max() + 1
    Bidx = np.random.default_rng(0).integers(0, n_sc, (B, n_sc))
    turn = np.array([e["cmd"] != "straight" for e in samples])
    gt = np.stack([e["gt"] for e in samples])

    def boot(v):
        s, c = np.bincount(scene_ids, v, n_sc), np.bincount(scene_ids, None, n_sc)
        return s[Bidx].sum(1) / c[Bidx].sum(1)
    vals = {}
    for row, p in preds.items():
        h = Z.horizons(Z.per_sample(p.astype(np.float64), samples))
        h["l2_avg"] = (h["l2_1s"] + h["l2_2s"] + h["l2_3s"]) / 3
        for c in ("col_vad", "col_bevp"):
            h[f"{c}_avg"] = (h[f"{c}_1s"] + h[f"{c}_2s"] + h[f"{c}_3s"]) / 3
        h["lon_err_3s"] = p[:, -1, 0] - gt[:, -1, 0]
        vals[row] = h
    rows, cmd_rows = [], []
    for row, h in vals.items():
        r = {"row": row, "n": len(toks), **{k: float(v.mean()) for k, v in h.items()}}
        for k in ("l2_avg", "col_vad_avg", "col_bevp_avg"):
            r[k + "_lo"], r[k + "_hi"] = np.percentile(boot(h[k]), [2.5, 97.5])
            d = h[k] - vals["cv"][k]
            r["d_" + k] = float(d.mean())
            r["d_" + k + "_lo"], r["d_" + k + "_hi"] = np.percentile(boot(d), [2.5, 97.5])
        rows.append(r)
        for grp, m in (("straight", ~turn), ("turn", turn)):
            cmd_rows.append({"row": row, "cmd": grp, "n": int(m.sum()), "l2_avg": float(h["l2_avg"][m].mean()),
                             "col_bevp_avg": float(h["col_bevp_avg"][m].mean())})
    df, dc = pd.DataFrame(rows), pd.DataFrame(cmd_rows)
    df.to_csv(rl.dir / "nusc_main.csv", index=False)
    dc.to_csv(rl.dir / "nusc_by_command.csv", index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(RESULTS / "nuscenes_main.csv", index=False)
    dc.round(4).to_csv(RESULTS / "nuscenes_by_command.csv", index=False)
    pd.set_option("display.width", 250)
    rl.log.info("\n%s", df[["row", "n", "l2_1s", "l2_2s", "l2_3s", "l2_avg", "l2_avg_lo", "l2_avg_hi", "d_l2_avg",
                            "d_l2_avg_lo", "d_l2_avg_hi", "col_vad_avg", "col_bevp_avg", "d_col_bevp_avg",
                            "d_col_bevp_avg_lo", "d_col_bevp_avg_hi", "lon_err_3s"]].round(3).to_string())
    rl.log.info("\n%s", dc.round(3).to_string())


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fetch-wod", "req-wod", "exam-wod", "req-nusc", "exam-nusc"))
    a = ap.parse_args()
    if a.cmd == "fetch-wod":
        fetch_wod()
    elif a.cmd == "req-wod":
        req_wod()
    elif a.cmd == "req-nusc":
        req_nusc()
    else:
        rl = RunLog("top10_t2", a.cmd)
        (exam_wod if a.cmd == "exam-wod" else exam_nusc)(rl)
        rl.close()


if __name__ == "__main__":
    main()
