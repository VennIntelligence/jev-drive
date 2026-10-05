"""Night-gap audit step 2: shipped openpilot (Cinque, existing serving preds) day vs night, open loop, CPU only.
  WOD-E2E val: the 1 437 frames with logged futures (479 rater + 958 extra, preds/op_cinque, the unified WOD interface). Night label = luma of the
    sequence (median front-camera mean luma over frames 100..230, ng_lum.py): night < NIGHT_T, dusk < DAY_T, day otherwise (WOD E2E shards carry no
    time_of_day). ADE/FDE at 3 s and 5 s, 3 s heading (chord angle, moving frames), RFS (rater frames).
  nuScenes val 'main' set: night = scene description contains the word night; L2 at 1/2/3 s.
  Controls: constant-velocity/arc baseline (jevdrive.waymo.baselines; nuScenes preds/cv.npz) on the same frames, so the gap of the SCENES (kinematics)
    can be told from the gap of the MODEL. Matched gap = direct standardisation of day onto the night (speed bin x turn bin) mix. All CIs are cluster
    bootstraps over sequences / scenes (B = 4000).
  $DATA_DIR/envs/jevdrive/bin/python experiments/leaderboard_audit/scripts/ng_analyze.py   -> $DATA_DIR/runs/night_gap/{wod_frames.csv,nusc_frames.csv,summary.json}"""
import json, os, pickle, re, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import waymo as W, wod_zeroshot as Z  # noqa: E402

D = Path(os.environ["DATA_DIR"])
OUT = D / "runs/night_gap"
NIGHT_T, DAY_T, B = 50.0, 120.0, 4000
rng = np.random.default_rng(0)


def chord(xy, k):
    return np.arctan2(xy[:, k, 1], xy[:, k, 0])


def wod_frames():
    S = Z.load_sets()
    lum = pd.read_parquet(OUT / "lum.parquet").groupby("sequence").luma.median()
    rows = []
    for st in ("rater", "extra"):
        s = S[st]
        names = s["name"].astype(str)
        pred = np.stack([np.load(Z.root("preds", "op_cinque") / f"{n}.npz")["wod"] for n in names]).astype(np.float64)
        gt = s["future"][..., :2].astype(np.float64)
        cv = W.baselines(s["past"])["cv"].astype(np.float64)
        v0 = W.init_speed(s["past"])
        o = pd.DataFrame(dict(set=st, name=names, seq=s["sequence"].astype(str), cluster=s["cluster"].astype(str), intent=s["intent"], v0=v0))
        for tag, p in (("op", pred), ("cv", cv)):
            e = np.linalg.norm(p - gt, axis=-1)
            o[f"{tag}_ade3"], o[f"{tag}_fde3"] = e[:, :12].mean(1), e[:, 11]
            o[f"{tag}_ade5"], o[f"{tag}_fde5"] = e.mean(1), e[:, 19]
            o[f"{tag}_lat3"] = np.abs(p[:, 11, 1] - gt[:, 11, 1])
            o[f"{tag}_lng3"] = p[:, 11, 0] - gt[:, 11, 0]                       # signed: > 0 = plans farther than the logged car went
            dh = np.degrees(np.angle(np.exp(1j * (chord(p, 11) - chord(gt, 11)))))
            o[f"{tag}_dhead3"] = np.where(np.linalg.norm(gt[:, 11], axis=-1) > 3.0, np.abs(dh), np.nan)
        o["gt_d3"] = np.linalg.norm(gt[:, 11], axis=-1)
        o["gt_ang3"] = np.degrees(chord(gt, 11))
        if st == "rater":
            o["op_rfs"] = W.rater_feedback_score(pred, s["traj"], s["scores"], v0)
            o["cv_rfs"] = W.rater_feedback_score(cv, s["traj"], s["scores"], v0)
        rows.append(o)
    o = pd.concat(rows, ignore_index=True)
    o["luma"] = o.seq.map(lum)
    o["lab"] = np.where(o.luma < NIGHT_T, "night", np.where(o.luma < DAY_T, "dusk", "day"))
    o["vbin"] = pd.cut(o.v0, [-1, 2, 6, 12, 99], labels=False)
    o["tbin"] = np.where(o.gt_d3 < 2, 0, np.where(np.abs(o.gt_ang3) < 8, 1, 2))     # stopped / straight / turning
    return o


def nusc_frames():
    ix = pickle.load(open(D / "processed/nusc_zs/val_index.pkl", "rb"))
    main = set(json.load(open(D / "processed/nusc_zs/sets.json"))["main"])
    sc = {x["name"]: x["description"] for x in json.load(open(D / "datasets/nuscenes/v1.0-trainval/scene.json"))}
    P = {t: i for i, t in enumerate(np.load(D / "runs/nusc_zs/preds/op_cinque_none.npz")["tokens"])}
    op = np.load(D / "runs/nusc_zs/preds/op_cinque_none.npz")["pred"].astype(np.float64)
    cvz = np.load(D / "runs/nusc_zs/preds/cv.npz")
    CP = {t: i for i, t in enumerate(cvz["tokens"])}
    rows = []
    for e in ix["samples"]:
        t = e["token"]
        if t not in main or t not in P or t not in CP:
            continue
        gt = np.asarray(e["gt"], np.float64)
        d = sc[e["scene"]].lower()
        r = dict(token=t, scene=e["scene"], night=bool(re.search(r"\bnight\b", d)), rain=bool(re.search(r"\brain", d)), gt_d3=np.linalg.norm(gt[-1]),
                 gt_ang3=np.degrees(np.arctan2(gt[-1, 1], gt[-1, 0])))
        for tag, p in (("op", op[P[t]]), ("cv", cvz["pred"][CP[t]].astype(np.float64))):
            err = np.linalg.norm(p - gt, axis=-1)
            r[f"{tag}_l2_1"], r[f"{tag}_l2_2"], r[f"{tag}_l2_3"] = err[1], err[3], err[5]       # at the step (UniAD style)
            r[f"{tag}_l2avg"] = err[[1, 3, 5]].mean()
        rows.append(r)
    o = pd.DataFrame(rows)
    o["lab"] = np.where(o.night, "night", "day")
    o["vbin"] = pd.cut(o.gt_d3 / 3.0, [-1, 2, 6, 12, 99], labels=False)                   # mean speed over the 3 s future as the speed proxy
    o["tbin"] = np.where(o.gt_d3 < 2, 0, np.where(np.abs(o.gt_ang3) < 8, 1, 2))
    return o


def gap(o, col, a="night", b="day", matched=False, idx=None):
    """mean(a) - mean(b); matched: b re-weighted to a's (vbin, tbin) mix, cells without b dropped from both."""
    x = o[o.lab == a].dropna(subset=[col]) if idx is None else o.loc[idx][lambda d: d.lab == a].dropna(subset=[col])
    y = o[o.lab == b].dropna(subset=[col]) if idx is None else o.loc[idx][lambda d: d.lab == b].dropna(subset=[col])
    if not matched:
        return x[col].mean() - y[col].mean()
    cx, cy = x.groupby(["vbin", "tbin"])[col].agg(["mean", "size"]), y.groupby(["vbin", "tbin"])[col].mean()
    c = cx.join(cy.rename("m_b"), how="inner")
    return float((c["mean"] * c["size"]).sum() / c["size"].sum() - (c["m_b"] * c["size"]).sum() / c["size"].sum())


def boot(o, cluster, fn):
    """fn(dataframe) -> float or dict; resample clusters with replacement."""
    ids = o[cluster].to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        out.append(fn(o.iloc[np.concatenate([groups[k] for k in pick])]))
    return out


def summarise(o, cluster, metrics, a="night", b="day"):
    res = {}
    for m in metrics:
        row = {"n_a": int((o.lab == a).sum()), "n_b": int((o.lab == b).sum()), "mean_a": float(o[o.lab == a][m].mean()), "mean_b": float(o[o.lab == b][m].mean())}
        for mt in (False, True):
            f = lambda d, mt=mt: gap(d.reset_index(drop=True), m, a, b, mt)  # noqa: E731
            pt = f(o.reset_index(drop=True))
            bs = np.array([v for v in boot(o, cluster, f) if np.isfinite(v)])
            row["matched" if mt else "raw"] = [float(pt), *map(float, np.percentile(bs, [2.5, 97.5]))]
        res[m] = row
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = wod_frames()
    w.to_csv(OUT / "wod_frames.csv", index=False)
    n = nusc_frames()
    n.to_csv(OUT / "nusc_frames.csv", index=False)
    S = {"wod_counts": w.groupby(["set", "lab"]).size().unstack().fillna(0).astype(int).to_dict(), "wod_seqs": w.groupby("lab").seq.nunique().to_dict(),
         "wod_cluster": w.groupby(["cluster", "lab"]).size().unstack().fillna(0).astype(int).to_dict(),
         "wod_mix": w.groupby("lab").agg(v0=("v0", "mean"), moving=("gt_d3", lambda x: (x >= 2).mean()), turn=("tbin", lambda x: (x == 2).mean())).to_dict()}
    wm = ["op_ade3", "op_fde3", "op_ade5", "op_fde5", "op_lat3", "op_dhead3", "cv_ade3", "cv_fde3", "cv_ade5", "cv_lat3", "cv_dhead3"]
    for ab in (("night", "day"), ("dusk", "day")):
        S[f"wod_{ab[0]}_vs_{ab[1]}"] = summarise(w, "seq", wm, *ab)
    r = w[w.set == "rater"].reset_index(drop=True)
    S["wod_rfs_night_vs_day"] = summarise(r, "seq", ["op_rfs", "cv_rfs"])
    S["wod_rfs_dusk_vs_day"] = summarise(r, "seq", ["op_rfs", "cv_rfs"], "dusk", "day")
    # difference in differences: how much of the night gap of the model is already the gap of a kinematic baseline on the same frames
    for m in ("ade3", "fde3", "ade5"):
        w[f"d_{m}"] = w[f"op_{m}"] - w[f"cv_{m}"]
    S["wod_op_minus_cv"] = summarise(w, "seq", ["d_ade3", "d_fde3", "d_ade5"])
    # sensitivity to the luma cut
    sens = {}
    for t in (35, 50, 65):
        w2 = w.copy(); w2["lab"] = np.where(w2.luma < t, "night", np.where(w2.luma >= DAY_T, "day", "dusk"))
        sens[str(t)] = {"n_night": int((w2.lab == "night").sum()), "ade3_gap": float(gap(w2, "op_ade3")), "ade3_gap_matched": float(gap(w2, "op_ade3", matched=True))}
    S["wod_luma_cut_sensitivity"] = sens
    # per cluster night share and per-cluster night-day op ade3 for clusters with >= 15 frames in each
    cl = {}
    for c, g in w.groupby("cluster"):
        a, b = g[g.lab == "night"], g[g.lab == "day"]
        if len(a) >= 10 and len(b) >= 10:
            cl[c] = {"n_night": len(a), "n_day": len(b), "op_ade3_night": float(a.op_ade3.mean()), "op_ade3_day": float(b.op_ade3.mean()), "cv_ade3_night": float(a.cv_ade3.mean()), "cv_ade3_day": float(b.cv_ade3.mean())}
    S["wod_by_cluster"] = cl
    S["nusc_counts"] = n.groupby("lab").agg(n=("token", "size"), scenes=("scene", "nunique")).to_dict()
    nm = ["op_l2_1", "op_l2_2", "op_l2_3", "op_l2avg", "cv_l2_1", "cv_l2_2", "cv_l2_3", "cv_l2avg"]
    S["nusc_night_vs_day"] = summarise(n, "scene", nm)
    n["d_l2avg"] = n.op_l2avg - n.cv_l2avg
    S["nusc_op_minus_cv"] = summarise(n, "scene", ["d_l2avg"])
    json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in S.items() if k in ("wod_counts", "wod_seqs", "wod_mix", "nusc_counts")}, indent=1, default=float))


if __name__ == "__main__":
    main()
