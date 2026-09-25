"""Fusion diagnostics Q2b: on WOD's longitudinal loss frames, is the cause object seen by openpilot?
Pre-registration: todos/2026-09-25-fusion-diagnostics.md (Q2, and the [Q2b] lines of the deviation log).

  corridor   SAM 3.1 detections on the front image (score > 0.5), ground-contact point lifted onto the flat ground
             with the sequence's own WOD calibration; an object is "in the corridor" when its BEV point is within
             1.5 m of the logged 5 s ego path (origin + the 20 future points) and <= 40 m from the ego
  labels     per frame and class group (pedestrian, cyclist, vehicle, cone|debris, emergency vehicle, and "road
             user" = vehicle | pedestrian | cyclist for Interections): any such object in the corridor
  probes     logistic probes, 4-fold cross-fit by sequence on the 19 663-frame subset, one per class group and
             feature (openpilot Cinque `temporal`, Qwen `L18_last`); threshold at 90 % specificity from out-of-fold
             scores inside the training folds (inner 3-fold)
  anatomy    every longitudinal loss frame of Q2a in the six clusters: seen but decided wrong / not seen / no visible
             cause, plus "Qwen sees it, openpilot does not"; shares with sequence-bootstrap CIs
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fusion_q4 as Q
from .common import data_dir, get_logger

log = get_logger(__name__)
HALF_W, RANGE = 1.5, 40.0
GROUPS = {"pedestrian": ("pedestrian",), "cyclist": ("cyclist",), "vehicle": ("vehicle",), "cone|debris": ("cone", "debris"),
          "emergency vehicle": ("emergency vehicle",), "road user": ("vehicle", "pedestrian", "cyclist")}
CLUSTER_GROUP = {"Pedestrian": "pedestrian", "Cyclist": "cyclist", "Cut_ins": "vehicle", "Multi-Lane Maneuvers": "vehicle",
                 "Foreign Object Debris": "cone|debris", "Special Vehicles": "emergency vehicle", "Interections": "road user"}
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "fusion-diagnostics" / "q2b"


def calibs() -> dict:
    out = {}
    for f in (data_dir() / "processed" / "wod_zeroshot" / "op_calib.json",
              data_dir() / "processed" / "drive_backbones" / "op_calib_trainval.json"):
        if f.exists():
            for seq, c in json.loads(f.read_text()).items():
                out.setdefault(seq, c["1"])
    return out


def extend(path: np.ndarray, reach: float = RANGE) -> np.ndarray:
    """Post-hoc sensitivity: the logged path continued along its last heading until it reaches `reach` metres
    from the origin (a stopping ego's 5 s path is a few metres long and cannot contain what it stops for)."""
    seg = np.diff(path, axis=0)
    good = np.linalg.norm(seg, axis=1) > 0.05
    h = seg[good][-1] / np.linalg.norm(seg[good][-1]) if good.any() else np.array([1.0, 0.0])
    end = path[-1]
    left = reach - np.linalg.norm(end)
    return np.r_[path, [end + h * max(left, 0.0)]] if left > 0 else path


def corridor_labels(det_dir: Path, extended: bool = False) -> pd.DataFrame:
    """One row per WOD frame of the list: for each class group, whether a detection lies in the corridor."""
    from . import waymo as W
    from .fusion_diag import project
    lst = pd.read_parquet(Q.root("lists") / "wod.parquet")
    cal = calibs()
    miss = sorted(set(lst.sequence) - set(cal))
    if miss:
        raise RuntimeError(f"{len(miss)} sequences without a front calibration, e.g. {miss[:3]}")
    d = Q.load_dets(det_dir)
    seq = d.key.map(lst.set_index("key").sequence)
    d = Q.lift_dets(d, seq.to_numpy(), cal)
    d = d[d.lift_ok]
    df = W.load_index()
    names = W.frame_names(df)
    at = pd.Series(np.arange(len(df)), index=names)
    _, future = W.load_ego()
    rows = []
    by = d.groupby("key")
    for key in lst.key:
        path = np.r_[[[0.0, 0.0]], future[at[key], :, :2]]
        row = {"frame_name": key, **{f"in_{g}": False for g in GROUPS}}
        if key in by.groups:
            g = by.get_group(key)
            pts = g[["gx", "gy"]].to_numpy(float)
            if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < 0.5:        # standing still: a short stub ahead
                path = np.array([[0.0, 0.0], [2.0, 0.0]])
            if extended:
                path = extend(path)
            _, dist_lat, _, _ = project(path, pts)     # distance to the polyline, ends included
            near = (np.abs(dist_lat) <= HALF_W) & (np.hypot(pts[:, 0], pts[:, 1]) <= RANGE) & (pts[:, 0] > 0)
            hit = set(g.prompt.to_numpy()[near])
            for grp, members in GROUPS.items():
                row[f"in_{grp}"] = bool(hit & set(members))
        rows.append(row)
    return pd.DataFrame(rows)


def probes(lab: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold probe scores and the 90 %-specificity decision per class group and feature."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from . import waymo as W
    from .waymo_ladder import subset_path
    sub = pd.read_parquet(subset_path())
    feats = {}
    for name, (st, arr) in {"op": ("op_cinque_p3", "temporal"), "qwen": ("qwenvid_p3", "L18_last")}.items():
        idx, a = W.load_flat_features(st, [arr])
        feats[name] = pd.Series(np.arange(len(idx)), index=idx.frame_name.to_numpy()), a[arr]
    common = set(lab.frame_name) & set(feats["op"][0].index) & set(feats["qwen"][0].index)
    t = lab[lab.frame_name.isin(common)].merge(sub[["frame_name", "sequence"]], on="frame_name").reset_index(drop=True)
    codes = pd.factorize(t.sequence)[0]
    fold = np.random.default_rng(seed).permutation(codes.max() + 1)[codes] % 4
    out = t[["frame_name"]].copy()
    for name, (pos, arr) in feats.items():
        X = np.asarray(arr[pos.reindex(t.frame_name).to_numpy()], np.float32)
        for grp in GROUPS:
            y = t[f"in_{grp}"].to_numpy()
            score, dec = np.full(len(t), np.nan), np.zeros(len(t), bool)
            for f in range(4):
                tr, te = fold != f, fold == f
                if y[tr].sum() < 5:
                    continue
                inner = np.full(tr.sum(), np.nan)
                ftr = fold[tr]
                for g in np.unique(ftr):        # inner cross-fit on the training folds for the threshold
                    a_, b_ = ftr != g, ftr == g
                    sc = StandardScaler().fit(X[tr][a_])
                    m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr][a_]), y[tr][a_])
                    inner[b_] = m.decision_function(sc.transform(X[tr][b_]))
                thr = np.quantile(inner[~y[tr]], 0.90)
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
                score[te] = m.decision_function(sc.transform(X[te]))
                dec[te] = score[te] > thr
            out[f"{name}_{grp}_score"], out[f"{name}_{grp}_seen"] = score, dec
            log.info("probe %s %s: %d positives of %d", name, grp, int(y.sum()), len(y))
    return out


def anatomy(lab: pd.DataFrame, pr: pd.DataFrame, b: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    q2a = pd.read_csv(Path(__file__).resolve().parents[1] / "research" / "results" / "fusion-diagnostics" / "q2a" /
                      "q2a_frames.csv")
    f = q2a[q2a.loss & q2a.category.str.startswith("longitudinal") & q2a.cluster.isin(list(CLUSTER_GROUP))].copy()
    f = f.merge(lab, on="frame_name", how="left").merge(pr, on="frame_name", how="left")
    grp = f.cluster.map(CLUSTER_GROUP)
    f["q2b_cause_class"] = grp
    f["q2b_cause_object"] = [bool(r[f"in_{g}"]) for (_, r), g in zip(f.iterrows(), grp)]
    f["q2b_probe_op"] = [r[f"op_{g}_score"] for (_, r), g in zip(f.iterrows(), grp)]
    f["q2b_probe_qwen"] = [r[f"qwen_{g}_score"] for (_, r), g in zip(f.iterrows(), grp)]
    op_seen = np.array([bool(r[f"op_{g}_seen"]) for (_, r), g in zip(f.iterrows(), grp)])
    qw_seen = np.array([bool(r[f"qwen_{g}_seen"]) for (_, r), g in zip(f.iterrows(), grp)])
    c = f.q2b_cause_object.to_numpy()
    f["q2b_category"] = np.where(~c, "no visible cause", np.where(op_seen, "seen, decided wrong", "not seen"))
    f["q2b_qwen_not_op"] = c & qw_seen & ~op_seen
    codes, uniq = pd.factorize(f.sequence)
    idx = np.random.default_rng(seed).integers(len(uniq), size=(b, len(uniq)))
    cnt = np.bincount(codes, minlength=len(uniq)).astype(float)
    rows = []
    for scope, m in [("six clusters pooled", np.ones(len(f), bool))] + [(k, (f.cluster == k).to_numpy()) for k in CLUSTER_GROUP]:
        if not m.any():
            continue
        for cat in ("seen, decided wrong", "not seen", "no visible cause", "Qwen sees, openpilot does not"):
            v = (f.q2b_qwen_not_op.to_numpy() if cat.startswith("Qwen") else (f.q2b_category == cat).to_numpy()) & m
            num = np.bincount(codes, v.astype(float), len(uniq))[idx].sum(1)
            den = np.bincount(codes, m.astype(float), len(uniq))[idx].sum(1)
            r = num / np.maximum(den, 1)
            rows.append({"scope": scope, "category": cat, "n_frames": int(m.sum()), "share": v.sum() / m.sum(),
                         "lo": float(np.quantile(r, 0.025)), "hi": float(np.quantile(r, 0.975))})
    return pd.DataFrame(rows), f


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", default=str(data_dir() / "processed" / "fusion_diag" / "sam" / "wod"))
    ap.add_argument("--extended", action="store_true", help="post-hoc sensitivity: corridor continued to 40 m")
    a = ap.parse_args()
    rl = RunLog("fusion_diag", "q2b-extended" if a.extended else "q2b")
    lab = corridor_labels(Path(a.dets), a.extended)
    lab.to_parquet(rl.dir / "corridor_labels.parquet", index=False)
    rl.info("corridor positives: " + str({g: int(lab[f"in_{g}"].sum()) for g in GROUPS}))
    pr = probes(lab)
    pr.to_parquet(rl.dir / "probe_scores.parquet", index=False)
    shares, frames = anatomy(lab, pr)
    shares.to_csv(rl.dir / "q2b_shares.csv", index=False)
    frames.to_csv(rl.dir / "q2b_frames.csv", index=False)
    rl.info("Q2b shares\n" + shares.to_markdown(index=False, floatfmt=".3f"))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
