"""Fusion diagnostics, stage 0 (CPU only): Q3, Q7, Q2a and the GT half of Q6.
Pre-registration: todos/2026-09-25-fusion-diagnostics.md (the item sections and the deviation log).

  q3     real-data mirror: the decision-40 (iii) per-frame predictions sliced by WOD cluster, zero fit.
         cls_late - cls ego (ADE, s_ego deciles 1-9, all val; RFS on the rater frames) and openpilot's native plan
         - cv (RFS on the rater frames; ADE on the 20 237-frame subset), sequence-bootstrap CIs
  q2a    WOD loss anatomy (CPU part): rater frames where Cinque's native plan or Cinque cls_late scores
         <= max rater score - 1, classified by direction (longitudinal stop / go, lateral) and, for lateral, by
         whether the routing intent explains the branch. The per-frame CSV has empty Q2b columns to fill later
  q7     mode vocabulary: GMMs (k = 1..8 by BIC) on P5 reactive-frame expert x+ - x- descriptors and on WOD
         rater_best - log descriptors, clusters named by a fixed rule and mapped to openpilot desires
  q6gt   three geometric rule gates on GT state, examined by `p5_exam.exam` unchanged

Q6 gate interface (shared with the later SAM-state run, Q6-SAM)
----------------------------------------------------------------
`gates(states, ego)` takes two tables and returns one row per (run, k) of `ego`:

  states   one row per object and frame: run (str, e.g. the P5 variant id or attempt), k (int, 20 Hz tick),
           obj (object / track id), cls ("vehicle" | "pedestrian" | other), x, y (m), vx, vy (m/s).
           BEV in the ego frame of that frame: origin at the ego vehicle's location, +x forward, +y left.
           Velocities are ground-truth (ego-motion compensated) velocities expressed in the same axes, not
           velocities relative to the moving ego.
  ego      one row per frame: run, k, v0 (ego speed, m/s), front (ego front overhang from its origin, m), and
           path: (m, 2) float array, the route centreline ahead in the same ego frame, starting at (0, 0).

GT states come from `actors.npz` + `pose.jsonl` + `route.json` (`gt_run`); a SAM run only has to produce the
same `states` table (ground-contact points lifted to BEV, velocities from track association) and reuse `ego`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
HEADS_RUN = "runs/drive_backbones/heads_train/20260925-110819"
HEADS_TAG = "p3drive_heads"
MODELS = ("cinque", "lebowski")
CLS_LATE = "cls_late op-{m} temporal"
CLS_EGO = "cls ego K1024"
Q3_CLUSTERS = ("Cut_ins", "Pedestrian", "Cyclist", "Foreign Object Debris", "Interections", "all")
Q2_CLUSTERS = ("Interections", "Special Vehicles", "Multi-Lane Maneuvers", "Pedestrian", "Cut_ins",
               "Foreign Object Debris")
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "fusion-diagnostics"


# ================================================================ WOD context

def wod_ctx() -> dict:
    """Decision 40 (iii)'s per-frame predictions on all val, with index metadata, s_ego deciles and rater arrays."""
    from . import waymo as W
    z = np.load(data_dir() / HEADS_RUN / f"{HEADS_TAG}_preds_dir0.npz", allow_pickle=True)
    df = W.load_index()
    past, future = W.load_ego()
    names = W.frame_names(df)
    at = pd.Series(np.arange(len(df)), index=names)
    fn = z["frame_name"].astype(str)
    idx = at.reindex(fn).to_numpy().astype(int)
    s = z["s_ego"]
    dec = np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11))[1:-1], s, "right"), 0, 9)  # rejudge's
    sub = W.subsets(df, past, future)
    r, rtraj, rscore = W.load_rater(df)
    rpos = pd.Series(np.arange(len(fn)), index=fn).reindex(names[r]).to_numpy()
    assert not np.isnan(rpos).any(), "rater frames missing from the heads run"
    return {"fn": fn, "idx": idx, "seq": df.sequence.astype(str).to_numpy()[idx],
            "cluster": df.cluster.astype(str).to_numpy()[idx], "intent": df.intent.to_numpy()[idx],
            "gt": z["fut"], "speed": z["speed"], "dec": dec, "pre_onset": sub["pre_onset"][idx],
            "past": past[idx], "preds": {k[5:]: z[k] for k in z.files if k.startswith("pred_")},
            "rpos": rpos.astype(int), "rtraj": rtraj, "rscore": rscore}


def native(names: np.ndarray, m: str, split: str = "") -> tuple[np.ndarray, np.ndarray]:
    """openpilot's own plan (WOD rear-axle waypoints) on the given frames: (preds, covered mask)."""
    from .drive_backbones import root
    z = np.load(root() / f"op_{m}{'_' + split if split else ''}_native.npz")
    pos = pd.Series(np.arange(len(z["name"])), index=z["name"].astype(str)).reindex(names).to_numpy()
    ok = ~np.isnan(pos)
    out = np.full((len(names), 20, 2), np.nan, np.float32)
    out[ok] = z["wod"][pos[ok].astype(int)]
    return out, ok


def ade(p, g):
    return np.linalg.norm(p - g, axis=-1).mean(-1)


# ================================================================ Q3

def q3(rl) -> dict:
    from . import waymo as W
    from . import waymo_p1 as P1
    c = wod_ctx()
    rp = c["rpos"]
    keep19 = c["dec"] <= 8
    cv_all = W.baselines(c["past"])["cv"]
    rcl, rseq, rsp = c["cluster"][rp], c["seq"][rp], c["speed"][rp]
    rfs = lambda p: W.rater_feedback_score(p[rp], c["rtraj"], c["rscore"], rsp)  # noqa: E731
    rows = []

    def add(pair, readout, v, base, seq, cl, m, role="primary"):
        for k in Q3_CLUSTERS:
            mk = m & (cl == k) if k != "all" else m.copy()
            rows.append({"pair": pair, "readout": readout, "role": role, "cluster": k,
                         **P1.paired(v, base, seq, mk), "base_value": float(base[mk].mean()) if mk.any() else np.nan})

    e_ego = ade(c["preds"][CLS_EGO], c["gt"])
    r_ego, r_cv = rfs(c["preds"][CLS_EGO]), rfs(cv_all)
    e_cv = ade(cv_all, c["gt"])
    for m in MODELS:
        late = c["preds"][CLS_LATE.format(m=m)]
        e_late = ade(late, c["gt"])
        pair = f"cls_late {m} - cls ego"
        add(pair, "ADE all val, s_ego deciles 1-9 (m, lower better)", e_late, e_ego, c["seq"], c["cluster"], keep19)
        add(pair, "ADE pre-onset, s_ego deciles 1-9 (m)", e_late, e_ego, c["seq"], c["cluster"],
            keep19 & c["pre_onset"], "side")
        add(pair, "RFS rater frames (higher better)", rfs(late), r_ego, rseq, rcl, np.ones(len(rp), bool))
        # native - cv: RFS on the rater frames and ADE on the 20 237-frame subset (decision 40 (iii)'s native file)
        nat, ok = native(c["fn"], m)
        assert ok[rp].all(), "subset native file does not cover every rater frame"
        pair = f"native {m} - cv"
        add(pair, "RFS rater frames (higher better)", rfs(np.nan_to_num(nat)), r_cv, rseq, rcl, np.ones(len(rp), bool))
        e_nat = np.where(ok, ade(np.nan_to_num(nat), c["gt"]), np.nan)
        add(pair, "ADE 20237 subset, s_ego deciles 1-9 (m)", np.nan_to_num(e_nat), e_cv, c["seq"], c["cluster"], ok & keep19)
        nat2, ok2 = native(c["fn"], m, "trainval")
        e_nat2 = np.where(ok2, ade(np.nan_to_num(nat2), c["gt"]), np.nan)
        add(pair, "ADE all val (trainval native), s_ego deciles 1-9 (m)", np.nan_to_num(e_nat2), e_cv, c["seq"],
            c["cluster"], ok2 & keep19, "side")
        both = ok & ok2
        rl.log.info("native %s: subset %d frames, trainval %d frames; mean |subset - trainval| on shared frames %.4f m",
                    m, ok.sum(), ok2.sum(), float(np.abs(nat[both] - nat2[both]).mean()))
    t = pd.DataFrame(rows)
    ref = {m: float(W.rfs_by_cluster(rfs(c["preds"][CLS_LATE.format(m=m)]), rcl)[0]) for m in MODELS}
    rl.log.info("check vs decision 40 (iii): cls_late RFS cluster mean %s (expected 7.64 / 7.73)", ref)
    return {"q3_deltas": t, "_check": ref}


# ================================================================ Q2a

LON_M, LAT_M, HEAD_DEG, SEG_MIN_M = 3.0, 2.0, 15.0, 0.5
BEARING_DEG = 5.0                          # waymo.ONSET_BEARING
I2S, I5S = 7, 19                           # waypoint index of 2 s and 5 s (4 Hz, first point at 0.25 s)


def direction(model: np.ndarray, best: np.ndarray, intent: np.ndarray) -> pd.DataFrame:
    """Direction of model - rater_best per frame, in rater_best's own longitudinal / lateral frame (RFS geometry)."""
    from . import waymo as W
    lng, lat = W._rater_frames(best[:, None].astype(np.float64))
    lng, lat = lng[:, 0], lat[:, 0]
    v = model.astype(np.float64) - best
    dl2, dl5 = ((lng[:, i] * v[:, i]).sum(-1) for i in (I2S, I5S))
    dlat5 = (lat[:, I5S] * v[:, I5S]).sum(-1)
    segs = [p[:, I5S] - p[:, I5S - 2] for p in (model, best)]
    seg_ok = np.all([np.linalg.norm(s, axis=-1) >= SEG_MIN_M for s in segs], 0)
    ang = [np.degrees(np.arctan2(s[:, 1], s[:, 0])) for s in segs]
    dhead = np.where(seg_ok, np.abs((ang[0] - ang[1] + 180) % 360 - 180), np.nan)
    lon = np.maximum(np.abs(dl2), np.abs(dl5)) > LON_M
    latc = (np.abs(dlat5) > LAT_M) | (np.nan_to_num(dhead) > HEAD_DEG)
    dl = np.where(np.abs(dl2) >= np.abs(dl5), dl2, dl5)
    d = np.where(latc, "lateral", np.where(lon, np.where(dl > 0, "longitudinal stop", "longitudinal go"), "none"))

    def bearing(p):
        x, y = p[:, I5S, 0], p[:, I5S, 1]
        return np.where(np.hypot(x, y) >= 3.0, np.degrees(np.arctan2(y, x)), 0.0)
    sgn = np.where(intent == 2, 1.0, np.where(intent == 3, -1.0, 0.0))
    toward = lambda p: sgn * bearing(p) > BEARING_DEG  # noqa: E731
    turn = np.isin(intent, (2, 3))
    sub = np.where(~latc, "", np.where(~turn, "route ambiguity",
                                      np.where(toward(best) & ~toward(model), "decision error (intent explains)",
                                               "lateral other")))
    return pd.DataFrame({"dlon2": dl2, "dlon5": dl5, "dlat5": dlat5, "dhead": dhead, "both": lon & latc,
                         "direction": d, "lateral_class": sub})


Q2_CATS = ("lateral: decision error (intent explains)", "lateral: route ambiguity", "lateral: lateral other",
           "longitudinal stop", "longitudinal go", "none")


def category(dirs: pd.DataFrame) -> np.ndarray:
    return np.where(dirs.direction == "lateral", "lateral: " + dirs.lateral_class, dirs.direction).astype(object)


def q2a(rl) -> dict:
    from . import traj
    from . import waymo as W
    c = wod_ctx()
    rp = c["rpos"]
    rt, rs = c["rtraj"], c["rscore"]
    best = rt[np.arange(len(rp)), rs.argmax(1)].astype(np.float64)
    top = rs.max(1)
    fn, seq, cl, it, sp = c["fn"][rp], c["seq"][rp], c["cluster"][rp], c["intent"][rp], c["speed"][rp]
    nat, ok = native(c["fn"][rp], "cinque")
    assert ok.all()
    models = {"native": nat, "cls_late": c["preds"][CLS_LATE.format(m="cinque")][rp]}
    rfs = {k: W.rater_feedback_score(p, rt, rs, sp) for k, p in models.items()}
    loss = {k: v <= top - 1 for k, v in rfs.items()}
    dirs = {k: direction(p, best, it) for k, p in models.items()}
    judged = np.where(loss["native"], "native", np.where(loss["cls_late"], "cls_late", ""))
    any_loss = loss["native"] | loss["cls_late"]
    cat = np.where(judged == "native", category(dirs["native"]), category(dirs["cls_late"]))
    cat = np.where(any_loss, cat, "")
    log_dir = direction(c["gt"][rp], best, it)                     # "model" := the logged future
    frame = pd.DataFrame({"frame_name": fn, "sequence": seq, "cluster": cl, "intent": np.array(W.INTENTS)[it],
                          "speed": sp, "max_rater": top, "rfs_native": rfs["native"], "rfs_cls_late": rfs["cls_late"],
                          "loss_native": loss["native"], "loss_cls_late": loss["cls_late"], "loss": any_loss,
                          "judged_model": judged, "category": cat})
    for k, d in dirs.items():
        frame = frame.join(d.add_prefix(f"{k}_"))
    frame = frame.join(log_dir[["dlon2", "dlon5", "dlat5", "dhead", "direction", "lateral_class"]].add_prefix("log_"))
    for col in ("q2b_cause_object", "q2b_cause_class", "q2b_probe_op", "q2b_probe_qwen", "q2b_category"):
        frame[col] = ""

    def table(mask, cats, tag):
        rows = []
        groups = [(k, cl == k) for k in Q2_CLUSTERS] + [("others", ~np.isin(cl, Q2_CLUSTERS)),
                                                        ("six clusters pooled", np.isin(cl, Q2_CLUSTERS)),
                                                        ("all rater frames", np.ones(len(cl), bool))]
        for g, gm in groups:
            m = gm & mask
            row = {"model": tag, "group": g, "rater_frames": int(gm.sum()), "loss_frames": int(m.sum()),
                   "loss_share": float(m.sum() / max(gm.sum(), 1))}
            for k in Q2_CATS:
                ind = (cats[m] == k).astype(float)
                lo, hi = traj.boot_ci(ind, seq[m]) if m.sum() else (np.nan, np.nan)
                row |= {f"{k}": float(ind.mean()) if m.sum() else np.nan, f"{k} lo": lo, f"{k} hi": hi,
                        f"{k} n": int(ind.sum())}
            rows.append(row)
        return rows
    rows = table(any_loss, cat, "union (judged vs the losing model; native if both)")
    for k in models:
        rows += table(loss[k], category(dirs[k]), k)
    t = pd.DataFrame(rows)
    rl.log.info("loss frames: native %d, cls_late %d, union %d of %d", loss["native"].sum(), loss["cls_late"].sum(),
                any_loss.sum(), len(fn))
    return {"q2a_shares": t, "q2a_frames": frame}


# ================================================================ Q7

K_MAX, LAT_MODE_M, LON_MODE_M = 8, 1.0, 1.0
TURN_M, LC_M = 6.0, 2.5
ONSET_M, ONSET_CENSOR = 0.5, 5.25


OFFPATH_M = 1.0


def offpath_max(a: np.ndarray, b: np.ndarray) -> float:
    """Largest lateral distance from either trajectory's points to the other one's polyline (origin prepended).
    Points beyond the other's end are ignored, so a car that stopped short on the path the other one drove on
    (whatever that path does afterwards) scores ~0."""
    def one(p, q):
        q = np.r_[[[0.0, 0.0]], q].astype(np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-3]
        q = q[keep]
        if len(q) < 2:
            return float(np.linalg.norm(p, axis=1).max())
        s, d, u, inside = project(q, p.astype(np.float64))
        return float(np.where(inside, np.abs(d), 0.0).max())
    return max(one(a, b), one(b, a))


def descriptors(d: np.ndarray) -> np.ndarray:
    """(n, 20, 2) difference of two trajectories in one ego frame -> 2 s longitudinal diff, signed max lateral diff."""
    j = np.abs(d[..., 1]).argmax(1)
    return np.stack([d[:, I2S, 0], d[np.arange(len(d)), j, 1]], 1)


def onset_wod(d: np.ndarray) -> np.ndarray:
    hit = (np.abs(d) > ONSET_M).any(-1)
    return np.where(hit.any(1), (hit.argmax(1) + 1) * 0.25, ONSET_CENSOR)


def gmm_bic(X: np.ndarray, seed: int = 0):
    from sklearn.mixture import GaussianMixture
    Z = (X - X.mean(0)) / X.std(0).clip(1e-9)
    fits, bic = {}, []
    for k in range(1, min(K_MAX, len(X) // 3) + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=seed).fit(Z)
        fits[k] = g
        bic.append({"k": k, "bic": g.bic(Z)})
    bic = pd.DataFrame(bic)
    k = int(bic.k[bic.bic.idxmin()])
    return fits[k].predict(Z), bic


def name_mode(lon2: float, lat: float) -> tuple[str, str]:
    if abs(lat) >= LAT_MODE_M:
        side = "Left" if lat > 0 else "Right"
        kind = "turn" if abs(lat) >= TURN_M else "laneChange" if abs(lat) >= LC_M else "keep"
        return f"lateral {kind} {side.lower()}", f"{kind}{side}"
    if lon2 < -LON_MODE_M:
        return "longitudinal slow / stop", "none"
    if lon2 > LON_MODE_M:
        return "longitudinal faster / go", "none"
    return "near-zero difference", "none"


def clusters_table(X: np.ndarray, lab: np.ndarray, data: str, fit: str, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for c in np.unique(lab):
        m = lab == c
        lon2, lat, on = np.median(X[m], 0)
        mode, desire = name_mode(lon2, lat)
        rows.append({"data": data, "fit": fit, "cluster": int(c), "n": int(m.sum()), "share": float(m.mean()),
                     "lon2_median": lon2, "lat_median": lat, "onset_median": on,
                     "lon2_iqr": np.subtract(*np.percentile(X[m, 0], [75, 25])),
                     "lat_iqr": np.subtract(*np.percentile(X[m, 1], [75, 25])), "mode": mode, "desire": desire,
                     "covered": desire != "none", **{k: float(np.median(v[m]) if k.endswith("_median") else v[m].mean())
                                                         for k, v in (extra or {}).items()}})
    return pd.DataFrame(rows)


def q7(rl) -> dict:
    from . import p5_pairs as P
    # P5: reactive pair frames under the exam's own tau_exp
    d = P.processed()
    t = pd.read_parquet(d / "index.parquet")
    fut = np.load(d / "future.npy")
    obs, null = pd.read_parquet(d / "obs.parquet"), pd.read_parquet(d / "null.parquet")
    pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str})
    tau = max(float(np.quantile(np.abs(null.d_expert), 0.95)), 0.5)
    o = obs[np.abs(obs.d_expert) > tau].merge(pairs[["base_id", "seed", "t_vis", "t_div"]], on=["base_id", "seed"])
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    dd = fut[pos[o.fn_plus].to_numpy()] - fut[pos[o.fn_minus].to_numpy()]
    Xp = np.c_[descriptors(dd), (o.t_div - o.t_vis).to_numpy() * P.TICK]
    lab_p, bic_p = gmm_bic(Xp)
    # post-hoc diagnostic (deviation log 18:35): largest off-path distance between the two expert futures; below
    # OFFPATH_M the two worlds drove the same path and only the longitudinal progress differs
    fp, fm = fut[pos[o.fn_plus].to_numpy()], fut[pos[o.fn_minus].to_numpy()]
    offpath = np.array([offpath_max(a, b) for a, b in zip(fp, fm)])
    tp = clusters_table(Xp, lab_p, "P5", "P5 reactive frames", {"same_path_share": (offpath < OFFPATH_M).astype(float),
                                                                "offpath_median": offpath})
    fam = pd.crosstab(pd.Series(lab_p, name="cluster"), o.family.to_numpy(), normalize="columns")
    rl.log.info("P5: tau_exp %.3f, %d reactive frames (%d pairs); BIC k = %d", tau, len(o),
                o.groupby(["base_id", "seed"]).ngroups, int(bic_p.k[bic_p.bic.idxmin()]))
    # WOD: rater_best - log on the rater frames; the top s_ego decile reported and refitted on its own
    c = wod_ctx()
    rp = c["rpos"]
    best = c["rtraj"][np.arange(len(rp)), c["rscore"].argmax(1)]
    dw = best - c["gt"][rp]
    Xw = np.c_[descriptors(dw), onset_wod(dw)]
    top = c["dec"][rp] == 9
    lab_w, bic_w = gmm_bic(Xw)
    tw = clusters_table(Xw, lab_w, "WOD", "WOD all rater frames", {"share_in_dec10": top.astype(float)})
    tw["n_dec10"] = [int((top & (lab_w == k)).sum()) for k in tw.cluster]
    lab_t, bic_t = gmm_bic(Xw[top])
    tt = clusters_table(Xw[top], lab_t, "WOD", "WOD s_ego decile 10")
    bic = pd.concat([bic_p.assign(fit="P5 reactive frames"), bic_w.assign(fit="WOD all rater frames"),
                     bic_t.assign(fit="WOD s_ego decile 10")])
    cl = pd.concat([tp, tw, tt], ignore_index=True)
    # vocabulary: modes pooled over clusters, share of frames per dataset
    voc = cl.groupby(["mode", "desire", "covered", "fit"], as_index=False).n.sum()
    voc["share"] = voc.n / voc.groupby("fit").n.transform("sum")
    vt = voc.pivot_table(index=["mode", "desire", "covered"], columns="fit", values="share", fill_value=0).reset_index()
    cov = voc[voc.covered].groupby("fit").share.sum().reindex(voc.fit.unique()).fillna(0)
    rl.log.info("coverage by desire: %s", cov.to_dict())
    frames = pd.concat([pd.DataFrame({"data": "P5", "id": o.fn_plus + "|" + o.fn_minus, "family_or_cluster": o.family,
                                      "lon2": Xp[:, 0], "lat": Xp[:, 1], "onset": Xp[:, 2], "gmm": lab_p}),
                        pd.DataFrame({"data": "WOD", "id": c["fn"][rp], "family_or_cluster": c["cluster"][rp],
                                      "lon2": Xw[:, 0], "lat": Xw[:, 1], "onset": Xw[:, 2], "gmm": lab_w, "dec10": top})])
    return {"q7_clusters": cl, "q7_bic": bic, "q7_vocabulary": vt,
            "q7_coverage": cov.rename("coverage").reset_index(), "q7_p5_family_by_cluster": fam.reset_index(),
            "q7_frames": frames, "_tau": tau}


# ================================================================ Q6 gates

TTC_S, HALF_W, PATH_T, PATH_MIN, TTC_RANGE = 3.0, 1.2, 3.0, 5.0, 60.0
PED_V, PED_CORR, PED_RANGE, DZ_MAX = 0.5, 4.0, 30.0, 8.0
GATES = ("ttc", "intrusion", "pedestrian")


def project(path: np.ndarray, pts: np.ndarray):
    """Points (n, 2) onto the polyline `path` (m, 2): arc length s, signed lateral offset d (left +), unit tangent,
    and `inside` (the foot falls on the polyline, not beyond either end)."""
    a, b = path[:-1], path[1:]
    ab = b - a
    L = np.linalg.norm(ab, axis=1).clip(1e-9)
    u = ab / L[:, None]
    ap = pts[:, None] - a[None]                                   # (n, m-1, 2)
    tr = (ap * u[None]).sum(-1)
    tc = np.clip(tr, 0, L[None])
    foot = a[None] + tc[..., None] * u[None]
    dist = np.linalg.norm(pts[:, None] - foot, axis=-1)
    j = dist.argmin(1)
    i = np.arange(len(pts))
    cum = np.r_[0.0, np.cumsum(L)]
    s = cum[j] + tc[i, j]
    uj = u[j]
    d = uj[:, 0] * ap[i, j, 1] - uj[:, 1] * ap[i, j, 0]       # cross(u, p - a): + on the left
    inside = ~(((j == 0) & (tr[i, j] < 0)) | ((j == len(L) - 1) & (tr[i, j] > L[j])))
    return s, d, uj, inside


def gate_frame(obj: pd.DataFrame, v0: float, front: float, path: np.ndarray) -> dict:
    out = dict.fromkeys(GATES, False)
    if not len(obj) or len(path) < 2:
        return out
    p = obj[["x", "y"]].to_numpy(float)
    v = obj[["vx", "vy"]].to_numpy(float)
    s, d, u, inside = project(path, p)
    ahead = inside & (s > 0)
    # (i) TTC along the corridor
    corr = ahead & (np.abs(d) <= HALF_W) & (s <= TTC_RANGE)
    if corr.any():
        closing = v0 - (v[corr] * u[corr]).sum(1)
        gap = np.maximum(s[corr] - front, 0.0)
        ttc = np.where(closing > 0, gap / np.where(closing > 0, closing, 1.0), np.inf)
        out["ttc"] = bool((ttc < TTC_S).any())
    # (ii) any object inside the next 3 s of path +- HALF_W
    L3 = max(v0 * PATH_T, PATH_MIN)
    out["intrusion"] = bool((ahead & (np.abs(d) <= HALF_W) & (s <= L3)).any())
    # (iii) a pedestrian heading for the corridor
    ped = (obj.cls.to_numpy() == "pedestrian") & ahead & (s <= PED_RANGE) & (np.hypot(p[:, 0], p[:, 1]) < PED_RANGE)
    if ped.any():
        nrm = np.stack([-u[:, 1], u[:, 0]], 1)                  # left normal
        toward = -np.sign(d) * (v * nrm).sum(1)
        out["pedestrian"] = bool((ped & (np.abs(d) - HALF_W < PED_CORR) & (toward > PED_V)).any())
    return out


def gates(states: pd.DataFrame, ego: pd.DataFrame) -> pd.DataFrame:
    """The three gates and their OR for every (run, k) of `ego` (see the module docstring for both tables)."""
    grp = {key: g for key, g in states.groupby(["run", "k"], sort=False)}
    empty = states.iloc[:0]
    rows = []
    for r in ego.itertuples(index=False):
        g = gate_frame(grp.get((r.run, r.k), empty), r.v0, r.front, r.path)
        rows.append({"run": r.run, "k": r.k, **g, "any": any(g.values())})
    return pd.DataFrame(rows)


def _cls(type_id: str) -> str:
    return "pedestrian" if type_id.startswith("walker.") else "vehicle"


def gt_run(adir: Path, run: str, ks) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GT (states, ego) tables of one recorded run at ticks `ks` (see the module docstring)."""
    pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    pose["k"] = (pose.t / 0.05).round().astype(int)
    pose = pose.set_index("k")
    f2k = pd.Series(pose.index.to_numpy(), index=pose.frame.to_numpy())
    a = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    meta = json.loads((adir / "meta.json").read_text())
    front = float(meta["bbox_extent"][0] + meta["bbox_location"][0])
    route = pd.read_json(adir / "route.json")
    rxy = np.c_[route.x.to_numpy(), -route.y.to_numpy()]                 # right-handed world
    ryaw = -route.yaw.to_numpy()
    ak = f2k.reindex(a["frame"]).to_numpy()
    st, eg = [], []
    ks = [k for k in ks if k in pose.index]
    for k in ks:
        e = pose.loc[k]
        th = -np.radians(e.yaw)
        c, s_ = np.cos(th), np.sin(th)
        R = np.array([[c, s_], [-s_, c]])                                # world -> ego
        exy = np.array([e.x, -e.y])
        # route ahead: nearest route point with a compatible heading, then forward until the length is covered
        dist = np.hypot(*(rxy - exy).T)
        dist[np.abs((ryaw - np.degrees(th) + 180) % 360 - 180) > 90] = np.inf
        j = int(dist.argmin())
        pts = (rxy[j:] - exy) @ R.T
        fwd = pts[:, 0] > 0                                               # drop the route points still behind
        pts = pts[int(fwd.argmax()):] if fwd.any() else pts[-1:]
        path = np.r_[[[0.0, 0.0]], pts]
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        path = path[: max(2, int(np.searchsorted(cum, TTC_RANGE + 2.0)) + 1)]
        v0 = float(np.hypot(e.vx, e.vy))
        eg.append({"run": run, "k": int(k), "v0": v0, "front": front, "path": path})
        m = (ak == k) & (np.abs(a["xyz"][:, 2] - e.z) < DZ_MAX)
        if m.any():
            p = (np.c_[a["xyz"][m, 0], -a["xyz"][m, 1]] - exy) @ R.T
            v = np.c_[a["v"][m, 0], -a["v"][m, 1]] @ R.T
            ids = a["id"][m]
            st.append(pd.DataFrame({"run": run, "k": int(k), "obj": ids,
                                    "cls": [_cls(kinds.get(str(i), ["vehicle."])[0]) for i in ids],
                                    "x": p[:, 0], "y": p[:, 1], "vx": v[:, 0], "vy": v[:, 1]}))
    cols = ["run", "k", "obj", "cls", "x", "y", "vx", "vy"]
    return (pd.concat(st, ignore_index=True) if st else pd.DataFrame(columns=cols)), pd.DataFrame(eg)


def _gt_gates(gen: Path, run: str, ks) -> pd.DataFrame | None:
    from . import p5_pairs as P
    a = P.attempt(gen, run)
    if a is None:
        return None
    st, eg = gt_run(a, run, sorted(set(ks)))
    if not len(eg):
        return None
    g = gates(st, eg)
    return g.merge(eg[["run", "k", "v0"]], on=["run", "k"]).assign(n_obj=g.k.map(st.groupby("k").size()).fillna(0).astype(int))


def q6gt(rl, workers: int | None = None) -> dict:
    import os
    from joblib import Parallel, delayed
    from . import p5_exam as E
    from . import p5_pairs as P
    gen = data_dir() / "runs" / "p5_pairs" / "gen"
    t, past, fut, obs, null, pairs = E.load()
    fold_row = E.folds(t, pairs)
    fold = dict(zip(t.base_id, fold_row))
    pk = pairs.set_index(["base_id", "seed"])
    obs = obs.assign(run_plus=[pk.loc[(b, s), "plus"] for b, s in zip(obs.base_id, obs.seed)],
                     run_minus=[pk.loc[(b, s), "minus"] for b, s in zip(obs.base_id, obs.seed)])
    obs[["run_plus", "run_minus"]] = obs[["run_plus", "run_minus"]].astype(int).astype(str)
    null = null.assign(run_plus=[P.variant_id(b, "plus", 0) for b in null.base_id],
                       run_null=[P.variant_id(b, "null", 0) for b in null.base_id])
    tr = t[(t.source == "p5") & (t.role == "train")].copy()
    tr["run"] = tr.route_id.astype(str)
    tr["k"] = tr.k.astype(int)
    need = {}
    for df_, cols in ((obs, ("run_plus", "run_minus")), (null, ("run_plus", "run_null"))):
        for c in cols:
            for r, k in zip(df_[c], df_.k):
                need.setdefault(r, set()).add(int(k))
    for r, k in zip(tr.run, tr.k):
        need.setdefault(r, set()).add(int(k))
    rl.log.info("%d runs, %d (run, tick) frames to gate", len(need), sum(len(v) for v in need.values()))
    workers = workers or len(os.sched_getaffinity(0))          # the taskset the job was started under
    res = Parallel(workers, verbose=5)(delayed(_gt_gates)(gen, r, ks) for r, ks in sorted(need.items()))
    G = pd.concat([x for x in res if x is not None], ignore_index=True).set_index(["run", "k"])
    rl.log.info("gated %d frames of %d runs", len(G), G.index.get_level_values(0).nunique())

    def trig(runs, ks, gate):   # 1.0 / 0.0, NaN where the frame could not be gated
        return G[gate].astype(float).reindex(pd.MultiIndex.from_arrays([runs.to_numpy(), ks.astype(int).to_numpy()])).to_numpy()

    # brake magnitude per eval fold: median expert v0 - v(2 s) on triggered P5 training frames of the other folds
    pos = np.flatnonzero((t.source == "p5").to_numpy() & (t.role == "train").to_numpy())
    v0 = np.linalg.norm(past[pos, -1, 2:4], axis=1)
    decel = v0 - P.v2(fut[pos])
    tr = tr.assign(decel=decel, fold=fold_row[pos])
    stats, mags = [], {}
    names = {g: f"gate {g}" for g in (*GATES, "any")}
    for g in names:
        tg = trig(tr.run, tr.k, g)
        ok = ~np.isnan(tg)
        tg = np.nan_to_num(tg).astype(bool)
        brake = tr.decel.to_numpy() > 1.0
        stats.append({"gate": g, "train_frames": int(ok.sum()), "trigger_rate": float(tg[ok].mean()),
                      "decel_median_triggered": float(np.median(tr.decel[tg])) if tg.any() else np.nan,
                      "decel_median_not": float(np.median(tr.decel[ok & ~tg])),
                      "precision_vs_brake_gt_1ms": float(brake[tg].mean()) if tg.any() else np.nan,
                      "recall_vs_brake_gt_1ms": float(tg[ok & brake].mean()) if (ok & brake).any() else np.nan})
        for f in range(E.K_FOLDS):
            m = tg & ok & (tr.fold.to_numpy() != f) & (tr.fold.to_numpy() >= 0)
            mags[(g, f)] = float(np.median(tr.decel[m])) if m.any() else np.nan
    stats = pd.DataFrame(stats)
    for g, col in names.items():
        # the gate's direction is braking by construction: a non-positive fitted median is floored (logged in magnitudes)
        mag = lambda bases: np.array([max(np.nan_to_num(mags[(g, fold[b])]), 1e-3) for b in bases])  # noqa: E731
        mo = mag(obs.base_id)
        tp, tm = trig(obs.run_plus, obs.k, g), trig(obs.run_minus, obs.k, g)
        miss = np.isnan(tp) | np.isnan(tm)
        obs[col] = np.where(miss, np.nan, -mo * (tp - tm))
        obs[f"{col} plus"], obs[f"{col} minus"] = tp, tm
        mn = mag(null.base_id)
        np_, nn = trig(null.run_plus, null.k, g), trig(null.run_null, null.k, g)
        missn = np.isnan(np_) | np.isnan(nn)
        null[col] = np.where(missn, np.nan, -mn * (np_ - nn))
    ex = list(names.values())
    r = E.exam(obs, null, pairs, ex)
    fam = r["obs"].groupby("family")[[f"{c} {w}" for c in ex for w in ("plus", "minus")]].mean().reset_index()
    mag = pd.DataFrame([{"gate": g, "fold": f, "magnitude": v} for (g, f), v in mags.items()])
    rl.log.info("tau_exp %.3f; pooled families %s", r["tau_exp"], r["pooled_families"])
    return {"q6_flip_rates_gt": r["flips"], "q6_label_validity": r["validity"], "q6_gate_train_stats": stats,
            "q6_trigger_rates_by_family": fam, "q6_magnitudes": mag,
            "_obs": r["obs"], "_null": null, "_summary": {"tau_exp": r["tau_exp"], "pooled_families": r["pooled_families"],
                                                        "taus": r["taus"]}}


# ================================================================ CLI

STEPS = {"q3": q3, "q2a": q2a, "q7": q7, "q6gt": q6gt}
SMALL = {"q3": ("q3_deltas",), "q2a": ("q2a_shares", "q2a_frames"),
         "q7": ("q7_clusters", "q7_bic", "q7_vocabulary", "q7_coverage", "q7_p5_family_by_cluster"),
         "q6gt": ("q6_flip_rates_gt", "q6_label_validity", "q6_gate_train_stats", "q6_trigger_rates_by_family",
                  "q6_magnitudes")}


def main():
    import argparse
    import time
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item", choices=list(STEPS))
    a = ap.parse_args()
    rl = RunLog("fusion_diag", a.item)
    rl.event("start", item=a.item)
    t0 = time.time()
    out = STEPS[a.item](rl)
    for name, v in out.items():
        if isinstance(v, pd.DataFrame):
            if name.startswith("_"):
                v.to_parquet(rl.dir / f"{name[1:]}.parquet", index=False)
                continue
            v.to_csv(rl.dir / f"{name}.csv", index=False)
            if name != "q2a_frames" and name != "q7_frames":
                rl.log.info("%s\n%s", name, v.to_markdown(index=False, floatfmt=".3f"))
        else:
            (rl.dir / f"{name.strip('_')}.json").write_text(json.dumps(v, indent=2, default=str))
            rl.log.info("%s: %s", name, v)
    rl.event("end", wall_s=round(time.time() - t0, 1), outputs=[k for k in out if not k.startswith("_")])
    rl.log.info("done in %.1f s -> %s", time.time() - t0, rl.dir)
    rl.close()


if __name__ == "__main__":
    main()
