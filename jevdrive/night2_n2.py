"""Night queue 2, N2: does openpilot carry what a bypass needs, and does it change lanes on a desire pulse
(todos/2026-09-26-night-queue-2.md, N2 and the [A-N2] entries, written before any number).

  labels   per indexed frame of a P5-style set: CARLA ground-truth labels from the run's route.json / pose.jsonl /
           actors.npz (no simulator): a = static vehicle / walker in the ego lane 0 < ds <= 30 m, b = vehicle in an
           adjacent lane |ds| <= 20 m (b_front: 0 <= ds <= 20 m), c = oncoming vehicle in the lane to the left
           0 < ds <= 50 m; plus v_ego and the route heading change 60 m ahead (desire target selection)
           -> processed/<set>/night2_labels.parquet
  probe    linear probes (standardised L2 logistic regression, GroupKFold(5) by base route, AUC) on openpilot
           temporal / vision taps and the YOLO image-plane token set -> research/results/night2/N2/probe_<set>.csv
  targets  desire-check target frames and their 8 s warm-up streams -> processed/<set>/night2_desire_targets.json
           (the runner is scripts/night2_desire.py in the openpilot venv)
  desire   the runner's output -> research/results/night2/N2/desire_<set>.csv and the per-bin verdict table

Every function takes a set name (P5_SET-style directory under processed/), so N1's frames reuse the same code.
Run on the box: python -m jevdrive.night2_n2 <step> --set carla_p5v1_ba
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "night2" / "N2"
HALF_LANE, ADJ = 1.75, 5.25
A_REACH, B_REACH, C_REACH, V_STATIC = 30.0, 20.0, 50.0, 0.5
HIDDEN_DZ = 5.0
LABELS = ("a", "b", "b_front", "c")
CAMS = ("front", "front_left", "front_right")
YOLO_CLASSES = ("pedestrian", "cyclist", "vehicle")
YOLO_K, YOLO_SCORE = 8, 0.25


def proc(set_name: str, *parts) -> Path:
    return data_dir() / "processed" / set_name / Path(*parts)


def adir_of(files) -> str:
    return list(files)[3].rsplit("/cams/", 1)[0]


# ---------------------------------------------------------------- ground-truth labels

def _project(P: np.ndarray, cum: np.ndarray, q: np.ndarray, lo: int = 0, hi: int | None = None):
    """Points q (m, 2) onto the polyline P[lo:hi] (arc length cum): (s, signed d, segment index)."""
    hi = len(P) - 1 if hi is None else min(hi, len(P) - 1)
    A, B = P[lo:hi], P[lo + 1:hi + 1]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((q[:, None, :] - A[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
    X = A[None] + t[..., None] * AB[None]
    dist = np.hypot(*(q[:, None, :] - X).transpose(2, 0, 1))
    j = dist.argmin(1)
    r = np.arange(len(q))
    cross = AB[j, 0] * (q[:, 1] - A[j, 1]) - AB[j, 1] * (q[:, 0] - A[j, 0])
    return cum[lo + j] + t[r, j] * np.sqrt(L2[j]), np.sign(cross) * dist[r, j], lo + j


def _run_labels(args):
    adir, rows = args
    d = Path(adir)
    try:
        route = pd.read_json(d / "route.json")
        pose = pd.read_json(d / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame")
        a = np.load(d / "actors.npz")
        kinds = json.loads((d / "actor_kinds.json").read_text())
    except (OSError, ValueError) as e:
        return [(fn, None) for fn, _ in rows], f"{adir}: {e!r}"
    P = route[["x", "y"]].to_numpy(np.float64)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-3]
    P = P[keep]
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    ryaw = np.degrees(np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])))
    ryaw = np.r_[ryaw, ryaw[-1:]]
    order = np.argsort(a["frame"], kind="stable")
    fr, ids, xyz, yaw, vel = a["frame"][order], a["id"][order], a["xyz"][order], a["yaw"][order], a["v"][order]
    starts = np.searchsorted(fr, [f for _, f in rows]), np.searchsorted(fr, [f for _, f in rows], side="right")
    is_walker = {int(k): v[0].startswith("walker.") for k, v in kinds.items()}
    out = []
    for (fn, frame), i0, i1 in zip(rows, *starts):
        if frame not in pose.index:
            out.append((fn, None))
            continue
        e = pose.loc[frame]
        eyaw = e.yaw
        # ego: nearest route segment whose heading agrees with the ego's (loops and junction overlaps)
        s_all, d_all, j_all = _project(P, cum, np.array([[e.x, e.y]]))
        cand = np.flatnonzero(np.abs((ryaw - eyaw + 180) % 360 - 180) < 60)
        if len(cand):
            A, B = P[cand[cand < len(P) - 1]], P[cand[cand < len(P) - 1] + 1]
            q = np.array([e.x, e.y])
            AB = B - A
            t = np.clip(((q - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-9), 0, 1)
            k = cand[cand < len(P) - 1][int(np.hypot(*(q - (A + t[:, None] * AB)).T).argmin())]
        else:
            k = int(j_all[0])
        s_ego, _, _ = _project(P, cum, np.array([[e.x, e.y]]), max(0, k - 2), k + 3)
        s_ego = float(s_ego[0])
        # route heading change over the next 60 m (straight-frame selection for the desire check)
        k60 = int(np.searchsorted(cum, s_ego + 60.0))
        seg = ryaw[k:max(k + 1, min(k60, len(ryaw)))]
        dhead = float(np.abs((seg - ryaw[k] + 180) % 360 - 180).max()) if len(seg) else 0.0
        rec = {"v_ego": float(np.hypot(e.vx, e.vy)), "s_ego": s_ego, "dhead60": dhead, "route_left": float(cum[-1] - s_ego)}
        lab = dict.fromkeys(LABELS, False)
        n = dict.fromkeys(LABELS, 0)
        if i1 > i0:
            q = xyz[i0:i1].astype(np.float64)
            vis = np.abs(q[:, 2] - e.z) <= HIDDEN_DZ
            lo = int(np.searchsorted(cum, s_ego - 40.0))
            hi = int(np.searchsorted(cum, s_ego + 60.0)) + 1
            s, dd, jj = _project(P, cum, q[:, :2], max(0, lo - 1), hi)
            ds = s - s_ego
            sp = np.hypot(*vel[i0:i1].T.astype(np.float64))
            walk = np.array([is_walker.get(int(x), False) for x in ids[i0:i1]])
            veh = ~walk
            dyaw = np.abs((yaw[i0:i1] - ryaw[jj] + 180) % 360 - 180)
            # CARLA is left-handed (y right): a positive cross product is to the right of the route
            ego_lane, adj = np.abs(dd) <= HALF_LANE, (np.abs(dd) > HALF_LANE) & (np.abs(dd) <= ADJ)
            left_lane = (dd < -HALF_LANE) & (dd >= -ADJ)
            m = {"a": vis & ego_lane & (sp < V_STATIC) & (ds > 0) & (ds <= A_REACH),
                 "b": vis & veh & adj & (np.abs(ds) <= B_REACH),
                 "b_front": vis & veh & adj & (ds >= 0) & (ds <= B_REACH),
                 "c": vis & veh & left_lane & (dyaw > 135) & (sp > V_STATIC) & (ds > 0) & (ds <= C_REACH)}
            for key, mm in m.items():
                lab[key], n[key] = bool(mm.any()), int(mm.sum())
        out.append((fn, {**rec, **lab, **{f"n_{k}": v for k, v in n.items()}}))
    return out, None


def labels(set_name: str, workers: int = 24, source: str | None = "p5") -> pd.DataFrame:
    t = pd.read_parquet(proc(set_name, "index.parquet"))
    if source is not None and "source" in t:
        t = t[t.source == source]
    t = t.assign(adir=t.files.map(adir_of))
    jobs = [(a, list(zip(g.frame_name, g.frame.astype(int)))) for a, g in t.groupby("adir")]
    res, errs = {}, []
    with Pool(workers) as p:
        for part, err in p.imap_unordered(_run_labels, jobs, chunksize=2):
            res.update(part)
            if err:
                errs.append(err)
    ok = [fn for fn in t.frame_name if res.get(fn) is not None]
    lab = pd.DataFrame([res[fn] for fn in ok], index=pd.Index(ok, name="frame_name"))
    lab = lab.join(t.set_index("frame_name")[["base_id", "world", "seed", "intent", "adir", "frame"]])
    lab.to_parquet(proc(set_name, "night2_labels.parquet"))
    log.info("%s: %d / %d frames labelled (%d run errors); positive rate %s", set_name, len(lab), len(t), len(errs),
             {k: round(float(lab[k].mean()), 4) for k in LABELS})
    for e in errs[:5]:
        log.warning(e)
    return lab


# ---------------------------------------------------------------- features

def op_features(set_name: str, frame_names: pd.Series, sub: str = "op_streams_vis") -> dict:
    from . import p5_openpilot
    os.environ["P5_SET"] = set_name
    t = pd.DataFrame({"frame_name": frame_names.to_numpy()})
    f = p5_openpilot.load(t, ("cinque", "lebowski"), arrays=("temporal", "vision"), sub=sub)
    return {"cinque temporal": f["op-cinque temporal"], "lebowski temporal": f["op-lebowski temporal"],
            "cinque vision (mean)": f["op-cinque vision"], "lebowski vision (view_40)": f["op-lebowski vision"]}


def yolo_tokens(det_root: Path, frame_names: pd.Series) -> np.ndarray | None:
    """(n, 216): per camera the top-8 detections by score, (class one-hot, u_c/W, v_c/H, w/W, h/H, score, mask);
    image plane only, no lift, no corridor. None when the detections do not cover these frames."""
    parts = sorted(Path(det_root).glob("*/part-*.parquet")) or sorted(Path(det_root).glob("part-*.parquet"))
    if not parts:
        return None
    d = pd.concat([pd.read_parquet(p, columns=["key", "prompt", "score", "x0", "y0", "x1", "y1", "H", "W"]) for p in parts],
                  ignore_index=True)
    d = d[(d.score >= YOLO_SCORE) & d.prompt.isin(YOLO_CLASSES)]
    fn, cam = d.key.str.split("|").str[0], d.key.str.split("|").str[1]
    pos = pd.Series(np.arange(len(frame_names)), index=frame_names.to_numpy())
    row = fn.map(pos)
    have = set(fn.unique())
    cover = float(np.mean([f in have for f in frame_names]))
    log.info("YOLO detections cover %.3f of the frames (frames with >= 1 detection)", cover)
    ok = row.notna().to_numpy()
    d, row, cam = d[ok], row[ok].astype(int).to_numpy(), cam[ok].map({c: i for i, c in enumerate(CAMS)}).to_numpy()
    d = d.assign(row=row, cam=cam).sort_values(["row", "cam", "score"], ascending=[True, True, False])
    d["rank"] = d.groupby(["row", "cam"]).cumcount()
    d = d[d["rank"] < YOLO_K]
    X = np.zeros((len(frame_names), len(CAMS), YOLO_K, 9), np.float32)
    r, c, k = d.row.to_numpy(), d.cam.to_numpy(), d["rank"].to_numpy()
    X[r, c, k, d.prompt.map({p: i for i, p in enumerate(YOLO_CLASSES)}).to_numpy()] = 1
    W, H = d.W.to_numpy(np.float32), d.H.to_numpy(np.float32)
    X[r, c, k, 3] = (d.x0 + d.x1).to_numpy() / 2 / W
    X[r, c, k, 4] = (d.y0 + d.y1).to_numpy() / 2 / H
    X[r, c, k, 5] = (d.x1 - d.x0).to_numpy() / W
    X[r, c, k, 6] = (d.y1 - d.y0).to_numpy() / H
    X[r, c, k, 7] = d.score.to_numpy()
    X[r, c, k, 8] = 1.0
    return X.reshape(len(frame_names), -1)


# ---------------------------------------------------------------- probe

def logreg_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, C: float = 1.0) -> dict:
    """Standardised (train-fold stats) L2 logistic regression, minimise sum BCE + ||w||^2 / (2C) (sklearn's objective)
    with full-batch L-BFGS on the GPU; GroupKFold by `groups`. Mean / sd of the fold AUCs and the pooled OOF AUC."""
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    yt = torch.as_tensor(y.astype(np.float32), device=dev)
    oof, aucs = np.full(len(y), np.nan), []
    for tr, te in GroupKFold(folds).split(X, y, groups):
        tr_t, te_t = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
        mu, sd = Xt[tr_t].mean(0), Xt[tr_t].std(0).clamp_min(1e-6)
        A, B = (Xt[tr_t] - mu) / sd, (Xt[te_t] - mu) / sd
        w = torch.zeros(A.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, tolerance_grad=1e-6, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt[tr_t], reduction="sum") \
                + (w * w).sum() / (2 * C)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            p = (B @ w + b).cpu().numpy()
        oof[te] = p
        yte = y[te]
        aucs.append(roc_auc_score(yte, p) if 0 < yte.sum() < len(yte) else np.nan)
    return {"auc_mean": float(np.nanmean(aucs)), "auc_sd": float(np.nanstd(aucs)), "auc_oof": float(roc_auc_score(y, oof)),
            "fold_aucs": [round(float(a), 4) for a in aucs]}


def verdict(auc: float) -> str:
    return "有信息，可激发" if auc >= 0.70 else "没有，要外接" if auc <= 0.60 else "之间"


def probe_table(lab: pd.DataFrame, feats: dict, label_cols=("a", "b", "b_front", "c"), tag: str = "") -> pd.DataFrame:
    groups = lab.base_id.astype(str).to_numpy()
    rows = []
    for lc in label_cols:
        y = lab[lc].to_numpy().astype(int)
        for name, X in feats.items():
            if X is None:
                continue
            r = logreg_auc(X, y, groups)
            rows.append({"set": tag, "probe": lc, "feature": name, "dim": X.shape[1], "n": len(y), "pos_rate": round(float(y.mean()), 4),
                         "n_routes": len(set(groups)), **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                         "verdict": verdict(r["auc_mean"])})
            log.info("%s probe %s %s: AUC %.3f +- %.3f (oof %.3f), n %d, pos %.3f", tag, lc, name, r["auc_mean"], r["auc_sd"],
                     r["auc_oof"], len(y), y.mean())
    return pd.DataFrame(rows)


def probe(set_name: str, det_root: str | None = None) -> pd.DataFrame:
    lab = pd.read_parquet(proc(set_name, "night2_labels.parquet"))
    feats = op_features(set_name, lab.index.to_series())
    if det_root:
        feats["YOLO26x-seg image-plane tokens"] = yolo_tokens(Path(det_root), lab.index.to_series())
    tab = probe_table(lab, feats, tag=set_name)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / f"probe_{set_name}.csv", index=False)
    return tab


def controls(set_name: str, det_root: str | None = None, v_min: float = 3.0) -> pd.DataFrame:
    """Post-hoc ([A-N2] 10:43): the same probes on ego speed alone, and every feature refitted on v_ego >= v_min frames."""
    lab = pd.read_parquet(proc(set_name, "night2_labels.parquet"))
    rows = [probe_table(lab, {"ego speed only": lab[["v_ego"]].to_numpy(np.float32)}, tag=set_name).assign(frames="all")]
    moving = (lab.v_ego >= v_min).to_numpy()
    feats = op_features(set_name, lab.index.to_series())
    if det_root:
        feats["YOLO26x-seg image-plane tokens"] = yolo_tokens(Path(det_root), lab.index.to_series())
    feats = {k: v[moving] for k, v in feats.items() if v is not None}
    feats["ego speed only"] = lab[["v_ego"]].to_numpy(np.float32)[moving]
    rows.append(probe_table(lab[moving], feats, tag=set_name).assign(frames=f"v_ego>={v_min:g}"))
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(RESULTS / f"probe_controls_{set_name}.csv", index=False)
    return tab


# ---------------------------------------------------------------- desire check

BINS = ((5.0, 10.0, "5-10"), (10.0, 15.0, "10-15"), (15.0, np.inf, ">15"))
WARM, READ = 40, (0, 1, 5)                     # 8 s warm-up from a zero state; read points in 5 Hz frames after the pulse
PER_BIN, PER_RUN, MIN_GAP_S = 150, 2, 10.0


def targets(set_name: str, seed: int = 0) -> dict:
    lab = pd.read_parquet(proc(set_name, "night2_labels.parquet"))
    ok = (lab.v_ego >= 5.0) & (lab.intent == 1) & (lab.dhead60 < 10.0)
    cand = lab[ok].copy()
    rng = np.random.RandomState(seed)
    frames_cache, picked = {}, []

    def run_frames(adir):
        if adir not in frames_cache:
            fr = sorted((json.loads(x) for x in open(Path(adir) / "frames.jsonl")), key=lambda r: r["frame"])
            frames_cache[adir] = fr
        return frames_cache[adir]

    for lo, hi, name in BINS:
        c = cand[(cand.v_ego >= lo) & (cand.v_ego < hi)]
        c = c.iloc[rng.permutation(len(c))]
        per_run, got = {}, []
        for fn, r in c.iterrows():
            if len(got) >= PER_BIN:
                break
            prev = per_run.get(r.adir, [])
            if len(prev) >= PER_RUN or any(abs(r.frame - f) < MIN_GAP_S * 20 for f in prev):
                continue
            fr = run_frames(r.adir)
            pos = next((i for i, x in enumerate(fr) if x["frame"] == r.frame), None)
            if pos is None or pos < WARM or pos + max(READ) >= len(fr):
                continue
            span = fr[pos - WARM: pos + max(READ) + 1]
            if any(b["frame"] - a["frame"] != 4 for a, b in zip(span, span[1:])):
                continue
            per_run.setdefault(r.adir, []).append(r.frame)
            got.append({"key": fn, "bin": name, "v_ego": float(r.v_ego), "pulse": WARM, "read": list(READ),
                        "files": [[f"{r.adir}/{x['files'][cam]}" for cam in CAMS] for x in span]})
        picked += got
        log.info("%s bin %s: %d candidates, %d targets", set_name, name, len(c), len(got))
    plan = {"set": set_name, "targets": picked}
    proc(set_name, "night2_desire_targets.json").write_text(json.dumps(plan))
    return {"targets": len(picked)}


def desire_table(set_names=("carla_p5v1_ba", "carla_p5v1_pdm")) -> pd.DataFrame:
    rows = []
    for s in set_names:
        f = proc(s, "night2_desire_out.jsonl")
        if f.exists():
            rows.append(pd.read_json(f, lines=True).assign(set=s))
    d = pd.concat(rows, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    d.drop(columns=[c for c in d.columns if c.startswith("plan")], errors="ignore").to_csv(RESULTS / "desire_frames.csv", index=False)
    out = []
    for (model, read), g in d.groupby(["model", "read"]):
        for sname, gs in list(g.groupby("set")) + [("pooled", g)]:
            for b in [x[2] for x in BINS]:
                gb = gs[gs.bin == b]
                if not len(gb):
                    out.append({"model": model, "read_s": read * 0.2, "set": sname, "bin": b, "n_targets": 0})
                    continue
                dl = np.r_[gb.d_left.to_numpy(), gb.d_right.to_numpy()]
                ok = np.r_[gb.d_left.to_numpy() > 0, gb.d_right.to_numpy() < 0]
                med = float(np.median(np.abs(dl)))
                v = "执行器可用" if med >= 1.5 and ok.mean() >= 0.9 else "openpilot 不按 desire 变道" if med < 0.8 else "之间"
                out.append({"model": model, "read_s": read * 0.2, "set": sname, "bin": b, "n_targets": len(gb),
                            "median_abs_dlat3": round(med, 3), "median_dlat3_left": round(float(np.median(gb.d_left)), 3),
                            "median_dlat3_right": round(float(np.median(gb.d_right)), 3),
                            "p90_abs_dlat3": round(float(np.percentile(np.abs(dl), 90)), 3),
                            "dir_correct": round(float(ok.mean()), 3), "verdict": v})
    tab = pd.DataFrame(out)
    tab.to_csv(RESULTS / "desire_bins.csv", index=False)
    return tab


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("labels", "probe", "controls", "targets", "desire"))
    ap.add_argument("--set", default="carla_p5v1_ba")
    ap.add_argument("--dets", default="", help="probe: YOLO detection root (E5: processed/elicit_e5/dets)")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.step == "labels":
        labels(a.set, a.workers)
    elif a.step == "probe":
        print(probe(a.set, a.dets or None).to_markdown(index=False))
    elif a.step == "controls":
        print(controls(a.set, a.dets or None).to_markdown(index=False))
    elif a.step == "targets":
        print(targets(a.set))
    else:
        print(desire_table().to_markdown(index=False))


if __name__ == "__main__":
    main()
