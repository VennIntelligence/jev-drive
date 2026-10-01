"""op-adapt step 3 readouts (op-train venv) on the feature pass of experiments/op_adapt_r1/archive/op_adapt_eval.py; pass lines in
fc65452:todos/2026-09-28-op-adapt.md.

  (a) nuScenes val keyframes: corridor-pedestrian probe, decision 42 D0 recipe (fold z-score, class-balanced L2
      logistic by L-BFGS, λ from an inner 20 % scene split, 5 scene folds dealt from a seed-0 permutation,
      out-of-fold AUC, scene-cluster bootstrap 500), original vs adapted `temporal` / `vision`
  (b) plan drift on normal frames (nuScenes val: no pedestrian / cyclist in the wide corridor; WOD val: no YOLO26x
      pedestrian / cyclist detection that is unlifted or lifted into |d| <= 4 m, 0 < s <= 40 m), lead drift, and WOD
      native-plan ADE against the logged future (rear-axle conversion of decision 34), sequence bootstrap
  (c) P5 v1 BA: D0 itself (p5_exam.probes + probe_auc_paired_scopes, scope pedestrian) on the adapted features

  python experiments/op_adapt_r1/lib/op_adapt_readout.py --run runs/op_adapt/train-lam1/<ts>
Output: <run>/readout/{a_probe.csv, b_drift.csv, c_d0.csv, summary.json}
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_data as D  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

LAMS = (1e-4, 1e-3, 1e-2, 1e-1)
TAPS = ("orig_temporal", "adapt_temporal", "orig_vision", "adapt_vision")


def oof_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k=5, seed=0) -> np.ndarray:
    from sklearn.model_selection import GroupShuffleSplit
    from jevdrive.p4_carla import _std, auc, logreg
    ug = np.array(sorted(set(groups)))
    fold = pd.Series(np.arange(len(ug)) % k, index=np.random.default_rng(seed).permutation(ug))[groups].to_numpy()
    Xt = torch.as_tensor(X, device="cuda", dtype=torch.float32)
    yt = torch.as_tensor(y.astype(np.int64), device="cuda")
    s = np.full(len(y), np.nan)
    for f in range(k):
        tr, ev = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        mu, sd = _std(Xt, tr)
        Z = (Xt - mu) / sd
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tr, groups=groups[tr]))
        fi, si = tr[a], tr[b]
        sel = [auc(y[si], (Z[si] @ W + c).softmax(1)[:, 1].cpu().numpy()) for W, c in (logreg(Z[fi], yt[fi], lam) for lam in LAMS)] \
            if len(np.unique(y[fi])) == 2 and len(np.unique(y[si])) == 2 else []
        lam = LAMS[int(np.argmax(sel))] if sel else 1e-2
        W, c = logreg(Z[tr], yt[tr], lam)
        s[ev] = (Z[ev] @ W + c).softmax(1)[:, 1].cpu().numpy()
    return s


def paired_boot(y, sa, sb, groups, b=500, seed=0):
    """AUC(a), AUC(b) and AUC(a) - AUC(b) with a cluster bootstrap over groups (same draws for both)."""
    from jevdrive.p4_carla import auc
    codes, uniq = pd.factorize(groups)
    mem = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    da, db = [], []
    for _ in range(b):
        m = np.concatenate([mem[c] for c in rng.integers(len(uniq), size=len(uniq))])
        if len(np.unique(y[m])) == 2:
            da.append(auc(y[m], sa[m]))
            db.append(auc(y[m], sb[m]))
    da, db = np.array(da), np.array(db)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
    return {"auc": auc(y, sa), "auc_ci": q(da), "auc_ref": auc(y, sb), "auc_ref_ci": q(db),
            "delta": auc(y, sa) - auc(y, sb), "delta_ci": q(da - db), "n": int(len(y)), "pos": int(y.sum()), "groups": len(uniq)}


def readout_a(ev, out):
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet").set_index("token")
    L = lab.loc[ev["key"]]
    g = L.scene.to_numpy()
    rows, scores = [], {}
    for task in ("ped_corr", "ped_wide", "vru_corr"):
        y = L[task].to_numpy(bool).astype(int)
        for tap in TAPS:
            scores[(task, tap)] = oof_probe(ev[tap], y, g)
        for kind in ("temporal", "vision"):
            r = paired_boot(y, scores[(task, f"adapt_{kind}")], scores[(task, f"orig_{kind}")], g)
            rows.append({"task": task, "tap": kind, **r})
    # descriptive (coordinator 2026-09-28): by distance of the nearest pedestrian, in-lane corridor vs wide corridor only;
    # positives of the bin against the clean negatives (no pedestrian in the wide corridor), same out-of-fold scores
    neg = ~L.ped_wide.to_numpy(bool)
    bins = ((0, 10), (10, 20), (20, 41))
    for task, cond, dcol in (("ped_corr", L.ped_corr.to_numpy(bool), "ped_dist"),
                             ("ped_wide", L.ped_wide.to_numpy(bool) & ~L.ped_corr.to_numpy(bool), "ped_dist_wide")):
        dd = L[dcol].to_numpy()
        for lo, hi in bins:
            p = cond & (dd > lo) & (dd <= hi)
            m = p | neg
            if p.sum() < 10:
                continue
            for kind in ("temporal", "vision"):
                r = paired_boot(p[m].astype(int), scores[(task, f"adapt_{kind}")][m], scores[(task, f"orig_{kind}")][m], g[m])
                rows.append({"task": f"{'in-lane corridor' if task == 'ped_corr' else 'wide corridor only'} {lo}-{hi} m "
                             "(descriptive)", "tap": kind, **r})
    # the aux heads themselves (trained on nuScenes train; val scenes are held out)
    y = L.ped_corr.to_numpy(bool).astype(int)
    for h in ("head_t", "head_v"):
        r = paired_boot(y, ev[f"adapt_{h}"][:, 0], scores[("ped_corr", "orig_temporal")], g)
        rows.append({"task": "ped_corr", "tap": f"aux {h} (vs orig temporal probe)", **r})
    # distance on corridor positives: ridge OOF MAE (descriptive)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    pos = L.ped_corr.to_numpy(bool)
    dist = L.ped_dist.to_numpy()[pos]
    for tap in ("orig_temporal", "adapt_temporal"):
        X = ev[tap][pos]
        pr = np.zeros(pos.sum())
        for tr, te in GroupKFold(5).split(X, groups=g[pos]):
            m = X[tr].mean(0), X[tr].std(0) + 1e-6
            pr[te] = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit((X[tr] - m[0]) / m[1], dist[tr]).predict((X[te] - m[0]) / m[1])
        rows.append({"task": "ped_dist_mae_m", "tap": tap, "auc": float(np.abs(pr - dist).mean()),
                     "auc_ref": float(np.abs(dist - dist.mean()).mean()), "n": int(pos.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "a_probe.csv", index=False)
    return df


def lead_x(lead):          # (n, 72) -> first lead hypothesis x at t = 0
    return lead.reshape(-1, 3, 6, 4)[:, 0, 0, 0]


def sig(x):
    return 1 / (1 + np.exp(-x))


def drift_rows(ev, mask, name):
    d = A.plan_drift(ev["adapt_plan"][mask], ev["orig_plan"][mask])
    lp_o, lp_a = sig(ev["orig_lead_prob"][mask, 0]), sig(ev["adapt_lead_prob"][mask, 0])
    has = lp_o > 0.5
    lx = np.abs(lead_x(ev["adapt_lead"][mask]) - lead_x(ev["orig_lead"][mask]))[has]
    return {"set": name, "n": int(mask.sum()), "drift_median": float(np.median(d)), "drift_p95": float(np.percentile(d, 95)),
            "drift_mean": float(d.mean()), "lead_prob_absdiff_median": float(np.median(np.abs(lp_a - lp_o))),
            "lead_x_absdiff_median": float(np.median(lx)) if len(lx) else np.nan, "n_lead": int(has.sum())}


def wod_normal(keys):
    dets = pd.read_parquet(data_dir() / "processed/real_transfer/yolo/wod_val/dets.parquet")
    v = dets[dets.prompt.isin(["pedestrian", "cyclist"])]
    bad = v[~v.lift_ok | ((v.d.abs() <= D.WIDE_W) & (v.s > 0) & (v.s <= D.WIDE_R))].frame_id.unique()
    have = set(pd.read_parquet(data_dir() / "processed/real_transfer/yolo/wod_val/frames.parquet").frame_id)
    return np.array([k in have and k not in set(bad) for k in keys]), np.array([k in have for k in keys])


def wod_ade(ev):
    """Native-plan ADE (20 x 0.25 s, rear axle) against the logged future, original and adapted."""
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    df = W.load_index()
    row = pd.Series(np.arange(len(df)), index=W.frame_names(df))
    fut = np.load(data_dir() / "processed/waymo_e2e/future.npy", mmap_mode="r")
    keys = ev["key"]
    gt = np.stack([fut[row[k]][:, :2] for k in keys]).astype(np.float64)
    res = {}
    for m in ("orig", "adapt"):
        p = ev[f"{m}_plan"]
        wp = np.stack([Z.openpilot_to_wod(p[i, :, 0:3], p[i, :, 11], A.T_IDXS,
                                          np.array(calib[k.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:2, 3])
                       for i, k in enumerate(keys)])
        res[m] = np.linalg.norm(wp[:, :, :2] - gt, axis=-1).mean(1)
    return res


def readout_b(evn, evw, out):
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet").set_index("token")
    rows = [drift_rows(evn, ~lab.loc[evn["key"]].vru_wide.to_numpy(bool), "nuScenes val normal"),
            drift_rows(evn, np.ones(len(evn["key"]), bool), "nuScenes val all (descriptive)")]
    extra = {}
    if evw is not None:
        norm, have = wod_normal(evw["key"])
        rows += [drift_rows(evw, norm, "WOD val normal"), drift_rows(evw, np.ones(len(norm), bool), "WOD val all (descriptive)")]
        ade = wod_ade(evw)
        seq = np.array([k.rsplit("-", 1)[0] for k in evw["key"]])
        d = ade["adapt"] - ade["orig"]
        codes, uniq = pd.factorize(seq)
        mem = [np.flatnonzero(codes == c) for c in range(len(uniq))]
        rng = np.random.default_rng(0)
        rel = []
        for _ in range(1000):
            m = np.concatenate([mem[c] for c in rng.integers(len(uniq), size=len(uniq))])
            rel.append(d[m].mean() / ade["orig"][m].mean())
        extra = {"wod_ade_orig": float(ade["orig"].mean()), "wod_ade_adapt": float(ade["adapt"].mean()),
                 "wod_ade_rel_change": float(d.mean() / ade["orig"].mean()),
                 "wod_ade_rel_ci": (float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))),
                 "wod_n": int(len(d)), "wod_sequences": len(uniq)}
    df = pd.DataFrame(rows)
    df.to_csv(out / "b_drift.csv", index=False)
    return df, extra


def readout_c(evp, out, rl):
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_exam as E
    t, past, fut, obs, null, pairs = E.load()
    pos = pd.Series(np.arange(len(evp["key"])), index=evp["key"])
    at = pos.reindex(t.frame_name).to_numpy()
    assert not np.isnan(at).any(), "P5 frames without features"
    at = at.astype(int)
    X = {k: evp[k][at] for k in TAPS}
    fold = E.folds(t, pairs)
    scores, _ = E.probes(t, X, fold, rl)
    ps = E.probe_auc_paired_scopes(obs, t, scores, [("adapt_temporal", "orig_temporal"), ("adapt_vision", "orig_vision")])
    ps.to_csv(out / "c_d0.csv", index=False)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--parts", nargs="+", default=["a", "b", "c"])
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "readout"
    out.mkdir(exist_ok=True)
    rl = RunLog("op_adapt", "readout")
    ev = {ds: dict(np.load(run / "eval" / f"{ds}.npz")) for ds in ("nusc", "wod", "p5") if (run / "eval" / f"{ds}.npz").exists()}
    summ = {}
    if "a" in a.parts:
        df = readout_a(ev["nusc"], out)
        rl.info("(a)\n" + df.to_string())
        summ["a"] = df.to_dict("records")
    if "b" in a.parts:
        df, extra = readout_b(ev["nusc"], ev.get("wod"), out)
        rl.info("(b)\n" + df.to_string() + f"\n{extra}")
        summ["b"] = {"rows": df.to_dict("records"), **extra}
    if "c" in a.parts and "p5" in ev:
        ps = readout_c(ev["p5"], out, rl)
        ped = ps[ps.scope == "pedestrian"]
        rl.info("(c)\n" + ped.to_string())
        summ["c"] = ped.to_dict("records")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    rl.event("end")


if __name__ == "__main__":
    main()
