"""W diagnosis (todos/2026-09-28-wm-loop.md; report tmp/2026-09-27-wm-loop-diagnosis.md): why the W world model failed.
Reuses nq4_w's universe (processed/nq4_w/{meta.parquet, z.npy}), folds, model and probes unchanged.

  confound  CPU. In the logged (expert) windows, how the future 2 s of actions relate to the future front distance,
            against the kinematic answer for the same windows
  stored    CPU. Re-read the stored W seed runs: probe quality on the test folds against the ground-truth label,
            the pair-AUC ceiling the label itself allows, the pair AUC on anchors where the x+ label is on
  pedprobe  GPU. Route-fold OOF pedestrian probe on real features of P5 v1 (Cinque temporal, V-JEPA 2 mean / last_mean,
            Qwen L18_last, SigLIP2, YOLO image-plane tokens)
  variant   GPU. Retrain W's predictor (seed 0, 5 folds) with one change: full z / V-JEPA only / openpilot only /
            future actions zeroed; latent error against persistence, action-shuffle sensitivity, brake-vs-hold on
            front distance and on ego speed
  report    tables -> research/results/nq4/wdiag/
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import nq4_w as W
from .common import data_dir, get_logger

log = get_logger(__name__)
OUT = W.REPO / "research" / "results" / "nq4" / "wdiag"
KINDS = ("full", "vj", "op", "noact")


def _meta():
    return pd.read_parquet(W.wdir("meta.parquet"))


def _anchors(meta):
    return np.flatnonzero(meta.win_start.to_numpy()) + W.HIST - 1


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def _kin_d(v0, acc):
    """Front distance change of a static object over the 10 future steps: -(ego travel) under the given accelerations."""
    v, s = v0.astype(np.float64).copy(), np.zeros(len(v0))
    for j in range(W.FUT):
        vn = np.maximum(0.0, v + acc[:, j] * W.DT)
        s += 0.5 * (v + vn) * W.DT
        v = vn
    return -s


# ================================================================ confound (CPU)

def confound() -> dict:
    """Anchors with v >= 3 m/s. a_bar = mean logged acceleration over the 10 future steps (what the predictor is fed).
    OLS of d_front(t+2 s) on [d_front(t), v, a_bar] within the occupied-lane anchors, the same coefficient for the
    kinematic prediction of a static object, and stratified brake-vs-steady contrasts."""
    m = _meta()
    an = _anchors(m)
    fut = an[:, None] + np.arange(W.FUT)
    a_bar = m.a_next.to_numpy()[fut].mean(1)
    d0, d10 = m.d_front.to_numpy()[an], m.d_front.to_numpy()[an + W.FUT]
    occ0, occ10 = m.occ.fillna(False).to_numpy(bool)[an], m.occ.fillna(False).to_numpy(bool)[an + W.FUT]
    v0 = m.v.to_numpy()[an]
    kin = d0 + _kin_d(v0, m.a_next.to_numpy()[fut])
    t = pd.DataFrame({"set": m.set.to_numpy()[an], "world": m.world.to_numpy()[an], "base_id": m.base_id.to_numpy()[an],
                      "v0": v0, "a_bar": a_bar, "d0": d0, "d10": d10, "kin10": kin, "occ0": occ0, "occ10": occ10})
    t = t[t.v0 >= W.V_HAZ]
    rows = []
    for name, g in (("all v>=3", t), ("lane occupied at t (d0<40)", t[t.d0 < W.D_MAX]),
                    ("hazard anchors (occ at t)", t[t.occ0])):
        X = np.c_[np.ones(len(g)), g.d0, g.v0, g.a_bar]
        b = np.linalg.lstsq(X, g.d10.to_numpy(), rcond=None)[0]
        bk = np.linalg.lstsq(X, g.kin10.to_numpy(), rcond=None)[0]
        # stratified contrast: braking (a_bar <= -1.5) vs steady (|a_bar| < 0.3) within (d0 5 m bin, v0 2 m/s bin)
        g = g.assign(grp=np.select([g.a_bar <= -1.5, g.a_bar.abs() < 0.3], ["brake", "steady"], "mid"),
                     cell=(g.d0 // 5).astype(int).astype(str) + "|" + (g.v0 // 2).astype(int).astype(str))
        c = g[g.grp != "mid"].groupby(["cell", "grp"]).agg(d10=("d10", "mean"), kin=("kin10", "mean"), occ10=("occ10", "mean"),
                                                             n=("d10", "size")).unstack("grp").dropna()
        wgt = np.minimum(c["n"]["brake"], c["n"]["steady"])
        diff = lambda col: float(np.average(c[col]["brake"] - c[col]["steady"], weights=wgt)) if len(c) else np.nan
        rows.append({"subset": name, "n": len(g), "n_routes": g.base_id.nunique(), "coef_a_bar_data_m_per_mps2": b[3],
                     "coef_a_bar_kinematic": bk[3], "n_brake": int((g.grp == "brake").sum()),
                     "brake_minus_steady_d10_data_m": diff("d10"), "brake_minus_steady_d10_kinematic_m": diff("kin"),
                     "brake_minus_steady_occ10_data": diff("occ10"), "strata": len(c)})
    # does braking in the future reveal a hazard that is not yet there? anchors with a free lane at t
    f = t[~t.occ0]
    fb = f[f.a_bar <= -1.5]
    fs = f[f.a_bar.abs() < 0.3]
    rows.append({"subset": "free lane at t: P(occ at t+2 s)", "n": len(f), "n_routes": f.base_id.nunique(),
                 "n_brake": len(fb), "p_occ10_brake": fb.occ10.mean(), "p_occ10_steady": fs.occ10.mean(),
                 "p_occ10_all": f.occ10.mean()})
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "confound.csv", index=False, float_format="%.4f")
    return r


# ================================================================ stored W runs (CPU)

def stored() -> dict:
    """From the stored W seed runs: (1) probe AUC against the ground-truth label on the test-fold anchors of each pair
    class (oracle src = the probe on real z_{t0+h}), (2) the pair AUC a perfect probe would get (the label itself),
    (3) the oracle / pred pair AUC restricted to pairs whose x+ label is on at t0+h, (4) the long action test split
    by the kinematic answer."""
    m = _meta()
    lab = {"ped": m.ped.fillna(False).to_numpy(bool), "cutin": m.occ.fillna(False).to_numpy(bool),
           "obstacle": (m.a.fillna(0).to_numpy() > 0.5)}
    rows, act = [], []
    for s, d in W._seed_dirs().items():
        pairs = pd.read_parquet(d / "pairs.parquet")
        sc = pd.read_parquet(d / "pair_scores.parquet")
        sc = sc[sc.probe.isin(list(W.CLS_PROBE.values()))]
        w = sc.pivot_table(index=["pair", "h", "src", "probe"], columns="side", values="score").reset_index()
        w = w.join(pairs[["cls", "kind", "r_plus", "r_other", "base_id"]], on="pair")
        w = w[(w.kind == "pair") & w.cls.isin(list(W.CLS_PROBE))]
        w = w[w.probe == w.cls.map(W.CLS_PROBE)]
        for (cls, h, src), g in w.groupby(["cls", "h", "src"]):
            y = lab[cls]
            yp, yo = y[g.r_plus.to_numpy() + h], y[g.r_other.to_numpy() + h]
            s_all, y_all = np.r_[g.plus, g.other], np.r_[yp, yo]
            on = yp & ~yo
            # ego motion matched: both sides' speed at t0 + h within 0.5 m/s and speed at t0 within 0.5 m/s (the expert
            # has not yet reacted differently), so the probe cannot be reading the ego's own braking
            vv = m.v.to_numpy()
            same = (np.abs(vv[g.r_plus.to_numpy() + h] - vv[g.r_other.to_numpy() + h]) < 0.5) & \
                   (np.abs(vv[g.r_plus.to_numpy()] - vv[g.r_other.to_numpy()]) < 0.5)
            onm = on & same
            rows.append({"pair_auc_label_on_speed_matched": _auc(np.r_[np.ones(onm.sum()), np.zeros(onm.sum())],
                                                                 np.r_[g.plus.to_numpy()[onm], g.other.to_numpy()[onm]]),
                         "n_label_on_speed_matched": int(onm.sum()),
                         "probe_auc_vs_label_speed_matched": _auc(np.r_[yp[same], yo[same]],
                                                                  np.r_[g.plus.to_numpy()[same], g.other.to_numpy()[same]]),
                         "speed_only_pair_auc_label_on": _auc(np.r_[np.ones(on.sum()), np.zeros(on.sum())],
                                                              -np.r_[vv[g.r_plus.to_numpy()[on] + h], vv[g.r_other.to_numpy()[on] + h]])})
            rows[-1].update({"seed": s, "cls": cls, "h": h, "src": src, "n_pairs": len(g),
                         "x_plus_label_on": yp.mean(), "x_minus_label_on": yo.mean(),
                         "label_pair_auc_ceiling": _auc(np.r_[np.ones(len(g)), np.zeros(len(g))], np.r_[yp, yo]),
                         "probe_auc_vs_label": _auc(y_all, s_all),
                         "pair_auc": _auc(np.r_[np.ones(len(g)), np.zeros(len(g))], s_all),
                         "pair_auc_label_on": _auc(np.r_[np.ones(on.sum()), np.zeros(on.sum())],
                                                   np.r_[g.plus.to_numpy()[on], g.other.to_numpy()[on]]),
                         "n_label_on": int(on.sum())})
        a = pd.read_parquet(d / "action_scores.parquet")
        a = a[(a.test == "long") & (a.h == 10)]
        v0 = m.v.to_numpy()[a.row.to_numpy()]
        kin = _kin_d(v0, W._brake(v0)) - _kin_d(v0, np.zeros((len(v0), W.FUT)))
        d0 = m.d_front.to_numpy()[a.row.to_numpy()]
        for name, k in (("all", np.ones(len(a), bool)), ("d0 < 10 m", d0 < 10), ("d0 10-20 m", (d0 >= 10) & (d0 < 20)),
                        ("d0 >= 20 m", d0 >= 20), ("v0 < 6", v0 < 6), ("v0 >= 6", v0 >= 6)):
            g = a[k]
            act.append({"seed": s, "subset": name, "n": int(k.sum()), "correct": float((g.d_alt > g.d_base).mean()),
                        "pred_brake_minus_hold_m": float((g.d_alt - g.d_base).mean()),
                        "kinematic_brake_minus_hold_m": float(kin[k].mean())})
    r = pd.DataFrame(rows).groupby(["cls", "h", "src"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    a = pd.DataFrame(act).groupby("subset", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "stored_probe_quality.csv", index=False, float_format="%.4f")
    a.to_csv(OUT / "stored_long_action.csv", index=False, float_format="%.4f")
    return {"probe": r, "action": a}


# ================================================================ pedestrian probe on real features (GPU)

def _feats_ba(names: pd.Series) -> dict:
    """P5 v1 BA features aligned to frame names: V-JEPA 2 last_mean, SigLIP2 patch_mean, Qwen L18_last, YOLO tokens."""
    import os
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from . import n6_backbones as N6, night2_n4 as N4
    t = pd.DataFrame({"frame_name": names.to_numpy()})
    out = {}
    f = N6.load_backbones(t, ("vjepa2", "siglip2", "qwen"))
    out["V-JEPA 2 last_mean"] = f["vjepa2 last_mean"]
    out["SigLIP2 patch_mean"] = f["siglip2 patch_mean"]
    out["Qwen3-VL-4B L18_last"] = f["qwen L18_last"]
    idx = pd.read_parquet(W.proc("carla_p5v1_ba", "index.parquet")).frame_name
    tok = np.load(N4.out("tokB.npy"), mmap_mode="r")
    out["YOLO26x image-plane tokens"] = np.asarray(tok[pd.Series(np.arange(len(idx)), index=idx)[names].to_numpy()], np.float32)
    return out


def _oof_logreg(X, y, groups, k: int = 5, seed: int = 0) -> np.ndarray:
    import torch
    rng = np.random.RandomState(seed)
    g = np.unique(groups)
    fold = pd.Series(np.arange(len(g)) % k, index=g[rng.permutation(len(g))])[groups].to_numpy()
    s = np.zeros(len(y), np.float32)
    Xt = torch.as_tensor(X, device="cuda", dtype=torch.float32)
    for f in range(k):
        tr, te = fold != f, fold == f
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0).clamp_min(1e-4)
        w, b = W.fit_logreg((Xt[tr] - mu) / sd, torch.as_tensor(y[tr], device="cuda", dtype=torch.float32))
        s[te] = (((Xt[te] - mu) / sd) @ w + b).cpu().numpy()
    return s


def pedprobe() -> pd.DataFrame:
    """Frames: P5 v1 BA and PDM obs rows of the pedestrian families (x+, x- and weather null worlds together) --
    the frames W's pedestrian pair test is made of. Route-grouped 5-fold OOF logistic probe (N2's objective)."""
    m = _meta()
    z = np.load(W.wdir("z.npy"), mmap_mode="r")
    rows = []
    for sets, extra in ((("carla_p5v1_ba",), True), (("carla_p5v1_pdm",), False)):
        k = np.flatnonzero(m.set.isin(sets).to_numpy() & (m.cls == "ped").to_numpy())
        mm = m.iloc[k]
        y = mm.ped.fillna(False).to_numpy(bool)
        feats = {"Cinque temporal": np.asarray(z[k, :W.D_OP], np.float32), "V-JEPA 2 mean": np.asarray(z[k, W.D_OP:], np.float32),
                 "W latent (temporal + V-JEPA mean)": np.asarray(z[k], np.float32),
                 "ego kinematics only (v, a_prev, w_prev)": mm[["v", "a_prev", "w_prev"]].to_numpy(np.float32)}
        if extra:
            feats.update(_feats_ba(mm.frame_name))
        # same-k x+ / x- pairs where the x+ label is on and the ego speed agrees within 0.5 m/s (expert not yet reacting)
        key = mm.base_id.astype(str) + "|" + mm.seed.astype(str) + "|" + mm.k.astype(str)
        mw = mm.world.to_numpy()
        mk = pd.Series(np.flatnonzero(mw == "minus"), index=key.to_numpy()[mw == "minus"])
        mk = mk[~mk.index.duplicated()]
        pl = np.flatnonzero((mw == "plus") & y)
        mi = mk.reindex(key.to_numpy()[pl]).to_numpy()
        pl, mi = pl[~np.isnan(mi)], mi[~np.isnan(mi)].astype(int)
        vv = mm.v.to_numpy()
        pl_m, mi_m = pl[np.abs(vv[pl] - vv[mi]) < 0.5], mi[np.abs(vv[pl] - vv[mi]) < 0.5]
        for name, X in feats.items():
            s = _oof_logreg(X, y.astype(np.float32), mm.base_id.to_numpy())
            plus = mm.world.to_numpy() == "plus"
            pair_on = _auc(np.r_[np.ones(len(pl_m)), np.zeros(len(mi_m))], np.r_[s[pl_m], s[mi_m]])
            rows.append({"set": sets[0], "feature": name, "dim": X.shape[1], "n": len(y), "n_routes": mm.base_id.nunique(),
                         "pos_rate": y.mean(), "oof_auc": _auc(y, s),
                         "oof_auc_x_plus_only": _auc(y[plus], s[plus]),
                         "oof_auc_label_vs_minus_world": _auc(np.r_[np.ones((plus & y).sum()), np.zeros((mm.world == "minus").sum())],
                                                              np.r_[s[plus & y], s[(mm.world == "minus").to_numpy()]]),
                         "pair_auc_label_on_speed_matched": pair_on, "n_pairs_speed_matched": len(pl_m)})
            log.info("%s %s: OOF AUC %.3f", sets[0], name, rows[-1]["oof_auc"])
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "pedprobe.csv", index=False, float_format="%.4f")
    return r


# ================================================================ predictor variants (GPU)

def _loss_for(kind):
    import torch
    if kind == "full":
        return W.block_mse
    return lambda pred, tgt: ((pred - tgt) ** 2).mean()


def variant(kind: str, rl, seed: int = 0, steps: int | None = None, folds_only=None) -> dict:
    """W's `run` with one change (kind): 'full' as W; 'vj' / 'op' drop one z block; 'noact' zeroes the future actions
    in training and at test. Readouts per test fold on every anchor and on W's long-test hazard anchors."""
    import torch
    torch.manual_seed(seed)
    meta = _meta()
    z = np.load(W.wdir("z.npy"), mmap_mode="r")
    cols = {"full": slice(None), "vj": slice(W.D_OP, None), "op": slice(0, W.D_OP), "noact": slice(None)}[kind]
    z = np.ascontiguousarray(z[:, cols])
    W.block_mse = _loss_for(kind)                 # train / evaluate read the module global
    fold, inner = W.folds(meta, seed)
    starts_all = np.flatnonzero(meta.win_start.to_numpy())
    anchors_all = starts_all + W.HIST - 1
    anc_fold = fold[anchors_all]
    rec, hz_rows = [], []
    for f in (folds_only or range(5)):
        t0 = time.time()
        tr_rows = np.flatnonzero((fold != f) & ~inner)
        va_rows = np.flatnonzero((fold != f) & inner)
        data = W.Data(meta, z, tr_rows)
        if kind == "noact":
            data.fact.zero_()
        st_tr = starts_all[(anc_fold != f) & ~inner[anchors_all]]
        st_va = starts_all[(anc_fold != f) & inner[anchors_all]]
        torch.manual_seed(seed * 100 + f)
        model = W.build_model(dz=z.shape[1]).cuda()
        info = W.train(model, data, st_tr, st_va, seed * 100 + f, rl, f"{kind}/fold{f}", steps=steps)
        P = W.probes(data, meta, tr_rows, va_rows)
        Xv = data.Z[torch.as_tensor(va_rows, device="cuda")].float()
        wv, muv, _ = W.fit_ridge(data.Z[torch.as_tensor(tr_rows, device="cuda")].float(),
                                 torch.as_tensor(meta.v.to_numpy(np.float32)[tr_rows], device="cuda"), Xv,
                                 torch.as_tensor(meta.v.to_numpy(np.float32)[va_rows], device="cuda"))
        vprobe = lambda X: (X @ wv + muv).cpu().numpy()
        # ---- latent error: pred / persistence / shuffled future actions, all test anchors
        st = starts_all[anc_fold == f]
        stt = torch.as_tensor(st, device="cuda")
        _, _, fa, zf = data.batch(stt)
        pred = W.predict(model, data, stt, None if kind != "noact" else torch.zeros_like(fa))
        g = torch.Generator(device="cuda").manual_seed(seed)
        perm = torch.randperm(len(st), device="cuda", generator=g)
        fshuf = fa[perm] * torch.as_tensor(W.ACT_SCALE, device="cuda")
        pshuf = W.predict(model, data, stt, torch.zeros_like(fshuf) if kind == "noact" else fshuf)
        zt0 = data.Z[stt + W.HIST - 1].float()[:, None]
        for h in (5, 10):
            rec.append({"fold": f, "h": h, "n": len(st), "mse_pred": float(W.block_mse(pred[:, h - 1], zf[:, h - 1])),
                        "mse_persist": float(W.block_mse(zt0[:, 0], zf[:, h - 1])),
                        "mse_shuffled_actions": float(W.block_mse(pshuf[:, h - 1], zf[:, h - 1])),
                        "best_val": info["best_val"]})
        del pred, pshuf, zf
        # ---- hazard anchors: W's long test (hold / brake) plus the logged actions
        an = anchors_all[anc_fold == f]
        mm = meta.iloc[an]
        hz = an[(mm.world.isin(["plus", "x10"]) & mm.occ.fillna(False).astype(bool) & (mm.v >= W.V_HAZ)).to_numpy()]
        fut = hz[:, None] + np.arange(W.FUT)
        w_exp, a_exp = meta.w_next.to_numpy(np.float32)[fut], meta.a_next.to_numpy(np.float32)[fut]
        v0 = meta.v.to_numpy(np.float32)[hz]
        acts = {"hold": np.stack([np.zeros_like(w_exp), w_exp], -1), "brake": np.stack([W._brake(v0), w_exp], -1),
                "logged": np.stack([a_exp, w_exp], -1)}
        sth = torch.as_tensor(hz - (W.HIST - 1), device="cuda")
        out = {"row": hz, "fold": np.full(len(hz), f)}
        for k, a in acts.items():
            p = W.predict(model, data, sth, torch.as_tensor(a, device="cuda") * (0 if kind == "noact" else 1))[:, -1]
            out[f"d_{k}"] = W.apply_probes({"d": P["d_front"]}, p)["d"]
            out[f"v_{k}"] = vprobe(p)
        zo = data.Z[torch.as_tensor(hz + W.FUT, device="cuda")].float()
        out["d_oracle"], out["v_oracle"] = W.apply_probes({"d": P["d_front"]}, zo)["d"], vprobe(zo)
        z0 = data.Z[torch.as_tensor(hz, device="cuda")].float()
        out["d_persist"], out["v_persist"] = W.apply_probes({"d": P["d_front"]}, z0)["d"], vprobe(z0)
        hz_rows.append(pd.DataFrame(out))
        rl.info(f"{kind} fold {f}: best inner-val {info['best_val']:.4f}, {len(st)} test anchors, {len(hz)} hazard anchors, "
                f"{time.time() - t0:.0f} s")
        del data, model
        torch.cuda.empty_cache()
    d = rl.dir
    pd.DataFrame(rec).to_csv(d / "latent_error.csv", index=False)
    pd.concat(hz_rows, ignore_index=True).to_parquet(d / "hazard.parquet", index=False)
    return {"kind": kind, "dir": str(d)}


# ================================================================ report

def _latest(kind) -> Path | None:
    fs = sorted((data_dir() / "runs" / "nq4" / "wdiag" / f"variant-{kind}").glob("*/hazard.parquet"))
    return fs[-1].parent if fs else None


def report() -> dict:
    m = _meta()
    rows = []
    for kind in KINDS:
        d = _latest(kind)
        if d is None:
            continue
        le = pd.read_csv(d / "latent_error.csv")
        hz = pd.read_parquet(d / "hazard.parquet")
        v0 = m.v.to_numpy()[hz.row.to_numpy()]
        fut = hz.row.to_numpy()[:, None] + np.arange(W.FUT)
        a_bar = m.a_next.to_numpy()[fut].mean(1)
        vt = m.v.to_numpy()[hz.row.to_numpy() + W.FUT]
        for h, g in le.groupby("h"):
            wt = g.n / g.n.sum()
            r = {"variant": kind, "h_s": h * W.DT, "mse_pred_over_persist": float((wt * g.mse_pred).sum() / (wt * g.mse_persist).sum()),
                 "mse_shuffled_over_true_actions": float((wt * g.mse_shuffled_actions).sum() / (wt * g.mse_pred).sum())}
            if h == 10:
                bh = hz.d_brake - hz.d_hold
                r.update({"n_hazard": len(hz), "brake_gt_hold_d_front": float((bh > 0).mean()),
                          "brake_minus_hold_d_front_m": float(bh.mean()),
                          "kinematic_brake_minus_hold_m": float((_kin_d(v0, W._brake(v0)) - _kin_d(v0, np.zeros((len(v0), W.FUT)))).mean()),
                          "brake_lt_hold_speed": float((hz.v_brake < hz.v_hold).mean()),
                          "brake_minus_hold_speed_mps": float((hz.v_brake - hz.v_hold).mean()),
                          "kinematic_brake_minus_hold_speed_mps": float((np.maximum(0, v0 + W._brake(v0).sum(1) * W.DT) - v0).mean()),
                          "corr_logged_a_bar_vs_pred_d_shift": float(np.corrcoef(a_bar, hz.d_logged - hz.d_hold)[0, 1]),
                          "corr_logged_a_bar_vs_pred_v_shift": float(np.corrcoef(a_bar, hz.v_logged - hz.v_hold)[0, 1]),
                          "speed_probe_mae_oracle_mps": float(np.abs(hz.v_oracle - vt).mean()),
                          "speed_mae_logged_pred_mps": float(np.abs(hz.v_logged - vt).mean())})
            rows.append(r)
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "variants.csv", index=False, float_format="%.4f")
    return {"variants": r}


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("confound", "stored", "pedprobe", "variant", "report"))
    ap.add_argument("--kind", default="full", choices=KINDS)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--folds", default="")
    a = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    if a.step == "variant":
        rl = RunLog("nq4", "wdiag", f"variant-{a.kind}")
        r = variant(a.kind, rl, steps=a.steps or None, folds_only=[int(x) for x in a.folds.split(",")] if a.folds else None)
        rl.info(json.dumps(r))
        rl.close()
        return
    r = {"confound": confound, "stored": stored, "pedprobe": pedprobe, "report": report}[a.step]()
    for v in (r.values() if isinstance(r, dict) else [r]):
        print(v.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
