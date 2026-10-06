"""op_probe analysis (plans/2026-10-06-dac-localize-prereg.md sections 3-5): token sets, probes per stage, decoders.

  select                      navtest token sets (F / PP / R / FF from op_parity's gap tables) + the small-read rows -> runs/op_probe/sets/
  probe  [--small]            ridge probes per stage -> runs/op_probe/probe[-small]/: predicted rasters + corridors on navtest, metrics M1 / M2
  decode [--small]            MLP plan decoders per stage (D-imit / D-hinge) -> navtest poses npz for opb_score.py

Probe = ridge on [stage features, E] (z-scored on the probe train rows, lambda picked on navsim/op-parity-full-dev logs by raster R^2), targets
= 1 m drivable SDF raster (64 x 48, clipped +-10 m) and, separately, the 6 corridor distances. Train rows: navtrain shards s2 (small) or
s2-s4 (full) minus dev logs. Evaluation: navtest (log-disjoint from navtrain). op-train env (torch), one GPU (pool job).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_probe"
SETS = ROOT / "sets"
CACHE = data_dir() / "runs" / "op_parity" / "cache"
GAP = data_dir() / "runs" / "op_parity" / "gap" / "gap_navtest_shap.npz"
J_DAC = 1                                         # TERMS index of DAC in the gap tables
X0, Y0, RES05 = -8.0, -24.0, 0.5                  # opb_labels raster
RES1, NH1, NW1 = 1.0, 64, 48
CLIP = 10.0
S_CORR = (5.0, 10.0, 20.0)
D_MAX = 15.0
# nuPlan ego footprint (Pacifica) about the rear axle: front 4.049 m, rear -1.127 m, half width 1.1485 m
CORNERS = np.array([[4.049, 1.1485], [4.049, -1.1485], [-1.127, 1.1485], [-1.127, -1.1485]])
T_POSE, T_DENSE = np.arange(1, 9) * 0.5, np.arange(0, 41) * 0.1


# ---------------------------------------------------------------- sets
def cmd_select(a):
    from jevdrive.run import Run
    with Run("op_probe", "select", config=vars(a)) as run:
        tab = np.load(CACHE / "lb_navtest" / "tab.npz")
        names = tab["names"]
        z = np.load(GAP, allow_pickle=True)
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        ix = np.array([pos[t] for t in names])
        p2 = z["X2"][0][ix, J_DAC] < 1                            # P2-F-s0 fails DAC (the model whose features are probed)
        p2s1 = z["X2"][1][ix, J_DAC] < 1
        wa = z["Xw"][ix, J_DAC] < 1
        sets = {"F": p2 & ~wa, "PP": ~p2 & ~wa, "R": wa & ~p2, "FF": p2 & wa, "F_both_seeds": p2 & p2s1 & ~wa}
        rng = np.random.default_rng(0)
        small = np.sort(np.r_[rng.choice(np.flatnonzero(sets["F"]), 200, replace=False), rng.choice(np.flatnonzero(sets["PP"]), 200, replace=False)])
        pp1500 = np.sort(rng.choice(np.flatnonzero(sets["PP"]), 1500, replace=False))
        SETS.mkdir(parents=True, exist_ok=True)
        np.savez(SETS / "navtest_sets.npz", tokens=names, log=tab["log"], **sets)
        np.save(SETS / "small400_rows.npy", small)
        np.save(SETS / "pp1500_rows.npy", pp1500)
        (SETS / "small400_tokens.txt").write_text("\n".join(names[small]) + "\n")
        ev = np.sort(np.r_[np.flatnonzero(sets["F"] | sets["R"] | sets["FF"]), pp1500])
        (SETS / "eval_tokens.txt").write_text("\n".join(names[ev]) + "\n")     # F u R u FF u PP-1500: the scoring set of decoders / ablations
        run.summary.update({k: int(v.sum()) for k, v in sets.items()} | {"small": len(small), "eval": len(ev)})
        run.info(json.dumps(run.summary))


# ---------------------------------------------------------------- geometry
def raster1m(sdf05):
    """(N, 128, 96) 0.5 m SDF -> (N, 64, 48) 1 m (2 x 2 mean), clipped."""
    s = sdf05.astype(np.float32).reshape(len(sdf05), NH1, 2, NW1, 2).mean((2, 4))
    return np.clip(s, -CLIP, CLIP)


def sample(grid, res, xy):
    """Bilinear SDF at points. grid (H, W) with cell centres X0 + (i + .5) res; xy (..., 2) -> (...)."""
    from scipy.ndimage import map_coordinates
    i = (xy[..., 0] - X0) / res - 0.5
    j = (xy[..., 1] - Y0) / res - 0.5
    return map_coordinates(grid, np.stack([i.ravel(), j.ravel()]), order=1, mode="nearest").reshape(xy.shape[:-1])


def path_points(fut):
    """Logged future (8, 3) -> points / headings at arc lengths S_CORR (straight extrapolation past the end)."""
    P = np.vstack([[0.0, 0.0, 0.0], fut])
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    out = []
    for sc in S_CORR:
        if sc <= s[-1] and s[-1] > 0.5:
            k = min(np.searchsorted(s, sc), len(s) - 1)
            k = max(k, 1)
            f = (sc - s[k - 1]) / max(seg[k - 1], 1e-6)
            p = P[k - 1, :2] + f * (P[k, :2] - P[k - 1, :2])
            h = P[k - 1, 2] + f * (P[k, 2] - P[k - 1, 2])
        else:
            h = P[-1, 2] if s[-1] > 0.5 else 0.0
            p = P[-1, :2] + (sc - s[-1]) * np.array([np.cos(h), np.sin(h)])
        out.append((p, h))
    return out


def corridor(grid, res, fut):
    """6 free distances (left / right at S_CORR) from the path to the first SDF < 0 along its normal, capped at D_MAX; 0 when the path point
    itself is outside."""
    d = np.arange(0, D_MAX + 1e-6, 0.25)
    res_ = []
    for p, h in path_points(fut):
        n = np.array([-np.sin(h), np.cos(h)])
        for sgn in (1, -1):
            v = sample(grid, res, p[None] + sgn * d[:, None] * n[None])
            k = np.flatnonzero(v < 0)
            res_.append(D_MAX if not len(k) else (0.0 if k[0] == 0 else d[k[0] - 1] + 0.25 * v[k[0] - 1] / (v[k[0] - 1] - v[k[0]] + 1e-9)))
    return np.array(res_, np.float32)


def dense(p8):
    P = np.vstack([[0, 0, 0], p8])
    t = np.r_[0, T_POSE]
    h = np.unwrap(P[:, 2])
    return np.stack([np.interp(T_DENSE, t, P[:, 0]), np.interp(T_DENSE, t, P[:, 1]), np.interp(T_DENSE, t, h)], -1)


def footprint(p8):
    """(41 * 4, 2) footprint corners of the raw plan linearly interpolated to 0.1 s."""
    d = dense(p8)
    c, s = np.cos(d[:, 2])[:, None], np.sin(d[:, 2])[:, None]
    x = d[:, :1] + c * CORNERS[None, :, 0] - s * CORNERS[None, :, 1]
    y = d[:, 1:2] + s * CORNERS[None, :, 0] + c * CORNERS[None, :, 1]
    return np.stack([x, y], -1).reshape(-1, 2)


def margin(grid, res, p8):
    """Smallest SDF over the raw plan's footprint corners (m; < 0 = a corner leaves the drivable surface)."""
    return float(sample(grid, res, footprint(p8)).min())


# ---------------------------------------------------------------- data
def labels(split_tag):
    z = np.load(ROOT / "labels" / f"{split_tag}.npz")
    return {t: i for i, t in enumerate(z["tokens"].tolist())}, z


def feat_files(model, data):
    d = ROOT / "feats" / model
    if model != "WA":
        return [d / f"{data}.npz"]
    return sorted(d.glob(f"{data}.s*of3.npz")) + ([d / f"{data}-small400.npz"] if (d / f"{data}-small400.npz").exists() else [])


def feats(model, data, stage, tokens):
    """Stage features (N, d) float32 for tokens of one op_parity cache dir. model: P2-F-s0 / P0 / WA; stage V / M / T / H / Cf / Ca."""
    zs = [np.load(f) for f in feat_files(model, data)]
    tok = np.concatenate([z["tokens"] for z in zs])
    pos = {t: i for i, t in enumerate(tok.tolist())}
    X = np.concatenate([z[stage] for z in zs])
    return np.stack([X[pos[t]] for t in tokens]).reshape(len(tokens), -1).astype(np.float32)


def has_feats(model, data, tokens):
    toks = set()
    for f in feat_files(model, data):
        if f.exists():
            toks |= set(np.load(f)["tokens"].tolist())
    return np.array([t in toks for t in tokens])


STAGES = [("E", None, None), ("P2-V", "P2-F-s0", "V"), ("P2-M", "P2-F-s0", "M"), ("P2-T", "P2-F-s0", "T"), ("P2-H", "P2-F-s0", "H"),
          ("P0-T", "P0", "T"), ("P0-H", "P0", "H"), ("WA-Cf", "WA", "Cf"), ("WA-Ca", "WA", "Ca"), ("WA-T", "WA", "T"), ("WA-H", "WA", "H")]


# ---------------------------------------------------------------- ridge
def ridge_fit(Xtr, Ys_tr, Xdv, Ys_dv, lams, dev):
    """z-scored primal ridge with one eigendecomposition for several target blocks; lambda (relative to the mean eigenvalue) picked per block
    by dev R^2 averaged over that block's outputs."""
    import torch
    X = torch.as_tensor(Xtr, device=dev)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    X = (X - mu) / sd
    G = (X.T @ X).double()
    ev, Q = torch.linalg.eigh(G)
    del G
    scale = float(ev.clamp_min(0).mean())
    Xd = ((torch.as_tensor(Xdv, device=dev) - mu) / sd).double()
    out = []
    for Ytr, Ydv in zip(Ys_tr, Ys_dv):
        Y = torch.as_tensor(Ytr, device=dev)
        ym = Y.mean(0)
        QtXtY = Q.T @ (X.T @ (Y - ym)).double()
        Yd = torch.as_tensor(Ydv, device=dev).double()
        var = ((Yd - Yd.mean(0)) ** 2).mean(0).clamp_min(1e-6)
        best = None
        for lam in lams:
            W = Q @ (QtXtY / (ev[:, None] + lam * scale))
            r2 = float((1 - ((Xd @ W + ym.double() - Yd) ** 2).mean(0) / var).mean())
            if best is None or r2 > best[0]:
                best = (r2, lam, W)
        r2, lam, W = best
        out.append(dict(W=W.float(), mu=mu, sd=sd, ym=ym, lam=lam, dev_r2=r2))
    return out


def ridge_predict(m, X, dev, bs=4096):
    import torch
    out = []
    for i in range(0, len(X), bs):
        x = (torch.as_tensor(X[i:i + bs], device=dev) - m["mu"]) / m["sd"]
        out.append((x @ m["W"] + m["ym"]).cpu().numpy())
    return np.concatenate(out)


def auc(score, y):
    """P(score of a positive > score of a negative), ties half."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def auc_ci(score, y, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_of = [np.flatnonzero(inv == g) for g in range(len(ug))]
    bs = []
    for _ in range(B):
        ix = np.concatenate([idx_of[g] for g in rng.integers(0, len(ug), len(ug))])
        bs.append(auc(score[ix], y[ix]))
    return np.nanpercentile(bs, [2.5, 97.5]).tolist()


def twoset(v, A, B, groups, n=2000, seed=0):
    """mean(v[A]) - mean(v[B]) with logs resampled (both sets drawn from the same resampled logs)."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    ok = np.isfinite(v)
    sA = np.bincount(inv, np.where(A & ok, v, 0), len(ug)); cA = np.bincount(inv, (A & ok).astype(float), len(ug))
    sB = np.bincount(inv, np.where(B & ok, v, 0), len(ug)); cB = np.bincount(inv, (B & ok).astype(float), len(ug))
    idx = rng.integers(0, len(ug), (n, len(ug)))
    d = sA[idx].sum(1) / cA[idx].sum(1) - sB[idx].sum(1) / cB[idx].sum(1)
    return float(sA.sum() / cA.sum() - sB.sum() / cB.sum()), *np.nanpercentile(d, [2.5, 97.5]).tolist()


def train_tokens(small):
    from jevdrive.data import splits
    shards = [2] if small else [2, 3, 4]
    toks = np.concatenate([np.load(CACHE / f"navtrain_full.s{k}of12" / "tab.npz")["names"] for k in shards])
    datas = np.concatenate([[f"navtrain_full.s{k}of12"] * len(np.load(CACHE / f"navtrain_full.s{k}of12" / "tab.npz")["names"]) for k in shards])
    dv = splits.load("navsim/op-parity-full-dev")
    is_dev = dv.mask(toks)
    return toks, datas, is_dev, dv


def stage_matrix(name, model, stage, toks, datas, ego):
    """[stage features, E] for tokens grouped by cache dir."""
    if model is None:
        return ego
    X = np.zeros((len(toks), 0), np.float32)
    parts = []
    for d in dict.fromkeys(datas):
        m = datas == d
        parts.append((m, feats(model, d, stage, toks[m])))
    X = np.zeros((len(toks), parts[0][1].shape[1]), np.float32)
    for m, x in parts:
        X[m] = x
    return np.concatenate([X, ego], 1)


def cmd_probe(a):
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    tag = "probe-small" if a.small else "probe"
    out = ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_probe", tag, config=vars(a)) as run:
        toks, datas, is_dev, dvs = train_tokens(a.small)
        run.use_split(dvs), run.use_split(splits.load("navsim/navtrain")), run.use_split(splits.load("navsim/navtest"))
        lab_pos, LZ = labels("navtrain_s23456")
        li = np.array([lab_pos[t] for t in toks])
        ok = LZ["ok"][li]
        # navtrain targets
        tabs = {d: np.load(CACHE / d / "tab.npz") for d in dict.fromkeys(datas)}
        fut = np.concatenate([tabs[d]["fut"] for d in dict.fromkeys(datas)])
        ego = np.concatenate([tabs[d]["ego"] for d in dict.fromkeys(datas)]).astype(np.float32)
        assert (np.concatenate([tabs[d]["names"] for d in dict.fromkeys(datas)]) == toks).all()
        Yr = raster1m(LZ["sdf"][li]).reshape(len(toks), -1)
        has_fut = ~np.isnan(fut[:, 0, 0])
        t0 = time.time()
        Yc = np.stack([corridor(LZ["sdf"][i].astype(np.float32), RES05, f) if h else np.full(6, np.nan, np.float32)
                       for i, f, h in zip(li, fut, has_fut)])
        run.info(f"train targets: {len(toks)} tokens ({is_dev.sum()} dev), corridors in {time.time() - t0:.0f} s")
        # navtest eval rows
        S = np.load(SETS / "navtest_sets.npz")
        tt = S["tokens"]
        ev_rows = np.load(SETS / "small400_rows.npy") if a.small else np.arange(len(tt))
        ttab = np.load(CACHE / "lb_navtest" / "tab.npz")
        assert (ttab["names"] == tt).all()
        tpos, TZ = labels("navtest")
        ti = np.array([tpos[t] for t in tt[ev_rows]])
        e_toks, e_ego, e_fut = tt[ev_rows], ttab["ego"][ev_rows].astype(np.float32), ttab["fut"][ev_rows]
        e_sdf05 = TZ["sdf"][ti].astype(np.float32)
        e_r1 = raster1m(TZ["sdf"][ti])
        e_hf = ~np.isnan(e_fut[:, 0, 0])
        Yc_true = np.stack([corridor(s, RES05, f) if h else np.full(6, np.nan, np.float32) for s, f, h in zip(e_sdf05, e_fut, e_hf)])
        Yc_true1 = np.stack([corridor(s, RES1, f) if h else np.full(6, np.nan, np.float32) for s, f, h in zip(e_r1, e_fut, e_hf)])
        p2 = np.load(ROOT / "feats" / "P2-F-s0" / "lb_navtest.npz")
        p2pos = {t: i for i, t in enumerate(p2["tokens"].tolist())}
        e_plan = p2["poses"][[p2pos[t] for t in e_toks]]
        m_true = np.array([margin(s, RES05, p) for s, p in zip(e_sdf05, e_plan)])
        m_true1 = np.array([margin(s, RES1, p) for s, p in zip(e_r1, e_plan)])
        sets = {k: S[k][ev_rows] for k in ("F", "PP", "R", "FF", "F_both_seeds")}
        res = {"tokens": e_toks, "log": S["log"][ev_rows], "corr_true": Yc_true, "corr_true1": Yc_true1, "margin_true": m_true,
               "margin_true1": m_true1, **{f"set_{k}": v for k, v in sets.items()}}
        lams = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
        rows = []
        for name, model, stage in STAGES:
            if model is not None and (not has_feats(model, datas[0], toks[:50]).all() or not has_feats(model, "lb_navtest", e_toks).all()):
                run.info(f"{name}: features missing, skipped")
                continue
            t1 = time.time()
            Xa = stage_matrix(name, model, stage, toks, datas, ego)
            use = ok & has_fut & ~np.isnan(Yc).any(1)
            tr, dv_ = use & ~is_dev, use & is_dev
            mr, mc = ridge_fit(Xa[tr], [Yr[tr], Yc[tr]], Xa[dv_], [Yr[dv_], Yc[dv_]], lams, dev)
            del Xa
            Xe = stage_matrix(name, model, stage, e_toks, np.array(["lb_navtest"] * len(e_toks)), e_ego)
            pr = ridge_predict(mr, Xe, dev).reshape(-1, NH1, NW1)
            pc = ridge_predict(mc, Xe, dev)
            pc_r = np.stack([corridor(g, RES1, f) if h else np.full(6, np.nan, np.float32) for g, f, h in zip(pr, e_fut, e_hf)])
            mp = np.array([margin(g, RES1, p) for g, p in zip(pr, e_plan)])
            res[f"{name}/corr_direct"], res[f"{name}/corr_raster"], res[f"{name}/margin"] = pc, pc_r, mp
            np.save(out / f"raster_{name}.npy", pr.astype(np.float16))
            r2_test = float(1 - ((pr.reshape(len(pr), -1) - e_r1.reshape(len(pr), -1)) ** 2).mean() / e_r1.reshape(len(pr), -1).var(0).mean())
            rows.append(dict(stage=name, lam_raster=mr["lam"], dev_r2_raster=mr["dev_r2"], lam_corr=mc["lam"], dev_r2_corr=mc["dev_r2"],
                             test_r2_raster=r2_test, dim=int(Xe.shape[1]), fit_s=time.time() - t1))
            run.info(json.dumps(rows[-1]))
            torch.cuda.empty_cache()
        np.savez(out / "probe_outputs.npz", **res)
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "fits.csv", index=False)
        summ = summarize(res, [r["stage"] for r in rows])
        summ.to_csv(out / "metrics.csv", index=False)
        run.info("\n" + summ.to_string())
        run.summary["out"] = str(out)


def summarize(res, stages):
    """M1 (corridor MAE, skill vs E) and M2 (AUC of -margin for F vs PP) per stage, with log-cluster CIs."""
    import pandas as pd
    from jevdrive import stats
    F, PP = res["set_F"], res["set_PP"]
    lg = res["log"]
    ct = res["corr_true"]
    rows = []
    base = {}
    for name in stages:
        r = {"stage": name}
        for kind in ("direct", "raster"):
            err = np.nanmean(np.abs(res[f"{name}/corr_{kind}"] - ct), 1)
            for sn, m in (("F", F), ("PP", PP)):
                b = stats.bootstrap(err[m], groups=lg[m])
                r[f"mae_{kind}_{sn}"], r[f"mae_{kind}_{sn}_lo"], r[f"mae_{kind}_{sn}_hi"] = b["mean"], b["lo"], b["hi"]
            if name == "E":
                base[kind] = err
            if kind in base:
                for sn, m in (("F", F), ("PP", PP)):
                    r[f"skill_{kind}_{sn}"] = 1 - np.nanmean(err[m]) / np.nanmean(base[kind][m])
            r[f"excess_{kind}_F_minus_PP"], r[f"excess_{kind}_lo"], r[f"excess_{kind}_hi"] = twoset(err, F, PP, lg)
        sel = F | PP
        y = F[sel]
        sc = -res[f"{name}/margin"][sel]
        r["auc_F_vs_PP"] = auc(sc, y)
        r["auc_lo"], r["auc_hi"] = auc_ci(sc, y, lg[sel])
        r["margin_F_med"] = float(np.median(res[f"{name}/margin"][F]))
        r["margin_PP_med"] = float(np.median(res[f"{name}/margin"][PP]))
        rows.append(r)
    sel = F | PP
    for nm, key in (("TRUE 0.5 m", "margin_true"), ("TRUE 1 m", "margin_true1")):
        rows.append({"stage": nm, "auc_F_vs_PP": auc(-res[key][sel], F[sel]), "margin_F_med": float(np.median(res[key][F])),
                     "margin_PP_med": float(np.median(res[key][PP]))})
    err1 = np.nanmean(np.abs(res["corr_true1"] - ct), 1)
    rows.append({"stage": "TRUE 1 m raster (resolution floor)", "mae_raster_F": float(np.nanmean(err1[F])), "mae_raster_PP": float(np.nanmean(err1[PP]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decoders (M3)
DEC_STAGES = ["E", "P2-V", "P2-M", "P2-T", "P2-H", "WA-Cf", "WA-Ca", "WA-H"]


def _interp_matrix():
    """(41, 9) linear interpolation of [origin, 8 poses] onto the 0.1 s grid."""
    t = np.r_[0, T_POSE]
    M = np.zeros((41, 9))
    for k, tt in enumerate(T_DENSE):
        j = min(np.searchsorted(t, tt, side="right") - 1, 7)
        f = (tt - t[j]) / (t[j + 1] - t[j])
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


def corners_torch(P, M, C):
    """P (B, 8, 3) poses -> (B, 41 * 4, 2) footprint corners of the linearly interpolated plan."""
    import torch
    P0 = torch.cat([torch.zeros_like(P[:, :1]), P], 1)
    d = torch.einsum("kj,bjc->bkc", M, P0)
    c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
    x = d[..., :1] + c * C[None, None, :, 0] - s * C[None, None, :, 1]
    y = d[..., 1:2] + s * C[None, None, :, 0] + c * C[None, None, :, 1]
    return torch.stack([x, y], -1).reshape(len(P), -1, 2)


def sdf_at(grid, xy):
    """grid (B, 1, 128, 96) 0.5 m SDF, xy (B, K, 2) metres -> (B, K) bilinear (border clamp)."""
    import torch
    g = torch.stack([(xy[..., 1] - Y0) / 24.0 - 1.0, (xy[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
    return torch.nn.functional.grid_sample(grid, g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]


def cmd_decode(a):
    import torch
    import torch.nn as nn
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    tag = "decode-small" if a.small else "decode"
    out = ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_probe", tag, seed=0, config=vars(a)) as run:
        toks, datas, is_dev, dvs = train_tokens(a.small)
        run.use_split(dvs), run.use_split(splits.load("navsim/navtrain")), run.use_split(splits.load("navsim/navtest"))
        lab_pos, LZ = labels("navtrain_s23456")
        li = np.array([lab_pos[t] for t in toks])
        tabs = {d: np.load(CACHE / d / "tab.npz") for d in dict.fromkeys(datas)}
        fut = np.concatenate([tabs[d]["fut"] for d in dict.fromkeys(datas)])
        ego = np.concatenate([tabs[d]["ego"] for d in dict.fromkeys(datas)]).astype(np.float32)
        use = LZ["ok"][li] & ~np.isnan(fut[:, 0, 0]) & ~is_dev
        sdf = torch.as_tensor(LZ["sdf"][li[use]], device=dev)[:, None]                        # (n, 1, 128, 96) fp16
        Y = torch.as_tensor(fut[use], device=dev).float()
        tt = np.array([t.strip() for t in open(SETS / "eval_tokens.txt") if t.strip()])
        if a.small:
            tt = np.array([t.strip() for t in open(SETS / "small400_tokens.txt") if t.strip()])
        ttab = np.load(CACHE / "lb_navtest" / "tab.npz")
        tp = {t: i for i, t in enumerate(ttab["names"].tolist())}
        e_ego = ttab["ego"][[tp[t] for t in tt]].astype(np.float32)
        M = torch.as_tensor(_interp_matrix(), device=dev, dtype=torch.float32)
        C = torch.as_tensor(CORNERS, device=dev, dtype=torch.float32)
        res = {"tokens": tt}
        rows = []
        for name, model, stage in [x for x in STAGES if x[0] in DEC_STAGES]:
            if model is not None and not has_feats(model, "lb_navtest", tt).all():
                run.info(f"{name}: navtest features missing, skipped")
                continue
            Xa = torch.as_tensor(stage_matrix(name, model, stage, toks[use], datas[use], ego[use]), device=dev)
            mu, sd = Xa.mean(0), Xa.std(0).clamp_min(1e-6)
            Xa = (Xa - mu) / sd
            Xe = (torch.as_tensor(stage_matrix(name, model, stage, tt, np.array(["lb_navtest"] * len(tt)), e_ego), device=dev) - mu) / sd
            for kind, lam_h in (("imit", 0.0), ("hinge", a.lam_hinge)):
                torch.manual_seed(0)
                net = nn.Sequential(nn.Dropout(0.1), nn.Linear(Xa.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 24)).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
                wu = max(1, a.steps // 20)
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / wu) * 0.5 * (1 + np.cos(np.pi * min(k, a.steps) / a.steps)))
                g = torch.Generator(device=dev).manual_seed(0)
                t1 = time.time()
                for step in range(a.steps):
                    b = torch.randint(0, len(Xa), (a.batch,), device=dev, generator=g)
                    P = net(Xa[b]).view(-1, 8, 3)
                    li_ = nn.functional.huber_loss(P[..., :2], Y[b, :, :2], delta=1.0) + 3.0 * nn.functional.huber_loss(P[..., 2], Y[b, :, 2], delta=0.1)
                    loss = li_
                    if lam_h > 0:
                        v = sdf_at(sdf[b].float(), corners_torch(P, M, C))
                        lh = torch.relu(a.hinge_margin - v).mean()
                        loss = loss + lam_h * lh
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                net.eval()
                with torch.no_grad():
                    Pe = torch.cat([net(Xe[i:i + 2048]).view(-1, 8, 3) for i in range(0, len(Xe), 2048)]).cpu().numpy()
                res[f"{name}|{kind}"] = Pe.astype(np.float32)
                rows.append(dict(stage=name, kind=kind, final_loss=float(loss), imit=float(li_), train_s=time.time() - t1))
                run.info(json.dumps(rows[-1]))
            del Xa, Xe
            torch.cuda.empty_cache()
        np.savez(out / "decoder_poses.npz", **res)
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "fits.csv", index=False)
        run.summary["out"] = str(out / "decoder_poses.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p = sp.add_parser("probe")
    p.add_argument("--small", action="store_true")
    p = sp.add_parser("decode")
    p.add_argument("--small", action="store_true")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lam-hinge", type=float, default=1.0)
    p.add_argument("--hinge-margin", type=float, default=0.3)
    a = ap.parse_args()
    {"select": cmd_select, "probe": cmd_probe, "decode": cmd_decode}[a.cmd](a)
