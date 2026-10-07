"""op_parity turn-probe (plans/2026-10-07-turn-probe-prereg.md): is junction / road-boundary geometry readable from the frozen Cinque tokens
on sharp turns, compared with WA-Cf? Read-out probes only; cached features and labels only (no extraction, no driving model).

  fit    [--small]   (GPU, pool) ridge + 2-layer MLP probes (opb_probe.ridge_fit / mlp_fit_predict, unchanged) on z-scored [X, E] for the arms
                     E / V (Cinque view_39) / WA (WA-Cf) / VJ21 (PCA 512); targets = 1 m ego-frame drivable SDF raster + 8 corridor distances;
                     predictions on the navtrain held-out logs and all of navtest -> $DATA_DIR/runs/op_parity/turn_probe[-small]/pred.npz
                     --small = 4 000 random train rows, ridge only, E / V / WA (the early-stop gate)
  report [--small]   (CPU) strata, log-cluster bootstrap, the pre-registered verdict, tables and figures -> .../turn_probe[-small]/report/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent), str(_R / "experiments/op_probe/scripts")]
import argparse, json, zlib  # noqa: E401,E402

import numpy as np  # noqa: E402

import opb_probe as P  # noqa: E402
import rep as REP  # noqa: E402

OUT = REP.D / "runs" / "op_parity"
X0, Y0 = P.X0, P.Y0
S_CORR = (5.0, 10.0, 15.0, 20.0)
D_MAX = P.D_MAX
ARMS = ("E", "V", "WA", "VJ21")
BINS = (("S5", 0, 5), ("T5-20", 5, 20), ("T20-45", 20, 45), ("T45", 45, 1e9))
BAND, BX, BY = 2.0, (8, 40), (8, 40)                       # |true SDF| <= 2 m; raster rows x in [0, 32) m, cols |y| < 16 m
NB = 4000
FAILS = _R / "experiments/op_parity/results/four_dirs/nav_tokens_navtest.csv"


# ---------------------------------------------------------------- geometry (vectorised opb_probe.sample / path_points / corridor)
def bil(G, res, xy):
    """Bilinear SDF with border clamp. G (N, H, W), xy (N, K, 2) -> (N, K)."""
    H, W = G.shape[1:]
    i = np.clip((xy[..., 0] - X0) / res - 0.5, 0, H - 1)
    j = np.clip((xy[..., 1] - Y0) / res - 0.5, 0, W - 1)
    i0, j0 = np.minimum(i.astype(int), H - 2), np.minimum(j.astype(int), W - 2)
    fi, fj, n = i - i0, j - j0, np.arange(len(G))[:, None]
    g = lambda a, b: G[n, a, b].astype(np.float32)  # noqa: E731
    return g(i0, j0) * (1 - fi) * (1 - fj) + g(i0 + 1, j0) * fi * (1 - fj) + g(i0, j0 + 1) * (1 - fi) * fj + g(i0 + 1, j0 + 1) * fi * fj


def path_points(fut, S):
    """Logged futures (N, 8, 3) -> points (N, len(S), 2) and headings at arc lengths S (straight extrapolation past the end)."""
    Pp = np.concatenate([np.zeros((len(fut), 1, 3)), fut.astype(np.float64)], 1)
    seg = np.linalg.norm(np.diff(Pp[..., :2], axis=1), axis=-1)
    s = np.concatenate([np.zeros((len(fut), 1)), np.cumsum(seg, 1)], 1)
    L, r = s[:, -1], np.arange(len(fut))
    he = np.where(L > 0.5, Pp[:, -1, 2], 0.0)
    pts, hs = [], []
    for sc in S:
        k = np.clip((s < sc).sum(1), 1, 8)
        f = (sc - s[r, k - 1]) / np.maximum(seg[r, k - 1], 1e-6)
        p = Pp[r, k - 1, :2] + f[:, None] * (Pp[r, k, :2] - Pp[r, k - 1, :2])
        h = Pp[r, k - 1, 2] + f * (Pp[r, k, 2] - Pp[r, k - 1, 2])
        ins = (sc <= L) & (L > 0.5)
        pe = Pp[:, -1, :2] + (sc - L)[:, None] * np.stack([np.cos(he), np.sin(he)], 1)
        pts.append(np.where(ins[:, None], p, pe)), hs.append(np.where(ins, h, he))
    return np.stack(pts, 1), np.stack(hs, 1)


def corridor(G, res, fut, S=S_CORR, chunk=2048):
    """(N, 2 len(S)) free distances [left, right] per arc length from the path to the first SDF < 0 along its normal (cap D_MAX; 0 when the
    path point is outside): opb_probe.corridor, batched."""
    d, sg, out = np.arange(0, D_MAX + 1e-6, 0.25), np.array([1.0, -1.0]), []
    for a in range(0, len(G), chunk):
        g, f = G[a:a + chunk], fut[a:a + chunk]
        p, h = path_points(f, S)
        nrm = np.stack([-np.sin(h), np.cos(h)], -1)
        xy = p[:, :, None, None] + sg[None, None, :, None, None] * d[None, None, None, :, None] * nrm[:, :, None, None]
        v = bil(g, res, xy.reshape(len(g), -1, 2)).reshape(len(g), len(S), 2, len(d))
        neg = v < 0
        k = neg.argmax(-1)
        km = np.maximum(k - 1, 0)
        v0, v1 = np.take_along_axis(v, km[..., None], -1)[..., 0], np.take_along_axis(v, k[..., None], -1)[..., 0]
        o = np.where(~neg.any(-1), D_MAX, np.where(k == 0, 0.0, d[km] + 0.25 * v0 / (v0 - v1 + 1e-9)))
        out.append(o.reshape(len(g), -1).astype(np.float32))
    return np.concatenate(out)


def entry(fut, pose):
    """Arc length (m) to the turn entry (first |heading| >= 5 deg on the 0.1 s interpolated logged future; inf if never) and the in-turn flag
    (|heading change over the 1.5 s history| >= 5 deg)."""
    Pp = np.concatenate([np.zeros((len(fut), 1, 3)), fut.astype(np.float64)], 1)
    t, td = np.r_[0, P.T_POSE], P.T_DENSE
    h = np.unwrap(Pp[..., 2], axis=1)
    D = np.stack([np.stack([np.interp(td, t, row[:, c]) for c in (0, 1)] + [np.interp(td, t, hh)], -1) for row, hh in zip(Pp, h)])
    s = np.concatenate([np.zeros((len(fut), 1)), np.cumsum(np.linalg.norm(np.diff(D[..., :2], axis=1), axis=-1), 1)], 1)
    hit = np.abs(np.degrees(D[..., 2])) >= 5
    k = hit.argmax(1)
    return np.where(hit.any(1), s[np.arange(len(fut)), k], np.inf), np.abs(np.degrees(pose[:, 0, 2])) >= 5


# ---------------------------------------------------------------- fit (GPU)
def cmd_fit(a):
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    out = OUT / ("turn_probe-small" if a.small else "turn_probe")
    out.mkdir(parents=True, exist_ok=True)
    arms = ARMS[:3] if a.small else ARMS
    with Run("op_parity", out.name + "-fit", seed=0, config=vars(a)) as run:
        toks, datas, _, _ = P.train_tokens(False)
        s_tr, s_dv, s_te = (splits.load(n) for n in ("navsim/op-parity-s234-train", "navsim/op-parity-s234-dev", "navsim/navtest"))
        for s in (s_tr, s_dv, s_te):
            run.use_split(s)
        splits.check_disjoint(s_tr, s_dv)
        lab_pos, LZ = P.labels("navtrain_s23456")
        li = np.array([lab_pos[t] for t in toks])
        tabs = [np.load(REP.CR / d / "tab.npz") for d in dict.fromkeys(datas)]
        cat = lambda k: np.concatenate([t[k] for t in tabs])  # noqa: E731
        assert (cat("names") == toks).all()
        fut, ego, log, pose = cat("fut"), cat("ego").astype(np.float32), cat("log"), cat("pose")
        sdf = LZ["sdf"][li]
        Yr = P.raster1m(sdf).reshape(len(toks), -1)
        Yc = corridor(sdf, P.RES05, np.nan_to_num(fut))
        chk = np.flatnonzero(~np.isnan(fut[:, 0, 0]))[:100]                    # the batched corridor reproduces opb_probe.corridor
        old_s, P.S_CORR = P.S_CORR, S_CORR
        ref = np.stack([P.corridor(sdf[i].astype(np.float32), P.RES05, fut[i]) for i in chk])
        P.S_CORR = old_s
        eq = float(np.abs(ref - Yc[chk]).max())
        run.info(f"corridor equivalence vs opb_probe.corridor on {len(chk)} rows: max abs diff {eq:.2e} m")
        assert eq < 1e-3, eq
        use = LZ["ok"][li] & ~np.isnan(fut[:, 0, 0])
        tr, dv = use & s_tr.mask(toks), use & s_dv.mask(toks)
        if a.small:
            tr &= np.isin(np.arange(len(toks)), np.random.default_rng(0).choice(np.flatnonzero(tr), 4000, replace=False))
        inner = np.array([zlib.crc32(g.encode()) % 10 == 0 for g in log])        # lambda selection: 10% of the train logs (hash), never dev / navtest
        # eval rows: navtrain held-out logs + all of navtest
        ttab = np.load(REP.CR / REP.TEST / "tab.npz")
        tpos, TZ = P.labels("navtest")
        ti = np.array([tpos[t] for t in ttab["names"]])
        assert s_te.mask(ttab["names"]).all()
        e_ok = TZ["ok"][ti] & ~np.isnan(ttab["fut"][:, 0, 0])
        e_tok = np.r_[toks[dv], ttab["names"][e_ok]]
        e_dat = np.r_[datas[dv], np.array([REP.TEST] * int(e_ok.sum()))]
        e_ego = np.r_[ego[dv], ttab["ego"][e_ok].astype(np.float32)]
        e_fut = np.r_[fut[dv], ttab["fut"][e_ok]]
        e_sdf = np.concatenate([sdf[dv], TZ["sdf"][ti[e_ok]]])
        res = dict(tokens=e_tok, log=np.r_[log[dv], ttab["log"][e_ok]], test=np.r_[np.zeros(dv.sum(), bool), np.ones(e_ok.sum(), bool)], fut=e_fut,
                   pose=np.r_[pose[dv], ttab["pose"][e_ok]], raster_true=P.raster1m(e_sdf).astype(np.float16),
                   corr_true=corridor(e_sdf, P.RES05, e_fut), sdf05=e_sdf)
        run.info(f"rows: train {tr.sum()} (ridge fit {int((tr & ~inner).sum())}, lambda rows {int((tr & inner).sum())}), navtrain held-out {dv.sum()}, "
                 f"navtest {e_ok.sum()} / {len(e_ok)}")
        pca = dict(np.load(REP.OUT / "vj21_pca.npz"))
        lams = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
        fits = []
        for arm in run.tqdm(arms, desc="arms"):
            src = lambda tk, dt: REP.source(arm, tk, dt, pca) if arm != "E" else np.zeros((len(tk), 0), np.float32)  # noqa: E731
            Xa = np.concatenate([src(toks[tr], datas[tr]), ego[tr]], 1)
            Xe = np.concatenate([src(e_tok, e_dat), e_ego], 1)
            it = inner[tr]
            mr, mc = P.ridge_fit(Xa[~it], [Yr[tr][~it], Yc[tr][~it]], Xa[it], [Yr[tr][it], Yc[tr][it]], lams, dev)
            res[f"{arm}/ridge/raster"] = P.ridge_predict(mr, Xe, dev).reshape(-1, P.NH1, P.NW1).astype(np.float16)
            res[f"{arm}/ridge/corr"] = P.ridge_predict(mc, Xe, dev)
            fits.append(dict(arm=arm, dim=int(Xa.shape[1]), lam_raster=mr["lam"], inner_r2_raster=mr["dev_r2"], lam_corr=mc["lam"], inner_r2_corr=mc["dev_r2"]))
            del mr, mc
            torch.cuda.empty_cache()
            if not a.small:
                qr, qc = P.mlp_fit_predict(Xa, [Yr[tr], Yc[tr]], Xe, dev)
                res[f"{arm}/mlp/raster"], res[f"{arm}/mlp/corr"] = qr.reshape(-1, P.NH1, P.NW1).astype(np.float16), qc
            run.info(json.dumps(fits[-1]))
            del Xa, Xe
            torch.cuda.empty_cache()
        np.savez(out / "pred.npz", **res)
        (out / "fits.json").write_text(json.dumps(fits, indent=1))
        run.summary.update(out=str(out / "pred.npz"), n_train=int(tr.sum()), n_dev=int(dv.sum()), n_test=int(e_ok.sum()), corridor_equiv=eq)


# ---------------------------------------------------------------- report (CPU)
class Boot:
    """Log-cluster bootstrap (ratio of sums), one resample matrix shared by every arm / stratum of a row set (paired)."""

    def __init__(self, logs, rows, B=NB, seed=0):
        ug = np.unique(logs[rows])
        self.rows, self.n = rows, len(ug)
        self.inv = np.minimum(np.searchsorted(ug, logs), self.n - 1)          # rows outside the set are never counted (mean masks with rows)
        self.idx = np.random.default_rng(seed).integers(0, self.n, (B, self.n))

    def mean(self, v, m):
        """(point, replicates (B,)) of mean(v[m]) over finite v."""
        m = m & self.rows & np.isfinite(v)
        s, c = np.bincount(self.inv, np.where(m, v, 0.0), self.n), np.bincount(self.inv, m.astype(float), self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (s.sum() / c.sum() if c.sum() else np.nan), s[self.idx].sum(1) / c[self.idx].sum(1)


def ci(pt, reps):
    lo, hi = np.nanpercentile(reps, [2.5, 97.5]) if np.isfinite(reps).any() else (np.nan, np.nan)
    return dict(mean=float(pt), lo=float(lo), hi=float(hi))


def token_metrics(z, key):
    """Per-token read-out errors of one probe (m). inside = the side of the logged turn."""
    true, pred, fut = z["raster_true"].astype(np.float32), z[f"{key}/raster"].astype(np.float32), z["fut"]
    band = np.zeros(true.shape, bool)
    band[:, BX[0]:BX[1], BY[0]:BY[1]] = np.abs(true[:, BX[0]:BX[1], BY[0]:BY[1]]) <= BAND
    with np.errstate(invalid="ignore"):
        m = dict(band=(np.abs(pred - true) * band).sum((1, 2)) / band.sum((1, 2)))
    ct = z["corr_true"].reshape(len(fut), len(S_CORR), 2)
    e = corridor(pred, P.RES1, fut).reshape(ct.shape) - ct
    side = np.where(fut[:, -1, 2] > 0, 0, 1)[:, None, None]                    # left turn: inside = left (index 0)
    ein, eout = np.take_along_axis(e, side, 2)[..., 0], np.take_along_axis(e, 1 - side, 2)[..., 0]
    m.update(corr=np.abs(e).mean((1, 2)), inside=np.abs(ein).mean(1), outside=np.abs(eout).mean(1), inside_bias=ein.mean(1),
             direct=np.abs(z[f"{key}/corr"].reshape(ct.shape) - ct).mean((1, 2)))
    for j, sc in enumerate(S_CORR):
        m[f"inside_{sc:.0f}m"] = np.abs(ein[:, j])
    m["sq"] = ((pred - true) ** 2).mean((1, 2))
    return m


def cmd_report(a):
    import pandas as pd
    from jevdrive.run import Run
    src = OUT / ("turn_probe-small" if a.small else "turn_probe")
    out = src / "report"
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", src.name + "-report", seed=0, config=vars(a)) as run:
        z = dict(np.load(src / "pred.npz"))
        keys = sorted({k.rsplit("/", 1)[0] for k in z if "/" in k})
        M = {k: token_metrics(z, k) for k in keys}
        test, lg, fut = z["test"], z["log"], z["fut"]
        dyaw = np.abs(np.degrees(fut[:, -1, 2]))
        tv = z["raster_true"][test].astype(np.float32).reshape(int(test.sum()), -1).var(0).mean()
        r2 = {k: float(1 - M[k]["sq"][test].mean() / tv) for k in keys}
        strata = {"all": np.ones(len(dyaw), bool)} | {n: (dyaw >= lo) & (dyaw < hi) for n, lo, hi in BINS}
        boots = {"navtest": (test, Boot(lg, test)), "navtrain-heldout": (~test, Boot(lg, ~test))}
        mets = ("band", "corr", "inside", "outside", "inside_bias", "direct") + tuple(f"inside_{s:.0f}m" for s in S_CORR)
        rows, con = [], []
        for split, (sm, B) in boots.items():
            for sn, st in strata.items():
                m = sm & st
                for met in mets:
                    R = {k: B.mean(M[k][met], m) for k in keys}
                    for k in keys:
                        arm, probe = k.split("/")
                        e = R[f"E/{probe}"]
                        rows.append(dict(split=split, stratum=sn, n=int(m.sum()), logs=len(np.unique(lg[m])), probe=probe, arm=arm, metric=met, **ci(*R[k]),
                                         skill_vs_E=float(1 - R[k][0] / e[0]) if met != "inside_bias" else np.nan))
                    for probe in sorted({k.split("/")[1] for k in keys}):
                        for x, y in (("V", "WA"), ("V", "VJ21"), ("V", "E"), ("WA", "E")):
                            if f"{x}/{probe}" in R and f"{y}/{probe}" in R:
                                (px, rx), (py, ry) = R[f"{x}/{probe}"], R[f"{y}/{probe}"]
                                con.append(dict(split=split, stratum=sn, n=int(m.sum()), probe=probe, metric=met, contrast=f"{x} - {y}", **ci(px - py, rx - ry),
                                                rel=float((px - py) / py) if py else np.nan))
            # difference in differences: turn-vs-straight degradation, V minus WA
            for probe in sorted({k.split("/")[1] for k in keys}):
                for met in ("band", "corr", "inside"):
                    g = lambda arm, sn: B.mean(M[f"{arm}/{probe}"][met], sm & strata[sn])  # noqa: E731
                    for sn in ("T45", "T20-45"):
                        (v1, rv1), (v0, rv0), (w1, rw1), (w0, rw0) = g("V", sn), g("V", "S5"), g("WA", sn), g("WA", "S5")
                        con.append(dict(split=split, stratum=f"{sn} - S5", n=int((sm & strata[sn]).sum()), probe=probe, metric=met, contrast="DiD (V - WA)",
                                        **ci((v1 - v0) - (w1 - w0), (rv1 - rv0) - (rw1 - rw0)), rel=np.nan))
                        con.append(dict(split=split, stratum=f"{sn} / S5", n=int((sm & strata[sn]).sum()), probe=probe, metric=met, contrast="ratio-of-ratios (V / WA)",
                                        **ci((v1 / v0) / (w1 / w0), (rv1 / rv0) / (rw1 / rw0)), rel=np.nan))
        S, C = pd.DataFrame(rows), pd.DataFrame(con)
        S.to_csv(out / "strata.csv", index=False), C.to_csv(out / "contrasts.csv", index=False)
        pd.DataFrame([dict(key=k, navtest_raster_r2=v) for k, v in r2.items()]).to_csv(out / "r2.csv", index=False)

        # ---- pre-registered verdict (navtest, band MAE)
        def pick(df, **kw):
            for k, v in kw.items():
                df = df[df[k] == v]
            assert len(df) == 1, (kw, len(df))
            return df.iloc[0]
        V = {}
        for probe in sorted({k.split("/")[1] for k in keys}):
            sk = {arm: float(pick(S, split="navtest", stratum="T45", probe=probe, arm=arm, metric="band").skill_vs_E) for arm in ("V", "WA")}
            d45 = pick(C, split="navtest", stratum="T45", probe=probe, metric="band", contrast="V - WA")
            did = pick(C, split="navtest", stratum="T45 - S5", probe=probe, metric="band", contrast="DiD (V - WA)")
            V[probe] = dict(skill_T45=sk, G0=bool(min(sk.values()) >= 0.15), delta45=[d45["mean"], d45.lo, d45.hi], delta45_rel=float(d45.rel),
                            R1=bool(d45.lo > 0 and d45.rel >= 0.10), did=[did["mean"], did.lo, did.hi], R2=bool(did.lo > 0))
        main = V["ridge" if a.small else "mlp"]
        verdict = ("invalid (G0)" if not main["G0"] else "representation lacks it (turn-specific)" if main["R1"] and main["R2"] else
                   "head-side" if not main["R1"] else "general encoder deficit, not turn-specific")
        agree = all(np.sign(V[p]["delta45"][0]) == np.sign(main["delta45"][0]) and np.sign(V[p]["did"][0]) == np.sign(main["did"][0]) for p in V)
        vd = dict(verdict=verdict, decided_by="ridge (small gate)" if a.small else "mlp", ridge_mlp_same_sign=bool(agree), rules=V, r2=r2, n_boot=NB,
                  n_test=int(test.sum()), n_heldout=int((~test).sum()))
        (out / "verdict.json").write_text(json.dumps(vd, indent=1))
        run.info(json.dumps(vd, indent=1))

        # ---- P2H failure subsets inside navtest T45 and distance to the turn entry
        sm, B = boots["navtest"]
        f = pd.read_csv(FAILS, low_memory=False)
        f = f[(f.bucket == "D1' >45") & f.key.isin(["P2Hs0", "P2Hs1"])]
        cut, anyf = set(f[f.inside == 1].token), set(f.token)
        tk = z["tokens"]
        t45 = sm & strata["T45"]
        is_cut, is_f = np.array([t in cut for t in tk]), np.array([t in anyf for t in tk])
        sets = {"T45 inside-cut fail": t45 & is_cut, "T45 other DAC fail": t45 & is_f & ~is_cut, "T45 pass": t45 & ~is_f}
        d_ent, in_turn = entry(fut, z["pose"])
        t20 = sm & (dyaw > 20)
        ebins = {"in the turn": t20 & in_turn, "entry < 5 m": t20 & ~in_turn & (d_ent < 5), "entry 5-15 m": t20 & ~in_turn & (d_ent >= 5) & (d_ent < 15),
                 "entry >= 15 m": t20 & ~in_turn & (d_ent >= 15)}
        frow, erow = [], []
        for name, groups, dst in (("fail", sets, frow), ("entry", ebins, erow)):
            for gn, m in groups.items():
                for probe in sorted({k.split("/")[1] for k in keys}):
                    for met in ("band", "inside", "inside_bias", "corr"):
                        R = {arm: B.mean(M[f"{arm}/{probe}"][met], m) for arm in ARMS if f"{arm}/{probe}" in M}
                        for arm, (p, r) in R.items():
                            dst.append(dict(group=gn, n=int(m.sum()), logs=len(np.unique(lg[m])), probe=probe, metric=met, arm=arm, **ci(p, r)))
                        dst.append(dict(group=gn, n=int(m.sum()), logs=len(np.unique(lg[m])), probe=probe, metric=met, arm="V - WA",
                                        **ci(R["V"][0] - R["WA"][0], R["V"][1] - R["WA"][1])))
            if name == "fail":                                                                    # excess on inside-cut failures over passes, per arm and V - WA
                for probe in sorted({k.split("/")[1] for k in keys}):
                    for met in ("band", "inside", "inside_bias"):
                        g = lambda arm, gn: B.mean(M[f"{arm}/{probe}"][met], sets[gn])  # noqa: E731
                        (vc, rvc), (vp, rvp), (wc, rwc), (wp, rwp) = g("V", "T45 inside-cut fail"), g("V", "T45 pass"), g("WA", "T45 inside-cut fail"), g("WA", "T45 pass")
                        for arm, p, r in (("V", vc - vp, rvc - rvp), ("WA", wc - wp, rwc - rwp), ("V - WA", (vc - vp) - (wc - wp), (rvc - rvp) - (rwc - rwp))):
                            frow.append(dict(group="inside-cut fail - pass", n=int(sets["T45 inside-cut fail"].sum()), logs=0, probe=probe, metric=met, arm=arm, **ci(p, r)))
        Fd, Ed = pd.DataFrame(frow), pd.DataFrame(erow)
        Fd.to_csv(out / "failsets.csv", index=False), Ed.to_csv(out / "entry.csv", index=False)

        # ---- markdown tables
        fm = lambda r: f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"  # noqa: E731
        L = []
        for probe in sorted({k.split("/")[1] for k in keys}):
            for split in boots:
                for met in ("band", "inside", "corr"):
                    L.append(f"\n**{split}, {probe}, {met} MAE (m)**\n\n| stratum | n (logs) | " + " | ".join(a_ for a_ in ARMS if f"{a_}/{probe}" in M) + " | V - WA | rel |\n|:--|--:|" +
                             ":--|" * (len([a_ for a_ in ARMS if f"{a_}/{probe}" in M]) + 1) + "--:|")
                    for sn in strata:
                        q = S[(S.split == split) & (S.stratum == sn) & (S.probe == probe) & (S.metric == met)].set_index("arm")
                        c = pick(C, split=split, stratum=sn, probe=probe, metric=met, contrast="V - WA")
                        L.append(f"| {sn} | {q.n.iloc[0]} ({q.logs.iloc[0]}) | " + " | ".join(fm(q.loc[a_]) for a_ in ARMS if a_ in q.index) + f" | {c['mean']:+.2f} [{c.lo:+.2f}, {c.hi:+.2f}] | {100 * c.rel:+.0f}% |")
            L.append(f"\n**{probe}: DiD and ratios**\n\n| split | strata | metric | contrast | value [95% CI] |\n|:--|:--|:--|:--|:--|")
            for _, r in C[(C.probe == probe) & C.contrast.str.contains("DiD|ratio")].iterrows():
                L.append(f"| {r.split} | {r.stratum} | {r.metric} | {r.contrast} | {r['mean']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] |")
            for nm, df in (("P2H failure subsets (navtest T45)", Fd), ("distance to the turn entry (navtest > 20 deg)", Ed)):
                L.append(f"\n**{probe}: {nm}**\n\n| group | n (logs) | metric | E | V | WA | VJ21 | V - WA |\n|:--|--:|:--|:--|:--|:--|:--|:--|")
                for (gn, met), q in df[df.probe == probe].groupby(["group", "metric"], sort=False):
                    q = q.set_index("arm")
                    L.append(f"| {gn} | {q.n.iloc[0]} ({q.logs.iloc[0]}) | {met} | " + " | ".join(fm(q.loc[a_]) if a_ in q.index else "-" for a_ in (*ARMS, "V - WA")) + " |")
        (out / "tables.md").write_text("\n".join(L) + "\n")
        run.info("\n".join(L))
        if not a.small:
            figures(out, z, M, S, Ed, sets, strata, test)
        run.summary.update(verdict=verdict, out=str(out))


def figures(out, z, M, S, Ed, sets, strata, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    col = {"E": ps.BASELINE, "V": ps.PALETTE["vermillion"], "WA": ps.PALETTE["blue"], "VJ21": ps.PALETTE["green"]}
    lab = {"E": "ego + command only", "V": "Cinque tokens", "WA": "WA-Cf", "VJ21": "V-JEPA 2.1 (frozen)"}
    # 1. error by turn stratum
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    names = [b[0] for b in BINS]
    for ax, met, yl in ((axs[0], "band", "boundary-band SDF MAE (m)"), (axs[1], "inside", "inside-boundary offset MAE (m)")):
        for j, arm in enumerate(ARMS):
            q = S[(S.split == "navtest") & (S.probe == "mlp") & (S.metric == met) & (S.arm == arm)].set_index("stratum").loc[names]
            x = np.arange(len(names)) + (j - 1.5) * 0.18
            ax.bar(x, q["mean"], 0.17, color=col[arm], label=lab[arm], yerr=[q["mean"] - q.lo, q.hi - q["mean"]], error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(np.arange(len(names)), ["< 5", "5-20", "20-45", "> 45"])
        ax.set_xlabel("logged heading change in 4 s (deg)"), ax.set_ylabel(yl), ps.bars(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.savefig(out / "strata.png", dpi=300)
    # 2. distance to the turn entry
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.4), constrained_layout=True)
    order = ["entry >= 15 m", "entry 5-15 m", "entry < 5 m", "in the turn"]
    for arm in ARMS:
        q = Ed[(Ed.probe == "mlp") & (Ed.metric == "band") & (Ed.arm == arm)].set_index("group").loc[order]
        ax.errorbar(range(4), q["mean"], yerr=[q["mean"] - q.lo, q.hi - q["mean"]], color=col[arm], marker="o", ms=3, capsize=1.5, label=lab[arm])
    ax.set_xticks(range(4), [">= 15 m", "5-15 m", "< 5 m", "in the turn"]), ax.set_xlabel("distance to the turn entry"), ax.set_ylabel("boundary-band SDF MAE (m)")
    ax.legend()
    fig.savefig(out / "entry.png", dpi=300)
    # 3. BEV examples on sharp turns: true vs read-out drivable boundary
    tk = z["tokens"]
    pickq = lambda m: np.flatnonzero(m)[np.argsort(tk[m])][[m.sum() // 4, m.sum() // 2, 3 * m.sum() // 4]]  # noqa: E731
    rows = np.r_[pickq(sets["T45 inside-cut fail"]), pickq(sets["T45 pass"])]
    xc, yc = X0 + np.arange(P.NH1) + 0.5, Y0 + np.arange(P.NW1) + 0.5
    fig, axs = plt.subplots(2, 3, figsize=(ps.DOUBLE_COLUMN_IN, 4.6), constrained_layout=True)
    for ax, i in zip(axs.ravel(), rows):
        t = z["raster_true"][i].astype(np.float32)
        ax.contourf(yc, xc, t, levels=[0, 1e3], colors=["#E8E8E8"])
        ax.contour(yc, xc, t, levels=[0], colors="k", linewidths=1.0)
        for arm in ("V", "WA"):
            ax.contour(yc, xc, z[f"{arm}/mlp/raster"][i].astype(np.float32), levels=[0], colors=col[arm], linewidths=0.9)
        f = np.vstack([[0, 0, 0], z["fut"][i]])
        ax.plot(f[:, 1], f[:, 0], color=ps.PALETTE["green"], marker=".", ms=2.5, lw=0.8)
        ax.set_xlim(20, -20), ax.set_ylim(-4, 36), ax.set_aspect("equal"), ax.grid(False)
        ax.set_title(f"P2H {'inside-cut fail' if i in rows[:3] else 'pass'}, {tk[i][:8]}\nband MAE: Cinque {M['V/mlp']['band'][i]:.2f} m, WA-Cf {M['WA/mlp']['band'][i]:.2f} m", fontsize=7)
    h = [plt.Line2D([], [], color=c, lw=1) for c in ("k", col["V"], col["WA"], ps.PALETTE["green"])]
    fig.legend(h, ["true drivable boundary", "read out of Cinque tokens", "read out of WA-Cf", "logged future (4 s)"], loc="outside lower center", ncol=4)
    fig.savefig(out / "bev.png", dpi=300)
    (out / "bev_tokens.txt").write_text("\n".join(tk[rows]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report"])
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    {"fit": cmd_fit, "report": cmd_report}[a.cmd](a)
