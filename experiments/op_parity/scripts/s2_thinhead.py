"""System 2 as a thin head on frozen VLM features that selects one of the F20 candidates of WP2's plan on WOD-E2E val
(plans/2026-10-08-s2-thinhead-prereg.md, results/s2_thinhead.md). Candidate sets, re-timing and RFS scoring are s2_gohold's.

  q0       (jevdrive env, CPU)     the ceiling of hindsight supervision: RFS of the F20 candidate chosen with the LOGGED future -> results/s2_thinhead/q0.*
  plans    (op-train env, one GPU) WP2 plans on WOD train rows through the training-protocol token path (wod_launch.py tok, part D): r2-train rows at
                                   frame % 4 == 0 -> $DATA_DIR/runs/op_parity/s2_thinhead/train_plans.npz (r2-dev rows: wod/launch/tok_dev.npz)
  extract  (jevdrive env, one GPU) the long-span multi-frame Qwen3-VL-4B features of the 479 rater frames (4 frames per camera at 0.5 s spacing,
                                   Qwen's video path, layer 18) -> features/qwenvid_s2th_s5
  prep     (jevdrive env, CPU)     one table of inputs and targets: val (479 rater frames x 2 WP2 seeds: streams, the RFS of the 20 candidates) and
                                   train (r2-train / r2-dev rows: streams, hindsight class); vision streams standardised + PCA-whitened on train rows
                                   -> $DATA_DIR/runs/op_parity/s2_thinhead/data.npz
  heads    s2_thinhead_heads.py    the arms (features x supervision x head), out-of-fold by sequence
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import s2_gohold as G  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

OUT, FIG = _R / "experiments/op_parity/results/s2_thinhead", _R / "experiments/op_parity/figs/s2_thinhead"
RUNS = data_dir() / "runs/op_parity/s2_thinhead"
ARMS = ("WP2-full-s0", "WP2-full-s1")
PRI = np.array([p_ * 4 + s_ for p_ in (G.KEEP, 1, 3, 0, 4) for s_ in range(4)])     # tie order: the plan itself, then the smaller change
PPRI = np.array([G.KEEP, 1, 3, 0, 4])
QL_SET, QL_STRIDE, QV_SET, Q1_SET, C_SET = "qwenvid_s2th_s5", 5, "qwenvid_train_t4", "qwen_front3", "op_cinque_p3_trainval"
KMAX, PCA_ROWS = 128, 20000
Q1_LAYERS = ("L09_mean", "L27_mean", "L36_mean", "L18_last")


def codebook():
    return json.loads((G.OUT / "codebook.json").read_text())


def board():
    """The 479 rater frames: Ctx, WP2's plans (2 seeds), their F20 candidates (n, 20, 20, 2), the RFS of every candidate J (2, 20, n), strata."""
    import wod_launch_report as R
    from jevdrive import wod_zeroshot as Z
    C, cb = R.Ctx(), codebook()
    P = [C.preds(t) for t in ARMS]
    F = [G.factored(p, C.v0, 3, cb) for p in P]
    J = np.stack([np.stack([C.rfs(np.ascontiguousarray(f[:, m])) for m in range(20)]) for f in F])
    assert np.abs(J[0, G.KEEP * 4] - C.rfs(P[0])).max() < 1e-9
    cluster = Z.load_sets()["rater"]["cluster"].astype(str)
    st = {"all": C.st["all"], "stopped": C.st["stopped"], "moving": C.st["moving (v>=0.5)"], "turn": C.intent >= 2, "straight": C.intent < 2}
    st |= {f"cluster {c}": cluster == c for c in sorted(set(cluster))}
    return SimpleNamespace(C=C, cb=cb, P=P, F=F, J=J, n=C.n, names=C.names[: C.n], v0=C.v0, st=st, cluster=cluster, base=J[:, G.KEEP * 4].mean(0),
                           best=J.max(1).mean(0), log=C.fut[: C.n])


def ci(C, d, rows=None, q=(2.5, 97.5)):
    """Ctx.ci with a free percentile pair (cluster mean of the per-frame difference, paired bootstrap over sequences)."""
    m = C.M if rows is None else C.M * np.asarray(rows, float)[:, None]
    cnt, sm = C.K @ m, C.K @ (m * d[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.nanmean(np.where(cnt > 0, sm / cnt, np.nan), 1)
    return (C.cm(d, rows), *np.nanpercentile(b, q))


def first(S, pri=PRI):
    """(K, n) scores -> best index per frame, ties toward the plan (order pri)."""
    return pri[(S[pri] >= S.max(0) - 1e-9).argmax(0)]


def hindsight(F, log, plan, v0, cb):
    """Candidates F (n, 20, 20, 2) chosen with the logged future (n, 20, 2): {rule: (n,) candidate index}, the ADE of every candidate (n, 20)."""
    d = np.linalg.norm(F - log[:, None], axis=-1)
    ade, fde = d.mean(-1), d[..., [11, 19]].mean(-1)
    lc, own = G.cls(G.arc(log), v0, 3, cb), G.cls(G.arc(plan), v0, 3, cb)
    sp, ii = np.where(lc == own, 0, 1 + lc), np.arange(len(F))
    end = np.stack([d[ii, p * 4 + sp, 19] for p in range(5)])
    return {"H-ade": first(-ade.T), "H-fde": first(-fde.T), "H-cls": first(-end, PPRI) * 4 + sp}, ade


def plan_desc(plan, v0, cb):
    """The plan stream (n, 15): arc length at 1 .. 5 s, lateral position at 3 / 5 s, heading of the last second, own V3 class one-hot, the three
    prototype distances at 5 s for this speed, v0."""
    s, n = G.arc(plan), len(plan)
    ch = plan[:, 19] - plan[:, 15]
    hd = np.where(np.linalg.norm(ch, axis=-1) > 0.1, np.arctan2(ch[:, 1], ch[:, 0]), 0.0)
    own = np.eye(3)[G.cls(s, v0, 3, cb)]
    proto = np.stack([G.decode(np.full(n, c), v0, 3, cb)[:, -1] for c in range(3)], 1)
    return np.concatenate([s[:, [3, 7, 11, 15, 19]], plan[:, [11, 19], 1], hd[:, None], own, proto, v0[:, None]], 1)


# ---------------------------------------------------------------- Q0
def cmd_q0(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-q0", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        Bd = board()
        C, n, ii = Bd.C, Bd.n, np.arange(Bd.n)
        run.info("reproduce: WP2 %.3f, F20 oracle d %+.3f, log %.3f", C.cm(Bd.base), C.cm(Bd.best - Bd.base), C.cm(C.rfs(Bd.log)))
        H = [hindsight(Bd.F[s], Bd.log, Bd.P[s], Bd.v0, Bd.cb) for s in (0, 1)]
        sc = {k: np.mean([Bd.J[s][H[s][0][k], ii] for s in (0, 1)], 0) for k in ("H-ade", "H-fde", "H-cls")}
        sc |= {"log itself": C.rfs(Bd.log), "F20 oracle (privileged)": Bd.best}
        rows = []
        for k, f in sc.items():
            for sn, m in Bd.st.items():
                c = ci(C, f - Bd.base, m)
                o = C.cm(Bd.best - Bd.base, m)
                rows.append({"selection": k, "stratum": sn, "n": int(m.sum()), "RFS WP2": C.cm(Bd.base, m), "RFS": C.cm(f, m), "d": c[0], "lo": c[1], "hi": c[2],
                             "oracle d": o, "share of oracle": c[0] / o if o > 1e-9 else np.nan})
        stats.write_table(rows, OUT / "q0")
        df = pd.DataFrame(rows)
        run.info("Q0:\n%s", df[df.stratum.isin(["all", "stopped", "moving", "turn"])].to_string(float_format=lambda v: f"{v:+.3f}"))
        ag = []
        for k in ("H-ade", "H-fde", "H-cls"):
            for sn in ("all", "stopped", "moving", "turn"):
                m = Bd.st[sn]
                pk = H[0][0][k]
                ag.append({"selection": k, "stratum": sn, "n": int(m.sum()),
                           "pick reaches the oracle best": float(np.mean([(Bd.J[s][H[s][0][k], ii] >= Bd.J[s].max(0) - 1e-9)[m].mean() for s in (0, 1)])),
                           "pick = keep / follow": float(np.mean([(H[s][0][k] == G.KEEP * 4)[m].mean() for s in (0, 1)])),
                           "pick better than WP2": float(np.mean([(Bd.J[s][H[s][0][k], ii] > Bd.J[s][G.KEEP * 4] + 1e-9)[m].mean() for s in (0, 1)])),
                           "pick worse than WP2": float(np.mean([(Bd.J[s][H[s][0][k], ii] < Bd.J[s][G.KEEP * 4] - 1e-9)[m].mean() for s in (0, 1)])),
                           "paths A/B/C/D/E (seed 0)": " / ".join(str(int(((pk // 4)[m] == j).sum())) for j in range(5)),
                           "speeds follow/hold/creep/go (seed 0)": " / ".join(str(int(((pk % 4)[m] == j).sum())) for j in range(4)),
                           "ADE of the pick to the log (m)": float(np.mean([H[s][1][ii, H[s][0][k]][m].mean() for s in (0, 1)])),
                           "ADE of WP2 to the log (m)": float(np.mean([H[s][1][:, G.KEEP * 4][m].mean() for s in (0, 1)]))})
        stats.write_table(ag, OUT / "q0_picks")
        run.info("picks:\n%s", pd.DataFrame(ag).T.to_string())
        pd.DataFrame({"name": Bd.names, "rfs_WP2": Bd.base, "rfs_oracle": Bd.best} | {f"rfs_{k}": v for k, v in sc.items()}
                     | {f"pick_s{s}_{k}": H[s][0][k] for s in (0, 1) for k in H[s][0]}).to_csv(OUT / "q0_frames.csv", index=False, float_format="%.4f")
        r = df[(df.selection == "H-ade") & (df.stratum == "all")].iloc[0]
        run.summary.update(h_ade=[float(r.d), float(r.lo), float(r.hi)])


# ---------------------------------------------------------------- WP2 plans on train rows (token path)
def cmd_plans(a):
    import torch
    import pp_train as T
    import wod_launch as WL
    import wod_parity as WP
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", "s2-thinhead-plans", config=vars(a)) as run:
        tr = splits.load("wod/r2-train")
        run.use_split(tr)
        models = {t: T.load_pmodel(t, dev) for t in ARMS}
        pi = torch.as_tensor(A.plan_index(models[ARMS[0]].net.slices), device=dev)
        cd = WP.CACHE / "wod_r2"
        tab, fi, ticks = dict(np.load(cd / "tab.npz")), np.load(cd / "front_idx.npy"), np.load(cd / "ticks.npy", mmap_mode="r")
        frame = np.array([int(x.rsplit("-", 1)[1]) for x in tab["names"].astype(str)])
        sel = np.flatnonzero(tr.mask(tab["log"]) & (frame % 4 == 0))
        if a.limit:
            sel = sel[: a.limit]
        plans = {t: np.zeros((len(sel), 20, 2), np.float32) for t in ARMS}
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(sel), 256):
                r = sel[i:i + 256]
                u, inv = np.unique(fi[r], return_inverse=True)
                H = torch.from_numpy(ticks[u]).to(dev)[torch.as_tensor(inv.reshape(fi[r].shape), device=dev)]
                ego, tc = torch.from_numpy(tab["ego"][r]).to(dev), torch.tensor([[1.0, 0.0]], device=dev).expand(len(r), 2)
                for t, m in models.items():
                    plans[t][i:i + 256] = WL.to_wod(m(H, ego, tc).float()[:, pi].view(-1, 33, 15).cpu().numpy(), tab["cam"][r][:, :2])
                if (i // 256) % 40 == 0:
                    run.info(f"{i}/{len(sel)} rows, {time.time() - t0:.0f} s")
        RUNS.mkdir(parents=True, exist_ok=True)
        np.savez(RUNS / ("train_plans.npz" if not a.limit else "train_plans_smoke.npz"), names=tab["names"][sel], seq=tab["log"][sel], intent=tab["intent"][sel],
                 ego=tab["ego"][sel], **{f"plan_{t}": p for t, p in plans.items()})
        run.summary.update(rows=len(sel), plan_s=time.time() - t0)


# ---------------------------------------------------------------- long-span multi-frame Qwen3 features of the rater frames
def cmd_extract(a):
    from jevdrive import waymo as W
    from jevdrive import waymo_qwenvid as QV
    from jevdrive import wod_zeroshot as Z
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-extract", config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        df = W.load_index()
        key = dict(zip(W.frame_names(df), range(len(df))))
        names = Z.load_sets()["rater"]["name"].astype(str)[: a.limit or None]
        items, idx, full = W.multicam_clip_items(df, np.array([key[x] for x in names]), QV.FRAMES - 1, QL_STRIDE, W.CAMS)
        run.info("%d of %d rater frames have the %d x %d-frame window", int(full.sum()), len(full), QV.FRAMES, QL_STRIDE)
        meta = W.extract_items(QL_SET + ("_smoke" if a.limit else ""), QV.make_fx(), items, idx, 2, frames_per_clip=QV.FRAMES, clip_stride=QL_STRIDE, layer=QV.LAYER,
                               complete=int(full.sum()))
        run.summary.update(rows=meta["rows"], ms_per_frame=meta["ms_per_frame"])


# ---------------------------------------------------------------- prep
def shard_feats(name, arrays, names, cache=None):
    """Rows of a shard-keyed feature set for the frame names (order of `names`): {array: (n, d) float32}, found mask."""
    import pandas as pd
    from jevdrive import par
    from jevdrive import waymo as W
    if cache is not None and cache.exists():
        z = np.load(cache)
        if len(z["found"]) == len(names):
            return {k: z[k] for k in arrays}, z["found"]
    root = W.out_dir("features", name)
    parts = sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists())
    pos = pd.Series(np.arange(len(names)), index=names)

    def one(d):
        fn = pd.read_parquet(d / "index.parquet", columns=["frame_name"]).frame_name.to_numpy()
        m = np.flatnonzero(np.isin(fn, names))
        return (pos[fn[m]].to_numpy(), {k: np.asarray(np.load(d / f"{k}.npy", mmap_mode="r")[m], np.float32) for k in arrays}) if len(m) else None
    res = par.pmap(one, parts, threads=True, workers=16, desc=name)
    res.raise_if_failed()
    out, found = None, np.zeros(len(names), bool)
    for r in res.values:
        if r is None:
            continue
        p, x = r
        if out is None:
            out = {k: np.zeros((len(names), v.shape[1]), np.float32) for k, v in x.items()}
        new = ~found[p]                                                         # first shard wins (a frame re-extracted under another index)
        for k in arrays:
            out[k][p[new]] = x[k][new]
        found[p[new]] = True
    if cache is not None:
        np.savez(cache, found=found, **out)
    return out, found


def flat_feats(name, array, names):
    import pandas as pd
    from jevdrive import waymo as W
    idx, f = W.load_flat_features(name, [array])
    pos = pd.Series(np.arange(len(idx)), index=idx.frame_name.to_numpy()).reindex(names).to_numpy()
    ok = np.isfinite(pos)
    out = np.zeros((len(names), f[array].shape[1]), np.float32)
    o = np.argsort(pos[ok])
    out[np.flatnonzero(ok)[o]] = np.asarray(f[array][np.sort(pos[ok]).astype(int)], np.float32)
    return out, ok


def _cand(args):
    plan, v0, log, cb = args
    F = G.factored(plan, v0, 3, cb)
    h, ade = hindsight(F, log, plan, v0, cb)
    return h["H-ade"], ade.astype(np.float32), plan_desc(plan, v0, cb).astype(np.float32)


def whiten(xtr, fit_rows, k=KMAX):
    """Standardise + PCA-whiten on the train rows `fit_rows` (unsupervised) -> function x -> (n, k) scores with unit train variance."""
    mu, sd = xtr[fit_rows].mean(0), xtr[fit_rows].std(0) + 1e-6
    z = (xtr[fit_rows] - mu) / sd
    w, V = np.linalg.eigh(z.T @ z / len(z))
    o = np.argsort(-w)[:k]
    V, s = V[:, o], np.sqrt(np.maximum(w[o], 1e-8))
    return lambda x: (((x - mu) / sd) @ V / s).astype(np.float32), float(w[o].sum() / w.sum())


def cmd_prep(a):
    import pp_wod as PW
    from jevdrive import par
    from jevdrive import waymo as W
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-prep", seed=0, config=vars(a)) as run:
        tr, dv, val = splits.load("wod/r2-train"), splits.load("wod/r2-dev"), splits.load("wod/val")
        for s in (tr, dv, val):
            run.use_split(s)
        splits.check_disjoint(tr, dv)
        splits.check_disjoint(tr, val)
        splits.check_disjoint(dv, val)
        Bd = board()
        C, n, cb = Bd.C, Bd.n, Bd.cb
        r = C.Z.load_sets()["rater"]
        assert set(C.seq[:n]) <= set(val.members)
        # ---- train rows: r2-train (plans from `plans`) and r2-dev (wod-launch's tok_dev), frame % 4 == 0, with the multi-frame cache
        tp, td = np.load(RUNS / "train_plans.npz"), np.load(data_dir() / "runs/op_parity/wod/launch/tok_dev.npz")
        cd = np.load(data_dir() / "runs/op_parity/cache/wod_r2/tab.npz")
        ego_tab = dict(zip(cd["names"].astype(str), range(len(cd["names"]))))
        tn = np.concatenate([tp["names"], td["names"]]).astype(str)
        tseq = np.concatenate([tp["seq"], td["seq"]]).astype(str)
        tplan = np.concatenate([tp[f"plan_{ARMS[0]}"], td[f"plan_{ARMS[0]}"]]).astype(np.float64)
        isdev = np.r_[np.zeros(len(tp["names"]), bool), np.ones(len(td["names"]), bool)]
        frame = np.array([int(x.rsplit("-", 1)[1]) for x in tn])
        assert tr.mask(tseq[~isdev]).all() and dv.mask(tseq[isdev]).all()
        df = W.load_index()
        key = dict(zip(W.frame_names(df), range(len(df))))
        past_all, fut_all = W.load_ego()
        qv, has_qv = shard_feats(QV_SET, ["L18_mean"], tn, RUNS / "cache_qv_train.npz")
        keep = has_qv & (frame % 4 == 0)
        run.info("train rows %d (r2-train %d, r2-dev %d); with the multi-frame cache and frame %% 4 == 0: %d / %d", len(tn), (~isdev).sum(), isdev.sum(),
                 (keep & ~isdev).sum(), (keep & isdev).sum())
        tn, tseq, tplan, isdev, qv = tn[keep], tseq[keep], tplan[keep], isdev[keep], qv["L18_mean"][keep]
        gi = np.array([key[x] for x in tn])
        tv0, tlog = W.init_speed(past_all[gi]).astype(np.float64), fut_all[gi][..., :2].astype(np.float64)
        ok = np.isfinite(tlog).all((1, 2))
        assert ok.all(), f"{(~ok).sum()} train rows without a full future"
        tego = cd["ego"][[ego_tab[x] for x in tn]].astype(np.float32)
        ch = [slice(i, i + 1000) for i in range(0, len(tn), 1000)]
        res = par.pmap(_cand, [(tplan[c], tv0[c], tlog[c], cb) for c in ch], workers=min(n_cpus(), 48), run=run, desc="train candidates")
        res.raise_if_failed()
        tlab, tade, tpd = (np.concatenate([x[j] for x in res.values]) for j in range(3))
        # ---- val rater frames
        vego = PW.wod_ego(r["past"], r["intent"])[0].astype(np.float32)
        vpd = np.stack([plan_desc(p, Bd.v0, cb) for p in Bd.P]).astype(np.float32)
        vh = [hindsight(Bd.F[s], Bd.log, Bd.P[s], Bd.v0, cb) for s in (0, 1)]
        tok = np.load(data_dir() / "runs/op_parity/wod/launch/tok_val.npz")
        tk = dict(zip(tok["names"].astype(str), range(len(tok["names"]))))
        both = np.array([x in tk for x in Bd.names])
        dtok = np.linalg.norm(tok[f"plan_{ARMS[0]}"][[tk[x] for x in Bd.names[both]]] - Bd.P[0][both], axis=-1)
        run.info("token-path vs harness WP2-s0 plan on %d rater frames: mean %.3f m, p95 %.3f m, at 5 s %.3f m", both.sum(), dtok.mean(), np.percentile(dtok, 95),
                 dtok[:, -1].mean())
        # ---- vision streams: whiten on train rows, apply to train and val
        fit = np.sort(np.random.default_rng(0).choice(np.flatnonzero(~isdev), min(PCA_ROWS, int((~isdev).sum())), replace=False))
        Z, meta = {}, {"pca_var": {}, "missing_val": {}}

        def stream(nm, xtr, xval, found=None):
            f, var = whiten(xtr, fit)
            Z[f"tr_{nm}"], Z[f"va_{nm}"] = f(xtr), f(xval)
            if found is not None and not found.all():
                Z[f"va_{nm}"][~found] = 0.0                                     # the train mean
            meta["pca_var"][nm], meta["missing_val"][nm] = var, [] if found is None else Bd.names[~found].tolist()
            run.info("stream %s: dim %d -> %d PCs (%.1f %% of the train variance), val rows missing %d", nm, xtr.shape[1], Z[f"tr_{nm}"].shape[1], 100 * var,
                     0 if found is None else int((~found).sum()))
        c_tr, f1 = flat_feats(C_SET, "temporal", tn)
        c_va, f2 = flat_feats(C_SET, "temporal", Bd.names)
        assert f1.all() and f2.all()
        stream("C", c_tr, c_va)
        qv_va, fv = shard_feats(QV_SET, ["L18_mean"], Bd.names)
        stream("QV", qv, qv_va["L18_mean"], fv)
        del qv
        q_tr, f1 = shard_feats(Q1_SET, ["L18_mean"], tn, RUNS / "cache_q1_train.npz")
        q_va, f2 = shard_feats(Q1_SET, ["L18_mean"] + list(Q1_LAYERS), Bd.names)
        assert f1.all() and f2.all()
        stream("Q1", q_tr["L18_mean"], q_va["L18_mean"])
        raw_v0 = {"C": c_va, "Q1": q_va["L18_mean"], "QV": qv_va["L18_mean"]}
        del q_tr
        sub = tn[fit]                                                           # the other layers: PCA on the fit rows only (descriptive layer curve)
        ql, f1 = shard_feats(Q1_SET, list(Q1_LAYERS), sub)
        for k in Q1_LAYERS:
            f, var = whiten(ql[k], np.arange(len(sub)))
            Z[f"va_Q1@{k}"] = f(q_va[k])
            meta["pca_var"][f"Q1@{k}"] = var
        if (W.out_dir("features", QL_SET) / "meta.json").exists():              # no train rows: whitened on the val rows themselves (unsupervised, no labels)
            x, fl = flat_feats(QL_SET, "L18_mean", Bd.names)
            f, var = whiten(x, np.flatnonzero(fl))
            Z["va_QL"] = f(x)
            Z["va_QL"][~fl] = 0.0
            meta["pca_var"]["QL"], meta["missing_val"]["QL"] = var, Bd.names[~fl].tolist()
            f, var = whiten(raw_v0["QV"], np.flatnonzero(fv))                   # same treatment for the short-span set, for a like-for-like comparison
            Z["va_QVv"] = f(raw_v0["QV"])
            Z["va_QVv"][~fv] = 0.0
            f, var = whiten(raw_v0["Q1"], np.arange(n))
            Z["va_Q1v"] = f(raw_v0["Q1"])
            raw_v0["QL"] = x
            run.info("stream QL: %d PCs fitted on the %d val rows (%.1f %% var), missing %d", Z["va_QL"].shape[1], fl.sum(), 100 * var, (~fl).sum())
        # ---- alignment check: v0 from each raw vision stream, ridge fitted on half the sequences
        half = (np.random.default_rng(0).permutation(n) % 2).astype(bool)
        for nm, x in raw_v0.items():
            z = (x - x[half].mean(0)) / (x[half].std(0) + 1e-6)
            w = np.linalg.solve(z[half].T @ z[half] + 100 * np.eye(z.shape[1]), z[half].T @ (Bd.v0[half] - Bd.v0[half].mean()))
            meta.setdefault("align_r2_v0", {})[nm] = float(1 - ((z[~half] @ w + Bd.v0[half].mean() - Bd.v0[~half]) ** 2).mean() / Bd.v0[~half].var())
        run.info("alignment (held-out R2 of v0 from the raw stream): %s", meta["align_r2_v0"])
        emu, esd = tego[~isdev].mean(0), tego[~isdev].std(0) + 1e-6
        pmu, psd = tpd[~isdev].mean(0), tpd[~isdev].std(0) + 1e-6
        lab_hist = {k: np.bincount(v, minlength=20).tolist() for k, v in (("r2-train", tlab[~isdev]), ("r2-dev", tlab[isdev]), ("val rater s0", vh[0][0]["H-ade"]),
                                                                           ("val rater s1", vh[1][0]["H-ade"]))}
        run.info("hindsight class shares (keep / follow): %s", {k: round(v[G.KEEP * 4] / sum(v), 3) for k, v in lab_hist.items()})
        meta |= {"label_hist": lab_hist, "n_train": int((~isdev).sum()), "n_dev": int(isdev.sum()), "train_seqs": len(set(tseq[~isdev])), "dev_seqs": len(set(tseq[isdev])),
                 "token_vs_harness_m": [float(dtok.mean()), float(np.percentile(dtok, 95))], "splits": [tr.id, dv.id, val.id]}
        np.savez(RUNS / "data.npz", va_names=Bd.names, va_seq=C.seq[:n], va_ego=(vego - emu) / esd, va_plan=(vpd - pmu) / psd, va_J=Bd.J,
                 va_hlab=np.stack([vh[s][0]["H-ade"] for s in (0, 1)]), tr_names=tn, tr_seq=tseq, tr_dev=isdev, tr_ego=(tego - emu) / esd, tr_plan=(tpd - pmu) / psd,
                 tr_lab=tlab, tr_ade=tade, **Z)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "data_meta.json").write_text(json.dumps(meta, indent=1))
        run.summary.update(n_train=meta["n_train"], n_dev=meta["n_dev"])


# ---------------------------------------------------------------- latency of the head path
def cmd_latency(a):
    """Batch 1, JPEG bytes -> selected candidate: preprocessing (read + decode + processor), Qwen3-VL-4B to layer 18, the head."""
    import subprocess
    import torch
    from jevdrive import features as Fx
    from jevdrive import stats
    from jevdrive import waymo as W
    from jevdrive import waymo_qwenvid as QV
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-latency", config=vars(a)) as run:
        df = W.load_index()
        key = dict(zip(W.frame_names(df), range(len(df))))
        rows = np.array([key[x] for x in Z.load_sets()["rater"]["name"].astype(str)[: a.n + 5]])
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        run.info("cards at start (index, util, memory): %s; CUDA_VISIBLE_DEVICES=%s", smi.replace("\n", " | "), __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
        out = []
        for nm, frames, stride in (("Q1: 3 cameras, 1 frame", 1, 1), ("QV / QL: 3 cameras x 4 frames (video path)", 4, 2)):
            if frames == 1:
                fx = Fx.QwenFeatures(n_images=len(W.CAMS), layers=[QV.LAYER], compile=False)
                fx.model.language_model.layers = fx.model.language_model.layers[: QV.LAYER]
            else:
                fx = QV.make_fx()
            items, _, _ = W.multicam_clip_items(df, rows, frames - 1, stride, W.CAMS)
            ds = W.Shards(items, fx.transform)
            pre, fwd = [], []
            for i in range(len(items)):
                t0 = time.perf_counter()
                b = fx.collate([ds[i]])
                t1 = time.perf_counter()
                o = fx(b)
                float(o[f"L{QV.LAYER:02d}_mean"].float().sum())                     # host sync
                t2 = time.perf_counter()
                pre.append(1e3 * (t1 - t0))
                fwd.append(1e3 * (t2 - t1))
            pre, fwd = np.array(pre[5:]), np.array(fwd[5:])
            out.append({"path": nm, "n": len(pre), "LLM tokens": int(fx.n_image_tokens), "preprocess ms (median)": float(np.median(pre)), "model ms (median)": float(np.median(fwd)),
                        "model ms (p95)": float(np.percentile(fwd, 95)), "peak VRAM GB": torch.cuda.max_memory_allocated() / 2 ** 30})
            del fx
            torch.cuda.empty_cache()
        x, Wm = np.random.default_rng(0).normal(size=(1, 2560)).astype(np.float32), np.random.default_rng(1).normal(size=(2560 + 35, 20)).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(1000):
            int((np.concatenate([x, np.zeros((1, 35), np.float32)], 1) @ Wm).argmax())
        out.append({"path": "head (linear, PCA folded into the weights; CPU, numpy)", "n": 1000, "model ms (median)": 1e3 * (time.perf_counter() - t0) / 1000})
        stats.write_table(out, OUT / "latency")
        (OUT / "latency_cards.txt").write_text(smi + "\n")
        run.info("latency: %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    sp_.add_parser("q0")
    for c in ("plans", "extract"):
        p = sp_.add_parser(c)
        p.add_argument("--limit", type=int, default=0)
    sp_.add_parser("prep")
    p = sp_.add_parser("latency")
    p.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    {"q0": cmd_q0, "plans": cmd_plans, "extract": cmd_extract, "prep": cmd_prep, "latency": cmd_latency}[a.cmd](a)
