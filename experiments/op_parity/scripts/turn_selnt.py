"""op_parity turn selector navtrain (plans/2026-10-08-turn-selector-navtrain-prereg.md): selectors trained on the navtrain held-out labels
(sh30_crossfit, 28 323 turn tokens, features from the fold model that did not see the token's log), read out on the navtest > 20 deg turn tokens
with the real SH30-F-s0 / s1 plans and their existing candidate scores. No new simulator scoring. Reuses turn_dewater / turn_selinput conventions.

  build    train-side arrays (E streams, fold-model hidden state / road edges, candidate margins, labels) from tsn_extract.py outputs; G-leak, G-margin
  fit      one arm: OOF on navtrain, final fit, navtest predictions (never reads a navtest score), learning curve   (GPU for NN arms, pool job)
  ctrl     G-E: the navtest-internal E arm of decision 186 (10 repeats) must reproduce -0.27
  hidden   G-hidden: fold-model hidden state vs SH30 hidden state on the navtest turn tokens
  report   tables, figures, verdict (reads navtest scores)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json, pickle, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import tsn_extract as TX  # noqa: E402
import turn_ceiling as TC  # noqa: E402
import turn_dewater as TD  # noqa: E402
import turn_selinput as TS  # noqa: E402

D = TC.D
OUT = D / "runs/op_parity/turn_selnt"
LAB = D / "runs/op_parity/sh30_crossfit/labels"
RES = _R / "experiments/op_parity/results/turn_selector_navtrain"
FIG = _R / "experiments/op_parity/figs/turn_selector_navtrain"
FK = TS.FK                                             # ["F19 x pc", "L9 x epcap"]
FKS = TS.FKS
NCAND = 19                                             # F19 = c00..c18 are the scored candidates
CFG = [(64, 0.1, 0.05), (128, 0.2, 0.05), (128, 0.3, 0.3), (256, 0.3, 0.3)]
FRACS = (0.05, 0.1, 0.25, 0.5, 0.75)
STREAMS = {"E": ["ego", "plan"], "N1": ["ego", "plan", "Cflat"], "N4": ["H", "ego", "plan"], "P1": ["ego", "plan", "Mflat"]}
MARG = {"N2": "C", "N3": "C", "P2": "M", "P3": "M"}
NN_SPEC = {"N5": dict(tok=1), "N6": dict(h=1), "N7": dict(tok=1, h=1, c="C"), "N8": dict(tok=1, c="C"), "P4": dict(c="M")}
ARMS = ["E", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "P1", "P2", "P3", "P4"]
CURVE_ARMS = {"N7": 3, "P4": 3, "P2": 2}
FOLDS_OUT, FOLDS_NN, INNER, EPOCHS, NINIT = 5, 3, 4, 30, 5


def fam_idx():
    C = TC.candidates()
    F = TD.fams(C)
    out = {}
    for f, cv in FKS:
        fk = TD.fkey(f, cv)
        ix = F[f]
        assert max(ix) < NCAND, f"{f} has candidates beyond the F19 block"
        out[fk] = (ix, np.array([C[i][1:] for i in ix], float))
    return C, F, out


_C, _F, FI_ = fam_idx()


# ---------------------------------------------------------------- 1. train side
def cmd_build(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.bench.navsim import SUBS
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    import multiprocessing as mp
    import sc_analyze as SC
    cdir = D / "runs/op_parity/cache"
    with Run("op_parity", f"turn_selnt/build-{a.tag}", seed=0, config=vars(a)) as run:
        for j in range(5):
            run.use_split(splits.load(f"navsim/op-parity-cf5f{j}-dev"))
        nt, ntr = splits.load("navsim/navtest"), splits.load("navsim/navtrain")
        run.use_split(nt), run.use_split(ntr)
        df = TX.tokens_of(a.tag)
        toks, n = df.token.to_numpy(), len(df)
        pos = {t: i for i, t in enumerate(toks.tolist())}
        # ---- extraction outputs, in token order
        Z = {k: None for k in ("select_4", "mean", "road_edges", "plan_pos", "model_fold", "diff")}
        sh_of, row_of = np.full(n, -1), np.full(n, -1)
        gates = []
        for i in range(TX.NSH):
            f = TX.out_dir(a.tag) / f"s{i}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            ix = np.array([pos[t] for t in z["tokens"].tolist()], int)
            for k in Z:
                if Z[k] is None:
                    Z[k] = np.zeros((n, *z[k].shape[1:]), z[k].dtype)
                Z[k][ix] = z[k]
            assert (z["fold"] == df.fold.to_numpy()[ix]).all()
            sh_of[ix], row_of[ix] = i, z["row"]
            gates.append(json.loads(str(z["gate"])))
        assert (sh_of >= 0).all(), f"{(sh_of < 0).sum()} tokens without extracted features"
        fold = df.fold.to_numpy()
        assert (Z["model_fold"] == fold).all()
        # ---- G-leak
        leak = dict(a_fold_hash_ok=bool(all(TX.fold_of_log(l) == f for l, f in zip(df.log, fold))), a_model_fold_ok=True,
                    b_max_diff_m=max(g["max_diff_m"] for g in gates), b_median_diff_m=float(np.median([g["median_diff_m"] for g in gates])),
                    c_ctrl_median_diff_m=float(np.median([g["ctrl_median_diff_m"] for g in gates if "ctrl_median_diff_m" in g])) if any("ctrl_median_diff_m" in g for g in gates) else None)
        leak["b_ok"] = leak["b_max_diff_m"] < TX.TOL_M
        leak["c_ok"] = leak["c_ctrl_median_diff_m"] is None or leak["c_ctrl_median_diff_m"] > 0.1
        # (d) the navtest tokens are not navtrain members and vice versa
        tt, _ = TC.bucket_tokens()
        leak["d_disjoint_ok"] = bool(not ntr.mask(tt).any() and not nt.mask(toks).any())
        # ---- labels (c00 = identity) and held-out poses
        zp = np.load(LAB / "poses.npz")
        ti = {t: i for i, t in enumerate(zp["tokens"].tolist())}
        li = np.array([ti[t] for t in toks])
        P0 = zp["h"][li].astype(np.float64)
        z33 = np.load(LAB / "poses_f33.npz")
        assert np.array_equal(z33["tokens"], zp["tokens"])
        poses = {TC.key(0, c): z33[f"c{c:02d}"][li].astype(np.float64) for c in range(NCAND)}
        assert np.array_equal(poses[TC.key(0, 0)], P0)
        cs = pd.read_csv(LAB / "cscore_all_F19.csv")
        C, F, FI = fam_idx()
        X = np.full((1, len(C), n, 9), np.nan)
        tix = {t: i for i, t in enumerate(toks.tolist())}
        cs = cs[cs.token.isin(tix)]
        for key, g in cs.groupby("key"):
            X[0, int(key[1:]), g.token.map(tix).to_numpy(), :8] = g[[SUBS[k] for k in TC.SUB8]].to_numpy(float)
        assert np.isfinite(X[0, :NCAND, :, :8]).all(), "cscore_all_F19 does not cover every (candidate, token)"
        sc = pd.read_csv(LAB / "score.csv")
        sc = sc[sc.token.isin(tix)].set_index("token").loc[toks]
        leak["e_c00_eq_h_max_abs"] = float(np.abs(X[0, 0, :, :8] - sc[[SUBS[k] for k in TC.SUB8]].to_numpy(float)).max())
        leak["e_ok"] = leak["e_c00_eq_h_max_abs"] == 0.0
        Y = {}
        for (f, cv), fk in zip(FKS, FK):
            Sc = TC.noec(TD.conv(X, C, cv))
            Y[fk] = (Sc[:, FI[fk][0]] - Sc[:, :1]).transpose(0, 2, 1)           # (1, n, K)
        # ---- tab streams
        ego, cam, speed = np.zeros((n, 20)), np.zeros((n, 3)), np.zeros(n)
        for i in range(TX.NSH):
            m = np.flatnonzero(sh_of == i)
            if not len(m):
                continue
            t = np.load(cdir / f"navtrain_full.s{i}of12/tab.npz")
            ego[m], cam[m], speed[m] = t["ego"][row_of[m]], t["cam"][row_of[m]], t["speed"][row_of[m]]
        # ---- margins
        zs = np.load(D / "runs/op_probe/labels/navtrain_all.npz")
        sidx = {t: i for i, t in enumerate(zs["tokens"].tolist())}
        sdf_i = np.array([sidx[t] for t in toks])
        assert zs["ok"][sdf_i].all() and (float(zs["x0"]), float(zs["y0"]), float(zs["res"])) == (SC.X0, SC.Y0, SC.RES)
        TS._D["sdf"] = zs["sdf"][sdf_i]
        TS._D["poses"] = poses
        ex, ey = SC.edges_ego(Z["road_edges"], cam)
        cal = json.loads(TS.CAL.read_text())["SH30-F-s0"]
        TS._D["ex"], TS._D["ey"], TS._D["cal"] = {0: ex}, {0: ey}, {0: (cal["s"], cal["b"])}
        units = [(0, c) for c in range(NCAND)]
        workers = max(1, min(n_cpus() // 4, 24, len(units)))
        M, Cm = np.zeros((1, n, NCAND, 4)), np.zeros((1, n, NCAND, 4))
        for fn, A_ in ((TS._m_sdf, M), (TS._m_curb, Cm)):
            r = par.pmap(fn, units, workers=workers, run=run, desc=fn.__name__, mp_context=mp.get_context("fork"))
            r.raise_if_failed()
            for (s, c), v in r.values:
                A_[0, :, c] = v
        M, Cm = np.clip(M, *TS.CLIP), np.clip(Cm, *TS.CLIP)
        dac_fail = X[0, 0, :, 1] < 1
        gm = dict(map_auc=TS._auc(-M[0, :, 0, 3], dac_fail), curb_auc=TS._auc(-Cm[0, :, 0, 3], dac_fail), dac_fail_rate=float(dac_fail.mean()))
        gm.update(map_ok=bool(gm["map_auc"] >= 0.85), curb_in_range=bool(0.58 <= gm["curb_auc"] <= 0.72))
        # ---- vision tokens (last context frame, 32 x 512)
        V3 = np.zeros((n, 32, 512), np.float16)
        for i in range(TX.NSH):
            m = np.flatnonzero(sh_of == i)
            if len(m):
                front = np.load(cdir / f"navtrain_full.s{i}of12@warp/front.npy", mmap_mode="r")
                o = np.argsort(row_of[m])
                V3[m[o]] = np.asarray(front[row_of[m][o], -1])
        H = np.concatenate([Z["select_4"], Z["mean"]], 1)
        plan = TD.plan_desc(P0)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"train_{a.tag}.npz", tokens=toks, log=df.log.to_numpy(), fold=fold, ego=ego, plan=plan, H=H, M=M.astype(np.float32), C=Cm.astype(np.float32),
                 plan_pos=Z["plan_pos"], speed=speed, **{f"Y|{fk}": Y[fk].astype(np.float32) for fk in FK})
        np.save(OUT / f"train_V3_{a.tag}.npy", V3)
        gate = dict(leak=leak, margin=gm, n=n, n_logs=int(df.log.nunique()), ok=bool(leak["b_ok"] and leak["c_ok"] and leak["d_disjoint_ok"] and leak["e_ok"] and leak["a_fold_hash_ok"] and gm["map_ok"]))
        (OUT / f"gates_build_{a.tag}.json").write_text(json.dumps(gate, indent=1))
        run.info("gates %s", json.dumps(gate))
        run.summary.update(gate=gate["ok"], map_auc=gm["map_auc"], curb_auc=gm["curb_auc"])
        assert gate["ok"], f"G-leak / G-margin failed: {gate}"


# ---------------------------------------------------------------- 2. sides
def _both(x):
    return np.stack([x, x])


def load_train(tag):
    z = np.load(OUT / f"train_{tag}.npz")
    s = {k: z[k] for k in z.files}
    n = len(s["tokens"])
    side = dict(n=n, S=1, tokens=s["tokens"], ego=s["ego"][None], plan=s["plan"][None], Hraw=s["H"].astype(np.float32)[None], M=s["M"].astype(np.float64), C=s["C"].astype(np.float64),
                lc=np.unique(s["log"], return_inverse=True)[1], log=s["log"], V3=np.load(OUT / f"train_V3_{tag}.npy"))
    for fk in FK:
        side["Y", fk] = s[f"Y|{fk}"].astype(np.float64)
    return side


def load_test():
    """Navtest turn tokens x SH30 s0 / s1: test-time inputs only (no score is read here)."""
    cdir = D / "runs/op_parity/cache"
    tok, dyaw = TC.bucket_tokens()
    log = TC.archived(TC.MODEL.format(TC.SEEDS[0]), tok)[1]
    tab = np.load(cdir / "lb_navtest/tab.npz")
    row = {t: i for i, t in enumerate(tab["names"].tolist())}
    use = np.array([row[t] for t in tok])
    P = np.load(TC.OUT / "poses.npz")
    assert np.array_equal(P["tokens"], tok)
    Hs = []
    for s in TC.SEEDS:
        h = np.load(TS.HID.format(s))
        assert (h["names"] == tab["names"]).all()
        Hs.append(np.concatenate([h["select_4"], h["mean"]], 1).astype(np.float32)[use])
    front = np.load(cdir / "lb_navtest@warp/front.npy", mmap_mode="r")
    Z = np.load(TS.OUT / "margins.npz")
    assert np.array_equal(Z["tokens"], tok)
    return dict(n=len(tok), S=2, tokens=tok, dyaw=dyaw, log=log, lc=np.unique(log, return_inverse=True)[1], ego=_both(tab["ego"][use].astype(np.float64)),
                plan=np.stack([TD.plan_desc(P[TC.key(s, 0)].astype(np.float64)) for s in TC.SEEDS]), Hraw=np.stack(Hs),
                M=Z["M"][:, :NCAND].transpose(0, 2, 1, 3).astype(np.float64), C=Z["C"][:, :NCAND].transpose(0, 2, 1, 3).astype(np.float64),
                V3=np.ascontiguousarray(np.asarray(front[:, -1])[use]))


def add_pca(tr, te, k=max(TD.KS)):
    """Whitened PCA of the hidden state, fitted on the training rows (features only, no labels), applied to both sides."""
    A = tr["Hraw"].reshape(-1, tr["Hraw"].shape[-1])
    mu, sd = A.mean(0), A.std(0) + 1e-6
    _, sv, Vt = np.linalg.svd((A - mu) / sd, full_matrices=False)
    W = Vt[:k].T / (sv[:k] / np.sqrt(len(A)))
    for side in (tr, te):
        side["H"] = (((side["Hraw"] - mu) / sd) @ W).astype(np.float64)


def sel(side, fk, key):
    """Candidate-axis slice of M / C for the family."""
    return side[key][:, :, FI_[fk][0]]


def xof(side, streams, k, fk):
    out = []
    for s in streams:
        if s == "H":
            out.append(side["H"][..., :k])
        elif s in ("Mflat", "Cflat"):
            m = sel(side, fk, s[0])
            out.append(m.reshape(m.shape[0], m.shape[1], -1))
        else:
            out.append(side[s])
    return np.concatenate(out, -1)


def folds_by_log(side, idx, k, salt):
    lg = np.unique(side["lc"][idx])
    fo = dict(zip(np.random.default_rng(salt).permutation(lg).tolist(), (np.arange(len(lg)) % k).tolist()))
    return np.array([fo[l] for l in side["lc"][idx]])


def gain(side, fk, idx, pred):
    return float(TD.take(side["Y", fk][:, idx], TD.picks_of(pred)).mean())


# ---------------------------------------------------------------- 3. heads: fit(tr) -> predictor(side, idx) -> (S, m, K)
def fit_L(tr_side, arm, fk, tr, salt):
    streams = STREAMS[arm]
    Y = tr_side["Y", fk]
    K = Y.shape[-1]
    ks = TD.KS if "H" in streams else TD.KS[:1]
    Xk = {k: xof(tr_side, streams, k, fk) for k in ks}
    g = folds_by_log(tr_side, tr, INNER, 1000 + salt)
    best, al = -np.inf, None
    for k, X in Xk.items():
        d = X.shape[-1]
        acc = np.zeros(len(TD.LAMS))
        for q in range(INNER):
            a_, b_ = tr[g != q], tr[g == q]
            P = TD.ridge_fit(X[:, a_].reshape(-1, d), Y[:, a_].reshape(-1, K))
            for li, lam in enumerate(TD.LAMS):
                acc[li] += TD.take(Y[:, b_], TD.picks_of(P(X[:, b_].reshape(-1, d), lam).reshape(X.shape[0], len(b_), K))).sum()
        for li in reversed(range(len(TD.LAMS))):
            if acc[li] > best + 1e-9:
                best, al = acc[li], (k, TD.LAMS[li])
    P = TD.ridge_fit(Xk[al[0]][:, tr].reshape(-1, Xk[al[0]].shape[-1]), Y[:, tr].reshape(-1, K))

    def pred(side, idx):
        X = xof(side, streams, al[0], fk)[:, idx]
        p = P(X.reshape(-1, X.shape[-1]), al[1]).reshape(X.shape[0], len(idx), K)
        p[..., 0] = 0.0
        return p
    pred.info = dict(k=al[0], lam=al[1])
    return pred


def fit_G(tr_side, arm, fk, tr, salt, threads=16):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    Y = tr_side["Y", fk]
    K = Y.shape[-1]
    cf = FI_[fk][1][1:]
    mk = MARG[arm]

    def rows(side, idx):
        E = xof(side, ["ego", "plan"], 0, fk)[:, idx]
        Mm = sel(side, fk, mk)[:, idx]
        S, m = E.shape[:2]
        fmax = Mm[..., 3].max(-1)
        x = np.broadcast_to(E[:, :, None, :], (S, m, K - 1, E.shape[-1]))
        c = np.broadcast_to(cf[None, None], (S, m, K - 1, 3))
        own, idm = Mm[:, :, 1:, :], np.broadcast_to(Mm[:, :, :1, :], (S, m, K - 1, 4))
        fm = np.broadcast_to(fmax[:, :, None, None], (S, m, K - 1, 1))
        return np.concatenate([x, c, own, idm, fm], -1).reshape(S * m * (K - 1), -1)
    with threadpool_limits(limits=threads, user_api="openmp"):
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                                          early_stopping=False, random_state=0).fit(rows(tr_side, tr), Y[:, tr, 1:].reshape(-1))

    def pred(side, idx):
        with threadpool_limits(limits=threads, user_api="openmp"):
            p = g.predict(rows(side, idx)).reshape(side["S"], len(idx), K - 1)
        return np.concatenate([np.zeros((side["S"], len(idx), 1)), p], -1)
    return pred


def rule_picks(side, fk, key):
    """(T, S, n) picks of the one-parameter rule for every tau (decision 187 P3 / N3)."""
    cand = FI_[fk][1]
    Mf = sel(side, fk, key)[..., 3]
    allowed = cand[:, 2] >= 1
    cost = np.abs(cand[:, 0]) / 0.5 + np.abs(np.log(cand[:, 1])) / np.log(1.15) + np.abs(np.log(cand[:, 2])) / np.log(1.2)
    P = []
    for tau in TS.TAUS:
        ok = (Mf >= tau) & allowed
        ok[..., 0] = False
        alt = np.where(ok, cost, np.inf).argmin(-1)
        best = np.where(allowed, Mf, -np.inf).argmax(-1)
        P.append(np.where(Mf[..., 0] >= tau, 0, np.where(ok.any(-1), alt, best)))
    return np.stack(P)


def fit_R(tr_side, arm, fk, tr, salt):
    key = MARG[arm]
    P = rule_picks(tr_side, fk, key)
    Y = tr_side["Y", fk]
    tau = int(np.argmax([TD.take(Y[:, tr], p[:, tr]).sum() for p in P]))
    K = Y.shape[-1]

    def pred(side, idx):
        pk = rule_picks(side, fk, key)[tau][:, idx]
        return (np.arange(K)[None, None] == pk[..., None]).astype(float) * (pk[..., None] > 0)
    pred.info = dict(tau=TS.TAUS[tau])
    return pred


_TORCH = {}


def fit_nn(tr_side, arm, fk, tr, seed, cfg, dev_cache=_TORCH):
    import torch
    import torch.nn as nn
    dev = torch.device("cuda")
    w, p_drop, wd = cfg
    spec = NN_SPEC[arm]
    Y = tr_side["Y", fk]
    K = Y.shape[-1]
    ins_np = lambda side: dict(e=xof(side, ["ego", "plan"], 0, fk), **({"h": side["Hraw"]} if spec.get("h") else {}),  # noqa: E731
                               **({"c": xof(side, [spec["c"] + "flat"], 0, fk)} if spec.get("c") else {}))
    lc = tr_side["lc"]
    lg = np.random.default_rng(900 + seed).permutation(np.unique(lc[tr]))
    va_l = lg[: max(1, len(lg) // 5)]
    va = tr[np.isin(lc[tr], va_l)]
    trn = tr[~np.isin(lc[tr], va_l)]
    ins = ins_np(tr_side)
    mu = {k: (v[:, trn].reshape(-1, v.shape[-1]).mean(0), v[:, trn].reshape(-1, v.shape[-1]).std(0) + 1e-6) for k, v in ins.items()}
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    Z = {k: T((v - mu[k][0]) / mu[k][1])[0] for k, v in ins.items()}                              # train side has one seed row
    V3 = dev_cache.get(id(tr_side["V3"]))
    if spec.get("tok") and V3 is None:
        V3 = dev_cache[id(tr_side["V3"])] = torch.from_numpy(tr_side["V3"]).to(dev)
    ysd = float(Y[:, trn].std()) + 1e-9
    Yt = T(Y[0, :, 1:] / ysd)
    torch.manual_seed(seed)

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            if spec.get("tok"):
                s.ln, s.proj, s.q = nn.LayerNorm(512), nn.Linear(512, 64), nn.Parameter(torch.randn(4, 64) * 0.1)
            s.e = nn.Sequential(nn.Linear(Z["e"].shape[-1], w), nn.GELU())
            s.h = nn.Sequential(nn.Linear(Z["h"].shape[-1], w), nn.GELU()) if "h" in Z else None
            s.c = nn.Sequential(nn.Linear(Z["c"].shape[-1], w), nn.GELU()) if "c" in Z else None
            d = w * (1 + ("h" in Z) + ("c" in Z)) + (256 if spec.get("tok") else 0)
            s.out = nn.Sequential(nn.Linear(d, 2 * w), nn.GELU(), nn.Dropout(p_drop), nn.Linear(2 * w, K - 1))

        def forward(s, tk, e, h, c):
            z = [s.e(e)]
            if spec.get("tok"):
                x = s.proj(s.ln(tk.float()))
                att = torch.softmax(torch.einsum("qd,bnd->bqn", s.q, x) / 8.0, -1)
                z.append(torch.einsum("bqn,bnd->bqd", att, x).flatten(1))
            if s.h is not None:
                z.append(s.h(h))
            if s.c is not None:
                z.append(s.c(c))
            return s.out(torch.cat(z, -1))
    net = Head().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)

    def batch(Zs, V, ti):
        return (V[ti] if spec.get("tok") else None), Zs["e"][ti], (Zs["h"][ti] if "h" in Zs else None), (Zs["c"][ti] if "c" in Zs else None)

    def predict_z(Zs, V, idx):
        net.eval()
        with torch.no_grad():
            ti = torch.as_tensor(idx, device=dev)
            return torch.cat([net(*batch(Zs, V, ti[b:b + 2048])) for b in range(0, len(ti), 2048)]).cpu().numpy()
    best, best_g = None, -np.inf
    Yva = Y[:, va]
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.default_rng(seed * 1000 + ep).permutation(trn)
        for b in range(0, len(perm), 512):
            ti = torch.as_tensor(perm[b:b + 512], device=dev)
            loss = ((net(*batch(Z, V3, ti)) - Yt[ti]) ** 2).mean()
            opt.zero_grad(), loss.backward(), opt.step()
        pv = np.concatenate([np.zeros((1, len(va), 1)), predict_z(Z, V3, va)[None]], -1)
        gv = TD.take(Yva, TD.picks_of(pv)).mean()
        if gv > best_g + 1e-12:
            best_g, best = gv, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best)

    def pred(side, idx):
        sin = ins_np(side)
        Vs = None
        if spec.get("tok"):
            Vs = V3 if side is tr_side else dev_cache.setdefault(id(side["V3"]), torch.from_numpy(side["V3"]).to(dev))
        out = []
        for s_ in range(side["S"]):
            Zs = {k: T((v[s_] - mu[k][0]) / mu[k][1]) for k, v in sin.items()}
            out.append(predict_z(Zs, Vs, idx))
        return np.concatenate([np.zeros((side["S"], len(idx), 1)), np.stack(out)], -1)
    pred.info = dict(val_gain=float(best_g), cfg=cfg)
    return pred


def log_subset(side, frac, rep):
    lg = np.random.default_rng(100 + rep).permutation(np.unique(side["lc"]))
    keep = lg[: max(INNER + 1, int(round(frac * len(lg))))]
    return np.flatnonzero(np.isin(side["lc"], keep))


def cmd_fit(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    arm = a.arm
    a.cfg_l9 = CFG[0]
    kind = "NN" if arm in NN_SPEC else "G" if arm in ("N2", "P2") else "R" if arm in ("N3", "P3") else "L"
    fks = FK if (kind != "NN" or arm in ("N7", "P4")) else FK[:1]
    t0 = time.time()
    with Run("op_parity", f"turn_selnt/fit-{a.tag}-{arm}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        tr, te = load_train(a.tag), load_test()
        add_pca(tr, te)
        n = tr["n"]
        allx = np.arange(n)
        res = dict(arm=arm, kind=kind, tag=a.tag, n_train=n, test_tokens=te["tokens"])
        for fk in fks:
            r = {}
            f5 = folds_by_log(tr, allx, FOLDS_OUT if kind != "NN" else FOLDS_NN, 11)
            if kind in ("L", "G", "R"):
                fitter = {"L": fit_L, "G": fit_G, "R": fit_R}[kind]
                oof = np.zeros((1, n, tr["Y", fk].shape[-1]))
                for j in range(f5.max() + 1):
                    p = fitter(tr, arm, fk, allx[f5 != j], j)
                    oof[:, f5 == j] = p(tr, allx[f5 == j])
                p = fitter(tr, arm, fk, allx, 99)
                r["info"] = getattr(p, "info", {})
                r["test_pred"] = p(te, np.arange(te["n"]))
                if arm in CURVE_ARMS and fk == FK[0]:
                    r["curve"] = {}
                    for fr in FRACS:
                        r["curve"][fr] = []
                        for rep in range(CURVE_ARMS[arm]):
                            sub = log_subset(tr, fr, rep)
                            r["curve"][fr].append(fitter(tr, arm, fk, sub, 50 + rep)(te, np.arange(te["n"])))
                    r["curve"][1.0] = [r["test_pred"]]
            else:
                # config selection by the out-of-fold realised gain on navtrain (3 folds by log); the selected config is read out
                cfgs = CFG if fk == FK[0] else [a.cfg_l9]
                oofs, scores = [], []
                for ci, cfg in enumerate(cfgs):
                    oof = np.zeros((1, n, tr["Y", fk].shape[-1]))
                    for j in range(f5.max() + 1):
                        oof[:, f5 == j] = fit_nn(tr, arm, fk, allx[f5 != j], 10 + j, cfg)(tr, allx[f5 == j])
                    oofs.append(oof)
                    scores.append(gain(tr, fk, allx, oof))
                    run.info("%s %s cfg %s: navtrain OOF gain %+.4f", arm, fk, cfg, 100 * scores[-1])
                bi = int(np.argmax(np.array(scores) + 1e-9 * -np.arange(len(cfgs))))                  # ties: earlier = smaller
                cfg, oof = cfgs[bi], oofs[bi]
                r.update(cfg=cfg, cfg_scores=dict(zip(map(str, cfgs), scores)))
                sel_cfg = cfg
                if fk == FK[0]:
                    a.cfg_l9 = cfg
                preds = [fit_nn(tr, arm, fk, allx, 200 + s, cfg)(te, np.arange(te["n"])) for s in range(a.ninit)]
                r["test_pred_each"] = np.stack(preds)
                r["test_pred"] = np.mean(preds, 0)
                if arm in CURVE_ARMS and fk == FK[0]:
                    r["curve"] = {1.0: preds[:CURVE_ARMS[arm]]}
                    for fr in FRACS:
                        r["curve"][fr] = [fit_nn(tr, arm, fk, log_subset(tr, fr, rep), 300 + rep, sel_cfg)(te, np.arange(te["n"])) for rep in range(CURVE_ARMS[arm])]
                run.info("%s %s: %d inits, cfg %s", arm, fk, a.ninit, cfg)
            r["oof_gain_navtrain"] = 100 * gain(tr, fk, allx, oof)
            r["oof_pred_picks"] = TD.picks_of(oof)[0].astype(np.int8)
            res[fk] = r
            run.info("%s %s: navtrain OOF gain %+.3f  (elapsed %.0f s)", arm, fk, r["oof_gain_navtrain"], time.time() - t0)
        res["wall_s"] = time.time() - t0
        (OUT / "fit" / a.tag).mkdir(parents=True, exist_ok=True)
        with open(OUT / "fit" / a.tag / f"{arm}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        run.summary.update(arm=arm, wall_s=res["wall_s"], **{f"oof_{fk}": res[fk]["oof_gain_navtrain"] for fk in fks})


# ---------------------------------------------------------------- 4. gates on the navtest side
def cmd_ctrl(a):
    """G-E: the navtest-internal E arm of decision 186 (turn_selinput.unit2 on turn_dewater's folds), 10 repeats, F19 x pc."""
    import multiprocessing as mp
    from jevdrive import par, stats
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    with Run("op_parity", "turn_selnt/ctrl", seed=0, config=vars(a)) as run:
        C, tok, dyaw, log, X, F = TS.prepare()
        units = [("oof", "E", FK[0], r, 1.0) for r in range(TS.REPS_L)]
        res = par.pmap(TS.unit2, units, workers=max(1, min(n_cpus() // 4, 10)), run=run, desc="E units", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        d = np.mean([r["d"] for _, r in res.values], 0)
        g = stats.paired(100 * d, np.zeros(len(d)), groups=log)
        out = dict(mean=g["mean"], lo=g["lo"], hi=g["hi"], reference=-0.27, ok=bool(abs(g["mean"] - (-0.27)) <= 0.01))
        (OUT / "gate_E.json").write_text(json.dumps(out, indent=1))
        run.info("G-E %s", out)
        run.summary.update(out)
        assert out["ok"], f"G-E failed: {out}"


def cmd_hidden(a):
    """G-hidden: per-coordinate Pearson correlation between the fold models' and SH30-F-s0's hidden state on the navtest turn tokens."""
    from jevdrive.run import Run
    with Run("op_parity", "turn_selnt/hidden", seed=0, config=vars(a)) as run:
        z = np.load(TX.out_dir(a.tag) / "navtest.npz")
        te = load_test()
        ref = te["Hraw"][0]
        rows = []
        for j in range(5):
            H = np.concatenate([z[f"select_4_{j}"], z[f"mean_{j}"]], 1).astype(np.float32)
            Hc, Rc = H - H.mean(0), ref - ref.mean(0)
            r = (Hc * Rc).sum(0) / (np.sqrt((Hc ** 2).sum(0) * (Rc ** 2).sum(0)) + 1e-9)
            rows.append(dict(fold=j, median_corr=float(np.median(r)), q10=float(np.quantile(r, 0.1)), q90=float(np.quantile(r, 0.9)),
                             median_corr_select4=float(np.median(r[:512])), median_corr_mean=float(np.median(r[512:]))))
        out = dict(per_fold=rows, median_of_medians=float(np.median([r["median_corr"] for r in rows])))
        out["ok"] = out["median_of_medians"] >= 0.8
        (OUT / f"gate_hidden_{a.tag}.json").write_text(json.dumps(out, indent=1))
        run.info("G-hidden %s", out)
        run.summary.update(ok=out["ok"], median=out["median_of_medians"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "fit", "ctrl", "hidden", "report"):
        p = sp.add_parser(name)
        p.add_argument("--tag", default="full")
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--out", default=str(RES))
        p.add_argument("--figs", default=str(FIG))
        if name == "fit":
            p.add_argument("--arm", required=True, choices=ARMS)
            p.add_argument("--ninit", type=int, default=NINIT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TD.OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
