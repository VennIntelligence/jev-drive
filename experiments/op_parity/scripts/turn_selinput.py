"""op_parity turn selector input (plans/2026-10-08-turn-selector-input-prereg.md): is the per-token pick of decision 186 unlearnable because the
selector lacks drivable-boundary geometry, or at this label scale whatever the input? Reuses turn_dewater's families, score conventions, folds
(by log), ridge / tree heads and statistics. No new simulator scoring; CPU except the neural arms (pool job).

  feats      per-candidate footprint margins (4 s, cumulative to 1 / 2 / 3 / 4 s): map SDF (PRIVILEGED, `M`) and SH30's own calibrated road edges (`C`);
             gates G1 / G2 -> $OUT/margins.npz, gates.json
  select     ridge (L), boosted-tree (G), one-parameter rule (R) arms + the E control -> $OUT/select2.pkl
  nn         small attention / MLP heads on unpooled vision tokens, SH30 hidden state, per-candidate own-edge margins (GPU, pool job) -> $OUT/nn.pkl
  report     tables + figures
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json, pickle  # noqa: E401,E402

import numpy as np  # noqa: E402

import turn_ceiling as TC  # noqa: E402
import turn_dewater as TD  # noqa: E402

D = TC.D
OUT = D / "runs/op_parity/turn_selinput"
RES = _R / "experiments/op_parity/results/turn_selector_input"
FIG = _R / "experiments/op_parity/figs/turn_selector_input"
FKS = (("F19", "pc"), ("L9", "epcap"))
FK = [TD.fkey(*x) for x in FKS]
S = TD.S
TIDX = [10, 20, 30, 40]                                # 1 / 2 / 3 / 4 s on the 0.1 s grid
CLIP = (-2.0, 4.0)
TAUS = (-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
L_ARMS = {"E": ["ego", "plan"], "P1": ["ego", "plan", "Mflat"], "N1": ["ego", "plan", "Cflat"], "N4": ["H", "ego", "plan"]}
MARG = {"P1": "M", "P2": "M", "P3": "M", "N1": "C", "N2": "C", "N3": "C"}
G_ARMS, R_ARMS, NN_ARMS = ("P2", "N2"), ("P3", "N3"), ("N5", "N6", "N7")
REPS_L, REPS_G, REPS_NN, CURVE_REPS = 10, 5, 5, {"L": 10, "R": 10, "G": 2, "NN": 3}
CURVE = TD.CURVE
PCA_STREAMS = ("T", "V", "H")
_D = TD._D
LAB = D / "runs/op_probe/labels/navtest.npz"
HID = OUT / "hidden/SH30-F-s{}-warp__lb_navtest.npz"
INF = D / "runs/op_parity/self_consist/infer/SH30-F-s{}-warp__lb_navtest.npz"
CAL = _R / "experiments/op_parity/results/self_consist/calibration.json"


# ---------------------------------------------------------------- 1. margins
def _m_sdf(u):
    import sc_analyze as SC
    s, c = u
    cor = SC.footprint(_D["poses"][TC.key(s, c)].astype(np.float64))
    v = SC.sdf_at(_D["sdf"], cor).min(-1)                                       # (n, 41) min over the 4 corners
    return (s, c), np.minimum.accumulate(v, 1)[:, TIDX]


def _m_curb(u):
    import sc_analyze as SC
    s, c = u
    cor = SC.footprint(_D["poses"][TC.key(s, c)].astype(np.float64))
    ex, ey, (sc_, b_) = _D["ex"][s], _D["ey"][s], _D["cal"][s]
    yl, yr = sc_ * ey[:, 0] + b_, sc_ * ey[:, 1] - b_
    out = np.empty((len(cor), 4))
    for i in range(len(cor)):
        cx, cy = cor[i, ..., 0], cor[i, ..., 1]                                  # (41, 4)
        m = np.minimum(np.interp(cx.ravel(), ex[i], yl[i]).reshape(cx.shape) - cy, cy - np.interp(cx.ravel(), ex[i], yr[i]).reshape(cx.shape)).min(1)
        out[i] = np.minimum.accumulate(m)[TIDX]
    return (s, c), np.where(np.isfinite(out), out, CLIP[1])


def _auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool).ravel()
    n1 = y.sum()
    return float((rankdata(np.asarray(score).ravel())[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def cmd_feats(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    import multiprocessing as mp
    import sc_analyze as SC
    with Run("op_parity", "turn_selinput/feats", seed=0, config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        C, tok, dyaw, log, X = TD.load()
        assert nt.mask(tok).all()
        Z = np.load(TC.OUT / "poses.npz")
        _D["poses"] = {TC.key(s, c): Z[TC.key(s, c)] for s in TC.SEEDS for c in range(len(C))}
        lab = np.load(LAB)
        lt = {t: i for i, t in enumerate(lab["tokens"].tolist())}
        _D["sdf"] = lab["sdf"][[lt[t] for t in tok]].astype(np.float32)
        tab = SC.tab_of("test")
        row = {t: i for i, t in enumerate(tab["names"].tolist())}
        use = np.array([row[t] for t in tok])
        cal = json.loads(CAL.read_text())
        _D["ex"], _D["ey"], _D["cal"] = {}, {}, {}
        for s in TC.SEEDS:
            nm, re_mu, *_ = SC.load_edges(f"SH30-F-s{s}", "test")
            assert (nm == tab["names"]).all()
            _D["ex"][s], _D["ey"][s] = SC.edges_ego(re_mu[use], tab["cam"][use])
            _D["cal"][s] = (cal[f"SH30-F-s{s}"]["s"], cal[f"SH30-F-s{s}"]["b"])
        units = [(s, c) for s in TC.SEEDS for c in range(len(C))]
        workers = max(1, min(n_cpus() // 4, 32, len(units)))
        M, Cm = np.zeros((len(TC.SEEDS), len(C), len(tok), 4)), np.zeros((len(TC.SEEDS), len(C), len(tok), 4))
        for fn, A_ in ((_m_sdf, M), (_m_curb, Cm)):
            r = par.pmap(fn, units, workers=workers, run=run, desc=fn.__name__, mp_context=mp.get_context("fork"))
            r.raise_if_failed()
            for (s, c), v in r.values:
                A_[s, c] = v
        M, Cm = np.clip(M, *CLIP), np.clip(Cm, *CLIP)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "margins.npz", M=M.astype(np.float32), C=Cm.astype(np.float32), tokens=tok)
        dac_fail = X[:, 0, :, 1] < 1                                             # identity candidate, DAC in bench TERMS order
        g1, g2 = _auc(-M[:, 0, :, 3], dac_fail), _auc(-Cm[:, 0, :, 3], dac_fail)
        gates = dict(G1_map_auc=g1, G1_ok=bool(g1 >= 0.85), G2_curb_auc=g2, G2_ok=bool(0.58 <= g2 <= 0.72), dac_fail_rate=float(dac_fail.mean()))
        (OUT / "gates.json").write_text(json.dumps(gates, indent=1))
        run.info("gates %s", gates)
        run.summary.update(gates)
        assert gates["G1_ok"] and gates["G2_ok"], f"gate failed: {gates}"


# ---------------------------------------------------------------- 2. shared setup
def prepare(neural=False):
    """Fill turn_dewater's shared arrays (scores, folds, E / T / V streams) and the margin, hidden-state and (neural) token streams."""
    import pandas as pd
    C, tok, dyaw, log, X = TD.load()
    F = TD.fams(C)
    feats, finfo = TD.features(tok)
    _D.update(feats)
    _D["lcode"] = pd.factorize(log)[0]
    Sraw = TC.noec(X)
    for f, cv in FKS:
        fk = TD.fkey(f, cv)
        Sc = TC.noec(TD.conv(X, C, cv))
        _D["Y", fk] = np.ascontiguousarray((Sc[:, F[f]] - Sc[:, :1]).transpose(0, 2, 1))
        _D["Yraw", fk] = np.ascontiguousarray((Sraw[:, F[f]] - Sraw[:, :1]).transpose(0, 2, 1))
        _D["cand", fk] = np.array([C[i][1:] for i in F[f]], float)
    Z = np.load(OUT / "margins.npz")
    assert np.array_equal(Z["tokens"], tok)
    for f, cv in FKS:
        fk = TD.fkey(f, cv)
        for nm in ("M", "C"):
            m = np.ascontiguousarray(Z[nm][:, F[f]].transpose(0, 2, 1, 3)).astype(np.float64)     # (S, n, K, 4)
            _D[f"{nm}|{fk}"], _D[f"{nm}flat|{fk}"] = m, m.reshape(m.shape[0], m.shape[1], -1)
    # SH30 hidden state (select_4 + mean) per seed: PCA fitted on the non-turn tokens of the same seed, no scores
    import sc_analyze as SC
    tab = SC.tab_of("test")
    row = {t: i for i, t in enumerate(tab["names"].tolist())}
    use = np.array([row[t] for t in tok])
    fit = np.setdiff1d(np.arange(len(row)), use)
    Hw, Hr = [], []
    for s in TC.SEEDS:
        h = np.load(HID.format(s))
        assert (h["names"] == tab["names"]).all()
        H = np.concatenate([h["select_4"], h["mean"]], 1).astype(np.float32)
        Hw.append(TD.pca_white(H, fit, use)), Hr.append(H[use])
    _D["H"], _D["Hraw"] = np.stack(Hw), np.stack(Hr).astype(np.float64)
    if neural:
        front = np.load(D / "runs/op_parity/cache/lb_navtest@warp/front.npy", mmap_mode="r")
        _D["V3"] = np.ascontiguousarray(np.asarray(front[:, -1])[use])               # (n, 32, 512) fp16
    return C, tok, dyaw, log, X, F


def xof(streams, k, fk):
    out = []
    for s in streams:
        if s in PCA_STREAMS:
            out.append(_D[s][..., :k])
        elif s in ("Mflat", "Cflat"):
            out.append(_D[f"{s}|{fk}"])
        else:
            out.append(_D[s])
    return np.concatenate(out, -1)


def _fold_train(tr, lc, opt, r, j):
    if opt >= 1.0:
        return tr
    lg = np.unique(lc[tr])
    return tr[np.isin(lc[tr], np.random.default_rng(7000 + r * 10 + j).permutation(lg)[: max(TD.INNER, int(round(opt * len(lg))))])]


def unit2(a):
    """One unit -> (tag, dict). kinds oof / curve (fraction `opt` of the training logs). Gains are seed means under the family's convention.
    d: per-token out-of-fold gain; picks (S, n); sc (S, n): score for the 'a repair exists' AUC."""
    kind, arm, fk, rep, opt = a
    if arm == "E":
        tag, r = TD.unit(a)
        return tag, dict(d=r["d"], picks=r["picks"])
    Y, lc = _D["Y", fk], _D["lcode"]
    n, K = Y.shape[1], Y.shape[2]
    tag = (kind, arm, fk, rep, opt)
    f = TD.folds(rep)
    d, pk, sc = np.zeros(n), np.zeros((S, n), int), np.zeros((S, n))
    if arm in R_ARMS:
        Mf = _D[f"{MARG[arm]}|{fk}"][..., 3]                                         # (S, n, K) 4 s margin
        cand = _D["cand", fk]
        allowed = cand[:, 2] >= 1
        cost = np.abs(cand[:, 0]) / 0.5 + np.abs(np.log(cand[:, 1])) / np.log(1.15) + np.abs(np.log(cand[:, 2])) / np.log(1.2)
        P = []
        for tau in TAUS:
            ok = (Mf >= tau) & allowed
            ok[..., 0] = False
            alt = np.where(ok, cost, np.inf).argmin(-1)
            best = np.where(allowed, Mf, -np.inf).argmax(-1)                          # all allowed candidates incl. the identity
            P.append(np.where(Mf[..., 0] >= tau, 0, np.where(ok.any(-1), alt, best)))
        P = np.stack(P)                                                              # (T, S, n)
        for j in range(TD.FOLDS):
            te, tr = np.flatnonzero(f == j), _fold_train(np.flatnonzero(f != j), lc, opt, rep, j)
            g = [TD.take(Y[:, tr], p[:, tr]).sum() for p in P]
            p = P[int(np.argmax(g))]
            d[te], pk[:, te] = TD.take(Y[:, te], p[:, te]).mean(0), p[:, te]
        sc = -Mf[..., 0]
        return tag, dict(d=d, picks=pk, sc=sc)
    if arm in G_ARMS:
        from sklearn.ensemble import HistGradientBoostingRegressor
        E = xof(["ego", "plan"], 0, fk)
        cf, Mm = _D["cand", fk][1:], _D[f"{MARG[arm]}|{fk}"]                         # (K-1, 3), (S, n, K, 4)
        fmax = Mm[..., 3].max(-1)

        def rows(idx):
            m = len(idx)
            x = np.broadcast_to(E[:, idx][:, :, None, :], (S, m, K - 1, E.shape[-1]))
            c = np.broadcast_to(cf[None, None], (S, m, K - 1, 3))
            own, idm = Mm[:, idx, 1:, :], np.broadcast_to(Mm[:, idx, :1, :], (S, m, K - 1, 4))
            fm = np.broadcast_to(fmax[:, idx, None, None], (S, m, K - 1, 1))
            return np.concatenate([x, c, own, idm, fm], -1).reshape(S * m * (K - 1), -1)
        for j in range(TD.FOLDS):
            te, tr = np.flatnonzero(f == j), _fold_train(np.flatnonzero(f != j), lc, opt, rep, j)
            g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                                              early_stopping=False, random_state=0).fit(rows(tr), Y[:, tr, 1:].reshape(-1))
            pr = np.concatenate([np.zeros((S, len(te), 1)), g.predict(rows(te)).reshape(S, len(te), K - 1)], -1)
            p = TD.picks_of(pr)
            d[te], pk[:, te], sc[:, te] = TD.take(Y[:, te], p).mean(0), p, pr[..., 1:].max(-1)
        return tag, dict(d=d, picks=pk, sc=sc)
    # ridge arms
    streams = L_ARMS[arm]
    Xk = {k: xof(streams, k, fk) for k in (TD.KS if any(s in PCA_STREAMS for s in streams) else TD.KS[:1])}
    for j in range(TD.FOLDS):
        te, tr = np.flatnonzero(f == j), _fold_train(np.flatnonzero(f != j), lc, opt, rep, j)
        al, _, Pr = TD.select_fit(Xk, Y, tr, TD.folds(rep * 10 + j, TD.INNER, 1000))
        Xl = Xk[al[0]]
        pr = Pr[al[0]](Xl[:, te].reshape(-1, Xl.shape[-1]), al[1]).reshape(S, len(te), K)
        pr[..., 0] = 0.0
        p = TD.picks_of(pr)
        d[te], pk[:, te], sc[:, te] = TD.take(Y[:, te], p).mean(0), p, pr[..., 1:].max(-1)
    return tag, dict(d=d, picks=pk, sc=sc)


def cmd_select(a):
    import multiprocessing as mp
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_selinput/select", seed=0, config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        C, tok, dyaw, log, X, F = prepare()
        assert nt.mask(tok).all()
        fam_arms = ["E", *L_ARMS.keys() - {"E"}, *R_ARMS]
        units = []
        for arm in fam_arms:
            kd = "R" if arm in R_ARMS else "L"
            for fk in FK:
                units += [("oof", arm, fk, r, 1.0) for r in range(REPS_L)]
                units += [("curve", arm, fk, r, fr) for fr in CURVE for r in range(CURVE_REPS[kd])]
        trees = [("oof", arm, fk, r, 1.0) for arm in G_ARMS for fk in FK for r in range(REPS_G)]
        trees += [("curve", arm, fk, r, fr) for arm in G_ARMS for fk in FK for fr in CURVE for r in range(CURVE_REPS["G"])]
        if a.smoke:
            units = [u for u in units if u[3] == 0 and u[2] == FK[0]]
            trees = [u for u in trees if u[3] == 0 and u[2] == FK[0] and u[0] == "oof"]
        workers = max(1, min(n_cpus() // 4, a.workers))
        run.info("%d ridge/rule units on %d workers, %d tree units", len(units), workers, len(trees))
        res = par.pmap(unit2, units, workers=workers, run=run, desc="ridge/rule units", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=min(workers, 24), user_api="openmp"):
            res.values += [unit2(u) for u in run.tqdm(trees, desc="tree units")]
        with open(OUT / ("select2_smoke.pkl" if a.smoke else "select2.pkl"), "wb") as fh:
            pickle.dump(dict(res=dict(res.values), tok=tok, dyaw=dyaw, log=log), fh)
        run.summary.update(n_units=len(units) + len(trees), workers=workers)


# ---------------------------------------------------------------- 3. neural heads (GPU)
def cmd_nn(a):
    import torch
    import torch.nn as nn
    from jevdrive.data import splits
    from jevdrive.run import Run
    torch.set_num_threads(4)
    with Run("op_parity", "turn_selinput/nn", seed=0, config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        C, tok, dyaw, log, X, F = prepare(neural=True)
        dev = torch.device("cuda")
        V3 = torch.from_numpy(_D["V3"]).to(dev)                                           # (n, 32, 512) fp16
        n = len(tok)
        lc = _D["lcode"]

        class Head(nn.Module):
            def __init__(s, k1, de, dh=0, dc=0, tok_=False):
                super().__init__()
                s.tok_ = tok_
                if tok_:
                    s.ln, s.proj, s.q = nn.LayerNorm(512), nn.Linear(512, 64), nn.Parameter(torch.randn(4, 64) * 0.1)
                s.e = nn.Sequential(nn.Linear(de, 64), nn.GELU())
                s.h = nn.Sequential(nn.Linear(dh, 64), nn.GELU()) if dh else None
                s.c = nn.Sequential(nn.Linear(dc, 64), nn.GELU()) if dc else None
                d = 64 * (1 + bool(dh) + bool(dc)) + (256 if tok_ else 0)
                s.out = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, k1))

            def forward(s, tk, e, h, c):
                z = [s.e(e)]
                if s.tok_:
                    x = s.proj(s.ln(tk.float()))
                    att = torch.softmax(torch.einsum("qd,bnd->bqn", s.q, x) / 8.0, -1)
                    z.append(torch.einsum("bqn,bnd->bqd", att, x).flatten(1))
                if s.h is not None:
                    z.append(s.h(h))
                if s.c is not None:
                    z.append(s.c(c))
                return s.out(torch.cat(z, -1))

        def make(arm, fk):
            E = xof(["ego", "plan"], 0, fk)
            spec = dict(N5=dict(tok=True), N6=dict(h=True), N7=dict(tok=True, h=True, c=True))[arm]
            ins = dict(e=E, h=_D["Hraw"] if spec.get("h") else None, c=_D[f"Cflat|{fk}"] if spec.get("c") else None)
            return spec, ins

        def run_fold(arm, fk, rep, opt, j, f, spec, ins):
            Y = _D["Y", fk]
            K = Y.shape[2]
            te = np.flatnonzero(f == j)
            tr_all = _fold_train(np.flatnonzero(f != j), lc, opt, rep, j)
            lg = np.random.default_rng(900 + rep * 10 + j).permutation(np.unique(lc[tr_all]))
            va_l = lg[: max(1, len(lg) // 5)]
            va = tr_all[np.isin(lc[tr_all], va_l)]
            tr = tr_all[~np.isin(lc[tr_all], va_l)]
            mu = {k: (v[:, tr].reshape(-1, v.shape[-1]).mean(0), v[:, tr].reshape(-1, v.shape[-1]).std(0) + 1e-6) for k, v in ins.items() if v is not None}
            T = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
            Z = {k: T((v - mu[k][0]) / mu[k][1]) for k, v in ins.items() if v is not None}       # (S, n, d)
            ysd = float(Y[:, tr].std()) + 1e-9
            Yt = T(Y[:, :, 1:] / ysd)
            torch.manual_seed(rep * 100 + j)
            net = Head(K - 1, Z["e"].shape[-1], Z["h"].shape[-1] if "h" in Z else 0, Z["c"].shape[-1] if "c" in Z else 0, spec.get("tok", False)).to(dev)
            opt_ = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)

            def batch(idx_s, idx_t):
                g = lambda k: Z[k][idx_s, idx_t] if k in Z else None  # noqa: E731
                return V3[idx_t] if spec.get("tok") else None, g("e"), g("h"), g("c")

            def predict(idx):
                net.eval()
                with torch.no_grad():
                    out = []
                    for s_ in range(S):
                        ti = torch.as_tensor(idx, device=dev)
                        out.append(torch.cat([net(*batch(s_, ti[b:b + 1024]))[:, :] for b in range(0, len(ti), 1024)]))
                    p = torch.stack(out).cpu().numpy()                                    # (S, m, K-1)
                return np.concatenate([np.zeros((S, len(idx), 1)), p], -1)
            rows = np.array([(s_, t) for s_ in range(S) for t in tr])
            best, best_g = None, -np.inf
            for ep in range(30):
                net.train()
                perm = np.random.default_rng(rep * 1000 + j * 50 + ep).permutation(len(rows))
                for b in range(0, len(perm), 512):
                    r_ = rows[perm[b:b + 512]]
                    si, ti = torch.as_tensor(r_[:, 0], device=dev), torch.as_tensor(r_[:, 1], device=dev)
                    loss = ((net(*batch(si, ti)) - Yt[si, ti]) ** 2).mean()
                    opt_.zero_grad(), loss.backward(), opt_.step()
                gv = TD.take(Y[:, va], TD.picks_of(predict(va))).mean()
                if gv > best_g + 1e-12:
                    best_g, best = gv, {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(best)
            return te, predict(te)

        units = [("oof", arm, fk, r, 1.0) for arm in NN_ARMS for fk in FK for r in range(REPS_NN)]
        units += [("curve", arm, fk, r, fr) for arm in NN_ARMS for fk in FK for fr in CURVE for r in range(CURVE_REPS["NN"])]
        if a.smoke:
            units = [u for u in units if u[3] == 0 and u[2] == FK[0] and u[0] == "oof"][:1]
        res = {}
        for kind, arm, fk, rep, opt in run.tqdm(units, desc="nn units"):
            spec, ins = make(arm, fk)
            Y = _D["Y", fk]
            f = TD.folds(rep)
            d, pk, sc = np.zeros(n), np.zeros((S, n), int), np.zeros((S, n))
            for j in range(TD.FOLDS):
                te, pr = run_fold(arm, fk, rep, opt, j, f, spec, ins)
                p = TD.picks_of(pr)
                d[te], pk[:, te], sc[:, te] = TD.take(Y[:, te], p).mean(0), p, pr[..., 1:].max(-1)
            res[kind, arm, fk, rep, opt] = dict(d=d, picks=pk, sc=sc)
            run.info("%s %s %s rep %d opt %.2f: gain %+.3f", kind, arm, fk, rep, opt, 100 * d.mean())
        with open(OUT / ("nn_smoke.pkl" if a.smoke else "nn.pkl"), "wb") as fh:
            pickle.dump(dict(res=res, tok=tok), fh)
        run.summary.update(n_units=len(units))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("feats", "select", "nn", "report"):
        p = sp.add_parser(name)
        p.add_argument("--out", default=str(RES))
        p.add_argument("--figs", default=str(FIG))
        p.add_argument("--workers", type=int, default=48)
        p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TD.OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
