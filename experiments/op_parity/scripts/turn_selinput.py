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
HID = str(OUT / "hidden/SH30-F-s{}-warp__lb_navtest.npz")
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


# ---------------------------------------------------------------- 4. report
ROWS = [("E", "L", "E: ego + plan (control, = decision 186)", ""), ("P1", "L", "P1: E + per-candidate map margin (ridge)", "PRIV"),
        ("P2", "G", "P2: E + map margin (trees on candidate rows)", "PRIV"), ("P3", "R", "P3: map-margin rule, 1 parameter", "PRIV"),
        ("N1", "L", "N1: E + own-edge margin (ridge)", ""), ("N2", "G", "N2: E + own-edge margin (trees)", ""), ("N3", "R", "N3: own-edge margin rule, 1 parameter", ""),
        ("N4", "L", "N4: E + SH30 hidden state (PCA, ridge)", ""), ("N5", "NN", "N5: E + unpooled vision tokens (attention head)", ""),
        ("N6", "NN", "N6: E + SH30 hidden state (MLP)", ""), ("N7", "NN", "N7: tokens + hidden + own-edge margins + E (all non-privileged)", "")]
NP, NN_ = 3, 7                                             # multiplicity: privileged arms, non-privileged arms
THRESH = dict(large=0.40, little=0.20, rising=0.3)


def cmd_report(a):
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    from scipy.stats import rankdata
    with Run("op_parity", "turn_selinput/report", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        out, figd = _pl.Path(a.out), _pl.Path(a.figs)
        out.mkdir(parents=True, exist_ok=True)
        C, tok, dyaw, log, X, F = prepare()
        R = {}
        for f in ("select2.pkl", "nn.pkl"):
            with open(OUT / f, "rb") as fh:
                R.update(pickle.load(fh)["res"])
        B = {k: v for k, v in TD.buckets(dyaw).items() if "left" not in k and "right" not in k}
        ceil = {fk: _D["Y", fk].max(-1).mean(0) for fk in FK}
        zero = np.zeros(len(log))

        def mean_of(kind, arm, fk, opt=1.0, key="d"):
            v = [r[key] for k, r in R.items() if k[:3] == (kind, arm, fk) and k[4] == opt and key in r]
            return np.mean(v, 0), len(v)

        def gci(d, m, alpha=0.05):
            return stats.paired(100 * d[m], zero[m], groups=log[m], alpha=alpha)
        G, rows, verdict = {}, [], {}
        for arm, head, lab, priv in ROWS:
            alpha = 0.05 / NP if priv else (0.05 / NN_ if arm != "E" else 0.05)
            for fk in FK:
                d, nrep = mean_of("oof", arm, fk)
                r = {"arm": arm, "head": head, "description": lab, "input": priv or "model-side", "family x convention": fk, "repeats": nrep}
                for b, m in B.items():
                    G[arm, fk, b] = gci(d, m)
                    r[b] = TC.cell(G[arm, fk, b])
                G[arm, fk, "adj"] = gci(d, B["> 20 deg"], alpha)
                r["Bonferroni CI"] = TC.cell(G[arm, fk, "adj"]) if arm != "E" else ""
                r["CI level %"] = f"{100 * (1 - alpha):.2f}" if arm != "E" else "95.00"
                q = TC.ratio_ci(d, ceil[fk], log)
                G[arm, fk, "rec"] = q
                r["recovery of the ceiling"] = f"{100 * q['mean']:.1f}% [{100 * q['lo']:.1f}, {100 * q['hi']:.1f}]"
                pk = np.stack([x["picks"] for k, x in R.items() if k[:3] == ("oof", arm, fk) and k[4] == 1.0])
                r["moved %"] = f"{100 * (pk != 0).mean():.1f}"
                rows.append(r)
        stats.write_table(rows, out / "selector", note="out-of-fold gain over SH30 of the selected candidate under the family's convention, EPDMS x 100 (no EC), seed mean, "
                          "mean over repeats of 5 folds by log; paired log-cluster bootstrap, B 10 000. PRIV = uses map geometry (upper bound, not a method). "
                          f"adj. CI = Bonferroni over {NP} privileged / {NN_} non-privileged arms.")
        # ---- verdict, as pre-registered
        f0, f1 = FK
        priv = {x: dict(lo=G[x, f0, "adj"]["lo"], gain=G[x, f0, "> 20 deg"]["mean"], rec=G[x, f0, "rec"]["mean"]) for x in ("P1", "P2", "P3")}
        nonp = {x: dict(lo=G[x, f0, "adj"]["lo"], lo95=G[x, f0, "> 20 deg"]["lo"], gain=G[x, f0, "> 20 deg"]["mean"], gain_l9=G[x, f1, "> 20 deg"]["mean"],
                        rec=G[x, f0, "rec"]["mean"]) for x in ("N1", "N2", "N3", "N4", "N5", "N6", "N7")}
        p_large = any(v["lo"] > 0 and v["rec"] >= THRESH["large"] for v in priv.values())
        p_little = all(v["rec"] < THRESH["little"] or (v["lo"] <= 0 and v["rec"] < THRESH["large"]) for v in priv.values())
        n_pos = sorted([k for k, v in nonp.items() if v["lo"] > 0 and v["gain_l9"] > 0], key=lambda k: -nonp[k]["gain"])
        n_hint = [k for k, v in nonp.items() if v["lo95"] > 0 and k not in n_pos]
        case = "iii" if n_pos else "i" if p_large else "ii" if p_little else "partial"
        verdict = dict(case=case, privileged_large=p_large, privileged_little=p_little, nonpriv_positive=n_pos, nonpriv_hint_unadjusted=n_hint, priv=priv, nonpriv=nonp,
                       ceiling_f0=float(100 * ceil[f0].mean()), thresholds=THRESH)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
        run.info("verdict: %s", json.dumps(verdict, default=float))
        # ---- learning curves
        rows, CV = [], {}
        fr_all = (*CURVE, 1.0)
        for arm, head, lab, priv_ in ROWS:
            for fk in FK:
                r = {"arm": arm, "family x convention": fk}
                for fr in fr_all:
                    d, _ = mean_of("oof", arm, fk) if fr == 1.0 else mean_of("curve", arm, fk, fr)
                    CV[arm, fk, fr] = gci(d, B["> 20 deg"])
                    r[f"{int(100 * fr)}%"] = TC.cell(CV[arm, fk, fr])
                r["rising 75 -> 100%"] = bool(CV[arm, fk, 1.0]["mean"] - CV[arm, fk, 0.75]["mean"] >= THRESH["rising"])
                rows.append(r)
        stats.write_table(rows, out / "learning_curve", note="gain > 20 deg vs the share of training logs; trees 2 repeats and neural heads 3 repeats at <100%")
        # ---- AUC: "a repair exists" (F19 x pc gain > 0)
        y = _D["Y", f0][:, :, 1:].max(-1) > 1e-9                                      # (S, n)
        lc = _D["lcode"]
        rng = np.random.default_rng(0)
        idx = [np.flatnonzero(lc == g) for g in range(lc.max() + 1)]
        boot = [np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))]) for _ in range(1000)]

        def auc_ci(sc):
            q = [_auc(sc[:, bi], y[:, bi]) for bi in boot]
            return f"{_auc(sc, y):.3f} [{np.quantile(q, 0.025):.3f}, {np.quantile(q, 0.975):.3f}]"
        f = TD.folds(0)
        Mz = np.load(OUT / "margins.npz")
        rows = [{"score": "base rate % of token-seeds", "AUC": f"{100 * y.mean():.1f}"},
                {"score": "- identity map margin (4 s), PRIV", "AUC": auc_ci(-Mz["M"][:, 0, :, 3])},
                {"score": "- identity own-edge margin (4 s)", "AUC": auc_ci(-Mz["C"][:, 0, :, 3])}]
        for lab, streams in (("ridge on the 0 / 1 label: E", ["ego", "plan"]), ("... E + map margins, PRIV", ["ego", "plan", "Mflat"]), ("... E + own-edge margins", ["ego", "plan", "Cflat"]),
                             ("... E + hidden state (32 PCs)", ["H", "ego", "plan"])):
            Xa, sc = xof(streams, 32, f0), np.zeros(y.shape)
            for j in range(TD.FOLDS):
                P = TD.ridge_fit(Xa[:, f != j].reshape(-1, Xa.shape[-1]), y[:, f != j].reshape(-1, 1).astype(float))
                sc[:, f == j] = P(Xa[:, f == j].reshape(-1, Xa.shape[-1]), 100.0).reshape(S, -1)
            rows.append({"score": lab, "AUC": auc_ci(sc)})
        for arm, head, lab, priv_ in ROWS:
            if arm == "E":
                continue
            sc, _ = mean_of("oof", arm, f0, key="sc")
            rows.append({"score": f"{arm} head: max predicted gain" if head != "R" else f"{arm} rule: - identity margin", "AUC": auc_ci(sc)})
        stats.write_table(rows, out / "auc_repair_exists", floatfmt=".1f", note="out-of-fold AUC of 'F19 x pc has a candidate with gain > 0' per (token, seed), > 20 deg; log-cluster bootstrap B 1000. "
                          "Decision 186 main arm E: 0.692 [0.660, 0.720].")
        # ---- contrasts against E
        rows = []
        for arm, head, lab, priv_ in ROWS[1:]:
            r = {"contrast": f"{arm} - E"}
            for fk in FK:
                dd = mean_of("oof", arm, fk)[0] - mean_of("oof", "E", fk)[0]
                r[fk] = TC.cell(gci(dd, B["> 20 deg"]))
            rows.append(r)
        stats.write_table(rows, out / "contrasts_vs_E", note="paired difference of out-of-fold gains, > 20 deg, same folds")
        # ---- what the privileged rule picks and a descriptive coverage reading
        Mf, Y = _D[f"M|{f0}"], _D["Y", f0]
        rep_ex = Y[:, :, 1:].max(-1) > 1e-9
        rows = [{"reading": "repair exists (F19 x pc), % of token-seeds", "value": 100 * rep_ex.mean()},
                {"reading": "... of which a candidate with 4 s map margin > 0 exists, %", "value": 100 * ((Mf[..., 1:, 3] > 0).any(-1) & rep_ex).sum() / rep_ex.sum()},
                {"reading": "identity 4 s map margin < 0 (would be DAC-risky), % of token-seeds", "value": 100 * (Mf[..., 0, 3] < 0).mean()},
                {"reading": "identity own-edge margin < 0, % of token-seeds", "value": 100 * (_D[f"C|{f0}"][..., 0, 3] < 0).mean()}]
        stats.write_table(rows, out / "descriptive", floatfmt=".1f")
        fig_report(figd, G, CV, ceil, verdict)
        run.summary.update(case=case, nonpriv_positive=",".join(n_pos))


def fig_report(FIGD, G, CV, ceil, vd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIGD.mkdir(parents=True, exist_ok=True)
    f0, f1 = FK
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[2.4, 1, 1]))
    names = [r[0] for r in ROWS]
    cols = [ps.PALETTE["black"] if r[3] == "" and r[0] == "E" else ps.PALETTE["vermillion"] if r[3] else ps.PALETTE["blue"] for r in ROWS]
    for ax, fk in ((axs[0], f0),):
        v = np.array([[G[x, fk, "> 20 deg"]["mean"], G[x, fk, "> 20 deg"]["mean"] - G[x, fk, "> 20 deg"]["lo"], G[x, fk, "> 20 deg"]["hi"] - G[x, fk, "> 20 deg"]["mean"]] for x in names])
        ax.bar(range(len(names)), v[:, 0], 0.7, yerr=v[:, 1:].T, color=cols, error_kw=dict(lw=0.6, capsize=1.2))
        ax.axhline(100 * ceil[fk].mean(), color=ps.BASELINE, ls="--", lw=0.8)
        ax.text(len(names) - 0.5, 100 * ceil[fk].mean() + 0.2, "ceiling", ha="right", va="bottom", fontsize=7, color=ps.BASELINE)
        ax.set_xticks(range(len(names)), names, fontsize=7)
        ax.set_ylabel("selector - SH30, out of fold, EPDMS x 100\n(F19 x pc, > 20 deg, 95% CI)")
        ps.bars(ax), ps.zero_line(ax)
        ax.text(0.02, 0.86, "red = PRIVILEGED (map margin); blue = model-side; black = control", transform=ax.transAxes, fontsize=6.5, va="top")
    best_p = max(("P1", "P2", "P3"), key=lambda k: vd["priv"][k]["gain"])
    best_n = max(("N1", "N2", "N3", "N4", "N5", "N6", "N7"), key=lambda k: vd["nonpriv"][k]["gain"])
    xs = (*CURVE, 1.0)
    for ax, fk in ((axs[1], f0), (axs[2], f1)):
        for arm, c in (("E", ps.PALETTE["black"]), (best_p, ps.PALETTE["vermillion"]), (best_n, ps.PALETTE["blue"])):
            v = np.array([[CV[arm, fk, x]["mean"], CV[arm, fk, x]["lo"], CV[arm, fk, x]["hi"]] for x in xs])
            ax.plot(xs, v[:, 0], color=c, marker="o", ms=2.5, label=arm + (" (PRIV)" if arm == best_p else ""))
            ax.fill_between(xs, v[:, 1], v[:, 2], color=c, alpha=0.12, lw=0)
        ax.set_xticks(xs, [f"{int(100 * x)}%" for x in xs], fontsize=7)
        ax.set_xlabel("share of training logs"), ax.set_title(fk, fontsize=8), ps.zero_line(ax), ax.legend(fontsize=6)
    fig.savefig(FIGD / "selector_inputs.png", dpi=300)
    plt.close(fig)


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
