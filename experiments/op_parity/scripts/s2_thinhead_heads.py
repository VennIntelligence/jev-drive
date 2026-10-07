"""s2-thinhead, the frozen-feature arms (plans/2026-10-08-s2-thinhead-prereg.md): features x supervision x head, out-of-fold by sequence.

  fit     (jevdrive env; CPU pool for the ridge heads, one small GPU for the MLP heads) -> $DATA_DIR/runs/op_parity/s2_thinhead/heads.npz + heads.json
  report  (jevdrive env, CPU) tables with paired bootstraps over sequences -> results/s2_thinhead/

Inputs: s2_thinhead.py prep (data.npz). Streams: vision (C = Cinque temporal, Q1 = Qwen3 single frame, QV = Qwen3 4 frames x 0.2 s, QL = Qwen3 4 frames
x 0.5 s; whitened PCs), ego (20), plan (15). Heads: L = multi-output ridge on the 20 candidates' RFS gain over keep / follow (a: multinomial logistic),
M = one linear encoder per stream -> GELU -> dropout -> 20 (listwise loss). Supervision: a = hindsight class on r2-train (selected on r2-dev),
b = rater scores by sequence k-fold, c = a then b.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import s2_thinhead as T  # noqa: E402

KEEP = T.G.KEEP * 4
LAMS = (3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e6)
KS, FOLDS, REPS, INNER, S = (8, 32, 128), 5, 10, 4, 2
N_PERM, PERM_REPS, CURVE = 200, 3, (0.25, 0.5, 0.75)
ARMS = {"E0": ["ego"], "E": ["ego", "plan"], "C+E": ["C", "ego", "plan"], "Q1+E": ["Q1", "ego", "plan"], "QV+E": ["QV", "ego", "plan"],
        "QL+E": ["QL", "ego", "plan"], "Q1+C+E": ["Q1", "C", "ego", "plan"], "QV+C+E": ["QV", "C", "ego", "plan"], "QL+C+E": ["QL", "C", "ego", "plan"]}
SENS = {"Q1v+E": ["Q1v", "ego", "plan"], "QVv+E": ["QVv", "ego", "plan"]} | {f"Q1@{k}+E": [f"Q1@{k}", "ego", "plan"] for k in T.Q1_LAYERS}
PRIMARY = ("C+E", "Q1+E", "QV+E", "QL+E")
TRAINABLE = ("E0", "E", "C+E", "Q1+E", "QV+E", "Q1+C+E", "QV+C+E")           # arms with train rows (supervision a / c)
NONVIS = ("ego", "plan", "alog")
MK, MH, TAU, WDS, WDS_C = 64, 32, 0.5, (1e-3, 1e-2, 1e-1), (1e-2, 1e-1, 1.0)
_D = {}                                                                     # arrays shared with forked workers


def X_val(streams, k, half=False):
    """(S, n, d) inputs of the rater frames, vision streams first; `half`: k / 2 PCs per vision stream when there are two."""
    nv = sum(s not in NONVIS for s in streams)
    kk = k // 2 if (half and nv > 1) else k
    cols = [_D[f"va_{s}"] if s in NONVIS else _D[f"va_{s}"][:, :kk] for s in streams]
    return np.stack([np.concatenate([c[s] if c.ndim == 3 else c for c in cols], 1) for s in range(S)]).astype(np.float64), nv * kk


def X_train(streams, k, half=False):
    nv = sum(s not in NONVIS for s in streams)
    kk = k // 2 if (half and nv > 1) else k
    return np.concatenate([_D[f"tr_{s}"] if s in NONVIS else _D[f"tr_{s}"][:, :kk] for s in streams], 1).astype(np.float32)


def ridge_fit(X, Y):
    mu, ym = X.mean(0), Y.mean(0)
    Xc = X - mu
    w, V = np.linalg.eigh(Xc.T @ Xc)
    A = V.T @ (Xc.T @ (Y - ym))
    return lambda Xq, lam: (Xq - mu) @ (V @ (A / (np.maximum(w, 0)[:, None] + lam))) + ym


def gains(pred, idx, zero_keep=True):
    """Predicted scores (S, m, 20) of frames idx -> realised RFS gain over WP2 of the selected candidate, seed mean (m,), and the picks (S, m)."""
    J, out, picks = _D["va_J"], 0.0, []
    for s in range(S):
        p = pred[s].copy()
        if zero_keep:
            p[:, KEEP] = 0.0
        pk = T.first(p.T)
        picks.append(pk)
        out = out + J[s][pk, idx] - J[s][KEEP, idx]
    return out / S, np.stack(picks)


def folds(rep, k=FOLDS, salt=0):
    sc = _D["scode"]
    return (np.random.default_rng(salt + rep).permutation(sc.max() + 1) % k)[sc]


def select_fit(Xk, Y, tr, inner):
    """(k, lam) by inner CV over the frames tr (largest realised gain; ties to the smaller k and the larger lam), then the fit on tr."""
    best, arg = -np.inf, None
    for k, X in Xk.items():
        acc, d = np.zeros(len(LAMS)), X.shape[2]
        for g in range(INNER):
            a_, b_ = tr[inner[tr] != g], tr[inner[tr] == g]
            P = ridge_fit(X[:, a_].reshape(-1, d), Y[:, a_].reshape(-1, 20))
            for li, lam in enumerate(LAMS):
                acc[li] += gains(P(X[:, b_].reshape(-1, d), lam).reshape(S, len(b_), 20), b_)[0].sum()
        for li in reversed(range(len(LAMS))):
            if acc[li] > best + 1e-9:
                best, arg = acc[li], (k, LAMS[li])
    X = Xk[arg[0]]
    return arg, ridge_fit(X[:, tr].reshape(-1, X.shape[2]), Y[:, tr].reshape(-1, 20))


def unit(a):
    """One ridge unit -> (tag, per-frame gain (n,), extras). kinds: oof (one repeat), ins (fit on all), perm (PERM_REPS repeats, inputs permuted), const."""
    kind, arm, streams, rep, opt = a
    Y = (_D["va_J"] - _D["va_J"][:, KEEP:KEEP + 1]).transpose(0, 2, 1)             # (S, n, 20)
    n, sc = Y.shape[1], _D["scode"]
    if kind == "const":
        d, f = np.zeros(n), folds(rep)
        for j in range(FOLDS):
            te = np.flatnonzero(f == j)
            d[te] = gains(np.broadcast_to(Y[:, f != j].mean((0, 1)), (S, len(te), 20)), te)[0]
        return (kind, arm, rep), d, None
    Xk = {}
    for k in ((opt.get("k"),) if opt.get("k") else KS):
        X, nvd = X_val(streams, k, opt.get("half", False))
        if kind == "perm":
            pi = np.random.default_rng(5000 + rep).permutation(n)
            X = X[:, pi] if opt["what"] == "all" else np.concatenate([X[:, pi, :nvd], X[:, :, nvd:]], 2)
        Xk[k] = X
        if not any(s not in NONVIS for s in streams):
            break                                                               # no vision stream: k is irrelevant
    if kind == "ins":
        arg, P = select_fit(Xk, Y, np.arange(n), folds(0, INNER, 900))
        X = Xk[arg[0]]
        d, pk = gains(P(X.reshape(-1, X.shape[2]), arg[1]).reshape(S, n, 20), np.arange(n))
        return (kind, arm, rep), d, {"hp": [arg], "picks": pk}
    d, hp, picks = np.zeros(n), [], np.zeros((S, n), int)
    reps = range(PERM_REPS) if kind == "perm" else (rep,)
    for r in reps:
        f = folds(r)
        for j in range(FOLDS):
            te, tr = np.flatnonzero(f == j), np.flatnonzero(f != j)
            if opt.get("frac", 1.0) < 1.0:
                sub = np.random.default_rng(7000 + r * 10 + j).permutation(sc.max() + 1)[: int(opt["frac"] * (sc.max() + 1))]
                tr = tr[np.isin(sc[tr], sub)]
            arg, P = select_fit(Xk, Y, tr, folds(r * 10 + j, INNER, 1000))
            X = Xk[arg[0]]
            g, pk = gains(P(X[:, te].reshape(-1, X.shape[2]), arg[1]).reshape(S, len(te), 20), te)
            d[te] += g / len(reps)
            picks[:, te] = pk
            hp.append(arg)
    return (kind, arm, rep, opt.get("what", ""), opt.get("frac", 1.0)), d.astype(np.float32), {"hp": hp, "picks": picks}


# ---------------------------------------------------------------- torch heads
def softmax_fit(Xtr, ytr, Xdv, ydv, wds, steps=400, lr=0.05):
    """Multinomial logistic, full batch, one model per weight decay -> (W (G, d + 1, 20), dev NLL (G,), dev accuracy (G,))."""
    import torch
    Xt, yt, Xd, yd = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xdv), torch.from_numpy(ydv)
    G_, d = len(wds), Xtr.shape[1]
    W, b = torch.zeros(G_, d, 20, requires_grad=True), torch.zeros(G_, 1, 20, requires_grad=True)
    wd = torch.tensor(wds)[:, None, None]
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lp = torch.log_softmax(torch.einsum("nd,gdc->gnc", Xt, W) + b, -1)
        (-(lp[:, torch.arange(len(yt)), yt]).mean(1).sum() + (wd * W * W).sum()).backward()
        opt.step()
    with torch.no_grad():
        lp = torch.log_softmax(torch.einsum("nd,gdc->gnc", Xd, W) + b, -1)
        nll = -(lp[:, torch.arange(len(yd)), yd]).mean(1)
        acc = (lp.argmax(-1) == yd).float().mean(1)
    return torch.cat([W, b], 1).detach().numpy(), nll.numpy(), acc.numpy()


class BMLP:
    """M independent heads trained in parallel: one linear encoder per stream (-> MH) -> GELU -> dropout 0.5 -> 20."""

    def __init__(self, dims, M, dev, init=None):
        import torch
        self.t, self.dims, self.M = torch, dims, M
        g = torch.Generator().manual_seed(0)
        if init is None:
            p = [torch.randn(M, d, MH, generator=g) / np.sqrt(d) for d in dims] + [torch.zeros(M, 1, MH) for _ in dims]
            p += [torch.randn(M, MH * len(dims), 20, generator=g) * 0.01, torch.zeros(M, 1, 20)]
        else:
            p = [x.expand(M, *x.shape[1:]).clone() for x in init]
        self.p = [x.to(dev).requires_grad_(True) for x in p]
        self.p0 = [x.detach().clone() for x in self.p]

    def __call__(self, xs, train=False):
        t, ns = self.t, len(self.dims)
        h = t.cat([t.einsum("nd,mdh->mnh", x, self.p[i]) + self.p[ns + i] for i, x in enumerate(xs)], -1)
        h = t.nn.functional.dropout(t.nn.functional.gelu(h), 0.5, train)
        return t.einsum("mnh,mhc->mnc", h, self.p[-2]) + self.p[-1]

    def reg(self, wd, to_init=False):
        return sum((wd * ((x - (x0 if to_init else 0)) ** 2).flatten(1).sum(1)).sum() for x, x0 in zip(self.p, self.p0))


def split_streams(X, streams, k, half=False):
    """(.., d) concatenated inputs -> list of per-stream arrays (the widths of X_val / X_train)."""
    nv = sum(s not in NONVIS for s in streams)
    kk = k // 2 if (half and nv > 1) else k
    w = [{"ego": 20, "plan": 15, "alog": 20}.get(s, kk) for s in streams]
    o = np.cumsum([0] + w)
    return [X[..., o[i]:o[i + 1]] for i in range(len(w))], w


def mlp_arm(arm, streams, dev, run):
    """Supervision a (r2-train, selected on r2-dev), b and c (sequence k-fold) for one feature arm with the M head -> {sup: gain (n,)}, info."""
    import torch
    half = True
    Xv, _ = X_val(streams, MK, half)
    n = Xv.shape[1]
    xs_v, dims = split_streams(torch.from_numpy(Xv.reshape(S * n, -1)).float().to(dev), streams, MK, half)
    J = torch.from_numpy(_D["va_J"]).float().to(dev)                              # (S, 20, n)
    tgt = torch.softmax(J.permute(0, 2, 1).reshape(S * n, 20) / TAU, -1)
    out, info, init = {}, {}, None
    if arm in TRAINABLE:                                                        # ---- a
        Xt = torch.from_numpy(X_train(streams, MK, half)).to(dev)
        y, dv_ = torch.from_numpy(_D["tr_lab"]).to(dev), torch.from_numpy(_D["tr_dev"]).to(dev)
        xt, _ = split_streams(Xt, streams, MK, half)
        m = BMLP(dims, len(WDS), dev)
        wd = torch.tensor(WDS, device=dev)
        opt = torch.optim.Adam(m.p, lr=3e-3)
        tr_i = torch.nonzero(~dv_).squeeze(1)
        g = torch.Generator(device=dev).manual_seed(0)
        for step in range(600):
            b = tr_i[torch.randint(len(tr_i), (4096,), device=dev, generator=g)]
            opt.zero_grad()
            lp = torch.log_softmax(m([x[b] for x in xt], True), -1)
            (-(lp[:, torch.arange(len(b), device=dev), y[b]]).mean(1).sum() + m.reg(wd)).backward()
            opt.step()
        with torch.no_grad():
            lp = torch.log_softmax(m([x[dv_] for x in xt]), -1)
            nll = -(lp[:, torch.arange(int(dv_.sum()), device=dev), y[dv_]]).mean(1)
            gi = int(nll.argmin())
            lv = m(xs_v)[gi].reshape(S, n, 20).cpu().numpy()
        out["a"] = gains(lv, np.arange(n), zero_keep=False)[0]
        info["a"] = {"wd": WDS[gi], "dev_nll": float(nll[gi]), "dev_acc": float((lp[gi].argmax(-1) == y[dv_]).float().mean())}
        init = [x[gi:gi + 1].detach().cpu() for x in m.p]
    # ---- b (random init) and c (a's weights, L2-SP)
    per = 1 + INNER
    for sup, wds, steps, lr in (("b", WDS, 300, 3e-3), ("c", WDS_C, 150, 1e-3)):
        if sup == "c" and init is None:
            continue
        Gn = len(wds)
        M = REPS * FOLDS * Gn * per
        mask, wd, meta = torch.zeros(M, n, device=dev), torch.zeros(M, device=dev), []
        for r in range(REPS):
            f = folds(r)
            for j in range(FOLDS):
                inner = folds(r * 10 + j, INNER, 1000)
                for gi, w in enumerate(wds):
                    for q in range(per):
                        i = ((r * FOLDS + j) * Gn + gi) * per + q
                        tr = (f != j) & ((inner != q - 1) if q else True)
                        mask[i] = torch.from_numpy(tr.astype(np.float32)).to(dev)
                        wd[i] = w
                        meta.append((r, j, gi, q))
        m = BMLP(dims, M, dev, init if sup == "c" else None)
        opt = torch.optim.Adam(m.p, lr=lr)
        mk = mask.repeat(1, S)
        for step in range(steps):
            opt.zero_grad()
            lp = torch.log_softmax(m(xs_v, True), -1)
            kl = (tgt * (torch.log(tgt + 1e-12) - lp)).sum(-1)
            (((kl * mk).sum(1) / mk.sum(1)).sum() + m.reg(wd, sup == "c")).backward()
            opt.step()
        with torch.no_grad():
            pk = m(xs_v).reshape(M, S, n, 20).argmax(-1)                          # plain argmax of the logits
            real = torch.stack([J[s].T.gather(1, pk[:, s].T).T - J[s, KEEP] for s in range(S)]).mean(0).cpu().numpy()   # (M, n) realised gain
        d, chosen = np.zeros(n), []
        for r in range(REPS):
            f = folds(r)
            for j in range(FOLDS):
                inner = folds(r * 10 + j, INNER, 1000)
                sc = []
                for gi in range(Gn):
                    base = ((r * FOLDS + j) * Gn + gi) * per
                    sc.append(sum(real[base + q][(f != j) & (inner == q - 1)].sum() for q in range(1, per)))
                gi = max(range(Gn), key=lambda g_: (sc[g_], g_))                # ties to the larger weight decay
                chosen.append(wds[gi])
                d[f == j] += real[((r * FOLDS + j) * Gn + gi) * per][f == j] / REPS
        out[sup] = d
        info[sup] = {"wd counts": {str(w): int(np.sum(np.array(chosen) == w)) for w in wds},
                     "train-fold gain (in-sample)": float(np.mean([real[i][mask[i].cpu().numpy() > 0].mean() for i in range(0, M, per)]))}
        del m, opt
        torch.cuda.empty_cache()
    run.info("M head %s: %s | %s", arm, {k: round(float(v.mean()), 3) for k, v in out.items()}, info)
    return out, info


def cmd_fit(a):
    import multiprocessing as mp
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-heads", seed=0, config=vars(a) | dict(lams=LAMS, ks=KS, folds=FOLDS, reps=REPS, inner=INNER, n_perm=N_PERM)) as run:
        for s in ("wod/val", "wod/r2-train", "wod/r2-dev"):
            run.use_split(splits.load(s))
        z = np.load(T.RUNS / "data.npz")
        _D.update({k: z[k] for k in z.files})
        import pandas as pd
        _D["scode"] = pd.factorize(pd.Series(_D["va_seq"].astype(str)))[0]
        n = len(_D["va_names"])
        arms = {k: v for k, v in (ARMS | SENS).items() if all(f"va_{s}" in _D for s in v)}
        run.info("arms: %s", list(arms))
        if a.part == "mlp":
            import torch
            dev, minfo, out = torch.device("cuda"), {}, {}
            for arm in ARMS:
                if arm in arms:
                    o, minfo[arm] = mlp_arm(arm, arms[arm], dev, run)
                    out |= {f"{arm}|{sup}|M": d for sup, d in o.items()}
            np.savez(T.RUNS / "heads_M.npz", **{k: np.asarray(v, np.float32) for k, v in out.items()})
            (T.RUNS / "heads_M.json").write_text(json.dumps(minfo, indent=1, default=float))
            run.summary.update(arms=len(minfo))
            return
        units = [("const", "const", None, r, {}) for r in range(REPS)]
        for arm, st in arms.items():
            units += [("oof", arm, st, r, {}) for r in range(REPS)] + [("ins", arm, st, 0, {})]
        for arm in ("Q1+C+E", "QV+C+E"):                                         # equal input width: k / 2 PCs per vision stream
            units += [("oof", arm + " (k/2 each)", arms[arm], r, {"half": True}) for r in range(REPS)]
        for arm in ("E", "C+E", "Q1+E", "QV+E", "Q1+C+E"):                       # learning curve
            units += [("oof", arm, arms[arm], r, {"frac": fr}) for r in range(REPS // 2) for fr in CURVE]
        n_perm = a.perms
        for arm in ("E",) + tuple(p for p in PRIMARY if p in arms):
            units += [("perm", arm, arms[arm], i, {"what": "all"}) for i in range(n_perm)]
            if arm != "E":
                units += [("perm", arm, arms[arm], i, {"what": "vis"}) for i in range(n_perm)]
        t0 = time.time()
        res = par.pmap(unit, units, workers=min(n_cpus(), a.workers), run=run, desc="ridge units", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        run.info("ridge units: %d in %.0f s", len(units), time.time() - t0)
        out, hp, picks, perm = {}, {}, {}, {}
        for tag, d, ex in res.values:
            kind, arm = tag[0], tag[1]
            if kind == "const":
                out["const|b|L"] = out.get("const|b|L", 0) + d / REPS
            elif kind == "ins":
                out[f"{arm}|b|L|ins"] = d
                hp[f"{arm}|ins"] = ex["hp"]
            elif kind == "perm":
                perm.setdefault(f"{arm}|{tag[3]}", []).append(d)
            elif tag[4] < 1.0:
                out[f"{arm}|b|L|frac{tag[4]}"] = out.get(f"{arm}|b|L|frac{tag[4]}", 0) + d / (REPS // 2)
            else:
                out[f"{arm}|b|L"] = out.get(f"{arm}|b|L", 0) + d / REPS
                hp.setdefault(arm, []).extend(ex["hp"])
                if tag[2] == 0:
                    picks[f"{arm}|b|L"] = ex["picks"]
                if tag[2] < PERM_REPS:
                    out[f"{arm}|b|L|rep3"] = out.get(f"{arm}|b|L|rep3", 0) + d / PERM_REPS
        # ---- supervision a (L: multinomial logistic on r2-train, selected on r2-dev) and c (a's logits as one more stream of the ridge)
        import torch
        torch.set_num_threads(min(n_cpus(), 32))
        dv_, lab = _D["tr_dev"].astype(bool), _D["tr_lab"].astype(np.int64)
        prior = np.bincount(lab[~dv_], minlength=20) + 1.0
        prior /= prior.sum()
        ainfo = {"prior": {"dev_nll": float(-np.log(prior[lab[dv_]]).mean()), "dev_acc": float((lab[dv_] == prior.argmax()).mean()),
                           "val_acc": float(np.mean([(_D["va_hlab"][s] == prior.argmax()).mean() for s in range(S)]))}}
        for arm in TRAINABLE:
            st, best = arms[arm], None
            for k in (KS if any(s not in NONVIS for s in st) else KS[:1]):
                Xt = X_train(st, k)
                W, nll, acc = softmax_fit(Xt[~dv_], lab[~dv_], Xt[dv_], lab[dv_], (1e-5, 1e-4, 1e-3, 1e-2))
                gi = int(nll.argmin())
                if best is None or nll[gi] < best[0]:
                    best = (float(nll[gi]), k, W[gi], float(acc[gi]), gi)
            nll, k, W, acc, gi = best
            Xv, _ = X_val(st, k)
            lv = Xv @ W[:-1] + W[-1]
            d, pk = gains(lv, np.arange(n), zero_keep=False)
            out[f"{arm}|a|L"], picks[f"{arm}|a|L"] = d, pk
            ainfo[arm] = {"k": k, "wd_index": gi, "dev_nll": nll, "dev_acc": acc, "val_acc": float(np.mean([(lv[s].argmax(1) == _D["va_hlab"][s]).mean() for s in range(S)])),
                          "val_keep_share": float((pk == KEEP).mean())}
            lt = X_train(st, k)[~dv_] @ W[:-1] + W[-1]
            lt, lv = lt - lt[:, KEEP:KEEP + 1], lv - lv[..., KEEP:KEEP + 1]
            mu, sd = lt.mean(0), lt.std(0) + 1e-6
            _D["va_alog"] = ((lv - mu) / sd).astype(np.float32)
            acc_d, hps = np.zeros(n), []
            for r in range(REPS):
                _, d, ex = unit(("oof", arm, st + ["alog"], r, {}))
                acc_d += d / REPS
                hps += ex["hp"]
            out[f"{arm}|c|L"], hp[f"{arm}|c"] = acc_d, hps
            out[f"{arm}|c|L|ins"] = unit(("ins", arm, st + ["alog"], 0, {}))[1]
            run.info("L head %s: a %+.3f (dev nll %.3f acc %.3f, prior %.3f / %.3f), b %+.3f, c %+.3f", arm, out[f"{arm}|a|L"].mean(), nll, acc,
                     ainfo["prior"]["dev_nll"], ainfo["prior"]["dev_acc"], out[f"{arm}|b|L"].mean(), acc_d.mean())
        np.savez(T.RUNS / "heads.npz", **{k: np.asarray(v, np.float32) for k, v in out.items()}, **{f"perm:{k}": np.stack(v) for k, v in perm.items()},
                 **{f"picks:{k}": v for k, v in picks.items()})
        (T.RUNS / "heads.json").write_text(json.dumps({"hp": {k: [list(map(float, x)) for x in v] for k, v in hp.items()}, "a": ainfo, "arms": arms}, indent=1,
                                                      default=float))
        run.summary.update(arms=len(arms), units=len(units), **{k: float(np.mean(out[f"{k}|b|L"])) for k in arms})


# ---------------------------------------------------------------- report
def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-report", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        Bd = T.board()
        C = Bd.C
        z, info = dict(np.load(T.RUNS / "heads.npz")), json.loads((T.RUNS / "heads.json").read_text())
        if (T.RUNS / "heads_M.npz").exists():
            z |= dict(np.load(T.RUNS / "heads_M.npz"))
            info["M"] = json.loads((T.RUNS / "heads_M.json").read_text())
        z = SimpleNamespace(files=list(z), **{"d": z})
        z = type("Z", (), {"files": z.files, "__getitem__": lambda self, k, _d=z.d: _d[k]})()
        assert (np.load(T.RUNS / "data.npz")["va_names"].astype(str) == Bd.names).all()
        orc = Bd.best - Bd.base
        main = ("all", "stopped", "moving", "turn")
        keys = [k for k in z.files if ":" not in k]
        rows, summ = [], []
        for k in keys:
            p = k.split("|")
            arm, sup, head, var = p[0], p[1], p[2], p[3] if len(p) > 3 else ""
            d = z[k].astype(np.float64)
            for sn, m in Bd.st.items():
                c, cb_, o = T.ci(C, d, m), T.ci(C, d, m, (0.625, 99.375)), C.cm(orc, m)
                rows.append({"features": arm, "supervision": sup, "head": head, "variant": var or "oof", "stratum": sn, "n": int(m.sum()), "d": c[0], "lo": c[1], "hi": c[2],
                             "lo 98.75": cb_[1], "hi 98.75": cb_[2], "oracle d": o, "share of oracle": c[0] / o if o > 1e-9 else np.nan})
        stats.write_table(rows, OUT / "arms")
        df = pd.DataFrame(rows)
        f3 = lambda r: f"{r.d:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"  # noqa: E731
        for (arm, sup, head), g in df[df.variant.isin(["oof", "ins"]) & df.stratum.isin(main)].groupby(["features", "supervision", "head"], sort=False):
            r = {"features": arm, "supervision": sup, "head": head}
            for sn in main:
                x = g[(g.stratum == sn) & (g.variant == "oof")]
                if len(x):
                    r[sn] = f3(x.iloc[0])
            x = g[(g.stratum == "all") & (g.variant == "oof")]
            if len(x):
                r["share of oracle (all)"] = float(x.iloc[0]["share of oracle"])
                r["98.75 % CI (all)"] = f"[{x.iloc[0]['lo 98.75']:+.3f}, {x.iloc[0]['hi 98.75']:+.3f}]"
            x = g[(g.stratum == "all") & (g.variant == "ins")]
            r["in-sample (all)"] = float(x.iloc[0].d) if len(x) else np.nan
            summ.append(r)
        stats.write_table(summ, OUT / "q1")
        run.info("Q1:\n%s", pd.DataFrame(summ).to_string())
        # ---- contrasts (paired, same frames)
        con = []

        def contrast(nm, ka, kb):
            if ka in z.files and kb in z.files:
                for sn in main:
                    c = T.ci(C, z[ka].astype(np.float64) - z[kb], Bd.st[sn])
                    con.append({"contrast": nm, "a": ka, "b": kb, "stratum": sn, "d": c[0], "lo": c[1], "hi": c[2]})
        for sup, head in (("b", "L"), ("c", "L"), ("a", "L"), ("b", "M"), ("c", "M"), ("a", "M")):
            sfx = f"|{sup}|{head}"
            for v in ("C", "Q1", "QV", "QL", "Q1+C", "QV+C", "QL+C"):
                contrast(f"vision over the floor: {v}+E - E", f"{v}+E{sfx}", f"E{sfx}")
            for q in ("Q1", "QV", "QL"):
                contrast(f"Qwen - Cinque: {q}+E - C+E", f"{q}+E{sfx}", f"C+E{sfx}")
                contrast(f"Qwen on top of Cinque: {q}+C+E - C+E", f"{q}+C+E{sfx}", f"C+E{sfx}")
            contrast("multi-frame - single-frame: QV+E - Q1+E", f"QV+E{sfx}", f"Q1+E{sfx}")
            contrast("multi-frame (0.5 s) - single-frame: QL+E - Q1+E", f"QL+E{sfx}", f"Q1+E{sfx}")
            contrast("multi-frame 0.5 s - 0.2 s spacing: QL+E - QV+E", f"QL+E{sfx}", f"QV+E{sfx}")
            contrast("val-whitened: QL+E - QVv+E", f"QL+E{sfx}", f"QVv+E{sfx}")
            contrast("val-whitened: QL+E - Q1v+E", f"QL+E{sfx}", f"Q1v+E{sfx}")
            contrast("plan stream: E - E0", f"E{sfx}", f"E0{sfx}")
            contrast("floor - constant policy: E - const", f"E{sfx}", "const|b|L")
        for arm in info["arms"]:
            for head in ("L", "M"):
                contrast(f"c - b: {arm} ({head})", f"{arm}|c|{head}", f"{arm}|b|{head}")
        stats.write_table(con, OUT / "contrasts")
        cd = pd.DataFrame(con)
        run.info("contrasts (all):\n%s", cd[cd.stratum == "all"].drop(columns=["a", "b", "stratum"]).to_string(float_format=lambda v: f"{v:+.3f}"))
        # ---- permutation control (ridge, b; PERM_REPS repeats on both sides)
        prow = []
        for k in [k for k in z.files if k.startswith("perm:")]:
            arm, what = k[5:].split("|")
            null = np.array([C.cm(d.astype(np.float64)) for d in z[k]])
            act = C.cm(z[f"{arm}|b|L|rep3"].astype(np.float64))
            prow.append({"features": arm, "permuted": what, "n perm": len(null), "actual (3 repeats)": act, "null mean": null.mean(), "null lo": np.percentile(null, 2.5),
                         "null hi": np.percentile(null, 97.5), "p (null >= actual)": float((1 + (null >= act).sum()) / (1 + len(null)))})
        stats.write_table(prow, OUT / "permutation")
        run.info("permutation:\n%s", pd.DataFrame(prow).to_string())
        # ---- supervision a diagnostics, hyperparameters, learning curve
        arow = [{"features": k} | v for k, v in info["a"].items()]
        stats.write_table(arow, OUT / "hindsight_fit")
        hrow = [{"arm": k, "n fits": len(v), "k (median)": float(np.median([x[0] for x in v])), "lambda (median)": float(np.median([x[1] for x in v])),
                 "share at lambda 1e6 (constant policy)": float(np.mean([x[1] >= 1e6 for x in v]))} for k, v in info["hp"].items()]
        stats.write_table(hrow, OUT / "hyperparameters", floatfmt=".3g")
        crow = []
        for k in keys:
            p = k.split("|")
            if len(p) > 3 and p[3].startswith("frac"):
                c = T.ci(C, z[k].astype(np.float64))
                crow.append({"features": p[0], "training fraction": float(p[3][4:]), "d": c[0], "lo": c[1], "hi": c[2]})
        stats.write_table(sorted(crow, key=lambda r: (r["features"], r["training fraction"])), OUT / "learning_curve")
        pd.DataFrame({"name": Bd.names, "cluster": Bd.cluster, "v0": Bd.v0, "rfs_WP2": Bd.base, "rfs_oracle": Bd.best}
                     | {f"d {k}": z[k] for k in keys if k.count("|") == 2}).to_csv(OUT / "frames.csv", index=False, float_format="%.4f")
        (OUT / "heads_info.json").write_text(json.dumps({"a": info["a"], "M": info.get("M", {})}, indent=1))
        # ---- verdict
        g = lambda k: df[(df.features == k.split("|")[0]) & (df.supervision == "b") & (df["head"] == "L") & (df.variant == "oof")].set_index("stratum")  # noqa: E731
        pm = {r["features"]: r for r in prow if r["permuted"] == "all"}
        ver = {}
        for arm in PRIMARY:
            if f"{arm}|b|L" not in z.files:
                continue
            x = g(arm)
            ok, mov = bool(x.loc["all", "lo"] > 0), bool(x.loc["moving", "hi"] >= 0 and x.loc["moving", "d"] >= -0.05)
            ver[arm] = {"d all": [x.loc["all", "d"], x.loc["all", "lo"], x.loc["all", "hi"]], "98.75 % CI": [x.loc["all", "lo 98.75"], x.loc["all", "hi 98.75"]],
                        "d moving": [x.loc["moving", "d"], x.loc["moving", "lo"], x.loc["moving", "hi"]], "CI excludes 0": ok, "moving ok": mov,
                        "beyond the permutation null": bool(arm in pm and pm[arm]["actual (3 repeats)"] > pm[arm]["null hi"]),
                        "delivers": bool(ok and mov and x.loc["all", "lo 98.75"] > 0 and arm in pm and pm[arm]["actual (3 repeats)"] > pm[arm]["null hi"])}
        (OUT / "verdict.json").write_text(json.dumps(ver, indent=1, default=float))
        run.info("verdict: %s", json.dumps(ver, default=float))
        run.summary.update(delivers={k: v["delivers"] for k, v in ver.items()})


OUT = T.OUT


# ---------------------------------------------------------------- Q2: candidate set, conservative rule, shipped as System 1
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0)
_T = {}                                                                     # task -> dict(J, keep, pri, X {k: (S, n, d)}, margins, Xa, Ja)


def gains2(pred, J, keep, pri, idx, margin=0.0):
    out, changed = 0.0, 0.0
    for s in range(len(J)):
        p = pred[s].copy()
        p[:, keep] = margin
        pk = T.first(p.T, pri)
        out, changed = out + J[s][pk, idx] - J[s][keep, idx], changed + (pk != keep)
    return out / len(J), changed / len(J)


def unit2(a):
    """Ridge on the candidates' gain over keep, (k, lambda, margin) by inner CV, one repeat of the sequence k-fold -> gain (n,), share changed, margins."""
    task, rep = a
    t = _T[task]
    J, keep, pri, Xk, mg = t["J"], t["keep"], t["pri"], t["X"], t["margins"]
    Y = (J - J[:, keep:keep + 1]).transpose(0, 2, 1)
    Sn, n, K = Y.shape
    Xa, Ja = t.get("Xa", Xk), t.get("Ja", J)
    f, d, ch, hp = folds(rep), np.zeros(n), np.zeros(n), []
    for j in range(FOLDS):
        te, tr = np.flatnonzero(f == j), np.flatnonzero(f != j)
        inner = folds(rep * 10 + j, INNER, 1000)
        best, arg = -np.inf, None
        for k, X in Xk.items():
            acc, dd = np.zeros((len(LAMS), len(mg))), X.shape[2]
            for g in range(INNER):
                a_, b_ = tr[inner[tr] != g], tr[inner[tr] == g]
                P = ridge_fit(X[:, a_].reshape(-1, dd), Y[:, a_].reshape(-1, K))
                for li, lam in enumerate(LAMS):
                    pr = P(X[:, b_].reshape(-1, dd), lam).reshape(Sn, len(b_), K)
                    for mi, m in enumerate(mg):
                        acc[li, mi] += gains2(pr, J, keep, pri, b_, m)[0].sum()
            for li in reversed(range(len(LAMS))):
                for mi in reversed(range(len(mg))):
                    if acc[li, mi] > best + 1e-9:
                        best, arg = acc[li, mi], (k, LAMS[li], mg[mi])
        k, lam, m = arg
        X, XA = Xk[k], Xa[k]
        P = ridge_fit(X[:, tr].reshape(-1, X.shape[2]), Y[:, tr].reshape(-1, K))
        d[te], ch[te] = gains2(P(XA[:, te].reshape(-1, XA.shape[2]), lam).reshape(len(XA), len(te), K), Ja, keep, pri, te, m)
        hp.append(arg)
    return task, d, ch, hp


def cmd_q2(a):
    import multiprocessing as mp
    import pandas as pd
    from jevdrive import par, stats
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-q2", seed=0, config=vars(a) | dict(margins=MARGINS, reps=REPS)) as run:
        run.use_split(splits.load("wod/val"))
        Bd = T.board()
        C, n, cb = Bd.C, Bd.n, Bd.cb
        z = np.load(T.RUNS / "data.npz")
        _D.update({k: z[k] for k in z.files})
        _D["scode"] = pd.factorize(pd.Series(_D["va_seq"].astype(str)))[0]
        ship = C.preds("shipped")
        Fs = T.G.factored(ship, Bd.v0, 3, cb)
        Js = np.stack([C.rfs(np.ascontiguousarray(Fs[:, m])) for m in range(20)])[None]
        F30 = [T.G.factored(p, Bd.v0, 5, cb) for p in Bd.P]
        J30 = np.stack([np.stack([C.rfs(np.ascontiguousarray(f[:, m])) for m in range(30)]) for f in F30])
        raw = np.stack([T.plan_desc(p, Bd.v0, cb) for p in Bd.P + [ship]])                    # plan stream standardised on the val frames (no labels)
        pd_ = (raw - raw[0].mean(0)) / (raw[0].std(0) + 1e-6)
        ego = _D["va_ego"].astype(np.float64)

        def X(desc, vis=None):
            return {k: np.stack([np.concatenate(([_D[f"va_{vis}"][:, :k]] if vis else []) + [ego, d_], 1) for d_ in desc]) for k in (KS if vis else KS[:1])}
        J20, pri5 = Bd.J, np.array([2, 1, 3, 0, 4])
        sets = {"F20": (J20, KEEP, T.PRI), "F30": (J30, T.G.KEEP * 6, np.array([p_ * 6 + s_ for p_ in pri5 for s_ in range(6)])),
                "speed only (4)": (J20[:, KEEP:KEEP + 4], 0, np.arange(4)), "path only (5)": (J20[:, ::4], 2, pri5)}
        base = {"WP2": Bd.base, "shipped": Js[0, KEEP]}
        tasks = {}
        for arm, vis in (("E", None), ("Q1+E", "Q1"), ("C+E", "C")):
            for nm, (J, keep, pri) in sets.items():
                tasks[f"WP2|{nm}|{arm}|argmax"] = dict(J=J, keep=keep, pri=pri, X=X(pd_[:2], vis), margins=(0.0,))
                tasks[f"WP2|{nm}|{arm}|margin"] = dict(J=J, keep=keep, pri=pri, X=X(pd_[:2], vis), margins=MARGINS)
            for rule, mg in (("argmax", (0.0,)), ("margin", MARGINS)):
                tasks[f"shipped|F20|{arm}|{rule}"] = dict(J=Js, keep=KEEP, pri=T.PRI, X=X(pd_[2:], vis), margins=mg)
                tasks[f"shipped|F20, selector fitted on WP2|{arm}|{rule}"] = dict(J=J20, keep=KEEP, pri=T.PRI, X=X(pd_[:2], vis), margins=mg, Xa=X(pd_[2:], vis), Ja=Js)
        _T.update(tasks)
        res = par.pmap(unit2, [(t, r) for t in tasks for r in range(REPS)], workers=min(n_cpus(), a.workers), run=run, desc="q2 units", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        d, ch, hp = {}, {}, {}
        for t, d_, c_, h_ in res.values:
            d[t], ch[t] = d.get(t, 0) + d_ / REPS, ch.get(t, 0) + c_ / REPS
            hp.setdefault(t, []).extend(h_)
        rows = []
        for t in tasks:
            s1, nm, arm, rule = t.split("|")
            Jo = tasks[t].get("Ja", tasks[t]["J"])
            orc = Jo.max(1).mean(0) - Jo[:, tasks[t]["keep"]].mean(0)
            r = {"System 1": s1, "candidate set": nm, "K": Jo.shape[1], "features": arm, "rule": rule, "RFS System 1": C.cm(base[s1])}
            for sn in ("all", "stopped", "moving", "turn"):
                c = T.ci(C, d[t], Bd.st[sn])
                r |= {f"d {sn}": c[0], f"{sn} lo": c[1], f"{sn} hi": c[2]}
            r |= {"oracle d": C.cm(orc), "share of oracle": r["d all"] / C.cm(orc), "frames changed": float(ch[t].mean()),
                  "margin (median over fits)": float(np.median([h[2] for h in hp[t]])), "lambda (median)": float(np.median([h[1] for h in hp[t]]))}
            rows.append(r)
        stats.write_table(rows, OUT / "q2")
        con = []
        for arm in ("E", "Q1+E", "C+E"):
            for nm, ka, kb in (("margin - argmax (F20)", f"WP2|F20|{arm}|margin", f"WP2|F20|{arm}|argmax"), ("F30 - F20", f"WP2|F30|{arm}|argmax", f"WP2|F20|{arm}|argmax"),
                               ("speed only - F20", f"WP2|speed only (4)|{arm}|argmax", f"WP2|F20|{arm}|argmax"),
                               ("path only - F20", f"WP2|path only (5)|{arm}|argmax", f"WP2|F20|{arm}|argmax"),
                               ("shipped: transferred - retrained", f"shipped|F20, selector fitted on WP2|{arm}|argmax", f"shipped|F20|{arm}|argmax")):
                c = T.ci(C, d[ka] - d[kb])
                con.append({"features": arm, "contrast": nm, "d": c[0], "lo": c[1], "hi": c[2]})
        for nm, ka, kb in (("Q1+E - E (F20, margin)", "WP2|F20|Q1+E|margin", "WP2|F20|E|margin"), ("C+E - E (F20, margin)", "WP2|F20|C+E|margin", "WP2|F20|E|margin"),
                           ("selected shipped vs selected WP2 (absolute RFS, E, margin)", None, None)):
            c = T.ci(C, d[ka] - d[kb]) if ka else T.ci(C, base["shipped"] + d["shipped|F20|E|margin"] - base["WP2"] - d["WP2|F20|E|margin"])
            con.append({"features": "", "contrast": nm, "d": c[0], "lo": c[1], "hi": c[2]})
        stats.write_table(con, OUT / "q2_contrasts")
        np.savez(T.RUNS / "q2.npz", **{t.replace("|", "__"): v.astype(np.float32) for t, v in d.items()})
        run.info("Q2:\n%s\n%s", pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:+.3f}"), pd.DataFrame(con).to_string(float_format=lambda v: f"{v:+.3f}"))


# ---------------------------------------------------------------- figures
def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    from jevdrive import wod_zeroshot as Z
    T.FIG.mkdir(parents=True, exist_ok=True)
    df, perm, cur = pd.read_csv(OUT / "arms.csv"), pd.read_csv(OUT / "permutation.csv"), pd.read_csv(OUT / "learning_curve.csv")
    feats = [f for f in ARMS if f in set(df.features)]
    col = {"a": "#7f7f7f", "b": "#1f77b4", "c": "#d62728"}
    # ---- 1. out-of-fold gain per arm (ridge head), in-sample next to it
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, sn in zip(axs, ("all", "stopped", "moving")):
        g = df[(df.stratum == sn) & (df["head"] == "L")]
        for i, f in enumerate(feats):
            for j, sup in enumerate("abc"):
                x = g[(g.features == f) & (g.supervision == sup) & (g.variant == "oof")]
                if len(x):
                    r = x.iloc[0]
                    ax.errorbar(i + (j - 1) * 0.22, r.d, yerr=[[r.d - r.lo], [r.hi - r.d]], fmt="o", color=col[sup], capsize=3, ms=5, label=sup if i == 1 else None)
                y = g[(g.features == f) & (g.supervision == sup) & (g.variant == "ins")]
                if len(y):
                    ax.plot(i + (j - 1) * 0.22, y.iloc[0].d, "x", color=col[sup], ms=7, alpha=0.6)
        o = g["oracle d"].iloc[0]
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(o, color="g", ls="--", lw=1)
        ax.text(len(feats) - 0.5, o, f"F20 oracle {o:+.2f}", color="g", ha="right", va="bottom", fontsize=8)
        ax.set_xticks(range(len(feats)), feats, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{sn} (n {int(g.n.iloc[0])})", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("RFS of the selected candidate - WP2")
    h = [plt.Line2D([], [], marker="o", ls="", color=col[k]) for k in "abc"] + [plt.Line2D([], [], marker="x", ls="", color="0.4")]
    axs[0].legend(h, ["a: hindsight class (r2-train)", "b: rater scores, out-of-fold", "c: a then b, out-of-fold", "in-sample fit (b, c)"], fontsize=7.5, loc="upper left")
    fig.suptitle("Thin head (ridge) selecting among F20 on WOD val: out-of-fold gain over WP2, 95 % CI (paired bootstrap over sequences)", fontsize=10)
    fig.savefig(T.FIG / "01_arms_oof.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # ---- 2. permutation null and learning curve
    z = dict(np.load(T.RUNS / "heads.npz"))
    Bd = T.board()
    pk = [k for k in z if k.startswith("perm:") and k.endswith("|all")]
    fig, axs = plt.subplots(1, len(pk) + 1, figsize=(3.6 * (len(pk) + 1), 3.6))
    for ax, k in zip(axs, pk):
        arm = k[5:].split("|")[0]
        null = np.array([Bd.C.cm(d.astype(np.float64)) for d in z[k]])
        act = Bd.C.cm(z[f"{arm}|b|L|rep3"].astype(np.float64))
        ax.hist(null, bins=30, color="0.6")
        ax.axvline(act, color="#d62728", lw=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{arm}: actual {act:+.3f}, null {null.mean():+.3f} (p {(1 + (null >= act).sum()) / (1 + len(null)):.3f})", fontsize=8)
        ax.set_xlabel("out-of-fold gain, inputs permuted across frames", fontsize=8)
    ax = axs[-1]
    full = df[(df.stratum == "all") & (df["head"] == "L") & (df.supervision == "b") & (df.variant == "oof")].set_index("features")
    for f, g in cur.groupby("features"):
        xs, ys = list(g["training fraction"]) + [1.0], list(g.d) + [full.loc[f, "d"]]
        ax.plot(xs, ys, "o-", label=f, ms=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("share of the training sequences of each fold", fontsize=8)
    ax.set_title("learning curve (b, ridge), out-of-fold gain", fontsize=8)
    ax.legend(fontsize=7)
    fig.savefig(T.FIG / "02_permutation_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # ---- 3. eight frames: the 4 largest out-of-fold gains and losses of the arm with the best point estimate (repeat 0 picks, seed 0)
    pri = [f for f in feats if f"picks:{f}|b|L" in z and f not in ("E0",)]
    arm = max(pri, key=lambda f: full.loc[f, "d"])
    picks = z[f"picks:{arm}|b|L"][0]
    ii = np.arange(Bd.n)
    g0 = Bd.J[0][picks, ii] - Bd.J[0][KEEP]
    o = np.argsort(-g0, kind="stable")
    sel = [(int(i), "gain") for i in o[:4]] + [(int(i), "loss") for i in o[::-1][:4]]
    jb = T.first(Bd.J[0])
    t2, C = Z.root("t2_jpg"), Bd.C
    fig, axs = plt.subplots(4, 4, figsize=(17, 17), gridspec_kw={"width_ratios": [1, 1.25, 1, 1.25]})
    rows = []
    for q, (i, why) in enumerate(sel):
        r, c = q % 4, (q // 4) * 2
        ax = axs[r, c]
        ax.imshow(Image.open(t2 / Bd.names[i] / "1.jpg").convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{why} {q % 4 + 1}: {Bd.cluster[i]}, v0 {Bd.v0[i]:.1f} m/s", fontsize=8, loc="left")
        ax = axs[r, c + 1]
        for m in range(20):
            p = Bd.F[0][i, m]
            ax.plot(-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], color="0.8", lw=0.8)
        order = np.argsort(-C.sc[i])
        for j, o_ in enumerate(order):
            t = C.traj[i, o_]
            ax.plot(-np.r_[0, t[:, 1]], np.r_[0, t[:, 0]], "-" if j == 0 else "--", color="k" if j == 0 else "0.45", lw=2 if j == 0 else 1.2,
                    label=f"rater {j + 1}: score {C.sc[i, o_]:.0f}")
        name = lambda m: f"{T.G.PATHS[m // 4][1]} / {T.G.SPEEDS[m % 4]}"  # noqa: E731
        for lab, m, cl, ls, lw in (("WP2", KEEP, "#d62728", "-", 2.2), ("head", picks[i], "#1f77b4", "-", 2.0), ("oracle", jb[i], "#2ca02c", (0, (3, 2)), 2.0)):
            p = Bd.F[0][i, m]
            ax.plot(-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], color=cl, ls=ls, lw=lw, label=f"{lab}: {name(m)}, RFS {Bd.J[0][m, i]:.1f}")
            ax.plot(-p[[11, 19], 1], p[[11, 19], 0], "^", color=cl, ms=5)
        allp = np.concatenate([C.traj[i].reshape(-1, 2), Bd.F[0][i, [KEEP, picks[i], jb[i]]].reshape(-1, 2)])
        lim = max(3.0, 1.2 * np.abs(allp[:, 1]).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.05 * max(3.0, allp[:, 0].max()), 1.1 * max(3.0, allp[:, 0].max()))
        ax.plot(0, 0, "k*", ms=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False, ncol=2)
        rows.append({"panel": f"{why} {q % 4 + 1}", "frame": Bd.names[i], "cluster": Bd.cluster[i], "v0": Bd.v0[i], "head pick": name(picks[i]), "oracle pick": name(jb[i]),
                     "RFS WP2": Bd.J[0][KEEP, i], "RFS head": Bd.J[0][picks[i], i], "RFS oracle": Bd.J[0][jb[i], i], "rater scores": " ".join(f"{x:.0f}" for x in C.sc[i][order])})
    fig.suptitle(f"Arm {arm}, supervision b, ridge head (out-of-fold picks of repeat 0, WP2 seed 0): left block = 4 largest gains, right block = 4 largest losses.\n"
                 "FRONT camera at t0; BEV: ego at the star heading up, lateral axis stretched, grey = the 20 candidates, markers at 3 s and 5 s", fontsize=10, y=0.9)
    fig.savefig(T.FIG / "03_frames.png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "figure_frames.csv", index=False, float_format="%.2f")
    print(arm, pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    p = sp_.add_parser("fit")
    p.add_argument("--perms", type=int, default=N_PERM)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--part", default="ridge", choices=("ridge", "mlp"))
    sp_.add_parser("report")
    sp_.add_parser("figs")
    p = sp_.add_parser("q2")
    p.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    {"fit": cmd_fit, "report": cmd_report, "figs": cmd_figs, "q2": cmd_q2}[a.cmd](a)
