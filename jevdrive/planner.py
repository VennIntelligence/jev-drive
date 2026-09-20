"""Stage-A planner: a fixed trajectory vocabulary scored by a thin head on frozen cached features.

Heads, all on the same rows, the same split and the same frozen features (traj.py builds the targets and the
vocabulary):
  cls        multinomial logistic regression over the K anchors, cross-entropy to the nearest anchor;
             cls_soft uses a soft target over the m nearest anchors instead. Prediction = argmax anchor.
  ridge      direct closed-form regression of the 2T waypoint coordinates.
  mlp_reg    one hidden layer (GELU + dropout) onto the waypoints; mlp_cls onto the K anchors.
  ego        the same heads on the 11-dim past ego state of labels.py.
  const_*    no training: constant velocity and constant turn rate (CTRV) from the current ego state.
  majority   no features: the anchors that are most often the nearest one on the training scenes.
  oracle     no model: the best anchor in the vocabulary, i.e. the coverage floor of any scorer on it.

Protocol: the official nuScenes split, by scene. Features are standardised with training-scene statistics only.
Each head picks its L2 strength on one scene-grouped inner split (20 % of the training scenes held out), scored
by the inner-val ADE, then refits on all training scenes; the val scenes are read once, for the numbers we
report. Confidence intervals resample whole scenes.

Solvers: ridge is closed form for every lambda at once (one eigendecomposition of the training gram); the
vocabulary classifier is a full-batch multinomial logistic regression solved by the batched L-BFGS of probe.py,
lambdas batched together and rows chunked so that K = 8192 fits; the MLPs are minibatch AdamW.
Fusion with the ego state follows the v0 warning that 11 ego dimensions drown in 2560 feature dimensions:
'+ego' projects the features onto their top PCA_DIM principal components, standardises both blocks and
concatenates them; ridge_late instead regresses what the ego-only ridge got wrong.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from . import probe, traj
from .common import get_logger
from .labels import EGO_COLS

log = get_logger(__name__)
DEV = "cuda"
LAM_RIDGE = np.logspace(-4, 3, 15)  # L2 on the mean squared error; features are standardised
LAM_CLS = np.logspace(-6, 1, 8)  # L2 on the mean cross-entropy
SOFT_M, SOFT_TAU = 5, 0.5  # soft target: m nearest anchors, softmax(-RMS displacement / tau metres)
PCA_DIM, INNER_FRAC = 64, 0.2
HIDDEN, DROPOUT, MLP_EPOCHS, MLP_BS, MLP_LR, MLP_WD = 512, 0.1, 40, 512, 1e-3, 1e-2
MAX_ITER, MAX_BATCH_PARAMS = 600, 6e7  # L-BFGS iterations, and parameters solved in one batch
TOPK = (1, 5, 10)
CI_COLS = ("ade", "fde")
COLS = ["ade", "fde", "ade@1s", "fde@1s", "ade@2s", "fde@2s", "minade1", "minade5", "minade10", "minfde10"]


class Split:
    """Row positions of the official split, plus one scene-grouped inner split of the training scenes."""

    def __init__(self, lab: pd.DataFrame, seed: int = 0):
        self.scenes = lab.scene.to_numpy()
        self.train = np.flatnonzero(lab.split.to_numpy() == "train")
        self.val = np.flatnonzero(lab.split.to_numpy() == "val")
        f, v = next(GroupShuffleSplit(1, test_size=INNER_FRAC, random_state=seed)
                    .split(self.train, groups=self.scenes[self.train]))
        self.fit, self.sel = self.train[f], self.train[v]

    def __repr__(self):
        u = lambda r: len(np.unique(self.scenes[r]))
        return (f"Split(train {len(self.train)} rows / {u(self.train)} scenes, inner fit/sel "
                f"{len(self.fit)}/{len(self.sel)} rows / {u(self.fit)}/{u(self.sel)} scenes, "
                f"val {len(self.val)} rows / {u(self.val)} scenes)")


def standardize(X: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    """Z-score every column with the statistics of `rows` (the training scenes)."""
    mu, sd = X[rows].double().mean(0), X[rows].double().std(0, correction=0)
    return (X - mu.float()) / torch.where(sd > 1e-6, sd, 1).float()


def gram_eigh(A: torch.Tensor):
    """Eigendecomposition of A^T A: the gram is accumulated in float32 on the GPU (float64 GEMMs are 1/64
    rate on this card) and diagonalised in float64 on the CPU, where d^3 is cheap."""
    ev, V = torch.linalg.eigh((A.T @ A).double().cpu())
    return ev.float().to(A.device), V.float().to(A.device)


def pca(X: torch.Tensor, rows: np.ndarray, p: int) -> torch.Tensor:
    """Project every row onto the top-p principal components of `rows`."""
    mu = X[rows].mean(0)
    _, V = gram_eigh(X[rows] - mu)
    return (X - mu) @ V[:, -p:].flip(1)


def ridge_solve(X: torch.Tensor, Y: torch.Tensor, rows: np.ndarray, lams) -> torch.Tensor:
    """W (P, d+1, o) minimising mean squared error + lam |W|^2 over `rows`, every lam from one eigh."""
    A, B = X[rows], Y[rows]
    mx, my = A.mean(0), B.mean(0)
    A, B = A - mx, B - my
    ev, V = gram_eigh(A)
    Z = V.T @ (A.T @ B)
    W = torch.stack([V @ (Z / (ev[:, None] + lam * len(rows))) for lam in lams])
    return torch.cat([W, (my - mx @ W).unsqueeze(1)], 1)


def linear_apply(W: torch.Tensor, X: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    """(P, len(rows), o) outputs of the affine maps W (P, d+1, o)."""
    d = X.shape[1]
    return X[rows] @ W[:, :d] + W[:, d].unsqueeze(1)


def ce_solve(X: torch.Tensor, tgt: tuple[np.ndarray, np.ndarray], rows: np.ndarray, lams, K: int,
             max_iter: int = MAX_ITER) -> tuple[torch.Tensor, dict]:
    """Multinomial logistic regression over K anchors on `rows`: mean cross-entropy to the (possibly soft)
    target `tgt` = (anchor ids (n, m), weights (n, m)) plus lam/2 |W|^2, every lam in as few batches as fit."""
    d, n, D = X.shape[1], len(rows), (X.shape[1] + 1) * K
    Xr, group = X[rows].contiguous(), max(1, int(MAX_BATCH_PARAMS // D))
    ti = torch.as_tensor(tgt[0][rows], device=DEV, dtype=torch.long)
    tw = torch.as_tensor(tgt[1][rows], device=DEV, dtype=torch.float32)
    out, iters = [], []
    for g in range(0, len(lams), group):
        lam = torch.as_tensor(np.asarray(lams[g:g + group], np.float32), device=DEV)
        chunk = max(256, int(2**25 // (len(lam) * K)))

        def fun(v, ids, lam=lam, chunk=chunk):
            p = len(ids)
            W = v.view(p, d + 1, K)
            loss = torch.zeros(p, device=DEV, dtype=torch.float64)
            gV, gb = torch.zeros(p, d, K, device=DEV), torch.zeros(p, K, device=DEV)
            for a in range(0, n, chunk):
                x = Xr[a:a + chunk]
                i2 = ti[a:a + chunk].unsqueeze(0).expand(p, -1, -1).contiguous()
                w2 = tw[a:a + chunk].unsqueeze(0).expand(p, -1, -1).contiguous()
                logp = (x @ W[:, :d] + W[:, d].unsqueeze(1)).log_softmax(-1)  # (p, chunk, K)
                loss -= (logp.gather(2, i2) * w2).sum((1, 2)).double()
                r = logp.exp_().scatter_add_(2, i2, -w2)  # softmax - target
                gV += x.T @ r
                gb += r.sum(1)
            W2 = W[:, :d]
            loss = loss / n + 0.5 * lam[ids].double() * W2.square().sum((1, 2)).double()
            grad = torch.cat([gV / n + lam[ids, None, None] * W2, (gb / n).unsqueeze(1)], 1)
            return loss.float(), grad.view(p, -1)

        w, it = probe.lbfgs(fun, torch.zeros(len(lam), D, device=DEV), max_iter)
        out.append(w.view(len(lam), d + 1, K))
        iters.append(it)
    it = np.concatenate(iters)
    return torch.cat(out), {"iter_mean": float(it.mean()), "iter_max": int(it.max()),
                            "not_converged": int((it >= max_iter).sum())}


def cls_topk(W: torch.Tensor, X: torch.Tensor, rows: np.ndarray, kmax: int, chunk: int = 1024) -> np.ndarray:
    """Top-kmax anchor ids by score, (len(rows), P, kmax), best first."""
    out = []
    for a in range(0, len(rows), chunk):
        s = linear_apply(W, X, rows[a:a + chunk])
        out.append(s.topk(min(kmax, s.shape[-1]), dim=-1).indices.permute(1, 0, 2).cpu().numpy())
    return np.concatenate(out)


def _mlp(d: int, out: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(d, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
                               torch.nn.Linear(HIDDEN, out)).to(DEV)


def mlp_solve(X: torch.Tensor, y: torch.Tensor, rows: np.ndarray, out: int, cls: bool, epochs: int,
              sel=None, seed: int = 0):
    """Train one MLP on `rows` with AdamW and a cosine schedule. With `sel` = (rows, score(output) -> float),
    also return the best score and the epoch it was reached at, so the refit can stop there."""
    torch.manual_seed(seed)
    net = _mlp(X.shape[1], out)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = torch.nn.functional.cross_entropy if cls else torch.nn.functional.mse_loss
    r = torch.as_tensor(rows, device=DEV)
    best = (np.inf, epochs)
    for ep in range(epochs):
        net.train()
        for b in r[torch.randperm(len(r), device=DEV)].split(MLP_BS):
            opt.zero_grad(set_to_none=True)
            lossf(net(X[b]), y[b]).backward()
            opt.step()
        sched.step()
        if sel is not None:
            best = min(best, (sel[1](mlp_predict(net, X, sel[0])), ep + 1))
    return net, best


@torch.inference_mode()
def mlp_predict(net: torch.nn.Module, X: torch.Tensor, rows: np.ndarray, chunk: int = 4096) -> torch.Tensor:
    net.eval()
    return torch.cat([net(X[b]) for b in torch.as_tensor(rows, device=DEV).split(chunk)])


def _pick(scores, lams, what: str) -> int:
    """Index of the best inner-val score, warning when it sits on the edge of the lambda grid."""
    i = int(np.argmin(scores))
    if i in (0, len(lams) - 1):
        log.warning("%s picked lambda=%.3g at the edge of the grid (inner-val ADE %.3f)", what, lams[i], scores[i])
    return i


class Heads:
    """The trainable heads on one feature matrix, sharing a split, the targets and a vocabulary."""

    def __init__(self, X: torch.Tensor, sp: Split, fut: np.ndarray, Y: torch.Tensor, rate: float):
        self.X, self.sp, self.fut, self.Y, self.rate = X, sp, fut, Y, rate
        self.T = fut.shape[1]

    def _ade(self, pred: np.ndarray, rows: np.ndarray) -> float:
        return float(np.linalg.norm(pred - self.fut[rows], axis=-1).mean())

    def ridge(self):
        sp = self.sp
        W = ridge_solve(self.X, self.Y, sp.fit, LAM_RIDGE)
        p = linear_apply(W, self.X, sp.sel).reshape(len(LAM_RIDGE), -1, self.T, 2).cpu().numpy()
        s = [self._ade(p[i], sp.sel) for i in range(len(LAM_RIDGE))]
        best = _pick(s, LAM_RIDGE, "ridge")
        W = ridge_solve(self.X, self.Y, sp.train, [LAM_RIDGE[best]])
        pv = linear_apply(W, self.X, sp.val).reshape(-1, self.T, 2).cpu().numpy()
        return pv[:, None], {"lam": float(LAM_RIDGE[best]), "sel_ade": s[best]}

    def cls(self, tgt, anchors: torch.Tensor, kmax: int = max(TOPK)):
        sp, K = self.sp, len(anchors)
        A = anchors.reshape(K, -1, 2).cpu().numpy()
        W, st = ce_solve(self.X, tgt, sp.fit, LAM_CLS, K)
        top = cls_topk(W, self.X, sp.sel, 1)
        s = [self._ade(A[top[:, i, 0]], sp.sel) for i in range(len(LAM_CLS))]
        best = _pick(s, LAM_CLS, "cls")
        del W
        W, st2 = ce_solve(self.X, tgt, sp.train, [LAM_CLS[best]], K)
        pv = A[cls_topk(W, self.X, sp.val, kmax)[:, 0]]
        del W
        torch.cuda.empty_cache()
        return pv, {"lam": float(LAM_CLS[best]), "sel_ade": s[best], **{f"sel_{k}": v for k, v in st.items()},
                    **st2}

    def mlp(self, anchors: torch.Tensor | None = None, tgt=None, kmax: int = max(TOPK)):
        sp, T = self.sp, self.T
        if anchors is not None:
            K = len(anchors)
            A = anchors.reshape(K, -1, 2).cpu().numpy()
            y = torch.as_tensor(tgt[0][:, 0], device=DEV, dtype=torch.long)
            score = lambda o: self._ade(A[o.argmax(1).cpu().numpy()], sp.sel)
            pred = lambda o: A[o.topk(min(kmax, K), -1).indices.cpu().numpy()]
        else:
            K, y = 0, self.Y
            score = lambda o: self._ade(o.reshape(-1, T, 2).cpu().numpy(), sp.sel)
            pred = lambda o: o.reshape(-1, T, 2).cpu().numpy()[:, None]
        _, (s, ep) = mlp_solve(self.X, y, sp.fit, K or 2 * T, K > 0, MLP_EPOCHS, (sp.sel, score))
        net, _ = mlp_solve(self.X, y, sp.train, K or 2 * T, K > 0, ep)
        return pred(mlp_predict(net, self.X, sp.val)), {"epochs": ep, "sel_ade": s}


def eval_preds(preds: np.ndarray, gt: np.ndarray, rate: float) -> dict[str, np.ndarray]:
    """Per-sample metrics of (n, kmax, T, 2) predictions ordered by score: top-1 ADE/FDE plus minADE/minFDE."""
    return traj.sample_metrics(preds[:, 0], gt, rate) | traj.min_metrics(preds, gt, TOPK)


def add(rows: list, per_sample: dict, m: dict[str, np.ndarray], scenes: np.ndarray, **fields) -> dict:
    """One result row: the mean of every per-sample metric, plus scene-bootstrap CIs on the headline ones."""
    ci = {f"{c}_{b}": v for c in CI_COLS if c in m for b, v in zip(("lo", "hi"), traj.boot_ci(m[c], scenes))}
    rows.append({**fields, "n": len(scenes), **{k: float(v.mean()) for k, v in m.items()}, **ci})
    per_sample[(fields["head"], fields["features"], fields["K"])] = m
    log.info("%-12s %-24s K=%-5s ADE %.3f FDE %.3f minADE10 %.3f", fields["head"], fields["features"],
             fields["K"], rows[-1].get("ade", np.nan), rows[-1].get("fde", np.nan),
             rows[-1].get("minade10", np.nan))
    return rows[-1]


def load(name: str, src: dict, ego: torch.Tensor, sp: Split) -> torch.Tensor:
    """Standardised features for '<set>/<array>', '<set>/<array>+ego' (PCA + ego state) or 'ego'."""
    if name == "ego":
        return standardize(ego, sp.train)
    f, pos = src[name.removesuffix("+ego")]
    X = standardize(torch.from_numpy(np.load(f, mmap_mode="r")[pos]).to(DEV).float(), sp.train)
    if name.endswith("+ego"):
        X = torch.cat([standardize(pca(X, sp.train, PCA_DIM), sp.train), standardize(ego, sp.train)], 1)
    return X


def run(version: str, out_dir, rl=None, horizon: float = traj.HORIZON, rate: float = traj.RATE,
        ks=(64, 256, 1024, 4096, 8192), k_ref: int = 1024, pattern: str = ".*", ref_set: str | None = None,
        seed: int = 0) -> pd.DataFrame:
    """The whole planner evaluation: vocabulary sweep, layer curve, head comparison and baselines."""
    lab, fut, vel, yaw_rate = traj.build(version, horizon, rate)
    sp = Split(lab, seed)
    log.info("%s", sp)
    scenes_val, gt = sp.scenes[sp.val], fut[sp.val]
    n, T = len(lab), fut.shape[1]
    F = torch.as_tensor(fut.reshape(n, -1), device=DEV)
    ego = torch.as_tensor(lab[EGO_COLS].to_numpy(np.float32), device=DEV)
    rows, per_sample = [], {}

    vocab, tgt, soft = {}, {}, {}
    hard_w = np.concatenate([np.ones((n, 1)), np.zeros((n, SOFT_M - 1))], 1).astype(np.float32)
    for k in ks:
        vocab[k] = traj.kmeans(F[sp.train], k, seed=seed)
        i, e = traj.nearest(F, vocab[k], SOFT_M)
        tgt[k], soft[k] = (i, hard_w), (i, traj.soft_target(e / np.sqrt(T), SOFT_TAU))
        np.save(out_dir / f"vocab_K{k}.npy", vocab[k].reshape(k, T, 2).cpu().numpy())

    # --- baselines without image features ----------------------------------------------------------
    for name, p in (("const_velocity", traj.const_velocity(vel, horizon, rate)),
                    ("const_turn_rate", traj.const_turn_rate(vel, yaw_rate, horizon, rate))):
        add(rows, per_sample, eval_preds(p[sp.val][:, None], gt, rate), scenes_val, head=name, features="-", K=0)
    mean_traj = fut[sp.train].mean(0)
    add(rows, per_sample, eval_preds(np.repeat(mean_traj[None, None], len(sp.val), 0), gt, rate), scenes_val,
        head="mean_traj", features="-", K=0)
    for k in ks:
        cnt = np.bincount(tgt[k][0][sp.train, 0], minlength=k)
        top = vocab[k].reshape(k, T, 2)[cnt.argsort()[::-1][:max(TOPK)].copy()].cpu().numpy()
        add(rows, per_sample, eval_preds(np.repeat(top[None], len(sp.val), 0), gt, rate), scenes_val,
            head="majority", features="-", K=k)
        o = traj.oracle_metrics(vocab[k], gt)
        add(rows, per_sample, {"ade": o["oracle_ade"], "fde": o["oracle_fde"]}, scenes_val,
            head="oracle", features="-", K=k)

    # --- one pass over every feature set: linear regression and linear vocabulary classifier ---------
    src = probe.feature_sources(version, lab.sample_token, pattern)
    names = ["ego", *src]
    log.info("%d feature sets: %s", len(names), ", ".join(names))

    def run_heads(name, X, which, k=k_ref):
        h = Heads(X, sp, fut, F, rate)
        for w in which:
            t0 = time.perf_counter()
            p, st = {"ridge": lambda: h.ridge(), "mlp_reg": lambda: h.mlp(),
                     "mlp_cls": lambda: h.mlp(vocab[k], tgt[k]), "cls": lambda: h.cls(tgt[k], vocab[k]),
                     "cls_soft": lambda: h.cls(soft[k], vocab[k])}[w]()
            r = add(rows, per_sample, eval_preds(p, gt, rate), scenes_val, head=w, features=name,
                    K=k if "cls" in w else 0, seconds=time.perf_counter() - t0,
                    **{f"fit_{a}": b for a, b in st.items()})
            if rl is not None:
                rl.event("planner_result", **r)

    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(load, names[0], src, ego, sp)
        for i, name in enumerate(tqdm(names, desc="feature sets", dynamic_ncols=True)):
            X = nxt.result()
            if i + 1 < len(names):
                nxt = ex.submit(load, names[i + 1], src, ego, sp)
            run_heads(name, X, ("ridge", "cls", "mlp_reg", "mlp_cls") if name == "ego" else ("ridge", "cls"))
            del X
            torch.cuda.empty_cache()

    # --- the reference feature set, chosen on the inner split, gets the full treatment ---------------
    cur = pd.DataFrame(rows)
    cand = cur[(cur["head"] == "cls") & (cur["features"] != "ego")]
    best = ref_set or cand.loc[cand.fit_sel_ade.idxmin(), "features"]
    log.info("reference feature set (best inner-val ADE of cls): %s", best)
    X = load(best, src, ego, sp)
    run_heads(best, X, ("cls_soft", "mlp_reg", "mlp_cls"))
    for k in ks:
        if k != k_ref:
            run_heads(best, X, ("cls",), k)
    del X
    torch.cuda.empty_cache()
    run_heads(best + "+ego", load(best + "+ego", src, ego, sp), ("ridge", "cls", "mlp_reg"))
    torch.cuda.empty_cache()

    # late fusion: the image head regresses what the ego-only ridge gets wrong
    Xe, Xi = load("ego", src, ego, sp), load(best, src, ego, sp)
    lam = cur[(cur["head"] == "ridge") & (cur["features"] == "ego")].fit_lam.iloc[0]
    base = linear_apply(ridge_solve(Xe, F, sp.train, [lam]), Xe, np.arange(n))[0]
    p, st = Heads(Xi, sp, (F - base).reshape(n, T, 2).cpu().numpy(), F - base, rate).ridge()
    add(rows, per_sample, eval_preds(p + base[sp.val].reshape(-1, 1, T, 2).cpu().numpy(), gt, rate), scenes_val,
        head="ridge_late", features=best + "+ego", K=0, **{f"fit_{a}": b for a, b in st.items()})
    del Xe, Xi
    torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "results.csv", index=False)
    np.savez_compressed(out_dir / "per_sample.npz", scenes=scenes_val,
                        **{"|".join(map(str, k)) + "|" + m: v for k, d in per_sample.items() for m, v in d.items()})
    (out_dir / "results.md").write_text(to_markdown(res, best, k_ref))
    log.info("results -> %s\n%s", out_dir, to_markdown(res, best, k_ref))
    if rl is not None:
        for r in res.to_dict("records"):
            rl.event("planner_result", **r)
        q = res[(res["head"] == "cls") & res["features"].str.contains("/L")]
        for r in q.to_dict("records"):
            bb, feat = r["features"].split("/")
            rl.scalar(f"planner_ade/{bb}_{feat[4:]}", r["ade"], int(feat[1:3]))
    return res


def _tab(df: pd.DataFrame, index) -> str:
    t = df.set_index(index)[[c for c in COLS if c in df and df[c].notna().any()]]
    return t.to_markdown(floatfmt=".3f")


def to_markdown(res: pd.DataFrame, best: str, k_ref: int) -> str:
    head, feat = res["head"], res["features"]
    main = feat.isin((best, best + "+ego", "ego", "-")) & res.K.isin((0, k_ref))
    out = [f"n = {int(res.n.iloc[0])} val samples; reference feature set `{best}`, reference K = {k_ref}. "
           "Regression heads have a single mode, so their minADE_k does not fall with k.\n"]
    k = res[head.isin(("oracle", "majority")) | ((head.isin(("cls", "cls_soft"))) & (feat == best))]
    out.append("### Vocabulary sweep (oracle = best anchor in the vocabulary)\n\n"
               + _tab(k.sort_values(["K", "head"]), ["K", "head"]) + "\n")
    c = res[head.isin(("ridge", "cls")) & (feat != "ego") & res.K.isin((0, k_ref))]
    out.append("### Layer curve\n\n" + _tab(c.sort_values(["features", "head"]), ["features", "head"]) + "\n")
    out.append("### Heads and baselines\n\n" + _tab(res[main], ["head", "features"]) + "\n")
    ci = res[main].assign(
        ADE=lambda d: d.apply(lambda r: f"{r.ade:.3f} [{r.ade_lo:.3f}, {r.ade_hi:.3f}]", axis=1),
        FDE=lambda d: d.apply(lambda r: f"{r.fde:.3f} [{r.fde_lo:.3f}, {r.fde_hi:.3f}]", axis=1))
    out.append("### Scene-bootstrap 95 % CI\n\n"
               + ci.set_index(["head", "features"])[["ADE", "FDE"]].to_markdown() + "\n")
    return "\n".join(out)


@torch.inference_mode()
def bench_heads(d: int, T: int, ks=(64, 256, 1024, 4096, 8192), reps: int = 1000, warmup: int = 100) -> list[dict]:
    """Batch-1 latency of the head itself, warm: score every anchor, take the top 10 and read off the waypoints.
    Feature extraction is measured separately (features.bench); this is only what the head adds on top."""
    x, out = torch.randn(1, d, device=DEV), []
    for name, K in [("ridge", 0), ("mlp_reg", 0)] + [(h, k) for k in ks for h in ("cls", "mlp_cls")]:
        anchors = torch.randn(max(K, 1), T, 2, device=DEV)
        o = K or 2 * T
        if name.startswith("mlp"):
            net = _mlp(d, o).eval()
            params, g = (d * HIDDEN + HIDDEN * o) / 1e6, (lambda: net(x))
        else:
            W, b = torch.randn(d, o, device=DEV), torch.randn(o, device=DEV)
            params, g = d * o / 1e6, (lambda: x @ W + b)
        f = (lambda: anchors[g().topk(10, -1).indices[0]]) if K else g
        for _ in range(warmup):
            f()
        torch.cuda.synchronize()
        ts = np.empty(reps)
        for i in range(reps):
            t0 = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ts[i] = (time.perf_counter() - t0) * 1e3
        out.append({"head": name, "K": K, "params_m": params, "mean_ms": float(ts.mean()),
                    "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))})
        log.info("head latency %-8s K=%-5d %6.3f ms mean, p50 %6.3f, p95 %6.3f", name, K, ts.mean(),
                 out[-1]["p50_ms"], out[-1]["p95_ms"])
    return out
