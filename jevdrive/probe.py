"""Steps 3-4: linear probes on every (backbone, layer, pooling) plus baselines, and the results table.

Every feature set gets the same probe as sklearn's make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS,
scoring="neg_log_loss")): standardize on the training scenes, multinomial logistic regression with an L2 penalty,
C picked by scene-grouped inner CV (neg log-loss), refit on all training scenes with the best C.
It runs on the GPU instead (see `solve`): per feature set, every (outer fold, inner fold or refit, C) problem is one
column block of a single batched L-BFGS solve, with each problem's training rows given by a 0/1 weight mask.
Outer protocols:
  val    official split: fit on train scenes, evaluate on val scenes
  loso   leave-one-scene-out (only when there are at most MAX_LOSO scenes, i.e. mini)
  kfold  scene-grouped K-fold (OUTER folds) otherwise; predictions are pooled over folds
Baselines: majority (train class prior, Laplace-smoothed), ego (same probe on past ego state) and ego_rule
(no training: extrapolate the current yaw rate over the horizon and apply the label thresholds).
Metrics on all eval samples and on the hard subset (|current yaw rate| small, see labels.py).
"""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .common import CLASSES, get_logger, processed_dir
from .labels import EGO_COLS, HORIZON, N, THRESH_DEG

log = get_logger(__name__)
K = len(CLASSES)
CS = np.logspace(-4, 2, 13)
INNER, OUTER, MAX_LOSO = 4, 5, 50
MAX_ITER, GTOL, HIST = 3000, 1e-4, 10  # sklearn's lbfgs defaults for max_iter/tol; L-BFGS history length
DEV = "cuda"


def feature_sources(version: str, tokens: pd.Series, pattern: str = ".*") -> dict[str, tuple[Path, np.ndarray]]:
    """Stored feature arrays matching `pattern`: {'<set>/<name>': (npy path, row positions aligned to `tokens`)}."""
    out = {}
    for d in sorted((processed_dir(version) / "features").glob("*/")):
        pos = pd.Index(pd.read_parquet(d / "index.parquet").sample_token).get_indexer(tokens)
        if (pos < 0).any():
            log.warning("%s misses %d labeled samples, skipped", d.name, (pos < 0).sum())
            continue
        out |= {f"{d.name}/{f.stem}": (f, pos) for f in sorted(d.glob("*.npy")) if re.fullmatch(pattern, f"{d.name}/{f.stem}")}
    return out


def ego_rule(yaw_rate_now: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """One-hot (eps-smoothed) turn class from constant-yaw-rate extrapolation; its NLL only reflects eps."""
    dyaw = yaw_rate_now * HORIZON
    return np.eye(K)[np.select([dyaw > THRESH_DEG, dyaw < -THRESH_DEG], [0, 2], 1)] * (1 - K * eps) + eps


def outer_folds(lab: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    groups = lab.scene.to_numpy()
    folds = {"val": [(np.flatnonzero(lab.split == "train"), np.flatnonzero(lab.split == "val"))]}
    if len(np.unique(groups)) <= MAX_LOSO:
        folds["loso"] = [(np.flatnonzero(groups != s), np.flatnonzero(groups == s)) for s in np.unique(groups)]
    else:
        folds["kfold"] = list(GroupKFold(OUTER).split(groups, groups=groups))
    return folds


class Problems:
    """All logistic regressions one feature set needs, as masks over the n samples.
    Problem p = (outer fold o, inner fold j or refit, C): trains on rows where mask[:, p] = 1."""

    def __init__(self, y: np.ndarray, groups: np.ndarray, outer: list[tuple[np.ndarray, np.ndarray]]):
        masks, self.o, self.c, self.inner, self.refit = [], [], [], [], []  # inner[o] = [(test rows, problem ids)]
        for o, (tr, _) in enumerate(outer):
            inner = GroupKFold(min(INNER, len(np.unique(groups[tr])))).split(tr, groups=groups[tr])
            self.inner.append([])
            for itr, ite in [*inner, (np.arange(len(tr)), None)]:
                m = np.zeros(len(y), bool)
                m[tr[itr]] = True
                ids = list(range(len(masks), len(masks) + len(CS)))
                if ite is None:
                    self.refit.append(ids)
                else:
                    self.inner[o].append((tr[ite], ids))
                masks += [m] * len(CS)
                self.o += [o] * len(CS)
                self.c += list(CS)
        self.outer, self.P = outer, len(masks)
        self.mask = torch.tensor(np.stack(masks, 1), dtype=torch.float32, device=DEV)  # (n, P)
        self.n_p = self.mask.sum(0)
        self.l2 = 1 / (torch.tensor(self.c, device=DEV, dtype=torch.float32) * self.n_p)  # sklearn: mean loss + |W|^2/(2Cn)
        self.o = torch.tensor(self.o, device=DEV)
        self.y = torch.tensor(y, device=DEV)
        self.onehot = torch.nn.functional.one_hot(self.y, K).float()
        outer_mask = torch.tensor(np.stack([np.isin(np.arange(len(y)), tr) for tr, _ in outer], 1), device=DEV).double()
        self.outer_mask = outer_mask / outer_mask.sum(0)  # (n, O) column-normalized, for per-fold mean / variance


def solve(X: torch.Tensor, pr: Problems, max_iter: int = MAX_ITER):
    """Batched L-BFGS over all problems of `pr` on features X (n, d) float32 on the GPU.

    Standardization is folded into the weights: X is centered and scaled once globally (so fp32 stays accurate),
    and each outer fold's StandardScaler becomes a per-fold scale A and shift B on the logits, which is exact.
    Returns logits(rows) -> (len(rows), P, K) and solver stats."""
    n, d = X.shape
    mu, sd = X.double().mean(0), X.double().std(0, correction=0)
    X = ((X - mu.float()) / torch.where(sd > 0, sd, 1).float()).contiguous()
    fm = pr.outer_mask.T @ X.double()  # (O, d) per-fold mean and std, as StandardScaler(ddof=0) on the fold's rows
    fs = (pr.outer_mask.T @ X.double() ** 2 - fm ** 2).clamp_min(0).sqrt()
    fs = torch.where(fs > 1e-12, fs, 1)  # constant within the fold: sklearn uses scale 1
    A, B = (1 / fs).float(), (fm / fs).float()  # standardized x = x*A - B

    def logits(W, ids, rows=slice(None)):  # W (p, d+1, K) -> (rows, p, K)
        V = W[:, :d] * A[pr.o[ids], :, None]
        c = W[:, d] - torch.einsum("pd,pdk->pk", B[pr.o[ids]], W[:, :d])
        return (X[rows] @ V.permute(1, 0, 2).reshape(d, -1)).view(-1, len(ids), K) + c

    def fun(w, ids):  # w (p, (d+1)*K) -> loss (p,), grad (p, (d+1)*K)
        W = w.view(len(ids), d + 1, K)
        Z = logits(W, ids)
        logp = Z.log_softmax(-1)
        sw = pr.mask[:, ids] / pr.n_p[ids]  # (n, p): mean over the problem's training rows
        l2 = pr.l2[ids]
        loss = -(sw * (logp * pr.onehot[:, None]).sum(-1)).sum(0) + 0.5 * l2 * W[:, :d].square().sum((1, 2))
        R = (logp.exp() - pr.onehot[:, None]) * sw[..., None]  # (n, p, K)
        gb = R.sum(0)
        gV = (X.T @ R.view(n, -1)).view(d, len(ids), K).permute(1, 0, 2)
        gW = gV * A[pr.o[ids], :, None] - B[pr.o[ids], :, None] * gb[:, None] + l2[:, None, None] * W[:, :d]
        return loss, torch.cat([gW, gb[:, None]], 1).view(len(ids), -1)

    w, n_iter = lbfgs(fun, torch.zeros(pr.P, (d + 1) * K, device=DEV), max_iter)
    stats = {"iter_mean": float(n_iter.mean()), "iter_max": int(n_iter.max()),
             "not_converged": int((n_iter >= max_iter).sum())}
    W = w.view(pr.P, d + 1, K)
    return (lambda rows, ids=slice(None): logits(W[ids], torch.arange(pr.P, device=DEV)[ids], rows)), stats


def lbfgs(fun, x: torch.Tensor, max_iter: int, gtol: float = GTOL, m: int = HIST, max_ls: int = 10):
    """Minimize P independent problems at once from x (P, D); fun(x_sub, ids) -> (f (p,), g (p, D)) with ids
    indexing the P problems. Returns the solutions and the iterations each problem took.
    Armijo backtracking per problem; a problem stops at max|g| <= gtol (scipy's criterion), when a line search
    fails to decrease f, or at max_iter. Converged problems are dropped from the batch (compaction)."""
    P, D = x.shape
    out, n_iter = x.clone(), torch.zeros(P, dtype=torch.long, device=x.device)
    ids = torch.arange(P, device=x.device)
    f, g = fun(x, ids)
    S, Y = x.new_zeros(m, P, D), x.new_zeros(m, P, D)
    rho, gamma = x.new_zeros(m, P), x.new_ones(P)
    for it in range(max_iter):
        done = g.abs().amax(1) <= gtol
        if it:
            done |= stalled
        if done.any():  # write back finished problems, keep the rest
            out[ids[done]], n_iter[ids[done]] = x[done], it
            keep = ~done
            ids, x, f, g, S, Y, rho, gamma = ids[keep], x[keep], f[keep], g[keep], S[:, keep], Y[:, keep], rho[:, keep], gamma[keep]
            if not len(ids):
                break
        # two-loop recursion over the ring buffer, newest first; rho = 0 marks skipped / empty pairs
        q, alpha, order = -g, [None] * m, [(it - 1 - j) % m for j in range(min(it, m))]
        for j in order:
            alpha[j] = rho[j] * (S[j] * q).sum(1)
            q = q - alpha[j][:, None] * Y[j]
        q = q * gamma[:, None]
        for j in reversed(order):
            q = q + (alpha[j] - rho[j] * (Y[j] * q).sum(1))[:, None] * S[j]
        gd = (g * q).sum(1)
        bad = gd >= 0  # not a descent direction: fall back to steepest descent
        q[bad], gd[bad] = -g[bad], -(g[bad] ** 2).sum(1)
        t = 1 / g.norm(dim=1) if it == 0 else torch.ones_like(f)  # first step of length 1, as scipy
        fn, gn = fun(x + t[:, None] * q, ids)
        for _ in range(max_ls):
            retry = ~(fn <= f + 1e-4 * t * gd)
            if not retry.any():
                break
            r = retry.nonzero().squeeze(1)
            t[r] *= 0.5
            fn[r], gn[r] = fun(x[r] + t[r, None] * q[r], ids[r])
        stalled = ~(fn < f)  # failed line search, or no fp32-visible decrease left
        s = t[:, None] * q
        s[stalled] = 0
        yv = gn - g
        sy = (s * yv).sum(1)
        ok = sy > 1e-10
        j = it % m
        S[j], Y[j], rho[j] = s, yv, torch.where(ok, 1 / sy.clamp_min(1e-10), 0)
        gamma = torch.where(ok, sy / (yv * yv).sum(1).clamp_min(1e-20), gamma)
        x = x + s
        f, g = torch.where(stalled, f, fn), torch.where(stalled[:, None], g, gn)
    else:
        out[ids], n_iter[ids] = x, max_iter
    return out, n_iter.cpu().numpy()


def probe_set(X: torch.Tensor, pr: Problems) -> tuple[list[np.ndarray], dict]:
    """Held-out class probabilities for every outer fold of `pr`, and solver stats."""
    logits, stats = solve(X, pr)
    probs, chosen = [], []
    for o, (_, te) in enumerate(pr.outer):
        score = torch.zeros(len(CS), device=DEV)  # sum over inner folds of mean held-out log-likelihood, per C
        for rows, ids in pr.inner[o]:
            r = torch.as_tensor(rows, device=DEV)
            logp = logits(r, ids).log_softmax(-1).clamp_min(np.log(np.finfo(np.float64).eps))  # log_loss clipping
            score += logp.gather(2, pr.y[r][:, None, None].expand(-1, len(ids), 1))[..., 0].mean(0)
        best = int(score.argmax())  # first maximum, as sklearn
        chosen.append(CS[best])
        p = logits(torch.as_tensor(te, device=DEV), [pr.refit[o][best]]).softmax(-1)[:, 0]
        probs.append(p.double().cpu().numpy())
    return probs, stats | {"C": chosen}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred, turn = p.argmax(1), y != 1
    return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, labels=range(K), average="macro", zero_division=0),
            "nll": log_loss(y, p, labels=range(K)), "brier": ((p - np.eye(K)[y]) ** 2).sum(1).mean(),
            "turn_recall": (pred[turn] == y[turn]).mean() if turn.any() else np.nan, "n": len(y), "n_turn": int(turn.sum())}


def run(version: str, out_dir: Path, rl=None, pattern: str = ".*") -> pd.DataFrame:
    """Probe every stored feature set whose '<set>/<name>' matches `pattern` (baselines always run)."""
    lab = pd.read_parquet(processed_dir(version) / "labels.parquet")
    y, groups, hard = lab.label.to_numpy(), lab.scene.to_numpy(), lab.hard.to_numpy()
    ego = torch.tensor(lab[EGO_COLS].to_numpy(np.float32), device=DEV)
    src = feature_sources(version, lab.sample_token, pattern)
    folds = outer_folds(lab)
    outer = [f for fs in folds.values() for f in fs]
    proto = [p for p, fs in folds.items() for _ in fs]
    pr = Problems(y, groups, outer)
    names = ["ego", *src, *(f"{k}+ego" for k in src if k.startswith("qwen"))]
    log.info("probing %d feature sets on %d samples; %d outer folds %s -> %d problems per set",
             len(names) + 2, len(y), len(outer), {k: len(v) for k, v in folds.items()}, pr.P)

    def load(name):  # float16 memmap rows -> float32 on the GPU; runs one set ahead in a thread
        if name == "ego":
            return ego
        f, pos = src[name.removesuffix("+ego")]
        X = torch.from_numpy(np.load(f, mmap_mode="r")[pos]).to(DEV).float()
        return torch.cat([X, ego], 1) if name.endswith("+ego") else X

    preds = {}  # (set, outer fold) -> probabilities
    for o, (tr, te) in enumerate(outer):
        prior = np.bincount(y[tr], minlength=K) + 1.0
        preds["majority", o] = np.tile(prior / prior.sum(), (len(te), 1))
        preds["ego_rule", o] = ego_rule(lab[f"yaw_rate_{N - 1}"].to_numpy()[te])
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(load, names[0])
        for i, name in enumerate(tqdm(names, desc="probe sets", dynamic_ncols=True)):
            X = nxt.result()
            if i + 1 < len(names):
                nxt = ex.submit(load, names[i + 1])
            probs, st = probe_set(X, pr)
            preds |= {(name, o): p for o, p in enumerate(probs)}
            log.debug("%s: %s", name, st)
            if st["not_converged"]:
                log.warning("%s: %d / %d problems hit max_iter=%d", name, st["not_converged"], pr.P, MAX_ITER)
            if rl is not None:
                rl.event("probe_set", set=name, **st)

    rows = []
    for name in ["majority", "ego_rule", *names]:
        for p_name in folds:
            os_ = [o for o in range(len(outer)) if proto[o] == p_name]
            te = np.concatenate([outer[o][1] for o in os_])
            p = np.concatenate([preds[name, o] for o in os_])
            for subset, m in (("all", slice(None)), ("hard", hard[te])):
                rows.append({"set": name, "protocol": p_name, "subset": subset, **metrics(y[te][m], p[m])})
    res = pd.DataFrame(rows)
    if rl is not None:
        log_layer_curves(res, rl)
    res.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.md").write_text(to_markdown(res))
    log.info("results -> %s\n%s", out_dir, to_markdown(res))
    return res


def log_layer_curves(res: pd.DataFrame, rl):
    """Qwen metric vs depth as TensorBoard curves (step = LLM layer, 0 = merger output), plus one event per row."""
    for r in res.to_dict("records"):
        rl.event("probe_result", **r)
    q = res[res.set.str.fullmatch(r"qwen\w*/(L\d+_\w+|vis_mean)(\+ego)?")]
    for r in q.to_dict("records"):
        bb, feat = r["set"].split("+")[0].split("/")
        ego = "+ego" if r["set"].endswith("+ego") else ""
        layer, pool = (0, "mean") if feat == "vis_mean" else (int(feat[1:3]), feat[4:])
        for m in ("acc", "macro_f1", "nll", "brier"):
            rl.scalar(f"probe_{r['protocol']}_{r['subset']}/{m}/{bb}_{pool}{ego}", r[m], layer)
            if pool == "mean" and layer == 0:  # vis_mean anchors both pooling curves at depth 0
                rl.scalar(f"probe_{r['protocol']}_{r['subset']}/{m}/{bb}_last{ego}", r[m], layer)


def to_markdown(res: pd.DataFrame) -> str:
    out = []
    order = list(dict.fromkeys(res.set))
    for proto in dict.fromkeys(res.protocol):
        r = res[res.protocol == proto]
        a = r[r.subset == "all"].set_index("set").loc[order]
        h = r[r.subset == "hard"].set_index("set").loc[order]
        t = pd.DataFrame({"acc": a.acc, "macro_f1": a.macro_f1, "nll": a.nll, "brier": a.brier, "turn_rec": a.turn_recall,
                          "hard_acc": h.acc, "hard_nll": h.nll, "hard_turn_rec": h.turn_recall})
        n = a.iloc[0]
        out.append(f"### {proto}: n={n.n} (turns {n.n_turn}), hard n={h.iloc[0].n} (turns {h.iloc[0].n_turn})\n\n"
                   + t.to_markdown(floatfmt=".3f") + "\n")
    return "\n".join(out)
