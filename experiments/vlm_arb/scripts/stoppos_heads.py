"""Small heads of the stop-position probe in torch (plan 2026-10-03-stoppos-probe.md, Heads and readouts).

All heads take standardised features X (n, d), per-frame weights w (mean 1 over the training frames: every attempt carries the
same total weight) and return numpy outputs. Cards: only the one this process sees (CUDA_VISIBLE_DEVICES); light use.
"""
import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)
LOGIT_C = (0.01, 0.1, 1.0)


def T(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=DEV)


def norm_w(w):
    w = np.asarray(w, np.float64)
    return w * len(w) / w.sum()


class Standardizer:
    def fit(self, X):
        X = T(X)
        self.m, self.s = X.mean(0), X.std(0)
        self.s = torch.where(self.s < 1e-6, torch.ones_like(self.s), self.s)
        return self

    def __call__(self, X, chunk=20000):
        out = []
        for i in range(0, len(X), chunk):
            out.append(((T(X[i:i + chunk]) - self.m) / self.s))
        return torch.cat(out) if out else T(np.zeros((0, len(self.m))))


class Pca:
    """Randomised PCA on standardised features; scores are whitened (unit variance) so they enter the heads like other taps."""

    def __init__(self, k=512, n_fit=20000, seed=0):
        self.k, self.n_fit, self.seed = k, n_fit, seed

    def fit(self, Xs):
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        idx = torch.randperm(len(Xs), generator=g)[: self.n_fit].to(Xs.device)
        A = Xs[idx]
        self.mu = A.mean(0)
        U, S, V = torch.svd_lowrank(A - self.mu, q=self.k + 20, niter=3)
        self.V = V[:, : self.k]
        self.sd = (S[: self.k] / np.sqrt(len(A) - 1)).clamp_min(1e-6)
        return self

    def __call__(self, Xs, chunk=20000):
        return torch.cat([((Xs[i:i + chunk] - self.mu) @ self.V) / self.sd for i in range(0, len(Xs), chunk)])


# ------------------------------------------------------------------------------------------------ ridge
def ridge_fit(X, y, w, alphas=RIDGE_ALPHAS):
    """Weighted ridge with an unpenalised intercept for each alpha. Returns [(coef, intercept)] (torch)."""
    n, d = X.shape
    w = T(norm_w(w))
    y = T(y)
    A = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
    Aw = A * w[:, None]
    G, b = (A.T @ Aw).double(), (Aw.T @ y).double()
    P = torch.eye(d + 1, device=X.device, dtype=torch.float64)
    P[-1, -1] = 0
    out = []
    for a in alphas:
        beta = torch.linalg.solve(G + a * P, b).float()
        out.append((beta[:-1], beta[-1]))
    return out


def ridge_predict(X, coef):
    return (X @ coef[0] + coef[1]).cpu().numpy()


def mae_w(pred, y, w):
    w = np.asarray(w)
    return float((np.abs(pred - y) * w).sum() / w.sum())


# ------------------------------------------------------------------------------------------------ logistic
def logistic_fit(X, y, w, n_class, C, iters=150):
    """Weighted multinomial (or binary, n_class = 2) logistic regression, L2 as in sklearn: mean loss + ||W||^2 / (2 C n)."""
    n, d = X.shape
    w = T(norm_w(w))
    y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=DEV)
    W = torch.zeros(d, n_class, device=DEV, requires_grad=True)
    b = torch.zeros(n_class, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(X @ W + b, y, reduction="none") * w).mean() + (W ** 2).sum() / (2 * C * n)
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def logits(X, Wb):
    return X @ Wb[0] + Wb[1]


def nll_w(lg, y, w):
    w = T(norm_w(w))
    y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=lg.device)
    return float((torch.nn.functional.cross_entropy(lg, y, reduction="none") * w).mean())


def fit_temperature(lg, y, w):
    """Scalar T minimising the weighted NLL of softmax(lg / T) on the calibration set."""
    w = T(norm_w(w))
    y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=lg.device)
    lt = torch.zeros(1, device=lg.device, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(lg / lt.exp(), y, reduction="none") * w).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(lt.exp())


# ------------------------------------------------------------------------------------------------ quantile
def quantile_fit(X, y, w, qs=(0.1, 0.5, 0.9), steps=400, lr=0.03, wd=1e-4, seed=0):
    """Linear quantile regression heads (pinball loss, weighted), Adam, full batch; y is scaled by 10 m inside."""
    torch.manual_seed(seed)
    n, d = X.shape
    w = T(norm_w(w))
    yt = T(y) / 10.0
    q = T(qs)
    lin = torch.nn.Linear(d, len(qs)).to(DEV)
    torch.nn.init.zeros_(lin.weight)
    with torch.no_grad():
        lin.bias.copy_(torch.quantile(yt, q))
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        e = yt[:, None] - lin(X)
        loss = (torch.maximum(q * e, (q - 1) * e) * w[:, None]).mean() + wd * (lin.weight ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
    return lin


def quantile_predict(X, lin):
    with torch.no_grad():
        return (lin(X) * 10.0).cpu().numpy()


def cqr_constant(lo, hi, y, w, alpha=0.2):
    """Conformalised quantile regression correction c: [lo - c, hi + c] (weighted split conformal on the calibration set)."""
    s = np.maximum(lo - y, y - hi)
    w = np.asarray(w, float)
    o = np.argsort(s)
    c = np.cumsum(w[o]) / w.sum()
    level = min(1.0, (1 - alpha) * (1 + 1 / max(len(s), 1)))
    return float(np.interp(level, c, s[o]))


# ------------------------------------------------------------------------------------------------ MLP
class Mlp(torch.nn.Module):
    def __init__(self, d, out, hidden=256, drop=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def _loss(task, out, y, w):
    if task == "reg":
        return (torch.nn.functional.smooth_l1_loss(out[:, 0], y, reduction="none", beta=1.0) * w).mean()
    return (torch.nn.functional.cross_entropy(out, y.long(), reduction="none") * w).mean()


def mlp_fit(X, y, w, task, Xv=None, yv=None, wv=None, epochs=30, seed=0, bs=512, lr=1e-3, wd=1e-3):
    """task 'reg' (y in metres, scaled by 10 inside) or 'cls' (y integer classes). With a validation set the epoch of the best
    validation loss is returned (early stopping); without one the model after `epochs` epochs is returned."""
    torch.manual_seed(seed)
    n, d = X.shape
    ncls = int(np.max(y)) + 1 if task == "cls" else 1
    ncls = max(ncls, 2) if task == "cls" else 1
    m = Mlp(d, ncls).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    scale = 10.0 if task == "reg" else 1.0
    yt, wt = T(y) / scale, T(norm_w(w))
    if Xv is not None:
        yvt, wvt = T(yv) / scale, T(norm_w(wv))
    best, best_ep, state = np.inf, epochs, None
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            _loss(task, m(X[idx]), yt[idx], wt[idx]).backward()
            opt.step()
        if Xv is not None:
            m.eval()
            with torch.no_grad():
                v = float(_loss(task, m(Xv), yvt, wvt))
            if v < best:
                best, best_ep = v, ep
    m.eval()
    return m, best_ep


def mlp_predict(m, X, task, chunk=20000):
    with torch.no_grad():
        out = torch.cat([m(X[i:i + chunk]) for i in range(0, len(X), chunk)])
    return (out[:, 0] * 10.0).cpu().numpy() if task == "reg" else torch.softmax(out, 1).cpu().numpy()
