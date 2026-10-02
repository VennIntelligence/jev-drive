"""Thin heads on cached hidden states (vlm_thin study): fit, select on validation routes, grouped CV, emit the chosen head.

  vlm_thin_fit.py fit  --run RUN --res r4573 [--gpu 0]   every (feature, N, head) of one resolution -> fit/<res>.npz
  vlm_thin_fit.py emit --run RUN                          retrain the chosen head (selection.json) and save its weights

Features per frame (cached by `vlm_thin_stage.py extract`): `ans` = hidden state at the answer position after N language
layers; `pool` = per-image mean of the image tokens at layer N (both images concatenated); `vis` = vision-tower output
only (mean and max over the tokens of each image). Heads: `lin` = multinomial logistic regression, class-balanced, L2
(lambda picked on the validation routes); `mlp` = 2560/5120 -> 256 -> 3, GELU, dropout 0.3, AdamW, a 3-seed ensemble
(fixed hyper-parameters). Train = train routes only; validation routes pick lambda; the 7-fold route-grouped CV re-fits
with the lambda chosen on the primary split and predicts every route out of fold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_thin_common import ANS3, CACHE, CUTS, LIGHT3, LIGHTS, RES, SPLIT, light_masks, log  # noqa: E402

POOL_LAYERS = [0] + CUTS
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
MLP = dict(hidden=256, drop=0.3, lr=3e-3, wd=0.05, steps=200, seeds=3)
N_FOLDS = 7


def frames_tag(df):
    import hashlib
    return "%d-%s" % (len(df), hashlib.sha1("\n".join(df.id).encode()).hexdigest()[:8])


def load_feats(df, res):
    """Cached features of one resolution, rows aligned with df (order of frames.csv)."""
    d = CACHE / "features" / frames_tag(df) / res
    parts = [np.load(p, allow_pickle=False) for p in sorted(d.glob("shard-*.npz"))]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "hid", "pool", "vismax", "zs")}
    order = pd.Series(range(len(out["ids"])), index=out["ids"]).loc[df.id].to_numpy()
    return {k: v[order] for k, v in out.items()}


def feature(F, feat, N):
    if feat == "ans":
        return F["hid"][:, N]
    if feat == "pool":
        return F["pool"][:, POOL_LAYERS.index(N)].reshape(len(F["pool"]), -1)
    return np.concatenate([F["pool"][:, 0].reshape(len(F["pool"]), -1), F["vismax"].reshape(len(F["pool"]), -1)], 1)


def class_w(torch, y):
    n = torch.bincount(y, minlength=3).float().clamp(min=1)
    return len(y) / (3 * n)


def fit_lin(torch, X, y, lam):
    """Mean class-balanced CE + lam/2 |W|^2, LBFGS, full batch. X standardised [n, d] (cuda), y in {0,1,2}."""
    d = X.shape[1]
    W = torch.zeros(d, 3, device=X.device, requires_grad=True)
    b = torch.zeros(3, device=X.device, requires_grad=True)
    w = class_w(torch, y)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y, weight=w) + 0.5 * lam * (W ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return dict(kind="lin", W=W.detach(), b=b.detach())


def fit_mlp(torch, X, y, seed):
    """An ensemble of MLP.seeds independent MLPs trained together (batched matmuls), full-batch AdamW."""
    S, d, h = MLP["seeds"], X.shape[1], MLP["hidden"]
    g = torch.Generator(device=X.device).manual_seed(seed)
    P = {"W1": torch.randn(S, d, h, device=X.device, generator=g) / d ** 0.5, "b1": torch.zeros(S, 1, h, device=X.device),
         "W2": torch.randn(S, h, 3, device=X.device, generator=g) / h ** 0.5, "b2": torch.zeros(S, 1, 3, device=X.device)}
    for v in P.values():
        v.requires_grad_(True)
    opt = torch.optim.AdamW(list(P.values()), lr=MLP["lr"], weight_decay=MLP["wd"])
    w = class_w(torch, y)
    with torch.enable_grad():
        for _ in range(MLP["steps"]):
            opt.zero_grad()
            z = torch.nn.functional.gelu(torch.bmm(X.expand(S, -1, -1), P["W1"]) + P["b1"])
            z = torch.nn.functional.dropout(z, MLP["drop"], True)
            lg = torch.bmm(z, P["W2"]) + P["b2"]
            loss = sum(torch.nn.functional.cross_entropy(lg[s], y, weight=w) for s in range(S))
            loss.backward()
            opt.step()
    return dict(kind="mlp", **{k: v.detach() for k, v in P.items()})


def probs(torch, m, X):
    if m["kind"] == "lin":
        return torch.softmax(X @ m["W"] + m["b"], -1)
    S = m["W1"].shape[0]
    z = torch.nn.functional.gelu(torch.bmm(X.expand(S, -1, -1), m["W1"]) + m["b1"])
    return torch.softmax(torch.bmm(z, m["W2"]) + m["b2"], -1).mean(0)


def standardise(torch, Xtr_np, X_np, dev):
    mu, sd = Xtr_np.mean(0), Xtr_np.std(0) + 1e-3 * Xtr_np.std() + 1e-6
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)   # noqa: E731
    return to((X_np - mu) / sd), to(mu), to(sd)


def quick_s(pred, df, M, rows):
    """S = red recall - red answered green + green recall - no-light false alarm, plain ratios on `rows` (bool mask);
    `pred` = class idx (0 red, 1 green, 2 none) per frame."""
    def r(mask, cls):
        m = M[mask] & rows if isinstance(mask, str) else mask & rows
        return float((pred[m] == cls).mean()) if m.sum() else np.nan
    return r("red", 0) - r("red", 1) + r("green", 1) - r("nolight", 0)


def fit(a):
    import torch
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    df = pd.read_csv(Path(a.run) / "frames.csv", dtype={"route": str})
    F = load_feats(df, a.res)
    M = {k: v.to_numpy() for k, v in light_masks(df).items()}
    y = df.y.to_numpy()
    part = df.part.to_numpy()
    tr, va = (part == "train") & (y >= 0), part == "val"
    routes = sorted(df.route.unique(), key=int)
    fold = df.route.map({r: i % N_FOLDS for i, r in enumerate(routes)}).to_numpy()
    out, info = {}, {}
    ytt = torch.tensor(y, device=dev)
    specs = [("ans", N) for N in CUTS] + [("pool", N) for N in CUTS] + [("vis", 0)]
    for feat, N in specs:
        X = feature(F, feat, N)
        Xs, mu, sd = standardise(torch, X[tr], X, dev)
        # linear: lambda by validation S
        best = None
        for lam in LAMS:
            m = fit_lin(torch, Xs[torch.tensor(tr, device=dev)], ytt[torch.tensor(tr, device=dev)], lam)
            p = probs(torch, m, Xs).argmax(-1).cpu().numpy()
            s = quick_s(p, df, M, va)
            if best is None or s > best[0] + 1e-9 or (abs(s - best[0]) <= 1e-9 and lam > best[1]):
                best = (s, lam, p)
        out["pred|%s|%d|lin" % (feat, N)] = best[2].astype(np.uint8)
        info["%s|%d|lin" % (feat, N)] = dict(lam=best[1], val_S=best[0])
        # mlp
        m = fit_mlp(torch, Xs[torch.tensor(tr, device=dev)], ytt[torch.tensor(tr, device=dev)], seed=0)
        p = probs(torch, m, Xs).argmax(-1).cpu().numpy()
        out["pred|%s|%d|mlp" % (feat, N)] = p.astype(np.uint8)
        info["%s|%d|mlp" % (feat, N)] = dict(val_S=quick_s(p, df, M, va))
        # grouped CV: every route predicted out of fold
        for head in ("lin", "mlp"):
            cv = np.zeros(len(df), np.uint8)
            for k in range(N_FOLDS):
                trk, tek = (fold != k) & (y >= 0), fold == k
                Xk, _, _ = standardise(torch, X[trk], X, dev)
                tt = torch.tensor(trk, device=dev)
                mk = fit_lin(torch, Xk[tt], ytt[tt], info["%s|%d|lin" % (feat, N)]["lam"]) if head == "lin" else \
                    fit_mlp(torch, Xk[tt], ytt[tt], seed=0)
                cv[tek] = probs(torch, mk, Xk).argmax(-1).cpu().numpy()[tek]
            out["cv|%s|%d|%s" % (feat, N, head)] = cv
        log("%s %s N=%d: val S lin %.3f (lam %g) mlp %.3f" % (a.res, feat, N, info["%s|%d|lin" % (feat, N)]["val_S"],
                                                               info["%s|%d|lin" % (feat, N)]["lam"], info["%s|%d|mlp" % (feat, N)]["val_S"]))
    out["zs_pred"] = F["zs"].argmax(1).astype(np.uint8)
    out["ids"] = df.id.to_numpy()
    d = Path(a.run) / "fit"
    d.mkdir(exist_ok=True)
    np.savez(d / (a.res + ".npz"), **out)
    (d / (a.res + ".json")).write_text(json.dumps(info, indent=1))


def emit(a):
    """Retrain the chosen head (train routes only, same seed / lambda as in the fit) and save its weights."""
    import torch
    sel = json.loads((Path(a.run) / "selection.json").read_text())["chosen"]
    if sel["feat"] == "zs":                                      # the zero-shot one-pass variant has no head
        log("chosen variant is zero-shot: no head to emit")
        return
    df = pd.read_csv(Path(a.run) / "frames.csv", dtype={"route": str})
    F = load_feats(df, sel["res"])
    y, part = df.y.to_numpy(), df.part.to_numpy()
    tr = (part == "train") & (y >= 0)
    X = feature(F, sel["feat"], sel["N"])
    Xs, mu, sd = standardise(torch, X[tr], X, "cuda")
    tt = torch.tensor(tr, device="cuda")
    yt = torch.tensor(y, device="cuda")
    m = fit_lin(torch, Xs[tt], yt[tt], sel["lam"]) if sel["head"] == "lin" else fit_mlp(torch, Xs[tt], yt[tt], seed=0)
    p = probs(torch, m, Xs)
    name = "head_%s_%s_%d_%s.pt" % (sel["res"], sel["feat"], sel["N"], sel["head"])
    torch.save(dict(**m, mu=mu, sd=sd, res=sel["res"], feat=sel["feat"], N=sel["N"]), Path(a.run) / name)
    np.save(Path(a.run) / "head_cached_probs.npy", p.cpu().numpy())
    s = json.loads((Path(a.run) / "selection.json").read_text())
    s["chosen"]["head_file"] = name
    (Path(a.run) / "selection.json").write_text(json.dumps(s, indent=1))
    log("emitted " + name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "emit"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--res", default="r4573")
    a = ap.parse_args()
    {"fit": fit, "emit": emit}[a.cmd](a)
