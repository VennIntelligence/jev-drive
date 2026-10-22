"""Usage probes on a grafted model: how much of a named feature tensor does the driving head read?

Model-agnostic measurement hooks (torch only; no repo imports, so it runs inside any model env):

  Probe(owner, attr)       wraps `owner.<attr>` (a bound method or an nn.Module, e.g. `model._encode_scene_context` or
                           `model.encoder`) whose output is the feature tensor [..., D]. Default mode is off: the wrapper
                           returns the original output object untouched, so the model is bit-identical (identity gate).
    .capture(moments, stop) accumulate token-level first / second moments of the feature (PCA fit); stop=True raises
                           Captured right after, so a fit pass skips the planner
    .sweep(call, ks, pca)  run `call()` once with the feature computed and cached (k = None, unprojected), then once per k
                           with the cached feature projected onto its top-k principal directions; the expensive producer
                           (the encoder) runs once per sample. Returns {k: call()}. Needs `call` to be deterministic given
                           the inputs (gate: `determinism_gate`).
  Moments / PCA            float64 accumulator -> eigendecomposition (mean, components, explained variance); npz on disk
  project(x, pca, k)       mu + (x - mu) U_k U_k^T in fp32, cast back to x.dtype; k = None or D keeps every direction
                           (discarded directions sit at the fit mean, not at zero)
  Projector(pca, variant)  artifact controls (opt-in; the default above is unchanged): norm restore (centred / raw), random
                           subspace, correlation-matrix PCA, planner-aligned PCA through the consumer's linear read-in
  .apply(fn)               replace the feature by fn(feature) (token pruning, slice ablations); .off() restores
  skip_blocks(blocks, idx) level (b): context manager that replaces the listed entries of an nn.ModuleList by identity

Projection unit: every row of the feature (all leading dims flattened, the last dim D) is one vector; the principal
directions live in the D-dim channel space and are shared by every token / view / time step.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

FULL = None   # k value meaning "no projection" (the unprojected cached feature)


class Captured(Exception):
    """Raised by a capturing probe with stop=True right after the feature is recorded (skips the rest of the forward)."""


@dataclass
class PCA:
    mean: np.ndarray        # (D,)
    comps: np.ndarray       # (D, D), columns = principal directions, descending variance
    var: np.ndarray         # (D,) eigenvalues, descending
    n: int                  # rows the fit saw

    @property
    def dim(self) -> int:
        return len(self.mean)

    def explained(self, k) -> float:
        """Fraction of the fit data's variance inside the top-k directions."""
        return 1.0 if k is None else float(self.var[:k].sum() / self.var.sum())

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, comps=self.comps, var=self.var, n=self.n)

    @classmethod
    def load(cls, path) -> "PCA":
        z = np.load(path)
        return cls(z["mean"], z["comps"], z["var"], int(z["n"]))


class Moments:
    """Streaming sum / sum of outer products of feature rows (float64, on the feature's device)."""

    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).double()
        if self.s is None:
            self.s = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
            self.ss = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += x.T @ x

    def state(self) -> dict:
        return dict(n=self.n, s=self.s.cpu().numpy(), ss=self.ss.cpu().numpy())

    @staticmethod
    def merge(states) -> dict:
        states = list(states)
        return dict(n=sum(int(s["n"]) for s in states), s=sum(s["s"] for s in states), ss=sum(s["ss"] for s in states))

    @staticmethod
    def fit(state: dict) -> PCA:
        n, s, ss = int(state["n"]), np.asarray(state["s"]), np.asarray(state["ss"])
        mu = s / n
        cov = (ss - n * np.outer(mu, mu)) / max(n - 1, 1)
        w, v = np.linalg.eigh(cov)
        o = np.argsort(w)[::-1]
        return PCA(mu, v[:, o], np.clip(w[o], 0, None), n)


class _Basis:
    """PCA moved to a device once; projection and residual energy per k."""

    def __init__(self, pca: PCA, device):
        self.mu = torch.as_tensor(pca.mean, dtype=torch.float32, device=device)
        self.U = torch.as_tensor(pca.comps, dtype=torch.float32, device=device)

    def project(self, x: torch.Tensor, k) -> torch.Tensor:
        if k is None:
            return x
        c = x.float() - self.mu
        U = self.U[:, :k]
        return ((c @ U) @ U.T + self.mu).to(x.dtype)

    def energy(self, x: torch.Tensor, ks) -> dict:
        """sum ||x - mu||^2 (key 'total') and sum ||x - P_k x||^2 per k over the rows of x (held-out explained variance)."""
        c = (x.detach().float() - self.mu).reshape(-1, x.shape[-1])
        z = c @ self.U                                          # coordinates in the PCA basis
        tot = float(c.pow(2).sum())
        cum = torch.cumsum(z.pow(2).sum(0), 0)
        return {"total": tot, **{k: tot - float(cum[min(k, len(cum)) - 1]) for k in ks if k is not None}}


def project(x: torch.Tensor, pca: PCA, k) -> torch.Tensor:
    return _Basis(pca, x.device).project(x, k)


VARIANTS = ("pca", "cnorm", "norm", "rand", "std", "wpca")


class Projector:
    """Projection variants for artifact controls (the default protocol is _Basis / Probe.use_pca, unchanged). Every variant
    is x' = mu + (c E_k) R_k^T with c = x - mu (row vectors), then an optional norm restore; mu = the fit's mean:
      pca    E = R = U (top-k principal directions, discarded directions at their fit mean): the original protocol
      cnorm  pca, then the centred part rescaled to the original ||x - mu|| per row (deviation energy restored)
      norm   pca, then x' rescaled to the original ||x|| per row (raw norm restored)
      rand   E = R = Q, k random orthonormal directions (QR of a Gaussian D x D, fixed seed; nested in k)
      std    PCA of the correlation matrix: standardise by the fit's per-dimension std s, project, unstandardise
             (E = Q / s, R = Q * s)
      wpca   planner-aligned: given the consumer's linear read-in W (m x D), V_k = top-k eigenvectors of W Cov W^T; x' is the
             least-squares (Gaussian conditional mean) reconstruction of x from the k coordinates V_k^T W c, so W x' = W mu +
             V_k V_k^T W c exactly (E = W^T V_k, R = Cov W^T V_k / lambda_k); k is clamped to rank(W Cov W^T)
    Linear algebra in float64 on the host, applied in fp32, cast back to x.dtype (as _Basis)."""

    def __init__(self, pca: PCA, variant: str = "pca", device="cpu", W=None, seed: int = 0):
        assert variant in VARIANTS, variant
        U, var, D = np.asarray(pca.comps, np.float64), np.asarray(pca.var, np.float64), pca.dim
        cov = (U * var) @ U.T
        if variant in ("pca", "cnorm", "norm"):
            E = R = np.asarray(pca.comps)     # as _Basis (same dtype and memory layout -> the same GEMM -> pca is bit-identical)
        elif variant == "rand":
            Q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
            E = R = Q * np.sign(np.diag(r))
        elif variant == "std":
            s = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            w, Q = np.linalg.eigh(cov / np.outer(s, s))
            Q = Q[:, np.argsort(w)[::-1]]
            E, R = Q / s[:, None], Q * s[:, None]
        else:
            W = np.asarray(W, np.float64)
            w, V = np.linalg.eigh(W @ cov @ W.T)
            o = np.argsort(w)[::-1]
            w, V = w[o], V[:, o]
            keep = w > w[0] * 1e-10
            w, V = w[keep], V[:, keep]
            E, R = W.T @ V, (cov @ W.T @ V) / w
        self.variant, self.kmax = variant, E.shape[1]
        self.renorm = {"cnorm": "centred", "norm": "raw"}.get(variant)
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.mu, self.E, self.R = f(pca.mean), f(E), f(R)

    def project(self, x: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, self.kmax)
        xf = x.float()
        c = xf - self.mu
        y = (c @ self.E[:, :k]) @ self.R[:, :k].T
        if self.renorm == "centred":
            y = y * (c.norm(dim=-1, keepdim=True) / y.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        y = y + self.mu
        if self.renorm == "raw":
            y = y * (xf.norm(dim=-1, keepdim=True) / y.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        return y.to(x.dtype)

    def stats(self, x: torch.Tensor, ks, W: torch.Tensor | None = None) -> dict:
        """Sums over the rows of x per k: rows, ||x - x'||^2, ||x - mu||^2, ||x'|| / ||x||, and with W (m x D, fp32) the
        consumer-side ||W (x - x')||^2 and ||W (x - mu)||^2 (how much of what the read-in sees the projection removes)."""
        x = x.detach().reshape(-1, x.shape[-1])
        xf = x.float()
        c = xf - self.mu
        out = {"rows": x.shape[0], "c2": float(c.pow(2).sum())}
        if W is not None:
            out["wc2"] = float((c @ W.T).pow(2).sum())
        for k in ks:
            d = xf - self.project(x, k).float()
            r = {"res2": float(d.pow(2).sum()), "ratio": float((xf - d).norm(dim=-1).div(xf.norm(dim=-1).clamp_min(1e-12)).sum())}
            if W is not None:
                r["wres2"] = float((d @ W.T).pow(2).sum())
            out[k] = r
        return out


class Probe:
    """Measurement hook on the output of `owner.<attr>`. Install with `with Probe(model, "_encode_scene_context") as p:`."""

    def __init__(self, owner, attr: str):
        self.owner, self.attr = owner, attr
        self.orig = getattr(owner, attr)
        self.mode = "off"        # off | capture | project | cached
        self.k = None
        self.moments: Moments | None = None
        self.basis: _Basis | None = None
        self.cache = None
        self.energy: dict = {}
        self.calls = 0
        self.stop = False

    # -- install / remove
    def __enter__(self):
        owner, orig, me = self.owner, self.orig, self
        if isinstance(owner, torch.nn.ModuleList) or isinstance(getattr(owner, self.attr, None), torch.nn.Module):
            mod = getattr(owner, self.attr)
            self._handle = mod.register_forward_hook(lambda m, i, o: me._after(o))
        else:
            def wrapped(*a, **kw):
                if me.mode == "cached" and me.cache is not None:
                    return me._after(None)
                return me._after(orig(*a, **kw))
            setattr(owner, self.attr, wrapped)
            self._handle = None
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
        else:
            delattr(self.owner, self.attr) if self.attr in vars(self.owner) else setattr(self.owner, self.attr, self.orig)
        return False

    def _after(self, out):
        self.calls += 1
        if self.mode == "off":
            return out
        if self.mode == "capture":
            self.moments.add(out)
            if self.stop:
                raise Captured
            return out
        if self.mode == "project":
            return self.basis.project(out, self.k)
        if self.mode == "apply":
            return self.fn(out)
        # cached: first call stores the producer's output; every call returns it projected for the current k
        if self.cache is None:
            self.cache = out
        if callable(self.k):
            return self.k(self.cache)
        return self.cache if self.k is None else self.basis.project(self.cache, self.k)

    # -- steps
    def capture(self, moments: Moments, stop: bool = False) -> "Probe":
        """Record the feature's moments on every call; stop=True raises Captured after recording (catch it around the call)."""
        self.mode, self.moments, self.stop = "capture", moments, stop
        return self

    def apply(self, fn) -> "Probe":
        """Replace the feature by fn(feature) on every call (ablations: token pruning, slice-to-mean, noise); off() undoes it."""
        self.mode, self.fn = "apply", fn
        return self

    def off(self) -> "Probe":
        self.mode = "off"
        return self

    def use_pca(self, pca: PCA, device) -> "Probe":
        self.basis = _Basis(pca, device)
        return self

    def sweep(self, call, ks, energy: bool = True, on_arm=None) -> dict:
        """{arm: call()} for arm in ks; the producer runs once per sample (cached), the rest of the model once per arm.
        An arm is None (unprojected), an int k (top-k PCA projection; needs use_pca) or a (name, fn) pair (fn(feature),
        e.g. token pruning; the result is keyed by name). on_arm(key) runs before each arm's call (e.g. to set a second
        probe that must change with the arm). With energy and a PCA, self.energy[k] accumulates the residual energy of the
        cached feature per int k (held-out explained variance)."""
        self.mode, self.cache, out = "cached", None, {}
        try:
            self.k = None
            if on_arm:
                on_arm(None)
            out[None] = call()
            ints = [k for k in ks if isinstance(k, int)]
            if energy and self.basis is not None and self.cache is not None:
                for key, v in self.basis.energy(self.cache, ints).items():
                    self.energy[key] = self.energy.get(key, 0.0) + v
            for k in ks:
                if k is None:
                    continue
                key, self.k = (k[0], k[1]) if isinstance(k, tuple) else (k, k)
                if on_arm:
                    on_arm(key)
                out[key] = call()
        finally:
            self.mode, self.cache, self.k = "off", None, None
            if on_arm:
                on_arm(None)
        return out


def explained_heldout(energy: dict) -> dict:
    """{k: 1 - residual / total} from Probe.energy (summed over shards)."""
    return {k: 1.0 - v / energy["total"] for k, v in energy.items() if k != "total"}


@contextlib.contextmanager
def skip_blocks(blocks: torch.nn.ModuleList, idx, passthrough=None):
    """Level (b): the listed blocks return their first input unchanged (residual blocks -> the layer is dropped).
    passthrough(x, *a, **kw) builds the return value when a block returns more than x (V-JEPA blocks: (x, attn) ->
    `lambda x, *a, **kw: (x, None)`)."""
    saved = {i: blocks[i].forward for i in idx}
    ident = passthrough or (lambda x, *a, **kw: x)
    try:
        for i in idx:
            blocks[i].forward = ident
        yield
    finally:
        for i, f in saved.items():
            blocks[i].forward = f


def max_abs_diff(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max()) if a.size else 0.0


def identity_gate(plain, hooked) -> dict:
    """Bit-for-bit comparison of outputs without the hook and with the hook installed but off (lists of arrays)."""
    d = [max_abs_diff(p, h) for p, h in zip(plain, hooked)]
    same = all(np.array_equal(np.asarray(p), np.asarray(h)) for p, h in zip(plain, hooked))
    return dict(n=len(d), bit_identical=bool(same), max_abs_diff=max(d) if d else 0.0)
