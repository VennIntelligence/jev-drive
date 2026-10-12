"""ROUTE-MEM (experiments/route_mem, plans/2026-10-12-route-mem-prereg.md): the route memory a driver keeps of the AlpaSim route messages it
has received, and the adapter branch that reads it.

Memory (numpy, shared by the offline row builders and the served driver, so both compute the same numbers):
  batches   every route message of the episode so far: the waypoints (rig frame of the pose at the message time, NaN padding dropped), each
            carried into the current rear-axle frame through the ego poses (local frame -> current frame, rigid)
  merge     all batches sample the same route polyline (AlpaSim builds it once per rollout and only extends it), oldest first: the merged
            polyline is the oldest batch followed by every newer batch's points that lie beyond its end (along its last segment)
  window    the ego's arc position on the merged polyline (its projection; extrapolated along the first / last segment when it lies before /
            beyond it), then N_PT = 40 points at the arc offsets OFFS = -6, -4, .., +72 m: valid where the polyline covers the offset
  features  per point [valid, x / 20, y / 20, cos(heading), sin(heading)] in the current rear-axle frame (x forward, y left), zeros where
            invalid -> DIM = 200 numbers. A first decision has only its own batch, which starts ~42 m ahead: the near points are invalid.

Branch (torch): RouteParityAdapter = parity_adapter.ParityAdapter (unchanged weights and forward on the first 20 ego numbers) + RouteHead, a
small transformer decoder whose 32 slot queries read [null, ego token, the valid route points]; its output layer starts at zero, so the
branch adds exactly 0 until trained and an init from any P2 checkpoint is that checkpoint. The fed row is ego (20) ++ route features (200);
`present` (ego[:, 0]) multiplies both biases (anchor rows stay the shipped model).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_PT, NF = 40, 5
OFFS = -6.0 + 2.0 * np.arange(N_PT)              # arc offsets (m) from the ego's projection on the merged route
DIM = N_PT * NF
I20 = int(np.flatnonzero(OFFS == 20.0)[0])       # the +20 m point (coverage flag of local_map / decision 279)
TOL = 1.0                                        # a newer batch's point extends the merged polyline when it lies > TOL m beyond its end
EGO_DIM = 20
SIDECAR = "route_mem.json"                       # next to ckpt-final.pt: the checkpoint carries the branch (its config)


def to_local(a, X):
    """Rear-axle frame points X (n, 2) of pose a = (x, y, yaw) -> local frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, np.float64).reshape(-1, 2)
    return np.c_[a[0] + c * X[:, 0] - s * X[:, 1], a[1] + s * X[:, 0] + c * X[:, 1]]


def in_ego(a, X):
    """Local frame points X (n, 2) -> rear-axle frame of pose a = (x, y, yaw)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, np.float64).reshape(-1, 2) - np.asarray(a[:2], np.float64)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def merge(batches):
    """Batches (list of (m, 2) polylines in one frame, oldest first; NaN rows dropped) -> merged (M, 2) polyline or None."""
    M = None
    for B in batches:
        B = np.asarray(B, np.float64).reshape(-1, 2)
        B = B[np.isfinite(B).all(1)]
        if not len(B):
            continue
        if M is None:
            M = B.copy()
            continue
        if len(M) < 2:                                     # a one-point memory: keep the newer batch if it is longer
            M = B.copy() if len(B) > 1 else M
            continue
        u = M[-1] - M[-2]
        u = u / max(np.hypot(*u), 1e-9)
        j = np.flatnonzero((B - M[-1]) @ u > TOL)
        if len(j):
            M = np.vstack([M, B[j[0]:]])
    return M


def window(M):
    """Merged polyline (M, 2) in the current rear-axle frame -> (N_PT, NF) features (module docstring)."""
    F = np.zeros((N_PT, NF), np.float32)
    if M is None or len(M) < 2:
        return F
    d = np.diff(M, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-6
    if not keep.any():
        return F
    M = np.vstack([M[:1], M[1:][keep]])
    d, L = d[keep], L[keep]
    u = d / L[:, None]
    cum = np.r_[0.0, np.cumsum(L)]
    t = np.clip(-(M[:-1] * u).sum(1), 0.0, L)                                 # projection of the origin on every segment
    q = M[:-1] + t[:, None] * u
    i = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    s_e = cum[i] + t[i]
    if i == 0 and t[0] <= 0.0:                                                # before the memory: extrapolate along the first segment
        s_e = float(-(M[0] * u[0]).sum())
    elif i == len(L) - 1 and t[i] >= L[i]:                                    # beyond its end: along the last segment
        s_e = cum[-1] + float(-((M[-1]) * u[-1]).sum())
    s = s_e + OFFS
    ok = (s >= 0.0) & (s <= cum[-1])
    if not ok.any():
        return F
    j = np.clip(np.searchsorted(cum, s[ok], side="right") - 1, 0, len(L) - 1)
    P = M[j] + (s[ok] - cum[j])[:, None] * u[j]
    F[ok] = np.c_[np.ones(ok.sum()), P[:, 0] / 20.0, np.clip(P[:, 1] / 20.0, -4, 4), u[j, 0], u[j, 1]].astype(np.float32)
    return F


def memory(batches_local, anchor):
    """Batches already in the local frame (oldest first) and the current rear-axle pose -> (DIM,) features."""
    M = merge(batches_local)
    return window(None if M is None else in_ego(anchor, M)).reshape(-1)


def summary(f):
    """Features (..., DIM) -> dict of numpy arrays: n valid points, nearest valid offset (nan if none), +20 m covered."""
    P = np.asarray(f).reshape(*np.shape(f)[:-1], N_PT, NF)
    v = P[..., 0] > 0.5
    first = np.where(v.any(-1), OFFS[np.argmax(v, -1)], np.nan)
    return dict(n=v.sum(-1), first=first, c20=v[..., I20])


def transform_rows(f, off):
    """Features (n, DIM) in the logged frame -> the perturbed frame of yaw-rate rows (ot_rows.off_hinge: off = (dy, dpsi) of the row's t0 pose
    in the logged frame)."""
    P = np.asarray(f, np.float32).reshape(-1, N_PT, NF).copy()
    dy, dp = np.asarray(off, np.float64)[:, :1], np.asarray(off, np.float64)[:, 1:]
    c, s = np.cos(dp), np.sin(dp)
    x, y = P[..., 1] * 20.0, P[..., 2] * 20.0 - dy
    v = P[..., 0] > 0.5
    P[..., 1] = np.where(v, (c * x + s * y) / 20.0, 0)
    P[..., 2] = np.where(v, np.clip((-s * x + c * y) / 20.0, -4, 4), 0)
    hc, hs = P[..., 3].copy(), P[..., 4].copy()
    P[..., 3] = np.where(v, c * hc + s * hs, 0)
    P[..., 4] = np.where(v, -s * hc + c * hs, 0)
    return P.reshape(-1, DIM)


class Tracker:
    """Served memory of one driver session: route messages (rig-frame waypoints at their time) and the ego poses -> features at a decision."""

    def __init__(self):
        self.routes = {}

    def add(self, ts: int, wp):
        self.routes[int(ts)] = np.asarray(wp, np.float64).reshape(-1, 2)

    def features(self, t0: int, pose_of, anchor):
        """pose_of(ts) -> (x, y, yaw) local rig pose or None; anchor = the pose at t0. Messages after t0 or without a pose are skipped."""
        B = []
        for ts in sorted(self.routes):
            if ts > t0:
                continue
            p = pose_of(ts)
            wp = self.routes[ts]
            wp = wp[np.isfinite(wp).all(1)]
            if p is not None and len(wp):
                B.append(to_local(p, wp))
        return memory(B, anchor).astype(np.float32)


# ---------------------------------------------------------------- torch branch
def _torch():
    import torch
    import torch.nn as nn
    import parity_adapter as PA

    class RouteHead(nn.Module):
        def __init__(self, d=128, layers=2, heads=4, T=32, D=512):
            super().__init__()
            self.pt = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
            self.pos = nn.Parameter(0.02 * torch.randn(N_PT, d))
            self.ego = nn.Sequential(nn.Linear(EGO_DIM - 1, d), nn.GELU(), nn.Linear(d, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.q = nn.Parameter(0.02 * torch.randn(T, d))
            layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, layers)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, ego, f):
            B, d = ego.shape[0], self.q.shape[1]
            P = f.float().view(B, N_PT, NF)
            pad = P[..., 0] < 0.5
            mem = torch.cat([self.null.expand(B, 1, d), self.ego(ego[:, 1:].float())[:, None], self.pt(P) + self.pos[None]], 1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=ego.device), pad], 1)
            h = self.dec(self.q[None].expand(B, -1, -1), mem, memory_key_padding_mask=pad)
            return self.out(self.norm(h))

    class RouteParityAdapter(PA._torch()):
        """ParityAdapter (use_ego only) + RouteHead on the route features appended to the ego row."""

        def __init__(self, **kw):
            super().__init__(use_ego=True, use_side=False)
            self.route = RouteHead(**kw)

        def forward(self, ego, side=None, side_mask=None, lead=None):
            ego = ego.float()
            e, f = ego[:, :EGO_DIM], ego[:, EGO_DIM:]
            b = super().forward(e)
            if f.shape[1] == DIM:
                b = b + self.route(e if self.in_tf is None else self.in_tf(e), f) * e[:, :1, None]
            return b

    return RouteParityAdapter


def attach(model, **kw):
    """Swap a P2 PModel's adapter for the route branch, keeping every existing adapter tensor (the branch starts at an exact zero)."""
    import torch
    old = model.adapter
    assert old is not None and old.use_ego and not old.use_side and not getattr(old, "use_lead", False), "route branch: a P2 adapter"
    dev = next(old.parameters()).device
    g = torch.random.get_rng_state()
    torch.manual_seed(1234)                                  # the branch's init draws do not touch the training streams
    new = _torch()(**kw).to(dev)
    torch.random.set_rng_state(g)
    miss = new.load_state_dict(old.state_dict(), strict=False)
    assert not miss.unexpected_keys and all(k.startswith("route.") for k in miss.missing_keys), miss
    new.in_tf = old.in_tf
    model.adapter = new
    return model


def has_branch(tag: str) -> bool:
    import pp_train as T
    return (T.proot("runs", tag) / SIDECAR).exists()


def load_pmodel(tag: str, dev):
    """pp_train.load_pmodel for a checkpoint with the route branch (its sidecar), else pp_train.load_pmodel itself."""
    import torch
    import pp_train as T
    if not has_branch(tag):
        return T.load_pmodel(tag, dev)
    kw = json.loads((T.proot("runs", tag) / SIDECAR).read_text()).get("head", {})
    ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
    m = T.PModel(ck["model"]["arm"]).to(dev)
    attach(m, **kw)
    m.load_state(ck["model"])
    return m.eval()


def write_sidecar(tag: str, head: dict, extra: dict):
    import pp_train as T
    (T.proot("runs", tag) / SIDECAR).write_text(json.dumps(dict(head=head, n_pt=N_PT, nf=NF, offs=OFFS.tolist()) | extra, indent=1))


def empty() -> np.ndarray:
    return np.zeros(DIM, np.float32)


if __name__ == "__main__":                                   # self-test: straight and curved routes, carried batches, perturbed frames
    a0 = np.array([0.0, 0.0, 0.0])
    B0 = np.c_[np.arange(42.1, 81, 4.21), np.zeros(10)]
    f = memory([B0], a0).reshape(N_PT, NF)
    assert f[OFFS < 42, 0].sum() == 0 and f[(OFFS >= 43) & (OFFS <= 72), 0].all(), f[:, 0]
    B1 = np.c_[np.arange(42.1, 81, 4.21) + 50, np.zeros(10)]
    f = memory([B0, B1], np.array([50.0, 0.0, 0.0])).reshape(N_PT, NF)
    assert f[OFFS >= -6, 0].all() and np.allclose(f[:, 1] * 20, OFFS) and np.allclose(f[:, 2], 0)
    r, th = 30.0, np.linspace(0, np.pi / 2, 200)                             # quarter circle, left turn
    C = np.c_[r * np.sin(th), r - r * np.cos(th)]
    f = memory([C], np.array([0.0, 0.0, 0.0])).reshape(N_PT, NF)
    v = f[:, 0] > 0
    assert np.allclose(np.hypot(f[v, 1] * 20, f[v, 2] * 20 - r), r, atol=0.05) and (f[v, 2] >= -1e-6).all()
    off = np.array([[0.5, 0.05]])
    g = transform_rows(memory([B0, B1], np.array([50.0, 0.0, 0.0]))[None], off).reshape(N_PT, NF)
    x, y = g[:, 1] * 20, g[:, 2] * 20                                        # back to the logged frame: x c - y s, dy + x s + y c
    c, s = np.cos(0.05), np.sin(0.05)
    assert np.allclose(0.5 + x * s + y * c, 0, atol=1e-4) and np.allclose(x * c - y * s, OFFS, atol=1e-4)
    print("rmem selftest ok", summary(memory([B0], a0)))
