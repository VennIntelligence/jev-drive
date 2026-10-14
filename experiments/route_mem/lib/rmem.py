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

Ordered layout (route v2, experiments/route_follow/plans/2026-10-15-route-v2-prereg.md, rm_train --rm-order; sidecar head `order: true`):
  points    the same window (arc offsets from the ego's projection on the merged route, 2 m), cut at +40 m: OFFS_O = -6, -4, .., +40
            (N_PT_O = 24; 3 points behind the ego). Points beyond +40 m are dropped (no slot), not masked.
  scalars   [d_turn / 60, d_turn known, min(trav, 200) / 100] appended: d_turn = arc offset of the first valid point in 0 .. +60 m whose heading
            differs >= 30 deg from the reference heading (and still differs 4 m on, or the memory ends there: a merge kink is not a turn) (the route's at offset 0 when that point is valid, else the ego's own heading, i.e.
            the near gap of the first decisions); none found: 60 when the +60 m point is valid, else unknown (value 0, bit 0). trav = arc
            length the ego has travelled since the first route message (chords between the poses of the messages up to t0, then to the
            anchor). DIM_O = 24 x 5 + 3 = 123. The ordered features are a pure function of the old window and trav (to_order), so the
            offline rows, the served Tracker and the trainer's on-the-fly negatives compute the same numbers.
  branch    RouteHead(order=True): 24 point tokens (position embeddings = the first 24 of an old head's), the scalars enter through a
            zero-initialised linear map added to the ego token (an old head converted by order_state() therefore changes only by the
            dropped far points).
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
I0 = int(np.flatnonzero(OFFS == 0.0)[0])         # the ego's projection
I60 = int(np.flatnonzero(OFFS == 60.0)[0])
N_PT_O = int(np.flatnonzero(OFFS == 40.0)[0]) + 1  # ordered layout: -6 .. +40 m (24 points)
OFFS_O = OFFS[:N_PT_O]
NSC = 3                                          # d_turn / 60, d_turn known, travelled / 100
DIM_O = N_PT_O * NF + NSC
TURN_DEG, TURN_CAP, TRAV_CAP = 30.0, 60.0, 200.0
SERVE_ORDER = False                              # Tracker() default in a driver: set from the served checkpoint's sidecar (head_order)


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


def pts(f):
    """Features of either layout (..., DIM | DIM_O) -> the point block (..., N_PT | N_PT_O, NF)."""
    f = np.asarray(f)
    n = N_PT if f.shape[-1] == DIM else N_PT_O
    assert f.shape[-1] in (DIM, DIM_O), f.shape
    return f[..., :n * NF].reshape(*f.shape[:-1], n, NF)


def summary(f):
    """Features (..., DIM | DIM_O) -> dict of numpy arrays: n valid points, nearest valid offset (nan if none), +20 m covered."""
    P = pts(f)
    v = P[..., 0] > 0.5
    first = np.where(v.any(-1), OFFS[np.argmax(v, -1)], np.nan)
    return dict(n=v.sum(-1), first=first, c20=v[..., I20])


def turn_dist(f):
    """Old-layout features (..., DIM) -> (...,) arc offset (m) of the next heading change >= TURN_DEG within 0 .. +60 m (module docstring),
    TURN_CAP when none and +60 m is covered, nan when unknown."""
    P = np.asarray(f, np.float32).reshape(*np.shape(f)[:-1], N_PT, NF)
    v = P[..., 0] > 0.5
    h = np.arctan2(P[..., 4], P[..., 3])
    ref = np.where(v[..., I0], h[..., I0], 0.0)
    dh = np.abs((h - ref[..., None] + np.pi) % (2 * np.pi) - np.pi)
    c = v & (dh >= np.radians(TURN_DEG))
    c[..., :-2] &= c[..., 2:] | ~v[..., 2:]                                    # persistent over 4 m (a merge kink is one segment, two points)
    c[..., :I0] = False
    c[..., I60 + 1:] = False
    return np.where(c.any(-1), OFFS[np.argmax(c, -1)], np.where(v[..., I60], TURN_CAP, np.nan))


def to_order(f, trav):
    """Old-layout features (..., DIM) and travelled arc (...,) -> ordered features (..., DIM_O) float32."""
    f = np.asarray(f, np.float32)
    d = turn_dist(f)
    ok = np.isfinite(d)
    tr = np.minimum(np.nan_to_num(np.asarray(trav, np.float32)), TRAV_CAP) / 100.0
    sc = np.stack(np.broadcast_arrays(np.where(ok, d / TURN_CAP, 0.0), ok.astype(np.float32), tr), -1)
    return np.concatenate([f[..., :N_PT_O * NF], sc], -1).astype(np.float32)


def to_order_t(f, trav):
    """torch twin of to_order (the trainer's on-the-fly negatives): f (n, DIM), trav (n,) -> (n, DIM_O)."""
    import torch
    P = f.float().view(-1, N_PT, NF)
    v = P[..., 0] > 0.5
    h = torch.atan2(P[..., 4], P[..., 3])
    ref = torch.where(v[:, I0], h[:, I0], torch.zeros_like(h[:, I0]))
    dh = ((h - ref[:, None] + np.pi) % (2 * np.pi) - np.pi).abs()
    c = v & (dh >= float(np.radians(TURN_DEG)))
    c[:, :-2] &= c[:, 2:] | ~v[:, 2:]
    c[:, :I0] = False
    c[:, I60 + 1:] = False
    offs = torch.as_tensor(OFFS, dtype=torch.float32, device=f.device)
    first = offs[c.int().argmax(-1)]
    cap = torch.full_like(first, TURN_CAP)
    d = torch.where(c.any(-1), first, torch.where(v[:, I60], cap, torch.full_like(first, float("nan"))))
    ok = torch.isfinite(d)
    sc = torch.stack([torch.where(ok, d / TURN_CAP, torch.zeros_like(d)), ok.float(), trav.float().nan_to_num().clamp(max=TRAV_CAP) / 100.0], -1)
    return torch.cat([f.float()[:, :N_PT_O * NF], sc], -1)


def travelled(poses, anchor=None):
    """xy (m, 2) of the route messages' poses so far (oldest first) [+ the anchor pose] -> sum of the chords (m)."""
    Q = [np.asarray(q, np.float64)[:2] for q in poses] + ([np.asarray(anchor, np.float64)[:2]] if anchor is not None else [])
    return float(np.hypot(*np.diff(np.array(Q), axis=0).T).sum()) if len(Q) > 1 else 0.0


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

    def __init__(self, order=None):
        self.routes = {}
        self.order = SERVE_ORDER if order is None else bool(order)

    def add(self, ts: int, wp):
        self.routes[int(ts)] = np.asarray(wp, np.float64).reshape(-1, 2)

    def features(self, t0: int, pose_of, anchor):
        """pose_of(ts) -> (x, y, yaw) local rig pose or None; anchor = the pose at t0. Messages after t0 or without a pose are skipped."""
        B, Q = [], []
        for ts in sorted(self.routes):
            if ts > t0:
                continue
            p = pose_of(ts)
            wp = self.routes[ts]
            wp = wp[np.isfinite(wp).all(1)]
            if p is not None:
                Q.append(np.asarray(p, np.float64)[:2])
            if p is not None and len(wp):
                B.append(to_local(p, wp))
        f = memory(B, anchor).astype(np.float32)
        return to_order(f, travelled(Q, anchor) if Q else 0.0) if self.order else f

    def polyline(self, t0: int, pose_of, anchor):
        """The merged route polyline (M, 2) in the rear-axle frame of anchor (same batches as `features`), or None
        (experiments/alpasim/lib/route_path.py: JEV_ROUTE serves this path)."""
        B = []
        for ts in sorted(self.routes):
            p, wp = (pose_of(ts) if ts <= t0 else None), self.routes[ts]
            wp = wp[np.isfinite(wp).all(1)]
            if p is not None and len(wp):
                B.append(to_local(p, wp))
        M = merge(B)
        return None if M is None else in_ego(anchor, M)


# ---------------------------------------------------------------- torch branch
def _torch():
    import torch
    import torch.nn as nn
    import parity_adapter as PA

    class RouteHead(nn.Module):
        def __init__(self, d=128, layers=2, heads=4, T=32, D=512, order=False):
            super().__init__()
            self.n_pt = N_PT_O if order else N_PT
            self.dim = DIM_O if order else DIM
            self.pt = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
            self.pos = nn.Parameter(0.02 * torch.randn(self.n_pt, d))
            self.ego = nn.Sequential(nn.Linear(EGO_DIM - 1, d), nn.GELU(), nn.Linear(d, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.q = nn.Parameter(0.02 * torch.randn(T, d))
            layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, layers)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            if order:                                        # the scalars: zero-initialised, added to the ego token
                self.sc = nn.Linear(NSC, d)
                nn.init.zeros_(self.sc.weight)
                nn.init.zeros_(self.sc.bias)

        def forward(self, ego, f):
            B, d = ego.shape[0], self.q.shape[1]
            if self.n_pt == N_PT:
                P = f.float().view(B, N_PT, NF)
                e = self.ego(ego[:, 1:].float())
            else:
                P = f.float()[:, :self.n_pt * NF].reshape(B, self.n_pt, NF)
                e = self.ego(ego[:, 1:].float()) + self.sc(f.float()[:, self.n_pt * NF:])
            pad = P[..., 0] < 0.5
            mem = torch.cat([self.null.expand(B, 1, d), e[:, None], self.pt(P) + self.pos[None]], 1)
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
            if f.shape[1] == self.route.dim:
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


def order_state(sd: dict) -> dict:
    """An old-layout adapter state (with `route.*`) -> the ordered head's: position embeddings of the first N_PT_O points, scalar map at 0."""
    import torch
    out = dict(sd)
    k = next(k for k in sd if k.endswith("route.pos"))
    if sd[k].shape[0] == N_PT_O:
        return out
    out[k] = sd[k][:N_PT_O].clone()
    pre = k[:-len("pos")]
    out[pre + "sc.weight"] = torch.zeros(sd[k].shape[1], NSC, dtype=sd[k].dtype)
    out[pre + "sc.bias"] = torch.zeros(sd[k].shape[1], dtype=sd[k].dtype)
    return out


def head_order(tag: str) -> bool:
    import pp_train as T
    p = T.proot("runs", tag) / SIDECAR
    return bool(p.exists() and json.loads(p.read_text()).get("head", {}).get("order", False))


def dim_of(tag: str) -> int:
    return DIM_O if head_order(tag) else DIM


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
    o = to_order(memory([C], np.array([0.0, 0.0, 0.0]))[None], np.array([0.0]))[0]           # quarter circle r 30 m: 30 deg at arc ~15.7 m
    assert o.shape == (DIM_O,) and abs(o[-3] * TURN_CAP - 16.0) < 1e-6 and o[-2] == 1 and o[-1] == 0, o[-3:]
    o = to_order(memory([B0], a0)[None], np.array([12.5]))[0]                                # gap: first point at +43 m, straight, 60 m covered
    assert o[-3] == 1.0 and o[-2] == 1 and abs(o[-1] - 0.125) < 1e-6 and (pts(o)[:, 0] == 0).all()
    assert np.isnan(turn_dist(np.zeros(DIM))) and (to_order(np.zeros((2, DIM)), np.zeros(2))[:, -3:] == 0).all()
    tr = Tracker(order=True)
    tr.add(0, B0)
    tr.add(1, B0)
    ps = {0: (0.0, 0.0, 0.0), 1: (3.0, 4.0, 0.0)}
    o = tr.features(1, ps.get, np.array([3.0, 4.0, 0.0]))
    assert o.shape == (DIM_O,) and abs(o[-1] - 0.05) < 1e-6, o[-3:]
    assert abs(travelled([(0, 0), (3, 4)], (6, 8)) - 10.0) < 1e-9
    try:
        import torch
        F = np.stack([memory([C], np.array([0.0, 0.0, 0.0])), memory([B0], a0), memory([B0, B1], np.array([50.0, 0.0, 0.0])), np.zeros(DIM, np.float32)])
        tv = np.array([0.0, 12.5, 50.0, 3.0], np.float32)
        assert np.allclose(to_order_t(torch.as_tensor(F), torch.as_tensor(tv)).numpy(), to_order(F, tv), atol=1e-6)
        print("to_order_t == to_order")
    except ImportError:
        pass
    print("rmem selftest ok", summary(memory([B0], a0)))
