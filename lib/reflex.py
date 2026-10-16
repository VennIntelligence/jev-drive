"""STACK reflex module (experiments/route_follow/plans/2026-10-16-stack-prereg.md 4.5; user design 2026-10-16 11:55 JST): a body / self component
inside the model that exchanges with the route memory and corrects the near-field action. Not a predictor of futures.

Self-state (efference copy), numpy, one function for the offline rows and the served drivers (selfstate()):
  the previous decision's served plan (8 rear-axle poses at 0.5 .. 4 s in the previous decision's frame) is carried into the current
  rear-axle frame by the measured ego motion (rel = the current pose in the previous frame; the same rigid transform as intent_hold.py),
  with the previous decision's own origin as its t = 0 pose; times are re-based to now (t_prev - dt). Features (SF = 14):
    [valid, for tau in 0, 0.5, 1.0, 1.5 s: (x_tau - v tau) / 2, y_tau, yaw_tau (the intent's pose at now + tau in the current frame; tau = 0 is
     the deviation of the actual state from the previous intent), (v_intent(now) - v) / 5]; clipped to +-5 (x, y) / +-1 rad (yaw) / +-2;
  zeros (valid 0) without a previous served plan (first decision, human / navtrain rows).
  The fed row is ego (20) ++ route features (rmem DIM / DIM_O) ++ self-state (SF).

Module (torch, ReflexAdapter = rmem's RouteParityAdapter + Self):
  token     s = MLP(SF) (d 128)
  exchange  s reads the route branch's memory tokens [null, ego token, route points] (one pre-norm cross-attention + FFN block); the route
            branch's 32 slot queries read [null, s] (one cross-attention, zero-initialised output map) -> a bias (B, 32, 512) added to the
            route bias on the 9 context frames (the route branch's carrier)
  reflex    6 numbers 0.5 tanh(W s) = lateral / longitudinal residual at 0.5 / 1.0 / 1.5 s (|.| <= 0.5 m), x a zero-initialised scalar gate;
            on the plan's 33 points: 0 at t = 0, linear between the knots, the 1.5 s value from 1.5 s on; added to the plan mean's x (ch 0)
            and y (ch 1, right +, so left residual enters with -) in ReflexPModel.forward
  valid     every output of the module x valid x present: rows without a self-state get exactly the base model. With the zero-initialised
            output map and gate the module adds exact zeros at step 0 (the model is its init checkpoint).
  serving   JEV_REFLEX_OFF=1: the driver feeds valid 0 (the module's switch-off; same weights).
"""
from __future__ import annotations

import os

import numpy as np

SF = 14
TAUS = (0.0, 0.5, 1.0, 1.5)
KNOTS = (0.5, 1.0, 1.5)
RES_MAX = 0.5
OFF = os.environ.get("JEV_REFLEX_OFF", "") not in ("", "0")
SIDE_KEY = "reflex"                                         # key in rmem's sidecar route_mem.json: {"sf": 14, "d": 128}


def empty() -> np.ndarray:
    return np.zeros(SF, np.float32)


def selfstate(prev_served, rel, dt: float, v: float) -> np.ndarray:
    """prev_served (8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in the previous decision's frame (None: no previous plan); rel (3,) the current
    rear-axle pose (x, y, yaw) in the previous decision's frame; dt (s) since the previous decision; v the measured speed now -> (SF,) f4."""
    if prev_served is None or not np.isfinite(np.asarray(prev_served, np.float64)).all() or not (0 < dt <= 2.5):
        return empty()
    P = np.r_[[[0.0, 0.0, 0.0]], np.asarray(prev_served, np.float64).reshape(8, 3)]
    tp = np.r_[0.0, 0.5 * np.arange(1, 9)] - float(dt)                 # times relative to now
    x0, y0, a0 = (float(q) for q in rel)
    c, s = np.cos(a0), np.sin(a0)
    dx, dy = P[:, 0] - x0, P[:, 1] - y0
    X, Y = c * dx + s * dy, -s * dx + c * dy
    A = np.unwrap(P[:, 2]) - a0
    out = [1.0]
    for tau in TAUS:
        x, y, a = (float(np.interp(tau, tp, q)) for q in (X, Y, A))
        a = (a + np.pi) % (2 * np.pi) - np.pi
        out += [np.clip((x - v * tau) / 2.0, -5, 5), np.clip(y, -5, 5), np.clip(a, -1, 1)]
    j = int(np.clip(np.searchsorted(tp, 0.0, side="right") - 1, 0, 7))   # segment holding now
    vi = np.hypot(X[j + 1] - X[j], Y[j + 1] - Y[j]) / 0.5
    out.append(np.clip((vi - v) / 5.0, -2, 2))
    return np.asarray(out, np.float32)


def basis(t_idx) -> np.ndarray:
    """(33,) plan times -> (33, 3) weights of the three knot residuals (0 at t = 0, linear between knots, held after the last)."""
    t = np.asarray(t_idx, np.float64)
    k = np.r_[0.0, KNOTS]
    B = np.zeros((len(t), len(KNOTS)))
    for j in range(len(KNOTS)):
        e = np.zeros(len(k))
        e[j + 1] = 1.0
        B[:, j] = np.interp(t, k, e, right=e[-1])
    return B


def _torch():
    import torch
    import torch.nn as nn
    import rmem as RM

    class Self(nn.Module):
        def __init__(self, d=128, heads=4, T=32, D=512):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(SF - 1, d), nn.GELU(), nn.Linear(d, d))
            self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            self.read = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.slot = nn.Parameter(0.02 * torch.randn(T, d))
            self.give = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.res = nn.Linear(d, 2 * len(KNOTS))
            self.gate = nn.Parameter(torch.zeros(1))

        def forward(self, sf, mem, pad):
            """sf (B, SF); mem (B, M, d) the route branch's memory tokens, pad (B, M) -> context bias (B, 32, 512), residual (B, 6)."""
            B, d = sf.shape[0], self.slot.shape[1]
            v = sf[:, :1]
            s = self.inp(sf[:, 1:]) * v
            q = self.n1(s)[:, None]
            s = s + self.read(q, mem, mem, key_padding_mask=pad, need_weights=False)[0][:, 0]
            s = s + self.ffn(self.n2(s))
            kv = torch.cat([self.null.expand(B, 1, d), self.n3(s)[:, None]], 1)
            h = self.slot[None].expand(B, -1, -1)
            h = h + self.give(h, kv, kv, need_weights=False)[0]
            bias = self.out(self.norm(h)) * v[:, :, None]
            res = RES_MAX * torch.tanh(self.res(s)) * self.gate * v
            return bias, res

    Base = RM._torch()

    class ReflexAdapter(Base):
        """rmem.RouteParityAdapter + Self on the self-state appended after the route features. last_res: the residual of the last call."""

        def __init__(self, rx=None, **kw):
            super().__init__(**kw)
            self.rx = Self(**(rx or {}))
            self.last_res = None

        def memory_tokens(self, e, f):
            r = self.route
            B, d = e.shape[0], r.q.shape[1]
            if r.dim == RM.DIM:
                P = f.float().view(B, RM.N_PT, RM.NF)
                et = r.ego(e[:, 1:].float())
            else:
                P = f.float()[:, :r.n_pt * RM.NF].reshape(B, r.n_pt, RM.NF)
                et = r.ego(e[:, 1:].float()) + r.sc(f.float()[:, r.n_pt * RM.NF:])
                if r.latfeat:
                    et = et + r.lat(RM.lat_feats(P))
            mem = torch.cat([r.null.expand(B, 1, d), et[:, None], r.pt(P) + r.pos[None]], 1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=e.device), P[..., 0] < 0.5], 1)
            return mem, pad

        def forward(self, ego, side=None, side_mask=None, lead=None):
            ego = ego.float()
            nr = RM.EGO_DIM + self.route.dim
            assert ego.shape[1] == nr + SF, f"reflex adapter: fed row {ego.shape[1]} != {nr} + {SF}"
            b = super().forward(ego[:, :nr])
            e = ego[:, :RM.EGO_DIM]
            e_in = e if self.in_tf is None else self.in_tf(e)
            mem, pad = self.memory_tokens(e_in, ego[:, RM.EGO_DIM:nr])
            bias, res = self.rx(ego[:, nr:], mem, pad)
            pres = e[:, :1]
            self.last_res = res * pres
            return b + bias * pres[:, :, None]

    return ReflexAdapter


def attach(model, rx=None):
    """A PModel with the route branch (rmem.attach done) -> the reflex adapter (every tensor kept; Self at its init) and ReflexPModel."""
    import torch
    old = model.adapter
    dev = next(old.parameters()).device
    head = dict(d=old.route.q.shape[1], layers=len(old.route.dec.layers), heads=old.route.dec.layers[0].self_attn.num_heads,
                order=old.route.dim != __import__("rmem").DIM, latfeat=bool(getattr(old.route, "latfeat", False)))
    g = torch.random.get_rng_state()
    torch.manual_seed(4321)                                  # the module's init draws do not touch the training streams
    new = _torch()(rx=rx, **head).to(dev)
    torch.random.set_rng_state(g)
    miss = new.load_state_dict(old.state_dict(), strict=False)
    assert not miss.unexpected_keys and all(k.startswith("rx.") for k in miss.missing_keys), miss
    new.in_tf = old.in_tf
    model.adapter = new
    _wrap(model)
    return model


def _wrap(model):
    import torch
    from jevdrive import op_adapt as A
    s = model.net.slices["plan"].start
    B = torch.as_tensor(basis(A.T_IDXS), dtype=torch.float32)
    ix = torch.as_tensor([s + 15 * i for i in range(33)])
    iy = ix + 1
    cls = model.__class__

    class ReflexPModel(cls):
        def forward(self, *a, **k):
            self.adapter.last_res = None
            o = cls.forward(self, *a, **k)
            r = self.adapter.last_res
            if r is None:                                    # inputs_on False / no adapter call: the base output
                return o
            out, tap = (o if isinstance(o, tuple) else (o, None))
            Bd = B.to(r.device)
            lat, lon = r[:, :3] @ Bd.T, r[:, 3:] @ Bd.T       # (B, 33)
            add = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
            add[:, ix.to(out.device)] = lon
            add[:, iy.to(out.device)] = -lat                  # plan y is right-positive
            out = (out.float() + add).to(out.dtype)
            return (out, tap) if tap is not None else out
    model.__class__ = ReflexPModel
    model.reflex = True


def params(model):
    return list(model.adapter.rx.parameters()) if getattr(model, "reflex", False) else []


def selftest():
    import math
    P = np.c_[np.arange(1, 9) * 5.0, np.zeros(8), np.zeros(8)]       # 10 m/s straight
    f = selfstate(P, (5.0, 0.0, 0.0), 0.5, 10.0)                     # moved exactly as intended
    assert abs(f[0] - 1) < 1e-9 and np.abs(f[1:]).max() < 1e-6, f
    f = selfstate(P, (5.0, 0.3, 0.0), 0.5, 10.0)                     # 0.3 m left of the intent: the intent lies 0.3 m right
    assert all(abs(f[2 + 3 * i] + 0.3) < 1e-6 for i in range(4)), f
    f = selfstate(P, (5.0, 0.0, 0.1), 0.5, 10.0)                     # heading 0.1 rad left of the intent
    assert abs(f[3] + 0.1) < 1e-6 and f[2 + 3] < 0, f
    assert not selfstate(None, (0, 0, 0), 0.5, 1.0).any()
    Bm = basis(np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0]))
    assert np.allclose(Bm, [[0, 0, 0], [.5, 0, 0], [1, 0, 0], [.5, .5, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]), Bm
    return math.isfinite(float(f.sum()))


if __name__ == "__main__":
    print("reflex selftest", selftest())
