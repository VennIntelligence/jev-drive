"""CRITIC-PILOT critic (experiments/route_follow/plans/2026-10-13-critic-pilot-prereg.md, "Critic"): a small transformer that reads one
decision's inputs and a candidate plan and predicts the closed-loop consequence of that decision.

Inputs (batch b): front (b, 8, 32, 512) the frozen Cinque tokens the policy saw (None for the blind arm), ego (b, 20) the adapter's ego vector
(standardised by the caller), rmem (b, 200) the route memory in rmem.py's layout (40 points x [valid, x / 20, y / 20, cos, sin]), plan (b, 8, 3)
rear-axle x, y (m), yaw (rad) in the own frame (differentiable: the gradient sanity read takes d risk / d plan).
Outputs: dict ev (b, 3, 2) logits collision / offroad / corridor x 2 / 4 s, anyz (b, 2) logits, spin (b,) logit, lat2 (b,) m, prog4 (b,) m,
wzpk (b,) rad/s.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PT, NF = 40, 5


class Critic(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, ff=1024, vision=True, route=True):
        super().__init__()
        self.vision, self.route = vision, route
        if vision:
            self.tok = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
            self.slot = nn.Parameter(torch.zeros(8, 1, d))
            self.idx = nn.Parameter(torch.zeros(1, 32, d))
        self.ego = nn.Linear(20, d)
        self.rpt = nn.Linear(NF - 1, d)
        self.roff = nn.Parameter(torch.zeros(N_PT, d))
        self.plan = nn.Linear(4, d)
        self.ptime = nn.Parameter(torch.zeros(8, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.type = nn.Parameter(torch.zeros(6, d))                # cls, ego, null, route, plan, vision
        for p in (self.slot, self.idx, self.roff, self.ptime, self.cls, self.null, self.type):
            nn.init.normal_(p, std=0.02)
        L = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(L, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 6 + 2 + 1 + 3)

    def forward(self, front, ego, rmem, plan):
        b = ego.shape[0]
        T = self.type
        xs = [self.cls.expand(b, 1, -1) + T[0], (self.ego(ego) + T[1])[:, None], self.null.expand(b, 1, -1) + T[2]]
        R = rmem.view(b, N_PT, NF)
        valid = R[..., 0] > 0.5
        if not self.route:
            valid = torch.zeros_like(valid)
        xs.append(self.rpt(R[..., 1:]) + self.roff + T[3])
        pf = torch.stack([plan[..., 0] / 20, plan[..., 1] / 20, torch.cos(plan[..., 2]), torch.sin(plan[..., 2])], -1)
        xs.append(self.plan(pf) + self.ptime + T[4])
        mask = [torch.zeros(b, 3, dtype=torch.bool, device=ego.device), ~valid, torch.zeros(b, 8, dtype=torch.bool, device=ego.device)]
        if self.vision:
            v = self.tok(front.float())                                            # (b, 8, 32, d)
            v = (v + self.slot[None] + self.idx[None]).reshape(b, 256, -1) + T[5]
            xs.append(v)
            mask.append(torch.zeros(b, 256, dtype=torch.bool, device=ego.device))
        h = self.enc(torch.cat(xs, 1), src_key_padding_mask=torch.cat(mask, 1))
        o = self.head(self.norm(h[:, 0])).float()
        return dict(ev=o[:, :6].view(b, 3, 2), anyz=o[:, 6:8], spin=o[:, 8], lat2=o[:, 9] * 2.0, prog4=o[:, 10] * 10.0 + 20.0,
                    wzpk=o[:, 11] * 0.2 + 0.1)


def masked_bce(logit, y):
    m = torch.isfinite(y)
    if not m.any():
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[m], y[m])


def masked_huber(p, y, delta):
    m = torch.isfinite(y)
    if not m.any():
        return p.sum() * 0.0
    return F.huber_loss(p[m] / delta, y[m] / delta, delta=1.0)


def loss(out, lab):
    """lab: dict of tensors ev (b, 3, 2), anyz (b, 2), spin, lat2, prog4, wzpk (nan = masked)."""
    parts = dict(ev=masked_bce(out["ev"], lab["ev"]), anyz=masked_bce(out["anyz"], lab["anyz"]), spin=masked_bce(out["spin"], lab["spin"]),
                 lat2=masked_huber(out["lat2"], lab["lat2"].clamp(-10, 10), 1.0), prog4=masked_huber(out["prog4"], lab["prog4"], 5.0),
                 wzpk=masked_huber(out["wzpk"], lab["wzpk"], 0.2))
    return parts["ev"] + parts["anyz"] + 0.5 * parts["spin"] + parts["lat2"] + 0.5 * parts["prog4"] + 0.5 * parts["wzpk"], parts


def signed_off(P, q):
    """Polylines P (b, n, 2) (NaN rows allowed only as whole-row padding at the ends is not supported: pass finite polylines), points
    q (b, 2) -> signed lateral offset (b,) of q from P (nearest segment, left +). Torch, vectorised (used to relabel lat2 under a route shift)."""
    A, B = P[:, :-1], P[:, 1:]
    d = B - A
    L2 = (d ** 2).sum(-1).clamp_min(1e-9)
    u = (((q[:, None] - A) * d).sum(-1) / L2).clamp(0, 1)
    C = A + u[..., None] * d
    r = q[:, None] - C
    j = (r ** 2).sum(-1).argmin(1)
    ar = torch.arange(len(q), device=q.device)
    rj, dj = r[ar, j], d[ar, j]
    return torch.sign(dj[:, 0] * rj[:, 1] - dj[:, 1] * rj[:, 0]) * rj.norm(dim=-1)
