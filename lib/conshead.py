"""STACK consequence head (experiments/route_follow/plans/2026-10-16-stack-prereg.md, C2 / 4.1): per-time contact of ANY candidate plan.

Inputs: front (B, 8, 32, 512) f16 the frozen Cinque tokens the policy was fed (slots oldest -> newest, 0.2 s apart; only SLOTS are read,
no gradient into them; a (B, 2, 32, 512) array already holding SLOTS is accepted too), ego (B, 20) f32 the parity_adapter ego features
(the first 20 columns of the fed row, raw: the head standardises them with its own train-set statistics), poses (B, 8, 3) or
(B, C, 8, 3) f32 rear-axle x, y (m, left +), yaw (rad, left +) at 0.5 .. 4.0 s in the row's own frame (gradient flows to the poses).
Output: logits (B, 2, 8) or (B, C, 2, 8): [agent, road] x 0.5 s bins; bin b = contact anywhere in (0.5 b, 0.5 (b + 1)] s, contact as
experiments/route_follow/scripts/rf_stack_c2.py labels it (AgentHinge signed distance < 0 to a counted box at margin 0 / side margin 0;
a Pacifica corner with road-raster sdf < 0), 0.1 s dense steps. sigmoid(logits) = contact probability.

Design: the two token slots (+ slot / index embeddings) and an ego token form a memory (one self-attention layer, encoded once per row and
shared by all C candidates); the 8 poses are 8 queries (pose b and pose b - 1 (origin for b = 0) as raw coordinates and Fourier features,
+ a time embedding) through `layers` decoder layers (self-attention over the 8, cross-attention to the memory); query b -> the 2 logits of
bin b.

    from conshead import load
    head = load()                       # frozen eval module on cuda, default $DATA_DIR/runs/route_follow/stack/cons/head.pt
    p = torch.sigmoid(head(S.front[r], S.ego[r, :20], poses))      # (B, 2, 8)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

SLOTS = (2, 7)                                          # t - 1.0 s and t (sh30_core.SLOT_T: -1.4 .. 0 s every 0.2 s)
FREQ_M = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)              # Fourier wavelengths (m) of the pose coordinates
DEFAULT = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/route_follow/stack/cons/head.pt"


def pose_features(P):
    """(..., 8, 3) -> (..., 8, F): pose b and pose b - 1 (origin first), each x / 20, y / 10, cos, sin yaw, Fourier(x, y); segment dx / 5,
    dy / 2."""
    prev = torch.cat([torch.zeros_like(P[..., :1, :]), P[..., :-1, :]], -2)
    w = 2 * torch.pi / torch.as_tensor(FREQ_M, device=P.device, dtype=P.dtype)

    def one(Q):
        x, y, h = Q[..., 0:1], Q[..., 1:2], Q[..., 2:3]
        f = torch.cat([x * w, y * w], -1)
        return torch.cat([x / 20, y / 10, torch.cos(h), torch.sin(h), torch.sin(f), torch.cos(f)], -1)
    d = P[..., :2] - prev[..., :2]
    return torch.cat([one(P), one(prev), d[..., :1] / 5, d[..., 1:2] / 2], -1)


N_PF = 2 * (4 + 4 * len(FREQ_M)) + 2


class ConsHead(nn.Module):
    def __init__(self, d=128, heads=4, layers=2, ff=512, drop=0.1):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, layers=layers, ff=ff, drop=drop)
        self.tok = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.slot = nn.Parameter(torch.zeros(len(SLOTS), 1, d))
        self.idx = nn.Parameter(torch.zeros(1, 32, d))
        self.ego = nn.Sequential(nn.Linear(20, d), nn.GELU(), nn.Linear(d, d))
        self.register_buffer("ego_mu", torch.zeros(20))
        self.register_buffer("ego_sd", torch.ones(20))
        L = lambda: nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True, activation="gelu")  # noqa: E731
        self.enc = nn.TransformerEncoder(L(), 1, enable_nested_tensor=False)
        self.pose = nn.Sequential(nn.Linear(N_PF, d), nn.GELU(), nn.Linear(d, d))
        self.ptime = nn.Parameter(torch.zeros(8, d))
        Ld = nn.TransformerDecoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(Ld, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        for p in (self.slot, self.idx, self.ptime):
            nn.init.normal_(p, std=0.02)

    def memory(self, front, ego):
        """-> (B, 1 + 2 x 32, d) the per-row memory (no gradient into the tokens)."""
        f = front.detach()
        if f.shape[1] != len(SLOTS):
            f = f[:, list(SLOTS)]
        v = self.tok(f.float()) + self.slot[None] + self.idx[None]                    # (B, 2, 32, d)
        e = self.ego((ego.float() - self.ego_mu) / self.ego_sd)[:, None]
        return self.enc(torch.cat([e, v.flatten(1, 2)], 1))

    def forward(self, front, ego, poses):
        many = poses.dim() == 4
        P = poses if many else poses[:, None]
        B, C = P.shape[:2]
        M = self.memory(front, ego)
        M = M[:, None].expand(B, C, *M.shape[1:]).reshape(B * C, *M.shape[1:])
        q = self.pose(pose_features(P.float())) + self.ptime                           # (B, C, 8, d)
        h = self.dec(q.reshape(B * C, 8, -1), M)
        o = self.out(h).view(B, C, 8, 2).transpose(-1, -2)                              # (B, C, 2, 8)
        return o if many else o[:, 0]


def load(path=None, dev="cuda") -> ConsHead:
    """The trained head as a frozen eval module (requires_grad False on every parameter; gradients still flow to the poses)."""
    ck = torch.load(Path(path or DEFAULT), map_location="cpu", weights_only=False)
    m = ConsHead(**ck["cfg"])
    m.load_state_dict(ck["state"])
    m = m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
