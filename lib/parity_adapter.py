"""Input-parity adapter for openpilot Cinque (experiments/op_parity): WA-JEPA's extra inputs into Cinque without touching its weights at init.

WA-JEPA reads 4 cameras x 4 frames at 2 Hz, `ego_status = [cmd one-hot (L/S/R), vx, vy, ax, ay]` and 4 ego poses (x, y, yaw) in the current
frame. Cinque reads only the road / wide front views. This adapter brings the rest in as ONE bias (32, 512) added to the hidden tokens of all
9 policy context frames: the slot op_adapt_l's IntentAdapter and op_route_ft's RouteAdapter use, so the serving ONNX's `intent_bias` input
(experiments/op_adapt_l/scripts/op_l_onnx.py build --bias-input) carries it unchanged in closed loop.

  ego    20 numbers (`ego_features`): [present, cmd L / S / R, vx / 10, vy / 10, ax / 3, ay / 3, 4 poses x / 10, y / 10, yaw (rad)], the
         NAVSIM AgentInput (rear-axle frame of the current pose, oldest pose first) -> 4 memory tokens through an MLP
  side   hidden tokens of CAM_L0 / CAM_R0 / CAM_B0 from Cinque's own frozen vision encoder (each camera rendered as an openpilot road + wide
         pair along its mounting yaw, image pair = (key k - 1, key k) at 2 Hz, k = 1..3) -> 3 cams x 3 times x 32 tokens, each projected to
         d plus a camera, time and token-slot embedding
  bias   32 learned slot queries, a 2-layer pre-norm transformer decoder over [ego tokens, side tokens], a linear d -> 512 that starts at
         ZERO: the untrained adapter adds exactly 0, i.e. the model is shipped Cinque. `present` (ego[:, 0]) multiplies the bias, so
         present = 0 (arm P1, "inputs zeroed") is the unconditioned model for any weights.

Arms: use_ego / use_side select the memory (P2: ego, P3: ego + side); a camera can be dropped per row with side_mask (ablation).
"""
from __future__ import annotations

import numpy as np

EGO_DIM = 20
SIDE_CAMS = ("CAM_L0", "CAM_R0", "CAM_B0")
SIDE_T = 3                                     # image pairs (key k - 1, key k), k = 1..3 of the 4 history keys (2 Hz)
N_EGO_TOK = 4


def ego_features(pose, vel, acc, cmd, present: float = 1.0) -> np.ndarray:
    """One sample (or a batch with a leading axis): pose (4, 3) x, y, yaw in the current rear-axle frame, oldest first; vel / acc (4, 2)
    body frame (the last row is t0); cmd (4,) NAVSIM one-hot [left, straight, right, unknown] of t0 (unknown -> all zero)."""
    pose, vel, acc, cmd = (np.asarray(x, np.float32) for x in (pose, vel, acc, cmd))
    lead = pose.shape[:-2]
    pr = np.full(lead + (1,), present, np.float32)
    p = np.concatenate([pose[..., :2] / 10.0, pose[..., 2:3]], -1).reshape(lead + (12,))
    return np.concatenate([pr, cmd[..., :3], vel[..., -1, :] / 10.0, acc[..., -1, :] / 3.0, p], -1).astype(np.float32)


def _torch():
    import torch
    import torch.nn as nn

    class ParityAdapter(nn.Module):
        def __init__(self, use_ego=True, use_side=True, d=256, layers=2, heads=8, n_cam=len(SIDE_CAMS), n_t=SIDE_T, T=32, D=512):
            super().__init__()
            self.use_ego, self.use_side, self.T, self.D = use_ego, use_side, T, D
            self.ego = nn.Sequential(nn.Linear(EGO_DIM - 1, d), nn.GELU(), nn.Linear(d, N_EGO_TOK * d)) if use_ego else None
            if use_side:
                self.side_in = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
                self.cam_emb = nn.Parameter(0.02 * torch.randn(n_cam, d))
                self.t_emb = nn.Parameter(0.02 * torch.randn(n_t, d))
                self.s_emb = nn.Parameter(0.02 * torch.randn(T, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))       # always-present memory token (keeps attention defined when all else is masked)
            self.q = nn.Parameter(0.02 * torch.randn(T, d))
            layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, layers)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, ego, side=None, side_mask=None):
            """ego (B, 20); side (B, n_cam, n_t, 32, 512) hidden tokens; side_mask (B, n_cam) True = camera present -> bias (B, 32, 512) fp32."""
            ego = ego.float()
            B, d = ego.shape[0], self.q.shape[1]
            mem, pad = [self.null.expand(B, 1, d)], [torch.zeros(B, 1, dtype=torch.bool, device=ego.device)]
            if self.use_ego:
                mem.append(self.ego(ego[:, 1:]).view(B, N_EGO_TOK, d))
                pad.append(torch.zeros(B, N_EGO_TOK, dtype=torch.bool, device=ego.device))
            if self.use_side and side is not None:
                _, C, Tt, T, _ = side.shape
                s = self.side_in(side.float()) + self.cam_emb[None, :C, None, None] + self.t_emb[None, None, :Tt, None] + self.s_emb[None, None, None]
                mem.append(s.reshape(B, C * Tt * T, d))
                m = torch.ones(B, C, dtype=torch.bool, device=ego.device) if side_mask is None else side_mask.bool()
                pad.append((~m)[:, :, None].expand(B, C, Tt * T).reshape(B, -1))
            h = self.dec(self.q[None].expand(B, -1, -1), torch.cat(mem, 1), memory_key_padding_mask=torch.cat(pad, 1))
            return self.out(self.norm(h)) * ego[:, :1, None]

        def apply(self, H, ego, side=None, side_mask=None):
            """H (B, 9, 32, 512) hidden tokens -> H + bias on every context frame (H's dtype)."""
            return H + self(ego, side, side_mask)[:, None].to(H.dtype)

    return ParityAdapter


def ParityAdapter(**kw):
    return _torch()(**kw)
