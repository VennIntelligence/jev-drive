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

Lane TR1 (experiments/op_parity/plans/2026-10-10-tr1-prereg.md), both off by default (the adapter is then the one above, bit for bit):
  in_tf     a transform of the ego row applied inside forward (`ego_transform`), so every caller (training, jevdrive.bench, the WOD bias
            export, the AlpaSim cores) feeds the adapter the same reduced input: `noax` zeroes the longitudinal acceleration, `hist_cv`
            re-spaces the 3 history poses along their own polyline at the current speed (the path shape and yaw stay, the speed history goes)
  use_lead  one more memory token from `lead_features`: the frozen base model's own lead outputs (3 lead probabilities, distance, speed and
            acceleration of the first lead at t = 0) and the fed speed. The token MLP is created last, so the other weights of a seed are
            initialised as without it; it is warm-started from a pre-training with its own head (scripts/tr1.py tok; decision 204).
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


LEAD_DIM = 7
LEAD_AT = (0, 2, 3)                            # x, v, a of lead 0 at t = 0 in the raw `lead` output (first 72 numbers = 3 leads x 6 times x (x, y, v, a))


def lead_features(xva, lp, vx):
    """Torch. xva (B, 3) raw lead 0 distance (m, from the camera), speed (m/s), acceleration at t = 0; lp (B, 3) lead_prob logits;
    vx (B,) the fed ego speed / 10 (ego[:, 4]) -> (B, 7) [p0, p1, p2, x / 50, v / 10, a / 3, vx / 10]."""
    import torch
    xva, lp = xva.float(), lp.float()
    return torch.cat([torch.sigmoid(lp.clamp(-11, 11)), xva[:, :1] / 50.0, xva[:, 1:2] / 10.0, xva[:, 2:3] / 3.0, vx.float()[:, None]], 1)


def cv_history(e):
    """Torch. Ego rows (B, 20) -> the 12 history numbers with the 3 past poses moved along the polyline of the fed history (newest to
    oldest, extended straight beyond the oldest pose) to the arc lengths speed x (0.5, 1.0, 1.5) s; yaw interpolated the same way."""
    import torch
    B = e.shape[0]
    P = e[:, 8:20].reshape(B, 4, 3)
    Q = torch.flip(P, [1])                                             # newest (t0) first
    d = Q[:, 1:, :2] - Q[:, :-1, :2]
    L = d.norm(dim=-1)
    c = torch.cat([L.new_zeros(B, 1), L.cumsum(1)], 1)
    a = torch.hypot(e[:, 4], e[:, 5])[:, None] * e.new_tensor([0.5, 1.0, 1.5])     # positions and speed are both in units of 10
    back = torch.stack([-torch.cos(Q[:, -1, 2]), -torch.sin(Q[:, -1, 2])], -1)
    u = torch.where(L[:, -1:] > 1e-4, d[:, -1] / L[:, -1:].clamp_min(1e-4), back)
    Qe = torch.cat([Q, torch.cat([Q[:, -1, :2] + 100.0 * u, Q[:, -1, 2:3]], -1)[:, None]], 1)
    ce = torch.cat([c, c[:, -1:] + 100.0], 1).contiguous()
    i = (torch.searchsorted(ce, a.contiguous(), right=True) - 1).clamp(0, 3)
    c0, c1 = ce.gather(1, i), ce.gather(1, i + 1)
    w = ((a - c0) / (c1 - c0).clamp_min(1e-6)).clamp(0, 1)[..., None]
    g = lambda j: Qe.gather(1, j[..., None].expand(-1, -1, 3))  # noqa: E731
    out = g(i) + w * (g(i + 1) - g(i))
    return torch.cat([torch.flip(out, [1]), P[:, 3:4]], 1).reshape(B, 12)


def ego_transform(noax: bool = False, hist_cv: bool = False):
    """The in_tf of an adapter (None when nothing is switched on)."""
    if not (noax or hist_cv):
        return None

    def f(ego):
        import torch
        e = ego.clone()
        if noax:
            e[:, 6] = 0
        if hist_cv:
            e = torch.cat([e[:, :8], cv_history(e)], 1)
        return e
    return f


def _torch():
    import torch
    import torch.nn as nn

    class ParityAdapter(nn.Module):
        def __init__(self, use_ego=True, use_side=True, d=256, layers=2, heads=8, n_cam=len(SIDE_CAMS), n_t=SIDE_T, T=32, D=512, use_lead=False):
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
            self.in_tf, self.use_lead = None, use_lead           # lane TR1 (module docstring); created last: the init draws above are unchanged
            if use_lead:
                self.lead = nn.Sequential(nn.Linear(LEAD_DIM, d), nn.GELU(), nn.Linear(d, d))

        def forward(self, ego, side=None, side_mask=None, lead=None):
            """ego (B, 20); side (B, n_cam, n_t, 32, 512) hidden tokens; side_mask (B, n_cam) True = camera present; lead (B, 7)
            lead_features (use_lead adapters) -> bias (B, 32, 512) fp32."""
            ego = ego.float()
            if self.in_tf is not None:
                ego = self.in_tf(ego)
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
            if self.use_lead:
                mem.append(self.lead(lead.float())[:, None])
                pad.append(torch.zeros(B, 1, dtype=torch.bool, device=ego.device))
            h = self.dec(self.q[None].expand(B, -1, -1), torch.cat(mem, 1), memory_key_padding_mask=torch.cat(pad, 1))
            return self.out(self.norm(h)) * ego[:, :1, None]

        def apply(self, H, ego, side=None, side_mask=None):
            """H (B, 9, 32, 512) hidden tokens -> H + bias on every context frame (H's dtype)."""
            return H + self(ego, side, side_mask)[:, None].to(H.dtype)

    return ParityAdapter


def ParityAdapter(**kw):
    return _torch()(**kw)
