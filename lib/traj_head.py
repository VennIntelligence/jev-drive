"""Trajectory head on the policy's plan-pathway hidden state: flow matching against regression (experiments/flowhead, lane FLOW1,
plans/2026-10-10-flow-head-prereg.md).

Tap     `select_4` of the Cinque graph, (512,): the state the shipped plan read-out is decoded from (lib/heading_aux.py). It lies
        downstream of the parity adapter, so the head is conditioned as the plan is (ego, history, command through the adapter's bias)
        and its gradient reaches the adapter and the temporal summarizer.
Output  the 8 poses (x, y, yaw; rear axle, left +) at 0.5 .. 4 s, i.e. the frame of the imitation target and of the exported
        prediction file: no lever arm, no resampling.
Net     f(h, x_t, t) -> x_hat in z-scored trajectory space (24 numbers): a residual MLP, the conditioning (LN(h) projected + time
        embedding) re-added in front of every block, zero-initialised output (x_hat = the mean train trajectory at step 0).
fm      x-prediction flow matching, the form WA-JEPA trains its trajectory head with: x_t = (1 - t) noise + t x, t = sigmoid(N(-0.2, 1.6))
        clamped to [1e-4, 1 - 1e-4], loss on x_hat; sampling = 4 Euler steps x += dt (x_hat - x) / max(1 - t, 1e-3) from t = 0, one fixed
        noise vector (row k of a seeded bank; k = 0 is served, the others are for the diversity read).
rg      the same net with x_t = 0 and t = 0: a deterministic regression head of equal capacity, trained by the same loss.
rh      rg with the recipe's own Huber (delta = 1 sigma) instead of L2: the registered follow-up if rg misses the stored recipe's level.
Loss    0.5 sum over (x, y, yaw) of ((x_hat - x) / sigma)^2, mean over the 8 poses, on the de-normalised x_hat with pp_train's per-time
        sigmas: the L2 form of the recipe's sigma-normalised Huber (equal to it below one sigma), so the hinge weight keeps its
        meaning. A per-dimension weighting does not move the minimiser (the conditional mean), so the flow stays a flow.
fit     also returns the trajectory the recipe's hinges act on: rg x_hat; fm a 4-step sample from fresh noise, with gradient.
"""
import math

import numpy as np
import torch
import torch.nn as nn

TAP = "select_4"
T8 = 0.5 * np.arange(1, 9)
SIG = np.stack([0.3 + 0.2 * T8, 0.1 + 0.1 * T8, np.radians(1.0 + 1.0 * T8)], -1)      # pp_train SIG_X / SIG_Y / SIG_PSI, (8, 3)
LOGIT_MEAN, LOGIT_STD, STEPS, N_NOISE, NOISE_SEED = -0.2, 1.6, 4, 16, 20261010
KINDS = ("fm", "rg", "rh")


class TrajHead(nn.Module):
    def __init__(self, kind: str, mean, std, d_in: int = 512, d: int = 512, blocks: int = 4, seed: int = 0):
        super().__init__()
        assert kind in KINDS, kind
        self.kind, self.seed, self.gen = kind, seed, None
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32)).reshape(24)  # noqa: E731
        self.register_buffer("mean", t(mean))
        self.register_buffer("std", t(std).clamp_min(1e-3))
        self.register_buffer("sig", t(SIG))
        self.register_buffer("noise", torch.randn(N_NOISE, 24, generator=torch.Generator().manual_seed(NOISE_SEED)))
        self.register_buffer("freq", torch.exp(-math.log(1e3) * torch.arange(32) / 31))
        self.cond = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.xin = nn.Linear(24, d)
        self.tin = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 24))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def net(self, h, xt, t):
        """h (B, 512), x_t (B, 24) z-scored, t (B,) -> x_hat (B, 24) z-scored."""
        a = 1e3 * t[:, None] * self.freq
        c = self.cond(h.float()) + self.tin(torch.cat([a.sin(), a.cos()], 1))
        z = self.xin(xt)
        for b in self.blocks:
            z = z + b(z + c)
        return self.out(z)

    def poses(self, z):
        return (z * self.std + self.mean).view(-1, 8, 3)

    def _randn(self, *shape):
        if self.gen is None:                                    # own stream: the trainer's other draws are untouched
            self.gen = torch.Generator(self.mean.device).manual_seed(1_000_003 * self.seed + 13)
        return torch.randn(*shape, device=self.mean.device, generator=self.gen)

    def sample(self, h, noise=None, k: int = 0):
        """(B, 8, 3) poses in metres / rad. fm: STEPS Euler steps from `noise` (B, 24) (default: row k of the fixed bank for every
        row, so a row's plan does not depend on its batch); rg: the regression output."""
        B = len(h)
        if self.kind != "fm":
            return self.poses(self.net(h, h.new_zeros(B, 24, dtype=torch.float32), h.new_zeros(B, dtype=torch.float32)))
        x = self.noise[k].expand(B, 24) if noise is None else noise
        for i in range(STEPS):
            t = torch.full((B,), i / STEPS, device=x.device)
            x = x + (self.net(h, x, t) - x) / (STEPS * max(1.0 - i / STEPS, 1e-3))
        return self.poses(x)

    def fit(self, h, tgt, traj: bool = True):
        """h (B, 512), tgt (B, 8, 3) -> (loss per row (B,), the trajectory (B, 8, 3) the hinges act on, or None)."""
        B = len(h)
        if self.kind != "fm":
            p = self.sample(h)
            return self._l2(p, tgt), (p if traj else None)
        t = torch.sigmoid(self._randn(B) * LOGIT_STD + LOGIT_MEAN).clamp(1e-4, 1 - 1e-4)
        x1 = (tgt.reshape(B, 24).float() - self.mean) / self.std
        xt = (1 - t[:, None]) * self._randn(B, 24) + t[:, None] * x1
        return self._l2(self.poses(self.net(h, xt, t)), tgt), (self.sample(h, self._randn(B, 24)) if traj else None)

    def _l2(self, p, tgt):
        z = (p - tgt.float()) / self.sig.view(8, 3)
        l = torch.nn.functional.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0) if self.kind == "rh" else 0.5 * z ** 2
        return l.sum(-1).mean(-1)

    def pack(self) -> dict:
        return {"kind": self.kind, "seed": self.seed, "state": {k: v.detach().cpu() for k, v in self.state_dict().items()}}


def load(path, dev) -> TrajHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["state"]
    hd = TrajHead(ck["kind"], st["mean"], st["std"], seed=ck.get("seed", 0), blocks=sum(k.endswith(".3.weight") and k.startswith("blocks") for k in st))
    hd.load_state_dict(st)
    return hd.to(dev).eval()


@torch.no_grad()
def serve(model, head, S, rows=None, ks=(0,), batch: int = 128) -> np.ndarray:
    """The head's plans of the store rows (default all) for the noise rows `ks` -> (len(ks), n, 8, 3) float32. `model` is a pp_train
    PModel (P2 arm); its tap is set for the pass and restored."""
    rows = np.arange(S.n) if rows is None else np.asarray(rows)
    tap, was = model.tap, (model.training, head.training)
    model.tap = TAP
    model.eval()
    head.eval()
    P = np.zeros((len(ks), len(rows), 8, 3), np.float32)
    for i in range(0, len(rows), batch):
        r = torch.as_tensor(rows[i:i + batch], device=S.ego.device)
        _, h = model(S.front[r], S.ego[r], S.tc[r], None, None, nv=None if S.nv is None else S.nv[r])
        for j, k in enumerate(ks):
            P[j, i:i + len(r)] = head.sample(h.float(), k=k).cpu().numpy()
    model.tap = tap
    model.train(was[0])
    head.train(was[1])
    return P


def dev_ade(model, head, S, rows) -> dict:
    """ADE (m) of the served plan to the log on the dev rows that have a logged future."""
    rows = np.asarray(rows)[S.has_fut[torch.as_tensor(rows, device=S.ego.device)].cpu().numpy()]
    P = serve(model, head, S, rows)[0]
    f = S.fut[torch.as_tensor(rows, device=S.ego.device)].cpu().numpy()
    return {"th_ade": float(np.hypot(P[..., 0] - f[..., 0], P[..., 1] - f[..., 1]).mean())}
