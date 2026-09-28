"""op-adapt: adapting openpilot's vision layer (todos/2026-09-28-op-adapt.md).

Model plumbing on top of the graph port `jevdrive.op_torch`:
  load            cinque / lebowski ONNX -> OnnxTorch
  Stepper         modeld-style 20 Hz stepping with the port's own recurrent state (equivalence check)
  vision/policy   the training decomposition for cinque: per-frame vision tokens H_f = V(img_{f-d}, img_f)
                  (`view_39`, 32 x 512), then the policy on the 9-frame context [H_{f-8d}, ..., H_f] -> outputs,
                  `select_4` (temporal) and `mean` (pooled vision). d = 2 WOD frames (10 Hz fed twice),
                  1 CARLA frame (5 Hz held 4 steps), or the 4-step lattice of the nuScenes 20 Hz clock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .op_torch import OnnxTorch

MODELS_DIR = Path.home() / "data/models/openpilot"
FILES = {"cinque": "cinque.ort.onnx", "lebowski": "lebowski.onnx"}
TAPS = {"cinque": {"temporal": "select_4", "vision": "mean"}, "lebowski": {"temporal": "select", "vision": "view_40"}}
CONTEXT = 9                    # cinque's policy reads the current and the 8 previous 5 Hz hidden states (1.6 s)
CONTEXT_FRAMES_WOD = 2 * (CONTEXT - 1)
H_SHAPE = (32, 512)


def load(model: str, dtype=torch.float32, **kw) -> OnnxTorch:
    return OnnxTorch(MODELS_DIR / FILES[model], dtype=dtype, **kw)


def _in_dtype(net, name):
    t = net.in_types[name]
    return torch.uint8 if t == 2 else net.dtype if t == 10 else torch.float32


class Stepper:
    """One model stepped at 20 Hz exactly as jevdrive.openpilot.model.OPModel (desire 0, traffic [1, 0])."""

    def __init__(self, net: OnnxTorch, model: str):
        self.net, self.model = net, model
        self.dev = next(iter(net.params.values())).device
        self.queued = "new_img" in net.in_types
        self.taps = TAPS[model]

    def _t(self, name, a):
        return torch.as_tensor(np.asarray(a)).to(self.dev, _in_dtype(self.net, name))

    def reset(self):
        n = self.net
        if self.queued:
            self.state = {k: torch.zeros(n.in_shapes[k], dtype=_in_dtype(n, k), device=self.dev)
                          for k in n.in_types if k.startswith("state_")}
        else:
            self.img_q = torch.zeros((2, 5, 6, 128, 256), dtype=torch.uint8, device=self.dev)
            self.desire_q = torch.zeros((100, 8), device=self.dev)
            self.feat_q = torch.zeros((96, 512), device=self.dev)
            self.prev = torch.zeros(512, device=self.dev)

    @torch.no_grad()
    def step(self, img2: np.ndarray, action_t):
        n = self.net
        tc, at = self._t("traffic_convention", [[1.0, 0.0]]), self._t("action_t", [action_t])
        want = ["outputs", self.taps["temporal"], self.taps["vision"]]
        if self.queued:
            f = {"new_img": self._t("new_img", img2), "desire": self._t("desire", np.zeros(8)),
                 "traffic_convention": tc, "action_t": at, **self.state}
            out = n.run(f, want + ["next_" + k for k in self.state])
            self.state = {k: out["next_" + k] for k in self.state}
        else:
            img = torch.as_tensor(img2).to(self.dev)
            self.img_q = torch.cat([self.img_q[:, 1:], img[:, None]], 1)
            self.desire_q = torch.cat([self.desire_q[1:], torch.zeros(1, 8, device=self.dev)])
            self.feat_q = torch.cat([self.feat_q[1:], self.prev[None]])
            f = {"img": self.img_q[0, ::4].reshape(1, 12, 128, 256), "big_img": self.img_q[1, ::4].reshape(1, 12, 128, 256),
                 "desire_pulse": self.desire_q.reshape(25, 4, 8).amax(1)[None].to(_in_dtype(n, "desire_pulse")),
                 "traffic_convention": tc, "action_t": at,
                 "features_buffer": self.feat_q[::4][None].to(_in_dtype(n, "features_buffer"))}
            out = n.run(f, want)
            self.prev = out["outputs"][0, n.slices["hidden_state"]].float()
        return [out[w].float().reshape(-1).cpu().numpy() for w in want]

    def run_stream(self, frames: np.ndarray, action_t, repeat: int = 2):
        """Every frame fed `repeat` times (WOD: 10 Hz -> 20 Hz); returns per-step raw, temporal, vision."""
        self.reset()
        res = [self.step(fr, action_t) for fr in frames for _ in range(repeat)]
        return [np.stack(x) for x in zip(*res)]


# ---------------------------------------------------------------- cinque training decomposition
VISION_IN = ("_unsafe_view", "_unsafe_view_1")   # (1, 12, 128, 256) road / wide = [img_{t-4}, img_t] channels


def vision_feeds(prev: torch.Tensor, cur: torch.Tensor) -> dict:
    """prev / cur: (B, 2, 6, 128, 256) uint8 [road, wide] -> batched feeds of the vision part."""
    pair = torch.cat([prev, cur], 2)                                  # (B, 2, 12, 128, 256)
    return {VISION_IN[0]: pair[:, 0:1], VISION_IN[1]: pair[:, 1:2]}


def policy_feeds(net: OnnxTorch, ctx: torch.Tensor, action_t) -> dict:
    """ctx: (B, 9, 32, 512) hidden states [H_{t-8}, ..., H_t] (zeros where the stream had not started)."""
    B = ctx.shape[0]
    past = torch.cat([ctx.new_zeros(B, 1, 24, *H_SHAPE), ctx[:, None, :-1]], 2)   # (B, 1, 32, 32, 512): 24 unused slots
    dt = net.dtype
    return {"_to_copy_1": past.to(dt), "view_39": ctx[:, None, -1].to(dt),
            "_to_copy": torch.zeros(B, 1, 33, 8, dtype=dt, device=ctx.device),
            "_to_copy_2": torch.tensor([[1.0, 0.0]], dtype=dt, device=ctx.device).expand(B, 1, 2),
            "_to_copy_3": torch.tensor([action_t], dtype=dt, device=ctx.device).expand(B, 1, 2)}


POLICY_OUT = ["outputs", "select_4", "mean"]


def decomposed_wod(net: OnnxTorch, frames: np.ndarray, action_t, dev, batch: int = 32):
    """The WOD stream protocol computed the training way: H_f per frame, policy per frame on [H_{f-16}, .., H_f]
    (step 2). Returns raw outputs and select_4 per frame, as recorded at each frame's second 20 Hz step."""
    fr = torch.as_tensor(frames).to(dev)
    prev = torch.cat([torch.zeros_like(fr[:2]), fr[:-2]])
    with torch.no_grad():
        H = torch.cat([net.run_batched(vision_feeds(prev[i:i + batch], fr[i:i + batch]), ["view_39"])["view_39"][:, 0]
                       for i in range(0, len(fr), batch)])
        Hp = torch.cat([torch.zeros(2 * (CONTEXT - 1), *H_SHAPE, dtype=H.dtype, device=dev), H])
        idx = torch.arange(len(fr), device=dev)[:, None] + torch.arange(0, 2 * CONTEXT, 2, device=dev)[None]
        raw, tmp = [], []
        for i in range(0, len(fr), batch):
            ctx = Hp[idx[i:i + batch]]
            o = net.run_batched(policy_feeds(net, ctx, action_t), POLICY_OUT)
            raw.append(o["outputs"].reshape(len(ctx), -1).float())
            tmp.append(o["select_4"].reshape(len(ctx), -1).float())
    return torch.cat(raw).cpu().numpy(), torch.cat(tmp).cpu().numpy()
