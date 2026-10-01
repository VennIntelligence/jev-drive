"""op-adapt: adapting openpilot's vision layer (fc65452:todos/2026-09-28-op-adapt.md).

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


def policy_feeds(net: OnnxTorch, ctx: torch.Tensor, action_t, traffic=None) -> dict:
    """ctx: (B, 9, 32, 512) hidden states [H_{t-8}, ..., H_t] (zeros where the stream had not started).
    traffic: (B, 2) traffic convention ([1, 0] right-hand, [0, 1] left-hand), default right-hand."""
    B, dt, dev = ctx.shape[0], net.dtype, ctx.device
    past = torch.cat([ctx.new_zeros(B, 1, 24, *H_SHAPE), ctx[:, None, :-1]], 2)   # (B, 1, 32, 32, 512): 24 unused slots
    tc = torch.tensor([[1.0, 0.0]], device=dev).expand(B, 2) if traffic is None else traffic
    return {"_to_copy_1": past.to(dt), "view_39": ctx[:, None, -1].to(dt),
            "_to_copy": torch.zeros(B, 1, 33, 8, dtype=dt, device=dev),
            "_to_copy_2": tc.to(dt)[:, None], "_to_copy_3": torch.tensor([action_t], dtype=dt, device=dev).expand(B, 1, 2)}


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


# ---------------------------------------------------------------- the adapted model (B: stage 4 unfrozen)
TRUNK_OUT = "permute_73"        # (1, 1024, 8, 16): stage 3 output after its LN, input of stage 4's downsample conv
STAGE4 = ("conv2d_36", "view_39")


def node_weights(onnx_path, first: str, last: str) -> list[str]:
    """Float initializers consumed by the nodes from the producer of `first` to the producer of `last`."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(onnx_path), load_external_data=False).graph
    fl = {t.name for t in g.initializer if t.data_type in (1, 10, 11, 16)}
    prod = {o: k for k, n in enumerate(g.node) for o in n.output}
    return sorted({i for n in g.node[prod[first]:prod[last] + 1] for i in n.input if i in fl})


def stage4_weights(model="cinque") -> list[str]:
    return node_weights(MODELS_DIR / FILES[model], *STAGE4)


def stage4_matmuls(model="cinque") -> list[str]:
    """The 2-D MLP weights of stage 4 (LoRA targets)."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(MODELS_DIR / FILES[model])).graph
    w = set(stage4_weights(model))
    return sorted(t.name for t in g.initializer if t.name in w and len(t.dims) == 2)


class AuxHeads(torch.nn.Module):
    """Pedestrian heads: (i) MLP on the temporal token, (ii) attention pooling over the current frame's 32 hidden
    tokens. Each outputs [corridor logit, wide-corridor logit, distance / 10]."""

    def __init__(self, d=512, h=256, k=3):
        super().__init__()
        self.t = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, k))
        self.q = torch.nn.Parameter(torch.zeros(4, d))
        self.v_norm = torch.nn.LayerNorm(d)
        self.v = torch.nn.Sequential(torch.nn.Linear(4 * d, h), torch.nn.GELU(), torch.nn.Linear(h, k))

    def forward(self, temporal, tokens):
        x = self.v_norm(tokens.float())                                             # (B, 32, 512)
        att = torch.softmax(torch.einsum("qd,bnd->bqn", self.q, x) / x.shape[-1] ** 0.5, -1)
        pooled = torch.einsum("bqn,bnd->bqd", att, x).flatten(1)
        return self.t(temporal.float()), self.v(pooled)


def stage4_policy(net: OnnxTorch, trunk: torch.Tensor, action_t, traffic=None, valid=None) -> dict:
    """trunk: (B, 9, 1024, 8, 16) stage-3 outputs of the 9 context frames (zeros rows = no stream yet, see `mask`)
    -> outputs (B, n), select_4 (B, 512), mean (B, 512), tokens (B, 32, 512) of the current frame."""
    B = trunk.shape[0]
    H = net.run_batched({TRUNK_OUT: trunk.reshape(B * CONTEXT, 1, *trunk.shape[2:]).to(net.dtype)}, ["view_39"])["view_39"]
    H = H.reshape(B, CONTEXT, *H_SHAPE)
    return _policy(net, H, action_t, traffic, valid)


def _policy(net, H, action_t, traffic, valid=None):
    if valid is not None:                       # frames before the stream start carry a zero hidden state
        H = H * valid[:, :, None, None].to(H.dtype)
    o = net.run_batched(policy_feeds(net, H, action_t, traffic), POLICY_OUT)
    B = H.shape[0]
    return {"outputs": o["outputs"].reshape(B, -1), "select_4": o["select_4"].reshape(B, -1),
            "mean": o["mean"].reshape(B, -1), "tokens": H[:, -1]}


def full_policy(net: OnnxTorch, prev: torch.Tensor, cur: torch.Tensor, action_t, traffic=None, grad_past=True) -> dict:
    """C: the whole vision stack on the 9 context frames. prev / cur: (B, 9, 2, 6, 128, 256) uint8 image pairs.
    grad_past=False runs the 8 past frames without autograd (C')."""
    B = cur.shape[0]
    f = lambda p, c: net.run_batched(vision_feeds(p.reshape(-1, *p.shape[2:]), c.reshape(-1, *c.shape[2:])),  # noqa: E731
                                     ["view_39"])["view_39"].reshape(p.shape[0], p.shape[1], *H_SHAPE)
    if grad_past:
        H = f(prev, cur)
    else:
        with torch.no_grad():
            Hp = f(prev[:, :-1], cur[:, :-1])
        H = torch.cat([Hp, f(prev[:, -1:], cur[:, -1:])], 1)
    return _policy(net, H, action_t, traffic)


# ---------------------------------------------------------------- distillation targets
MDN_HALF = ("plan", "lead", "lane_lines", "road_edges", "pose", "wide_from_device_euler", "road_transform", "action")
FULL = ("lane_lines_prob", "meta", "desire_pred", "lead_prob", "desire_state")


def distill_index(slices: dict) -> np.ndarray:
    """Output-vector positions distilled: the MDN means (first half) of every MDN head plus the logit heads; the
    queued hidden state and the MDN spreads are left out."""
    idx = []
    for k, s in slices.items():
        if k in MDN_HALF:
            idx.append(np.arange(s.start, s.start + (s.stop - s.start) // 2))
        elif k in FULL:
            idx.append(np.arange(s.start, s.stop))
    return np.concatenate(idx)


def plan_index(slices: dict) -> np.ndarray:
    """Positions of the plan MDN mean (33 x 15) inside the output vector."""
    return np.arange(slices["plan"].start, slices["plan"].start + 495)


T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
T5 = np.flatnonzero(T_IDXS <= 5.0)


def plan_drift(p_new: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """Per sample mean L2 distance of the plan positions (x, y) over the plan points up to 5 s. p: (n, 33, 15)."""
    return np.linalg.norm(p_new[:, T5, :2] - p_ref[:, T5, :2], axis=-1).mean(1)
