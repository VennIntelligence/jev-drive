"""junction_eyes step 1 model: Cinque's own vision trunk made trainable (LoRA with Conv terms, or fully unfrozen), two
heads on its per-frame tokens `view_39` (32 x 512), the frozen factory policy for the keep term (plan 4.1, 4.3, 5.2).

Arm kinds (`vis`):
  lora   LoRA r 16 / alpha 16 (B zero) on every trunk MatMul (the ConvNeXt MLPs) and every dense Conv kernel (stem 4 x 4,
         downsample 2 x 2, head 1 x 1; on (out, in * k * k)); the 36 depthwise 7 x 7 kernels fully trainable (fp32 masters)
  full   every float initializer of the trunk trainable (fp32 masters), the input normalisation constants excepted
  none   frozen (Z; the identity reference)
The policy, its adapter-free inputs and every non-vision weight stay at the factory values; nothing else is trained.

Heads: experiments/corridor/scripts/head1_train.py `build()` unchanged (3-layer encoder + 2-layer query decoder, d 256), on
[tokens of the t0 slot (slot 7 of its 8 x 32 token input; `keep` reads only those), standardised 20-dim ego]:
  ra  nq 15, nc 2    R-A: 5 arc lengths x (centre, left, right) x (x, y), targets standardised by P's mean / std
  rb  nq 36, nc 192  R-B: 6 x 6 blocks of 8 x 8 cells x 3 classes -> (48, 48, 3) logits (`rb_logits`)
Views (the 3V arm, implemented, unused): n_view token sets go into the last n_view slots, each with a learned camera-id
embedding added to its tokens (zero init).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_R = Path(__file__).resolve().parents[3]
for _p in (_R, _R / "lib", _R / "experiments/corridor/scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.op_torch import _key  # noqa: E402

RANK, ALPHA = 16, 16
NQ_RA, NC_RA, NQ_RB, NC_RB = 15, 2, 36, 192
GRID, BLK = 48, 8


def rb_logits(o: torch.Tensor) -> torch.Tensor:
    """(B, 36, 192) head output -> (B, 48, 48, 3): query q = 6 * bi + bj owns cells [8 bi, 8 bi + 8) x [8 bj, 8 bj + 8),
    its 192 outputs are (8 rows, 8 columns, 3 classes) row-major. Row index = x bin (0 .. 24 m ahead), column = y bin."""
    B, n = o.shape[0], GRID // BLK
    return o.view(B, n, n, BLK, BLK, 3).permute(0, 1, 3, 2, 4, 5).reshape(B, GRID, GRID, 3)


def trunk_names(kind: str) -> tuple[dict, list]:
    """-> (lora {name: rank}, trainable [names]) of a vision kind."""
    if kind == "none":
        return {}, []
    t = A.vision_lora_targets("cinque")
    if kind == "lora":
        return {w: RANK for w in t["matmul"] + t["conv"]}, t["dw"]
    if kind == "full":
        import onnx
        g = onnx.load(str(A.MODELS_DIR / A.FILES["cinque"]), load_external_data=False).graph
        names = A.node_weights(A.MODELS_DIR / A.FILES["cinque"], g.node[0].output[0], "view_39")
        return {}, [w for w in names if not w.endswith(("_mean", "_std"))]
    raise ValueError(kind)


class JE(nn.Module):
    def __init__(self, vis: str = "lora", n_view: int = 1, dtype=torch.bfloat16, heads: bool = True):
        super().__init__()
        from head1_train import build
        self.vis, self.n_view = vis, n_view
        lora, tr = trunk_names(vis)
        self.net = A.load("cinque", dtype, trainable=tr, lora=lora or None, lora_alpha=ALPHA)
        self.vkeys = {_key(w) for w in tr}
        if heads:
            self.ra, self.rb = build(True, nq=NQ_RA, nc=NC_RA), build(True, nq=NQ_RB, nc=NC_RB)
        self.cam_emb = nn.Parameter(torch.zeros(n_view, 512)) if n_view > 1 else None
        for k, v in dict(emu=torch.zeros(20), esd=torch.ones(20), ramu=torch.zeros(NQ_RA, NC_RA), rasd=torch.ones(NQ_RA, NC_RA)).items():
            self.register_buffer(k, v)
        self.compiled = False

    # ---- parameter groups (plan 4.4)
    def groups(self) -> dict:
        lora = list(self.net.lora.parameters())
        vis = [p for k, p in self.net.params.items() if p.requires_grad]
        head = [p for m in ("ra", "rb") if hasattr(self, m) for p in getattr(self, m).parameters()]
        head += [self.cam_emb] if self.cam_emb is not None else []
        return {"lora": lora, "vis": vis, "head": head}

    def n_params(self) -> dict:
        return {k: sum(p.numel() for p in v) for k, v in self.groups().items()}

    # ---- trunk
    def encode(self, prev, cur, grad=False, chunk=128):
        """(n, 2, 6, 128, 256) uint8 pairs on the device -> (n, 32, 512) tokens (compute dtype)."""
        import pixel_store as PX
        if grad:
            return PX.fast_encode(self.net, prev, cur, grad=True, compiled=self.compiled)
        return PX.fast_encode(self.net, prev, cur, grad=False, chunk=chunk)

    def policy(self, H, valid=None, action_t=(0.275, 0.525)):
        """H (B, 9, 32, 512) hidden states [t0 - 1.6 s .. t0] -> raw outputs (B, n_out) of the frozen factory policy."""
        if valid is not None:
            H = H * valid[:, :, None, None].to(H.dtype)
        o = self.net.run_batched(A.policy_feeds(self.net, H.to(self.net.dtype), action_t), ["outputs"])["outputs"]
        return o.reshape(H.shape[0], -1)

    # ---- heads
    def head_in(self, tok):
        """tok (B, 32, 512) or (B, n_view, 32, 512) -> (V (B, 8, 32, 512), keep): the views in the last slots."""
        if tok.dim() == 3:
            tok = tok[:, None]
        B, nv = tok.shape[:2]
        if self.cam_emb is not None:
            tok = tok + self.cam_emb[None, :, None].to(tok.dtype)
        V = torch.cat([tok.new_zeros(B, 8 - nv, 32, 512), tok], 1)
        return V, torch.arange((8 - nv) * 32, 256, device=tok.device)

    def heads(self, tok, ego):
        """-> ra (B, 15, 2) standardised, rb logits (B, 48, 48, 3); fp32."""
        V, keep = self.head_in(tok)
        E = (ego.float() - self.emu) / self.esd
        return self.ra(V, E, keep), rb_logits(self.rb(V, E, keep))

    # ---- checkpoints: the trained part only (the factory ONNX is the rest)
    def trained_state(self) -> dict:
        st = {"vis": self.vis, "n_view": self.n_view,
              "lora": {k: v.detach().float().cpu() for k, v in self.net.lora.items()},
              "train": {k: p.detach().float().cpu() for k, p in self.net.params.items() if p.requires_grad},
              "buf": {k: getattr(self, k).cpu() for k in ("emu", "esd", "ramu", "rasd")}}
        for m in ("ra", "rb"):
            if hasattr(self, m):
                st[m] = getattr(self, m).state_dict()
        if self.cam_emb is not None:
            st["cam_emb"] = self.cam_emb.detach().cpu()
        return st

    def load_trained(self, st: dict):
        assert st["vis"] == self.vis and st["n_view"] == self.n_view, (st["vis"], st["n_view"])
        with torch.no_grad():
            for k, v in st["lora"].items():
                self.net.lora[k].copy_(v)
            for k, v in st["train"].items():
                self.net.params[k].copy_(v)
            for k, v in st["buf"].items():
                getattr(self, k).copy_(v)
            if self.cam_emb is not None:
                self.cam_emb.copy_(st["cam_emb"])
        for m in ("ra", "rb"):
            if hasattr(self, m) and m in st:
                getattr(self, m).load_state_dict(st[m])
        return self


def load_ckpt(path, dev, dtype=torch.bfloat16, heads=True) -> tuple[JE, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["model"]
    m = JE(st["vis"], st["n_view"], dtype, heads=heads)
    m.load_trained(st)
    return m.to(dev).eval(), ck


def distill_cols(net) -> tuple[np.ndarray, np.ndarray]:
    """Keep-term columns (vt.py / pp_train lam_d convention: MDN means of every MDN head + the logit heads; hidden state and
    spreads out) and the plan columns among them. The plan head is distilled at the same per-column weight as the others."""
    di = A.distill_index(net.slices)
    return di, np.isin(di, A.plan_index(net.slices))
