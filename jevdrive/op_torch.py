"""openpilot driving models (cinque, lebowski ONNX) as trainable PyTorch modules.

A direct interpreter of the ONNX graph: every node runs as the matching torch op, initializers become parameters,
so the numerics follow the graph exactly and every value the graph names can be fed or read. It is written for
one sample, the graph's own batch-1 shapes; batching is `torch.func.vmap` over the sample axis (`run_batched`),
which also keeps the hard-coded batch-1 reshapes valid. Autograd runs through it in fp32 or bf16.

  dtype        compute dtype for the graph's float16 values (float32 for equivalence, bfloat16 for training,
               float16 to mirror the graph literally); graph float32 values stay float32
  trainable    initializer names that become requires_grad parameters (fp32 master copies)
  lora         {initializer name: rank} adds a low-rank update x @ (W + A @ B * alpha / rank) to MatMul / Gemm
               weights; A, B are fp32 parameters, B starts at zero so the model starts exactly at the original

Split points (cinque): the vision trunk ends in `view_39` (1, 32, 512), the per-frame hidden state that the model
queues; the policy reads 9 of 33 queued frames (`index`, (1, 9, 32, 512)) and emits `outputs` and the temporal
token `select_4`. `cinque_policy_feeds` builds those inputs from per-frame hidden states, so training can run the
vision part per frame and the policy per sample without the 20 Hz queue.
"""
from __future__ import annotations

import base64
import pickle
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

ONNX_TO_TORCH = {1: torch.float32, 10: torch.float16, 7: torch.int64, 9: torch.bool, 2: torch.uint8, 6: torch.int32}


def _key(name: str) -> str:
    return name.replace(".", "__").replace("/", "_")


class OnnxTorch(torch.nn.Module):
    def __init__(self, path: str | Path, dtype=torch.float32, trainable=(), lora=None, lora_alpha=None):
        super().__init__()
        m = onnx.load(str(path))
        g = m.graph
        self.dtype = dtype
        meta = {p.key: p.value for p in m.metadata_props}
        self.slices = pickle.loads(base64.b64decode(meta["output_slices"])) if "output_slices" in meta else {}
        self.params = torch.nn.ParameterDict()
        self.cnames: set[str] = set()
        self.ints: dict[str, list] = {}
        trainable = set(trainable)
        self.half = set()
        for t in g.initializer:
            a = numpy_helper.to_array(t)
            if a.dtype.kind == "f":
                tr = t.name in trainable
                if a.dtype == np.float16:
                    self.half.add(_key(t.name))
                p = torch.nn.Parameter(torch.from_numpy(a.astype(np.float32)).to(torch.float32 if tr else self._fdt(a)),
                                       requires_grad=tr)
                self.params[_key(t.name)] = p
            else:
                self.register_buffer("c_" + _key(t.name), torch.from_numpy(a.copy()), persistent=False)
                self.cnames.add(t.name)
                if a.dtype.kind in "iu" and a.ndim <= 1:
                    self.ints[t.name] = a.reshape(-1).tolist()
        self.fnames = {t.name for t in g.initializer if numpy_helper.to_array(t).dtype.kind == "f"}
        self.nodes = [(n.op_type, list(n.input), list(n.output),
                       {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in g.node]
        self.inputs = [i.name for i in g.input]
        self.in_types = {i.name: i.type.tensor_type.elem_type for i in g.input}
        self.in_shapes = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in g.input}
        self.outputs = [o.name for o in g.output]
        self.producer = {o: k for k, n in enumerate(self.nodes) for o in n[2]}
        self._plans, self._drops = {}, {}
        self.lora = torch.nn.ParameterDict()
        self.lora_scale = {}
        for w, r in (lora or {}).items():
            W = self.params[_key(w)]
            din, dout = W.shape if W.ndim == 2 else (None, None)
            # Gemm weights here are (out, in) with transB=1, MatMul weights (in, out): orient A @ B like W
            self.lora[_key(w) + "__A"] = torch.nn.Parameter(torch.randn(din, r) / r ** 0.5)
            self.lora[_key(w) + "__B"] = torch.nn.Parameter(torch.zeros(r, dout))
            self.lora_scale[w] = (lora_alpha or r) / r

    def _fdt(self, a):
        return self.dtype if a.dtype == np.float16 else torch.float32

    def to_dtype(self, dtype):
        """Change the compute dtype; frozen fp16-origin weights follow it, trainable fp32 masters stay fp32."""
        self.dtype = dtype
        for k, p in self.params.items():
            if k in self.half and not p.requires_grad:
                p.data = p.data.to(dtype)
        return self

    # ---- graph slicing ----
    def plan(self, have: tuple, want: tuple):
        key = (have, want)
        if key not in self._plans:
            have_s, need, stack = set(have), set(), list(want)
            while stack:
                v = stack.pop()
                if v in have_s or v in self.fnames or v in self.cnames or v == "" or v in need:
                    continue
                if v not in self.producer:
                    raise KeyError(f"value {v} is neither fed nor produced")
                need.add(v)
                stack.extend(self.nodes[self.producer[v]][1])
            ks = sorted({self.producer[v] for v in need})
            self._plans[key] = ks
        return self._plans[key]

    def _drop(self, key):
        """Per node of plan(*key): the values no later node reads. run forgets them there, so a pass holds its live values only (the
        encoder at batch 128 needs 1.2 GB instead of 29; autograd keeps what backward needs). Same outputs."""
        if key not in self._drops:
            last, want, d = {}, set(key[1]), {}
            for k in self.plan(*key):
                last.update(dict.fromkeys(self.nodes[k][1], k))
                for o in self.nodes[k][2]:
                    last.setdefault(o, k)
            for v, k in last.items():
                if v and v not in want:
                    d.setdefault(k, []).append(v)
            self._drops[key] = d
        return self._drops[key]

    def _w(self, name):
        return self.params[_key(name)] if name in self.fnames else self._c(name)

    def _c(self, name):
        return getattr(self, "c_" + _key(name))

    def run(self, feeds: dict, want: list[str]) -> dict:
        """One sample: feeds {value name: tensor} -> {name: tensor} for the wanted values."""
        env, key = dict(feeds), (tuple(sorted(feeds)), tuple(want))
        drop = self._drop(key)
        for k in self.plan(*key):
            op, ins, outs, at = self.nodes[k]
            x = [env[i] if i in env else (self._w(i) if i else None) for i in ins]
            y = getattr(self, "op_" + op)(x, at, ins)
            if not isinstance(y, (list, tuple)):
                y = [y]
            env.update(zip(outs, y))
            del x, y
            for v in drop.get(k, ()):
                env.pop(v, None)
        return {w: env[w] for w in want}

    def run_batched(self, feeds: dict, want: list[str], in_dims=None) -> dict:
        """vmap of run over dim 0 of every fed tensor (or per in_dims {name: dim or None})."""
        names = sorted(feeds)
        dims = tuple((in_dims or {}).get(n, 0) for n in names)
        f = lambda *xs: self.run(dict(zip(names, xs)), want)  # noqa: E731
        return torch.func.vmap(f, in_dims=dims)(*[feeds[n] for n in names])

    # ---- ops (one sample, ONNX semantics) ----
    def _ints(self, name, x):
        return self.ints[name] if name in self.ints else x.reshape(-1).tolist()

    def op_Identity(self, x, at, ins):
        return x[0]

    def op_Cast(self, x, at, ins):
        dt = ONNX_TO_TORCH[at["to"]]
        return x[0].to(self.dtype if dt == torch.float16 else dt)

    def op_Slice(self, x, at, ins):
        d = x[0]
        st, en = self._ints(ins[1], x[1]), self._ints(ins[2], x[2])
        ax = self._ints(ins[3], x[3]) if len(x) > 3 and x[3] is not None else list(range(len(st)))
        sp = self._ints(ins[4], x[4]) if len(x) > 4 and x[4] is not None else [1] * len(st)
        idx = [slice(None)] * d.dim()
        for s, e, a, p in zip(st, en, ax, sp):
            n = d.shape[a]
            assert p > 0, "negative Slice step"
            s = max(0, min(n, s + n if s < 0 else s))
            e = max(0, min(n, e + n if e < 0 else e))
            idx[a] = slice(s, e, p)
        return d[tuple(idx)]

    def op_Unsqueeze(self, x, at, ins):
        y = x[0]
        axes = sorted(a % (y.dim() + len(self._ints(ins[1], x[1]))) for a in self._ints(ins[1], x[1]))
        for a in axes:
            y = y.unsqueeze(a)
        return y

    def op_Squeeze(self, x, at, ins):
        y = x[0]
        for a in sorted((a % y.dim() for a in self._ints(ins[1], x[1])), reverse=True):
            y = y.squeeze(a)
        return y

    def op_Concat(self, x, at, ins):
        dt = x[0].dtype
        return torch.cat([t.to(dt) for t in x], at["axis"])

    def op_Reshape(self, x, at, ins):
        sh = self._ints(ins[1], x[1])
        if not at.get("allowzero", 0):
            sh = [x[0].shape[i] if s == 0 else s for i, s in enumerate(sh)]
        return x[0].reshape(sh)

    def op_Gather(self, x, at, ins):
        d, ax = x[0], at.get("axis", 0)
        i = x[1]
        if i.dim() == 0:
            v = self.ints[ins[1]][0] if ins[1] in self.ints else int(i)
            return d.select(ax, v % d.shape[ax])
        i = torch.where(i < 0, i + d.shape[ax], i).to(d.device)
        return torch.index_select(d, ax, i.reshape(-1)).reshape(*d.shape[:ax], *i.shape, *d.shape[ax + 1:])

    def op_GatherND(self, x, at, ins):
        assert at.get("batch_dims", 0) == 0
        i = x[1]
        assert i.shape[-1] == 1
        return x[0][i[..., 0].to(x[0].device)]

    def op_ReduceMax(self, x, at, ins):
        axes = self._ints(ins[1], x[1])
        return torch.amax(x[0], dim=axes, keepdim=bool(at.get("keepdims", 1)))

    def op_ReduceMean(self, x, at, ins):
        axes = self._ints(ins[1], x[1])
        return torch.mean(x[0], dim=axes, keepdim=bool(at.get("keepdims", 1)))

    def op_Transpose(self, x, at, ins):
        return x[0].permute(*at["perm"])

    def _bin(self, x):
        a, b = x
        if a.dtype != b.dtype:  # a trainable fp32 master or a float constant meets the compute path
            a, b = a.to(self.dtype), b.to(self.dtype)
        return a, b

    def op_Add(self, x, at, ins):
        a, b = self._bin(x)
        return a + b

    def op_Sub(self, x, at, ins):
        a, b = self._bin(x)
        return a - b

    def op_Mul(self, x, at, ins):
        a, b = self._bin(x)
        return a * b

    def op_Div(self, x, at, ins):
        a, b = self._bin(x)
        return a / b

    def op_Conv(self, x, at, ins):
        p = at.get("pads", [0, 0, 0, 0])
        assert p[:2] == p[2:], p
        w = x[1].to(x[0].dtype)
        b = x[2].to(x[0].dtype) if len(x) > 2 else None
        return F.conv2d(x[0], w, b, at.get("strides", [1, 1]), p[:2], at.get("dilations", [1, 1]), at.get("group", 1))

    def op_LayerNormalization(self, x, at, ins):
        assert at.get("axis", -1) == -1
        y = x[0]
        n = y.shape[-1]
        out = F.layer_norm(y.float(), (n,), x[1].float(), x[2].float() if len(x) > 2 else None, at["epsilon"])
        return out.to(y.dtype)

    def _lora(self, name, w):
        k = _key(name)
        if k + "__A" not in self.lora:
            return w
        d = self.lora[k + "__A"] @ self.lora[k + "__B"] * self.lora_scale[name]
        return w + d.to(w.dtype)

    def op_MatMul(self, x, at, ins):
        a, b = x
        if ins[1] in self.fnames:
            b = self._lora(ins[1], b)
        return torch.matmul(a, b.to(a.dtype))

    def op_Gemm(self, x, at, ins):
        a, b = x[0], x[1]
        if ins[1] in self.fnames:
            b = self._lora(ins[1], b)
        if at.get("transA", 0):
            a = a.transpose(-1, -2)
        if at.get("transB", 0):
            b = b.transpose(-1, -2)
        y = at.get("alpha", 1.0) * torch.matmul(a, b.to(a.dtype))
        if len(x) > 2 and x[2] is not None:
            y = y + at.get("beta", 1.0) * x[2].to(a.dtype)
        return y

    def op_Gelu(self, x, at, ins):
        ap = at.get("approximate", b"none")
        return F.gelu(x[0], approximate=ap.decode() if isinstance(ap, bytes) else ap)

    def op_Sigmoid(self, x, at, ins):
        return torch.sigmoid(x[0])

    def op_Softmax(self, x, at, ins):
        return torch.softmax(x[0].float(), at.get("axis", -1)).to(x[0].dtype)

    def op_Not(self, x, at, ins):
        return ~x[0]

    def op_Where(self, x, at, ins):
        c, a, b = x
        a = a.to(b.dtype) if torch.is_tensor(a) else a
        return torch.where(c.to(b.device), a.to(b.device), b)

    def op_Expand(self, x, at, ins):
        sh = self._ints(ins[1], x[1])
        return x[0].expand(torch.broadcast_shapes(tuple(x[0].shape), tuple(sh)))

    def op_Split(self, x, at, ins):
        ax = at.get("axis", 0)
        if len(x) > 1 and x[1] is not None:
            return list(torch.split(x[0], self._ints(ins[1], x[1]), ax))
        return list(torch.chunk(x[0], at["num_outputs"], ax))


# ---------------------------------------------------------------- cinque split helpers
CINQUE_VISION_IN = ("_unsafe_view", "_unsafe_view_1")        # (1, 12, 128, 256) road / wide, frames [t-4, t], uint8
CINQUE_HIDDEN = "view_39"                                    # (1, 32, 512) per-frame vision tokens = queued hidden
CINQUE_VISION_MEAN = "mean"                                  # (1, 512) pooled vision tap ("vision" in drive_backbones)
CINQUE_TEMPORAL = "select_4"                                 # (1, 512) temporal token ("temporal")
