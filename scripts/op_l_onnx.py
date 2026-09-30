#!/usr/bin/env python
"""op-adapt L checkpoints as serving ONNX files for the closed-loop server (todos/2026-10-01-op-adapt-L-b2d-prereg.md).

The training port (jevdrive.op_adapt_l.LModel) trains fp32 copies of stage 4 (and optionally the policy pathway) and an intent
adapter `H' = H + E[intent]` on the hidden tokens of all 9 context frames. The closed loop serves the stock queued ONNX on
onnxruntime (scripts/op_arb_server.py), so a checkpoint becomes a copy of `cinque.ort.onnx` with
  * the trained initializers replaced (cast back to the graph's fp16),
  * an extra input `intent_bias` (1, 32, 512) fp16 = E[intent] (zeros = no intent), added to the current frame's hidden tokens
    (`view_39`, the policy's input and its pooled `mean`) and to the 8 past slots of the policy's 33-slot queue (`_to_copy_1`),
    but NOT to the copy of `view_39` that goes back into the model's own queue, so every step adds the CURRENT intent to all
    context frames, as in training (the 24 unused past slots stay zero).
`E.npy` (4, 32, 512) fp16 is written next to the ONNX; the server picks E[intent] per request.

  build  (op-train env)    op_l_onnx.py build --ckpt CKPT --out DIR/name.onnx [--no-adapter]
  ref    (op-train env)    op_l_onnx.py ref --ckpt CKPT|none --out ref.npz     port outputs on the stored WOD frame streams
  check  (openpilot env)   op_l_onnx.py check --onnx DIR/name.onnx --ref ref.npz   onnxruntime outputs of the served model vs the port
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
BASE = Path.home() / "data/models/openpilot/cinque.ort.onnx"
INTENT_PATTERN = np.array([1, 1, 2, 2, 3, 3, 0, 1], np.int64)       # cycled over the frames of a check stream


def _key(name):
    return name.replace(".", "__").replace("/", "_")


def build(a):
    import onnx
    import torch
    from onnx import TensorProto as TP, helper, numpy_helper
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st = ck["model"]
    m = onnx.load(str(BASE))
    g = m.graph
    inits = {_key(t.name): t for t in g.initializer}
    for k, v in st["net"].items():
        t = inits[k]
        dt = helper.tensor_dtype_to_np_dtype(t.data_type)
        arr = v.numpy().astype(dt)
        assert arr.shape == tuple(t.dims), (k, arr.shape, t.dims)
        t.CopyFrom(numpy_helper.from_array(arr, t.name))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if st.get("adapter") is not None and not a.no_adapter:
        E = (st["adapter"]["E"] * st["adapter"]["on"]).numpy().astype(np.float16)       # intent 0 adds nothing
        names = {n.output[0]: i for i, n in enumerate(g.node)}
        cur, past = names["unsqueeze_1"], names["cat_3"]
        assert g.node[cur].input[0] == "view_39" and g.node[names["mean"]].input[0] == "view_39"
        assert list(g.node[past].input) == ["_to_copy_1", "unsqueeze_1"]
        g.input.append(helper.make_tensor_value_info("intent_bias", TP.FLOAT16, [1, 32, 512]))
        mask = np.zeros((1, 32, 1, 1), np.float16)
        mask[:, 24:] = 1                                                                 # the 8 real past slots
        g.initializer.extend([numpy_helper.from_array(mask, "l_mask"),
                              numpy_helper.from_array(np.array([1, 1, 32, 512], np.int64), "l_shape")])
        new = [helper.make_node("Add", ["view_39", "intent_bias"], ["l_view_39"]),
               helper.make_node("Reshape", ["intent_bias", "l_shape"], ["l_bias4"]),
               helper.make_node("Mul", ["l_mask", "l_bias4"], ["l_bias_past"]),
               helper.make_node("Add", ["_to_copy_1", "l_bias_past"], ["l_past"])]
        nodes = list(g.node)
        nodes[cur].input[0] = "l_view_39"
        nodes[names["mean"]].input[0] = "l_view_39"
        nodes[past].input[0] = "l_past"
        first = min(cur, names["mean"], past)
        assert max(names["view_39"], names["_to_copy_1"]) < first
        del g.node[:]
        g.node.extend(nodes[:first] + new + nodes[first:])
        np.save(out.with_suffix(".E.npy"), E)
    onnx.save(m, str(out))
    print("wrote", out, "adapter" if out.with_suffix(".E.npy").exists() and not a.no_adapter else "no adapter", "trained tensors", len(st["net"]))


def ref(a):
    import torch
    from jevdrive import op_adapt as A
    from jevdrive import op_adapt_l as L
    dev = torch.device("cuda")
    model = L.load_model(None if a.ckpt == "none" else a.ckpt, dev)
    res = {}
    ref_dir = Path.home() / "data/runs/op_adapt/ref"
    for k in (0, 1):
        frames = np.load(ref_dir / f"frames_{k}.npz")["frames"]
        n = len(frames)
        intent = INTENT_PATTERN[(np.arange(n) // 3) % len(INTENT_PATTERN)]
        fr = torch.as_tensor(frames).to(dev)
        prev = torch.cat([torch.zeros_like(fr[:2]), fr[:-2]])
        with torch.no_grad():
            H = torch.cat([model.net.run_batched(A.vision_feeds(prev[i:i + 32], fr[i:i + 32]), ["view_39"])["view_39"][:, 0]
                           for i in range(0, n, 32)])
            Hp = torch.cat([torch.zeros(2 * (A.CONTEXT - 1), *A.H_SHAPE, dtype=H.dtype, device=dev), H])
            idx = torch.arange(n, device=dev)[:, None] + torch.arange(0, 2 * A.CONTEXT, 2, device=dev)[None]
            valid = (idx >= 2 * (A.CONTEXT - 1)).to(dev)
            tc = torch.tensor([[1.0, 0.0]], device=dev).expand(32, 2)
            outs = []
            for i in range(0, n, 32):
                ctx = Hp[idx[i:i + 32]]
                o = model.policy(ctx, valid[i:i + 32], tc[:len(ctx)], torch.from_numpy(intent[i:i + 32]).to(dev))
                outs.append(o["outputs"].float().cpu().numpy())
        res[f"out_{k}"], res[f"intent_{k}"] = np.concatenate(outs), intent
    np.savez(a.out, **res)
    print("wrote", a.out, {k: v.shape for k, v in res.items()})


def check(a):
    from jevdrive.openpilot.model import OPModel
    z = np.load(a.ref)
    E = np.load(Path(a.onnx).with_suffix(".E.npy")) if Path(a.onnx).with_suffix(".E.npy").exists() else None
    m = OPModel(str(a.onnx) if "/" in str(a.onnx) else a.onnx, a.backend)
    sl = m.slices
    cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in ("plan", "lead", "lead_prob", "desire_state", "meta")])
    ref_dir = Path.home() / "data/runs/op_adapt/ref"
    rows = []
    for k in (0, 1):
        frames = np.load(ref_dir / f"frames_{k}.npz")["frames"]
        intent = z[f"intent_{k}"]
        m.reset()
        got = []
        for f, fr in enumerate(frames):
            m.extra = {"intent_bias": E[int(intent[f])][None]} if E is not None else {}
            m.step(fr)
            got.append(m.step(fr))                        # recorded at the frame's second 20 Hz step, as the stream protocol
        got = np.stack(got)[:, cols]
        ref = z[f"out_{k}"][:, cols]
        ok = np.arange(len(frames)) >= 2 * 8
        d = np.abs(got[ok] - ref[ok])
        pl = np.arange(sl["plan"].start, sl["plan"].start + 495)
        pg = got[ok][:, :495].reshape(-1, 33, 15)[:, :, :2] - ref[ok][:, :495].reshape(-1, 33, 15)[:, :, :2]
        rows.append((k, int(ok.sum()), float(d.max()), float(np.percentile(d, 99)), float(np.abs(pg).max()), float(np.linalg.norm(pg, axis=-1).mean())))
        print("stream %d frames %d | all cols max %.4f p99 %.5f | plan xy max %.4f m, mean dist %.5f m" % rows[-1])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--no-adapter", action="store_true")
    r = sp.add_parser("ref")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--onnx", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--backend", default="cuda-iob")
    a = ap.parse_args()
    {"build": build, "ref": ref, "check": check}[a.cmd](a)
