"""A guard candidate (serving ONNX) as the op-adapt training port, for lines that read stored stage-3 trunk banks (op-train env).

The port (`experiments.op_adapt_l.lib.op_adapt_l.LModel`, a graph port of cinque.ort.onnx) is what op_adapt_L / op_adapt_H / op_img_cmd
used for their WOD-val capture / false-start and no-overlay drift readouts; it reads the original model's stage-3 outputs from disk
(`$DATA_DIR/runs/op_adapt_L/t/trunk`, `runs/op_img_cmd/ft/bank`). A serving ONNX made by op_l_onnx.py is cinque.ort.onnx with some
initializers replaced (+ an optional `intent_bias` input, zero here = no command). This module copies every initializer that differs
from the shipped graph into the port, so any such candidate runs on the port without its checkpoint. A candidate that changes a
weight below stage 4 (the trunk the banks store) cannot be read from the banks: `load` refuses it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _d in ("", "scripts", "experiments/op_adapt_r2/lib", "experiments/op_adapt_l/scripts"):
    if str(REPO / _d) not in sys.path:
        sys.path.insert(0, str(REPO / _d))


def changed_weights(onnx_path) -> dict:
    """{initializer name: float32 array} of the candidate's initializers that differ from the shipped cinque.ort.onnx."""
    import onnx
    from onnx import numpy_helper
    from jevdrive import op_adapt as A
    base = {t.name: t for t in onnx.load(str(A.MODELS_DIR / A.FILES["cinque"]), load_external_data=False).graph.initializer}
    out = {}
    for t in onnx.load(str(onnx_path)).graph.initializer:
        b = base.get(t.name)
        a = numpy_helper.to_array(t)
        if a.dtype.kind != "f" or (b is None and t.name == "l_mask"):     # l_mask: op_l_onnx's intent_bias plumbing, not a weight
            continue
        if b is None or b.raw_data != t.raw_data or not np.array_equal(numpy_helper.to_array(b), a):
            out[t.name] = a.astype(np.float32)
    return out


def trunk_weights() -> set:
    """Float initializers consumed below stage 4 (first node .. the producer of the stored trunk tensor)."""
    import onnx
    from jevdrive import op_adapt as A
    g = onnx.load(str(A.MODELS_DIR / A.FILES["cinque"]), load_external_data=False).graph
    return set(A.node_weights(A.MODELS_DIR / A.FILES["cinque"], g.node[0].output[0], A.TRUNK_OUT))


def load(onnx_path, dev):
    """Port model of a candidate: None = the shipped model (LModel default, may read the original's H cache), else the shipped port
    with stage 4 marked trainable (so hidden tokens are recomputed from the trunks, never read from the original's H cache) and the
    candidate's changed initializers copied in. Returns (model, info dict)."""
    import torch
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from jevdrive.op_torch import _key
    if onnx_path is None:
        return L.load_model(None, dev), {"port": "shipped", "changed": 0}
    ch = changed_weights(onnx_path)
    bad = sorted(set(ch) & trunk_weights())
    if bad:
        raise SystemExit(f"{onnx_path}: {len(bad)} changed weights below stage 4 (e.g. {bad[:3]}); the stored trunk banks are the "
                         "shipped model's, so this candidate needs re-rendered frames (not wired)")
    m = L.LModel(L.LCfg("guard", s4=True, pol=False, intent="none")).to(dev).eval()
    miss = [k for k in ch if _key(k) not in m.net.params]
    if miss:
        raise SystemExit(f"{onnx_path}: initializers not in the port graph: {miss[:5]}")
    with torch.no_grad():
        for k, a in ch.items():
            p = m.net.params[_key(k)]
            p.data.copy_(torch.from_numpy(a).to(p.device, p.dtype))
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"port": "onnx->port", "changed": len(ch)}


def plans_from_trunks(model, T, rows, slot_valid, tc, dev, bs=96) -> np.ndarray:
    """Plan MDN mean (n, 33, 15) of bank rows: T (N, 9, 1024, 8, 16) fp16 memmap, rows (n,), slot_valid (N, 9), tc (N, 2).
    The same forward as experiments/op_img_cmd/scripts/img2_eval.py cmd_plans."""
    import torch
    from jevdrive import op_adapt as A
    pi = A.plan_index(model.net.slices)
    rows = np.asarray(rows)
    mu = np.zeros((len(rows), 33, 15), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            o = model(torch.from_numpy(np.stack([np.asarray(T[k]) for k in r])).to(dev), torch.from_numpy(slot_valid[r]).to(dev),
                      torch.from_numpy(np.asarray(tc[r], np.float32)).to(dev).half())["outputs"].float()
            mu[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
    return mu
