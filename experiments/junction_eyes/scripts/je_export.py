"""junction_eyes merge export and its gates (research/next-round/junction-eyes-plan.md 5.3).

  export --tag T [--which best|last]   W' = W + A @ B * alpha / r (Conv kernels reshaped from (out, in * k * k)), depthwise /
                                       unfrozen kernels replaced by their trained masters, written into the same-named
                                       initializers of a COPY of the factory ONNX -> step1/onnx/<T>.onnx (+ .json); the factory
                                       file's sha256 is checked before and after (never written)
  gate   --tag T [--n 512]            equivalence: (a) view_39 of the torch model (fp32, unmerged LoRA) vs the merged ONNX on
                                       onnxruntime (its vision subgraph, CUDA, the graph's fp16) on n pairs, against the same
                                       comparison for the factory weights (the fp16 noise floor); (b) the plan (x, y) of the first 15
                                       points along a comma1M 20 Hz stream, torch Stepper vs the merged ONNX stepped by OPModel
                                       (line 0.5 m), with the factory pair as the floor
  serve0                               G-serve: the B = 0 model merged -> every changed initializer byte-identical to the factory,
                                       and its plans along the stream bit-identical to the factory ONNX's (onnxruntime CPU, eager)
Outputs: step1/gates/{equiv-<T>,serve0}.json.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "experiments/junction_eyes/lib")]
import argparse, hashlib, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import je_model as M  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.op_torch import write_merged  # noqa: E402

STEP1 = data_dir() / "runs" / "junction_eyes" / "step1"
BASE = A.MODELS_DIR / A.FILES["cinque"]
AT = (0.275, 0.525)
NPT = 15


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_path(tag, which="best"):
    return STEP1 / "ckpt" / tag / f"{which}.pt"


def export(m: M.JE, out: _pl.Path, info: dict) -> dict:
    h0 = sha(BASE)
    w = m.net.merged()
    err = write_merged(BASE, out, w)
    h1 = sha(BASE)
    assert h0 == h1, "the factory ONNX changed"
    rec = info | {"base": str(BASE), "base_sha256": h0, "out": str(out), "out_sha256": sha(out), "n_weights": len(w),
                  "max_fp16_rounding": max(err.values()) if err else 0.0}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    return rec


def cmd_export(a):
    from jevdrive.run import Run
    with Run("junction_eyes", f"step1/export-{a.tag}", config=vars(a)) as run:
        m, ck = M.load_ckpt(ckpt_path(a.tag, a.which), "cpu", torch.float32, heads=False)
        rec = export(m, STEP1 / "onnx" / f"{a.tag}.onnx", {"tag": a.tag, "which": a.which, "step": ck["step"]})
        run.summary.update(rec)
        run.info(json.dumps(rec))


# ---------------------------------------------------------------- gates
def vision_onnx(src: _pl.Path) -> _pl.Path:
    """The vision subgraph (_unsafe_view, _unsafe_view_1 -> view_39) of an ONNX file, beside it."""
    import onnx.utils
    dst = src.with_name(src.stem + ".vision.onnx")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        onnx.utils.extract_model(str(src), str(dst), list(A.VISION_IN), ["view_39"])
    return dst


def ort_vision(path, prev, cur):
    import onnxruntime as ort
    from jevdrive.openpilot import model as OM  # noqa: F401  (preloads the CUDA libraries for onnxruntime)
    s = ort.InferenceSession(str(vision_onnx(path)), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    pair = np.concatenate([prev, cur], 2)                                  # (n, 2, 12, 128, 256)
    return np.stack([s.run(None, {A.VISION_IN[0]: p[0:1], A.VISION_IN[1]: p[1:2]})[0].reshape(32, 512) for p in pair]).astype(np.float32)


def torch_vision(net, prev, cur, dev):
    with torch.no_grad():
        p, c = torch.from_numpy(prev).to(dev), torch.from_numpy(cur).to(dev)
        return torch.cat([net.run_batched(A.vision_feeds(p[i:i + 32], c[i:i + 32]), ["view_39"])["view_39"].reshape(-1, 32, 512)
                          for i in range(0, len(p), 32)]).float().cpu().numpy()


def stream(n_steps: int, seg: str = ""):
    """20 Hz model frames (n, 2, 6, 128, 256) of one comma1M segment."""
    import je_data as D
    root = data_dir() / "datasets" / "comma1M"
    if not seg:                                   # default: the first held-out keep segment (never trained on) with both cameras
        from jevdrive.data import splits
        seg = [x for x in sorted(splits.load("comma1m/junction-eyes-heldout")) if (root / x / "ecamera.hevc").exists()][0]
    return D.comma_frames(seg, n_steps), seg


def plans_torch(net, frames):
    st = A.Stepper(net, "cinque")
    raw = st.run_stream(frames, AT, repeat=1)[0]
    return raw[:, A.plan_index(net.slices)].reshape(len(frames), 33, 15)


def plans_ort(path, frames, backend="cuda"):
    from jevdrive.openpilot.model import OPModel
    m = OPModel(str(path), backend)
    m.reset()
    pi = A.plan_index(m.slices)
    return np.stack([m.step(x, desire=np.zeros(8), traffic=(1, 0), action_t=AT)[pi].reshape(33, 15) for x in frames])


def pairs_from_stream(frames, n):
    """n (prev, cur) pairs 4 steps apart along the stream."""
    t = np.linspace(4, len(frames) - 1, n).astype(int)
    return frames[t - 4], frames[t]


def dstat(x, y):
    d = np.abs(x - y)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), ref_rms=float(np.sqrt((y ** 2).mean())))


def cmd_gate(a):
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("junction_eyes", f"step1/gate-{a.tag}", config=vars(a)) as run:
        m, ck = M.load_ckpt(ckpt_path(a.tag, a.which), dev, torch.float32, heads=False)
        onx = STEP1 / "onnx" / f"{a.tag}.onnx"
        if not onx.exists():
            export(m, onx, {"tag": a.tag, "which": a.which, "step": ck["step"]})
        frames, seg = stream(max(a.steps, 1200), a.seg)
        prev, cur = pairs_from_stream(frames, a.n)
        res = {"tag": a.tag, "step": ck["step"], "n_pairs": a.n, "segment": seg, "stream_steps": a.steps}
        f0 = M.JE("none", dtype=torch.float32, heads=False).to(dev)
        hv = {"arm_torch": torch_vision(m.net, prev, cur, dev), "arm_onnx": ort_vision(onx, prev, cur),
              "fac_torch": torch_vision(f0.net, prev, cur, dev), "fac_onnx": ort_vision(BASE, prev, cur)}
        res["view39_arm_torch_vs_onnx"] = dstat(hv["arm_torch"], hv["arm_onnx"])
        res["view39_factory_torch_vs_onnx"] = dstat(hv["fac_torch"], hv["fac_onnx"])
        res["view39_arm_vs_factory_torch"] = dstat(hv["arm_torch"], hv["fac_torch"])            # how far training moved the tokens
        fr = frames[:a.steps]
        pa_t, pa_o = plans_torch(m.net, fr), plans_ort(onx, fr)
        pf_t, pf_o = plans_torch(f0.net, fr), plans_ort(BASE, fr)
        dxy = lambda x, y: np.abs(x[:, :NPT, :2] - y[:, :NPT, :2]).max((1, 2))  # noqa: E731
        for k, (x, y) in {"plan_arm_torch_vs_onnx": (pa_t, pa_o), "plan_factory_torch_vs_onnx": (pf_t, pf_o),
                          "plan_arm_vs_factory_onnx": (pa_o, pf_o)}.items():
            e = dxy(x, y)
            res[k] = dict(max_m=float(e.max()), median_m=float(np.median(e)), share_over_0p5=float((e > 0.5).mean()))
        r, f = res["view39_arm_torch_vs_onnx"], res["view39_factory_torch_vs_onnx"]
        res["pass_view39"] = bool(r["mean_abs"] <= 2 * f["mean_abs"] + 1e-6 and r["max_abs"] <= 2 * f["max_abs"] + 1e-6)
        res["pass_plan"] = bool(res["plan_arm_torch_vs_onnx"]["max_m"] <= 0.5)
        res["pass"] = res["pass_view39"] and res["pass_plan"]
        out = STEP1 / "gates" / f"equiv-{a.tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        run.info(json.dumps(res))


def cmd_serve0(a):
    """G-serve: the arm at its start (B = 0, depthwise masters = factory) merged and served."""
    import onnx
    from onnx import numpy_helper
    from jevdrive.run import Run
    with Run("junction_eyes", "step1/serve0", config=vars(a)) as run:
        m = M.JE("lora", dtype=torch.float32, heads=False)
        onx = STEP1 / "onnx" / "JE-zero.onnx"
        rec = export(m, onx, {"tag": "JE-zero"})
        g0 = {t.name: t for t in onnx.load(str(BASE)).graph.initializer}
        g1 = {t.name: t for t in onnx.load(str(onx)).graph.initializer}
        changed = set(m.net.merged())
        same = [numpy_helper.to_array(g0[k]).tobytes() == numpy_helper.to_array(g1[k]).tobytes() for k in changed]
        res = {"n_changed_initializers": len(changed), "bytes_identical": int(sum(same)), "export": rec}
        fr, seg = stream(a.steps, a.seg)
        res["segment"] = seg
        for be in ("cpu", "cuda"):
            p0, p1 = plans_ort(BASE, fr, be), plans_ort(onx, fr, be)
            e = np.abs(p0 - p1).max((1, 2))
            res[f"plans_{be}"] = dict(steps=len(fr), bitwise_identical_steps=int((e == 0).sum()), max_abs=float(e.max()))
        res["pass"] = bool(res["bytes_identical"] == len(changed) and res["plans_cpu"]["bitwise_identical_steps"] == len(fr))
        out = STEP1 / "gates" / "serve0.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        run.info(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "gate", "serve0"):
        p = sp.add_parser(name)
        if name != "serve0":
            p.add_argument("--tag", required=True)
            p.add_argument("--which", default="best", choices=("best", "last"))
        if name != "export":
            p.add_argument("--steps", type=int, default=400, help="20 Hz steps of the comma1M stream")
            p.add_argument("--seg", default="")
        if name == "gate":
            p.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    {"export": cmd_export, "gate": cmd_gate, "serve0": cmd_serve0}[a.cmd](a)
