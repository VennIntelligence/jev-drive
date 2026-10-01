"""op-adapt step 1: onnxruntime reference outputs for the PyTorch port (openpilot venv).

Renders a few WOD val streams with the WOD exam's renderer (scripts/drive_backbones_openpilot.render), runs cinque and
lebowski through `OPModel` exactly as the feature runs do (each WOD frame fed twice for 20 Hz, desire 0, traffic
[1, 0], ACTION_T), and stores per 20 Hz step the raw output vector and the `temporal` / `vision` taps, plus the packed
input frames, so experiments/op_adapt_r1/archive/op_adapt_equiv.py can replay the identical inputs through the port with its own recurrent
state. Backends: cpu (ORT CPU), cuda (ORT CUDA EP, the graph's fp16), trt (TensorRT fp16, what the features used).

  CUDA_VISIBLE_DEVICES=2 python experiments/op_adapt_r1/archive/op_adapt_ref.py --streams 3 --frames 150
Output: $DATA_DIR/runs/op_adapt/ref/{frames_<k>.npz, <model>_<backend>_<k>.npz}
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("scripts",)]
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import drive_backbones_openpilot as R  # noqa: E402
import wod_zeroshot_openpilot as WZ  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def fp32_model(model: str) -> str:
    """An all-float32 copy of the model (fp16 initializers, casts, inputs, outputs and value types -> float32) under
    MODELS_DIR/fp32/, so ORT CPU gives an fp32 reference; returns the OPModel name ("fp32/<model>32")."""
    import onnx
    from onnx import numpy_helper
    from jevdrive.openpilot.model import MODELS_DIR, prepare_onnx
    dst = MODELS_DIR / "fp32" / f"{model}32.onnx"
    if not dst.exists():
        m = onnx.load(str(prepare_onnx(model)))
        g = m.graph
        for t in g.initializer:
            if t.data_type == onnx.TensorProto.FLOAT16:
                t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
        for n in g.node:
            for at in n.attribute:
                if n.op_type == "Cast" and at.name == "to" and at.i == onnx.TensorProto.FLOAT16:
                    at.i = onnx.TensorProto.FLOAT
                if at.type == onnx.AttributeProto.TENSOR and at.t.data_type == onnx.TensorProto.FLOAT16:
                    at.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(at.t).astype(np.float32)))
        for v in list(g.input) + list(g.output) + list(g.value_info):
            if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                v.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        dst.parent.mkdir(exist_ok=True)
        onnx.save(m, str(dst), save_as_external_data=False)
    return f"fp32/{model}32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--frames", type=int, default=150, help="WOD frames per stream (2 model steps each)")
    ap.add_argument("--models", nargs="+", default=["cinque", "lebowski"])
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda", "trt"])
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--fp32", action="store_true", help="ORT CPU on an all-fp32 copy of the graph (backend tag cpu32)")
    a = ap.parse_args()
    if a.fp32:                      # the fp32 copy of lebowski is > 2 GB, past the protobuf limit of the tap copies
        a.backends, a.models = ["cpu32"], ["cinque"]
    from jevdrive.openpilot.model import OPModel
    out = data_dir() / "runs" / "op_adapt" / "ref"
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((D.root() / D.plan_name("subset")).read_text())
    WZ._init(plan["spans"], json.loads((Z.root() / "op_calib.json").read_text()),
             str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    seen, picks = set(), []
    for s in plan["streams"]:               # long streams from distinct sequences, spread over the plan
        if len(s["names"]) >= a.frames and s["sequence"] not in seen:
            seen.add(s["sequence"])
            picks.append(s)
    picks = picks[:: max(1, len(picks) // a.streams)][: a.streams]
    for k, s in enumerate(picks):
        names = s["names"][: a.frames]
        f = out / f"frames_{k}.npz"
        if not f.exists():
            np.savez(f, frames=R.render(names), names=np.array(names))
        frames = np.load(f)["frames"]
        for model in a.models:
            taps = D.OP_TAPS[model]
            for be in a.backends:
                dst = out / f"{model}_{be}_{k}.npz"
                if dst.exists():
                    continue
                m = OPModel(fp32_model(model) if be == "cpu32" else model, "cpu" if be == "cpu32" else be,
                            threads=a.threads, taps=[taps["temporal"], taps["vision"]])
                raw, tmp, vis = [], [], []
                t0 = time.perf_counter()
                for fr in frames:
                    for _ in range(2):
                        raw.append(m.step(fr, action_t=WZ.ACTION_T))
                        tmp.append(m.tap_values[taps["temporal"]])
                        vis.append(m.tap_values[taps["vision"]])
                dt = time.perf_counter() - t0
                np.savez(dst, raw=np.stack(raw), temporal=np.stack(tmp), vision=np.stack(vis))
                print(f"stream {k} {model} {be}: {len(raw)} steps, {1e3 * dt / len(raw):.1f} ms/step", flush=True)
                del m


if __name__ == "__main__":
    main()
