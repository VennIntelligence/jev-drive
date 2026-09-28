"""op-adapt step 1: onnxruntime reference outputs for the PyTorch port (openpilot venv).

Renders a few WOD val streams with the WOD exam's renderer (scripts/drive_backbones_openpilot.render), runs cinque and
lebowski through `OPModel` exactly as the feature runs do (each WOD frame fed twice for 20 Hz, desire 0, traffic
[1, 0], ACTION_T), and stores per 20 Hz step the raw output vector and the `temporal` / `vision` taps, plus the packed
input frames, so scripts/op_adapt_equiv.py can replay the identical inputs through the port with its own recurrent
state. Backends: cpu (ORT CPU), cuda (ORT CUDA EP, the graph's fp16), trt (TensorRT fp16, what the features used).

  CUDA_VISIBLE_DEVICES=2 python scripts/op_adapt_ref.py --streams 3 --frames 150
Output: $DATA_DIR/runs/op_adapt/ref/{frames_<k>.npz, <model>_<backend>_<k>.npz}
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_backbones_openpilot as R  # noqa: E402
import wod_zeroshot_openpilot as WZ  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--frames", type=int, default=150, help="WOD frames per stream (2 model steps each)")
    ap.add_argument("--models", nargs="+", default=["cinque", "lebowski"])
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda", "trt"])
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
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
                m = OPModel(model, be, threads=a.threads, taps=[taps["temporal"], taps["vision"]])
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
