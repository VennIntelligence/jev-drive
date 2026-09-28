"""op-adapt step 1: numerical equivalence of the PyTorch port (jevdrive.op_torch) against onnxruntime (op-train venv).

Replays the packed frames stored by scripts/op_adapt_ref.py through the port with the port's own recurrent state
(cinque: the ONNX state_* tensors; lebowski: modeld's host queues with the port's own hidden outputs), one model step
per ORT step, and compares plan / lead / lead_prob / temporal / vision against every ORT backend. Also checks
that the training decomposition (vision per frame -> policy on the gathered 9-frame context, jevdrive.op_adapt)
reproduces the streamed outputs, and that bf16 / fp16 compute stays close.

  CUDA_VISIBLE_DEVICES=2 python scripts/op_adapt_equiv.py
Output: $DATA_DIR/runs/op_adapt/equiv/<ts>/{equiv.csv, curves.npz, log.txt, events.jsonl}
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

ACTION_T = (0.275, 0.525)


def quantities(raw, taps, slices):
    """The compared quantities of a (steps, n) raw output block: MDN means of plan / lead, lead_prob logits, taps."""
    s = lambda k: raw[:, slices[k]]  # noqa: E731
    plan = s("plan")[:, :495].reshape(-1, 33, 15)
    return {"plan_pos": plan[:, :, 0:3], "plan_vel": plan[:, :, 3:6], "plan_all": plan,
            "lead": s("lead")[:, :72], "lead_prob": s("lead_prob"), "action": s("action")[:, :2] if "action" in slices else None,
            "temporal": taps["temporal"], "vision": taps["vision"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cinque", "lebowski"])
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16", "float16"])
    a = ap.parse_args()
    log = RunLog("op_adapt", "equiv")
    ref = data_dir() / "runs" / "op_adapt" / "ref"
    streams = sorted(int(p.stem.split("_")[1]) for p in ref.glob("frames_*.npz"))
    rows, curves = [], {}
    dev = torch.device("cuda")
    for model in a.models:
        for dt in a.dtypes:
            net = A.load(model, getattr(torch, dt)).to(dev)
            st = A.Stepper(net, model)
            for k in streams:
                frames = np.load(ref / f"frames_{k}.npz")["frames"]
                t0 = time.perf_counter()
                raw, tmp, vis = st.run_stream(frames, ACTION_T)
                ms = 1e3 * (time.perf_counter() - t0) / len(raw)
                mine = quantities(raw, {"temporal": tmp, "vision": vis}, net.slices)
                for be in ("cpu", "cuda", "trt"):
                    f = ref / f"{model}_{be}_{k}.npz"
                    if not f.exists():
                        continue
                    r = np.load(f)
                    theirs = quantities(r["raw"], {"temporal": r["temporal"], "vision": r["vision"]}, net.slices)
                    for q in mine:
                        if mine[q] is None:
                            continue
                        d = np.abs(mine[q] - theirs[q]).reshape(len(raw), -1)
                        per_step = d.max(1)
                        scale = float(np.percentile(np.abs(theirs[q]), 99))
                        rows.append(dict(model=model, dtype=dt, stream=k, ref=be, q=q, steps=len(raw),
                                         max_abs=float(d.max()), p99_abs=float(np.percentile(d, 99)),
                                         mean_abs=float(d.mean()), ref_p99=scale,
                                         first50_max=float(per_step[:50].max()), last50_max=float(per_step[-50:].max()),
                                         ms_per_step=ms))
                        curves[f"{model}/{dt}/{k}/{be}/{q}"] = per_step
                log.info(f"{model} {dt} stream {k}: {len(raw)} steps, {ms:.1f} ms/step; plan_pos max|d| vs cpu "
                         + f"{[r['max_abs'] for r in rows if r['model'] == model and r['dtype'] == dt and r['stream'] == k and r['ref'] == 'cpu' and r['q'] == 'plan_pos']}")
            # training decomposition == stream (same dtype, same frames)
            if model == "cinque":
                for k in streams:
                    frames = np.load(ref / f"frames_{k}.npz")["frames"]
                    raw, tmp, vis = st.run_stream(frames, ACTION_T)
                    raw_d, tmp_d = A.decomposed_wod(net, frames, ACTION_T, dev)
                    ok = slice(A.CONTEXT_FRAMES_WOD, None)        # frames with the full 1.6 s context
                    rs = raw[1::2][ok]
                    for q, x, y in (("plan_pos", quantities(rs, {"temporal": tmp[1::2][ok], "vision": vis[1::2][ok]}, net.slices)["plan_pos"],
                                     quantities(raw_d[ok], {"temporal": tmp_d[ok], "vision": tmp_d[ok]}, net.slices)["plan_pos"]),
                                    ("temporal", tmp[1::2][ok], tmp_d[ok])):
                        d = np.abs(x - y)
                        rows.append(dict(model=model, dtype=dt, stream=k, ref="decomposed", q=q, steps=len(x),
                                         max_abs=float(d.max()), p99_abs=float(np.percentile(d, 99)), mean_abs=float(d.mean()),
                                         ref_p99=float(np.percentile(np.abs(y), 99))))
            del net, st
            torch.cuda.empty_cache()
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(log.dir / "equiv.csv", index=False)
    np.savez(log.dir / "curves.npz", **curves)
    summ = (df.groupby(["model", "dtype", "ref", "q"])[["max_abs", "p99_abs", "ref_p99", "first50_max", "last50_max"]]
            .max().reset_index())
    log.info("\n" + summ.to_string())
    log.event("end", n=len(df))


if __name__ == "__main__":
    main()
