"""openpilot on the WOD-E2E rater frames with NAVSIM's input timeline (todos/2026-09-25-openpilot-openloop-comparison.md, G1).
Rendering is the zero-shot exam's (scripts/wod_zeroshot_openpilot.py via wod_openpilot_rigs, `base` geometry); only the
frames fed and the step schedule change. Every variant starts from a zero state and reads the plan at t0.

  ctx1.5   the 10 Hz frames of the last 1.5 s (f-15 .. f)
  nav2hz   NAVSIM's four frames f-15, f-10, f-5, f (t = -1.5, -1.0, -0.5, 0 s)

Both run on the schedule of scripts/navsim_zs_openpilot.py: small / Cinque step the 20 Hz clock t = -1.5 .. 0 (31 steps),
Lebowski its context-rate phases t = -1.4 .. 0 (8 steps); each step shows the latest frame at or before t.
Predictions: $DATA_DIR/processed/wod_zeroshot/preds/op_<model>@<variant>/<frame>.npz (resumable).

    CUDA_VISIBLE_DEVICES=3 $DATA_DIR/envs/openpilot/bin/python scripts/wod_openpilot_timeline.py --model cinque
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wod_openpilot_rigs as R  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402

VARIANTS = {"ctx1.5": 0.1, "nav2hz": 0.5}     # frame spacing (s) within the 1.5 s window
T_HIST = 1.5


def frame_times(variant: str) -> np.ndarray:
    return -np.round(np.arange(round(T_HIST / VARIANTS[variant]), -1, -1) * VARIANTS[variant], 3)   # oldest first


def schedule(variant: str, context_rate: bool) -> list[int]:
    """Index into frame_times(variant) of the frame shown at each step (navsim_zs_openpilot.schedule, generalised)."""
    ts = np.round(np.arange(-7, 1) * 0.2, 3) if context_rate else np.round(np.arange(-30, 1) * 0.05, 3)
    ft = frame_times(variant)
    return [int(np.searchsorted(ft, t + 1e-6) - 1) for t in ts]


def render(name):
    """{variant: (n_frames, 2, 6, 128, 256)} with the exam's renderer. A history frame before the sequence's first
    indexed frame (one rater frame starts its sequence, wod-e2e.md) is replaced by that first frame, i.e. held."""
    seq, f = name.rsplit("-", 1)
    spans = R._ctx["spans"]
    first = min(int(n.rsplit("-", 1)[1]) for n in spans if n.startswith(seq + "-"))
    out = {}
    for v in VARIANTS:
        names = [f"{seq}-{max(first, int(f) + int(round(t * 10))):03d}" for t in frame_times(v)]
        assert all(n in spans for n in names), f"{name}: history frame missing inside the sequence"
        out[v] = names
    allnames = sorted({n for ns in out.values() for n in ns})
    fr = _render_names(allnames)
    at = {n: i for i, n in enumerate(allnames)}
    return name, {v: fr[[at[n] for n in ns]] for v, ns in out.items()}


def _render_names(names):
    """R.model_frames renders a target's whole exam history; this renders exactly `names` with the same code path."""
    import io
    from PIL import Image
    from jevdrive import camgeom as G
    idx = R._maps(names[0].rsplit("-", 1)[0])
    out = np.empty((len(names), 2, 6, 128, 256), np.uint8)
    black = np.array([[0, 128, 128]], np.uint8)
    fh = {}
    for j, n in enumerate(names):
        sp = R._ctx["spans"][n]
        fo = fh.get(sp[0]) or fh.setdefault(sp[0], open(R._ctx["shard_dir"] / sp[0], "rb"))
        planes = []
        for k in range(3):
            fo.seek(sp[1 + 2 * k])
            im = Image.open(io.BytesIO(fo.read(sp[2 + 2 * k])))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes + [black])
        for m, k in enumerate(("road", "wide")):
            out[j, m] = R._pack(cat[idx[("base", k)]].reshape(G.OP_H, G.OP_W, 3))
    for fo in fh.values():
        fo.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("small", "cinque", "lebowski"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from wod_zeroshot_openpilot import ACTION_T, MODELS
    from jevdrive.common import data_dir
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    from jevdrive.runlog import RunLog
    log = RunLog("wod_zeroshot", f"op_timeline_{a.model}")
    spans, _ = Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    outdir = {v: Z.root("preds", f"op_{a.model}@{v}") for v in VARIANTS}
    todo = sorted(n for n in map(str, Z.load_sets()["rater"]["name"])
                  if not all((outdir[v] / f"{n}.npz").exists() for v in VARIANTS))[: a.limit or None]
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    with ProcessPoolExecutor(a.workers, initializer=R._init, initargs=(spans, op_calib, str(shard_dir), ["base"])) as ex:
        list(ex.map(int, range(a.workers)))            # fork the renderers before the TensorRT session exists
        cr = a.model == "lebowski"
        m = OPModel(a.model, MODELS[a.model], context_rate=cr)
        sched = {v: schedule(v, cr) for v in VARIANTS}
        log.info(f"{len(todo)} targets, model {a.model}, schedules {sched}")
        t0 = time.time()
        for i, (name, frames) in enumerate(ex.map(render, todo, chunksize=1)):
            dev = np.array(op_calib[name.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
            for v, fr in frames.items():
                m.reset()
                for s in sched[v]:
                    raw = m.step(fr[s], action_t=ACTION_T)
                d = decode(raw, m.slices, 10.0, ACTION_T)
                wod = Z.openpilot_to_wod(d["plan_pos"], d["plan_yaw"], T_IDXS, dev)
                np.savez(outdir[v] / f"{name}.npz", wod=wod, plan_pos=d["plan_pos"], plan_yaw=d["plan_yaw"],
                         steps=len(sched[v]))
            if (i + 1) % 100 == 0:
                log.info(f"[{i + 1}/{len(todo)}] {(time.time() - t0) / (i + 1):.3f} s/target")
    log.info(f"done: {len(todo)} targets in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
