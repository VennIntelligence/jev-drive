"""WOD-E2E zero-shot exam: openpilot runner (openpilot venv). Pre-registration:
todos/2026-09-24-zeroshot-exam/wod-e2e.md.

Per target frame f: WOD frames max(f-100, first)..f are read from the slim shards (FRONT, FRONT_LEFT,
FRONT_RIGHT), decoded straight to YCbCr, and rendered into openpilot's road / wide calib-frame model frames
(jevdrive.camgeom, nearest neighbour, chroma 2x2 mean) by a process pool. Each WOD frame is fed twice (20 Hz) to
small / Cinque from a zero state; Lebowski is stepped once per 5 Hz context step (OPModel(context_rate=True),
exact at the output phase), on frames f, f-2, ... . The last step's plan is converted to WOD rear-axle waypoints.
Predictions: $DATA_DIR/processed/wod_zeroshot/preds/op_<model>/<frame>.npz (resumable).

  python scripts/wod_zeroshot_openpilot.py --set rater extra
"""
import argparse, io, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

MODELS = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}
ACTION_T = (0.275, 0.525)
_ctx = {}


def _init(spans, op_calib, shard_dir):
    _ctx.update(spans=spans, calib=op_calib, shard_dir=Path(shard_dir), maps={})


def _maps(seq):
    if seq not in _ctx["maps"]:
        cal = {int(c): d for c, d in _ctx["calib"][seq].items()}
        cal = {c: cal[c] for c in Z.OP_SRC}
        sizes = [(cal[c]["width"], cal[c]["height"]) for c in Z.OP_SRC]
        idx = {}
        for k in ("road", "wide"):
            src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), cal)
            assert (src >= 0).all(), f"{seq} {k}: model frame not covered"
            idx[k] = G.nn_gather_index(src, U, V, sizes).ravel()
        _ctx["maps"] = {seq: idx}  # one sequence at a time per worker is enough
    return _ctx["maps"][seq]


def _pack(ycc: np.ndarray) -> np.ndarray:
    """(256, 512, 3) YCbCr -> (6, 128, 256) uint8 in frames_to_tensor's channel order."""
    Y = ycc[..., 0]
    uv = ycc[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))
    uv = np.rint(uv).astype(np.uint8)
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])


def model_frames(name: str) -> tuple[str, list, np.ndarray]:
    """History names and their packed (n, 2, 6, 128, 256) [road, wide] model frames for target `name`."""
    from PIL import Image
    spans = _ctx["spans"]
    names = [n for n in Z.history_names(name, Z.OP_WARMUP) if n in spans]
    seq = name.rsplit("-", 1)[0]
    idx = _maps(seq)
    out = np.empty((len(names), 2, 6, 128, 256), np.uint8)
    fh = {}
    for j, n in enumerate(names):
        sp = spans[n]
        f = fh.get(sp[0]) or fh.setdefault(sp[0], open(_ctx["shard_dir"] / sp[0], "rb"))
        planes = []
        for k in range(3):
            f.seek(sp[1 + 2 * k])
            im = Image.open(io.BytesIO(f.read(sp[2 + 2 * k])))
            im.draft("YCbCr", im.size)  # libjpeg's own YCbCr (BT.601 full range), no RGB round trip
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes)
        for m, k in enumerate(("road", "wide")):
            out[j, m] = _pack(cat[idx[k]].reshape(G.OP_H, G.OP_W, 3))
    for f in fh.values():
        f.close()
    return name, names, out


def run_one(m, name, names, frames, dev_xy, T_IDXS, decode):
    m.reset()
    if m.skip == 1:  # context-rate stepping: frames f, f-2, ... (0.2 s apart), oldest first
        seq = frames[::-1][::2][::-1]
        steps = [(s, 1) for s in seq]
    else:            # 20 Hz: every WOD frame twice
        steps = [(s, 2) for s in frames]
    for s, rep in steps:
        for _ in range(rep):
            raw = m.step(s, action_t=ACTION_T)
    d = decode(raw, m.slices, 10.0, ACTION_T)
    wod = Z.openpilot_to_wod(d["plan_pos"], d["plan_yaw"], T_IDXS, dev_xy)
    return wod, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=["rater"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    from jevdrive.common import data_dir
    log = RunLog("wod_zeroshot", "openpilot")
    log.event("start", args=vars(a), backends=MODELS, action_t=ACTION_T, warmup=Z.OP_WARMUP)
    sets = Z.load_sets()
    spans, _ = Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    outdir = {k: Z.root("preds", f"op_{k}") for k in a.models}
    todo = [str(n) for w in a.set for n in sets[w]["name"]]
    todo = [n for n in todo if not all((outdir[k] / f"{n}.npz").exists() for k in a.models)][: a.limit or None]
    todo.sort()  # consecutive targets of one sequence share a worker's maps more often
    models = {k: OPModel(k, MODELS[k], context_rate=(k == "lebowski")) for k in a.models}
    log.info(f"{len(todo)} targets, models {list(models)}, {a.workers} decode workers")
    t0, tm, n = time.time(), {k: 0.0 for k in models}, 0
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(spans, op_calib, str(shard_dir))) as ex:
        for name, names, frames in ex.map(model_frames, todo, chunksize=1):
            seq = name.rsplit("-", 1)[0]
            dev = np.array(op_calib[seq]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
            for k, m in models.items():
                t = time.perf_counter()
                wod, d = run_one(m, name, names, frames, dev, T_IDXS, decode)
                tm[k] += time.perf_counter() - t
                np.savez(outdir[k] / f"{name}.npz", wod=wod, plan_pos=d["plan_pos"], plan_yaw=d["plan_yaw"],
                         plan_vel=d["plan_vel"], lead=d["lead"], lead_prob=d["lead_prob"], n_hist=len(names),
                         first=names[0], dev_xy=dev)
            if n == 0:
                np.save(log.dir / f"model_frames_{name}.npy", frames[-1])
            n += 1
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                log.info(f"[{n}/{len(todo)}] {el / n:.2f} s/target wall; model s/target "
                         + ", ".join(f"{k} {v / n:.3f}" for k, v in tm.items()) + f"; ETA {(len(todo) - n) * el / n / 60:.0f} min")
                log.event("progress", n=n, s_per_target=el / n, **{f"gpu_s_{k}": v / n for k, v in tm.items()})
    log.event("end", targets=n, seconds=time.time() - t0)
    log.info(f"done: {n} targets in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
