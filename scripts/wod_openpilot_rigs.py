"""openpilot on WOD-E2E rater frames with other input constructions: the same inputs as the zero-shot exam
(scripts/wod_zeroshot_openpilot.py: FRONT / FRONT_LEFT / FRONT_RIGHT, rotation-only, nearest neighbour, 10 s warm-up,
each 10 Hz frame fed twice) plus

  hX.XX       virtual-height reprojection: the model frames are rendered as seen from a camera X.XX m above the
              road instead of the real 1.81 m WOD roof mount. Road-surface rays are re-aimed through the ground plane
              (exact for the road, objects above it are flattened); rays at or above the horizon are unchanged.
  front-only  FRONT alone (47 deg): what the model gets from a single narrow camera; the wide frame's sides
              (beyond +-23.5 deg) are black.

  run    (envs/openpilot) predictions -> $DATA_DIR/processed/wod_zeroshot/preds/op_<model>@<variant>/<frame>.npz
  score  (project venv) RFS / ADE / signed longitudinal bias per variant, paired bootstrap against `base`

    CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/openpilot/bin/python scripts/wod_openpilot_rigs.py run --model lebowski
    .venv/bin/python scripts/wod_openpilot_rigs.py score
"""
import argparse, io, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402

VARIANTS = {"base": (None, (1, 2, 3)), "h1.52": (1.52, (1, 2, 3)), "h1.22": (1.22, (1, 2, 3)),
            "h0.92": (0.92, (1, 2, 3)), "front-only": (None, (1,))}
_ctx = {}


def height_rays(rays, h_cam, h_virtual):
    """Vehicle-frame rays (x fwd, y left, z up) of a camera at h_virtual -> rays of the real camera at h_cam (same
    x, y) to the same road point; rays at or above the horizon unchanged."""
    if h_virtual is None:
        return rays
    down = -rays[..., 2]
    g = down > 1e-4
    X = rays * np.where(g, h_virtual / np.where(g, down, 1.0), 0.0)[..., None]
    X[..., 2] -= h_cam - h_virtual
    X = np.where(g[..., None], X, rays)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def _init(spans, op_calib, shard_dir, variants):
    _ctx.update(spans=spans, calib=op_calib, shard_dir=Path(shard_dir), variants=variants, maps={})


def _maps(seq):
    if seq not in _ctx["maps"]:
        cal_all = {int(c): d for c, d in _ctx["calib"][seq].items()}
        h_cam = float(np.array(cal_all[1]["extrinsic"]).reshape(4, 4)[2, 3])
        base = [(cal_all[c]["width"], cal_all[c]["height"]) for c in Z.OP_SRC]
        idx = {}
        for v in _ctx["variants"]:
            hv, cams = VARIANTS[v]
            cal = {c: cal_all[c] for c in cams}
            for k in ("road", "wide"):
                rays = height_rays(G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), h_cam, hv)
                src, U, V = G.choose_sources(np, rays, cal)
                # indices into the concatenation of all three front cameras, as the exam runner decodes them
                src = np.where(src >= 0, np.array(cams)[np.maximum(src, 0)] - 1, -1)
                idx[(v, k)] = G.nn_gather_index(src, U, V, base).ravel()
        _ctx["maps"] = {seq: idx}
    return _ctx["maps"][seq]


def _pack(ycc):
    Y = ycc[..., 0]
    uv = np.rint(ycc[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))).astype(np.uint8)
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])


def model_frames(name):
    """{variant: (n, 2, 6, 128, 256)} for the history of target `name` (same history as the exam runner)."""
    from PIL import Image
    spans = _ctx["spans"]
    names = [n for n in Z.history_names(name, Z.OP_WARMUP) if n in spans]
    idx = _maps(name.rsplit("-", 1)[0])
    out = {v: np.empty((len(names), 2, 6, 128, 256), np.uint8) for v in _ctx["variants"]}
    black = np.array([[0, 128, 128]], np.uint8)          # uncovered pixels: YCbCr black (the exam's RGB 0)
    fh = {}
    for j, n in enumerate(names):
        sp = spans[n]
        f = fh.get(sp[0]) or fh.setdefault(sp[0], open(_ctx["shard_dir"] / sp[0], "rb"))
        planes = []
        for k in range(3):
            f.seek(sp[1 + 2 * k])
            im = Image.open(io.BytesIO(f.read(sp[2 + 2 * k])))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes + [black])           # index -1 -> the black pixel
        for v in _ctx["variants"]:
            for m, k in enumerate(("road", "wide")):
                out[v][j, m] = _pack(cat[idx[(v, k)]].reshape(G.OP_H, G.OP_W, 3))
    for f in fh.values():
        f.close()
    return name, names, out


def cmd_run(a):
    from wod_zeroshot_openpilot import ACTION_T, MODELS, run_one
    from jevdrive.common import data_dir
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    from jevdrive.runlog import RunLog
    log = RunLog("wod_zeroshot", f"op_rigs_{a.model}")
    sets = Z.load_sets()
    spans, _ = Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    outdir = {v: Z.root("preds", f"op_{a.model}@{v}") for v in a.variants}
    todo = sorted(n for n in map(str, sets["rater"]["name"]) if not all((outdir[v] / f"{n}.npz").exists()
                                                                         for v in a.variants))[: a.limit or None]
    m = OPModel(a.model, MODELS[a.model], context_rate=(a.model == "lebowski"))
    log.info(f"{len(todo)} targets, model {a.model}, variants {a.variants}")
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(spans, op_calib, str(shard_dir), a.variants)) as ex:
        for i, (name, names, frames) in enumerate(ex.map(model_frames, todo, chunksize=1)):
            dev = np.array(op_calib[name.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
            for v in a.variants:
                wod, d = run_one(m, name, names, frames[v], dev, T_IDXS, decode)
                np.savez(outdir[v] / f"{name}.npz", wod=wod, plan_pos=d["plan_pos"], plan_yaw=d["plan_yaw"],
                         n_hist=len(names))
            if i == 0:
                np.savez_compressed(log.dir / f"model_frames_{name}.npz", **{v: f[-1] for v, f in frames.items()})
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                log.info(f"[{i + 1}/{len(todo)}] {el / (i + 1):.2f} s/target, ETA {(len(todo) - i - 1) * el / (i + 1) / 60:.0f} min")
    log.info(f"done in {time.time() - t0:.0f} s")
    log.close()


def cmd_score(a):
    import pandas as pd
    from jevdrive import waymo as W
    from jevdrive.runlog import RunLog
    log = RunLog("wod_zeroshot", "op_rigs_score")
    r = Z.load_sets()["rater"]
    names = [str(x) for x in r["name"]]
    traj, sc, cl = r["traj"], r["scores"], r["cluster"].astype(str)
    speed, log_xy = W.init_speed(r["past"]), r["future"][..., :2]
    best = traj[np.arange(len(traj)), sc.argmax(1)]
    rng = np.random.default_rng(0)
    fidx = rng.integers(0, len(names), (a.boot, len(names)))
    rows, per = [], {}
    for d in sorted(Z.root("preds").glob("op_*@*")):
        if not all((d / f"{n}.npz").exists() for n in names):
            continue
        p = np.stack([np.load(d / f"{n}.npz")["wod"] for n in names])
        rfs = W.rater_feedback_score(p, traj, sc, speed, details=True)[0]
        per[d.name] = dict(rfs=rfs, ade5=np.linalg.norm(p - best, axis=-1).mean(-1),
                           lon5=p[:, -1, 0] - log_xy[:, -1, 0], v0=speed)
    for k, q in per.items():
        model = k.split("@")[0]
        ref = per.get(f"{model}@base")
        row = dict(row=k, n=len(names), rfs=float(pd.Series(q["rfs"]).groupby(cl).mean().mean()),
                   rfs_frame=float(q["rfs"].mean()), ade5_rater=float(q["ade5"].mean()),
                   lon5_bias=float(q["lon5"].mean()), lon5_bias_moving=float(q["lon5"][speed > 5].mean()))
        if ref is not None:
            dd = q["rfs"] - ref["rfs"]
            bs = dd[fidx].mean(1)
            row |= dict(d_rfs_frame=float(dd.mean()), d_lo=float(np.percentile(bs, 2.5)),
                        d_hi=float(np.percentile(bs, 97.5)))
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(log.dir / "results.csv", index=False)
    log.info("\n" + res.round(3).to_string())
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--model", default="lebowski")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("score")
    p.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)
