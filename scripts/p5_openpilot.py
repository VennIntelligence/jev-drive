"""openpilot `temporal` on the P5 CARLA pairs: feature runner (openpilot venv).
Pre-registration: todos/2026-09-25-openpilot-temporal-p5-and-route.md (experiment 1). Plan from
`python -m jevdrive.p5_openpilot prepare` (processed/carla_p5/op_plan.json).

Every recorded run is one stream from a zero state, like a car engaged at the start of the route. The three P4-rig
JPEGs are rendered into openpilot's road / wide model frames by the WOD extraction's renderer
(wod_zeroshot_openpilot._maps / _pack; only the calibration record differs). Frames arrive at 5 Hz, the models'
context rate: Cinque gets each frame held for 4 steps of its 20 Hz clock (the rate study's `ctx5-hold`, bit-identical
to native at the output phase), Lebowski one context-rate step per frame (OPModel(context_rate=True)).
Output: processed/carla_p5/op_streams/<model>/<stream>.npz (name, temporal, hist), resumable.

  CUDA_VISIBLE_DEVICES=2 python scripts/p5_openpilot.py --shard 0/2 --workers 6
  python scripts/p5_openpilot.py --check 16      # equivalence checks before the batch
"""
import argparse, io, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wod_zeroshot_openpilot as WZ  # noqa: E402
from drive_backbones_openpilot import bounded_map  # noqa: E402
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import p5_openpilot as P  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

SEQ = "carla"
HOLD = 4          # 20 Hz steps per 5 Hz frame for the queued (20 Hz clock) models


def render(files: list) -> np.ndarray:
    """Packed (n, 2, 6, 128, 256) [road, wide] model frames from (front, front_left, front_right) JPEG triplets."""
    from PIL import Image
    idx = WZ._maps(SEQ)
    out = np.empty((len(files), 2, 6, 128, 256), np.uint8)
    for j, trip in enumerate(files):
        planes = []
        for f in trip:
            im = Image.open(io.BytesIO(Path(f).read_bytes()))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes)
        for m, k in enumerate(("road", "wide")):
            out[j, m] = WZ._pack(cat[idx[k]].reshape(G.OP_H, G.OP_W, 3))
    return out


def job(st):
    return st, render(st["files"])


def run_stream(m, frames, targets) -> dict:
    """One stream through one model from a zero state; `temporal` at every target."""
    m.reset()
    tap, tset, rows = D.OP_TAPS[m.name]["temporal"], set(targets), {}
    for j in range(len(frames)):
        for _ in range(1 if m.skip == 1 else HOLD):
            m.step(frames[j], action_t=WZ.ACTION_T)
        if j in tset:
            rows[j] = m.tap_values[tap].copy()
    return rows


def save(path, st, rows):
    tg = sorted(rows)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, name=np.array([st["names"][j] for j in tg]), temporal=np.stack([rows[j] for j in tg]),
             hist=np.array(tg))
    tmp.replace(path)


def geometry_check(calib) -> dict:
    """Independent check of the calibration record: the model-frame pixels the renderer reads (camgeom, WOD
    convention) against the P4 recorder's own camera model (CARLA axes, centred pinhole render, and its
    fixed-point inverse of the Waymo distortion from scripts/p4_carla_agent.distortion_maps)."""
    rw, rh, c = 1088, 1560, P.CAM
    u, v = np.meshgrid(np.arange(c["w"], dtype=np.float64), np.arange(c["h"], dtype=np.float64))
    xd, yd = (u - c["cu"]) / c["f"], (v - c["cv"]) / c["f"]
    xu, yu = xd.copy(), yd.copy()
    for _ in range(10):
        r2 = xu * xu + yu * yu
        s = 1.0 + c["k1"] * r2 + c["k2"] * r2 * r2
        xu, yu = xd / s, yd / s
    mx, my = c["f"] * xu + (rw - 1) / 2.0, c["f"] * yu + (rh - 1) / 2.0
    out = {}
    for k in ("road", "wide"):
        rays = G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H)
        src, U, V = G.choose_sources(np, rays, {i: calib[str(i)] for i in (1, 2, 3)})
        err = []
        for i, (_, *_, yaw) in enumerate(P.RIG):
            msk = src == i
            if not msk.any():
                continue
            r = rays[msk] * np.array([1, -1, 1])                  # Waymo (y left) -> CARLA (y right)
            ps = np.radians(-yaw)                                   # CARLA yaw is clockwise
            fwd, right = np.array([np.cos(ps), np.sin(ps), 0]), np.array([-np.sin(ps), np.cos(ps), 0])
            z, x, y = r @ fwd, r @ right, -r[:, 2]
            px, py = c["f"] * x / z + (rw - 1) / 2, c["f"] * y / z + (rh - 1) / 2
            ui, vi = np.clip(np.rint(U[msk]), 0, c["w"] - 1).astype(int), np.clip(np.rint(V[msk]), 0, c["h"] - 1).astype(int)
            # the render pixel the recorder wrote into the JPEG pixel the renderer reads, vs the ray's own render pixel
            err.append(np.hypot(mx[vi, ui] - px, my[vi, ui] - py))
        e = np.concatenate(err)
        out[k] = {"covered": float((src >= 0).mean()), "per_cam": np.bincount(src.ravel() + 1, minlength=4)[1:].tolist(),
                  "render_px_err_median": float(np.median(e)), "render_px_err_max": float(e.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(P.MODELS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", type=int, default=0, help="equivalence check on this many targets, no batch")
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel
    si, sn = map(int, a.shard.split("/"))
    log = RunLog("p5_openpilot", "check" if a.check else f"stream-{si}of{sn}")
    plan = json.loads((P.root() / "op_plan.json").read_text())
    calib = plan["calib"]
    WZ._init({}, {SEQ: calib}, ".")
    outdir = {k: P.root("op_streams", k) for k in a.models}
    log.event("start", args=vars(a), taps={k: D.OP_TAPS[k]["temporal"] for k in a.models})

    if a.check:
        g = geometry_check(calib)
        log.info(f"geometry: {g}")
        log.event("geometry", **g)
        sts = [s for s in plan["streams"] if len(s["targets"]) >= 4]
        sts = [sts[i] for i in np.linspace(0, len(sts) - 1, max(1, a.check // 4)).astype(int)]
        with ProcessPoolExecutor(2, initializer=WZ._init, initargs=({}, {SEQ: calib}, ".")) as ex:
            list(ex.map(int, range(2)))
            models = {k: OPModel(k, WZ.MODELS[k], context_rate=(k == "lebowski"), taps=[D.OP_TAPS[k]["temporal"]])
                      for k in a.models}
            batch = {s["key"]: (s, fr) for s, fr in ex.map(job, sts)}
        np.save(log.dir / "model_frame_example.npy", batch[sts[0]["key"]][1][sts[0]["targets"][-1]])
        res = []
        for key, (s, fr) in batch.items():
            tg = s["targets"][-4:]
            solo = render(s["files"][: tg[-1] + 1])                  # main process, no pool, fresh model state
            assert np.array_equal(solo, fr[: tg[-1] + 1]), "pool render differs from in-process render"
            for k, m in models.items():
                full = run_stream(m, fr, s["targets"])
                again = run_stream(m, solo, tg)
                for j in tg:
                    d = float(np.abs(full[j] - again[j]).max())
                    res.append({"stream": key, "model": k, "target": s["names"][j], "max_abs_diff": d,
                                "norm": float(np.linalg.norm(full[j]))})
        log.info("16-row check: " + json.dumps(res))
        log.event("check", rows=res, max_abs_diff=max(r["max_abs_diff"] for r in res))
        return

    items = [s for s in plan["streams"][si::sn] if not all((outdir[k] / f"{s['key']}.npz").exists() for k in a.models)]
    items = items[: a.limit or None]
    del plan
    with ProcessPoolExecutor(a.workers, initializer=WZ._init, initargs=({}, {SEQ: calib}, ".")) as ex:
        list(ex.map(int, range(a.workers)))     # fork before the TensorRT sessions exist (see drive_backbones_openpilot)
        models = {k: OPModel(k, WZ.MODELS[k], context_rate=(k == "lebowski"), taps=[D.OP_TAPS[k]["temporal"]])
                  for k in a.models}
        n_frames = sum(len(s["names"]) for s in items)
        log.info(f"{len(items)} streams, {n_frames} frames, models {list(models)}, {a.workers} render workers")
        t0, tm, n, nf = time.time(), {k: 0.0 for k in models}, 0, 0
        for st, frames in bounded_map(ex, job, items, 2 * a.workers):
            for k, m in models.items():
                t = time.perf_counter()
                rows = run_stream(m, frames, st["targets"])
                tm[k] += time.perf_counter() - t
                save(outdir[k] / f"{st['key']}.npz", st, rows)
            n, nf = n + 1, nf + len(st["names"])
            if n % 20 == 0 or n == len(items):
                el = time.time() - t0
                log.info(f"[{n}/{len(items)}] {nf} frames, {1e3 * el / nf:.2f} ms/frame wall; model ms/frame "
                         + ", ".join(f"{k} {1e3 * v / nf:.2f}" for k, v in tm.items())
                         + f"; ETA {(n_frames - nf) * el / nf / 60:.0f} min")
                log.event("progress", n=n, frames=nf, wall_s=el, **{f"gpu_s_{k}": v for k, v in tm.items()})
    el = time.time() - t0
    log.event("end", streams=n, frames=nf, wall_s=el, **{f"gpu_s_{k}": v for k, v in tm.items()})
    log.info(f"done: {n} streams, {nf} frames in {el:.0f} s")


if __name__ == "__main__":
    main()
