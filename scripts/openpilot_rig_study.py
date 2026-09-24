"""openpilot camera-rig robustness on real comma video (comma1M): how much do the plan errors grow when the model
frames come from another rig, mount or image pipeline instead of the comma road + wide cameras?

  frames  one segment: decode fcamera/ecamera once, render every image variant's packed model frames
          (jevdrive.openpilot.rigsim) into $DATA_DIR/runs/openpilot_rigs/frames/<sid>/<variant>.npy
  eval    run small / Cinque / Lebowski on every (segment, variant), plus the timing variants that only re-index
          the native frames, and score them like scripts/openpilot_replay.py (plan vs the localizer's future)
  sheet   one sample model frame (road | wide luma) per variant, for the figure

    PY=$DATA_DIR/envs/openpilot/bin/python
    for s in $(...segments...); do $PY scripts/openpilot_rig_study.py frames --seg $s; done
    CUDA_VISIBLE_DEVICES=0 $PY scripts/openpilot_rig_study.py eval
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive.openpilot import rigsim as RS  # noqa: E402
from jevdrive.openpilot.frames import decode_hevc, load_segment_meta  # noqa: E402

ROOT = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "runs" / "openpilot_rigs"
COMMA = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "datasets" / "comma1M"
N_FRAMES = 1200


def segments():
    return [r["sid"] for f in ("selected.json", "extra.json") if (COMMA / f).exists()
            for r in json.loads((COMMA / f).read_text())]


# ------------------------------------------------------------------ image-pipeline ops on the source planes
# comma's HEVC is BT.601 limited range (Y spans 16..245 on comma1M); ops that need RGB go through it and back.
def yuv_to_rgb(Y, U, V):
    import cv2
    up = lambda a: cv2.resize(a, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128  # noqa: E731
    y = (Y.astype(np.float32) - 16) * 1.164
    u, v = up(U), up(V)
    return np.clip(np.stack([y + 1.596 * v, y - 0.392 * u - 0.813 * v, y + 2.017 * u], -1), 0, 255)


def rgb_to_yuv(rgb):
    import cv2
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
    U = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    V = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    half = lambda a: cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)  # noqa: E731
    q = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8)  # noqa: E731
    return q(Y), q(half(U)), q(half(V))


def op_rgb(fn):
    return lambda Y, U, V, rng: rgb_to_yuv(fn(yuv_to_rgb(Y, U, V), rng))


def op_blur(sigma):
    def f(Y, U, V, rng):
        import cv2
        g = lambda a, s: cv2.GaussianBlur(a, (0, 0), s)  # noqa: E731
        return g(Y, sigma), g(U, sigma / 2), g(V, sigma / 2)
    return f


def op_jpeg(q):
    def f(rgb, rng):
        import cv2
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    return f


def op_noise(sigma):
    def f(Y, U, V, rng):
        n = rng.normal(0, sigma, Y.shape).astype(np.float32)
        return np.clip(Y + n, 0, 255).astype(np.uint8), U, V
    return f


def op_full_range(Y, U, V, rng):
    """Limited-range YUV read as if it were full range (the WOD / NAVSIM runners feed JPEG's full-range YCbCr):
    here the inverse, full-range values fed where limited is expected, i.e. contrast stretched by 255/219."""
    y = (Y.astype(np.float32) - 16) * 255 / 219
    c = lambda a: np.clip(np.rint((a.astype(np.float32) - 128) * 255 / 224 + 128), 0, 255).astype(np.uint8)  # noqa: E731
    return np.clip(np.rint(y), 0, 255).astype(np.uint8), c(U), c(V)


def op_squeeze(Y, U, V, rng):
    """Full-range data converted with a limited-range formula twice (contrast compressed by 219/255)."""
    y = 16 + Y.astype(np.float32) * 219 / 255
    c = lambda a: np.clip(np.rint((a.astype(np.float32) - 128) * 224 / 255 + 128), 0, 255).astype(np.uint8)  # noqa: E731
    return np.clip(np.rint(y), 0, 255).astype(np.uint8), c(U), c(V)


OPS = {
    "blur1.5": op_blur(1.5), "blur3": op_blur(3.0),
    "dark0.5": op_rgb(lambda x, r: x * 0.5), "bright1.6": op_rgb(lambda x, r: np.clip(x * 1.6, 0, 255)),
    "lowcontrast": op_rgb(lambda x, r: 128 + 0.6 * (x - 128)),
    "gray": lambda Y, U, V, r: (Y, np.full_like(U, 128), np.full_like(V, 128)),
    "fullrange": op_full_range, "squeeze": op_squeeze,
    "jpeg75": op_rgb(op_jpeg(75)), "jpeg30": op_rgb(op_jpeg(30)),
    "noise6": op_noise(6.0),
}


# ------------------------------------------------------------------ rigs
def rigs(meta):
    """{variant: (Rig, op name or None)} for one segment (the comma cameras carry its calibration)."""
    fcam, ecam = RS.comma_cams(meta["rpy_calib"])
    level = RS.Cam                                      # rig cameras: mounted level and straight in the calib frame
    pair = {"road": [0], "wide": [1]}                   # comma-like rigs: road <- camera 0, wide <- camera 1
    out = {"native": (RS.Rig([fcam, ecam], use=pair), None)}
    # one camera for both model frames
    out["wide-only"] = (RS.Rig([ecam], src={"road": "ecam", "wide": "ecam"}), None)
    # wide-only with the road frame warped through the segment's wide_from_device rotation (both signs, since
    # which way liveCalibration's wideFromDeviceEuler applies is exactly what this tests); wide frame as native
    for sign in (1, -1):
        R = (RS.VIEW_FROM_DEVICE @ RS.rot_from_euler(sign * np.asarray(meta["wide_from_device_euler"]))
             @ RS.rot_from_euler(meta["rpy_calib"])).T
        out[f"wide-only-wfd{'+' if sign > 0 else '-'}"] = (
            RS.Rig([ecam, ecam], belief=[ecam.replace(R=R), ecam], src={"road": "ecam", "wide": "ecam"}, use=pair), None)
    for hfov in (60, 90, 120):
        f = 960 / np.tan(np.radians(hfov / 2))
        out[f"single-pinhole{hfov}"] = (RS.Rig([level(w=1920, h=1080, f=f)]), None)
    fe = 960 / (np.pi / 2)                            # equidistant 180 deg across 1920 px
    fish = level(w=1920, h=1080, f=fe, model="equidistant")
    out["fisheye180"] = (RS.Rig([fish]), None)
    out["fisheye180-as-pinhole"] = (RS.Rig([fish], belief=[fish.replace(model="pinhole")]), None)
    fish720 = level(w=1280, h=720, f=640 / (np.pi / 2), model="equidistant")
    out["fisheye180-720p"] = (RS.Rig([fish720]), None)
    # multi-camera rigs, rotation-only stitching (the adapter used for WOD-E2E and NAVSIM)
    f0 = dict(w=1920, h=1080, f=1545.0, cx=960.0, cy=560.0)
    out["nuplan-f0"] = (RS.Rig([level(**f0)]), None)
    out["nuplan-l0f0r0"] = (RS.Rig([level(**f0, yaw=-55), level(**f0), level(**f0, yaw=55)]), None)
    wod = dict(w=972, h=1079, f=1115.0, cx=486.0, cy=719.0)
    out["waymo-front3"] = (RS.Rig([level(**wod, yaw=-45), level(**wod), level(**wod, yaw=45)]), None)
    nus = dict(w=1600, h=900, f=800 / np.tan(np.radians(35)))
    out["b2d-nuscenes3"] = (RS.Rig([level(**nus, yaw=-55), level(**nus), level(**nus, yaw=55)]), None)
    out["tfpp-110"] = (RS.Rig([level(w=1024, h=512, f=512 / np.tan(np.radians(55)))]), None)
    # mounting: miscalibrated orientation (physical camera rotated, adapter believes the calibration)
    for name, (dp, dy) in {"pitch+2": (2, 0), "pitch-2": (-2, 0), "pitch+5": (5, 0), "yaw+2": (0, 2)}.items():
        rot = RS.rot_from_euler(np.radians([0, dp, dy]))
        true = [c.replace(R=rot @ c.R) for c in (fcam, ecam)]
        out[name] = (RS.Rig(true, belief=[fcam, ecam], use=pair), None)
    # mounting height (ground-plane parallax); comma nominal 1.22 m
    for h in (0.92, 1.43, 1.60, 1.81, 2.20):
        out[f"height{h:.2f}"] = (RS.Rig([fcam.replace(), ecam.replace()], dh=h - RS.NOMINAL_HEIGHT_M, use=pair), None)
    # sensor resolution: the comma cameras with fewer pixels (same FOV)
    for k in (0.5, 0.25):
        true = [c.replace(w=round(c.w * k), h=round(c.h * k), f=c.f * k, cx=c.cx * k, cy=c.cy * k) for c in (fcam, ecam)]
        out[f"res{k:g}"] = (RS.Rig(true, use=pair), None)
    for name in OPS:
        out[name] = (RS.Rig([fcam, ecam], use=pair), name)
    # target-like combinations (geometry + mount + compression)
    out["wod-like"] = (RS.Rig(out["waymo-front3"][0].true, dh=1.81 - RS.NOMINAL_HEIGHT_M), "jpeg75")
    out["navsim-like"] = (RS.Rig([level(**f0)], dh=1.60 - RS.NOMINAL_HEIGHT_M), "jpeg75")
    return out, {"fcam": fcam, "ecam": ecam}


def cmd_frames(a):
    import cv2
    cv2.setNumThreads(1)
    seg = COMMA / a.seg
    meta = load_segment_meta(seg)
    out_dir = ROOT / "frames" / a.seg
    out_dir.mkdir(parents=True, exist_ok=True)
    table, sources = rigs(meta)
    names = [n for n in table if (not a.variants or n in a.variants)]
    t0 = time.time()
    render = {n: RS.Renderer(table[n][0], sources) for n in names}
    print(f"{a.seg}: maps for {len(names)} variants in {time.time() - t0:.1f} s", flush=True)
    mm = {n: np.lib.format.open_memmap(out_dir / f"{n}.npy.part", "w+", np.uint8, (N_FRAMES, 2, 6, 128, 256))
          for n in names}
    dec = [decode_hevc(seg / "fcamera.hevc"), decode_hevc(seg / "ecamera.hevc")]
    rng = np.random.default_rng(0)
    n_ops = sorted({table[n][1] for n in names if table[n][1]})

    def one(n, planes, op_planes, i):
        op = table[n][1]
        mm[n][i] = render[n](op_planes[op] if op else planes, {} if op else shared)

    t0, n_done = time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for i in range(N_FRAMES):
            try:
                planes = {"fcam": next(dec[0]), "ecam": next(dec[1])}
            except StopIteration:
                break
            seeds = rng.integers(0, 2**31, len(n_ops))
            op_planes = dict(zip(n_ops, ex.map(lambda o, s: {k: OPS[o](*p, np.random.default_rng(s))
                                                           for k, p in planes.items()}, n_ops, seeds)))
            shared = {}
            list(ex.map(lambda n: one(n, planes, op_planes, i), names))
            n_done = i + 1
            if i % 200 == 0:
                print(f"  frame {i}: {(time.time() - t0) / (i + 1):.2f} s/frame", flush=True)
    for n in names:
        mm[n].flush()
        arr = mm.pop(n)
        del arr
        p = out_dir / f"{n}.npy.part"
        if n_done < N_FRAMES:  # a short segment: rewrite with the real length
            full = np.load(p, mmap_mode="r")[:n_done]
            np.save(out_dir / f"{n}.npy", full)
            p.unlink()
        else:
            p.rename(out_dir / f"{n}.npy")
    (out_dir / "done.json").write_text(json.dumps({"n": n_done, "variants": names, "s": time.time() - t0}))
    print(f"{a.seg}: {n_done} frames x {len(names)} variants in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ evaluation
def timing_index(name, n, rng):
    """Frame index per 20 Hz step for the timing variants on the native frames: (road_idx, wide_idx)."""
    i = np.arange(n)
    if name == "t-10hz":            # WOD-E2E runner: each 10 Hz frame fed twice
        return i - i % 2, i - i % 2
    if name == "t-wide-lag50ms":    # wide frame one step older than the road frame
        return i, np.maximum(i - 1, 0)
    if name == "t-jitter":          # CARLA sensor_tick: 20% of frames one step early or late
        j = np.clip(i + rng.choice([-1, 0, 1], n, p=[.1, .8, .1]), 0, n - 1)
        return j, j
    raise KeyError(name)


TIMING = ("t-10hz", "t-wide-lag50ms", "t-jitter")
RESET = ("t-navsim-2hz-1.5s", "t-cold-1.5s")   # zero state, 1.5 s of history, scored every 10th frame


def cmd_eval(a):
    from openpilot_replay import ACTION_T, ground_truth, run_model, scores
    from jevdrive.openpilot.model import OPModel, decode
    from jevdrive.runlog import RunLog
    backends = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}
    models = {m: OPModel(m, backends[m]) for m in a.models}
    log = RunLog("openpilot_rigs", "eval")
    rows_path = ROOT / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {(r["seg"], r["variant"], r["model"]) for r in map(json.loads, rows_path.open())}
    fh = rows_path.open("a")
    for sid in segments():
        fdir = ROOT / "frames" / sid
        if not (fdir / "done.json").exists():
            log.info(f"{sid}: frames missing, skipped")
            continue
        meta = load_segment_meta(COMMA / sid)
        native = np.load(fdir / "native.npy", mmap_mode="r")
        n = len(native)
        gt = ground_truth(meta, n) | {"t": meta["t_loc"][:n] - meta["t_loc"][0]}
        variants = [p.name[:-4] for p in sorted(fdir.glob("*.npy"))] + list(TIMING) + list(RESET)
        for v in variants:
            todo = [m for m in models if (sid, v, m) not in done]
            if not todo:
                continue
            if v in RESET:
                frames = np.asarray(native)
            elif v in TIMING:
                ri, wi = timing_index(v, n, np.random.default_rng(0))
                frames = np.stack([native[ri, 0], native[wi, 1]], 1)
            else:
                frames = np.asarray(np.load(fdir / f"{v}.npy", mmap_mode="r"))
            for name in todo:
                t0 = time.perf_counter()
                if v in RESET:
                    pred, sub = reset_rollouts(models[name], frames, gt["speed"], v, decode, ACTION_T)
                    row = scores_subset(gt, pred, name, sub, scores)
                else:
                    pred = run_model(models[name], frames, gt["speed"])
                    row = scores(gt, pred, name)
                    np.save(ROOT / "frames" / sid / f"pred_{v}_{name}.npy",
                            np.concatenate([pred["plan_pos"].reshape(n, -1), pred["curvature"][:, None]], 1))
                row |= dict(seg=sid, variant=v, s=time.perf_counter() - t0)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                log.info(f"{sid[:8]} {v:22s} {name:8s} lat@2s {row['lat@2s']:.2f} lon@2s {row['lon@2s']:.2f} "
                         f"lat@4s {row['lat@4s']:.2f} curv_mae {row.get('curv_mae', float('nan')) * 1e3:.2f}/km "
                         f"({row['s']:.0f} s)")
    log.close()


RESET_EVERY, RESET_STEPS = 10, 31   # 1.5 s at 20 Hz: t0-1.5 s .. t0


def reset_rollouts(m, frames, speed, v, decode, action_t):
    """Zero-state rollouts over 1.5 s ending at every 10th frame (from frame 100): NAVSIM's timeline (2 Hz
    frames held on the 20 Hz clock, as navsim_zs_openpilot.py) or the same 1.5 s at the full 20 Hz."""
    targets = np.arange(100, len(frames), RESET_EVERY)
    plan_pos, curv = np.zeros((len(frames), 33, 3), np.float32), np.zeros(len(frames), np.float32)
    for t0 in targets:
        m.reset()
        for k in range(RESET_STEPS):
            t = t0 - (RESET_STEPS - 1) + k
            src = t0 - 10 * ((t0 - t + 9) // 10) if v == "t-navsim-2hz-1.5s" else t   # newest 2 Hz frame <= t
            raw = m.step(frames[src], action_t=action_t)
        d = decode(raw, m.slices, float(speed[t0]), action_t)
        plan_pos[t0], curv[t0] = d["plan_pos"], d["curvature"]
    return dict(plan_pos=plan_pos, curvature=curv), targets


def scores_subset(gt, pred, name, sub, scores):
    """Plan errors on the reset-rollout target frames only (curvature / accel agreement not scored)."""
    from openpilot_replay import HORIZONS, at_horizon
    row = dict(model=name, n=int(len(sub)))
    for h in HORIZONS:
        gp, gv = at_horizon(gt["gt_pos"][sub], h), at_horizon(gt["gt_v"][sub], h)
        pp = at_horizon(pred["plan_pos"][sub], h)
        ok = np.isfinite(gv)
        row[f"lon@{h:g}s"] = float(np.abs(pp[ok, 0] - gp[ok, 0]).mean())
        row[f"lat@{h:g}s"] = float(np.abs(pp[ok, 1] - gp[ok, 1]).mean())
    return row


def cmd_sheet(a):
    """Frame 600 of one segment, road | wide luma per variant -> sheet.npz (for the figure on the Mac)."""
    from jevdrive.openpilot.frames import unpack_luma
    fdir = ROOT / "frames" / a.seg
    out = {}
    for p in sorted(fdir.glob("*.npy")):
        if p.name.startswith("pred_"):
            continue
        f = np.load(p, mmap_mode="r")[a.frame]
        out[p.name[:-4]] = np.concatenate([unpack_luma(f[0]), unpack_luma(f[1])], 1)
    np.savez_compressed(ROOT / f"sheet_{a.seg[:8]}_{a.frame}.npz", **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("frames")
    p.add_argument("--seg", required=True)
    p.add_argument("--variants", nargs="*")
    p.add_argument("--threads", type=int, default=2)
    p = sp.add_parser("eval")
    p.add_argument("--models", nargs="+", default=["small", "cinque", "lebowski"])
    p = sp.add_parser("sheet")
    p.add_argument("--seg", required=True)
    p.add_argument("--frame", type=int, default=600)
    a = ap.parse_args()
    {"frames": cmd_frames, "eval": cmd_eval, "sheet": cmd_sheet}[a.cmd](a)
