"""Replay comma1M segments through openpilot driving models at 20 Hz and score the plan against the
localizer's ground-truth future motion.

Per segment: decode fcamera/ecamera, warp to the model frames with the segment calibration (cached as
model_frames.npy next to the video), step every model from a zero state with the car's speed as v_ego,
decode plan / desired curvature / desired accel like modeld, and save everything to one npz per segment.

  python scripts/openpilot_replay.py --models small cinque lebowski --backend small=trt cinque=trt lebowski=trt
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.openpilot.frames import future_in_calib, load_segment_meta, segment_model_frames
from jevdrive.openpilot.model import LONG_SMOOTH_S, T_IDXS, OPModel, decode, smooth
from jevdrive.runlog import RunLog

ACTION_T = (0.275, 0.525)  # lat 0.2 s + long 0.15 s actuator delay (+0.3 s long smoothing), + 1.5 frames, as modeld
HORIZONS = (1.0, 2.0, 4.0, 6.0)
WARMUP = 100  # frames (5 s) before the 5 s feature queue is full; excluded from the scores


def ground_truth(meta, n):
    pos, vel = zip(*(future_in_calib(meta, i, T_IDXS) for i in range(n)))
    speed = np.linalg.norm(meta["vel"][:n], axis=1)
    curv = -meta["omega_dev"][:n, 2] / np.maximum(speed, 1.0)  # device z is down; openpilot curvature is left-positive
    accel = np.gradient(speed, meta["t_loc"][:n])
    return dict(gt_pos=np.stack(pos), gt_v=np.stack(vel), speed=speed, gt_curv=curv, gt_accel=accel)


def run_model(m, frames, speed):
    m.reset()
    keys = ("plan_pos", "plan_vel", "curvature", "accel", "accel_smooth", "lead_x", "lead_prob", "lane_prob",
            "engaged", "step_ms")
    out = {k: [] for k in keys}
    prev_acc = 0.0
    for i, f in enumerate(frames):
        t = time.perf_counter()
        raw = m.step(f, action_t=ACTION_T)
        out["step_ms"].append((time.perf_counter() - t) * 1e3)
        d = decode(raw, m.slices, float(speed[i]), ACTION_T)
        prev_acc = smooth(d["accel"], prev_acc, LONG_SMOOTH_S)
        for k in ("plan_pos", "plan_vel", "curvature", "accel", "lead_prob", "lane_prob", "engaged"):
            out[k].append(d[k])
        out["accel_smooth"].append(prev_acc)
        out["lead_x"].append(d["lead"][0, 0, 0])
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def scores(gt, pred, model):
    """Plan position / speed errors at HORIZONS and action agreement, frames >= WARMUP with ground truth."""
    s = slice(WARMUP, None)
    row = dict(model=model, n=int(len(gt["speed"][s])))
    for h in HORIZONS:
        j = np.interp(h, T_IDXS, np.arange(33))
        at = lambda a: np.stack([np.interp(j, np.arange(33), a[..., k]) for k in range(a.shape[-1])], -1)  # noqa: E731
        gp = at(gt["gt_pos"][s])
        pp = at(gt["cv_pos"][s]) if model == "const-vel" else at(pred["plan_pos"][s])
        ok = np.isfinite(gp[:, 0])
        row[f"lon@{h:g}s"] = float(np.abs(pp[ok, 0] - gp[ok, 0]).mean())
        row[f"lat@{h:g}s"] = float(np.abs(pp[ok, 1] - gp[ok, 1]).mean())
        gv = np.array([np.interp(j, np.arange(33), r) for r in gt["gt_v"][s]])
        pv = gt["speed"][s] if model == "const-vel" else \
            np.array([np.interp(j, np.arange(33), r) for r in pred["plan_vel"][s][:, :, 0]])
        okv = np.isfinite(gv)
        row[f"v@{h:g}s"] = float(np.abs(pv[okv] - gv[okv]).mean())
    if model != "const-vel":
        t = gt["t"]
        moving = gt["speed"] > 5
        k_gt = np.interp(t + ACTION_T[0], t, gt["gt_curv"])
        a_gt = np.interp(t + ACTION_T[1], t, gt["gt_accel"])
        m = moving & (np.arange(len(t)) >= WARMUP)
        row["curv_mae"] = float(np.abs(pred["curvature"][m] - k_gt[m]).mean())
        row["curv_r"] = float(np.corrcoef(pred["curvature"][m], k_gt[m])[0, 1])
        row["accel_mae"] = float(np.abs(pred["accel_smooth"][m] - a_gt[m]).mean())
        row["accel_r"] = float(np.corrcoef(pred["accel_smooth"][m], a_gt[m])[0, 1])
        row["step_ms_p50"] = float(np.percentile(pred["step_ms"][WARMUP:], 50))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "data/datasets/comma1M")
    ap.add_argument("--segs", nargs="*", help="segment ids (default: selected.json)")
    ap.add_argument("--models", nargs="+", default=["small", "cinque", "lebowski"])
    ap.add_argument("--backend", nargs="*", default=[], help="model=backend overrides (default cuda-iob)")
    a = ap.parse_args()
    segs = a.segs or [r["sid"] for r in json.loads((a.root / "selected.json").read_text())]
    backends = {m: "cuda-iob" for m in a.models} | dict(kv.split("=") for kv in a.backend)
    log = RunLog("openpilot_replay", "_".join(a.models))
    models = {m: OPModel(m, backends[m]) for m in a.models}
    rows = []
    for sid in segs:
        d = a.root / sid
        meta = load_segment_meta(d)
        cache = d / "model_frames.npy"
        t0 = time.perf_counter()
        if cache.exists():
            frames = np.load(cache, mmap_mode="r")
        else:
            frames, _ = segment_model_frames(d, meta)
            np.save(cache, frames)
        n = len(frames)
        log.info(f"{sid}: {n} frames, prep {time.perf_counter() - t0:.1f}s, calib rpy {np.round(meta['rpy_calib'], 4)}")
        gt = ground_truth(meta, n) | {"t": meta["t_loc"][:n] - meta["t_loc"][0]}
        gt["cv_pos"] = np.zeros_like(gt["gt_pos"])
        gt["cv_pos"][..., 0] = gt["speed"][:, None] * T_IDXS[None]
        res = {f"gt/{k}": v for k, v in gt.items()}
        rows.append(dict(seg=sid) | scores(gt, None, "const-vel"))
        for name, m in models.items():
            pred = run_model(m, frames, gt["speed"])
            res |= {f"{name}/{k}": v for k, v in pred.items()}
            rows.append(dict(seg=sid, backend=backends[name]) | scores(gt, pred, name))
            log.event("segment_scores", **rows[-1])
            r = rows[-1]
            log.info(f"  {name:9s} lat@2s {r['lat@2s']:.2f} m lon@2s {r['lon@2s']:.2f} m v@2s {r['v@2s']:.2f} m/s "
                     f"curv r {r['curv_r']:.2f} accel r {r['accel_r']:.2f} step {r['step_ms_p50']:.2f} ms")
        np.savez_compressed(log.dir / f"{sid}.npz", **res, rpy_calib=meta["rpy_calib"],
                            sample_frames=np.asarray(frames[600]))
    (log.dir / "scores.json").write_text(json.dumps(dict(backends=backends, action_t=ACTION_T, warmup=WARMUP,
                                                         rows=rows), indent=1))
    log.info(f"results -> {log.dir}")
    log.close()
