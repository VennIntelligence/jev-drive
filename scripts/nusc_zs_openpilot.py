#!/usr/bin/env python
"""openpilot zero-shot on nuScenes open-loop planning (todos/2026-09-24-zeroshot-exam/nuscenes-physicalai.md).
Runs in envs/openpilot.

Each val scene is driven continuously from its first keyframe, as modeld would: a 20 Hz clock (10 steps per
keyframe interval, ~0.05 s, the last step exactly at the keyframe), each step fed the latest CAM_FRONT frame
(keyframe or sweep, ~12 Hz) at or before its time. road (focal 910) and wide (focal 455) model frames are
rendered from CAM_FRONT by the rotation-only reprojection (calib frame = the nuScenes ego axes, level and
straight), nearest sampling of libjpeg's YCbCr with the chroma 2x2 mean, as the WOD-E2E and NAVSIM runners.
The plan at every keyframe step -> the LIDAR_TOP point at the GT times.

    PY=$DATA_DIR/envs/openpilot/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY scripts/nusc_zs_openpilot.py
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.common import dataroot, n_cpus  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

MODELS = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}   # backends of the openpilot smoke run
ACTION_T = (0.275, 0.525)
DESIRE = {"left": 1, "right": 2}                                     # command -> desire turnLeft / turnRight
STEPS = 10                                                           # 20 Hz steps per 0.5 s keyframe interval
_idx = {}


def op_index(scene: dict) -> tuple[list, float]:
    """Flat gather indices into CAM_FRONT (road, wide) and the covered fraction (1.0 expected)."""
    cal = {"CAM_FRONT": Z.cam_calib(scene, "CAM_FRONT")}
    out, cov = [], []
    for k in ("road", "wide"):
        src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), cal)
        cov.append(float((src >= 0).mean()))
        idx = G.nn_gather_index(np.where(src >= 0, 0, -1), U, V, [(1600, 900)]).ravel()
        out.append(np.where(idx >= 0, idx, 0))       # uncovered (none expected) would read pixel 0
    return out, min(cov)


def pack(ycc: np.ndarray, idx: np.ndarray) -> np.ndarray:
    p = ycc.reshape(-1, 3)[idx].reshape(256, 512, 3)
    Y = p[..., 0]
    uv = np.rint(p[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))).astype(np.uint8)
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])


def scene_plan(name: str) -> dict:
    """Step times, the CAM_FRONT frame of every step and the keyframe steps of one scene (all its keyframes: the
    tail ones, without a full 3 s future, are only used for the adapter check)."""
    idx = _idx["idx"]
    sc = idx["scenes"][name]
    keys = [e for e in idx["samples"] if e["scene"] == name]
    kt = np.array([e["t0"] for e in keys], np.float64)
    t = [kt[0]] + [kt[k - 1] + j * (kt[k] - kt[k - 1]) / STEPS for k in range(1, len(kt)) for j in range(1, STEPS + 1)]
    fr = [Z.latest_frame(sc, "CAM_FRONT", x + 1) for x in t]
    return {"t": np.array(t), "frame": np.array(fr), "key_step": np.arange(len(kt)) * STEPS, "keys": keys}


def _init():
    _idx["idx"] = Z.load_index()


def scene_frames(name: str):
    """Worker: packed (n_unique, 2, 6, 128, 256) model frames and the step -> row map of one scene."""
    from PIL import Image
    plan = scene_plan(name)
    sc = _idx["idx"]["scenes"][name]
    ix, cov = op_index(sc)
    uniq, inv = np.unique(plan["frame"], return_inverse=True)
    out = np.empty((len(uniq), 2, 6, 128, 256), np.uint8)
    for r, f in enumerate(uniq):
        im = Image.open(dataroot() / sc["cams"]["CAM_FRONT"]["path"][f])
        im.draft("YCbCr", im.size)
        ycc = np.asarray(im.convert("YCbCr"))
        out[r, 0], out[r, 1] = pack(ycc, ix[0]), pack(ycc, ix[1])
    return name, out, inv, cov


def main(a, log):
    from tqdm import tqdm
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    _init()
    idx = _idx["idx"]
    scenes = sorted(idx["scenes"])[:a.limit or None]
    models = {m: OPModel(m, MODELS[m]) for m in a.models.split(",")}
    variants = a.desires.split(",")
    log.event("start", models=list(models), desires=variants, scenes=len(scenes), action_t=ACTION_T)
    res = {(m, d): {} for m in models for d in variants}
    tail = {(m, d): {} for m in models for d in variants}      # raw plans at tail keyframes (adapter check only)
    tm = {m: 0.0 for m in models}
    covs, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers or min(16, n_cpus()), initializer=_init) as ex:
        for name, frames, inv, cov in tqdm(ex.map(scene_frames, scenes), total=len(scenes), desc="op scenes"):
            covs.append(cov)
            plan = scene_plan(name)
            sc = idx["scenes"][name]
            tc = (0, 1) if sc["location"].startswith("singapore") else (1, 0)
            keyat = {s: e for s, e in zip(plan["key_step"], plan["keys"])}
            # the command of the last keyframe at or before the step (tail keyframes have none -> straight)
            cmd_of_step = [plan["keys"][s // STEPS].get("cmd", "straight") for s in range(len(plan["t"]))]
            dev = sc["cams"]["CAM_FRONT"]["t_ego"][:2]
            for m, mod in models.items():
                for d in variants:
                    mod.reset()
                    t1 = time.perf_counter()
                    for s in range(len(plan["t"])):
                        desire = np.zeros(8, np.float32)
                        if d == "cmd":
                            desire[DESIRE.get(cmd_of_step[s], 0)] = 1
                        raw = mod.step(frames[inv[s]], desire=desire, traffic=tc, action_t=ACTION_T)
                        e = keyat.get(s)
                        if e is None:
                            continue
                        o = decode(raw, mod.slices, Z.cv_speed(sc, e["t0"]), ACTION_T)
                        if not e["valid"]:
                            tail[(m, d)][e["token"]] = np.c_[o["plan_pos"], o["plan_yaw"]]
                            continue
                        p = np.stack([o["plan_pos"][:, 0], -o["plan_pos"][:, 1]], -1)       # calib: y left
                        psi = -np.asarray(o["plan_yaw"], np.float64)
                        c, s_ = np.cos(psi), np.sin(psi)
                        rear = dev + p - np.stack([c * dev[0] - s_ * dev[1], s_ * dev[0] + c * dev[1]], -1)
                        pred, _ = Z.to_lidar_point(T_IDXS[1:], rear[1:], psi[1:], sc["lidar_xyz"], e["fut_t"])
                        res[(m, d)][e["token"]] = (pred.astype(np.float32), np.c_[o["plan_pos"], o["plan_yaw"]])
                    tm[m] += time.perf_counter() - t1
    out = Z.root("preds")
    for (m, d), r in res.items():
        toks = sorted(r)
        np.savez(out / f"op_{m}_{d}.npz", tokens=np.array(toks), pred=np.stack([r[t][0] for t in toks]),
                 plan=np.stack([r[t][1] for t in toks]).astype(np.float32))
        tt = sorted(tail[(m, d)])
        np.savez(out / f"op_{m}_{d}_tail.npz", tokens=np.array(tt), plan=np.stack([tail[(m, d)][t] for t in tt]))
    s = {"scenes": len(scenes), "n": len(res[next(iter(res))]), "wall_s": time.time() - t0,
         "gpu_s": tm, "min_coverage": float(min(covs))}
    log.event("end_run", **s)
    log.info(json.dumps(s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small,cinque,lebowski")
    ap.add_argument("--desires", default="none,cmd")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    a = ap.parse_args()
    log = RunLog("nusc_zs", "openpilot")
    log.info(f"args {vars(a)} -> {log.dir}")
    main(a, log)
    log.event("end")
