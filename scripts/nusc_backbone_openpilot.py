#!/usr/bin/env python
"""openpilot as a frozen feature extractor on nuScenes train + val (decisions 40, follow-up (i); pre-registration in
todos/2026-09-24-driving-backbones/README.md). Runs in envs/openpilot.

Input protocol is the nuScenes zero-shot exam's, unchanged (scripts/nusc_zs_openpilot.py): every scene driven
continuously from its first keyframe on a 20 Hz clock, each step fed the latest CAM_FRONT frame, road / wide
rendered by the rotation-only reprojection, desire none, traffic convention by location. The only differences:
all 850 trainval scenes instead of the 150 val ones, and at every keyframe step we keep the tapped intermediate
tensors of the WOD feature run (`temporal`, `vision`: jevdrive.drive_backbones.OP_TAPS) plus the native plan as
rear-axle waypoints at 0.25 ... 3.0 s in the t0 ego frame.

Output: $DATA_DIR/processed/drive_backbones/nusc_op/<model>/<scene>.npz (resumable per scene).

    PY=$DATA_DIR/envs/openpilot/bin/python
    CUDA_VISIBLE_DEVICES=2 taskset -c ... $PY scripts/nusc_backbone_openpilot.py --workers 4
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nusc_zs_openpilot as NO  # noqa: E402  (the exam runner: renderer, step plan, model settings)
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

T_OUT = np.arange(1, 13) * 0.25          # native plan waypoints, the replication's target times


def _init():
    NO._idx["idx"] = Z.load_index("trainval")


def rear_waypoints(o: dict, dev: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Native plan (calib frame at the camera, y right) -> rear-axle xy in the t0 ego frame at T_OUT (the exam's
    conversion: rear(t) = d + p(t) - R(psi_t) d)."""
    p = np.stack([o["plan_pos"][:, 0], -o["plan_pos"][:, 1]], -1)
    psi = -np.asarray(o["plan_yaw"], np.float64)
    c, s = np.cos(psi), np.sin(psi)
    rear = dev + p - np.stack([c * dev[0] - s * dev[1], s * dev[0] + c * dev[1]], -1)
    t = np.r_[0.0, t_idx[1:]]                 # as Z.to_lidar_point: the origin at t = 0, then the plan's points
    rear = np.r_[[[0.0, 0.0]], rear[1:]]
    return np.stack([np.interp(T_OUT, t, rear[:, k]) for k in (0, 1)], -1).astype(np.float32)


def main(a, log):
    from tqdm import tqdm
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    _init()
    idx = NO._idx["idx"]
    out = {m: D.root("nusc_op", m) for m in a.models.split(",")}
    scenes = [s for s in sorted(idx["scenes"]) if not all((out[m] / f"{s}.npz").exists() for m in out)]
    scenes = scenes[:a.limit or None]
    models = {m: OPModel(m, NO.MODELS[m], taps=list(D.OP_TAPS[m].values())) for m in out}
    log.event("start", models=list(models), scenes=len(scenes), done_before=len(idx["scenes"]) - len(scenes))
    tm, t0 = {m: 0.0 for m in models}, time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for name, frames, inv, cov in tqdm(ex.map(NO.scene_frames, scenes), total=len(scenes), desc="op scenes"):
            plan = NO.scene_plan(name)
            sc = idx["scenes"][name]
            tc = (0, 1) if sc["location"].startswith("singapore") else (1, 0)
            keyat = {s: e for s, e in zip(plan["key_step"], plan["keys"])}
            dev = sc["cams"]["CAM_FRONT"]["t_ego"][:2]
            for m, mod in models.items():
                taps = D.OP_TAPS[m]
                rec = {"token": [], "temporal": [], "vision": [], "native": []}
                mod.reset()
                t1 = time.perf_counter()
                for s in range(len(plan["t"])):
                    t2 = time.perf_counter()
                    raw = mod.step(frames[inv[s]], desire=np.zeros(8, np.float32), traffic=tc, action_t=NO.ACTION_T)
                    if a.duty < 1:     # yield the shared card: busy at most `duty` of the wall time
                        time.sleep((time.perf_counter() - t2) * (1 / a.duty - 1))
                    e = keyat.get(s)
                    if e is None:
                        continue
                    o = decode(raw, mod.slices, Z.cv_speed(sc, e["t0"]), NO.ACTION_T)
                    rec["token"].append(e["token"])
                    rec["temporal"].append(mod.tap_values[taps["temporal"]])
                    rec["vision"].append(mod.tap_values[taps["vision"]].astype(np.float16))
                    rec["native"].append(rear_waypoints(o, dev, T_IDXS))
                tm[m] += time.perf_counter() - t1
                np.savez(out[m] / f"{name}.npz", coverage=cov, **{k: np.array(v) for k, v in rec.items()})
    s = {"scenes": len(scenes), "wall_s": time.time() - t0, "gpu_s": tm}
    log.event("end_run", **s)
    log.info(str(s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cinque,lebowski")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N pending scenes only (smoke test)")
    ap.add_argument("--duty", type=float, default=1.0,
                    help="GPU duty cycle: sleep after each step so the card is ours at most this share of the time "
                         "(0.35 kept the Alpamayo exam on the same card within ~10%%)")
    a = ap.parse_args()
    log = RunLog("nusc_backbones", "openpilot")
    log.info(f"args {vars(a)} -> {log.dir}")
    main(a, log)
    log.event("end")
