"""Offline replay of Cinque on the frames saved in HUGSIM runs (video.mp4), with the model inputs rebuilt in variants. No simulator.

The ego path is the logged one (open loop): the question is whether the plan the model gives on the *same frames* depends
on how the openpilot road / wide inputs are built. Variants (jevdrive.hugsim_zs.OpenpilotFrames is the baseline):
  base        3 forward cameras, as in the exam (the wide input takes its left / right edge from FRONT_LEFT / FRONT_RIGHT)
  front       CAM_FRONT only for both inputs (pixels outside its field of view stay black)
  yaw:X       the virtual camera turned X degrees left (+) / right (-) of the ego heading (X signed), both inputs
  ramp:R      yaw rotating by R degrees per simulator step (a synthetic yaw rate), to measure how the plan responds to
              perceived yaw rate
Same request sequence as experiments/hugsim/archive/zs_agent.py (warm-up 100 model steps on frame 0, then 4 per step,
speed = ego speed x 1.25, command -> desire, traffic flag, forward_only, straight_stop).

    CUDA_VISIBLE_DEVICES=<idle card> $DATA_DIR/envs/openpilot/bin/python spin_input_replay.py <jobs.json> <out_dir> [--images]
jobs.json: [{"run_dir", "dataset", "tag", "n_steps", "traffic"}], run_dir holds video.mp4, infos.pkl, zs_steps.jsonl.
Writes <out_dir>/<tag>__<run>.json = {variant: [plan per step]} (+ logged plans) and, with --images, model-input PNGs.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path(".")
for p in (ROOT, Path("/root/autodl-tmp/ujs/jev-drive"), Path("/root/autodl-tmp/ujs/jev-drive/scripts")):
    sys.path.insert(0, str(p))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402

CAMYAML = "/root/autodl-tmp/ujs/third_party/HUGSIM/configs/sim/{}_camera.yaml"
DIL = 1.25


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def build_frames(cal, cams, yaw_deg=0.0):
    """Z.OpenpilotFrames with the virtual camera rays turned by yaw_deg about the vehicle's up axis (left positive)."""
    obj = Z.OpenpilotFrames.__new__(Z.OpenpilotFrames)
    sub = {c: cal[c] for c in cams}
    obj.cams = list(sub)
    sizes = [(sub[c]["width"], sub[c]["height"]) for c in obj.cams]
    obj.idx, obj.coverage, obj.src_frac = {}, {}, {}
    for k in ("road", "wide"):
        rays = G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H) @ rot_z(yaw_deg).T
        src, U, V = G.choose_sources(np, rays, sub)
        obj.coverage[k] = float((src >= 0).mean())
        obj.src_frac[k] = {c: float((src == j).mean()) for j, c in enumerate(obj.cams)}
        obj.idx[k] = G.nn_gather_index(src, U, V, sizes).ravel()
    return obj


def read_video(path):
    import cv2
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = f.shape[0] // 2, f.shape[1] // 3
        out.append({"CAM_FRONT_LEFT": f[:h, :w], "CAM_FRONT": f[:h, w:2 * w], "CAM_FRONT_RIGHT": f[:h, 2 * w:]})


def plan_from(out, speed_dummy=None):
    plan = Z.openpilot_to_plan(out["pos"], out["t"], DIL)
    plan = Z.forward_only(plan)
    return Z.straight_stop(plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--variants", default="base,front,yaw:2,yaw:-2,yaw:4,yaw:-4,ramp:0.5,ramp:-0.5,ramp:1,ramp:-1")
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel, T_IDXS, decode
    m = OPModel("cinque", "trt", context_rate=False)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for job in json.load(open(a.jobs)):
        d = Path(job["run_dir"])
        infos = pickle.load(open(d / "infos.pkl", "rb"))
        n = min(job["n_steps"], len(infos))
        rect = Z.rect_matrix(CAMYAML.format(job["dataset"]))
        cal = Z.calibs(infos[0]["cam_params"], rect)
        vid = read_video(d / "video.mp4")
        logged = [json.loads(x) for x in open(d / "zs_steps.jsonl")][1:]
        res = {"logged": [r["plan"] for r in logged[:n]], "coverage": {}}
        for var in a.variants.split(","):
            kind, _, val = var.partition(":")          # base | front | yaw:+2 | ramp:-0.5
            val = float(val) if val else 0.0
            cams = ("CAM_FRONT",) if kind == "front" else Z.CAMS[:3]
            m.reset()
            plans = []
            for k in range(min(n, 14) if kind == "ramp" else n):
                yaw = val if kind == "yaw" else val * k if kind == "ramp" else 0.0
                key = (kind == "front", round(yaw, 4), job["dataset"])
                if key not in cache:
                    cache[key] = build_frames(cal, cams, yaw)
                img2 = cache[key].pack(vid[k])
                reps = 4 if k else 100
                desire = np.zeros(8, np.float32)
                desire[Z.DESIRE[int(infos[k]["command"])]] = 1
                for _ in range(reps):
                    raw = m.step(img2, desire=desire, traffic=tuple(job["traffic"]))
                dd = decode(raw, m.slices, float(infos[k]["ego_velo"]) * DIL)
                plans.append(np.round(plan_from({"pos": dd["plan_pos"], "t": T_IDXS.astype(np.float32)}), 3).tolist())
                if a.images and var in ("base", "front") and k in job.get("img_steps", []):
                    import cv2
                    from jevdrive.openpilot.frames import unpack_luma
                    cv2.imwrite(str(out_dir / f"{job['tag']}__{d.name}__{var}__step{k}.png"),
                                np.concatenate([unpack_luma(img2[0]), unpack_luma(img2[1])], 0))
                    cv2.imwrite(str(out_dir / f"{job['tag']}__{d.name}__raw__step{k}.png"),
                                cv2.cvtColor(np.concatenate([vid[k][c] for c in Z.CAMS[:3]], 1), cv2.COLOR_RGB2BGR))
            res[var] = plans
        json.dump(res, open(out_dir / f"{job['tag']}__{d.name}.json", "w"))
        print("done", job["tag"], d.name, flush=True)


if __name__ == "__main__":
    main()
