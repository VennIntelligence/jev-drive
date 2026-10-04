"""Stage 2 (openpilot env, one leased card): the real stream (G) and the reprojection arms (R) through shipped Cinque.

Per anchor (segment, 5 Hz frame i): the real frames i - PRE .. i are the shared history (zero state, every frame held 4 steps of the 20 Hz clock,
the p5_openpilot protocol); then the branches continue for K = 10 steps (2 s):
  G           the real frames i + 1 .. i + K (ground truth of the nominal motion)
  nom         anchor frame i re-projected to the real poses i + 1 .. i + K (jevdrive.op_interp.warp_frame: road plane at the camera height, static
              world, 60 m sphere beyond; both model views)
  yaw+-1/2    the same with the ego heading offset by +-1 / 2 deg from step 1 on (wm_common.perturb); lat+-0.5: a 3-step swerve ending 0.5 m off the path
Saved per branch and step (history frames i - 7 .. i for every branch, then the K future steps): temporal tap (512), plan position (33, 3),
plan yaw (33), action head curvature (decode with the real speed), phi1 = direction of the 1 s plan point (deg, left +).
  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/openpilot/bin/python experiments/world_model/scripts/wm_op.py [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).parent)]
import wm_common as C  # noqa: E402

OUT = C.RUN / "op"
HOLD = 4
ACTION_T = (0.275, 0.525)
TAP = "select_4"                                    # drive_backbones.OP_TAPS["cinque"]["temporal"]


def readout(m, raw, v):
    from jevdrive.openpilot.model import T_IDXS, decode
    d = decode(raw, m.slices, v, ACTION_T)
    pos = d["plan_pos"]                              # x fwd, y right (openpilot calib frame)
    x1, y1 = np.interp(1.0, T_IDXS, pos[:, 0]), np.interp(1.0, T_IDXS, pos[:, 1])
    return dict(plan=pos.astype(np.float32), yaw=d["plan_yaw"].astype(np.float32), curv=d["curvature"],
                phi1=float(-np.degrees(np.arctan2(y1, max(x1, 1e-3)))))


def run_branch(m, frames, v, keep_from):
    """Zero state, every frame held HOLD steps; outputs after the last hold of frames >= keep_from."""
    m.reset()
    rows = []
    for j, f in enumerate(frames):
        for _ in range(HOLD):
            raw = m.step(f, action_t=ACTION_T)
        if j >= keep_from:
            rows.append((m.tap_values[TAP].copy(), readout(m, raw, float(v[j]))))
    return {"z": np.stack([r[0] for r in rows]), "plan": np.stack([r[1]["plan"] for r in rows]), "yaw": np.stack([r[1]["yaw"] for r in rows]),
            "curv": np.array([r[1]["curv"] for r in rows]), "phi1": np.array([r[1]["phi1"] for r in rows])}


def main():
    from jevdrive import op_interp as I
    from jevdrive.openpilot.model import OPModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = json.load(open(C.RUN / "anchors.json"))[: a.limit or None]
    m = OPModel("cinque", "trt", taps=[TAP])
    t0 = time.time()
    for n, an in enumerate(anchors):
        out = OUT / f"{an['seg'][:12]}_{an['i']}.npz"
        if out.exists():
            continue
        z = np.load(C.RUN / "prep" / f"{an['seg']}.npz")
        op, pose, v, cam = z["op"], z["pose"], z["v"], z["cam"]
        i = an["i"]
        start = max(0, i - C.PRE)
        P, step_v = C.nominal(pose[i: i + C.K + 1])
        pose_arms = C.arms(P, step_v)
        hist = op[start: i + 1]
        keep_from = i - start - (C.HIST - 1)
        res = {"G": run_branch(m, op[start: i + C.K + 1], v[start: i + C.K + 1], keep_from)}
        for name, Q in pose_arms.items():
            fut = [I.warp_frame(op[i], cam, Q[j], np.zeros(3)) for j in range(1, C.K + 1)]
            res[name] = run_branch(m, np.concatenate([hist, np.stack(fut)]), v[start: i + C.K + 1], keep_from)
        flat = {f"{b}/{k}": x for b, d in res.items() for k, x in d.items()}
        flat |= {f"pose/{b}": Q for b, Q in pose_arms.items()}
        np.savez(out.with_suffix(".tmp.npz"), anchor=json.dumps(an), v=v[i - C.HIST + 1: i + C.K + 1], step_v=step_v, **flat)
        out.with_suffix(".tmp.npz").replace(out)
        print(f"anchor {n + 1}/{len(anchors)} {an['cat']} {an['seg'][:12]} i={i}: {len(res)} branches, {time.time() - t0:.0f}s", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
