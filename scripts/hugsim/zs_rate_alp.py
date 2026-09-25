#!/usr/bin/env python
"""How much does HUGSIM's 4 Hz frame rate cost Alpamayo 1.5? Offline on nuScenes open-loop planning, with the
inputs of the nuScenes exam (scripts/nusc_zs_alpamayo.py: 4 virtual f-theta views, nav text from the VAD command)
and only the time axis changed. todos/2026-09-25-hugsim-exam.

  native     4 frames at t0-0.3, -0.2, -0.1, t0 (each camera's nearest ~12 Hz sweep), egomotion from the 20 Hz poses
             with full rotations (the nuScenes exam).
  h4-hold    the HUGSIM adapter: frames exist only at t0 - 0.25 j; each 10 Hz slot takes the nearest of them
             (-> t0-0.25, t0-0.25, t0, t0); egomotion from 4 Hz poses, linear in position and yaw, yaw-only rotations.
  h4-spread  the alternative: the last four 4 Hz frames (t0-0.75 ... t0) presented as 10 Hz; egomotion as h4-hold.

Same samples, order, batch composition and seeds in every variant, so the differences are paired. Closed-loop
model config (flow matching 5 steps, as scripts/zeroshot_policy_server.py).

    CUDA_VISIBLE_DEVICES=1 $DATA_DIR/third_party/alpamayo1.5/.venv/bin/python scripts/hugsim/zs_rate_alp.py --n 400
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import navsim_zs_alpamayo as NA  # noqa: E402
import nusc_zs_alpamayo as NZA  # noqa: E402
from jevdrive import nuscenes_zs as Z  # noqa: E402

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
VARIANTS = ("native", "h4-hold", "h4-spread")


def slot_times(t0, variant):
    want = t0 - np.arange(3, -1, -1) * 100_000
    if variant == "native":
        return want
    grid = t0 - np.arange(8) * 250_000
    if variant == "h4-spread":
        return grid[3::-1]
    return np.array([grid[np.argmin(np.abs(grid - w) - 1e-9 * grid)] for w in want])


def hist_4hz(sc, t0):
    """Egomotion as the HUGSIM agent builds it: poses every 0.25 s, 10 Hz by linear interpolation, yaw only."""
    tg = t0 - np.arange(7, -1, -1) * 250_000
    xyz, R = Z.ego_at(sc, tg)
    R0t = R[-1].T
    rel = (xyz - xyz[-1]) @ R0t.T
    yaw = np.unwrap(Z.yaw_of(R0t @ R))
    tq = t0 + (np.arange(16) - 15) * 100_000
    x, y, h = (np.interp(tq, tg, v) for v in (rel[:, 0], rel[:, 1], yaw))
    c, s = np.cos(h), np.sin(h)
    rot = np.zeros((16, 3, 3), np.float32)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    return np.stack([x, y, np.zeros(16)], -1).astype(np.float32), rot


class Inputs(NZA.Inputs):
    def __init__(self, idx, variant):
        super().__init__(idx)
        self.variant = variant

    def __call__(self, e):
        sc = self.idx["scenes"][e["scene"]]
        m = self.maps_for(e["scene"])
        need = {(c, s) for v in m.views for c, s, *_ in v}
        slots = [m.render({(c, s): self.decode(sc["cams"][c]["path"][Z.frame_at(sc, c, t)], s) for c, s in need})
                 for t in slot_times(e["t0"], self.variant)]
        xyz, rot = Z.alpamayo_history(sc, e["t0"]) if self.variant == "native" else hist_4hz(sc, e["t0"])
        return {"token": e["token"], "frames": np.stack(slots, 1), "cam_idx": np.asarray(m.cam_idx), "xyz": xyz,
                "rot": rot, "nav": Z.NAV_TEXT.get(e.get("cmd"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(D / "runs" / "hugsim-exam" / "rate_alp.jsonl"))
    a = ap.parse_args()
    idx = Z.load_index()
    ents = NZA.entries_of(idx, "main")
    rng = np.random.default_rng(0)
    ents = [ents[k] for k in sorted(rng.choice(len(ents), a.n, replace=False))]
    model = NA.Model("sdpa", ("visual", "expert"), 5)
    by = {e["token"]: e for e in ents}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    for v in VARIANTS:
        t0 = time.time()
        rows = []

        def emit(r):
            e = by[r["token"]]
            sc = idx["scenes"][e["scene"]]
            p = Z.to_lidar_point(NZA.ALP_T, np.asarray(r["xyz"])[:, :2], np.asarray(r["yaw"]), sc["lidar_xyz"],
                                 e["fut_t"])[0]
            err = np.linalg.norm(p - e["gt"], axis=-1)
            rows.append(1)
            fh.write(json.dumps({"variant": v, "token": r["token"], "scene": e["scene"], "cmd": e["cmd"],
                                 "l2_1s": float(err[:2].mean()), "l2_2s": float(err[:4].mean()),
                                 "l2_3s": float(err.mean()), "l2pt_3s": float(err[-1]), "ms": r["ms"]}) + "\n")
            fh.flush()
        NZA.run_batches(model, NZA.prepared(model, Inputs(idx, v), ents, ("nav",), a.workers), ("nav",), a.batch, emit)
        print(f"{v}: {len(rows)} samples in {time.time() - t0:.0f} s", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
