#!/usr/bin/env python3
"""BODY1 gate G2 harness, part 1: dump the model-side inputs of decision 220's nuPlan decisions.

Replays COL1's serialized driver messages through the real driver class by calling swv1_replay.main() unchanged (its output is
redirected to $DATA_DIR/runs/body1/g2/replay/, the original swv1 files are not touched) and, through class-level wrappers around
Core.plan / Driver.drive, stores for every decision what a contact predictor may use at test time. Read-only on the navtest logs;
nothing is fitted here. One GPU, a pool job:

  python -m jevdrive.cl submit --name body1-g2-dump --owner body1 --vram 8 --cpu 8 --ram 24 --log-dir $DATA_DIR/runs/body1/g2/pool -- \
    env ALPASIM_SRC=$DATA_DIR/third_party/alpasim $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/g2_dump.py

Writes $DATA_DIR/runs/body1/g2/dump/<group>.npz (arrays are per decision, in scene order of spec.json, then decision order):
  scene (N,) str, k (N,) int32 decision index inside the scene, now_us (N,) int64, cmd (N,) int8 (0 L, 1 S, 2 R, 3 unknown),
  tokens (N, 4, *H_SHAPE) float16 view_39 vision tokens, slot i valid iff valid[n, i] (padded with zeros), valid (N, 4) bool,
  ego (N, 20) float32 ego feature vector, poses (N, 8, 3) float64 served plan (rear axle x, y, yaw at 0.5 .. 4 s, ego frame of t0),
  mu (N, 33, 15) float32 plan mean, plan_raw (N, 990) float32 whole plan slice (mean 495 + log-std logits 495),
  frames (N, 4, 2, 6, 128, 256) uint8 model-input frames (road + wide openpilot YUV planes) of the 4 history slots, exactly as fed to
  the Cinque encoder (the encoder pair of slot i is (frames[i-1] or zeros, frames[i])), cam_K / cam_R / cam_t / cam_D (N, ...) float64
  camera intrinsics / rotation / position (z = height) / distortion, cam_xy (N, 2), n_slots (N,) int.
meta.json holds shapes, sizes and the identity gate against the stored swv1 replay.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALP = HERE.parents[1] / "alpasim"
sys.path[:0] = [str(ALP / "scripts"), str(ALP / "lib"), str(HERE.parents[2])]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = DATA / "runs/body1/g2"
STORED = DATA / "runs/alpasim/swv1/replay"
GROUPS = ("P2H10-F-s0", "P2H10-F-s1", "ctrl")


def install(buf):
    """Class-level wrappers; every drive() appends one record dict to `buf`."""
    import sh30_driver as D
    core_cls, drv_cls, plans = D.C.Core, D.Driver, []
    plan0, drive0 = core_cls.plan, drv_cls.drive

    def plan(self, *a, **k):
        if getattr(self, "_g2_out", None) is None:
            self._g2_out = {}
            self.model.register_forward_hook(lambda m, args, out: self._g2_out.update(out=out))
        o = plan0(self, *a, **k)
        plans.append(dict(tokens=o["tokens"].detach().cpu().numpy().astype(np.float16), valid=np.asarray(o["valid"], bool), ego=np.asarray(o["ego"], np.float32),
                          poses=np.asarray(o["poses"], np.float64), mu=np.asarray(o["mu"], np.float32), frames=np.asarray(o["cur"], np.uint8),
                          plan_raw=self._g2_out["out"].float()[0, self.model.net.slices["plan"]].cpu().numpy()))
        return o

    def drive(self, req, ctx):
        n = len(plans)
        r = drive0(self, req, ctx)
        if len(plans) != n + 1:
            raise RuntimeError(f"drive produced {len(plans) - n} plan() calls, expected 1")
        s = self.sessions[req.session_uuid]
        rec = plans.pop()
        rec.update(now_us=int(req.time_now_us), cmd=int(np.argmax(s.cmd)), cam={k: np.asarray(v, np.float64) for k, v in s.cam.items()})
        buf.append(rec)
        return r

    core_cls.plan, drv_cls.drive = plan, drive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(GROUPS)), ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from jevdrive.run import Run
    out = Path(a.out)
    import swv1_replay as R
    R.O = out / "replay"
    buf = []
    with Run("body1", "g2-dump", config=vars(a)) as run:
        install(buf)
        (out / "dump").mkdir(parents=True, exist_ok=True)
        meta = {}
        for g in a.groups:
            buf.clear()
            sys.argv = ["swv1_replay", "--groups", g] + (["--limit", str(a.limit)] if a.limit else [])
            R.main()
            fresh = pickle.load(open(R.O / "replay" / f"{g}.pkl", "rb"))
            stored = pickle.load(open(STORED / f"{g}.pkl", "rb"))
            scenes, ks, flat = [], [], []
            for sc, recs in fresh.items():
                for k, r in enumerate(recs):
                    scenes.append(sc), ks.append(k), flat.append(r)
            if len(flat) != len(buf):
                raise RuntimeError(f"{g}: {len(flat)} replayed decisions but {len(buf)} captured")
            for r, b in zip(flat, buf):
                if int(r["now"]) != b["now_us"] or np.abs(r["poses"] - b["poses"]).max() > 0:
                    raise RuntimeError(f"{g}: capture order does not match the replay at {r['now']}")
            # identity gate against the stored swv1 replay (served poses, P0 poses, every head slice)
            d_pose, d_head = [], 0.0
            for sc, recs in fresh.items():
                for r, s in zip(recs, stored[sc]):
                    d_pose.append(np.abs(r["poses"] - s["poses"]).max())
                    for name in ("FT", "P0"):
                        for key, v in r[name].items():
                            d_head = max(d_head, float(np.abs(v.astype(np.float32) - s[name][key].astype(np.float32)).max()))
            d_pose = np.array(d_pose)
            gate = dict(n=len(d_pose), max_abs_pose_diff_m=float(d_pose.max()), share_over_0p03_m=float((d_pose > 0.03).mean()), max_abs_head_diff=d_head,
                        scenes_match=bool(list(fresh) == list(stored)))
            run.info(f"identity gate {g}: {gate}")
            arrs = dict(scene=np.array(scenes), k=np.array(ks, np.int32), now_us=np.array([b["now_us"] for b in buf], np.int64),
                        cmd=np.array([b["cmd"] for b in buf], np.int8), n_slots=np.array([int(b["valid"].sum()) for b in buf]),
                        cam_xy=np.array([b["cam"]["t"][:2] for b in buf]))
            for key in ("valid", "ego", "poses", "mu", "plan_raw", "frames"):
                arrs[key] = np.stack([b[key] for b in buf])
            for key in buf[0]["cam"]:
                arrs[f"cam_{key}"] = np.stack([b["cam"][key] for b in buf])
            # tokens exist for the valid slots only (in order); scatter them into the 4 slots, zeros elsewhere
            tok = np.zeros((len(buf), 4) + buf[0]["tokens"].shape[1:], np.float16)
            for n, b in enumerate(buf):
                tok[n, arrs["valid"][n]] = b["tokens"]
            arrs["tokens"] = tok
            np.savez(out / "dump" / f"{g}.npz", **arrs)
            sz = (out / "dump" / f"{g}.npz").stat().st_size
            meta[g] = dict(gate=gate, n=len(buf), bytes=sz, shapes={k: list(v.shape) for k, v in arrs.items()})
            run.info(f"{g}: {len(buf)} decisions, {sz / 1e9:.2f} GB")
            run.summary[g] = meta[g]
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            buf.clear()


if __name__ == "__main__":
    main()
