#!/usr/bin/env python
"""nuScenes open-loop zero-shot exam, project venv (todos/2026-09-24-zeroshot-exam/nuscenes-physicalai.md).

  index   val keyframe index (jevdrive.nuscenes_zs.build_index) -> $DATA_DIR/processed/nusc_zs/val_index.pkl
  sets    the pre-registered evaluation sets (main / full-history / Alpamayo subsets) -> sets.json next to it
  cv      constant-velocity baseline predictions (straight ahead at the current speed) and the logged future

    .venv/bin/python scripts/nusc_zs.py index
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

MIN_HIST_S = 1.5    # main set: t0 at least 1.5 s after the scene's first keyframe (Alpamayo's egomotion window)
FULL_HIST_S = 5.0   # full-history subset: openpilot's ~5 s context is filled


def sets_path() -> Path:
    return Z.index_path().with_name("sets.json")


def build_sets(idx: dict, seed: int = 0) -> dict:
    s = idx["samples"]
    valid = [e["token"] for e in s if e["valid"]]
    main = [e["token"] for e in s if e["valid"] and e["hist_s"] >= MIN_HIST_S - 1e-3]
    full = [e["token"] for e in s if e["valid"] and e["hist_s"] >= FULL_HIST_S - 1e-3]
    rng = np.random.default_rng(seed)
    perm = [main[k] for k in rng.permutation(len(main))]
    return {"valid": valid, "main": main, "fullhist": full,
            "quarter": sorted(perm[:len(main) // 4], key=main.index),   # seed-0 1/4 of main, in index order
            "half": sorted(perm[:len(main) // 2], key=main.index)}


def cmd_index(a, log):
    idx = Z.build_index(log.info)
    p = Z.index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(idx, f, protocol=5)
    sets = build_sets(idx)
    json.dump(sets, open(sets_path(), "w"))
    s = {e["token"]: e for e in idx["samples"]}
    for k, v in sets.items():
        cmds = [s[t]["cmd"] for t in v]
        log.info(f"set {k}: n={len(v)} scenes={len({s[t]['scene'] for t in v})} "
                 f"cmd={ {c: cmds.count(c) for c in Z.CMD_NAMES} }")
    miss = {c: sum(not (Z.dataroot() / p).exists() for sc in idx["scenes"].values() for p in sc["cams"][c]["path"])
            for c in Z.CAMS}
    log.info(f"missing files per camera over the val scenes: {miss}")
    log.info(f"-> {p}")


def cmd_cv(a, log):
    """Baselines through our own pipeline: cv (straight at the current speed, BEV-Planner's GoStraight) and cvv
    (constant velocity vector, heading of the last 0.5 s displacement); both at the LIDAR_TOP point."""
    idx = Z.load_index()
    sc = idx["scenes"]
    rows = {"cv": {}, "cvv": {}}
    for e in idx["samples"]:
        if not e["valid"]:
            continue
        scene = sc[e["scene"]]
        v = Z.cv_speed(scene, e["t0"])
        t = e["fut_t"]
        rows["cv"][e["token"]] = Z.to_lidar_point(t, np.c_[v * t, 0 * t], 0 * t, scene["lidar_xyz"], t)[0]
        xyz, R = Z.ego_at(scene, [e["t0"] - 5e5, e["t0"]])
        d = (xyz[1] - xyz[0]) @ R[1]          # displacement in the t0 frame
        h = np.arctan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 0.1 else 0.0
        rows["cvv"][e["token"]] = Z.to_lidar_point(t, np.c_[v * t * np.cos(h), v * t * np.sin(h)], 0 * t + h,
                                                   scene["lidar_xyz"], t)[0]
    for k, r in rows.items():
        toks = sorted(r)
        np.savez(Z.root("preds") / f"{k}.npz", tokens=np.array(toks), pred=np.stack([r[t] for t in toks]).astype(np.float32))
    log.info(f"cv / cvv for {len(rows['cv'])} valid samples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("index", "cv"))
    a = ap.parse_args()
    log = RunLog("nusc_zs", a.cmd)
    {"index": cmd_index, "cv": cmd_cv}[a.cmd](a, log)
    log.event("end")
