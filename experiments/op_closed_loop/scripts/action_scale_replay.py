#!/usr/bin/env python
"""Shipped Cinque on real comma1M drives (native comma 3/3X rig, real 20 Hz frames): action head vs what the car did.

The only local real data with openpilot's own camera rig. Windows of WARM + SCORE frames are picked from the localizer
(speed > 3 m/s over most of the scored part, stratified by peak yaw rate so curves and junctions are represented), the
model is stepped on every frame (its temporal queue needs 20 Hz) and per frame we keep the raw action slice, the plan
mean and the localizer motion (speed, yaw rate in the calibrated frame, z down: + = right, openpilot's curvature sign).

  python action_scale_replay.py pick <out_dir> [--n 32]
  python action_scale_replay.py run <out_dir> [--shard K --nshard N --threads 12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ROOT = Path.home() / "data/datasets/comma1M"
WARM, SCORE = 120, 200            # 6 s warm-up (> the 5 s context), 10 s scored


def motion(meta):
    from jevdrive.openpilot.frames import rot_from_euler
    cfd = rot_from_euler(meta["rpy_calib"]).T                      # calib_from_device
    om = meta["omega_dev"] @ cfd.T                                 # (N, 3) in the calibrated frame
    return np.linalg.norm(meta["vel"], axis=1), om[:, 2]


def cmd_pick(a):
    from jevdrive.openpilot.frames import load_segment_meta
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not ((d / "fcamera.hevc").exists() and (d / "ecamera.hevc").exists()):
            continue
        try:
            meta = load_segment_meta(d)
        except Exception as e:  # noqa: BLE001
            print("skip", d.name, e)
            continue
        if meta["fcam_wh"] != (1928, 1208) or not meta["has_ecam"]:
            continue
        v, yr = motion(meta)
        n = len(v)
        for s in range(0, n - WARM - SCORE, 100):
            sc = slice(s + WARM, s + WARM + SCORE)
            if (v[sc] > 3).mean() < 0.9:
                continue
            rows.append(dict(seg=d.name, start=s, peak=float(np.abs(yr[sc]).max()), vmed=float(np.median(v[sc]))))
    rng = np.random.default_rng(0)
    pick, used = [], set()
    # strata of peak yaw rate (rad/s): straight < 0.05, curve 0.05-0.2, junction-like > 0.2; at most one window per segment per stratum
    for lo, hi, k in ((0.0, 0.05, a.n // 4), (0.05, 0.2, a.n * 3 // 8), (0.2, 9.0, a.n - a.n // 4 - a.n * 3 // 8)):
        c = [r for r in rows if lo <= r["peak"] < hi]
        rng.shuffle(c)
        for r in c:
            if len([p for p in pick if lo <= p["peak"] < hi]) >= k:
                break
            if (r["seg"], lo) in used:
                continue
            used.add((r["seg"], lo))
            pick.append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(pick, open(out / "windows.json", "w"), indent=1)
    print(len(rows), "candidate windows,", len(pick), "picked;", [sum(lo <= p["peak"] < hi for p in pick) for lo, hi in ((0, .05), (.05, .2), (.2, 9))])


def cmd_run(a):
    from jevdrive.openpilot.frames import load_segment_meta, segment_model_frames
    from jevdrive.openpilot.model import OPModel, mdn_mu
    out = Path(a.out)
    win = json.load(open(out / "windows.json"))[a.shard::a.nshard]
    m = OPModel("cinque", "cpu", threads=a.threads)
    sl = m.slices
    for w in win:
        f = out / f"{w['seg']}_{w['start']}.npz"
        if f.exists():
            continue
        meta = load_segment_meta(ROOT / w["seg"])
        hi = w["start"] + WARM + SCORE
        fr, _ = segment_model_frames(ROOT / w["seg"], meta, max_frames=hi)
        fr = fr[w["start"]:hi]
        m.reset()
        act, plan = [], []
        for x in fr:
            raw = m.step(x, desire=np.zeros(8), traffic=(1, 0), action_t=(0.275, 0.525))
            act.append(raw[sl["action"]])
            plan.append(mdn_mu(raw[sl["plan"]], (33, 15)))
        v, yr = motion(meta)
        idx = np.arange(w["start"], w["start"] + len(fr))
        np.savez_compressed(f, act=np.array(act, np.float32), plan=np.array(plan, np.float32), t=meta["t_loc"], v=v, yr=yr,
                            idx=idx, warm=WARM, pos=meta["pos"], R=meta["R"], rpy_calib=meta["rpy_calib"])
        print("done", f.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "run"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)
