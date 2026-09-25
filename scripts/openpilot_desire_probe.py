"""Does an openpilot model act on turn / lane-change desires? On real comma1M video (the native model frames of the
rig study), replay each segment once without desire and once per desire kind with a 1 s desire held every 10 s
(modeld feeds only the rising edge, so this is one pulse per event), and compare the plan's lateral position at
2 / 4 s in the 3 s after each pulse with the no-desire replay of the same frames (paired, same frames).

    CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/openpilot/bin/python scripts/openpilot_desire_probe.py
"""
import json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.openpilot.model import OPModel, T_IDXS, decode  # noqa: E402

ROOT = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "runs" / "openpilot_rigs"
BACKEND = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}
DESIRES = {"turnLeft": 1, "turnRight": 2, "laneChangeLeft": 3, "laneChangeRight": 4}
ACTION_T = (0.275, 0.525)


def replay(m, frames, desire_idx):
    m.reset()
    y2, y4 = [], []
    for i, f in enumerate(frames):
        d = np.zeros(8, np.float32)
        if desire_idx and i >= 100 and (i - 100) % 200 < 20:        # held 1 s every 10 s from 5 s on
            d[desire_idx] = 1
        out = decode(m.step(f, desire=d, action_t=ACTION_T), m.slices, 10.0, ACTION_T)
        p = out["plan_pos"]
        y2.append(np.interp(2.0, T_IDXS, p[:, 1]))
        y4.append(np.interp(4.0, T_IDXS, p[:, 1]))
    return np.array(y2), np.array(y4)


def main():
    segs = sorted(p.parent.name for p in ROOT.glob("frames/*/native.npy"))
    rows = []
    for name, bk in BACKEND.items():
        m = OPModel(name, bk)
        for sid in segs:
            frames = np.load(ROOT / "frames" / sid / "native.npy", mmap_mode="r")
            base = replay(m, frames, 0)
            win = np.array([i for i in range(len(frames)) if i >= 100 and (i - 100) % 200 < 60])  # 3 s after onset
            for kind, k in DESIRES.items():
                y2, y4 = replay(m, frames, k)
                rows.append(dict(model=name, seg=sid, desire=kind, n=len(win),
                                 dy2=float(np.mean(y2[win] - base[0][win])), dy4=float(np.mean(y4[win] - base[1][win])),
                                 dy4_max=float(np.max(np.abs(y4[win] - base[1][win])))))
                r = rows[-1]
                print(f"{name:8s} {sid[:8]} {kind:15s} mean dy@2s {r['dy2']:+.2f} m  dy@4s {r['dy4']:+.2f} m  "
                      f"max|dy@4s| {r['dy4_max']:.2f} m", flush=True)
    (ROOT / "desire_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
