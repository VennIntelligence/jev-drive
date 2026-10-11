"""Box: the logged ego attitude and altitude over each public scene's window, from the MTGS asset (frame_infos[*].ego2global).
In the closed loop (physics NONE) AlpaSim renders the ego level (pose quaternion about z only) at the altitude of the hand-over
(rollout.asl actor_poses / render requests); the logged car pitches with the road. pitch_* is the logged nose-up pitch (deg) relative to
the world horizontal, dz_* the logged altitude change from the hand-over frame (m), over frames 18 .. 18 + 50 (the 5 s scene: frame 18 is
the render at k = 0, checked on one scene).

usage: python asset_grade.py <assets dir> <out.csv>     (AlpaSim venv)
"""
import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

F0, N = 18, 51


def one(p: Path) -> dict:
    v = list(pickle.load(open(p, "rb")).values())[0]
    fi = v["frame_infos"][F0:F0 + N]
    E = np.array([np.asarray(f["ego2global"], float) for f in fi])
    fwd = E[:, :3, 0]                                          # ego x axis in the world
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 2], -1, 1)))  # nose up positive
    left = E[:, :3, 1]
    roll = np.degrees(np.arcsin(np.clip(left[:, 2], -1, 1)))
    z = E[:, 2, 3]
    h = min(5, len(z) - 1)                                     # hand-over at 0.5 s
    return dict(scene=p.parent.name, n=len(fi), pitch_ho=pitch[h], pitch_mean=pitch[h:].mean(), pitch_absmax=np.abs(pitch[h:]).max(),
                dpitch_absmax=np.abs(pitch[h:] - pitch[h]).max(), roll_absmax=np.abs(roll[h:]).max(), dz_end=z[-1] - z[h],
                dz_absmax=np.abs(z[h:] - z[h]).max())


if __name__ == "__main__":
    ps = sorted(Path(sys.argv[1]).glob("*/video_scene_dict.pkl"))
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(one, ps, chunksize=16))
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items()} for r in rows)
    print(len(rows), "->", sys.argv[2])
