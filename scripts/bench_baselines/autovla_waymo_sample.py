"""Sample WOD-E2E val frames by stratum and write them where the AutoVLA venv can read them.

AutoVLA runs in its own venv without `jevdrive`, so sampling and extraction happen here and the benchmark
(`autovla_waymo.py`) only reads the manifest. Strata are our own subset definitions (jevdrive/waymo.subsets),
not the authors' evaluation setup:

  straight   `straight_yaw`: not turning now and not turning within the 3 s onset horizon
  turning    `turn_yaw`: |yaw rate| >= 5 deg/s, the car is already turning
  pre_onset  not turning yet (|yaw rate| < 1 deg/s) but turning within 3 s: where the ego prior cannot know

Sampling rule: of the val frames whose 4-frame image window at stride 5 (0.5 s, so 2 Hz over 1.5 s) is
strictly complete for all three cameras, take `--per-stratum` frames per stratum uniformly without
replacement, at most one per sequence, with `--seed`.

  cd ~/data/jev-drive && UV_PROJECT_ENVIRONMENT=$DATA_DIR/envs/jevdrive PYTHONPATH=. uv run --no-sync \
    python scripts/bench_baselines/autovla_waymo_sample.py    # ~570 MB of JPEGs into $DATA_DIR/processed/autovla_waymo
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from jevdrive import waymo as W

p = argparse.ArgumentParser()
p.add_argument("--per-stratum", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-back", type=int, default=3)
p.add_argument("--stride", type=int, default=5)
p.add_argument("--out", type=Path, default=Path(os.environ["DATA_DIR"]) / "processed/autovla_waymo")
args = p.parse_args()

df, (past, future) = W.load_index(), W.load_ego()
val = (df.split == "val").to_numpy()
masks = W.subsets(df, past, future)
complete = np.zeros(len(df), bool)
targets = np.flatnonzero(val)
complete[targets] = W.history_set(df, args.n_back, args.stride, targets=targets)

rng = np.random.default_rng(args.seed)
sequence = df.sequence.astype(str).to_numpy()
picked: dict[str, np.ndarray] = {}
for stratum in ("straight_yaw", "turn_yaw", "pre_onset"):
    pool = np.flatnonzero(val & complete & masks[stratum])
    rng.shuffle(pool)
    seen, keep = set(), []
    for row in pool:                              # at most one frame per sequence, so scenes are independent
        if sequence[row] not in seen:
            seen.add(sequence[row])
            keep.append(row)
        if len(keep) == args.per_stratum:
            break
    picked[stratum] = np.array(sorted(keep), np.int64)
    print(f"{stratum}: pool {len(pool)} frames in {len(set(sequence[pool]))} sequences -> {len(keep)} sampled")

rows = np.concatenate(list(picked.values()))
stratum_of = {int(r): s for s, rs in picked.items() for r in rs}
hist, dt, exact = W.history_rows(df, args.n_back, args.stride, targets=rows)
assert exact.all(), "sampled a frame with an incomplete window"

args.out.mkdir(parents=True, exist_ok=True)
frames_dir = args.out / "frames"
frames_dir.mkdir(exist_ok=True)
shard_of = (W.shard_dir().as_posix() + "/" + df.shard.astype(str)).to_numpy()
manifest = []
for i, row in enumerate(rows):
    name = f"{df.sequence.iloc[row]}-{df.frame.iloc[row]:03d}"
    images = {}
    for cam in W.CAMS:                            # oldest first, as AutoVLA's prompt expects
        paths = []
        for slot in range(args.n_back, -1, -1):
            src = hist[i, slot]
            off, length = int(df[f"{cam}_off"].iloc[src]), int(df[f"{cam}_len"].iloc[src])
            path = frames_dir / f"{name}_{cam}_{args.n_back - slot}.jpg"
            if not path.exists():
                fd = os.open(shard_of[src], os.O_RDONLY)
                try:
                    path.write_bytes(os.pread(fd, length, off))
                finally:
                    os.close(fd)
            paths.append(str(path))
        images[f"{cam}_camera"] = paths
    kin = W.past_kinematics(past[row:row + 1])
    manifest.append({
        "name": name, "row": int(row), "stratum": stratum_of[int(row)],
        "intent": W.INTENTS[int(df.intent.iloc[row])],
        "command": {"GO_STRAIGHT": "go straight", "GO_LEFT": "turn left", "GO_RIGHT": "turn right",
                    "UNKNOWN": "go straight"}[W.INTENTS[int(df.intent.iloc[row])]],
        "velocity": past[row, -1, 2:4].astype(float).tolist(),      # vel x/y at t=0, m/s
        "acceleration": past[row, -1, 4:6].astype(float).tolist(),  # accel x/y at t=0, m/s^2
        "speed": float(np.hypot(*past[row, -1, 2:4])),
        "yaw_rate_deg": float(np.degrees(kin["w"][0])),
        "images": images,
        "future_xy": W.future_xy(future[row:row + 1])[0].astype(float).tolist(),
    })
(args.out / "manifest.json").write_text(json.dumps(manifest))
print(f"wrote {len(manifest)} frames to {args.out}")
