"""Pair inventory, WOD-E2E part: what the data can give without a map. CPU only; run with envs/jevdrive.
  python pair_inv_wod.py --out <dir>    -> <dir>/wod_summary.json
WOD-E2E carries no map and no route: the only command signal is the per-frame `intent` (UNKNOWN / straight / left /
right) plus the logged ego future. So different-command pairs (type 1) cannot be built; history-perturbation and
heading-offset pairs (types 2, 3) are counted. Train/val only (test has no future states).
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive import waymo as W

SPEED_BINS = [0, 1, 3, 6, 10, 15, 1e9]
SPEED_LAB = ["<1", "1-3", "3-6", "6-10", "10-15", ">=15"]
HIST_S = (2.0, 4.0)      # seconds of contiguous image history (10 frames/s)


def contiguous_back(df):
    """For every row: number of consecutive earlier frame indices of the same sequence that are present (exact steps of 1)."""
    o = np.lexsort((df.frame.to_numpy(), df.sequence.to_numpy()))
    seq = df.sequence.to_numpy()[o]; fr = df.frame.to_numpy()[o]
    new = np.r_[True, (seq[1:] != seq[:-1]) | (fr[1:] != fr[:-1] + 1)]
    run_start = np.maximum.accumulate(np.where(new, np.arange(len(fr)), 0))
    back = np.arange(len(fr)) - run_start
    out = np.empty(len(df), int); out[o] = back
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    df = W.load_index(); past, future = W.load_ego()
    S = {"index_rows": len(df), "by_split": df.split.value_counts().to_dict()}
    shards_on_disk = len(list((W.shard_dir()).glob("*.tfrecord-*")))
    S["shards_on_disk"] = shards_on_disk
    S["shards_in_index"] = int(df.shard.nunique())
    back = contiguous_back(df)
    v = np.linalg.norm(past[:, -1, 2:4], axis=1)
    sp = pd.cut(pd.Series(v), SPEED_BINS, labels=SPEED_LAB, right=False)
    ss = W.subsets(df, past, future)
    for split in ("train", "val"):
        m = (df.split == split).to_numpy() & df.has_future.to_numpy()
        R = {"frames": int(m.sum()), "sequences": int(df.sequence[m].nunique())}
        R["intent_counts"] = df.intent[m].map(dict(enumerate(W.INTENTS))).value_counts().to_dict()
        R["frames_by_speed"] = sp[m].value_counts().reindex(SPEED_LAB).astype(int).to_dict()
        for h in HIST_S:
            ok = m & (back >= round(h / W.FRAME_DT))
            R[f"hist_ge_{h:g}s_frames"] = int(ok.sum())
            R[f"hist_ge_{h:g}s_by_speed"] = sp[ok].value_counts().reindex(SPEED_LAB).astype(int).to_dict()
        R["past_states_4s_kinematic_history_frames"] = int(m.sum())   # past_states is always present (16 x 4 Hz)
        # "approach" proxies without a map: the logged intent is a turn, and the car is not turning yet / turning now
        for k in ("turn_intent", "pre_onset", "turn_yaw"):
            mk = m & ss[k]
            R[k] = {"frames": int(mk.sum()), "sequences": int(df.sequence[mk].nunique()),
                    "by_speed": sp[mk].value_counts().reindex(SPEED_LAB).astype(int).to_dict()}
        mk = m & ss["pre_onset"] & (back >= 20)
        R["pre_onset_hist_ge_2s"] = int(mk.sum())
        for itn, nm in ((1, "straight"), (2, "left"), (3, "right")):
            R[f"pre_onset_intent_{nm}"] = int((m & ss["pre_onset"] & (df.intent == itn).to_numpy()).sum())
        # speed < 3 m/s starts (the low-speed condition of the common-cause hypothesis)
        R["low_speed_lt3_hist_ge_2s"] = int((m & (v < 3) & (back >= 20)).sum())
        S[split] = R
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(S, open(out / "wod_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
