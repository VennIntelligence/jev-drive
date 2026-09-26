"""Real-data transfer G0 and the shared YOLO26x-seg detections / 64-d embeddings of this round
(todos/2026-09-26-real-data-transfer.md, G0 and deviation-log entries [G0], written before any G0 number).

  lists     image lists of every frame set (current frame x front / front_left / front_right), interleaved by set
            priority into disjoint slices, one per detector process
  (detect)  scripts/real_g0_detect.sh: jevdrive.fastperc detect on each slice, E5's detector config unchanged
  embed     detections -> flat-ground BEV with each frame's own calibration -> agent-legal corridor -> E5's k = 8
            embedding (64 dims) per frame set, plus READY markers; see HANDOFF.md in the output root
  students  E5's students refitted with the E5 code (checked against the stored run), weights and CARLA statistics
            kept, so the students can be applied to foreign rows
  wod / navsim / navsim-table / figs    G0 readouts, E1's functions unchanged

Run on the box (envs/jevdrive; the detector in envs/ultralytics): python -m jevdrive.real_g0 <step>
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
CAMS = ("front", "front_left", "front_right")
NAV_CAMS = {"front": "CAM_F0", "front_left": "CAM_L0", "front_right": "CAM_R0"}
SETS = ("i3", "wod_val", "navtest", "wod_train", "navtrain")       # detection priority order
YOLO = "yolo:yolo26x-seg.pt:640:half"


def root(*p) -> Path:
    d = data_dir() / "processed" / "real_transfer" / "yolo"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


# ---------------------------------------------------------------- frame sets and image lists

def _wod(names: np.ndarray) -> pd.DataFrame:
    from . import waymo as W
    df = W.load_index()
    at = pd.Series(np.arange(len(df)), index=W.frame_names(df)).reindex(names)
    assert at.notna().all(), f"{int(at.isna().sum())} WOD frames not in the index"
    r = df.iloc[at.astype(int).to_numpy()]
    sh = [str(W.shard_dir() / x) for x in r.shard]
    fr = pd.DataFrame({"frame_id": names, "sequence": r.sequence.astype(str).to_numpy(), "row": at.astype(int).to_numpy()})
    li = pd.concat([pd.DataFrame({"key": [f"{n}|{c}" for n in names], "path": "", "shard": sh,
                                  "off": r[f"{c}_off"].to_numpy(), "len": r[f"{c}_len"].to_numpy()}) for c in CAMS])
    return fr, li


def frame_sets() -> dict:
    """{set: (frames, image list)}; frames are one row per frame (frame_id first), lists one row per image."""
    from . import navsim_zs as Z
    from .waymo_ladder import subset_path
    D = data_dir()
    out = {}
    s = pd.read_parquet(subset_path())
    out["wod_val"] = _wod(s.frame_name.to_numpy())
    tr = D / "processed/waymo_e2e/features/qwenvid_train_t4"
    names = pd.concat([pd.read_parquet(sh / "index.parquet").frame_name for sh in sorted(tr.iterdir())
                       if sh.name.startswith("training_") and (sh / "index.parquet").exists()], ignore_index=True)
    out["wod_train"] = _wod(names.to_numpy())
    for split in ("navtest", "navtrain"):
        tok = np.load(D / "runs/navsim_zs/openpilot" / split / "cinque_temporal.npz")["tokens"]
        idx = {e["token"]: e["cams"][-1] for e in Z.load_index(split)}
        fr = pd.DataFrame({"frame_id": tok})
        li = pd.concat([pd.DataFrame({"key": [f"{t}|{c}" for t in tok], "path": [idx[t][NAV_CAMS[c]]["path"] for t in tok]})
                        for c in CAMS])
        out[split] = (fr, li)
    t = pd.read_parquet(D / "processed/hugsim_pairs/index.parquet")
    fr = t[["frame_name", "route_id", "base_id", "world", "role"]].rename(columns={"frame_name": "frame_id"})
    li = []
    for i, c in enumerate(CAMS):
        p = t.files.map(lambda f, i=i: f[4 * i + 3])
        assert p.str.contains(f"/{c}/").all()
        li.append(pd.DataFrame({"key": t.frame_name + f"|{c}", "path": p}))
    out["i3"] = (fr.reset_index(drop=True), pd.concat(li))
    return out


def lists(n_slices: int) -> dict:
    """Write frames.parquet per set and n_slices disjoint detector lists (keys prefixed '<set>:')."""
    fs = frame_sets()
    parts = []
    for name in SETS:
        fr, li = fs[name]
        d = root(name)
        d.mkdir(exist_ok=True)
        fr.to_parquet(d / "frames.parquet", index=False)
        li = li.assign(key=name + ":" + li.key).reset_index(drop=True)
        for c in ("shard", "off", "len"):
            if c not in li:
                li[c] = "" if c == "shard" else 0
        parts.append(li.sample(frac=1.0, random_state=0))          # mix sequences so every slice sees every set
        log.info("%s: %d frames, %d images", name, len(fr), len(li))
    allimg = pd.concat(parts, ignore_index=True)
    sl = root("slices")
    sl.mkdir(exist_ok=True)
    for k in range(n_slices):
        allimg.iloc[k::n_slices].to_parquet(sl / f"slice_{k:02d}.parquet", index=False)
    summ = {n: int((allimg.key.str.split(":").str[0] == n).sum()) for n in SETS}
    (sl / "summary.json").write_text(json.dumps({"slices": n_slices, "images": len(allimg), **summ}, indent=1))
    log.info("%d images in %d slices: %s", len(allimg), n_slices, summ)
    return summ


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("lists",))
    ap.add_argument("--slices", type=int, default=30)
    a = ap.parse_args()
    if a.step == "lists":
        lists(a.slices)


if __name__ == "__main__":
    main()
