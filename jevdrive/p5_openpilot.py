"""openpilot `temporal` on the P5 counterfactual pairs (todos/2026-09-25-openpilot-temporal-p5-and-route.md,
experiment 1). The exam itself is `p5_exam` unchanged; this module only adds openpilot as a feature source.

  prepare   one stream per recorded run in the P5 index (P4 training routes and P5 worlds): every 5 Hz camera
            frame of the run from its first up to its last indexed frame, as JSON for the openpilot venv
  finalize  the runner's per-stream npz -> processed/carla_p5/op_<model>/{index.parquet, temporal.npy}
  load      features aligned to the P5 index, keyed "op-<model> temporal" (the examinee's tap name)

The runner is scripts/p5_openpilot.py. The cameras are P4's Waymo-calibrated rig (scripts/p4_carla_agent.py:
972 x 1079, f = 1113.5 px, Waymo principal point and k1 / k2), written here as a WOD calibration record so the
WOD extraction's renderer (`camgeom` rotation-only reprojection, nearest neighbour, chroma 2x2 mean) runs as is.
Imported by the openpilot venv: numpy only at import time.
"""
import json
from pathlib import Path

import numpy as np

from .common import data_dir

MODELS = ("cinque", "lebowski")
# P4 rig (p4_carla_agent.WAYMO_CAMS / DEFAULT): name, x, y, z in Waymo's rear-axle vehicle frame (+y left), yaw deg
RIG = (("front", 1.519, 0.026, 1.806, 0.0), ("front_left", 1.445, 0.153, 1.806, 45.0),
       ("front_right", 1.482, -0.116, 1.806, -45.0))
CAM = {"w": 972, "h": 1079, "f": 1113.5, "cu": 488.1, "cv": 719.2, "k1": -0.0736, "k2": -0.0366}


def carla_calib() -> dict:
    """The rig as `wod_zeroshot` calibration records, keyed like WOD CameraName (1 FRONT, 2 LEFT, 3 RIGHT)."""
    out = {}
    for i, (_, x, y, z, yaw) in enumerate(RIG, 1):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        E = np.eye(4)
        E[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]      # camera (x optical axis, y left, z up) -> vehicle
        E[:3, 3] = x, y, z
        out[str(i)] = {"intrinsic": [CAM["f"], CAM["f"], CAM["cu"], CAM["cv"], CAM["k1"], CAM["k2"], 0.0, 0.0, 0.0],
                       "extrinsic": E.ravel().tolist(), "width": CAM["w"], "height": CAM["h"]}
    return out


def root(*parts) -> Path:
    """p5_pairs.processed() without its pandas import (this module is loaded by the openpilot venv)."""
    import os
    p = data_dir() / "processed" / os.environ.get("P5_SET", "carla_p5") / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepare() -> dict:
    """op_plan.json: streams (attempt dir, frame names, JPEG triplets, target positions)."""
    import pandas as pd
    t = pd.read_parquet(root() / "index.parquet")
    adir = t.files.map(lambda f: f[0].rsplit("/cams/", 1)[0])
    streams = []
    for d, g in t.groupby(adir, sort=True):
        rid = g.route_id.iloc[0]
        want = set(g.frame_name)
        fr = sorted((json.loads(line) for line in open(Path(d) / "frames.jsonl")), key=lambda r: r["frame"])
        names = [f"{rid}-{r['frame']:07d}" for r in fr]
        last = max(i for i, n in enumerate(names) if n in want)
        fr, names = fr[: last + 1], names[: last + 1]
        tgt = [i for i, n in enumerate(names) if n in want]
        assert len(tgt) == len(want), f"{d}: {len(want) - len(tgt)} indexed frames not in frames.jsonl"
        streams.append({"key": f"{g.source.iloc[0]}_{rid}", "names": names, "targets": tgt,
                        "files": [[f"{d}/{r['files'][c]}" for c, *_ in RIG] for r in fr],
                        "gaps": int((np.diff([r["frame"] for r in fr]) != 4).sum())})
    plan = {"calib": carla_calib(), "streams": streams}
    (root() / "op_plan.json").write_text(json.dumps(plan))
    info = {"streams": len(streams), "frames": sum(len(s["names"]) for s in streams),
            "targets": sum(len(s["targets"]) for s in streams), "gaps": sum(s["gaps"] for s in streams),
            "indexed": len(t)}
    assert info["targets"] == len(t)
    return info


def reuse(src_set: str = "carla_p5", sub: str = "op_streams") -> dict:
    """Link the runner's per-stream outputs from an earlier set for every stream of this plan that is the same
    stream there: same frame names, same targets and the same JPEG files (directories resolved, so P5 v1's
    BehaviorAgent worlds, which are symlinks to v0's runs, match v0's streams). The runner skips linked streams."""
    import os
    src = data_dir() / "processed" / src_set
    if src == root() or not (src / "op_plan.json").exists():
        return {"linked": 0}
    real = {}

    def rp(f):
        d, b = f.rsplit("/", 1)
        if d not in real:
            real[d] = os.path.realpath(d)
        return real[d] + "/" + b

    def sig(s):
        return s["names"], s["targets"], [[rp(f) for f in trip] for trip in s["files"]]
    old = {s["key"]: s for s in json.loads((src / "op_plan.json").read_text())["streams"]}
    n = 0
    for s in json.loads((root() / "op_plan.json").read_text())["streams"]:
        o = old.get(s["key"])
        if o is None or o["names"] != s["names"] or o["targets"] != s["targets"] or sig(o) != sig(s):
            continue
        for m in MODELS:
            f, dst = src / sub / m / f"{s['key']}.npz", root(sub, m) / f"{s['key']}.npz"
            if f.exists() and not dst.exists():
                dst.symlink_to(f)
        n += 1
    return {"linked_streams": n, "from": str(src / sub)}


def _dst(model: str, sub: str) -> Path:
    """op_streams -> op_<model>; op_streams_<x> -> op_<model>_<x> (the reactivity D0 set keeps its own copy)."""
    return root(f"op_{model}" + sub.removeprefix("op_streams"))


def finalize(model: str, arrays=("temporal",), sub: str = "op_streams") -> dict:
    import pandas as pd
    files = sorted(root(sub, model).glob("*.npz"))
    parts = []
    for f in files:
        with np.load(f) as z:
            parts.append({k: z[k] for k in ("name", "hist", *arrays)})
    fn = np.concatenate([p["name"] for p in parts]).astype(str)
    dst = _dst(model, sub)
    for k in arrays:
        np.save(dst / f"{k}.npy", np.concatenate([p[k] for p in parts]).astype(np.float16))
    pd.DataFrame({"frame_name": fn, "hist": np.concatenate([p["hist"] for p in parts])}).to_parquet(dst / "index.parquet")
    return {"model": model, "arrays": list(arrays), "rows": len(fn), "streams": len(files)}


def load(t, models=MODELS, arrays=("temporal",), sub: str = "op_streams") -> dict:
    """{"op-<model> <array>": (n, d) float32} aligned to the P5 index `t`."""
    import pandas as pd
    out = {}
    for m in models:
        d = _dst(m, sub)
        names = pd.read_parquet(d / "index.parquet").frame_name
        pos = pd.Series(np.arange(len(names)), index=names)
        at = t.frame_name.map(pos)
        assert at.notna().all(), f"{m}: {int(at.isna().sum())} frames without features"
        for k in arrays:
            out[f"op-{m} {k}"] = np.load(d / f"{k}.npy", mmap_mode="r")[at.astype(int).to_numpy()].astype(np.float32)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "reuse", "finalize"))
    ap.add_argument("--arrays", default="temporal")
    ap.add_argument("--sub", default="op_streams")
    ap.add_argument("--from-set", default="carla_p5")
    a = ap.parse_args()
    if a.step == "prepare":
        print(prepare())
    elif a.step == "reuse":
        print(reuse(a.from_set, a.sub))
    else:
        for m in MODELS:
            print(finalize(m, tuple(a.arrays.split(",")), a.sub))


if __name__ == "__main__":
    main()
