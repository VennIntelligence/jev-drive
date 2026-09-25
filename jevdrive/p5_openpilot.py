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
    from .p5_pairs import processed
    return processed(*parts)


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


def finalize(model: str) -> dict:
    import pandas as pd
    files = sorted(root("op_streams", model).glob("*.npz"))
    parts = []
    for f in files:
        with np.load(f) as z:
            parts.append({k: z[k] for k in ("name", "temporal", "hist")})
    fn = np.concatenate([p["name"] for p in parts]).astype(str)
    X = np.concatenate([p["temporal"] for p in parts]).astype(np.float16)
    dst = root(f"op_{model}")
    np.save(dst / "temporal.npy", X)
    pd.DataFrame({"frame_name": fn, "hist": np.concatenate([p["hist"] for p in parts])}).to_parquet(dst / "index.parquet")
    return {"model": model, "rows": len(fn), "streams": len(files)}


def load(t, models=MODELS) -> dict:
    """{"op-<model> temporal": (n, 512) float32} aligned to the P5 index `t`."""
    import pandas as pd
    out = {}
    for m in models:
        d = root(f"op_{m}")
        names = pd.read_parquet(d / "index.parquet").frame_name
        pos = pd.Series(np.arange(len(names)), index=names)
        at = t.frame_name.map(pos)
        assert at.notna().all(), f"{m}: {int(at.isna().sum())} frames without features"
        out[f"op-{m} temporal"] = np.load(d / "temporal.npy")[at.astype(int).to_numpy()].astype(np.float32)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "finalize"))
    a = ap.parse_args()
    if a.step == "prepare":
        print(prepare())
    else:
        for m in MODELS:
            print(finalize(m))


if __name__ == "__main__":
    main()
