#!/usr/bin/env python
"""Rule-8 equivalence check of the closed-loop head arms (night queue 3, lane B; the [B] entry's item 6).

The head server dumps, for every request of a smoke route, the inputs (three JPEGs, desire, ego input) and what it
computed (model frames, Cinque `temporal`, Qwen `L18_last` / token row, the head's path). Here the offline code recomputes
each from the dumped inputs in a fresh process and compares bit for bit:

  op    (envs/openpilot)   model frames (p5_openpilot.render_blobs), a fresh Cinque session stepped in the dumped order
                           (4 steps per frame with the dumped desire, as p5_openpilot.run_stream), and the head (Heads)
  qwen  (repo .venv)       the 4-frame x 3-camera clip rebuilt from the consecutive dumps -> Qwen extractor, batch 1
  yolo  (envs/ultralytics) the three current JPEGs -> Detector + build_tokens
  ego   (repo .venv)       jevdrive.nq3_cl.ego_past against p4_carla.route_rows on recorded P5 v1 BA routes (the same
                           pose track through both), and the intent against the index

    python scripts/nq3_cl_check.py op <attempt dir> ... --out result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def dumps(adir):
    fs = sorted(Path(adir, "frames").glob("*.npz"))
    return [dict(np.load(f, allow_pickle=False)) for f in fs]


def same(a, b):
    return a is not None and b is not None and a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def check_op(adirs):
    import p5_openpilot as P5
    import wod_zeroshot_openpilot as WZ
    from jevdrive import drive_backbones as D
    from jevdrive import nq3_cl as CL
    from jevdrive.openpilot.model import OPModel
    from jevdrive.p5_openpilot import carla_calib
    WZ._init({}, {P5.SEQ: carla_calib()}, ".")
    heads = CL.Heads()
    taps = D.OP_TAPS["cinque"]
    m = OPModel("cinque", WZ.MODELS["cinque"], context_rate=False, taps=list(taps.values()))
    res = []
    for adir in adirs:
        ds = dumps(adir)
        m.reset()
        n = {"requests": len(ds), "img2": 0, "op": 0, "path": 0, "op_max_abs": 0.0, "path_max_abs": 0.0}
        for d in ds:
            img2 = P5.render_blobs([[d["jpg0"], d["jpg1"], d["jpg2"]]])[0]
            n["img2"] += same(img2, d["img2"])
            desire = np.zeros(8, np.float32)
            desire[int(d["desire"])] = 1
            for _ in range(P5.HOLD):
                m.step(img2, desire=desire, action_t=WZ.ACTION_T)
            op = m.tap_values[taps["temporal"]].copy()
            n["op"] += same(op, d["op"])
            n["op_max_abs"] = max(n["op_max_abs"], float(np.abs(op - d["op"]).max()))
            arm = str(d["arm"])
            if arm == "mc" and "q" not in d:
                arm = "ridge_late"
            path = heads.predict(arm, d["ego"], op, q=d.get("q"), tok=d.get("tok"))
            n["path"] += same(path, d["path"])
            n["path_max_abs"] = max(n["path_max_abs"], float(np.abs(path - d["path"]).max()))
        res.append({"attempt": str(adir), **n})
    return res


def check_feat(kind, adirs):
    import nq3_feat_server as FS
    f = FS.Qwen() if kind == "qwen" else FS.Yolo()
    key = "q" if kind == "qwen" else "tok"
    res = []
    for adir in adirs:
        ds = dumps(adir)
        n = {"requests": 0, "identical": 0, "max_abs": 0.0}
        for i, d in enumerate(ds):
            if key not in d:
                continue
            if kind == "qwen":
                clip = ds[i - 3:i + 1]
                blobs = [c["jpg%d" % cam] for cam in range(3) for c in clip]
            else:
                blobs = [d["jpg0"], d["jpg1"], d["jpg2"]]
            x = f.features(blobs)[key]
            n["requests"] += 1
            n["identical"] += same(x.astype(d[key].dtype), d[key])
            n["max_abs"] = max(n["max_abs"], float(np.abs(x - d[key]).max()))
        res.append({"attempt": str(adir), **n})
    return res


def check_ego(n_routes=6):
    import pandas as pd
    from jevdrive import nq3_cl as CL, p4_carla as p4
    t = pd.read_parquet(Path(__import__("os").environ["DATA_DIR"]) / "processed/carla_p5v1_ba/index.parquet")
    adirs = sorted({str(Path(f[0]).parents[2]) for f in t.files.iloc[:: max(1, len(t) // 200)]})[:n_routes]
    res = []
    for adir in adirs:
        a = Path(adir)
        meta = json.loads((a / "meta.json").read_text())
        rows = p4.route_rows(a, meta["route_id"], meta["town"])
        if rows is None:
            continue
        tt, past, _, route = rows
        pose = pd.read_json(a / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame").sort_index()
        f0 = pose.index.min()
        yaw = np.radians(pose.yaw.to_numpy())
        xy = np.stack([pose.x.to_numpy() + CL.REAR_AXLE_X * np.cos(yaw), pose.y.to_numpy() + CL.REAR_AXLE_X * np.sin(yaw)], -1)
        ra, th = CL.rh_track(xy, yaw)
        rx = route[["x", "y"]].to_numpy(float)
        cmd = route.option.to_numpy()
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(rx, axis=0), axis=1))]
        mx, it_ok = 0.0, 0
        for j, r in enumerate(tt.itertuples()):
            k = int(r.frame - f0)
            mx = max(mx, float(np.abs(CL.ego_past(ra, th, k) - past[j]).max()))
            it_ok += CL.intent(rx, cmd, s, xy[k], yaw[k]) == int(p4.route_intent(route, np.array([r.route_progress]), CL.INTENT_LOOKAHEAD_M)[0])
        res.append({"attempt": adir, "rows": len(tt), "ego_max_abs": mx, "intent_equal": it_ok})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("op", "qwen", "yolo", "ego"))
    ap.add_argument("adirs", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = check_op(a.adirs) if a.kind == "op" else check_ego() if a.kind == "ego" else check_feat(a.kind, a.adirs)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
