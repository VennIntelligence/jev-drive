#!/usr/bin/env python
"""Collect per-step traces of HUGSIM exam runs into compact JSON (box side, envs/hugsim), for the figures and the
failure analysis (todos/2026-09-25-hugsim-exam). Per run: the ego track from zs_steps.jsonl (or infos.pkl for the
baselines), its signed lateral offset (+ right) and heading error to the recorded route, speed, command, the plans
sent, the end reason and the HD-Score terms per step.

    $DATA_DIR/envs/hugsim/bin/python scripts/hugsim/zs_collect.py $DATA_DIR/runs/hugsim-exam/checklist --out traces.json
"""
import argparse
import csv
import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
H = D / "datasets" / "hugsim"
_routes = {}


def route(ds, scene):
    if (ds, scene) not in _routes:
        d = H / "scenes" / ds / scene / "ground_param.pkl"
        if d.exists():
            cam_poses, _, cmds = pickle.load(open(d, "rb"))
        else:
            with zipfile.ZipFile(H / "scenes" / ds / f"{scene}.zip") as z:
                cam_poses, _, cmds = pickle.load(io.BytesIO(z.read(f"{scene}/ground_param.pkl")))
        xz = cam_poses[:, [0, 2], 3]
        fwd = cam_poses[:, [0, 2], 2]
        # densify to ~0.1 m for the nearest-point search
        seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        sq = np.arange(0, s[-1], 0.1)
        dx = np.stack([np.interp(sq, s, xz[:, k]) for k in (0, 1)], -1)
        th = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 1]))
        _routes[(ds, scene)] = (dx, np.interp(sq, s, th), xz, np.asarray(cmds))
    return _routes[(ds, scene)]


def offsets(pos, theta, rt):
    dx, th, _, _ = rt
    i = np.argmin(((dx[None] - pos[:, None]) ** 2).sum(-1), axis=1)
    r = np.stack([np.cos(th[i]), -np.sin(th[i])], -1)
    lat = ((pos - dx[i]) * r).sum(-1)
    herr = (theta - th[i] + np.pi) % (2 * np.pi) - np.pi
    return lat, herr, i * 0.1


def one(run_dir, row):
    ds = row["dataset"]
    steps = []
    zs = run_dir / "zs_steps.jsonl"
    if zs.exists():
        lines = [json.loads(x) for x in zs.read_text().splitlines() if x.strip()]
        setup = next((x for x in lines if x.get("setup")), {})
        steps = [x for x in lines if "step" in x]
    else:
        setup = {}
        infos = pickle.load(open(run_dir / "infos.pkl", "rb"))
        from scipy.spatial.transform import Rotation
        for k, info in enumerate(infos):
            R = Rotation.from_euler("XYZ", info["ego_rot"]).as_matrix()
            steps.append({"step": k, "t": info["timestamp"], "pos": [info["ego_pos"][0], info["ego_pos"][2]],
                          "theta": float(np.arctan2(R[0, 2], R[2, 2])), "v": float(info["ego_velo"]),
                          "cmd": int(info["command"])})
    scene = row["scene"]
    rt = route(ds, scene)
    pos = np.array([s["pos"] for s in steps], float)
    th = np.unwrap(np.array([s["theta"] for s in steps], float))
    lat, herr, s_along = offsets(pos, th, rt) if len(pos) else (np.zeros(0),) * 3
    try:
        ev = json.load(open(run_dir / "eval.json"))
        det = ev.get("details", {})
    except (OSError, ValueError):
        ev, det = {}, {}
    keep = ("t", "v", "steer", "cmd", "plan", "raw_plan", "model_plan", "desire", "nav", "oracle", "infer_ms", "cot",
            "model_pos", "model_xy", "slots_dt", "lead_prob", "reps", "n_obj")
    return {"tag": row["tag"], "scenario": row["scenario"], "dataset": ds, "scene": scene, "end": row["end"],
            "hdscore": ev.get("hdscore"), "rc": ev.get("rc"), "setup": setup,
            "route": np.round(rt[2][::2], 2).tolist(), "route_len": float(len(rt[0]) * 0.1),
            "pos": np.round(pos, 3).tolist(), "theta": np.round(th, 4).tolist(), "lat": np.round(lat, 3).tolist(),
            "herr": np.round(herr, 4).tolist(), "s": np.round(s_along, 2).tolist(),
            "steps": [{k: s[k] for k in keep if k in s} for s in steps],
            "pdms_t": {k: v for k, v in det.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    root = Path(a.root)
    rows = list({(r["scenario"], r["tag"]): r for r in csv.DictReader(open(root / "results.csv"))}.values())
    want = set(a.tags.split(",")) if a.tags else None
    out = []
    for r in rows:
        if want and r["tag"] not in want:
            continue
        if not r.get("run_dir"):                      # rows written before the runner logged it
            sc, mode, k = r["scenario"].rsplit("-", 2)
            ad = {"cv": "jev", "route": "jev", "ltf": "ltf"}.get(r["agent"], "zs")
            r["scene"], r["run_dir"] = sc, str(root / r["tag"] / ad / f"{sc}_{mode}_{k}")
        try:
            out.append(one(Path(r["run_dir"]), r))
        except Exception as e:  # noqa: BLE001 - a broken run must not stop the collection
            print("skip", r["tag"], r["scenario"], repr(e))
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} runs -> {a.out}")


if __name__ == "__main__":
    main()
