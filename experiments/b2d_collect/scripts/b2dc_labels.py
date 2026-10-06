#!/usr/bin/env python
"""Labels and dataset index of a B2D collection run (CPU; envs/simlingo: carla.Map from the OpenDrive files, scipy, cv2).

  $DATA_DIR/envs/simlingo/bin/python experiments/b2d_collect/scripts/b2dc_labels.py --data $DATA_DIR/runs/b2d_collect/data/<tag> \
      [--workers N] [--sdf-stride 2] [--force]

Per route of <data>/attempts/<rid>/: the newest attempt whose clip/DONE exists -> clip/labels.npz (lib/b2dc_labels.tick_labels, all ticks)
and clip/sdf.npz (drivable SDF every --sdf-stride ticks: `ticks`, `sdf` (k, 128, 96) float16, `footprint`). Then <data>/index.csv, one row
per route: route id, town, scenario type, attempt, leaderboard status / score / infractions (collisions, red lights, ...), ticks, sim and
wall seconds, real-time factor, video MB, labelled ticks with a full 4 s future; and <data>/index.json (totals).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE.parent / "lib")]
import b2dc_labels as L  # noqa: E402
from jevdrive import par  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", data_dir() / "third_party/carla/CARLA_0.9.15"))
MANIFEST = data_dir() / "runs/b2d_collect/routes/v1/manifest.csv"


def xodr(town):
    p = CARLA_ROOT / f"CarlaUE4/Content/Carla/Maps/{town}/OpenDrive/{town}.xodr"
    return p if p.exists() else CARLA_ROOT / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"


def pick_attempt(rdir: Path):
    """Newest attempt with a complete clip, else None."""
    for a in sorted(rdir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1, reverse=True):
        if (a / "clip" / "DONE").exists():
            return a
    return None


def result_row(adir: Path) -> dict:
    out = {}
    f = adir / "results.json"
    if f.exists():
        try:
            r = json.loads(f.read_text())["_checkpoint"]["records"][0]
            inf = r.get("infractions", {})
            out = {"lb_status": r.get("status"), "score_route": r["scores"]["score_route"], "score_penalty": r["scores"]["score_penalty"],
                   "ds": r["scores"]["score_composed"], "route_length": r.get("meta", {}).get("route_length"),
                   **{f"n_{k}": len(v) for k, v in inf.items()}}
        except (KeyError, IndexError, ValueError):
            pass
    a = adir / "attempt.json"
    if a.exists():
        out["attempt_status"] = json.loads(a.read_text()).get("status")
    return out


def work(args):
    rid, adir, stride, force = args
    clip = Path(adir) / "clip"
    lf, sf = clip / "labels.npz", clip / "sdf.npz"
    ego, route, meta = L.load_clip(clip)
    if force or not lf.exists() or lf.stat().st_mtime < (clip / "ego.npz").stat().st_mtime:
        lab = L.tick_labels(ego, route)
        np.savez_compressed(lf, **lab)
    else:
        lab = dict(np.load(lf))
    if force or not sf.exists():
        lanes = L.lane_points(meta["town"], str(xodr(meta["town"])))
        ticks = np.arange(0, len(ego["t"]), stride)
        sdf = L.clip_sdf(lanes, lab["xy_world"], lab["heading"], ticks)
        np.savez_compressed(sf, ticks=ticks, sdf=sdf, footprint=L.footprint(meta), x0=L.X0, y0=L.Y0, res=L.RES)
    t = meta.get("timing", {})
    return {"route_id": rid, "attempt": Path(adir).name, "town": meta["town"], "ticks": len(ego["t"]), "sim_s": round(len(ego["t"]) * L.DT, 2),
            "wall_s": t.get("wall_s"), "rtf": t.get("rtf"), "video_mb": t.get("video_mb"), "fut_ok": int(lab["fut_ok"].sum()),
            "turn_ticks": int((lab["cmd"][:, 0] + lab["cmd"][:, 2]).sum()), "moving_s": round(float((lab["speed"] > 0.5).sum() * L.DT), 1),
            "vid_missing": int((ego["vid"] < 0).sum()), "clip": str(clip), **result_row(Path(adir))}


def main(a):
    data = Path(a.data)
    man = {}
    if MANIFEST.exists():
        with open(MANIFEST) as fh:
            man = {r["route_id"]: r for r in csv.DictReader(fh)}
    with Run("b2d_collect", f"labels-{data.name}", config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("b2d/b2dc-train"))
        jobs, missing = [], []
        for rdir in sorted((data / "attempts").iterdir()) if (data / "attempts").exists() else []:
            adir = pick_attempt(rdir)
            (jobs.append((rdir.name, str(adir), a.sdf_stride, a.force)) if adir else missing.append(rdir.name))
        run.info(f"{len(jobs)} clips with DONE, {len(missing)} routes without a complete clip")
        res = par.pmap(work, jobs, run=run, workers=a.workers or None)
        rows = [r for r in res.values if isinstance(r, dict)]
        for r in rows:
            m = man.get(str(r["route_id"]), {})
            r.update(type=m.get("type", ""), src=m.get("src", ""), turn=m.get("turn", ""))
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(data / "index.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        tot = {"clips": len(rows), "missing": missing, "failed_units": len(res.errors), "ticks": int(sum(r["ticks"] for r in rows)),
               "sim_h": round(sum(r["sim_s"] for r in rows) / 3600, 3), "video_gb": round(sum(r["video_mb"] or 0 for r in rows) / 1e3, 3),
               "collisions": int(sum(r.get("n_collisions_vehicle", 0) + r.get("n_collisions_pedestrian", 0) + r.get("n_collisions_layout", 0)
                                     for r in rows)),
               "mean_ds": float(np.mean([r["ds"] for r in rows if r.get("ds") is not None])) if rows else None}
        (data / "index.json").write_text(json.dumps(tot, indent=1))
        run.summary.update(tot)
        res.raise_if_failed()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--sdf-stride", type=int, default=2)
    cli_args(ap)
    main(ap.parse_args())
