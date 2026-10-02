"""Scene context at the step a Cinque episode starts to yaw: how much drivable ground lies left versus right of the ego, and
whether scene geometry blocks the lane ahead. Run on the box (hugsim env; open3d), CPU only.
    python spin_scene_context.py <spin_episodes.csv> <results.csv of scored-op> <out.csv> <runs_root>
For every cinque / lebowski PR#57 run: at step s (the divergence start from spin_analysis for spin episodes, 8 otherwise) the
ego pose (infos.pkl), ground points (ground.ply, camera coordinates x right, z forward) in the ego frame within 3..25 m ahead:
asym = (L - R) / (L + R) with L / R the ground points more than 3 m left / right of the ego line; blocked = scene points
(scene.ply non-ground) within 1.2 m of the ego line, 3..15 m ahead, at the ego height band (|y - ego_y| < 1.0).
"""
import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def one(job):
    import open3d as o3d
    d, step = job
    infos = pickle.load(open(Path(d) / "infos.pkl", "rb"))
    s = min(step, len(infos) - 1)
    i = infos[s]
    x0, y0, z0 = i["ego_pos"]
    R = np.asarray(i["ego_rot"])
    from scipy.spatial.transform import Rotation
    M = Rotation.from_euler("XYZ", R).as_matrix()
    th = np.arctan2(M[0, 2], M[2, 2])
    f = np.array([np.sin(th), np.cos(th)])
    r = np.array([np.cos(th), -np.sin(th)])
    out = {}
    for name in ("ground", "scene"):
        P = np.asarray(o3d.io.read_point_cloud(str(Path(d) / f"{name}.ply")).points)
        rel = P[:, [0, 2]] - [x0, z0]
        fw, rt = rel @ f, rel @ r
        if name == "ground":
            m = (fw > 3) & (fw < 25)
            L, Rr = int(((rt < -3) & m).sum()), int(((rt > 3) & m).sum())
            out["ground_left"], out["ground_right"] = L, Rr
            out["asym"] = (L - Rr) / max(L + Rr, 1)
        else:
            m = (fw > 3) & (fw < 15) & (np.abs(rt) < 1.2) & (np.abs(P[:, 1] - y0) < 1.0)
            out["blocked_pts"] = int(m.sum())
    return out


def main(spin_csv, results_csv, out, root):
    spin = {(r["scenario"], r["agent"], r["controller"]): r for r in csv.DictReader(open(spin_csv))}
    jobs, meta = [], []
    for r in csv.DictReader(open(results_csv)):
        if r["controller"] != "fixed":
            continue
        s = spin.get((r["scenario"], r["agent"], "fixed"))
        step = int(s["start"]) if s and s["spin"] == "True" else 8
        d = Path(r["run_dir"])
        d = Path(root) / "scored-op" / r["tag"] / d.parent.name / d.name
        jobs.append((str(d), step))
        meta.append(dict(scenario=r["scenario"], agent=r["agent"], dataset=r["dataset"], difficulty=r["difficulty"],
                         spin=s["spin"] if s else "", first_dir=s.get("first_dir", "") if s else "", step=step,
                         end=r["end"]))
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    rows = [dict(m, **x) for m, x in zip(meta, res)]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "runs")


if __name__ == "__main__":
    main(*sys.argv[1:5])
