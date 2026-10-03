"""Per-run traces and scene features for the HUGSIM spin attribution (CPU only, box, hugsim env).
Plan: experiments/hugsim/plans/2026-10-04-spin-attribution-prereg.md
    spin_attr_extract.py <out.json> <results.csv> [<results.csv> ...]     (results.csv with a run_dir column; box paths)
One entry per run: scenario, tag, per-step v / theta (deg, + right as the simulator) / pos / steer / lead_prob / engaged / plan point at 1 s
(x right, y forward after the agent's processing) / raw plan / nearest actor in the ego lane (obj_boxes) / n_obj, and scene features at step 0:
scene.ply points blocking the ego line (3..25 m ahead, |lateral| < 1.2 m, ego height band), ground.ply lateral extents (3..25 m ahead),
route heading over the first 10 / 20 / 40 m relative to the start.
"""
import csv
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def one(d):
    d = Path(d)
    out = {"dir": str(d)}
    L = [json.loads(x) for x in open(d / "zs_steps.jsonl")]
    setup = L[0]
    st = [x for x in L if "step" in x]
    out["steps"] = [dict(step=s["step"], v=s["v"], th=s["theta"], pos=s["pos"], steer=s["steer"], lp=s.get("lead_prob"),
                         eng=s.get("engaged"), plan=s.get("plan"), raw=s.get("raw_plan"), mp=s.get("model_pos"),
                         n_obj=s.get("n_obj"), cmd=s.get("cmd")) for s in st]
    infos = pickle.load(open(d / "infos.pkl", "rb"))
    lead = []
    for i in infos:
        eb = i["ego_box"]
        yaw = eb[6]
        best = None
        for o in i["obj_boxes"]:
            dx, dy = o[0] - eb[0], o[1] - eb[1]
            fw = dx * np.cos(yaw) + dy * np.sin(yaw)
            lt = -dx * np.sin(yaw) + dy * np.cos(yaw)
            if 0 < fw < 60 and abs(lt) < 2.0:
                best = fw if best is None else min(best, fw)
        lead.append(best)
    out["lead_actor"] = lead
    out["ego_yaw"] = [float(i["ego_box"][6]) for i in infos]
    try:
        import open3d as o3d
        from scipy.spatial.transform import Rotation
        i = infos[0]
        x0, y0, z0 = i["ego_pos"]
        M = Rotation.from_euler("XYZ", np.asarray(i["ego_rot"])).as_matrix()
        th = np.arctan2(M[0, 2], M[2, 2])
        f = np.array([np.sin(th), np.cos(th)])
        r = np.array([np.cos(th), -np.sin(th)])
        for name in ("ground", "scene"):
            P = np.asarray(o3d.io.read_point_cloud(str(d / f"{name}.ply")).points)
            rel = P[:, [0, 2]] - [x0, z0]
            fw, rt = rel @ f, rel @ r
            if name == "ground":
                m = (fw > 3) & (fw < 25) & (np.abs(rt) < 15)
                a = rt[m]
                out["ground_n"] = int(m.sum())
                if len(a) > 50:
                    out["gl_p90"], out["gr_p90"] = float(-np.percentile(a, 5)), float(np.percentile(a, 95))
                    out["gl_med"], out["gr_med"] = float(-np.median(a[a < 0])) if (a < 0).any() else 0.0, float(np.median(a[a > 0])) if (a > 0).any() else 0.0
            else:
                m = (fw > 3) & (fw < 25) & (np.abs(rt) < 1.2) & (np.abs(P[:, 1] - y0) < 1.0)
                out["blocked_pts"] = int(m.sum())
                out["blocked_near"] = float(fw[m].min()) if m.any() else None
                out["blocked_pts_15"] = int((m & (fw < 15)).sum())
    except Exception as e:  # noqa: BLE001
        out["scene_err"] = repr(e)
    return out


def route_feats(ds, scene, H):
    import io
    import zipfile
    zp = H / "scenes" / ds / f"{scene}.zip"
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith("ground_param.pkl"))
        cam_poses, _, cmds = pickle.load(io.BytesIO(z.read(name)))
    xz = cam_poses[:, [0, 2], 3]
    fwd = cam_poses[:, [0, 2], 2]
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 1]))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    o = {}
    for D in (10, 20, 40):
        k = int(np.searchsorted(s, D))
        k = min(k, len(s) - 1)
        o[f"route_dyaw{D}"] = float(np.degrees(yaw[k] - yaw[0]))
    return o


def main():
    out, csvs = sys.argv[1], sys.argv[2:]
    rows = []
    for c in csvs:
        for r in csv.DictReader(open(c)):
            rows.append(r)
    dirs = [r["run_dir"] for r in rows]
    with ProcessPoolExecutor(32) as ex:
        res = list(ex.map(one, dirs))
    H = Path.home() / "data" / "datasets" / "hugsim"
    cache = {}
    for r, o in zip(rows, res):
        o.update({k: r[k] for k in ("scenario", "dataset", "difficulty", "agent", "controller", "tag", "hdscore", "end", "scene")})
        key = (r["dataset"], r["scene"])
        if key not in cache:
            try:
                cache[key] = route_feats(*key, H)
            except Exception as e:  # noqa: BLE001
                cache[key] = {"route_err": repr(e)}
        o.update(cache[key])
    json.dump(res, open(out, "w"))
    print(len(res), "runs ->", out)


if __name__ == "__main__":
    main()
