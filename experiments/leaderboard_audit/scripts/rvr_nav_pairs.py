"""Real vs render, navhard: for every stage-2 synthetic scene, the offset of each of its 4 frames from the logged ego pose at the
same timestamp (synthetic frames share the log's timestamps), so frames rendered at (nearly) the logged pose can be paired with the
real CAM_F0 of that log frame. navsim2 env (nuplan pickles), CPU:
  $DATA_DIR/envs/navsim2/bin/python experiments/leaderboard_audit/scripts/rvr_nav_pairs.py
Writes $DATA_DIR/runs/real_vs_render/nav_pairs.csv (one row per synthetic frame)."""
import csv, os, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
NAV = D / "datasets/navsim"
OUT = D / "runs/real_vs_render"
_LOGS = {}


def log(name):
    if name not in _LOGS:
        L = pickle.load(open(NAV / "navsim_logs/test" / f"{name}.pkl", "rb"))
        _LOGS[name] = {x["timestamp"]: x for x in L}
    return _LOGS[name]


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def one(p):
    d = pickle.load(open(p, "rb"))
    md = d["scene_metadata"]
    try:
        L = log(md["log_name"])
    except FileNotFoundError:
        return []
    rows = []
    for k, f in enumerate(d["frames"]):
        r = L.get(f["timestamp"])
        if r is None:
            continue
        x, y, h = f["ego_status"]["ego_pose"]
        X, Y = r["ego2global_translation"][:2]
        H = yaw_of(r["ego2global_rotation"])
        dx, dy = x - X, y - Y
        c, s = np.cos(H), np.sin(H)
        rows.append(dict(syn=md["scene_token"], log=md["log_name"], k=k, ts=f["timestamp"], real_token=r["token"],
                         dlon=c * dx + s * dy, dlat=-s * dx + c * dy, dyaw=float(np.degrees((h - H + np.pi) % (2 * np.pi) - np.pi)),
                         syn_f0=str(f["camera_dict"]["cam_f0"]["data_path"]), real_f0=r["cams"]["CAM_F0"]["data_path"],
                         v=float(np.hypot(*f["ego_status"]["ego_velocity"]))))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ps = sorted((NAV / "navhard_two_stage/synthetic_scene_pickles").glob("*.pkl"))
    rows = []
    with ProcessPoolExecutor(32) as ex:
        for r in ex.map(one, ps, chunksize=32):
            rows += r
    with open(OUT / "nav_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = np.array([[r["dlon"], r["dlat"], r["dyaw"]] for r in rows])
    dist = np.hypot(a[:, 0], a[:, 1])
    print(len(ps), "scenes", len(rows), "frames")
    for t in (0.05, 0.2, 0.5, 1.0):
        m = (dist < t) & (np.abs(a[:, 2]) < 1)
        print(f"frames within {t} m and 1 deg: {m.sum()}  (k=3: {(m & (np.array([r['k'] for r in rows]) == 3)).sum()})")
