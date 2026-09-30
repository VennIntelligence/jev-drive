"""CARLA static scene geometry near the routes, for S_jev's NC (prereg v4 amendment). CARLA env (Python 3.8), talks to a
running server (scripts/op_adapt_carla_objects.sh starts one).

  python scripts/op_adapt_carla_objects.py --port P --points-dir DIR --out-dir DIR [--towns T ...]

Per town: load the world, walk the spectator over the route points (1 km cells, so a large map streams every tile the
routes touch), and keep every environment object whose bounding box centre lies within 60 m of a route point.
Output <out>/<town>.npz: id, label (carla.CityObjectLabel int), name, centre (n, 3), extent (n, 3) half sizes,
yaw / pitch / roll (deg), CARLA world frame.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import carla

CELL = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--points-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--towns", nargs="*")
    ap.add_argument("--radius", type=float, default=60.0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = carla.Client("127.0.0.1", a.port)
    c.set_timeout(600.0)
    for f in sorted(Path(a.points_dir).glob("*.npy")):
        town = f.stem
        if a.towns and town not in a.towns:
            continue
        if (out / f"{town}.npz").exists():
            continue
        t0 = time.time()
        w = c.load_world(town)
        pts = np.load(f).astype(float)
        objs = {}
        cells = np.unique(np.floor(pts / CELL), axis=0)
        sp = w.get_spectator()
        for cx, cy in cells:
            m = (np.floor(pts[:, 0] / CELL) == cx) & (np.floor(pts[:, 1] / CELL) == cy)
            ctr = pts[m].mean(0)
            sp.set_transform(carla.Transform(carla.Location(float(ctr[0]), float(ctr[1]), 50.0)))
            for _ in range(20):
                w.wait_for_tick(60.0)
            for o in w.get_environment_objects(carla.CityObjectLabel.Any):
                if o.id in objs:
                    continue
                bb = o.bounding_box
                objs[o.id] = (int(o.type), o.name, (bb.location.x, bb.location.y, bb.location.z), (bb.extent.x, bb.extent.y, bb.extent.z),
                              (bb.rotation.yaw, bb.rotation.pitch, bb.rotation.roll))
        ids = np.array(list(objs), np.uint64)
        cen = np.array([v[2] for v in objs.values()], float).reshape(-1, 3)
        # keep objects near a route point (coarse grid test, then exact distance)
        keep = np.zeros(len(cen), bool)
        from collections import defaultdict
        grid = defaultdict(list)
        for p in pts:
            grid[(int(p[0] // 50), int(p[1] // 50))].append(p)
        grid = {k: np.array(v) for k, v in grid.items()}
        ext = np.array([v[3] for v in objs.values()], float).reshape(-1, 3)
        for i, (x, y, _) in enumerate(cen):
            r = a.radius + float(np.hypot(ext[i, 0], ext[i, 1]))
            gi, gj = int(x // 50), int(y // 50)
            k = int(np.ceil(r / 50))
            near = [grid[(gi + di, gj + dj)] for di in range(-k, k + 1) for dj in range(-k, k + 1) if (gi + di, gj + dj) in grid]
            keep[i] = bool(near) and bool((np.hypot(*(np.vstack(near) - (x, y)).T) <= r).any())
        v = list(objs.values())
        sel = np.flatnonzero(keep)
        np.savez_compressed(out / f"{town}.npz", id=ids[sel], label=np.array([v[i][0] for i in sel], np.int32),
                            name=np.array([v[i][1] for i in sel]), centre=cen[sel], extent=ext[sel],
                            rot=np.array([v[i][4] for i in sel], float).reshape(-1, 3))
        print(town, "objects", len(objs), "kept", int(keep.sum()), "cells", len(cells), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
