"""OpenDRIVE lane samples of one CARLA town near given route points, for the S_jev drivable / lane-direction raster
(experiments/op_adapt_r2/archive/op_adapt_score_data.py `maps-carla`). Runs in the CARLA env (Python 3.8, carla 0.9.15 client, no server:
carla.Map is built offline from the .xodr).

  python experiments/op_adapt_r2/archive/op_adapt_carla_lanes.py <town> <xodr> <points.npy> <out.npz> [--radius 60]

points.npy: (n, 2) CARLA world x, y of the routes. Every Driving waypoint (0.5 m apart) within `radius` of a route point
is kept together with its lateral neighbours of type Driving / Bidirectional / Parking (walking left and right across
the road, through non-drivable lanes). Output columns (CARLA frame, yaw in degrees): x, y, yaw, width, type (1 driving,
2 bidirectional, 3 parking), road, section, lane, junction (-1 none).
"""
import argparse

import numpy as np

import carla

TYPES = {carla.LaneType.Driving: 1, carla.LaneType.Bidirectional: 2, carla.LaneType.Parking: 3}
CELL = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("town")
    ap.add_argument("xodr")
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=0.5)
    a = ap.parse_args()
    m = carla.Map(a.town, open(a.xodr).read())
    pts = np.load(a.points)
    r = int(np.ceil(a.radius / CELL))
    cells = {(int(i), int(j)) for i, j in np.floor(pts / CELL).astype(int)}
    near = {(i + di, j + dj) for i, j in cells for di in range(-r, r + 1) for dj in range(-r, r + 1)}
    rows, seen = [], set()

    def add(w):
        t = TYPES.get(w.lane_type)
        if t is None:
            return
        key = (w.road_id, w.section_id, w.lane_id, round(w.s, 2))
        if key in seen:
            return
        seen.add(key)
        loc = w.transform.location
        rows.append((loc.x, loc.y, w.transform.rotation.yaw, w.lane_width, t, w.road_id, w.section_id, w.lane_id,
                     w.junction_id if w.is_junction else -1))

    for w in m.generate_waypoints(a.step):
        loc = w.transform.location
        if (int(np.floor(loc.x / CELL)), int(np.floor(loc.y / CELL))) not in near:
            continue
        add(w)
        for side in ("get_left_lane", "get_right_lane"):
            q = w
            for _ in range(8):
                q = getattr(q, side)()
                if q is None:
                    break
                add(q)
    z = np.array(rows, dtype=np.float64).reshape(-1, 9)
    np.savez_compressed(a.out, x=z[:, 0], y=z[:, 1], yaw=z[:, 2], width=z[:, 3], type=z[:, 4].astype(np.int8),
                        road=z[:, 5].astype(np.int32), section=z[:, 6].astype(np.int32), lane=z[:, 7].astype(np.int32),
                        junction=z[:, 8].astype(np.int32), step=a.step)
    print(a.town, len(z), "samples")


if __name__ == "__main__":
    main()
