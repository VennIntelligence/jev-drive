"""CARLA counterfactual route pairs, stage 3 step 3: pack the rendered poses into the layout of the op_adapt_H domains and write the route sidecar.

  <root>/samples/route_carla/imgs.npy   (n_poses, 10, 2, 6, 128, 256) uint8 [road, wide] packed model frames, 5 Hz, t0 - 1.8 s ... t0
  <root>/samples/route_carla/tab.npz    pose-level table, the op_adapt_H `Samples` columns (id, split, v0, bin, cluster, tc, img_t, img_valid,
                                        slot_valid) + CARLA columns (town, junction, d, profile, weather, n_exits, lane_off_m, ...)
  <root>/route.npz                      one row per (pose, exit), keyed by `id` = "<pose id>-x<index from left>", the fields of the real-data
                                        sidecars (lib/route_poly.attach reads it) + CARLA columns; `pose_row` indexes imgs.npy / tab.npz

Join: `Samples("route_carla")` with OP_H_ROOT=<root> reads imgs + tab; `route_poly.attach(<root>/route.npz, ids)` gives the polylines of the
exit rows. A trainer samples an exit row, takes `imgs[pose_row]` and `poly` (noise at train time with route_poly.noise_polyline).

  python carla_pairs_pack.py --plan <poses.pkl> --render <render dir> --root <out root> [--drop-frames]
"""
import argparse, json, math, pickle, sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
T5 = np.round(-0.2 * np.arange(9, -1, -1), 3)


def speed_bin(v):
    return np.select([v < 0.5, v < 3.0, v < 8.0], ["stop", "low", "mid"], "high")


def hist_ego(h):
    """History poses in the t0 rear-axle frame (x forward, y left, yaw left+ rad): (10, 3)."""
    x, y, yaw = np.asarray(h["x"]), -np.asarray(h["y"]), -np.radians(h["yaw"])
    c, s = math.cos(yaw[-1]), math.sin(yaw[-1])
    dx, dy = x - x[-1], y - y[-1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, yaw - yaw[-1]], -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--drop-frames", action="store_true", help="delete the per-pose npz after they are packed and verified")
    a = ap.parse_args()
    root = Path(a.root)
    d = root / "samples" / "route_carla"
    d.mkdir(parents=True, exist_ok=True)
    P = [p for p in pickle.load(open(a.plan, "rb")) if (Path(a.render) / "frames" / p["town"] / (p["id"] + ".npz")).exists()]
    n = len(P)
    assert n and len({p["id"] for p in P}) == n
    imgs = np.lib.format.open_memmap(d / "imgs.npy.tmp", mode="w+", dtype=np.uint8, shape=(n, 10, 2, 6, 128, 256))
    for i, p in enumerate(P):
        with np.load(Path(a.render) / "frames" / p["town"] / (p["id"] + ".npz")) as z:
            imgs[i] = z["frames"]
    imgs.flush()
    del imgs
    (d / "imgs.npy.tmp").replace(d / "imgs.npy")
    v0 = np.array([p["v0"] for p in P], np.float32)
    tab = dict(id=np.array([p["id"] for p in P]), split=np.array([p["split"] for p in P]), v0=v0, bin=speed_bin(v0),
               cluster=np.array([p["cluster"] for p in P]), slot_valid=np.ones((n, 9), bool), img_valid=np.ones((n, 10), bool),
               img_t=np.tile(T5, (n, 1)), tc=np.tile(np.array([1.0, 0.0], np.float32), (n, 1)),
               town=np.array([p["town"] for p in P]), junction=np.array([p["junction"] for p in P], np.int32),
               d=np.array([p["d"] for p in P], np.float32), profile=np.array([p["profile"] for p in P]),
               accel=np.array([-p["a"] for p in P], np.float32), weather=np.array([p["weather"] for p in P]),
               n_exits=np.array([p["n_exits"] for p in P], np.int16), lane=np.array([p["lane"] for p in P], np.int16),
               road=np.array([p["road"] for p in P], np.int32), lane_off_m=np.array([p["lane_off_m"] for p in P], np.float32),
               n_lanes=np.array([p["n_lanes"] for p in P], np.int16), pose_hist=np.stack([hist_ego(p["hist"]) for p in P]))
    np.savez(d / "tab.npz", **tab)
    R = [(i, p, e) for i, p in enumerate(P) for e in p["exits"]]
    m = len(R)
    f = lambda k, dt=None: np.array([e[k] for _, _, e in R], dt)  # noqa: E731
    g = lambda k, dt=None: np.array([p[k] for _, p, _ in R], dt)  # noqa: E731
    cls = f("cls")
    route = dict(
        id=np.array(["%s-x%d" % (p["id"], e["index_from_left"]) for _, p, e in R]), split=g("split"), cluster=g("cluster"), scene=g("town"),
        cmd=cls, v0=g("v0", np.float32), poly=np.stack([e["poly"] for _, _, e in R]).astype(np.float32), pmask=np.stack([e["pmask"] for _, _, e in R]),
        plen=f("plen", np.float32), dur=np.full(m, np.nan, np.float32), turn_deg=f("turn_deg", np.float32), turn_s=f("turn_s", np.float32),
        turn_end_s=f("turn_end_s", np.float32), turn_rmin=f("turn_rmin", np.float32), n_turn=f("n_turn", np.int16), in_turn=f("in_turn", bool),
        max_turn_deg=f("max_turn_deg", np.float32), jct_s=g("d", np.float32), turn_junction=np.ones(m, bool), jct_dist=g("d", np.float32),
        n_exit=g("n_exits", np.int16), taken_cls=cls, status=np.full(m, "branch"),
        # CARLA columns
        pose_id=g("id"), pose_row=np.array([i for i, _, _ in R], np.int32), junction=g("junction", np.int32), road=g("road", np.int32),
        lane=g("lane", np.int16), d=g("d", np.float32), profile=g("profile"), weather=g("weather"), index_from_left=f("index_from_left", np.int16),
        angle=f("angle", np.float32), legal=f("legal", bool), lane_change=f("lane_change", bool), exit_road=np.array([e["exit"][0] for _, _, e in R], np.int32),
        exit_dir=np.array([e["exit"][1] for _, _, e in R], np.int8), conn_road=np.array([e["conn"][0] for _, _, e in R], np.int32),
        conn_lane=np.array([e["conn"][1] for _, _, e in R], np.int16), lane_off_m=g("lane_off_m", np.float32), n_lanes=g("n_lanes", np.int16))
    assert len(set(route["id"].tolist())) == m, "duplicate row ids"
    np.savez(root / "route.npz", **route)
    S = dict(poses=n, rows=m, dev_poses=int((tab["split"] == "dev").sum()), towns=dict(Counter(tab["town"].tolist())),
             junctions=len(set(zip(tab["town"].tolist(), tab["junction"].tolist()))), n_exits=dict(Counter(tab["n_exits"].tolist())),
             by_cls=dict(Counter(cls.tolist())), lane_change_rows=int(route["lane_change"].sum()), profile=dict(Counter(tab["profile"].tolist())),
             imgs_gb=round(n * 10 * 2 * 6 * 128 * 256 / 1e9, 2))
    (root / "pack_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S))
    if a.drop_frames:
        for p in P:
            (Path(a.render) / "frames" / p["town"] / (p["id"] + ".npz")).unlink()


if __name__ == "__main__":
    main()
