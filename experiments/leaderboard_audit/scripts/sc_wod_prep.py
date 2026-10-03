"""Scale check, WOD perception frames: unpack the 10 sceneflow TFRecords (v1 Frame protos, 10 Hz, with laser labels) into one
npz per segment: the front-three camera JPEGs, calibration, vehicle pose, ego velocity, vehicle-frame 3D boxes, and the
map features of frame 0 (lane centres / road lines / edges as polylines in the world frame).
env p3-wodprep (CPU): $DATA_DIR/envs/p3-wodprep/bin/python experiments/leaderboard_audit/scripts/sc_wod_prep.py"""
import os, pickle, struct, sys
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
SRC, DST = D / "datasets/waymo_perception/sceneflow", D / "runs/scale_check/wod_sf"
DST.mkdir(parents=True, exist_ok=True)
from waymo_open_dataset import dataset_pb2 as dp  # noqa: E402


def records(path):
    with open(path, "rb") as f:
        while h := f.read(12):
            n = struct.unpack("<Q", h[:8])[0]
            p = f.read(n)
            f.read(4)
            yield p


def poly(feat, kind):
    if kind == "lane":
        return [(p.x, p.y, p.z) for p in feat.lane.polyline]
    if kind == "line":
        return [(p.x, p.y, p.z) for p in feat.road_line.polyline]
    return [(p.x, p.y, p.z) for p in feat.road_edge.polyline]


for path in sorted(SRC.glob("*.tfrecord")):
    seg = path.name.split("segment-")[1].split("_with")[0]
    out = DST / f"{seg}.pkl"
    if out.exists():
        continue
    fr = []
    cal = None
    mp = None
    for k, rec in enumerate(records(path)):
        F = dp.Frame.FromString(rec)
        if cal is None:
            cal = {c.name: dict(intrinsic=np.array(c.intrinsic, np.float64), extrinsic=np.array(c.extrinsic.transform).reshape(4, 4),
                                width=c.width, height=c.height) for c in F.context.camera_calibrations if c.name in (1, 2, 3)}
        if mp is None and len(F.map_features):
            mp = []
            for f in F.map_features:
                w = f.WhichOneof("feature_data")
                if w == "lane":
                    mp.append(("lane", f.id, poly(f, "lane"), [(b.boundary_feature_id, b.lane_start_index, b.lane_end_index) for b in f.lane.left_boundaries],
                               [(b.boundary_feature_id, b.lane_start_index, b.lane_end_index) for b in f.lane.right_boundaries],
                               [n.feature_id for n in f.lane.left_neighbors], [n.feature_id for n in f.lane.right_neighbors]))
                elif w == "road_line":
                    mp.append(("line", f.id, poly(f, "line"), int(f.road_line.type)))
                elif w == "road_edge":
                    mp.append(("edge", f.id, poly(f, "edge"), int(f.road_edge.type)))
        im = {i.name: i for i in F.images}
        boxes = np.array([[l.box.center_x, l.box.center_y, l.box.center_z, l.box.length, l.box.width, l.box.height, l.box.heading,
                           l.type, l.metadata.speed_x, l.metadata.speed_y, l.num_lidar_points_in_box, hash(l.id) % (1 << 30)]
                          for l in F.laser_labels], np.float64).reshape(-1, 12)
        fr.append(dict(t=F.timestamp_micros, pose=np.array(F.pose.transform).reshape(4, 4),
                       vel=(im[1].velocity.v_x, im[1].velocity.v_y, im[1].velocity.w_z), jpg={c: im[c].image for c in (1, 2, 3)},
                       pose_cam={c: np.array(im[c].pose.transform).reshape(4, 4) for c in (1,)}, boxes=boxes))
    pickle.dump(dict(seg=seg, cal=cal, map=mp, frames=fr), open(out.with_suffix(".tmp"), "wb"))
    os.replace(out.with_suffix(".tmp"), out)
    print(seg, len(fr), "frames", None if mp is None else len(mp), "map feats", flush=True)
