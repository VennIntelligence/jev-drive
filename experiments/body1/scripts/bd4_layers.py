"""BODY1 arm 4.3, item C check (prereg Amendment 4 note vi, Amendment 5): which nuPlan layers are the road areas of AlpaSim's offroad scorer?

The scorer (alpasim src/eval/src/eval/scorers/offroad.py) calls the ego on the road when one lane polygon, or the union of the lanes within 6 m,
contains its box; otherwise, on nuPlan maps (no road-edge layer), when the union of the RoadArea elements covers it. trajdata's nuPlan
conversion (dataset_specific/nuplan/nuplan_utils.py) builds RoadArea from the devkit's `drivable_area` = road_segments + intersections +
generic_drivable_areas + carpark_areas. Whether the public scenes' shipped maps were built that way is checked here on the maps themselves:
every road-area polygon of a scene map (c1_extract.py map, rollout-local frame) is matched to the nuPlan gpkg polygons of the same city by
area and vertex count (both invariant under the rigid local frame). Map polygons only; no scene, rollout or score is read; analysis only.

  $DATA_DIR/envs/navsim2/bin/python experiments/body1/scripts/bd4_layers.py [--sets zero-s0 clean-s0]     (CPU)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

CITY = {"las_vegas": "us-nv-las-vegas-strip", "boston": "us-ma-boston", "pittsburgh": "us-pa-pittsburgh-hazelwood", "singapore": "sg-one-north"}
GPKG = ("road_segments", "intersections", "generic_drivable_areas", "carpark_areas", "lane_groups_polygons", "lane_group_connectors", "lanes_polygons",
        "gen_lane_connectors_scaled_width_polygons")
OUT = B.REPO / "experiments/body1/results/loss"


def main(a):
    import shapely
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(data_dir() / "datasets/navsim/maps"))
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_db
    db = get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"])
    with Run("body1", "scorer-layers", config=vars(a)) as run:
        ref = {}

        def city(loc):
            name = next(v for k, v in CITY.items() if k in loc.lower() or loc == v)
            if name not in ref:
                ref[name] = {}
                for ly in GPKG:
                    g = db.load_vector_layer(name, ly).geometry
                    P = [p for x in g for p in (x.geoms if hasattr(x, "geoms") else [x]) if p is not None and p.geom_type == "Polygon"]
                    ref[name][ly] = np.array([[p.area, len(p.exterior.coords)] for p in P]) if P else np.zeros((0, 2))
            return ref[name]
        cnt, area, n, tot = {ly: 0 for ly in GPKG} | {"none": 0}, {ly: 0.0 for ly in GPKG} | {"none": 0.0}, 0, 0.0
        scenes = 0
        for st in a.sets:
            mp = pickle.load(open(data_dir() / "runs/body1/diag/x" / st / "map.pkl", "rb"))
            for s, m in mp.items():
                R = city(m["location"])
                scenes += 1
                for ar in m["areas"]:
                    if len(ar["ext"]) < 3:
                        continue
                    p = shapely.Polygon(ar["ext"].astype(np.float64))
                    A, nv = p.area, len(p.exterior.coords)
                    hit = [ly for ly in GPKG if len(R[ly]) and ((np.abs(R[ly][:, 0] - A) <= a.tol * max(A, 1.0)) & (np.abs(R[ly][:, 1] - nv) <= 1)).any()]
                    n += 1
                    tot += A
                    for ly in hit or ["none"]:
                        cnt[ly] += 1
                        area[ly] += A
        res = dict(sets=a.sets, scenes=scenes, road_areas=n, matched_by_layer=cnt, area_share_by_layer={k: v / max(tot, 1e-9) for k, v in area.items()}, tol=a.tol,
                   code="trajdata nuplan_utils.populate_vector_map: RoadArea <- NuPlanMap._vector_map['drivable_area'] = road_segments + intersections + "
                        "generic_drivable_areas + carpark_areas (devkit _initialize_drivable_area)")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "scorer_layers.json").write_text(json.dumps(res, indent=1) + "\n")
        run.info(json.dumps(res, indent=1))
        run.summary.update(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["zero-s0", "clean-s0"])
    ap.add_argument("--tol", type=float, default=2e-4, help="relative area tolerance of a match (scene maps are float32 in a local frame)")
    main(ap.parse_args())
