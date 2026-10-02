"""Camera input audit for navhard (navsim2 env, CPU): how the openpilot road / wide inputs are built from NAVSIM and whether stage-2
synthesized views differ from stage-1 frames; plus side-road / junction context flags for the opposite-side class.

Facts about the build (jevdrive/navsim_zs.py OpenpilotMaps, scripts/navsim_zs_openpilot.py render_token): both model inputs are cut from CAM_F0
only (no side cameras): the virtual cameras (road f = 910 px, wide f = 455 px, 512 x 256, calibration = the NAVSIM ego axes, level and
straight) are filled by projecting their rays through the real CAM_F0 calibration (extrinsic rotation, intrinsics, Brown distortion) with
nearest sampling; the t0 frame's calibration is applied to all four history frames.
Outputs: tables/camera_calibration.csv, camera_summary.json, tables/q1_branch_context.csv.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402

OUT = REPO / "experiments/skill_pack/results/navhard-offroad"


def ypr(R):
    """Camera forward axis (R column 2, in the ego frame) -> yaw (left +), pitch (up +) in degrees and the roll of the x axis."""
    f = R[:, 2]
    x = R[:, 0]
    return float(np.degrees(np.arctan2(f[1], f[0]))), float(np.degrees(np.arcsin(f[2]))), float(np.degrees(np.arcsin(-x[2])))


def main():
    idx = L.index()
    rows = []
    for e in idx:
        cams = [c["CAM_F0"] for c in e["cams"]]
        yp = np.array([ypr(np.asarray(c["R"], float)) for c in cams])
        r = dict(token=e["token"], stage=e["stage"], map=e["map"],
                 yaw=yp[-1, 0], pitch=yp[-1, 1], roll=yp[-1, 2],
                 yaw_range_4f=float(np.ptp(yp[:, 0])), pitch_range_4f=float(np.ptp(yp[:, 1])),
                 t_x=float(cams[-1]["t"][0]), t_y=float(cams[-1]["t"][1]), t_z=float(cams[-1]["t"][2]),
                 fx=float(cams[-1]["K"][0, 0]), fy=float(cams[-1]["K"][1, 1]), cx=float(cams[-1]["K"][0, 2]), cy=float(cams[-1]["K"][1, 2]),
                 k1=float(cams[-1]["D"][0]), k2=float(cams[-1]["D"][1]),
                 K_range_4f=float(np.ptp([c["K"][0, 0] for c in cams])), D_range_4f=float(np.ptp([c["D"][0] for c in cams])),
                 t_range_4f=float(np.ptp([c["t"][2] for c in cams])))
        rows.append(r)
    C = pd.DataFrame(rows)
    # image sizes and the virtual cameras' coverage by CAM_F0, on a sample
    rng = np.random.default_rng(0)
    samp = {st: rng.choice([i for i, e in enumerate(idx) if e["stage"] == st], 150, replace=False) for st in ("one", "two")}
    size, cov = {}, []
    for st, ii in samp.items():
        for i in ii:
            e = idx[i]
            size.setdefault(st, {})
            wh = Image.open(e["cams"][-1]["CAM_F0"]["path"]).size
            size[st][str(wh)] = size[st].get(str(wh), 0) + 1
            m = Z.OpenpilotMaps(e["cams"][-1]["CAM_F0"])
            cov.append(dict(stage=st, cov_road=m.coverage[0], cov_wide=m.coverage[1]))
    cov = pd.DataFrame(cov)
    summ = {"image_sizes_sample150": size, "coverage_by_stage": {f"{a}_{b}": v for (a, b), v in cov.groupby("stage").agg(["mean", "min"]).round(4).to_dict().items() for v in [{k: x for k, x in v.items()}]}}
    g = C.groupby("stage")
    cols = ["yaw", "pitch", "roll", "yaw_range_4f", "pitch_range_4f", "t_x", "t_y", "t_z", "fx", "cx", "cy", "k1", "k2", "K_range_4f", "D_range_4f", "t_range_4f"]
    tab = pd.concat({st: d[cols].describe().T[["mean", "std", "min", "50%", "max"]] for st, d in g}, axis=1).round(4)
    tab.to_csv(OUT / "tables/camera_calibration.csv")
    f = C.fx.median()
    summ["fov_deg"] = dict(cam_f0_horizontal=2 * np.degrees(np.arctan(960 / f)), road_input=2 * np.degrees(np.arctan(256 / 910)), wide_input=2 * np.degrees(np.arctan(256 / 455)),
                           road_vertical=2 * np.degrees(np.arctan(128 / 910)), wide_vertical=2 * np.degrees(np.arctan(128 / 455)))
    summ["n_distinct_calibrations"] = {st: int(d[["yaw", "pitch", "roll", "fx", "cx", "cy", "k1"]].round(5).drop_duplicates().shape[0]) for st, d in g}
    (OUT / "camera_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    print(tab.to_string())

    # branch / junction context for each stage-2 token: intersection polygon in the forward fan, drivable width at x = 20 m
    import shapely
    from shapely.geometry import Polygon
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    from shapely.ops import unary_union
    cp = L.cache_paths()
    ctx = []
    for e in idx:
        mc = L.load_cache(cp[e["token"]])
        am = mc.drivable_area_map
        o = np.array(mc.ego_state.rear_axle.serialize())
        c, s = np.cos(o[2]), np.sin(o[2])
        g_ = lambda x, y: (o[0] + c * x - s * y, o[1] + s * x + c * y)  # noqa: E731
        fan = Polygon([g_(3, 0), g_(40, 40 * np.tan(np.radians(35))), g_(40, -40 * np.tan(np.radians(35)))])
        inter = am.get_indices_of_map_type([S.INTERSECTION])
        conn = am.get_indices_of_map_type([S.LANE_CONNECTOR])
        area = am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA])
        ia = [k for k in inter if am._geometries[k].intersects(fan)]
        d_inter = min([float(np.hypot(*(np.array(shapely.ops.nearest_points(am._geometries[k], shapely.Point(o[0], o[1]))[0].coords[0]) - o[:2]))) for k in ia] or [np.nan])
        n_conn_fan = sum(am._geometries[k].intersects(fan) for k in conn)
        u = unary_union([am._geometries[k] for k in area])
        from shapely.geometry import LineString
        seg = u.intersection(LineString([g_(20, -40), g_(20, 40)]))
        w20 = float(seg.length) if not seg.is_empty else 0.0
        ctx.append(dict(token=e["token"], inter_ahead=bool(ia), inter_dist=d_inter, n_conn_in_fan=int(n_conn_fan), width20=w20))
    X = pd.DataFrame(ctx)
    tab2 = pd.read_pickle(L.OUT / "token_table.pkl").merge(X, on="token")
    tab2.to_pickle(L.OUT / "branch_context.pkl")
    out = []
    for m in ("native", "n4"):
        opp = (tab2[f"{m}_end_y"] * tab2.ref_end_y < 0) & (tab2[f"{m}_end_y"].abs() > 1) & (tab2.ref_end_y.abs() > 1) & ((tab2[f"{m}_end_y"] - tab2.ref_end_y).abs() > 2)
        tab2[f"{m}_opp"] = opp
        for st in ("one", "two"):
            for nm, mk in (("intersection in forward fan", tab2.inter_ahead), ("no intersection ahead", ~tab2.inter_ahead),
                           ("lane connectors in fan >= 1", tab2.n_conn_in_fan >= 1), ("none", tab2.n_conn_in_fan == 0),
                           ("drivable width at 20 m > 20 m", tab2.width20 > 20), ("width <= 20 m", tab2.width20 <= 20)):
                gg = tab2[(tab2.stage == st) & mk]
                out.append(dict(model=m, stage=st, subset=nm, n=len(gg), opp_rate=float(gg[f"{m}_opp"].mean()), dac_fail=float((gg[f"{m}_dac"] == 0).mean())))
    pd.DataFrame(out).round(4).to_csv(OUT / "tables/q1_branch_context.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
