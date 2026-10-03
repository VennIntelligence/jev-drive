"""Class M eye-check sheet (plans/2026-10-04-navhard-dac-gated-comp-prereg.md, section 4).

Seeded random sample (seed 0, without replacement) of shipped-model DAC failures whose primary class is M (every off-polygon corner lies on
nuPlan generic_drivable_areas / carpark_areas). One panel per token: nuPlan CAM_F0 at t0 with the scorer's drivable polygon (yellow),
the raw generic drivable area (cyan), the shipped plan (blue) and the worst off-polygon corner at the first exit (red ring);
and a BEV (scorer polygon grey, generic drivable area cyan, plan blue, ego box at the first exit red, route lanes pale green).

  navsim2 env, CPU:  nhdac_msheet.py [--n 40] [--per-page 4]   -> $DATA_DIR/runs/leaderboard_audit/navhard_dac/msheet/{p01.jpg ..., sample.csv}
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402
from jevdrive.navsim_zs import project_nuplan  # noqa: E402

DD = L.D / "runs/leaderboard_audit/navhard_dac"
OUT = DD / "msheet"
FONT = ImageFont.truetype(str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 14)
CW, CH, BW, BH, HD = 600, 338, 220, 338, 40
YEL, CYAN, BLUE, RED = (255, 230, 0), (0, 220, 230), (40, 140, 255), (255, 40, 40)
_W = {}


def init():
    import glob
    sim, _, _, _, samp = L.setup_scoring()
    _W.update(sim=sim, maps={}, cp={Path(p).parent.name: p for p in glob.glob(str(L.MCACHE / "*/*/*/metric_cache.pkl"))},
              P=L.poses_by_token(L.D / "runs/op_lb/lb_navhard/preds/gimm-cinque__base.npz"), idx={e["token"]: e for e in L.index()})


def gda(name):
    if name not in _W["maps"]:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        a = get_maps_api(str(L.D / "datasets/navsim/maps"), "nuplan-maps-v1.0", name)
        g = []
        for ln in ("generic_drivable_areas", "carpark_areas"):
            g += [x for x in a._load_vector_map_layer(ln).geometry.values if x is not None and not x.is_empty]
        _W["maps"][name] = g
    return _W["maps"][name]


def panel(task):
    k, t, first = task
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    from shapely.geometry import box as sbox
    from shapely.ops import unary_union
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    mc, e = L.load_cache(_W["cp"][t]), _W["idx"][t]
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    to_ego = lambda p: np.stack([c * (p[:, 0] - o[0]) + s * (p[:, 1] - o[1]), -s * (p[:, 0] - o[0]) + c * (p[:, 1] - o[1])], 1)  # noqa: E731
    am = mc.drivable_area_map
    area = am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA])
    G = am._geometries
    win = sbox(o[0] - 60, o[1] - 60, o[0] + 60, o[1] + 60)
    da = unary_union([G[i] for i in area if G[i].intersects(win)])
    gd = unary_union([g for g in gda(mc.map_parameters.map_name) if g.intersects(win)])
    route = [G[i] for i in am.get_indices_of_map_type([S.LANE, S.LANE_CONNECTOR]) if am.tokens[i] in set(mc.route_lane_ids) and G[i].intersects(win)]
    rings = lambda g: [to_ego(np.asarray(r.coords)) for p in (getattr(g, "geoms", None) or [g]) if not p.is_empty for r in [p.exterior, *p.interiors]]  # noqa: E731
    st = L.simulate(_W["sim"], mc, _W["P"][t])
    cor = state_array_to_coords_array(st[None], mc.ego_state.car_footprint.vehicle_parameters)[0]       # (41, 5, 2)
    ins = am.points_in_polygons(cor[None, :, :-1, :])[area].any(axis=0)[0]
    fi = int(np.flatnonzero(~ins.all(axis=1))[0])
    d = shapely.distance(da, shapely.points(cor[fi, :4]))
    worst = int(np.argmax(np.where(~ins[fi], d, -1)))
    wp, ovs = to_ego(cor[fi, worst][None])[0], float(d.max())
    plan = L.dense_from_poses(_W["P"][t])
    camf = e["cams"][-1]["CAM_F0"]
    ct = np.asarray(camf["t"], float)

    def dens(p, step=0.5):
        seg = [p[0]]
        for a_, b_ in zip(p[:-1], p[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b_ - a_) / step)))
            seg += [a_ + (b_ - a_) * j / n for j in range(1, n + 1)]
        return np.asarray(seg)

    def pc(pts):
        uv, ok, _ = project_nuplan(np.c_[pts, np.zeros(len(pts))] - ct, camf, 1)
        return uv * (CW / 1920), ok & (pts[:, 0] > ct[0] + 2.0)

    def line(dr, uv, ok, col, w):
        for a_, b_, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                dr.line([tuple(a_), tuple(b_)], fill=col, width=w)

    im = Image.new("RGB", (CW + BW, HD + CH), (25, 25, 25))
    cam = Image.open(camf["path"]).convert("RGB").resize((CW, CH))
    cd = ImageDraw.Draw(cam)
    for r in rings(gd):
        line(cd, *pc(dens(r)), CYAN, 2)
    for r in rings(da):
        line(cd, *pc(dens(r)), YEL, 3)
    line(cd, *pc(plan[:, :2]), BLUE, 3)
    uv, ok = pc(wp[None])
    if ok[0]:
        x, y = uv[0]
        cd.ellipse([x - 9, y - 9, x + 9, y + 9], outline=RED, width=3)
    im.paste(cam, (0, HD))
    bev = Image.new("RGB", (BW, BH), (250, 250, 250))
    b = ImageDraw.Draw(bev)
    sx = BH / 45.0                                        # 45 m tall, x from -5 m; 220 px = 24.4 m wide
    tp = lambda q: [(BW / 2 - y * sx, BH - (x + 5) * sx) for x, y in q[:, :2]]  # noqa: E731
    for r in rings(gd):
        b.polygon(tp(r), fill=(190, 240, 245))
    for r in rings(da):
        b.polygon(tp(r), fill=(150, 150, 150))
    for g in route:
        for r in rings(g):
            b.polygon(tp(r), fill=(200, 235, 205))
    for r in rings(da):
        b.line(tp(r), fill=(200, 160, 0), width=2)
    b.line(tp(plan), fill=BLUE, width=3)
    b.polygon(tp(to_ego(cor[fi, :4])), outline=RED, width=3)
    x, y = tp(wp[None])[0]
    b.ellipse([x - 6, y - 6, x + 6, y + 6], outline=RED, width=2)
    b.text((4, 3), "BEV: grey = scorer polygon, cyan = generic drivable area", fill=(0, 0, 0), font=ImageFont.truetype(FONT.path, 9))
    im.paste(bev, (CW, HD))
    dr = ImageDraw.Draw(im)
    dr.text((6, 2), f"#{k:02d}  {t}  stage {e['stage']}", fill=(255, 255, 255), font=FONT)
    dr.text((6, 20), f"first exit {fi / 10:.1f} s, max {ovs:.2f} m outside the polygon (red ring)", fill=(210, 210, 210), font=FONT)
    return k, im, dict(k=k, token=t, first=fi / 10, over=ovs, stage=e["stage"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per-page", type=int, default=4)
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()
    d = pd.read_csv(DD / "nhdac_tokens.csv")
    d = d[(d.arm == "native") & (d.primary == "map_narrow")]
    toks = sorted(d.token)
    pick = np.random.default_rng(0).choice(len(toks), a.n, replace=False)
    tasks = [(i + 1, toks[j], 0) for i, j in enumerate(pick)]
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(min(a.procs, len(tasks)), initializer=init) as pool:
        res = sorted(pool.map(panel, tasks), key=lambda r: r[0])
    pd.DataFrame([r[2] for r in res]).to_csv(OUT / "sample.csv", index=False)
    pw, ph = CW + BW, HD + CH
    n = a.per_page
    for p in range(0, len(res), n):
        pg = Image.new("RGB", (2 * pw, ((n + 1) // 2) * ph), (255, 255, 255))
        for j, (_, im, _) in enumerate(res[p:p + n]):
            pg.paste(im, ((j % 2) * pw, (j // 2) * ph))
        pg.save(OUT / f"p{p // n + 1:02d}.jpg", quality=80)
    print(f"{len(d)} native M tokens, sampled {len(res)}, pages {-(-len(res) // n)}")


if __name__ == "__main__":
    main()
