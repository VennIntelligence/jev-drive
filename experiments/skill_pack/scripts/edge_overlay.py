"""Lane EDGE M2 (plan: experiments/skill_pack/plans/2026-10-04-roadedge-diagnosis-plan.md): CAM_F0 overlays of the map
and openpilot's lane lines / road edges, for checking the calibration, the scorer map and the model by eye.

navsim2 env, CPU. Map lines are ground points (ego frame, z = --zg) projected through the CAM_F0 calibration
(jevdrive.navsim_zs.project_nuplan). Model points are drawn twice:
  ray    the model's (x, y, z) taken as a ray from the camera: the pixel where the model put the line;
  metric the model's (x, y) on the ground plane at its metric distance: where the scorer sees it.
Colours: scorer polygon boundary green, + nuPlan generic drivable areas cyan, walkways magenta, lane polygon edges white,
model lane lines yellow (ray) / orange (metric), model road edges red (ray) / pink (metric).
Usage: edge_overlay.py --board navtest --tokens t1,t2,... --out dir   (one PNG per token, 960 x 540).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402
import edge_sections as E  # noqa: E402
from jevdrive.navsim_zs import project_nuplan  # noqa: E402

COL = dict(scorer=(0, 220, 0), ext=(0, 220, 255), walk=(255, 0, 255), lane=(255, 255, 255),
           ll_ray=(255, 230, 0), ll_met=(255, 140, 0), re_ray=(255, 0, 0), re_met=(255, 120, 200))


def _lines_of(geom):
    from shapely.geometry import MultiPolygon, Polygon
    gs = geom.geoms if isinstance(geom, MultiPolygon) or hasattr(geom, "geoms") else [geom]
    out = []
    for g in gs:
        if isinstance(g, Polygon) and not g.is_empty:
            out.append(np.asarray(g.exterior.coords))
            out += [np.asarray(r.coords) for r in g.interiors]
    return out


def _densify(p, step=0.5):
    if len(p) < 2:
        return np.zeros((0, 2))
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.asarray(out)


def draw(board, t, out, scale=2, zg=0.0):
    from shapely.geometry import box
    from shapely.ops import unary_union
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    i = E._G["row"][t]
    e = E._G["idx"][t]
    cam = e["cams"][-1]["CAM_F0"]
    ct = np.asarray(cam["t"], float)
    mc = L.load_cache(E._G["cp"][t])
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    to_ego = lambda p: np.stack([c * (p[:, 0] - o[0]) + s * (p[:, 1] - o[1]), -s * (p[:, 0] - o[0]) + c * (p[:, 1] - o[1])], 1)  # noqa: E731
    g = lambda x, y: (o[0] + c * x - s * y, o[1] + s * x + c * y)  # noqa: E731
    roi = box(*np.r_[np.min([g(0, -40), g(0, 40), g(80, -40), g(80, 40)], 0), np.max([g(0, -40), g(0, 40), g(80, -40), g(80, 40)], 0)])
    am = mc.drivable_area_map
    G = am._geometries
    dac = unary_union([G[k] for k in am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.CARPARK_AREA])]).intersection(roi)
    lanes = [G[k].intersection(roi) for k in am.get_indices_of_map_type([S.LANE, S.LANE_CONNECTOR])]
    raw = E._raw_layers(mc.map_parameters.map_name)
    near = lambda key: [raw[key][1][j] for j in raw[key][0].query(roi)]  # noqa: E731
    ext = unary_union([dac] + near("gda")).intersection(roi)
    walk = unary_union(near("walk")).intersection(roi) if near("walk") else None

    im = Image.open(cam["path"]).convert("RGB")
    W, H = im.size
    im = im.resize((W // scale, H // scale))
    d = ImageDraw.Draw(im)

    def poly(pts_ego, col, w):
        if len(pts_ego) < 2:
            return
        P = np.c_[pts_ego, np.full(len(pts_ego), zg)] - ct
        uv, ok, _ = project_nuplan(P, cam, scale)
        ok &= pts_ego[:, 0] > ct[0] + 2.0
        for a, b, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                d.line([tuple(a), tuple(b)], fill=col, width=w)

    def rays(R, col, w):
        uv, ok, _ = project_nuplan(R, cam, scale)
        for a, b, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                d.line([tuple(a), tuple(b)], fill=col, width=w)

    for gm in lanes:
        for p in _lines_of(gm):
            poly(_densify(to_ego(p)), COL["lane"], 1)
    if walk is not None and not walk.is_empty:
        for p in _lines_of(walk):
            poly(_densify(to_ego(p)), COL["walk"], 2)
    for p in _lines_of(ext):
        poly(_densify(to_ego(p)), COL["ext"], 3)
    for p in _lines_of(dac):
        poly(_densify(to_ego(p)), COL["scorer"], 3)

    h, sl = E._G["heads"][i], E._G["sl"]
    ll, _ = E._mdn(h, sl["lane_lines"], (4, 33, 2))
    llp = 1 / (1 + np.exp(-h[sl["lane_lines_prob"]:sl["lane_lines_prob"] + 8][1::2]))
    re, _ = E._mdn(h, sl["road_edges"], (2, 33, 2))
    m = E.X_IDXS > 2
    for arr, n, cr, cm in ((ll, 4, "ll_ray", "ll_met"), (re, 2, "re_ray", "re_met")):
        for k in range(n):
            if n == 4 and llp[k] < 0.3:
                continue
            R = np.stack([E.X_IDXS, -arr[k, :, 0], -arr[k, :, 1]], 1)[m]          # calib (y right, z down) -> ego axes
            rays(R, COL[cr], 3)
            poly(np.stack([E.X_IDXS + ct[0], ct[1] - arr[k, :, 0]], 1)[m], COL[cm], 2)
    d.rectangle([0, 0, 520, 22], fill=(0, 0, 0))
    d.text((4, 4), f"{board} {e['stage']} {t} {e['map']} lane p={np.round(llp, 2).tolist()}", fill=(255, 255, 255))
    im.save(Path(out) / f"{board}_{t}_zg{zg:+.2f}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=list(E.BOARDS), required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--zg", type=float, default=0.0, help="ground height in the NAVSIM ego (rear-axle) frame")
    a = ap.parse_args()
    E._init(a.board)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for t in a.tokens.split(","):
        draw(a.board, t, a.out, a.scale, a.zg)
