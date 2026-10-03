"""Loss-budget examples (tmp/lbx), NAV lane step 2: one GIF per picked token (lbx_nav_pick.py).

Each frame has four panels:
  top left   the original nuPlan CAM_F0 (t0 keyframe; during the history phase the latest keyframe at or before t), with the
             scorer's drivable-area boundary (yellow), route lane polygons (thin white), the reference (green; navhard PDM-Closed,
             navtest human), the shipped plan (blue) and the best driver's plan (orange), a dot per trajectory at time t;
  top right  BEV (ego frame at t0, x up): drivable area (grey), route lanes (pale green), agents at time t (navhard: the scorer's reactive
             traffic simulated against the shipped plan; navtest: log replay), red if they overlap the shipped ego box, the three simulated ego boxes (LQR + bicycle, the scorer's own
             replay) at t; the shipped box turns red while a corner is off the drivable area;
  bottom     openpilot's own input tensors (road 910 px focal, wide 455 px), decoded from the packed YUV420 frames the
             shipped run used: the four keyframes from the navsim_zs frames cache and the six GIMM frames (op_lb gimm.npy),
             with the same map / trajectory overlays and the shipped model's road edges (red) and lane lines (pink, p > 0.3), 3-60 m ahead.
Phase 1 steps through the 1.5 s input history (model frames at their own times), phase 2 steps the 4 s plan at 0.25 s.

  navsim2 env, CPU:  lbx_nav_render.py navhard|navtest [--procs 16] [--only token,...]
Output: $DATA_DIR/runs/leaderboard_audit/loss_budget/lbx_media/<board>_<class>_<token>.gif
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402
from jevdrive.navsim_zs import project_nuplan  # noqa: E402

LB = L.D / "runs/leaderboard_audit/loss_budget"
OUT = LB / "lbx_media"
BEST = "gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base.npz"
SRC = {"navhard": dict(data="lb_navhard", split="navhard_two_stage", idx=L.IDX, mc=L.MCACHE),
       "navtest": dict(data="lb_navtest", split="navtest", idx=L.D / "runs/navsim_zs/index/navtest_slim.pkl",
                       mc=L.D / "runs/navsim/metric_cache/v1_navtest")}
K_OP = {"road": np.array([[910.0, 0, 256.0], [0, 910.0, 47.6], [0, 0, 1]]),
        "wide": np.array([[455.0, 0, 256.0], [0, 455.0, 151.8], [0, 0, 1]])}
X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
T_KEY = np.array([-1.5, -1.0, -0.5, 0.0])
C = dict(ref=(40, 210, 90), native=(40, 140, 255), best=(255, 150, 0), da=(255, 230, 0), lane=(235, 235, 235),
         edge=(255, 40, 40), ll=(255, 120, 200), agent=(120, 130, 150), hit=(230, 30, 30))
NAMES = dict(road_edge_wide="off-road, road edge too wide", early_clip="early clip", wrong_direction="wrong direction",
             collisions="collision (NC or TTC)", stopped_slow="stopped / too slow", no_command="turn missed, no command")
FONT = ImageFont.truetype(str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 15)
FONT_S = ImageFont.truetype(str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 12)
CW, CH, BW, MW, MH, HDR = 880, 495, 400, 640, 320, 44
_W = {}


def yuv_rgb(p):
    """Packed (6, 128, 256) uint8 -> RGB (256, 512, 3), JPEG YCbCr full range (jevdrive.op_interp.to_rgb in numpy)."""
    Y = np.empty((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = p[0], p[1], p[2], p[3]
    U = np.repeat(np.repeat(p[4].astype(np.float32), 2, 0), 2, 1) - 128
    V = np.repeat(np.repeat(p[5].astype(np.float32), 2, 0), 2, 1) - 128
    rgb = np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def init(board):
    sim, _, policy, _, samp = L.setup_scoring()
    s = SRC[board]
    mt = json.loads((L.D / "runs/op_lb" / s["data"] / "meta.json").read_text())
    import glob
    hz = np.load(L.D / "runs/op_lb" / s["data"] / "plans/gimm@cinque.npz")
    _W.update(board=board, sim=sim, samp=samp, policy=policy, row={t: i for i, t in enumerate(mt["names"])}, kix=mt["index"], cam=mt["cam"],
              syn_t=np.round(mt["syn_t"], 3),
              keys=np.load(L.D / "runs/navsim_zs/openpilot" / s["split"] / "frames.npy", mmap_mode="r"),
              gimm=np.load(L.D / "runs/op_lb" / s["data"] / "gimm.npy", mmap_mode="r"),
              heads=hz["heads"], hrow={t: i for i, t in enumerate(hz["names"].tolist())}, sl=json.loads(str(hz["info"]))["heads_slices"],
              idx={e["token"]: e for e in pickle.load(open(s["idx"], "rb"))},
              cp={Path(p).parent.name: p for p in glob.glob(str(s["mc"] / "*/*/*/metric_cache.pkl"))},
              cp2={Path(p).parent.name: p for p in glob.glob(str(L.D / "runs/navsim/metric_cache/v2_navtest/*/*/*/metric_cache.pkl"))} if board == "navtest" else {},
              P={a: L.poses_by_token(L.D / "runs/op_lb" / s["data"] / "preds" / f) for a, f in (("native", "gimm-cinque__base.npz"), ("best", BEST))})


def mdn(h, base, shape):
    return h[base:base + int(np.prod(shape))].reshape(shape)


def model_frames(t):
    """{time: packed (2, 6, 128, 256)} of the 10 frames the shipped run fed (4 keyframes + 6 GIMM)."""
    i = _W["row"][t]
    k = np.asarray(_W["keys"][_W["kix"][i]])
    g = np.asarray(_W["gimm"][i])
    fr = {float(tk): (k[j], "key") for j, tk in enumerate(T_KEY)}
    fr.update({float(ts): (g[j], "GIMM") for j, ts in enumerate(_W["syn_t"])})
    return dict(sorted(fr.items()))


def render(pick):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    W, board, t = _W, _W["board"], pick["token"]
    e, mc = W["idx"][t], L.load_cache(W["cp"][t])
    o = np.array(mc.ego_state.rear_axle.serialize())
    c, s = np.cos(o[2]), np.sin(o[2])
    to_ego = lambda p: np.stack([c * (p[:, 0] - o[0]) + s * (p[:, 1] - o[1]), -s * (p[:, 0] - o[0]) + c * (p[:, 1] - o[1])], 1)  # noqa: E731

    # trajectories (dense 0.1 s, ego frame) and the scorer's replay of each
    if board == "navhard":
        ref = L.pdm_ref_ego(mc, W["samp"])
    else:
        ref = L.dense_from_poses(np.asarray(L.load_cache(W["cp2"][t]).human_trajectory.poses, np.float64))   # v1 cache has no human
    plans = dict(ref=ref, native=L.dense_from_poses(W["P"]["native"][t]), best=L.dense_from_poses(W["P"]["best"][t]))
    am = mc.drivable_area_map
    area = am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA])
    box, inside, sts = {}, {}, {}
    for a, d in plans.items():
        st = sts[a] = L.simulate(W["sim"], mc, L.poses_from_dense(d))
        cor = state_array_to_coords_array(st[None], mc.ego_state.car_footprint.vehicle_parameters)[0]      # (41, 5, 2) global
        inside[a] = am.points_in_polygons(cor[None, :, :-1, :])[area].any(axis=0)[0].all(-1)
        box[a] = np.stack([to_ego(q) for q in cor])

    # map (ego frame)
    G = am._geometries
    ego_poly = mc.ego_state.car_footprint.oriented_box.geometry
    near = lambda ids: [G[k] for k in ids if G[k].distance(ego_poly) < 80]  # noqa: E731
    da = unary_union(near(area))
    lanes = near(am.get_indices_of_map_type([S.LANE, S.LANE_CONNECTOR]))
    route = [G[k] for k in am.get_indices_of_map_type([S.LANE, S.LANE_CONNECTOR]) if am.tokens[k] in set(mc.route_lane_ids) and G[k].distance(ego_poly) < 80]
    rings = lambda g: [to_ego(np.asarray(r.coords)) for p in (getattr(g, "geoms", None) or [g]) if not p.is_empty for r in [p.exterior, *p.interiors]]  # noqa: E731
    da_r, lane_r, route_r = rings(da), [r for g in lanes for r in rings(g)], [r for g in route for r in rings(g)]

    if board == "navhard":      # stage-2 scoring uses reactive traffic: the agents as simulated against the shipped plan
        tracks = W["policy"].simulate_environment(sts["native"], mc)
        objs = [[(o.track_token, o.box.geometry) for o in tr.tracked_objects.tracked_objects] for tr in tracks]
    else:                       # navtest v1: non-reactive log replay
        objs = [list(zip(mc.observation[ti]._tokens, mc.observation[ti]._geometries)) for ti in range(41)]

    def agents(ti):
        out = []
        for tok, g in objs[ti]:
            q = to_ego(np.asarray(g.exterior.coords))
            hit = Polygon(box["native"][ti][:4]).intersects(Polygon(q)) if "red_light" not in tok else False
            out.append((q, "red_light" in tok, hit))
        return out

    # model view data
    h = W["heads"][W["hrow"][t]]
    re = mdn(h, W["sl"]["road_edges"], (2, 33, 2))
    ll = mdn(h, W["sl"]["lane_lines"], (4, 33, 2))
    llp = 1 / (1 + np.exp(-h[W["sl"]["lane_lines_prob"]:W["sl"]["lane_lines_prob"] + 8][1::2]))
    camp = np.asarray(W["cam"][W["row"][t]], float)
    camf = e["cams"][-1]["CAM_F0"]
    ct = np.asarray(camf["t"], float)
    frames = model_frames(t)

    def dens(p, step=0.5):
        if len(p) < 2:
            return p
        seg = [p[0]]
        for a_, b_ in zip(p[:-1], p[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b_ - a_) / step)))
            seg += [a_ + (b_ - a_) * k / n for k in range(1, n + 1)]
        return np.asarray(seg)

    def proj_cam(pts):          # ego ground points (n, 2) -> CAM_F0 pixels at the panel size
        P = np.c_[pts, np.zeros(len(pts))] - ct
        uv, ok, _ = project_nuplan(P, camf, 1)
        return uv * (CW / 1920), ok & (pts[:, 0] > ct[0] + 2.0)

    def proj_op(dev, view):     # device-frame points (n, 3) x fwd, y right, z down -> model-panel pixels
        x = dev[:, 0]
        ok = x > 1.0
        xs = np.where(ok, x, 1.0)
        K = K_OP[view]
        u, v = K[0, 0] * dev[:, 1] / xs + K[0, 2], K[1, 1] * dev[:, 2] / xs + K[1, 2]
        ok &= (u > -500) & (u < 1012) & (v > -500) & (v < 756)
        return np.stack([u, v], 1) * (MW / 512), ok

    ego2dev = lambda pts: np.stack([pts[:, 0] - camp[0], -(pts[:, 1] - camp[1]), np.full(len(pts), camp[2])], 1)  # noqa: E731

    def line(d, uv, ok, col, w):
        for a_, b_, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                d.line([tuple(a_), tuple(b_)], fill=col, width=w)

    def overlay(d, proj, ti, plan_phase, model=None):
        for r in route_r:
            line(d, *proj(dens(r)), C["lane"], 1)
        for r in da_r:
            line(d, *proj(dens(r)), C["da"], 3)
        if model:
            for k in range(4):
                if llp[k] > 0.3:
                    line(d, *model(np.stack([X_IDXS, ll[k, :, 0], ll[k, :, 1]], 1)[(X_IDXS > 3) & (X_IDXS < 60)]), C["ll"], 2)
            for k in range(2):
                line(d, *model(np.stack([X_IDXS, re[k, :, 0], re[k, :, 1]], 1)[(X_IDXS > 3) & (X_IDXS < 60)]), C["edge"], 3)
        if plan_phase:
            for a in ("ref", "best", "native"):
                uv, ok = proj(plans[a][:, :2])
                line(d, uv, ok, C[a], 4 if a == "native" else 3)
                if ok[ti]:
                    x_, y_ = uv[ti]
                    d.ellipse([x_ - 6, y_ - 6, x_ + 6, y_ + 6], fill=C[a], outline=(0, 0, 0))

    # static images
    cam_imgs = {tk: Image.open(cc["CAM_F0"]["path"]).convert("RGB").resize((CW, CH)) for tk, cc in zip(T_KEY, e["cams"])}
    op_imgs = {tm: [Image.fromarray(yuv_rgb(p[v])).resize((MW, MH), Image.NEAREST) for v in (0, 1)] for tm, (p, _) in frames.items()}

    sc = pick["scores"]
    def terms(a):
        x = sc[a]
        return f"{x['score']:.2f} (NC {x['no_at_fault_collisions']:.0f} DAC {x['drivable_area_compliance']:.0f} DDC {x['driving_direction_compliance']:.1f} " \
               f"EP {x['ego_progress']:.2f} TTC {x['time_to_collision_within_bound']:.0f})"
    hdr1 = f"{board} stage {e['stage']} | {t} | class: {NAMES[pick['cls']]} | cmd {pick['cmd']}, v0 {pick['v0']:.1f} m/s"
    hdr2 = f"token score  shipped {terms('native')}   best {sc['best']['score']:.2f}   {'PDM ref' if board == 'navhard' else 'human'} {sc['ref']['score']:.2f}"

    def frame(phase, tm, ti):
        im = Image.new("RGB", (CW + BW, HDR + CH + MH), (20, 20, 20))
        d = ImageDraw.Draw(im)
        d.text((6, 3), hdr1, fill=(255, 255, 255), font=FONT)
        d.text((6, 23), hdr2, fill=(210, 210, 210), font=FONT_S)
        kt = T_KEY[T_KEY <= tm + 1e-6][-1] if phase == 0 else 0.0
        ci = cam_imgs[float(kt)].copy()
        overlay(ImageDraw.Draw(ci), proj_cam, ti, phase == 1)
        im.paste(ci, (0, HDR))
        lab = f"nuPlan CAM_F0 keyframe t = {kt:+.1f} s" + ("" if phase == 0 else f"   plan t = {tm:.2f} s (1.7x speed)")
        d.rectangle([0, HDR, 8 + d.textlength(lab, font=FONT), HDR + 20], fill=(0, 0, 0))
        d.text((4, HDR + 2), lab, fill=(255, 255, 255), font=FONT)
        # BEV
        bev = Image.new("RGB", (BW, CH), (250, 250, 250))
        b = ImageDraw.Draw(bev)
        sx = BW / 50.0                                    # 50 m wide, x from -12 m
        tp = lambda q: [(BW / 2 - y * sx, CH - (x + 12) * sx) for x, y in q[:, :2]]  # noqa: E731
        for r in da_r:
            b.polygon(tp(r), fill=(215, 215, 215))
        for r in route_r:
            b.polygon(tp(r), fill=(200, 235, 205), outline=(170, 200, 175))
        for r in lane_r:
            b.line(tp(r), fill=(170, 170, 170), width=1)
        for q, red, hit in agents(ti):
            b.polygon(tp(q), fill=(255, 120, 120) if red else C["hit"] if hit else C["agent"])
        for a in ("ref", "best", "native"):
            b.line(tp(plans[a]), fill=C[a], width=2)
        for a in ("ref", "best", "native"):
            q = box[a][ti][:4]
            b.polygon(tp(q), outline=C["hit"] if (a == "native" and not inside[a][ti]) else C[a], width=4 if a == "native" else 2)
        b.text((4, 4), (f"BEV, t = {tm:+.2f} s" if phase else "BEV at t0 (input phase)") + "; agents " + ("reactive (vs shipped plan)" if board == "navhard" else "log replay"), fill=(0, 0, 0), font=FONT_S)
        y0 = 22
        for a, n in (("native", "shipped plan"), ("best", "best (it_dw3 + selector)"), ("ref", "PDM reference" if board == "navhard" else "human")):
            b.rectangle([6, y0 + 3, 20, y0 + 11], fill=C[a]); b.text((26, y0), n, fill=(0, 0, 0), font=FONT_S); y0 += 16
        b.text((6, y0), "red box: shipped ego off drivable area", fill=C["hit"], font=FONT_S); y0 += 16
        b.text((6, y0), "red agent: overlaps shipped ego", fill=C["hit"], font=FONT_S)
        im.paste(bev, (CW, HDR))
        # model view
        mt = max(k for k in op_imgs if k <= tm + 1e-6) if phase == 0 else 0.0
        for v, view in enumerate(("road", "wide")):
            oi = op_imgs[mt][v].copy()
            od = ImageDraw.Draw(oi)
            overlay(od, lambda pts, view=view: proj_op(ego2dev(pts), view), ti, phase == 1,
                    model=(lambda dev, view=view: proj_op(dev, view)) if phase == 1 else None)
            im.paste(oi, (v * MW, HDR + CH))
            lab = f"openpilot input, {view} ({frames[mt][1]} frame t = {mt:+.1f} s)" + (", model road edges red, lane lines pink" if phase == 1 else "")
            d.rectangle([v * MW, HDR + CH, v * MW + 8 + d.textlength(lab, font=FONT_S), HDR + CH + 17], fill=(0, 0, 0))
            d.text((v * MW + 4, HDR + CH + 2), lab, fill=(255, 255, 255), font=FONT_S)
        return im

    seq = [(0, tm, 0) for tm in frames] + [(1, ti / 10, ti) for ti in range(0, 41, 2)]   # history, then the plan at 0.2 s
    ims = [frame(*q).resize((1120, 748), Image.LANCZOS) for q in seq]      # keeps each GIF under ~3 MB
    dur = [160] * len(frames) + [125] * 20 + [2500]
    dur[len(frames) - 1] = 900
    keep = list(C.values()) + [(255, 255, 255), (0, 0, 0), (20, 20, 20), (210, 210, 210), (250, 250, 250), (215, 215, 215), (200, 235, 205),
                               (170, 170, 170), (255, 120, 120), (40, 40, 40)]
    mosaic = Image.new("RGB", (ims[0].width, 2 * ims[0].height))
    mosaic.paste(ims[0], (0, 0)); mosaic.paste(ims[-1], (0, ims[0].height))
    base = mosaic.quantize(colors=256 - len(keep), method=Image.Quantize.MEDIANCUT).getpalette()[:3 * (256 - len(keep))]
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for col in keep for v in col] + base)        # overlay colours first: exact matches survive
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{board}_{pick['cls']}_{t}.gif"
    q[0].save(f, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True)
    ims[-1].convert("RGB").save(OUT / f"{board}_{pick['cls']}_{t}_end.jpg", quality=82)
    return str(f), f.stat().st_size / 1e6, dict(native_off=[float(x / 10) for x in np.where(~inside["native"])[0][:1]],
                                                 hits=sorted({ti / 10 for ti in range(41) if any(h_ for _, _, h_ in agents(ti))})[:1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("board", choices=list(SRC))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    picks = json.loads((LB / f"lbx_{a.board}.json").read_text())["picks"]
    if a.only:
        picks = [p for p in picks if p["token"] in a.only.split(",")]
    with mp.get_context("fork").Pool(min(a.procs, len(picks)), initializer=init, initargs=(a.board,)) as pool:
        for p, r in zip(picks, pool.imap(render, picks)):
            print(p["cls"], p["token"], r, flush=True)
