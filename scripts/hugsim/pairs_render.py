"""I3 renderer: HUGSIM 3DGS scenes along the LOGGED ego trajectory, with and without one inserted actor
(todos/2026-09-25-reactivity-program/i3-hugsim-pairs.md; geometry, actors and labels in jevdrive/hugsim_pairs.py).

Runs in the HUGSIM venv from the HUGSIM repo root (third-party code imported as shipped). Per scene it loads the
scene Gaussians once, builds each world's actor track, checks it (road coverage, scene obstacles), renders the
three front cameras at 5 Hz for every valid world and writes JPEGs, frames.jsonl per world and meta.json.
The camera poses are HUGSIM's env poses (sim/hugsim_env/envs/hug_sim.py: ego = rt2pose((0, th, 0), (a,
ground_height(a, b), b)), camera = ego @ v2front @ inv(v2c) @ cam_rect) with (a, b, th) from the log.

  cd $DATA_DIR/third_party/HUGSIM && CUDA_VISIBLE_DEVICES=1 $DATA_DIR/envs/hugsim/bin/python \
      $DATA_DIR/../jev-drive/scripts/hugsim/pairs_render.py [--keys k1 k2 ...] [--validate]

--validate adds the checks of the 5-scene validation: x- rendered twice (determinism) and against HUGSimEnv's own
_get_obs at the same poses, changed pixels outside the actor's projected box, occlusion consistency against an
actor-only render, and saves the figure frames.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.getcwd())          # HUGSIM repo root
sys.path.insert(0, str(REPO))
from omegaconf import OmegaConf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from scene.obj_model import ObjModel  # noqa: E402
from sim.utils.sim_utils import dense_cam_poses, fov2focal, load_camera_cfg, rt2pose  # noqa: E402

from jevdrive import hugsim_pairs as H  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

REALCAR = H.data_dir() / "datasets" / "hugsim" / "3DRealCar"


class Scene:
    def __init__(self, ds: str, scene: str):
        self.ds, self.dir = ds, H.unpack(ds, scene)
        cfg = OmegaConf.load(self.dir / "cfg.yaml")
        self.g = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
        params, _ = torch.load(self.dir / "scene.pth", weights_only=False)
        self.g.restore(params, None)
        self.sh_degree = cfg.model.sh_degree
        self.bg = torch.tensor([1, 1, 1] if cfg.model.white_background else [0, 0, 0], dtype=torch.float32,
                               device="cuda")
        cam_params, _, rect = load_camera_cfg(OmegaConf.load(f"configs/sim/{ds}_camera.yaml"))
        v2front = cam_params["CAM_FRONT"]["v2c"]
        self.cams = {}
        for c in H.CAMS:
            p = cam_params[c]
            K = np.eye(4)
            K[0, 0] = fov2focal(p["intrinsic"]["fovx"], p["intrinsic"]["W"])
            K[1, 1] = fov2focal(p["intrinsic"]["fovy"], p["intrinsic"]["H"])
            K[0, 2], K[1, 2] = p["intrinsic"]["cx"], p["intrinsic"]["cy"]
            self.cams[c] = (K, v2front @ np.linalg.inv(p["v2c"]) @ rect, p["intrinsic"]["W"], p["intrinsic"]["H"])
        import pickle
        with open(self.dir / "ground_param.pkl", "rb") as f:
            cp, ch, cmds = pickle.load(f)
        self.dense, _ = dense_cam_poses(cp, cmds)
        self.cam_height = ch
        with torch.no_grad():
            sem_full = torch.argmax(self.g.get_full_3D_features, dim=-1)
            road = self.g.get_full_xyz[sem_full == 0].cpu().numpy()
            sem = torch.argmax(self.g.get_3D_features, dim=-1)
            obst = self.g.get_xyz[(sem > 1) & (sem != 10) & (self.g.get_opacity[:, 0] > 0.8)].cpu().numpy()
        road = road[np.isfinite(road).all(1)]
        obst = obst[np.isfinite(obst).all(1)]
        self.road_xz, self.road_y = road[:, [0, 2]], road[:, 1]
        self.road_tree = cKDTree(self.road_xz)
        self.pc = _FrozenPC(self.g)
        self.obst = obst
        self.obst_tree = cKDTree(obst[:, [0, 2]])
        self.dyn = {}

    # HUGSimEnv.ground_height (the env's ego height)
    def env_height(self, u, v):
        d = (self.dense[:, 0, 3] - u) ** 2 + (self.dense[:, 2, 3] - v) ** 2
        c2w = self.dense[np.argmin(d)]
        w2c = np.linalg.inv(c2w)
        loc = w2c[:3, :3] @ np.array([u, 0, v]) + w2c[:3, 3]
        loc[1] = 0
        return (c2w[:3, :3] @ loc + c2w[:3, 3])[1]

    def c2ws(self, a, b, th):
        ego = rt2pose(np.array([0.0, th, 0.0]), np.array([a, self.env_height(a, b), b]))
        return {c: ego @ self.cams[c][1] for c in H.CAMS}

    def road_height(self, a, b, r=1.0):
        idx = self.road_tree.query_ball_point([a, b], r)
        return (float(np.median(self.road_y[idx])), len(idx)) if len(idx) >= 20 else (None, len(idx))

    def load_actor(self, asset: str):
        if asset not in self.dyn:
            m = ObjModel(self.sh_degree, feat_mutable=False)
            params, _ = torch.load(REALCAR / asset / "gs.pth", weights_only=False)
            m.restore(list(params), None)
            xyz = m.get_xyz.detach().cpu().numpy()
            lo, hi = np.percentile(xyz, 0.1, axis=0) - 0.2, np.percentile(xyz, 99.9, axis=0) + 0.2
            wlh = json.loads((REALCAR / asset / "wlh.json").read_text())
            self.dyn[asset] = (m, lo, hi, wlh)
        return self.dyn[asset]

    def render(self, cam: str, c2w: np.ndarray, actor=None, pc=None):
        """One view as HUGSimEnv._get_obs renders it; actor = (asset, b2w 4x4) or None. Returns the render package.
        Each Camera gets its own `dynamics` dict: Camera's default is a shared mutable {} that render() writes the
        planned actors into, so a camera made without one would carry the last x+ actor into every later x-."""
        K, _, W, Hh = self.cams[cam]
        view = Camera(K=K, c2w=c2w, width=W, height=Hh, image=np.zeros((Hh, W, 3)), image_name="", dynamics={})
        planning = [{}, {}]
        dyn = {}
        if actor is not None:
            dyn = {"actor": self.dyn[actor[0]][0]}
            planning = [{"actor": torch.tensor(actor[1]).float().cuda()}, {}]
        with torch.no_grad():
            return render(viewpoint=view, prev_viewpoint=None, pc=pc or self.pc, dynamic_gaussians=dyn, unicycles={},
                          bg_color=self.bg, planning=planning)


def to_rgb(pkg):
    return (torch.permute(pkg["render"].clamp(0, 1), (1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8)


def proj_box(S: Scene, cam, c2w, asset, M, margin=12):
    """Pixel box (u0, v0, u1, v1) of the actor's Gaussian extent; None when behind the camera, the whole image
    when it straddles the image plane."""
    K, _, W, Hh = S.cams[cam]
    _, lo, hi, _ = S.dyn[asset]
    cx = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pc = (np.linalg.inv(c2w) @ M @ cx.T).T[:, :3]
    if (pc[:, 2] < 0.1).all():
        return None
    if (pc[:, 2] < 0.1).any():
        return (0, 0, W, Hh)
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return (int(max(u.min() - margin, 0)), int(max(v.min() - margin, 0)), int(min(u.max() + margin, W)),
            int(min(v.max() + margin, Hh)))


def check_track(S: Scene, track, t_sim, conflict, asset, t_end):
    """Road coverage of the footprint and scene obstacles inside the box, over t_sim <= min(first conflict, t_end)."""
    _, lo, hi, (w, l, h) = S.dyn[asset]
    first = t_sim[np.flatnonzero(conflict)[0]] if conflict.any() else np.inf
    use = np.flatnonzero(t_sim <= min(first, t_end) + 1e-6)
    gx, gz = np.meshgrid(np.linspace(-l / 2, l / 2, 5), np.linspace(-w / 2, w / 2, 3))
    cover, obst = [], []
    for i in use:
        a, b, th = track[i, :3]
        f, lf = H.fwd_of(th), H.left_of(th)
        pts = np.array([a, b]) + gx.reshape(-1, 1) * f + gz.reshape(-1, 1) * lf
        dist, _ = S.road_tree.query(pts)
        cover.append(float((dist < H.ROAD_R).mean()))
        idx = S.obst_tree.query_ball_point([a, b], np.hypot(l, w) / 2)
        if idx:
            d = S.obst[idx][:, [0, 2]] - np.array([a, b])
            along, across = d @ f, d @ lf
            y0 = track[i, 4]
            inside = (np.abs(along) < l / 2) & (np.abs(across) < w / 2) & (S.obst[idx][:, 1] < y0) & \
                     (S.obst[idx][:, 1] > y0 - h)
            obst.append(int(inside.sum()))
        else:
            obst.append(0)
    cover, obst = np.array(cover), np.array(obst)
    ok = cover.mean() >= H.ROAD_MEAN and cover.min() >= H.ROAD_MIN and obst.max() <= H.OBST_PTS
    return ok, {"road_mean": round(float(cover.mean()), 3), "road_min": round(float(cover.min()), 3),
                "obst_max": int(obst.max())}


def build_worlds(S: Scene, L: H.Logged, key: str, tc: float, t_render, t_sim, assets):
    """Actor tracks (with ground height as column 4), conflicts and validity per world."""
    worlds = {"minus": {"rendered": True}}
    t_end = t_render[-1]
    cut_side = None
    for fam in ("static", "cutin", "oncoming", "null"):
        asset = H.asset_for(key, fam, assets)
        _, _, _, (w, l, h) = S.load_actor(asset)
        sides = (1.0, -1.0) if fam == "cutin" or (fam == "null" and cut_side is None) else \
            (cut_side,) if fam == "null" else (0.0,)
        best = None
        for side in sides:
            if side is None:
                break
            tr = H.actor_track(fam, side, tc, t_sim, L, l)
            ys = []
            for a, b in tr[:, :2]:
                y, _ = S.road_height(a, b)
                ys.append(np.nan if y is None else y)
            ys = np.array(ys)
            env_y = np.array([S.env_height(a, b) + S.cam_height for a, b in tr[:, :2]])
            y = np.where(np.isnan(ys), env_y, ys)
            tr = np.concatenate([tr, y[:, None]], 1)
            conf = H.conflicts(L, t_sim, tr, (w, l))
            ok, info = check_track(S, tr, t_sim, conf, asset, t_end)
            info.update(side=side, asset=asset, wlh=[w, l, h], ground_fallback=int(np.isnan(ys).sum()),
                        y_minus_env=round(float(np.nanmedian(ys - env_y)), 3) if (~np.isnan(ys)).any() else None)
            cand = (ok, info, tr, conf)
            if best is None or (ok and not best[0]) or (ok == best[0] and info["road_mean"] > best[1]["road_mean"]):
                best = cand
            if ok and fam not in ("cutin", "null"):
                break
        if best is None:
            worlds[fam] = {"rendered": False, "reason": "no_side"}
            continue
        ok, info, tr, conf = best
        if fam == "cutin":
            cut_side = info["side"] if ok else None
        reason = "ok" if ok else ("offroad" if info["road_mean"] < H.ROAD_MEAN or info["road_min"] < H.ROAD_MIN
                                  else "obstacle")
        if fam == "null" and conf.any():
            ok, reason = False, "null_conflict"
        worlds[fam] = {"rendered": ok, "reason": reason, **info, "track": np.round(tr, 4).tolist(),
                       "conflict": conf.astype(int).tolist(),
                       "t_conflict": float(t_sim[np.flatnonzero(conf)[0]]) if conf.any() else None}
    return worlds


def render_scene(S: Scene, L: H.Logged, row, out: Path, validate: bool, pool, rl):
    key, tc = row["key"], row["t_c"]
    t_render, t_sim = H.window(tc, L)
    assets = H.scenario_assets()
    t0 = time.perf_counter()
    worlds = build_worlds(S, L, key, tc, t_render, t_sim, assets)
    t_tracks = time.perf_counter() - t0
    ea, eb, eth = L.pose_at(t_render)
    poses = [S.c2ws(ea[k], eb[k], eth[k]) for k in range(len(t_render))]
    k_sim = np.searchsorted(t_sim, t_render - 1e-6)
    stats = {w: {"vis_px": [], "any_px": [], "out_px": [], "out_px_thr": []} for w in worlds}
    minus = {}
    futs = []
    t_gpu = 0.0
    val = {"det_max": 0, "env_max": None, "occ": {}}
    fig = {}
    for w, m in worlds.items():
        if not m["rendered"]:
            continue
        d = out / w
        for c in H.CAM_KEYS:
            (d / "cams" / c).mkdir(parents=True, exist_ok=True)
        lines = []
        for k in range(len(t_render)):
            actor = None
            if w != "minus":
                a, b, th, _, y = m["track"][k_sim[k]]
                actor = (m["asset"], H.b2w(a, b, th, y))
            vis, anyp, outp, outt = [], [], [], []
            files = {}
            for c, ck in zip(H.CAMS, H.CAM_KEYS):
                g0 = time.perf_counter()
                pkg = S.render(c, poses[k][c], actor)
                rgb = to_rgb(pkg)
                t_gpu += time.perf_counter() - g0
                if w == "minus":
                    minus[(k, c)] = rgb
                    if validate:
                        rgb2 = to_rgb(S.render(c, poses[k][c], None, pc=S.g))
                        val["det_max"] = max(val["det_max"], int(np.abs(rgb2.astype(int) - rgb).max()))
                    vis.append(0), anyp.append(0), outp.append(0), outt.append(0)
                else:
                    diff = np.abs(rgb.astype(np.int16) - minus[(k, c)]).max(-1)
                    ch = diff > H.DIFF_THR
                    box = proj_box(S, c, poses[k][c], m["asset"], actor[1])
                    inb = np.zeros_like(ch)
                    if box is not None:
                        inb[box[1]:box[3], box[0]:box[2]] = True
                    vis.append(int(ch.sum())), anyp.append(int((diff > 0).sum()))
                    outp.append(int((diff > 0)[~inb].sum())), outt.append(int(ch[~inb].sum()))
                    if validate and c == "CAM_FRONT":
                        occ = occlusion(S, c, poses[k][c], actor, diff)
                        acc = val["occ"].setdefault(w, np.zeros(4, np.int64))
                        acc += occ
                        fig.setdefault(w, []).append((k, rgb, minus[(k, c)]))
                f = f"cams/{ck}/{4 * k:07d}.jpg"
                files[ck] = f
                futs.append(pool.submit(cv2.imwrite, str(d / f), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 95]))
            stats[w]["vis_px"].append(vis), stats[w]["any_px"].append(anyp), stats[w]["out_px"].append(outp)
            stats[w]["out_px_thr"].append(outt)
            lines.append(json.dumps({"frame": 4 * k, "t": float(t_render[k]), "files": files}))
        (d / "frames.jsonl").write_text("\n".join(lines) + "\n")
    for f in futs:
        assert f.result(), "jpeg write failed"
    for w in worlds:
        worlds[w].update(stats[w]) if worlds[w]["rendered"] else None
    if validate:
        val["env_max"] = env_check(S, row, poses, minus)
        val["occ"] = {w: v.tolist() for w, v in val["occ"].items()}
        save_fig_frames(out, fig, t_render, tc)
        val["probe"] = occlusion_probe(S, L, tc, out, H.asset_for(key, "static", assets))
    meta = {"key": key, "dataset": row["dataset"], "scene": row["scene"], "t_c": tc, "t_render": t_render.tolist(),
            "t_sim": t_sim.tolist(), "worlds": worlds, "cams": {c: {"K": S.cams[c][0].tolist(),
                                                                       "c2front": S.cams[c][1].tolist()}
                                                                   for c in H.CAMS},
            "timing": {"tracks_s": round(t_tracks, 2), "gpu_s": round(t_gpu, 2)}, "validation": val}
    (out / "meta.json").write_text(json.dumps(meta, default=_js))
    return meta


def _js(o):
    return o.tolist() if isinstance(o, np.ndarray) else o.item()


def occlusion(S, cam, c2w, actor, diff):
    """Occlusion consistency of one x+ view against an actor-only render (same renderer, empty scene):
    actor pixels (alpha > 0.5) where an x- scene object (semantic class > 1, not sky) is > 1 m in front of the
    actor must stay unchanged, and pixels where the x- surface is > 1 m behind it must change. Ground classes
    (road 0, sidewalk 1) are not occluders: at grazing angles their alpha-weighted depth is not a surface depth,
    and the car stands on them. Returns [occluded, occluded_changed, clear, clear_changed]."""
    empty = _EmptyPC.of(S.g)
    pa = S.render(cam, c2w, actor, pc=empty)
    pm = S.render(cam, c2w, None)
    alpha = pa["alphas"][0, ..., 0].cpu().numpy()
    da = pa["depth"][0].cpu().numpy()
    dm = pm["depth"][0].cpu().numpy()
    sem = torch.argmax(pm["feats"], dim=0).cpu().numpy()
    on = alpha > 0.5
    occ = on & (dm < da - 1.0) & (sem > 1) & (sem != 10)
    clr = on & (dm > da + 1.0)
    chg = diff > H.DIFF_THR
    return np.array([occ.sum(), (occ & chg).sum(), clr.sum(), (clr & chg).sum()], np.int64)


class _FrozenPC:
    """The scene's Gaussians with their activations evaluated once. GaussianModel's get_full_* properties re-run
    the activations and re-concatenate the ground model on every access, i.e. on every rendered view; render()
    reads exactly these six tensors (plus the appearance model when `affine`), so a frozen copy renders the same
    pixels (checked bit-exact, see the sub-doc) at 44 instead of 68 ms per 800x450 view (4.7 M Gaussians)."""

    def __init__(self, g):
        full = g.ground_model is not None
        src = [g.get_full_xyz, g.get_full_opacity, g.get_full_scaling, g.get_full_rotation, g.get_full_features,
               g.get_full_3D_features] if full else [g.get_xyz, g.get_opacity, g.get_scaling, g.get_rotation,
                                                     g.get_features, g.get_3D_features]
        with torch.no_grad():
            (self.get_xyz, self.get_opacity, self.get_scaling, self.get_rotation, self.get_features,
             self.get_3D_features) = [x.detach() for x in src]
        self.ground_model, self.affine, self.active_sh_degree = None, g.affine, g.active_sh_degree
        if g.affine:
            self.pos_enc, self.dir_enc, self.appearance_model = g.pos_enc, g.dir_enc, g.appearance_model


class _EmptyPC:
    """A Gaussian model with no Gaussians (render() concatenates the actor onto it): the actor-only render."""
    _cache = {}

    @classmethod
    def of(cls, g):
        if id(g) not in cls._cache:
            e = cls()
            full = g.ground_model is not None
            src = [g.get_full_xyz, g.get_full_opacity, g.get_full_scaling, g.get_full_rotation, g.get_full_features,
                   g.get_full_3D_features] if full else [g.get_xyz, g.get_opacity, g.get_scaling, g.get_rotation,
                                                         g.get_features, g.get_3D_features]
            (e.get_xyz, e.get_opacity, e.get_scaling, e.get_rotation, e.get_features,
             e.get_3D_features) = [x[:0].detach() for x in src]
            e.ground_model, e.affine, e.active_sh_degree = None, False, g.active_sh_degree
            cls._cache[id(g)] = e
        return cls._cache[id(g)]


def env_check(S: Scene, row, poses, minus, n=4):
    """Max |x- - HUGSimEnv._get_obs()| over n frames: our renders against the shipped env at the same ego state."""
    import tempfile
    from sim.hugsim_env.envs.hug_sim import HUGSimEnv
    ds = row["dataset"]
    scen = OmegaConf.create({"plan_list": [], "load_HD_map": False, "start_euler": [0, 0, 0], "start_ab": [0, 0],
                             "start_velo": 0, "start_steer": 0, "scene_name": row["scene"], "mode": "i3"})
    cfg = OmegaConf.merge({"scenario": scen}, {"base": {"realcar_path": str(REALCAR)}},
                          {"camera": OmegaConf.load(f"configs/sim/{ds}_camera.yaml")},
                          {"kinematic": OmegaConf.load("configs/sim/kinematic.yaml")})
    cfg.update(OmegaConf.load(S.dir / "cfg.yaml"))
    cfg.model_path = str(S.dir)
    with tempfile.TemporaryDirectory() as tmp:
        env = HUGSimEnv(cfg, tmp)
        env.render_kwargs["planning"] = [{}, {}]
        L = H.Logged(S.dir, ds)
        tr, _ = H.window(row["t_c"], L)
        worst = 0
        for k in np.linspace(0, len(tr) - 1, n).astype(int):
            a, b, th = [x[0] for x in L.pose_at(np.array([tr[k]]))]
            env.vab, env.vr = np.array([a, b]), np.array([0.0, th, 0.0])
            obs = env._get_obs()
            for c in H.CAMS:
                worst = max(worst, int(np.abs(obs["rgb"][c].astype(int) - minus[(k, c)]).max()))
        del env
    torch.cuda.empty_cache()
    return worst


def save_fig_frames(out, fig, t_render, tc, dts=(-3.0, -2.0, -1.0)):
    """For the figure: every actor world's front-camera x+ and x- at t_c + dt (nearest render frame)."""
    keep = {}
    for w, fr in fig.items():
        by_k = {f[0]: f for f in fr}
        for dt in dts:
            k = int(np.argmin(np.abs(t_render - (tc + dt))))
            if k in by_k:
                keep[f"{w}_{dt:+.0f}_plus"], keep[f"{w}_{dt:+.0f}_minus"] = by_k[k][1], by_k[k][2]
    np.savez_compressed(out / "fig_frames.npz", **keep)


def occlusion_probe(S: Scene, L: H.Logged, tc, out, asset, t_off=-4.0):
    """A parked car put behind scene objects on purpose: at t_c + t_off, a stopped car 15-35 m ahead on the logged
    path, 4.5-9 m to either side (only placements whose box holds <= OBST_PTS obstacle Gaussians, i.e. not inside
    a scene object). Occlusion counts (see `occlusion`) for every placement; the one with the most occluded actor
    pixels is saved for the figure."""
    _, _, _, (w, l, h) = S.load_actor(asset)
    t = tc + t_off
    a0, b0, th0 = [x[0] for x in L.pose_at(np.array([t]))]
    c2w = S.c2ws(a0, b0, th0)["CAM_FRONT"]
    pm = S.render("CAM_FRONT", c2w, None)
    rm = to_rgb(pm)
    rows, best = [], None
    for ahead in (15.0, 20.0, 25.0, 30.0, 35.0):
        for d in (4.5, -4.5, 6.0, -6.0, 7.5, -7.5, 9.0, -9.0):
            a, b, th = [float(x) for x in L.path(np.array([L.s_at(t) + ahead]), np.array([d]))]
            y, _ = S.road_height(a, b, r=2.0)
            y = S.env_height(a, b) + S.cam_height if y is None else y
            tr = np.array([[a, b, th, 0.0, y]])
            _, info = check_track(S, tr, np.array([0.0]), np.array([False]), asset, 1.0)
            if info["obst_max"] > H.OBST_PTS:
                continue
            M = H.b2w(a, b, th, y)
            pp = S.render("CAM_FRONT", c2w, (asset, M))
            rp = to_rgb(pp)
            diff = np.abs(rp.astype(np.int16) - rm).max(-1)
            occ = occlusion(S, "CAM_FRONT", c2w, (asset, M), diff)
            rows.append({"ahead": ahead, "lateral": d, "occ": int(occ[0]), "occ_changed": int(occ[1]),
                         "clear": int(occ[2]), "clear_changed": int(occ[3])})
            if best is None or occ[0] > best[0]:
                best = (occ[0], rp)
    if best is not None and best[0] > 0:
        np.savez_compressed(out / "probe_frames.npz", plus=best[1], minus=rm)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--tag", default="render")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"), help="scenes i::n")
    a = ap.parse_args()
    import pandas as pd
    rl = RunLog("i3-hugsim-pairs", a.tag)
    sc = pd.read_csv(H.root() / "scenes.csv")
    sc = sc[sc.t_c.notna()]
    if a.keys:
        sc = sc.set_index("key").loc[a.keys].reset_index()
    sc = sc.iloc[a.shard[0]::a.shard[1]]
    rl.log.info("%d scenes, validate=%s", len(sc), a.validate)
    from tqdm import tqdm
    with ThreadPoolExecutor(3) as pool:
        for _, row in tqdm(list(sc.iterrows()), desc=f"shard {a.shard[0]}/{a.shard[1]}", unit="scene"):
            out = H.root("scenes", row["key"])
            if a.skip_done and (out / "meta.json").exists():
                continue
            t0 = time.perf_counter()
            S = Scene(row["dataset"], row["scene"])
            t_load = time.perf_counter() - t0
            L = H.Logged(S.dir, row["dataset"])
            meta = render_scene(S, L, row, out, a.validate, pool, rl)
            wall = time.perf_counter() - t0
            ws = {w: {k: m.get(k) for k in ("reason", "side", "road_mean", "road_min", "obst_max", "t_conflict",
                                             "y_minus_env", "ground_fallback")} | {"vis_frames": int(
                      (np.array(m.get("vis_px", [[0]])).sum(1) >= H.VIS_PX).sum()), "out_px": int(np.array(
                      m.get("out_px", [[0]])).sum())} for w, m in meta["worlds"].items()}
            rl.log.info("%s: %.1f s (load %.1f, tracks %.1f, render %.1f), peak %.2f GB; worlds %s; val %s",
                        row["key"], wall, t_load, meta["timing"]["tracks_s"], meta["timing"]["gpu_s"],
                        torch.cuda.max_memory_allocated() / 1e9, ws, meta["validation"] if a.validate else "-")
            rl.event("scene_done", key=row["key"], wall_s=round(wall, 2), load_s=round(t_load, 2), worlds=ws,
                     **meta["timing"], validation=meta["validation"] if a.validate else None)
            del S
            torch.cuda.empty_cache()
    rl.close()


if __name__ == "__main__":
    main()
