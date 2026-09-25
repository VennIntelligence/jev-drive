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
        self.road_xz, self.road_y = road[:, [0, 2]], road[:, 1]
        self.road_tree = cKDTree(self.road_xz)
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
            return render(viewpoint=view, prev_viewpoint=None, pc=pc or self.g, dynamic_gaussians=dyn, unicycles={},
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
        sides = (1.0, -1.0) if fam == "cutin" else (cut_side,) if fam == "null" else (0.0,)
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
            if ok and fam != "cutin":
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
    stats = {w: {"vis_px": [], "any_px": [], "out_px": []} for w in worlds}
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
            vis, anyp, outp = [], [], []
            files = {}
            for c, ck in zip(H.CAMS, H.CAM_KEYS):
                g0 = time.perf_counter()
                pkg = S.render(c, poses[k][c], actor)
                rgb = to_rgb(pkg)
                t_gpu += time.perf_counter() - g0
                if w == "minus":
                    minus[(k, c)] = rgb
                    if validate:
                        rgb2 = to_rgb(S.render(c, poses[k][c], None))
                        val["det_max"] = max(val["det_max"], int(np.abs(rgb2.astype(int) - rgb).max()))
                    vis.append(0), anyp.append(0), outp.append(0)
                else:
                    diff = np.abs(rgb.astype(np.int16) - minus[(k, c)]).max(-1)
                    ch = diff > H.DIFF_THR
                    box = proj_box(S, c, poses[k][c], m["asset"], actor[1])
                    inb = np.zeros_like(ch)
                    if box is not None:
                        inb[box[1]:box[3], box[0]:box[2]] = True
                    vis.append(int(ch.sum())), anyp.append(int((diff > 0).sum())), outp.append(int((diff > 0)[~inb].sum()))
                    if validate and c == "CAM_FRONT":
                        occ = occlusion(S, c, poses[k][c], actor, diff)
                        acc = val["occ"].setdefault(w, np.zeros(4, np.int64))
                        acc += occ
                        fig.setdefault(w, []).append((k, int(ch.sum()), rgb, minus[(k, c)]))
                f = f"cams/{ck}/{4 * k:07d}.jpg"
                files[ck] = f
                futs.append(pool.submit(cv2.imwrite, str(d / f), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 95]))
            stats[w]["vis_px"].append(vis), stats[w]["any_px"].append(anyp), stats[w]["out_px"].append(outp)
            lines.append(json.dumps({"frame": 4 * k, "t": float(t_render[k]), "files": files}))
        (d / "frames.jsonl").write_text("\n".join(lines) + "\n")
    for f in futs:
        assert f.result(), "jpeg write failed"
    for w in worlds:
        worlds[w].update(stats[w]) if worlds[w]["rendered"] else None
    if validate:
        val["env_max"] = env_check(S, row, poses, minus)
        val["occ"] = {w: v.tolist() for w, v in val["occ"].items()}
        save_fig_frames(out, fig, worlds, t_render)
    meta = {"key": key, "dataset": row["dataset"], "scene": row["scene"], "t_c": tc, "t_render": t_render.tolist(),
            "t_sim": t_sim.tolist(), "worlds": worlds, "cams": {c: {"K": S.cams[c][0].tolist(),
                                                                       "c2front": S.cams[c][1].tolist()}
                                                                   for c in H.CAMS},
            "timing": {"tracks_s": round(t_tracks, 2), "gpu_s": round(t_gpu, 2)}, "validation": val}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def occlusion(S, cam, c2w, actor, diff):
    """Occlusion consistency of one x+ view against an actor-only render (same renderer, empty scene):
    actor pixels (alpha > 0.5) where the x- scene surface is > 1 m in front of the actor must be unchanged, and
    those where it is > 1 m behind must change. Returns counts [occluded, occluded_changed, clear, clear_changed]."""
    empty = _EmptyPC.of(S.g)
    pa = S.render(cam, c2w, actor, pc=empty)
    pm = S.render(cam, c2w, None)
    alpha = pa["alphas"][0, ..., 0].cpu().numpy()
    da = pa["depth"][0].cpu().numpy()
    dm = pm["depth"][0].cpu().numpy()
    on = alpha > 0.5
    occ = on & (dm < da - 1.0)
    clr = on & (dm > da + 1.0)
    chg = diff > H.DIFF_THR
    return np.array([occ.sum(), (occ & chg).sum(), clr.sum(), (clr & chg).sum()], np.int64)


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


def save_fig_frames(out, fig, worlds, t_render):
    """For the figure: per plus world the front-camera frame with the most changed pixels before the conflict."""
    keep = {}
    for w, fr in fig.items():
        tc = worlds[w].get("t_conflict") or np.inf
        fr = [f for f in fr if t_render[f[0]] < tc - 0.3] or fr
        k, px, plus, minus = max(fr, key=lambda f: f[1])
        keep[w] = (k, plus, minus)
    np.savez_compressed(out / "fig_frames.npz", **{f"{w}_plus": v[1] for w, v in keep.items()},
                        **{f"{w}_minus": v[2] for w, v in keep.items()},
                        **{f"{w}_k": np.array(v[0]) for w, v in keep.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--tag", default="render")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    rl = RunLog("i3-hugsim-pairs", a.tag)
    sc = pd.read_csv(H.root() / "scenes.csv")
    sc = sc[sc.t_c.notna()]
    if a.keys:
        sc = sc.set_index("key").loc[a.keys].reset_index()
    rl.log.info("%d scenes, validate=%s", len(sc), a.validate)
    with ThreadPoolExecutor(3) as pool:
        for _, row in sc.iterrows():
            out = H.root("scenes", row["key"])
            if a.skip_done and (out / "meta.json").exists():
                continue
            t0 = time.perf_counter()
            S = Scene(row["dataset"], row["scene"])
            t_load = time.perf_counter() - t0
            L = H.Logged(S.dir, row["dataset"])
            meta = render_scene(S, L, row, out, a.validate, pool, rl)
            wall = time.perf_counter() - t0
            ws = {w: (m["reason"] if not m["rendered"] else "ok") for w, m in meta["worlds"].items()}
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
