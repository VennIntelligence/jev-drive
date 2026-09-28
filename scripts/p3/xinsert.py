"""P3 cross-scene insertion (user 2026-09-28 ~15:30): any reconstructed pedestrian node of any processed segment can be
inserted into any target segment, with the approved grounded standard (LiDAR ground contact, contact shadow, exposure
gain cap 1.33, view gap <= 20 deg). todos/2026-09-26-night-queue-4.md, P section.

  bank   (GPU, one donor scene k)  every pedestrian node of the OmniRe run of scene k: height and gaussian extent, walking
         windows (51 frames, 0.5-2.5 m/s, straight: heading change <= 25 deg, path / chord <= 1.15), the viewing
         azimuths it was observed from (3 front cameras, <= 50 m, within 25 deg of the camera axis; azimuth = bearing
         from the pedestrian to the camera relative to its body heading), box PSNR (x+ vs log, front views with box
         height >= 40 px) and the donor / local-background ratio per RGB channel in the log (the exposure / white-balance
         reference).  -> runs/nq4/p3/xinsert/bank/p3_<k>.json
  plan   (CPU, any target scene)  anchor t* = crc32-minimal frame in [5, 15] s with ego >= 2 m/s; for every donor x window
         x entry side, the donor walks the target lane perpendicular to the logged path (from the entry side), centred
         at t* on the lane centre with TTR 3 s; view gap = max over the frames it is in the front camera of the smallest
         azimuth difference to what the log saw of the donor. Donor rules: box PSNR >= 22 dB, height 1.0-2.1 m,
         gaussian vertical extent / box height 0.8-1.25. The minimal-gap placement wins (ties: higher PSNR); an item
         exists when that gap is <= 20 deg. TTR 2 / 4 s reuse it; the null walks along the path 4.5 m off the lane on
         the entry side, direction of the smaller gap.  -> xinsert/plan/p3_<k>.json
  render (GPU, one target with a plan)  the target OmniRe run plus the donor's gaussians and its own deformation network
         (evaluated at the donor's frame), placed rigidly, feet on the target LiDAR ground (1.1 s median), SH evaluated
         in the donor's own frame (view directions rotated back), per-channel gain on the donor's colours so that its
         donor / background ratio equals the one in its own log (clip [0.75, 1.33]), and an ambient-occlusion contact
         shadow only where the ground under the feet is lit (local luminance >= 0.6 x the lit-lane reference). Outputs
         real, minus (the target re-render), ins2 / ins3 / ins4 / null in the insert.py layout, meta.json, clips.
  yield  (CPU) summary of the plans.

  CUDA_VISIBLE_DEVICES=6 $DATA_DIR/envs/drivestudio/bin/python scripts/p3/xinsert.py bank --scene 0
  $DATA_DIR/envs/drivestudio/bin/python scripts/p3/xinsert.py plan --targets 0 1 ... 65
  CUDA_VISIBLE_DEVICES=6 $DATA_DIR/envs/drivestudio/bin/python scripts/p3/xinsert.py render --target 0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insert as INS  # noqa: E402  (Path2D, wrap, crc, constants; also imports ds as DSM)

DSM = INS.DSM
DATA = Path(os.environ["DATA_DIR"])
X = DATA / "runs/nq4/p3/xinsert"
PROC = DATA / "processed/waymo_ds/training"
HZ, PRE, POST, FRONT, V_FLOOR = 10, 30, 20, 4.0, 3.0
AZ_MAX, PSNR_MIN, GAIN_MAX, SHADE = 20.0, 22.0, 1.33, 0.6
H_RANGE, SPAN_RANGE, SPEED_RANGE, TURN_MAX, STRAIGHT = (1.0, 2.1), (0.8, 1.25), (0.5, 2.5), 25.0, 1.15
WIN = PRE + POST + 1


def scene_geom(k: int) -> dict:
    d = PROC / f"{k:03d}"
    n = len(list((d / "ego_pose").glob("*.txt")))
    E0 = np.linalg.inv(np.loadtxt(d / "ego_pose/000.txt"))
    E = np.stack([E0 @ np.loadtxt(d / "ego_pose" / f"{t:03d}.txt") for t in range(n)])
    ext = [np.loadtxt(d / "extrinsics" / f"{i}.txt") for i in range(3)]
    intr = [np.loadtxt(d / "intrinsics" / f"{i}.txt") for i in range(3)]
    img = __import__("PIL.Image", fromlist=["Image"]).open(d / "images/000_0.jpg").size
    return {"dir": d, "n": n, "E": E, "ext": ext, "intr": intr, "wh": img}


def cam_axis(g, t, c=0):
    """World position and forward axis (x of the Waymo camera frame) of camera c at frame t."""
    cw = g["E"][t] @ g["ext"][c]
    return cw[:3, 3], cw[:3, 0]


def in_fov(g, t, p, c=0, dist=50.0):
    pos, ax = cam_axis(g, t, c)
    d = p[:2] - pos[:2]
    if np.linalg.norm(d) > dist or np.linalg.norm(d) < 1.0:
        return False
    half = np.arctan(g["wh"][0] / 2 / g["intr"][c][0])
    return abs(INS.wrap(np.arctan2(d[1], d[0]) - np.arctan2(ax[1], ax[0]))) <= min(half, np.radians(25))


def azimuth(g, t, p, yaw, c=0):
    pos, _ = cam_axis(g, t, c)
    d = pos[:2] - p[:2]
    return float(INS.wrap(np.arctan2(d[1], d[0]) - yaw))


class Ground:
    def __init__(self, g, frames):
        pts = []
        for t in range(max(min(frames) - 30, 0), min(max(frames) + 30, g["n"]), 3):
            L = np.fromfile(g["dir"] / "lidar" / f"{t:03d}.bin", dtype=np.float32).reshape(-1, 14)
            L = L[L[:, 10] == 1][:, 3:6].astype(np.float64)
            pts.append(L @ g["E"][t][:3, :3].T + g["E"][t][:3, 3])
        self.p = np.concatenate(pts)

    def __call__(self, xy):
        d = np.linalg.norm(self.p[:, :2] - xy, axis=1)
        for r in (0.8, 2.0):
            if (d <= r).sum() >= 5:
                return float(np.median(self.p[d <= r, 2]))
        return float("nan")


# ---------------------------------------------------------------- bank
def bank(a):
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    k = a.scene
    kk = f"{k:03d}"
    run = DATA / "ckpt/nq4_p3/p3" / kk
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = DSM._load(run, None)
    g = scene_geom(k)
    nf = g["n"]
    T0 = node.instances_trans.detach().cpu().numpy().astype(np.float64)
    R0 = quaternion_to_matrix(node.quat_act(node.instances_quats.detach())).cpu().numpy().astype(np.float64)
    fv = node.instances_fv.detach().cpu().numpy()
    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()  # noqa: E731
    out = []
    for j in range(len(keys)):
        if keys[j] not in peds:
            continue
        dk = keys[j]
        h = float(ps.instances_size[dk][2])
        M = node._means.detach()[node.point_ids[:, 0] == j].cpu().numpy()
        if len(M) < 200:
            continue
        span = float((np.percentile(M[:, 2], 99) - np.percentile(M[:, 2], 1)) / h)
        wins = []
        for s0 in range(1, nf - WIN - 1, 5):
            if not fv[s0 - 1:s0 + WIN + 1, j].all():
                continue
            P = T0[s0:s0 + WIN, j, :2]
            chord = np.linalg.norm(P[-1] - P[0])
            speed = chord / ((WIN - 1) / HZ)
            if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
                continue
            a1, a2 = P[WIN // 2] - P[0], P[-1] - P[WIN // 2]
            turn = abs(np.degrees(INS.wrap(np.arctan2(a2[1], a2[0]) - np.arctan2(a1[1], a1[0]))))
            plen = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
            if turn <= TURN_MAX and plen / max(chord, 1e-6) <= STRAIGHT:
                wins.append({"s0": s0, "speed": round(float(speed), 2), "turn": round(float(turn), 1)})
        if not wins:
            continue
        az = []
        for t in np.flatnonzero(fv[:, j]):
            yaw = np.arctan2(R0[t, j, 1, 0], R0[t, j, 0, 0])
            for c in range(3):
                if in_fov(g, t, T0[t, j], c):
                    az.append(round(np.degrees(azimuth(g, t, T0[t, j], yaw, c)), 1))
        # box PSNR and the per-channel donor / background ratio in the log (front views, box >= 40 px)
        ps_l, rat = [], []
        vis = [t for t in range(nf) if fv[t, j]]
        with torch.no_grad():
            for t in vis[::max(len(vis) // 16, 1)]:
                ii, cc = ds.full_image_set.get_image(t * ps.num_cams, 1)
                ii = {q: v.cuda() if torch.is_tensor(v) else v for q, v in ii.items()}
                cc = {q: v.cuda() if torch.is_tensor(v) else v for q, v in cc.items()}
                K, c2w = cc["intrinsics"].cpu().numpy().astype(np.float64), cc["camera_to_world"].cpu().numpy().astype(np.float64)
                real = to8(ii["pixels"])
                r = DSM._box_mask(ps.instances_pose[t, dk].cpu().numpy(), ps.instances_size[dk].cpu().numpy(), K, c2w, *real.shape[:2], 0)
                if not r or r[3] - r[1] < 40:
                    continue
                plus = tr(ii, cc)["rgb"]
                e = (plus[r[1]:r[3], r[0]:r[2]] - ii["pixels"][r[1]:r[3], r[0]:r[2]]).float()
                ps_l.append(float(-10 * torch.log10((e ** 2).mean().clamp_min(1e-10))))
                fv0 = node.instances_fv.clone()
                node.instances_fv[:, j] = False
                minus = tr(ii, cc)["rgb"]
                node.instances_fv.copy_(fv0)
                m = to8(plus).astype(int) - to8(minus).astype(int)
                m = np.abs(m).max(-1) > INS.DONOR_THR
                box = np.zeros_like(m)
                box[r[1]:r[3], r[0]:r[2]] = True
                m &= box
                if m.sum() >= 200:
                    ring = INS._ring(m)
                    rat.append([float(np.median(real[..., c][m]) / max(np.median(real[..., c][ring]), 1.0)) for c in range(3)])
                if len(ps_l) >= 8:
                    break
        out.append({"scene": k, "node": j, "waymo_id": wid_of[j], "height": round(h, 2), "span": round(span, 2),
                    "psnr": round(float(np.mean(ps_l)), 2) if ps_l else None, "ratio": np.median(rat, 0).round(3).tolist() if rat else None,
                    "n_az": len(az), "az": az[::max(len(az) // 400, 1)], "windows": wins, "ckpt": ckpt})
    (X / "bank").mkdir(parents=True, exist_ok=True)
    (X / "bank" / f"p3_{kk}.json").write_text(json.dumps(out))
    print(json.dumps({"scene": k, "donors": len(out)}))


# ---------------------------------------------------------------- placement
def load_bank():
    out = []
    for f in sorted((X / "bank").glob("p3_*.json")):
        for d in json.loads(f.read_text()):
            ok = (d["psnr"] is not None and d["psnr"] >= PSNR_MIN and H_RANGE[0] <= d["height"] <= H_RANGE[1]
                  and SPAN_RANGE[0] <= d["span"] <= SPAN_RANGE[1] and d["ratio"] is not None and d["n_az"] >= 10)
            if ok:
                out.append(d)
    return out


def donor_track(d, cache={}):
    """Donor node poses (trans, yaw) over all its frames, from the checkpoint (node poses cached per checkpoint)."""
    import torch
    if d["ckpt"] not in cache:
        sd = torch.load(d["ckpt"], map_location="cpu", weights_only=False)["models"]["DeformableNodes"]
        q = sd["instances_quats"].double()
        q = q / q.norm(dim=-1, keepdim=True)
        w, x, y, z = q.unbind(-1)
        cache[d["ckpt"]] = (sd["instances_trans"].double().numpy(), torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).numpy())
    T, yaw = cache[d["ckpt"]]
    return T[:, d["node"]], yaw[:, d["node"]]


def cam_table(g, frames):
    """Front camera centre (xy), axis angle and half field of view per frame."""
    C = np.array([cam_axis(g, t)[0][:2] for t in frames])
    A = np.array([np.arctan2(*cam_axis(g, t)[1][1::-1]) for t in frames])
    return C, A, min(np.arctan(g["wh"][0] / 2 / g["intr"][0][0]), np.radians(25))


def place(g, path, t_star, d, win, side, ttr, null=False, cams=None):
    """Placed donor window in target geometry g (vectorised over the 51 frames): the rotation, the lane-centre point and
    the view gap (max over frames in the front camera of the smallest azimuth difference to the observed ones)."""
    T, yaw = donor_track(d)
    s0 = win["s0"]
    s_star = s0 + PRE
    frames = np.arange(t_star - PRE, t_star + POST + 1)
    C, A, half = cams if cams is not None else cam_table(g, frames)
    u = T[s0 + WIN - 1, :2] - T[s0, :2]
    head = np.arctan2(u[1], u[0])
    S = path.Se[t_star] + FRONT + ttr * max(path.v[t_star], V_FLOOR)
    Q, th, _ = path.at(S)
    if null:
        Q = Q + side * INS.NULL_LAT * np.array([-np.sin(th), np.cos(th)])
        cands = [th, th + np.pi]
    else:
        cands = [th - side * np.pi / 2]
    obs = np.radians(np.asarray(d["az"]))
    rel = T[s0:s0 + WIN, :2] - T[s_star, :2]
    best = None
    for target in cands:
        phi = INS.wrap(target - head)
        c, s_ = np.cos(phi), np.sin(phi)
        xy = rel @ np.array([[c, s_], [-s_, c]]) + Q
        yw = yaw[s0:s0 + WIN] + phi
        dv = xy - C
        dist = np.linalg.norm(dv, axis=1)
        fov = (dist <= 50) & (dist >= 1) & (np.abs(INS.wrap(np.arctan2(dv[:, 1], dv[:, 0]) - A)) <= half)
        if fov.sum() < 5:
            continue
        az = INS.wrap(np.arctan2(-dv[:, 1], -dv[:, 0]) - yw)
        gaps = np.degrees(np.abs(INS.wrap(obs[None, :] - az[:, None])).min(1))[fov]
        gap = float(gaps.max())
        if best is None or gap < best["gap"]:
            best = {"gap": gap, "phi": float(phi), "Q": Q.tolist(), "n_fov": int(fov.sum())}
    return best


def plan_one(k: int, bank_: list) -> dict:
    g = scene_geom(k)
    path = INS.Path2D(g["E"])
    seg = json.loads((DATA / "runs/nq4/p3/targets" / f"{k:03d}.json").read_text())["segment"] if \
        (DATA / "runs/nq4/p3/targets" / f"{k:03d}.json").exists() else f"scene{k}"
    ok_t = [t for t in range(max(50, PRE + 1), min(151, g["n"] - POST - 1)) if path.v[t] >= 2.0]
    if not ok_t:
        return {"scene": k, "segment": seg, "item": False, "reason": "ego never >= 2 m/s in [5, 15] s"}
    ts = min(ok_t, key=lambda t: INS.crc(f"{seg}/xinsert/{t}"))
    gr = Ground(g, range(ts - PRE, ts + POST + 1))
    Q, _, _ = path.at(path.Se[ts] + FRONT + 3.0 * max(path.v[ts], V_FLOOR))
    if not np.isfinite(gr(Q)):
        return {"scene": k, "segment": seg, "item": False, "t_star": ts, "reason": "no LiDAR ground at the placement"}
    cands = []
    cams = cam_table(g, range(ts - PRE, ts + POST + 1))
    for d in bank_:
        for w in d["windows"]:
            for side in (1.0, -1.0):
                b = place(g, path, ts, d, w, side, 3.0, cams=cams)
                if b:
                    cands.append((b["gap"], -d["psnr"], d, w, side, b))
    if not cands:
        return {"scene": k, "segment": seg, "item": False, "t_star": ts, "reason": "no donor in view"}
    cands.sort(key=lambda c: (round(c[0], 1), c[1]))
    gap, _, d, w, side, b = cands[0]
    res = {"scene": k, "segment": seg, "t_star": ts, "ego_v": float(path.v[ts]), "item": gap <= AZ_MAX, "view_gap": round(gap, 1),
           "donor": {kk: d[kk] for kk in ("scene", "node", "waymo_id", "height", "span", "psnr", "ratio", "ckpt")},
           "window": w, "side": side, "n_candidates": len(cands),
           "runner_up": [{"gap": round(c[0], 1), "donor_scene": c[2]["scene"], "node": c[2]["node"]} for c in cands[1:4]]}
    res["variants"] = {}
    for name, ttr, null in (("ins2", 2.0, False), ("ins3", 3.0, False), ("ins4", 4.0, False), ("null", 3.0, True)):
        bb = place(g, path, ts, d, w, side, ttr, null, cams=cams)
        res["variants"][name] = {"phi": bb["phi"], "Q": bb["Q"], "view_gap": round(bb["gap"], 1)} if bb else None
    return res


def plan(a):
    b = load_bank()
    (X / "plan").mkdir(parents=True, exist_ok=True)
    for k in a.targets:
        if not (PROC / f"{k:03d}/lidar").exists():
            continue
        r = plan_one(k, b)
        (X / "plan" / f"p3_{k:03d}.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({q: r.get(q) for q in ("scene", "item", "view_gap", "reason")} | {"donor": (r.get("donor") or {}).get("scene")}), flush=True)


def yield_(a):
    rows = [json.loads(f.read_text()) for f in sorted((X / "plan").glob("p3_*.json"))]
    b = load_bank()
    gaps = [r["view_gap"] for r in rows if "view_gap" in r]
    s = {"targets": len(rows), "items": sum(r["item"] for r in rows), "donors_in_bank": len(b),
         "donor_scenes": sorted({d["scene"] for d in b}), "gap_median": float(np.median(gaps)) if gaps else None,
         "no_item_reasons": {}}
    for r in rows:
        if not r["item"]:
            why = r.get("reason") or f"best view gap {r['view_gap']:.0f} deg > {AZ_MAX:.0f}"
            s["no_item_reasons"][f"p3_{r['scene']:03d}"] = why
    (X / "yield.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


# ---------------------------------------------------------------- render
class XDonor:
    """The donor's gaussians as an extra node class of the target scene graph."""

    def __init__(self, d, device):
        import torch
        from omegaconf import OmegaConf
        from models.modules import ConditionalDeformNetwork
        sd = torch.load(d["ckpt"], map_location=device, weights_only=False)["models"]["DeformableNodes"]
        j = d["node"]
        m = sd["points_ids"][:, 0] == j
        self.means, self._scales, self._quats = sd["_means"][m], sd["_scales"][m], sd["_quats"][m]
        self.fdc, self.frest, self._op = sd["_features_dc"][m], sd["_features_rest"][m], sd["_opacities"][m]
        self.embed, self.size = sd["instances_embedding"][j], sd["instances_size"][j]
        self.trans, self.iq = sd["instances_trans"][:, j].clone(), sd["instances_quats"][:, j].clone()
        cfg = OmegaConf.load(Path(d["ckpt"]).parent / "config.yaml")
        ncfg = cfg.model.DeformableNodes
        self.net = ConditionalDeformNetwork(input_ch=3, **ncfg.networks).to(device)
        self.net.load_state_dict({q[len("deform_network."):]: v for q, v in sd.items() if q.startswith("deform_network.")})
        self.net.eval()
        self.ts = torch.linspace(0, 1, self.trans.shape[0], device=device)
        self.sh = int(ncfg.get("sh_degree", 3))
        self.cur, self.frame_map, self.pose, self.gain = None, {}, {}, torch.ones(3, device=device)
        self.step, self.in_test_set = 10 ** 6, False

    def set_cur_frame(self, t):
        self.cur = int(t)

    def get_gaussians(self, cam):
        import torch
        from models.gaussians.basics import quat_mult, quat_to_rotmat, spherical_harmonics
        if self.cur not in self.pose:
            return None
        s = self.frame_map[self.cur]
        R, T, Rd = self.pose[self.cur]                                  # placed rotation, translation; donor's own rotation
        n = self.means.shape[0]
        x = self.means / self.size[2] * 2
        dxyz, dq, dsc = self.net(x, self.ts[s].reshape(1, 1).repeat(n, 1), self.embed[None].repeat(n, 1))
        world = (self.means + (dxyz if dxyz is not None else 0)) @ R.T + T
        qR = __import__("pytorch3d.transforms", fromlist=["matrix_to_quaternion"]).matrix_to_quaternion(R)
        q = self._quats / self._quats.norm(dim=-1, keepdim=True) + (dq if dq is not None else 0)
        quats = quat_mult(qR[None].repeat(n, 1), q / q.norm(dim=-1, keepdim=True))
        scales = torch.exp(self._scales) + (dsc if dsc is not None else 0)
        vd = world.detach() - cam.camtoworlds.data[..., :3, 3]
        vd = (vd @ R) @ Rd.T                                            # into the donor's own world frame
        vd = vd / vd.norm(dim=-1, keepdim=True)
        colors = torch.cat((self.fdc[:, None, :], self.frest), dim=1)
        rgb = torch.clamp(spherical_harmonics(self.sh, vd, colors) + 0.5, 0.0, 1.0) * self.gain
        return {"_means": world, "_opacities": torch.sigmoid(self._op), "_rgbs": rgb.clamp(0, 1), "_scales": scales,
                "_quats": quats / quats.norm(dim=-1, keepdim=True)}


def render(a):
    import torch
    from PIL import Image, ImageDraw, ImageFilter
    from scipy.ndimage import binary_dilation
    from pytorch3d.transforms import quaternion_to_matrix
    t_start = time.time()
    k = a.target
    kk = f"{k:03d}"
    key = f"p3_{kk}"
    pl = json.loads((X / "plan" / f"{key}.json").read_text())
    assert pl["item"], f"{key}: no cross-scene item ({pl.get('reason') or pl.get('view_gap')})"
    run = DATA / "ckpt/nq4_p3/p3" / kk
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = DSM._load(run, None)
    from models.trainers.base import GSModelType                      # importable once _load put drivestudio on the path
    g = scene_geom(k)
    path = INS.Path2D(g["E"])
    ts = pl["t_star"]
    frames = list(range(ts - PRE, ts + POST + 1))
    gr = Ground(g, frames)
    dev = node.instances_trans.device
    d = pl["donor"]
    xd = XDonor(d, dev)
    Rq = quaternion_to_matrix(xd.iq / xd.iq.norm(dim=-1, keepdim=True)).double().cpu().numpy()
    Tn = xd.trans.double().cpu().numpy()
    Mloc = xd.means.double().cpu().numpy()
    s0 = pl["window"]["s0"]
    s_star = s0 + PRE
    fmap = {t: s0 + i for i, t in enumerate(frames)}
    xd.frame_map = fmap
    tr.models["XDonor"] = xd
    tr.gaussian_classes["XDonor"] = GSModelType.DeformableNodes

    def poses(v):
        phi = v["phi"]
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        Q = np.asarray(v["Q"])
        raw = {}
        for t in frames:
            s = fmap[t]
            xy = (Rz[:2, :2] @ (Tn[s, :2] - Tn[s_star, :2])) + Q
            Rw = Rz @ Rq[s]
            feet = np.percentile((Mloc @ Rw.T)[:, 2], 1)               # lowest gaussians relative to the node origin
            raw[t] = (xy, Rw, gr(xy) - feet, Rq[s])
        z = np.array([raw[t][2] for t in frames])
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
        zs = np.array([np.median(z[max(i - 5, 0):i + 6]) for i in range(len(z))])
        out = {}
        for i, t in enumerate(frames):
            xy, Rw, _, Rd = raw[t]
            out[t] = (torch.tensor(Rw, dtype=torch.float32, device=dev), torch.tensor(np.r_[xy, zs[i]], dtype=torch.float32, device=dev),
                      torch.tensor(Rd, dtype=torch.float32, device=dev))
        return out, {t: (raw[t][0], zs[i] + np.percentile((Mloc @ raw[t][1].T)[:, 2], 1)) for i, t in enumerate(frames)}

    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()  # noqa: E731

    camK = {}

    def view(t, ci=0):
        ii, cc = ds.full_image_set.get_image(t * ps.num_cams + ci, 1)
        ii = {q: v.cuda() if torch.is_tensor(v) else v for q, v in ii.items()}
        cc = {q: v.cuda() if torch.is_tensor(v) else v for q, v in cc.items()}
        camK[(t, ci)] = (cc["intrinsics"].cpu().numpy().astype(np.float64), cc["camera_to_world"].cpu().numpy().astype(np.float64))
        return ii, cc

    out = a.out / key
    V = {n: poses(v) for n, v in pl["variants"].items() if v}
    # exposure / white balance: per-channel gain from TTR 3 s frames (donor / ring ratio vs its own log), clipped
    rat0 = np.asarray(d["ratio"])
    gains = []
    with torch.no_grad():
        xd.pose = V["ins3"][0]
        for t in frames[::5]:
            ii, cc = view(t)
            xd.pose = V["ins3"][0]
            wi = to8(tr(ii, cc)["rgb"])
            xd.pose = {}
            wo = to8(tr(ii, cc)["rgb"])
            m = np.abs(wi.astype(int) - wo.astype(int)).max(-1) > INS.DONOR_THR
            if m.sum() < 150:
                continue
            ring = INS._ring(m)
            gains.append([rat0[c] * max(np.median(wo[..., c][ring]), 1.0) / max(np.median(wi[..., c][m]), 1.0) for c in range(3)])
    gain = np.clip(np.median(gains, 0), 1 / GAIN_MAX, GAIN_MAX) if gains else np.ones(3)
    xd.gain = torch.tensor(gain, dtype=torch.float32, device=dev)
    # lit-lane reference and per-frame shade test on the target re-render (front camera)
    names = ["real", "minus"] + list(V)
    rows = {n: [] for n in names}
    shadow_on = {}
    with torch.no_grad():
        for n in names:
            for c in INS.CAMS:
                (out / n / "cams" / c).mkdir(parents=True, exist_ok=True)
            for t in frames:
                rec = {}
                for ci, c in enumerate(INS.CAMS):
                    ii, cc = view(t, ci)
                    xd.pose = V[n][0] if n in V else {}
                    im = to8(ii["pixels"]) if n == "real" else to8(tr(ii, cc)["rgb"])
                    fn = f"{2 * t:07d}.jpg"
                    Image.fromarray(im).save(out / n / "cams" / c / fn, quality=95)
                    rec[c] = f"cams/{c}/{fn}"
                rows[n].append({"frame": 2 * t, "t": t / HZ, "files": rec})
            (out / n / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[n]))
        xd.pose = {}

        def disc_mask(t, ci, xy, zg, r, shape):
            K, c2w = camK[(t, ci)]
            ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            P = np.c_[xy[0] + r * np.cos(ang), xy[1] + r * np.sin(ang), np.full(32, zg), np.ones(32)]
            pc = (np.linalg.inv(c2w) @ P.T).T[:, :3]
            if not np.isfinite(zg) or (pc[:, 2] <= 0.5).any():
                return None, 0
            uv = (K @ pc.T).T
            uv = uv[:, :2] / uv[:, 2:]
            mk = Image.new("L", (shape[1], shape[0]), 0)
            ImageDraw.Draw(mk).polygon([tuple(p) for p in uv], fill=255)
            return np.asarray(mk) > 0, float(uv[:, 0].max() - uv[:, 0].min())

        def lum(im):
            return im[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])

        # lit-lane reference per frame: 90th percentile of the lane pixels 8-40 m ahead in the target re-render
        for n in V:
            feet = V[n][1]
            sh = {}
            for t in frames:
                mi = np.asarray(Image.open(out / "minus/cams/front" / f"{2 * t:07d}.jpg"))
                lanes = []
                for sd_ in np.arange(8, 40, 1.0):
                    q, th, _ = path.at(path.Se[t] + sd_)
                    for lat in (-1.0, 0.0, 1.0):
                        p = q + lat * np.array([-np.sin(th), np.cos(th)])
                        mk, _ = disc_mask(t, 0, p, gr(p), 0.3, mi.shape)
                        if mk is not None and mk.any():
                            lanes.append(np.median(lum(mi)[mk]))
                xy, zf = feet[t]
                mk, _ = disc_mask(t, 0, xy, zf, 0.6, mi.shape)
                if mk is None or not mk.any() or len(lanes) < 5:
                    sh[t] = None
                    continue
                sh[t] = bool(np.median(lum(mi)[mk]) >= SHADE * np.percentile(lanes, 90))
            vals = [v for v in sh.values() if v is not None]
            default = (sum(vals) >= len(vals) / 2) if vals else True
            seq = [sh[t] if sh[t] is not None else default for t in frames]
            on = {t: bool(np.mean(seq[max(i - 5, 0):i + 6]) >= 0.5) for i, t in enumerate(frames)}
            shadow_on[n] = on
            for t in frames:
                if not on[t]:
                    continue
                xy, zf = feet[t]
                fn = f"{2 * t:07d}.jpg"
                for ci, c in enumerate(INS.CAMS):
                    im = np.asarray(Image.open(out / n / "cams" / c / fn)).astype(np.float64)
                    mi = np.asarray(Image.open(out / "minus/cams" / c / fn))
                    donor = np.abs(im.astype(int) - mi.astype(int)).max(-1) > INS.DONOR_THR
                    mk, wpx = disc_mask(t, ci, xy, zf, INS.R_SHADOW, im.shape)
                    if mk is None or not mk.any():
                        continue
                    blur = Image.fromarray((mk * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(max(1.5, 0.25 * wpx))))
                    shd = np.asarray(blur) / 255.0 * ~binary_dilation(donor, iterations=2)
                    im = im * (1 - INS.ALPHA_SHADOW * shd[..., None])
                    Image.fromarray(np.clip(im + 0.5, 0, 255).astype(np.uint8)).save(out / n / "cams" / c / fn, quality=95)
    meta = {"scene": key, "target_segment": pl["segment"], "ckpt": ckpt, "t_star": ts, "frames": frames, "plan": pl,
            "donor_segment": json.loads((DATA / "runs/nq4/p3/targets" / f"{d['scene']:03d}.json").read_text())["segment"],
            "gain_rgb": [float(x) for x in gain], "gain_views": len(gains),
            "shadow_on_fraction": {n: float(np.mean(list(v.values()))) for n, v in shadow_on.items()},
            "render_s": round(time.time() - t_start, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps({"scene": key, "donor": [d["scene"], d["node"]], "view_gap": pl["view_gap"], "gain": meta["gain_rgb"],
                      "shadow": meta["shadow_on_fraction"], "render_s": meta["render_s"]}))


def clip(a):
    """Before | after for the docs: log | same-scene insertion (grounded standard; 'excluded' when that scene had no
    donor within 20 deg) | cross-scene insertion, TTR 3 s, front camera, 400 x 267 each, 5 fps over [t* - 2, t* + 1] s."""
    from PIL import Image, ImageDraw
    import review_sheet as RS
    f1, f2 = RS._font(15), RS._font(13)
    for k in a.targets:
        key = f"p3_{k:03d}"
        xm = json.loads((X / "items" / key / "meta.json").read_text())
        same = DATA / "runs/nq4/p3/insert_batch" / key
        sm = json.loads((same / "meta.json").read_text()) if (same / "meta.json").exists() else {}
        frames = []
        ts = xm["t_star"]
        for t in range(ts - 20, ts + 11, 2):
            G = Image.new("RGB", (1200, 267 + 40), (18, 18, 18))
            fn = f"{2 * t:07d}.jpg"
            tiles = [(X / "items" / key / "real/cams/front" / fn, "log")]
            if sm.get("chosen"):
                st = sm["t_star"]
                sfn = f"{2 * (st + t - ts):07d}.jpg"
                tiles.append((same / ("ins3f" if (same / "ins3f").exists() else "ins3") / "cams/front" / sfn, "same-scene donor"))
            else:
                tiles.append((None, "same-scene: excluded (no donor <= 20 deg)"))
            tiles.append((X / "items" / key / "ins3/cams/front" / fn, "cross-scene donor"))
            for i, (f, lab) in enumerate(tiles):
                if f is not None and Path(f).exists():
                    G.paste(Image.open(f).convert("RGB").resize((400, 267), Image.LANCZOS), (i * 400, 0))
                ImageDraw.Draw(G).text((i * 400 + 6, 4), lab, fill=(255, 255, 255), font=f1, stroke_width=2, stroke_fill=(0, 0, 0))
            dn = xm["plan"]["donor"]
            ImageDraw.Draw(G).text((8, 270), f"{key}  TTR 3 s   t - t* {(t - ts) / HZ:+.1f} s   donor p3_{dn['scene']:03d} node {dn['node']}  "
                                   f"view gap {xm['plan']['view_gap']:.0f} deg  gain {'/'.join(f'{x:.2f}' for x in xm['gain_rgb'])}  "
                                   f"shadow {xm['shadow_on_fraction'].get('ins3', 0):.0%} of frames", fill=(220, 220, 220), font=f2)
            frames.append(G)
        p = a.out / f"xinsert_{key}.webp"
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=200, loop=0, quality=70, method=6)
        print(json.dumps({"clip": str(p), "bytes": p.stat().st_size}))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bank")
    p.add_argument("--scene", type=int, required=True)
    p = sp.add_parser("plan")
    p.add_argument("--targets", type=int, nargs="+", required=True)
    sp.add_parser("yield")
    p = sp.add_parser("clip")
    p.add_argument("--targets", type=int, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=X / "clips")
    p = sp.add_parser("render")
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--out", type=Path, default=X / "items")
    a = ap.parse_args()
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if a.cmd == "clip":
        a.out.mkdir(parents=True, exist_ok=True)
    {"bank": bank, "plan": plan, "yield": yield_, "render": render, "clip": clip}[a.cmd](a)


if __name__ == "__main__":
    main()
