"""P3 insertion feasibility smoke (todos/2026-09-26-night-queue-4.md, P section, amendment 2, option 3).

A reconstructed real pedestrian node of the same OmniRe scene (the donor) is moved into the ego lane: its own trajectory
and deformation over the render window, rigidly rotated and translated in the ground plane so that it crosses the lane
perpendicular to the logged ego path, entering from the side it was originally on, and sits on the lane centre at the
anchor t* when the ego is TTR in {2, 3, 4} s away (ego speed floored at 3 m/s). Height follows the ground (difference of
the ego pose z at the two places). x- = the same scene with the donor hidden; the insertion null moves the donor to
4.5 m outside the lane on its own side, walking along the path.
Donor: pedestrian nodes by FRONT-labelled frame count (jevdrive.nq4_p3_filter donors), ties by crc32; it must be tracked
(valid node pose) over the whole window around t*, its box PSNR (x+ against the log, original place, front camera) must be
>= 22 dB, and every rendered front view of it after the move must lie within 45 deg of a viewing azimuth the log saw it
from (else the next donor). t* = the crc32-minimal frame in [5, 15] s with ego >= 2 m/s at which the donor is tracked.

Outputs per scene: <out>/<key>/<variant>/cams/<cam>/<frame>.jpg and frames.jsonl (variants real, plus, minus, ins2,
ins3, ins4, null; 10 Hz, [t* - 3 s, t* + 2 s], three front cameras), meta.json, a review clip <key>.mp4 (10 fps, full
size) and <key>.webp (960 px, 5 fps, for the docs).

  CUDA_VISIBLE_DEVICES=6 $DATA_DIR/envs/drivestudio/bin/python scripts/p3/insert.py --scene 0 --out <dir>
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds as DSM  # noqa: E402  (scripts/p3/ds.py: the OmniRe loader and box projection)

DATA = Path(os.environ["DATA_DIR"])
HZ, FRONT, V_FLOOR, PRE, POST = 10, 4.0, 3.0, 30, 20
TTRS, NULL_LAT, AZ_MAX, PSNR_MIN = (2.0, 3.0, 4.0), 4.5, 45.0, 22.0
CAMS = DSM.CAMS


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class Path2D:
    """Logged ego path in the reconstruction's world frame (ego frame of timestep 0), extended 150 m straight ahead."""

    def __init__(self, E: np.ndarray):
        xy, z = E[:, :2, 3], E[:, 2, 3]
        yaw = np.arctan2(E[:, 1, 0], E[:, 0, 0])
        self.P = np.vstack([xy, xy[-1] + 150 * np.array([np.cos(yaw[-1]), np.sin(yaw[-1])])])
        self.z = np.r_[z, z[-1]]
        self.cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))]
        self.Se = self.cum[:-1]
        self.v = np.zeros(len(xy))
        self.v[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) * HZ / 2
        self.v[0], self.v[-1] = self.v[1], self.v[-2]

    def at(self, S: float):
        """Point, tangent angle and ground-level ego z at arc S."""
        k = int(np.clip(np.searchsorted(self.cum, S) - 1, 0, len(self.P) - 2))
        seg = self.P[k + 1] - self.P[k]
        ln = max(np.linalg.norm(seg), 1e-6)
        u = (S - self.cum[k]) / ln
        return self.P[k] + u * seg, np.arctan2(seg[1], seg[0]), self.z[k] + u * (self.z[k + 1] - self.z[k])

    def locate(self, p: np.ndarray):
        """Arc S and signed lateral L (left +) of a point."""
        a, ab = self.P[:-1], np.diff(self.P, axis=0)
        n2 = np.maximum((ab ** 2).sum(1), 1e-9)
        u = np.clip(((p - a) * ab).sum(1) / n2, 0, 1)
        dist = np.linalg.norm(p - (a + u[:, None] * ab), axis=1)
        j = int(dist.argmin())
        ln = np.sqrt(n2[j])
        return self.cum[j] + u[j] * ln, float((ab[j, 0] * (p - a[j])[1] - ab[j, 1] * (p - a[j])[0]) / ln)

ALPHA_SHADOW, R_SHADOW, DONOR_THR = 0.5, 0.35, 15


def _lum(im):
    return im[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _ring(mask):
    """Background ring around a donor mask: its bounding box widened by half its width each side and extended 30 %
    below, minus the (dilated) donor itself."""
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    r = np.zeros_like(mask)
    r[max(ys.min(), 0):min(ys.max() + int(0.3 * h) + 1, mask.shape[0]), max(xs.min() - w // 2, 0):min(xs.max() + w // 2 + 1, mask.shape[1])] = True
    from scipy.ndimage import binary_dilation
    return r & ~binary_dilation(mask, iterations=3)


def post_fix(a, out, variants, frames, j, dk, T0, p0, ground, view, apply, tr, ps, to8, DSM):
    """Exposure match and contact shadow on the grounded variants (their saved frames are rewritten in place).
    Exposure: the donor keeps the donor / local-background luminance ratio it has in the log at its original place
    (median over up to 8 front views where its box is >= 40 px); one gain per variant, clipped to [0.5, 2].
    Shadow: an ambient-occlusion blob, a disc of radius R_SHADOW on the LiDAR ground under the feet, projected into each
    camera, Gaussian-softened, darkening non-donor pixels by up to ALPHA_SHADOW."""
    import torch
    from PIL import Image, ImageDraw, ImageFilter
    from scipy.ndimage import binary_dilation
    ratios = []
    with torch.no_grad():
        vis = [t for t in range(ps.instances_pose.shape[0]) if bool(ps.per_frame_instance_mask[t, dk])]
        for t in vis[::max(len(vis) // 16, 1)]:
            ii, cc = view(t)
            K, c2w = cc["intrinsics"].cpu().numpy().astype(np.float64), cc["camera_to_world"].cpu().numpy().astype(np.float64)
            real = to8(ii["pixels"])
            r = DSM._box_mask(ps.instances_pose[t, dk].cpu().numpy(), ps.instances_size[dk].cpu().numpy(), K, c2w, *real.shape[:2], 0)
            if not r or r[3] - r[1] < 40:
                continue
            apply("plus")
            plus = to8(tr(ii, cc)["rgb"])
            apply("minus")
            minus = to8(tr(ii, cc)["rgb"])
            m = np.abs(plus.astype(int) - minus.astype(int)).max(-1) > DONOR_THR
            box = np.zeros_like(m)
            box[r[1]:r[3], r[0]:r[2]] = True
            m &= box
            if m.sum() < 200:
                continue
            ring = _ring(m)
            Y = _lum(real)
            ratios.append(np.median(Y[m]) / max(np.median(Y[ring]), 1.0))
            if len(ratios) >= 8:
                break
        apply("plus")
    ratio0 = float(np.median(ratios)) if ratios else float("nan")
    res = {"exposure_ratio_orig": ratio0, "exposure_views": len(ratios)}
    for name, g in variants.items():
        if not g.get("fixed"):
            continue
        d = out / name
        gains = []
        for t in frames:
            fn = f"{2 * t:07d}.jpg"
            im = np.asarray(Image.open(d / "cams/front" / fn))
            mi = np.asarray(Image.open(out / "minus/cams/front" / fn))
            m = np.abs(im.astype(int) - mi.astype(int)).max(-1) > DONOR_THR
            if m.sum() >= 200:
                ring = _ring(m)
                gains.append(ratio0 / (np.median(_lum(im)[m]) / max(np.median(_lum(mi)[ring]), 1.0)))
        gain = float(np.clip(np.median(gains), 0.5, 2.0)) if gains and np.isfinite(ratio0) else 1.0
        res[f"gain_{name}"] = gain
        for t in frames:
            fn = f"{2 * t:07d}.jpg"
            xy = (g["Rz"] @ (T0[t, j] - p0) + g["off"])[:2]
            zg = ground(xy)
            for ci, c in enumerate(CAMS):
                im = np.asarray(Image.open(d / "cams" / c / fn)).astype(np.float64)
                mi = np.asarray(Image.open(out / "minus/cams" / c / fn))
                m = np.abs(im.astype(int) - mi.astype(int)).max(-1) > DONOR_THR
                soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))) / 255.0
                im = im * (1 + (gain - 1) * soft[..., None])
                _, cc = view(t, ci)
                K, c2w = cc["intrinsics"].cpu().numpy().astype(np.float64), cc["camera_to_world"].cpu().numpy().astype(np.float64)
                ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                disc = np.c_[xy[0] + R_SHADOW * np.cos(ang), xy[1] + R_SHADOW * np.sin(ang), np.full(32, zg), np.ones(32)]
                pc = (np.linalg.inv(c2w) @ disc.T).T[:, :3]
                if np.isfinite(zg) and (pc[:, 2] > 0.5).all():
                    uv = (K @ pc.T).T
                    uv = uv[:, :2] / uv[:, 2:]
                    sh = Image.new("L", (im.shape[1], im.shape[0]), 0)
                    ImageDraw.Draw(sh).polygon([tuple(p) for p in uv], fill=255)
                    rad = max(1.5, 0.25 * (uv[:, 0].max() - uv[:, 0].min()))
                    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(rad))) / 255.0
                    sh = sh * ~binary_dilation(m, iterations=2)
                    im = im * (1 - ALPHA_SHADOW * sh[..., None])
                Image.fromarray(np.clip(im + 0.5, 0, 255).astype(np.uint8)).save(d / "cams" / c / fn, quality=95)
    return res


def compare_clip(a, out, key, frames, ts, variants, diag, ttf):
    """Before | after (TTR 3 s): full front view on top, a 2.5x crop around the donor's feet below."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    f1, f2 = ImageFont.truetype(str(ttf), 18), ImageFont.truetype(str(ttf), 16)
    before, after = variants["ins3"], variants["ins3f"]
    wr = imageio.get_writer(a.out / f"{key}_fix.mp4", fps=HZ, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    small = []
    for i, t in enumerate(frames):
        fn = f"{2 * t:07d}.jpg"
        mi = np.asarray(Image.open(out / "minus/cams/front" / fn))
        G = Image.new("RGB", (1280, 854 + 58), (20, 20, 20))
        for k, (name, title) in enumerate((("ins3", "before: rigid move"), ("ins3f", "after: grounded + exposure + contact shadow"))):
            im = Image.open(out / name / "cams/front" / fn)
            arr = np.asarray(im)
            m = np.abs(arr.astype(int) - mi.astype(int)).max(-1) > DONOR_THR
            G.paste(im.resize((640, 427), Image.LANCZOS), (k * 640, 0))
            if m.sum() >= 50:
                ys, xs = np.nonzero(m)
                cx, cy = int(np.median(xs)), int(ys.max())
            else:
                cx, cy = arr.shape[1] // 2, arr.shape[0] // 2
            x0, y0 = int(np.clip(cx - 128, 0, arr.shape[1] - 256)), int(np.clip(cy - 120, 0, arr.shape[0] - 171))
            G.paste(im.crop((x0, y0, x0 + 256, y0 + 171)).resize((640, 427), Image.LANCZOS), (k * 640, 427))
            d = ImageDraw.Draw(G)
            d.text((k * 640 + 8, 6), title, fill=(255, 255, 255), font=f1, stroke_width=2, stroke_fill=(0, 0, 0))
            off = before["foot_offset_m"].get(t, float("nan")) if name == "ins3" else 0.0
            d.text((k * 640 + 8, 427 + 6), f"zoom x2.5   foot above LiDAR ground {off:+.2f} m", fill=(255, 255, 255), font=f2,
                   stroke_width=2, stroke_fill=(0, 0, 0))
        ImageDraw.Draw(G).text((10, 862), f"{key}  insertion, TTR 3 s   t - t* = {(t - ts) / HZ:+.1f} s   exposure gain "
                               f"{diag.get('gain_ins3f', 1):.2f}   donor at its own place: foot {diag['orig_foot_offset_m']:+.2f} m",
                               fill=(230, 230, 230), font=f1)
        wr.append_data(np.asarray(G))
        if i % 2 == 0:
            small.append(G.resize((1024, 730), Image.LANCZOS))
    wr.close()
    small[0].save(a.out / f"{key}_fix.webp", save_all=True, append_images=small[1:], duration=200, loop=0, quality=78, method=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--donors", type=Path, default=DATA / "runs/nq4/p3/filter/donors")
    ap.add_argument("--fix", action="store_true", help="also render the grounded variants (<name>f): feet snapped to the LiDAR "
                    "ground, local exposure matched, contact shadow; plus a before | after clip of TTR 3 s")
    a = ap.parse_args()
    # nvdiffrast JIT needs this env's ninja on PATH (as in ds.py main)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    t_start = time.time()
    kk = f"{a.scene:03d}"
    key = f"p3_{kk}"
    run = DATA / "ckpt/nq4_p3/p3" / kk
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = DSM._load(run, None)
    tg = json.loads((DATA / "runs/nq4/p3/targets" / f"{kk}.json").read_text())
    proc = Path(cfg.data.data_root) / kk
    nf = ps.num_timesteps if hasattr(ps, "num_timesteps") else ds.num_img_timesteps
    E0 = np.linalg.inv(np.loadtxt(proc / "ego_pose/000.txt"))
    E = np.stack([E0 @ np.loadtxt(proc / "ego_pose" / f"{t:03d}.txt") for t in range(nf)])
    ext = [np.loadtxt(proc / "extrinsics" / f"{i}.txt") for i in range(len(CAMS))]
    path = Path2D(E)
    lab = json.loads((a.donors / f"{key}.json").read_text())["front_labelled"]
    fv = node.instances_fv.detach().cpu().numpy()                     # (frames, nodes)
    trans0 = node.instances_trans.detach().clone()
    quats0 = node.instances_quats.detach().clone()
    R0 = quaternion_to_matrix(node.quat_act(quats0)).cpu().numpy()   # (frames, nodes, 3, 3)
    T0 = trans0.cpu().numpy()
    cand = [j for j in range(len(keys)) if keys[j] in peds]
    cand.sort(key=lambda j: (-lab.get(wid_of[j], 0), crc(f"{tg['segment']}/{wid_of[j]}")))
    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()  # noqa: E731

    def view(t, ci=0):
        ii, cc = ds.full_image_set.get_image(t * ps.num_cams + ci, 1)
        return ({k: v.cuda() if torch.is_tensor(v) else v for k, v in ii.items()},
                {k: v.cuda() if torch.is_tensor(v) else v for k, v in cc.items()})

    def azimuth(cam_xy, p_xy, R):
        d = cam_xy - p_xy
        return wrap(np.arctan2(d[1], d[0]) - np.arctan2(R[1, 0], R[0, 0]))

    report, chosen = [], None
    for j in cand[:6]:
        w = wid_of[j]
        ok_t = [t for t in range(max(50, PRE + 1), min(151, nf - POST - 1)) if path.v[t] >= 2.0 and fv[t - PRE - 1:t + POST + 2, j].all()]
        rec = {"donor": w, "node": j, "front_labelled": lab.get(w, 0), "t_candidates": len(ok_t)}
        if not ok_t:
            report.append({**rec, "reject": "not tracked over any window"})
            continue
        ts = min(ok_t, key=lambda t: crc(f"{tg['segment']}/{w}/{t}"))
        # box PSNR of the unmodified re-render at the donor's original place (front camera, frames it is visible)
        ps_list = []
        dk = keys[j]
        with torch.no_grad():
            for t in [t for t in range(nf) if fv[t, j]][::5]:
                ii, cc = view(t)
                K = cc["intrinsics"].cpu().numpy().astype(np.float64)
                c2w = cc["camera_to_world"].cpu().numpy().astype(np.float64)
                real = ii["pixels"]
                H, W = real.shape[:2]
                r = DSM._box_mask(ps.instances_pose[t, dk].cpu().numpy(), ps.instances_size[dk].cpu().numpy(), K, c2w, H, W, 0)
                if not r or (r[3] - r[1]) < 30:
                    continue
                plus = tr(ii, cc)["rgb"]
                e = (plus[r[1]:r[3], r[0]:r[2]] - real[r[1]:r[3], r[0]:r[2]]).float()
                ps_list.append(float(-10 * torch.log10((e ** 2).mean().clamp_min(1e-10))))
                if len(ps_list) >= 12:
                    break
        psnr = float(np.mean(ps_list)) if ps_list else float("nan")
        # observed viewing azimuths (all three cameras, donor within 60 m and +-30 deg of the camera axis)
        az_obs = []
        for t in np.flatnonzero(fv[:, j]):
            for c in range(len(CAMS)):
                cw = E[t] @ ext[c]
                d = T0[t, j, :2] - cw[:2, 3]
                if np.linalg.norm(d) <= 60 and abs(wrap(np.arctan2(d[1], d[0]) - np.arctan2(cw[1, 0], cw[0, 0]))) <= np.radians(30):
                    az_obs.append(azimuth(cw[:2, 3], T0[t, j, :2], R0[t, j]))
        az_obs = np.array(az_obs)
        # placements
        p0 = T0[ts, j]
        u = T0[min(ts + 5, nf - 1), j, :2] - T0[max(ts - 5, 0), j, :2]
        if np.linalg.norm(u) < 0.3:                                 # standing: use its facing
            u = R0[ts, j, :2, 0]
        S0, L0 = path.locate(p0[:2])
        _, _, z0 = path.at(S0)
        side = 1.0 if L0 > 0 else -1.0
        frames = list(range(ts - PRE, ts + POST + 1))
        variants = {}
        for tau in (*TTRS, "null"):
            Ss = path.Se[ts] + FRONT + (3.0 if tau == "null" else tau) * max(path.v[ts], V_FLOOR)
            Q, th, zq = path.at(Ss)
            if tau == "null":
                Q = Q + side * NULL_LAT * np.array([-np.sin(th), np.cos(th)])
                target = th if np.dot(u, [np.cos(th), np.sin(th)]) >= 0 else th + np.pi
            else:
                target = th - side * np.pi / 2                     # walk from its own side across the lane
            phi = wrap(target - np.arctan2(u[1], u[0]))
            Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
            off = np.r_[Q, p0[2] + (zq - z0)]
            variants[f"ins{int(tau)}" if tau != "null" else "null"] = {"Rz": Rz, "off": off, "S": Ss, "phi": float(phi), "dz": float(zq - z0)}
        # azimuth rule on the front camera views after the move (inserted variants)
        worst = 0.0
        for name, g in variants.items():
            if name == "null":
                continue
            for t in frames:
                cw = E[t] @ ext[0]
                pn = g["Rz"] @ (T0[t, j] - p0) + g["off"]
                az = azimuth(cw[:2, 3], pn[:2], g["Rz"] @ R0[t, j])
                worst = max(worst, float(np.degrees(np.abs(wrap(az_obs - az)).min())) if len(az_obs) else 180.0)
        rec.update({"t_star": ts, "psnr_box": round(psnr, 2), "n_psnr_views": len(ps_list), "az_worst_deg": round(worst, 1),
                    "L0": round(L0, 2), "donor_speed": round(float(np.linalg.norm(u)), 2)})
        if not (psnr >= PSNR_MIN):
            report.append({**rec, "reject": f"box PSNR {psnr:.1f} < {PSNR_MIN}"})
            continue
        if worst > AZ_MAX:
            report.append({**rec, "reject": f"viewing azimuth {worst:.0f} deg > {AZ_MAX:.0f}"})
            continue
        chosen = (j, ts, p0, variants, frames, rec)
        report.append({**rec, "reject": None})
        break
    out = a.out / key
    out.mkdir(parents=True, exist_ok=True)
    if chosen is None:
        (out / "meta.json").write_text(json.dumps({"scene": key, "donors": report, "chosen": None}, indent=1, default=float))
        print(json.dumps({"scene": key, "chosen": None, "donors": report}, default=float))
        return
    j, ts, p0, variants, frames, rec = chosen
    dk = keys[j]
    # ---- ground contact diagnostics (fix iteration 2026-09-28): LiDAR ground (drivestudio ground label) around the donor
    gp = []
    for t in range(max(frames[0] - 30, 0), min(frames[-1] + 30, nf), 3):
        Lr = np.fromfile(proc / "lidar" / f"{t:03d}.bin", dtype=np.float32).reshape(-1, 14)
        Lr = Lr[Lr[:, 10] == 1][:, 3:6].astype(np.float64)
        gp.append(Lr @ E[t][:3, :3].T + E[t][:3, 3])
    gp = np.concatenate(gp)

    def ground(xy):
        d = np.linalg.norm(gp[:, :2] - xy, axis=1)
        for r in (0.8, 2.0):
            if (d <= r).sum() >= 5:
                return float(np.median(gp[d <= r, 2]))
        return float("nan")

    Mloc = node._means.detach()[node.point_ids[:, 0] == j].cpu().numpy().astype(np.float64)
    mod = list(range(frames[0] - 1, frames[-1] + 2))
    feet = {t: float(np.percentile((Mloc @ R0[t, j].T)[:, 2], 1) + T0[t, j, 2]) for t in mod}   # lowest 1 % of the gaussians
    orig_off = [feet[t] - ground(T0[t, j, :2]) for t in frames]
    diag = {"orig_foot_offset_m": float(np.nanmedian(orig_off)),
            "jitter_z_m": float(np.std(np.diff(T0[frames, j, 2], 2))), "jitter_xy_m": float(np.std(np.linalg.norm(np.diff(T0[frames, j, :2], 2, axis=0), axis=1)))}
    if a.fix:
        for name in list(variants):
            g = variants[name]
            xy = {t: (g["Rz"] @ (T0[t, j] - p0) + g["off"])[:2] for t in mod}
            off = np.array([feet[t] - p0[2] + g["off"][2] - ground(xy[t]) for t in mod])
            off = np.where(np.isfinite(off), off, np.nanmedian(off))
            sm = np.array([np.median(off[max(i - 5, 0):i + 6]) for i in range(len(off))])      # 1.1 s median: no jitter added
            g["foot_offset_m"] = {int(t): float(o) for t, o in zip(mod, off)}
            variants[name + "f"] = {**g, "dzs": {int(t): float(-o) for t, o in zip(mod, sm)}, "fixed": True}
    pose0 = ps.instances_pose[:, dk].clone()
    fv0 = node.instances_fv.detach().clone()

    def apply(name):
        node.instances_trans.data.copy_(trans0)
        node.instances_quats.data.copy_(quats0)
        node.instances_fv.copy_(fv0)
        ps.instances_pose[:, dk] = pose0
        if name == "minus":
            node.instances_fv[:, j] = False
        elif name in variants:
            g = variants[name]
            Rz = torch.tensor(g["Rz"], dtype=torch.float32, device=trans0.device)
            off = torch.tensor(g["off"], dtype=torch.float32, device=trans0.device)
            p0t = torch.tensor(p0, dtype=torch.float32, device=trans0.device)
            mod = list(range(frames[0] - 1, frames[-1] + 2))          # +-1 frame: pose interpolation at the edges
            fr = torch.tensor(mod, device=trans0.device)
            node.instances_trans.data[fr, j] = (trans0[fr, j] - p0t) @ Rz.T + off
            Rn = Rz @ quaternion_to_matrix(node.quat_act(quats0[fr, j]))
            node.instances_quats.data[fr, j] = matrix_to_quaternion(Rn)
            dv = pose0.device
            Rp, op_, p0p = Rz.to(dv), off.to(dv), p0t.to(dv)
            M = pose0[torch.tensor(mod, device=dv)].clone()
            M[:, :3, :3] = Rp @ M[:, :3, :3]
            M[:, :3, 3] = (M[:, :3, 3] - p0p) @ Rp.T + op_
            ps.instances_pose[torch.tensor(mod, device=dv), dk] = M
            if "dzs" in g:                                        # feet snapped to the LiDAR ground, per frame
                dz = torch.tensor([g["dzs"][t] for t in mod], dtype=torch.float32)
                node.instances_trans.data[fr, j, 2] += dz.to(trans0.device)
                ps.instances_pose[torch.tensor(mod, device=dv), dk, 2, 3] += dz.to(dv)

    names = ["real", "plus", "minus", "ins2", "ins3", "ins4", "null"] + ([f"{n}f" for n in ("ins2", "ins3", "ins4", "null")] if a.fix else [])
    rows = {n: [] for n in names}
    with torch.no_grad():
        for n in names:
            apply(n if n != "real" else "plus")
            for c in CAMS:
                (out / n / "cams" / c).mkdir(parents=True, exist_ok=True)
            for t in frames:
                rec_f = {}
                for ci, c in enumerate(CAMS):
                    ii, cc = view(t, ci)
                    im = to8(ii["pixels"]) if n == "real" else to8(tr(ii, cc)["rgb"])
                    fn = f"{2 * t:07d}.jpg"
                    Image.fromarray(im).save(out / n / "cams" / c / fn, quality=95)
                    rec_f[c] = f"cams/{c}/{fn}"
                rows[n].append({"frame": 2 * t, "t": t / HZ, "files": rec_f})
            (out / n / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[n]))
    apply("plus")
    if a.fix:
        diag.update(post_fix(a, out, variants, frames, j, dk, T0, p0, ground, view, apply, tr, ps, to8, DSM))
    # inserted donor geometry per frame (distance ahead and lateral on the logged path) for the captions
    geo = {}
    for name, g in variants.items():
        for t in frames:
            pn = g["Rz"] @ (T0[t, j] - p0) + g["off"]
            S, L = path.locate(pn[:2])
            geo[f"{name}/{t}"] = (float(S - path.Se[t]), L)
    meta = {"scene": key, "segment": tg["segment"], "ckpt": ckpt, "donors": report, "chosen": rec, "t_star": ts, "frames": frames,
            "variants": {k: {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in g.items()} for k, g in variants.items()},
            "ego_v_t_star": float(path.v[ts]), "diagnostics": diag, "render_s": round(time.time() - t_start, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    # review clip: log | x- | null  /  TTR 2 | TTR 3 | TTR 4
    ttf = Path(sys.executable).parents[1] / "lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
    import imageio.v2 as imageio
    grid_names = [("real", "log"), ("minus", "x- (donor hidden)"), ("null", "null: donor 4.5 m off the lane"),
                  ("ins2", "x+ TTR 2 s"), ("ins3", "x+ TTR 3 s"), ("ins4", "x+ TTR 4 s")]
    wr = imageio.get_writer(a.out / f"{key}.mp4", fps=HZ, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    small = []
    for i, t in enumerate(frames):
        tiles = []
        for n, title in grid_names:
            im = Image.open(out / n / "cams/front" / f"{2 * t:07d}.jpg").resize((640, 427), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            d.text((8, 6), title, fill=(255, 255, 255), font=ImageFont.truetype(str(ttf), 20), stroke_width=2, stroke_fill=(0, 0, 0))
            if n in variants:
                s, l = geo[f"{n}/{t}"]
                d.text((8, 400), f"d {s:.1f} m, lat {l:+.1f} m", fill=(255, 255, 255), font=ImageFont.truetype(str(ttf), 17),
                       stroke_width=2, stroke_fill=(0, 0, 0))
            tiles.append(im)
        G = Image.new("RGB", (1920, 854 + 40), (20, 20, 20))
        for k, im in enumerate(tiles):
            G.paste(im, ((k % 3) * 640, (k // 3) * 427))
        ImageDraw.Draw(G).text((10, 862), f"{key}  insertion smoke   t - t* = {(t - ts) / HZ:+.1f} s   ego {path.v[t]:.1f} m/s   "
                                            f"donor {rec['donor'][:8]} (box PSNR {rec['psnr_box']:.1f} dB, azimuth gap {rec['az_worst_deg']:.0f} deg)",
                               fill=(230, 230, 230), font=ImageFont.truetype(str(ttf), 20))
        wr.append_data(np.asarray(G))
        if i % 2 == 0:
            small.append(G.resize((960, 447), Image.LANCZOS))
    wr.close()
    small[0].save(a.out / f"{key}.webp", save_all=True, append_images=small[1:], duration=200, loop=0, quality=80, method=6)
    if a.fix:
        compare_clip(a, out, key, frames, ts, variants, diag, ttf)
    print(json.dumps({"scene": key, "chosen": rec, "diagnostics": diag, "render_s": meta["render_s"]}, default=float))


if __name__ == "__main__":
    main()
