"""Insertion-tool comparison (research/insertion-options.md): one inserted pedestrian and one inserted vehicle in the
same OmniRe scene, harmonised by each open tool, on identical frames, measured with one metric table.

Options (front camera at the reconstruction's 960 x 640, the 49 frames t* - 2.9 s .. t* + 1.9 s at 10 Hz; TTR 3 s):
  ped_omni   OmniRe rigid move of a reconstructed pedestrian (scripts/p3/insert.py, variant ins3)
  ped_fix    the P3 manual fix (insert.py --fix, variant ins3f: feet on the LiDAR ground, exposure gain, contact blob)
  ped_r3d2   R3D2 (zenseact/R3D2, one-step SD-Turbo harmoniser trained on Waymo FRONT 3DGS renders) on ped_omni
  ped_vace   Wan2.1-VACE-14B masked video-to-video on ped_omni
  veh_omni   a 3DRealCar Gaussian asset (HUGSIM release) rendered with gsplat into x+ as a static car in the ego lane
             (TTC 3 s at t*), wheels on the LiDAR ground, depth-composited against the OmniRe depth
  veh_r3d2, veh_vace   the same tools on veh_omni
Every tool output is pasted back into its input inside the edit region only (feathered), so outside the region each
option is its base render (x- for the pedestrian, x+ for the vehicle) pixel for pixel. The edit region per frame is the
inserted actor's box widened by 0.8 h each side, 0.45 h below and 0.1 h above (h = box height in pixels).

Steps (each in its own env; <run> = $DATA_DIR/runs/nq4/p3/ins_tools/<key>):
  geom     (drivestudio env, CPU)  camera, ground contact points, edit regions             -> geom.npz
  veh      (drivestudio env, GPU)  x+ colour + depth, asset render, composite              -> opts/veh_{base,omni}
  r3d2     (r3d2 env, GPU)         --src ped_omni|veh_omni                                  -> opts/<cls>_r3d2{,_raw}
  vace     (vace env, GPU)         --src ped_omni|veh_omni                                  -> opts/<cls>_vace{,_raw}
  masks    (sam3 env, GPU)         SAM 3.1 actor masks on every option                      -> masks/<opt>.npz
  metrics  (any env, CPU)                                                                   -> metrics.csv / .json
  clip     (any env, CPU)          side-by-side WebP per class                              -> <cls>.webp / .mp4
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
REPO = Path(__file__).resolve().parents[2]
HZ, W, H = 10, 960, 640
PRE, POST = 29, 19                                   # 49 frames = 4 * 12 + 1 (Wan's frame grid)
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
LABELS = {"ped_omni": "(a) OmniRe as is", "ped_fix": "(b) manual fix", "ped_r3d2": "(c) R3D2", "ped_vace": "(d) VACE-14B",
          "veh_omni": "(a) 3DGS asset as is", "veh_r3d2": "(c) R3D2", "veh_vace": "(d) VACE-14B"}
PROMPT = {"ped": "Dashcam video of a sunny street. A pedestrian walks across the asphalt road in front of the car, "
                 "lit by the same sunlight as the scene, feet on the ground, casting a soft shadow on the road.",
          "veh": "Dashcam video of a sunny street. A car stands in the lane ahead on the asphalt road, lit by the same "
                 "sunlight as the scene, wheels on the ground, casting a shadow on the road under and beside it."}


def run_dir(key):
    return DATA / "runs/nq4/p3/ins_tools" / key


def src_dir(key):
    return DATA / "runs/nq4/p3/insert_fix" / key


def jpg(t):
    return f"{2 * t:07d}.jpg"


def load_rgb(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


def save_rgb(p, a):
    from PIL import Image
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8) if a.dtype != np.uint8 else a).save(p, quality=95)


def opt_frames(key, opt):
    return run_dir(key) / "opts" / opt


def frames_of(meta):
    ts = meta["t_star"]
    return list(range(ts - PRE, ts + POST + 1))


def box_region(box, hh=None):
    """Edit region (x0, y0, x1, y1) around an actor box."""
    x0, y0, x1, y1 = box
    h = hh or (y1 - y0)
    return (max(int(x0 - 0.8 * h), 0), max(int(y0 - 0.1 * h), 0), min(int(x1 + 0.8 * h) + 1, W), min(int(y1 + 0.45 * h) + 1, H))


def soft_region(reg, feather=4.0):
    """Feathered [0, 1] weight of an edit region (1 inside, fading over ~2 feather px at the border)."""
    from scipy.ndimage import gaussian_filter
    m = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = reg
    m[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = 1
    return np.clip(gaussian_filter(m, feather) * 1.0, 0, 1)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return None if len(ys) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def project(K, c2w, P):
    """World points (n, 3) -> pixels (n, 2) and depths (n,)."""
    w2c = np.linalg.inv(c2w)
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:], pc[:, 2]


# ------------------------------------------------------------------------------------------------ geometry (CPU)

def scene_geometry(key, meta):
    """Front-camera K (at 960 x 640) and c2w per frame in the reconstruction's world frame, and a LiDAR ground lookup."""
    kk = key.split("_")[1]
    proc = DATA / "processed/waymo_ds/training" / kk
    f = np.loadtxt(proc / "intrinsics/0.txt")
    s = W / 1920.0
    K = np.array([[f[0] * s, 0, f[2] * s], [0, f[1] * s, f[3] * s], [0, 0, 1]])
    E0 = np.linalg.inv(np.loadtxt(proc / "ego_pose/000.txt"))
    nf = len(list((proc / "ego_pose").glob("*.txt")))
    E = np.stack([E0 @ np.loadtxt(proc / "ego_pose" / f"{t:03d}.txt") for t in range(nf)])
    c2e = np.loadtxt(proc / "extrinsics/0.txt") @ OPENCV2DATASET
    frames = frames_of(meta)
    gp = []
    for t in range(max(frames[0] - 30, 0), min(frames[-1] + 60, nf), 3):
        L = np.fromfile(proc / "lidar" / f"{t:03d}.bin", dtype=np.float32).reshape(-1, 14)
        L = L[L[:, 10] == 1][:, 3:6].astype(np.float64)       # drivestudio ground label, points in the ego frame
        gp.append(L @ E[t][:3, :3].T + E[t][:3, 3])
    gp = np.concatenate(gp)

    def ground(xy, r0=0.8):
        d = np.linalg.norm(gp[:, :2] - np.asarray(xy)[:2], axis=1)
        for r in (r0, 2.0, 4.0):
            if (d <= r).sum() >= 5:
                return float(np.median(gp[d <= r, 2]))
        return float("nan")

    return K, E, c2e, ground, proc


def geom(a):
    """Edit regions and ground contact points of the inserted pedestrian (ins3 and ins3f, the same donor path)."""
    import torch
    key = a.key
    meta = json.loads((src_dir(key) / "meta.json").read_text())
    K, E, c2e, ground, proc = scene_geometry(key, meta)
    frames = frames_of(meta)
    ck = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)["models"]["DeformableNodes"]
    j, ts = meta["chosen"]["node"], meta["t_star"]
    T0 = ck["instances_trans"][:, j].double().numpy()
    q = torch.nn.functional.normalize(ck["instances_quats"][:, j].double(), dim=-1).numpy()   # wxyz
    w_, x_, y_, z_ = q.T
    R0 = np.stack([np.stack([1 - 2 * (y_ ** 2 + z_ ** 2), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)], -1),
                   np.stack([2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ ** 2 + z_ ** 2), 2 * (y_ * z_ - x_ * w_)], -1),
                   np.stack([2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ ** 2 + y_ ** 2)], -1)], 1)
    Mloc = ck["_means"][ck["points_ids"][:, 0] == j].double().numpy()
    p0 = T0[ts]
    out = {"frames": np.array(frames), "K": K, "c2w": np.stack([E[t] @ c2e for t in frames])}
    for name in ("ins3", "ins3f"):
        g = meta["variants"][name]
        Rz, off = np.array(g["Rz"]), np.array(g["off"])
        dzs = {int(k): v for k, v in g.get("dzs", {}).items()}
        P, feet, gz = [], [], []
        for t in frames:
            pn = Rz @ (T0[t] - p0) + off
            pn[2] += dzs.get(t, 0.0)
            P.append(pn)
            feet.append(np.percentile((Mloc @ R0[t].T)[:, 2], 1) + pn[2])
            gz.append(ground(pn[:2]))
        P, feet, gz = np.array(P), np.array(feet), np.array(gz)
        cp = np.array([project(K, E[t] @ c2e, np.array([[*P[i, :2], gz[i]]]))[0][0] for i, t in enumerate(frames)])
        out[f"{name}_pos"], out[f"{name}_feet_z"], out[f"{name}_ground_z"], out[f"{name}_contact_px"] = P, feet, gz, cp
    # actor pixels (x+ vs x- difference, the P3 fix's threshold) -> boxes -> edit regions (union of ins3 and ins3f)
    regs, boxes = [], []
    for t in frames:
        mi = load_rgb(src_dir(key) / "minus/cams/front" / jpg(t)).astype(int)
        m = np.zeros((H, W), bool)
        for name in ("ins3", "ins3f"):
            m |= np.abs(load_rgb(src_dir(key) / name / "cams/front" / jpg(t)).astype(int) - mi).max(-1) > 15
        b = bbox(m)
        boxes.append(b or (0, 0, 0, 0))
        regs.append(box_region(b) if b else (0, 0, 0, 0))
    out["ped_box"], out["ped_region"] = np.array(boxes), np.array(regs)
    rd = run_dir(key)
    rd.mkdir(parents=True, exist_ok=True)
    old = dict(np.load(rd / "geom.npz")) if (rd / "geom.npz").exists() else {}
    np.savez(rd / "geom.npz", **{**old, **out})
    # the pedestrian options that come straight from the insert run
    for opt, name in (("ped_base", "minus"), ("ped_omni", "ins3"), ("ped_fix", "ins3f"), ("real", "real")):
        d = opt_frames(key, opt)
        d.mkdir(parents=True, exist_ok=True)
        for t in frames:
            (d / jpg(t)).unlink(missing_ok=True)
            os.link(src_dir(key) / name / "cams/front" / jpg(t), d / jpg(t))
    print(json.dumps({"key": key, "frames": [frames[0], frames[-1]], "ped_box_h_px": [int(np.median([b[3] - b[1] for b in boxes]))],
                      "foot_minus_ground_m": {n: float(np.nanmedian(out[f"{n}_feet_z"] - out[f"{n}_ground_z"])) for n in ("ins3", "ins3f")}}))


# ------------------------------------------------------------------------------------------------ vehicle (GPU)

def veh(a):
    """Static 3DRealCar asset in the ego lane, rear towards the ego, TTC 3 s at t*, composited into x+ by depth."""
    import torch
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(REPO / "scripts/p3"))
    import ds as DSM
    from insert import Path2D
    from gsplat import rasterization, spherical_harmonics
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
    t0 = time.time()
    key = a.key
    meta = json.loads((src_dir(key) / "meta.json").read_text())
    K, E, c2e, ground, proc = scene_geometry(key, meta)
    frames, ts = frames_of(meta), meta["t_star"]
    path = Path2D(E)
    root = DATA / "datasets/hugsim/3DRealCar"
    aid = a.asset or min(os.listdir(root), key=lambda s: zlib.crc32(s.encode()))
    g = torch.load(root / aid / "gs.pth", map_location="cpu", weights_only=False)[0]
    w, l, h = json.loads((root / aid / "wlh.json").read_text())
    xyz = g[1].double()
    op = torch.sigmoid(g[7][:, 0].double())
    keep = op > 0.005
    # asset frame: x along the car, y down (ground at y = 0), z across -> ours: x forward, y left, z up
    M = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float64)
    if a.flip:
        M = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)) @ M
    loc = xyz @ M.T
    bottom = float(torch.quantile(loc[op > 0.3, 2], 0.002))
    S = path.Se[ts] + 4.0 + 3.0 * max(path.v[ts], 3.0) + l / 2
    Q, th, _ = path.at(S)
    Rw = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    corners = np.array([[sx * l / 2, sy * w / 2] for sx in (-1, 1) for sy in (-1, 1)]) @ Rw[:2, :2].T + Q
    zg = float(np.nanmedian([ground(c, 1.0) for c in [*corners, Q]]))
    Rt = torch.tensor(Rw, dtype=torch.float64) @ M
    tvec = torch.tensor([Q[0], Q[1], zg - bottom], dtype=torch.float64)
    dev = "cuda"
    means = (xyz @ Rt.T + tvec)[keep].float().to(dev)
    qr = matrix_to_quaternion(Rt[None].float())[0]
    quats = quaternion_multiply(qr[None], torch.nn.functional.normalize(g[6].float(), dim=-1))[keep].to(dev)
    scales = torch.exp(g[5].float())[keep].to(dev)
    opac = op[keep].float().to(dev)
    coeffs = torch.cat([g[2], g[3]], 1).float()[keep].to(dev)
    Rt_d = Rt.float().to(dev)
    # x+ colour and depth from the reconstruction
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = DSM._load(Path(meta["ckpt"]).parent, meta["ckpt"])
    base_d, omni_d = opt_frames(key, "veh_base"), opt_frames(key, "veh_omni")
    boxes, regs, cps = [], [], []
    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()  # noqa: E731
    with torch.no_grad():
        for t in frames:
            ii, cc = ds.full_image_set.get_image(t * ps.num_cams + 0, 1)
            ii = {k: v.cuda() if torch.is_tensor(v) else v for k, v in ii.items()}
            cc = {k: v.cuda() if torch.is_tensor(v) else v for k, v in cc.items()}
            o = tr(ii, cc)
            rgb, depth = o["rgb"], o["depth"].reshape(H, W)
            c2w = cc["camera_to_world"].float()
            Kc = cc["intrinsics"].float()
            dirs = means - c2w[:3, 3]
            col = spherical_harmonics(3, (dirs @ Rt_d), coeffs)          # view direction in the asset frame
            col = torch.clamp(col + 0.5, min=0.0)
            rc, ra, _ = rasterization(means, quats, scales, opac, col, torch.linalg.inv(c2w)[None], Kc[None], W, H,
                                      render_mode="RGB+ED", packed=False)
            crgb, cdep, calpha = rc[0, ..., :3], rc[0, ..., 3], ra[0, ..., 0]
            vis = (cdep < depth - 0.05) | (depth <= 0)
            al = (calpha * vis).clamp(0, 1)[..., None]
            comp = al * crgb.clamp(0, 1) + (1 - al) * rgb
            save_rgb(base_d / jpg(t), to8(rgb))
            save_rgb(omni_d / jpg(t), to8(comp))
            b = bbox((calpha > 0.05).cpu().numpy())
            boxes.append(b or (0, 0, 0, 0))
            regs.append(box_region(b) if b else (0, 0, 0, 0))
            c2w_np = c2w.double().cpu().numpy()
            pts = np.c_[corners, np.full(4, zg)]
            uv, z = project(K, c2w_np, pts)
            cps.append(uv[np.argmax(uv[:, 1])])                         # nearest bottom corner = lowest contact row
    rd = run_dir(key)
    old = dict(np.load(rd / "geom.npz"))
    old.update({"veh_box": np.array(boxes), "veh_region": np.array(regs), "veh_contact_px": np.array(cps),
                "veh_pose": np.r_[Q, zg, th], "veh_wlh": np.array([w, l, h])})
    np.savez(rd / "geom.npz", **old)
    info = {"asset": aid, "wlh": [w, l, h], "S_ahead_m": float(S - path.Se[ts]), "ego_v_t_star": float(path.v[ts]),
            "ground_z": zg, "flip": a.flip, "wall_s": round(time.time() - t0, 1)}
    (rd / "veh.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


# ------------------------------------------------------------------------------------------------ tools (GPU)

def _regions(key, cls):
    g = np.load(run_dir(key) / "geom.npz")
    return g["frames"], g[f"{cls}_region"], g[f"{cls}_box"]


def _paste(inp, out, reg):
    w = soft_region(tuple(reg))[..., None]
    return inp.astype(np.float32) * (1 - w) + out.astype(np.float32) * w


def r3d2(a):
    """R3D2 per frame on the 2x-upsampled render (1920 x 1280), a 1080-row window that holds the edit regions."""
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    key, src = a.key, a.src
    cls = src.split("_")[0]
    frames, regs, _ = _regions(key, cls)
    mp = str(DATA / "models/r3d2" / a.model)
    pipe = DiffusionPipeline.from_pretrained(mp, custom_pipeline=mp, torch_dtype=torch.float16 if a.fp16 else torch.float32)
    pipe.to("cuda")
    yc = int(np.median([(r[1] + r[3]) for r in regs]))            # 2x the region centre row
    y0 = int(np.clip(yc - 540, 0, 2 * H - 1080))
    name = f"{cls}_r3d2" + ("" if a.model == "R3D2" else "big")
    od, rd_ = opt_frames(key, name), opt_frames(key, name + "_raw")
    tt = []
    for t, reg in zip(frames, regs):
        inp = load_rgb(opt_frames(key, src) / jpg(int(t)))
        up = np.asarray(Image.fromarray(inp).resize((2 * W, 2 * H), Image.BICUBIC))
        x = to_tensor(up[y0:y0 + 1080]).to("cuda", torch.float16 if a.fp16 else torch.float32)
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            y = pipe(x).images[0]
        torch.cuda.synchronize()
        tt.append(time.time() - t1)
        full = up.copy()
        full[y0:y0 + 1080] = np.asarray(y.convert("RGB").resize((2 * W, 1080), Image.BICUBIC))
        raw = np.asarray(Image.fromarray(full).resize((W, H), Image.BICUBIC))
        save_rgb(rd_ / jpg(int(t)), raw)
        save_rgb(od / jpg(int(t)), _paste(inp, raw, reg))
    info = {"model": a.model, "fp16": a.fp16, "window_rows_2x": [y0, y0 + 1080], "s_per_frame_median": float(np.median(tt[2:])),
            "frames": len(tt), "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    (run_dir(key) / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def vace(a):
    """Wan2.1-VACE-14B at 480 x 832 on a fixed crop that holds every edit region; the model repaints the (binary) edit
    region of the composite, which it also sees (VACE's reactive frames), everything else is its inactive context."""
    import torch
    sys.path[:0] = [str(DATA / "third_party/ins/VACE"), str(DATA / "third_party/ins/VACE/vace")]
    from models.wan import WanVace
    from models.wan.configs import WAN_CONFIGS
    key, src = a.key, a.src
    cls = src.split("_")[0]
    frames, regs, _ = _regions(key, cls)
    cw, ch = 832, 480
    cx = int(np.median([(r[0] + r[2]) / 2 for r in regs]))
    cy = int(np.median([(r[1] + r[3]) / 2 for r in regs]))
    x0, y0 = int(np.clip(cx - cw // 2, 0, W - cw)), int(np.clip(cy - ch // 2, 0, H - ch))
    inp = np.stack([load_rgb(opt_frames(key, src) / jpg(int(t))) for t in frames])
    msk = np.zeros((len(frames), H, W), np.float32)
    for i, r in enumerate(regs):
        msk[i, r[1]:r[3], r[0]:r[2]] = 1
    vid = torch.from_numpy(inp[:, y0:y0 + ch, x0:x0 + cw]).permute(3, 0, 1, 2).float().div(127.5).sub(1).cuda()
    mk = torch.from_numpy(msk[:, y0:y0 + ch, x0:x0 + cw])[None].cuda()
    t0 = time.time()
    model = WanVace(config=WAN_CONFIGS["vace-14B"], checkpoint_dir=str(DATA / "models/vace/Wan2.1-VACE-14B"), device_id=0, rank=0,
                    t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    t_load = time.time() - t0
    torch.cuda.synchronize()
    t1 = time.time()
    out = model.generate(PROMPT[cls], [vid], [mk], [None], size=(cw, ch), frame_num=len(frames), shift=a.shift,
                         sample_solver="unipc", sampling_steps=a.steps, guide_scale=5.0, seed=a.seed, offload_model=False)
    torch.cuda.synchronize()
    t_gen = time.time() - t1
    out = ((out.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).cpu().numpy()     # (F, ch, cw, 3)
    name = f"{cls}_vace"
    od, rd_ = opt_frames(key, name), opt_frames(key, name + "_raw")
    for i, (t, reg) in enumerate(zip(frames, regs)):
        raw = inp[i].astype(np.float32).copy()
        raw[y0:y0 + ch, x0:x0 + cw] = out[i]
        save_rgb(rd_ / jpg(int(t)), raw)
        save_rgb(od / jpg(int(t)), _paste(inp[i], raw, reg))
    info = {"crop_xywh": [x0, y0, cw, ch], "steps": a.steps, "shift": a.shift, "seed": a.seed, "load_s": round(t_load, 1),
            "generate_s": round(t_gen, 1), "frames": len(frames), "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    (run_dir(key) / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


# ------------------------------------------------------------------------------------------------ actor masks (GPU)

def masks(a):
    """SAM 3.1 on a square crop around each frame's actor box; the instance with the best IoU to the rendered actor box."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO))
    from jevdrive.sam_detect import Detector, build
    model, _ = build()
    det = {c: Detector(model, prompts=(p,)) for c, p in (("ped", "person"), ("veh", "car"))}
    rd = run_dir(a.key)
    (rd / "masks").mkdir(exist_ok=True)
    opts = a.opts or sorted(p.name for p in (rd / "opts").iterdir() if p.name.split("_")[0] in ("ped", "veh")
                            and not p.name.endswith("_base"))
    for opt in opts:
        cls = opt.split("_")[0]
        frames, regs, boxes = _regions(a.key, cls)
        res = np.zeros((len(frames), H, W), bool)
        score = np.full(len(frames), np.nan)
        for i, (t, b) in enumerate(zip(frames, boxes)):
            if b[2] <= b[0]:
                continue
            im = torch.from_numpy(load_rgb(opt_frames(a.key, opt) / jpg(int(t)))).permute(2, 0, 1).cuda()
            s = int(max(b[2] - b[0], b[3] - b[1]) * 2.2) + 32
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            x0, y0 = int(np.clip(cx - s // 2, 0, max(W - s, 0))), int(np.clip(cy - s // 2, 0, max(H - s, 0)))
            crop = im[:, y0:y0 + s, x0:x0 + s]
            r = det[cls]([crop], keep=0.2)[0][0]
            if len(r["scores"]) == 0:
                continue
            ref = torch.tensor([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], dtype=torch.float32, device="cuda")
            bx = r["boxes"]
            iw = (torch.minimum(bx[:, 2], ref[2]) - torch.maximum(bx[:, 0], ref[0])).clamp(min=0)
            ih = (torch.minimum(bx[:, 3], ref[3]) - torch.maximum(bx[:, 1], ref[1])).clamp(min=0)
            inter = iw * ih
            iou = inter / ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) + (ref[2] - ref[0]) * (ref[3] - ref[1]) - inter)
            k = int(iou.argmax())
            if iou[k] < 0.3:
                continue
            m = r["masks"][k].cpu().numpy()
            res[i, y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
            score[i] = float(r["scores"][k])
        np.savez_compressed(rd / "masks" / f"{opt}.npz", mask=np.packbits(res, axis=-1), score=score, shape=np.array(res.shape))
        print(opt, "frames with a mask:", int(np.isfinite(score).sum()), "/", len(frames))


def load_mask(rd, opt):
    z = np.load(rd / "masks" / f"{opt}.npz")
    n, h, w = z["shape"]
    return np.unpackbits(z["mask"], axis=-1)[..., :w].astype(bool).reshape(n, h, w), z["score"]


# ------------------------------------------------------------------------------------------------ metrics (CPU)

def _lum(im):
    return im[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _lab(im):
    from skimage.color import rgb2lab
    return rgb2lab(im.astype(np.float64) / 255.0)


def _psnr(x, y, m):
    e = ((x.astype(np.float64) - y.astype(np.float64)) ** 2)[m].mean()
    return float(10 * np.log10(255.0 ** 2 / max(e, 1e-10)))


def _ring(mask):
    from scipy.ndimage import binary_dilation
    b = bbox(mask)
    hh, ww = b[3] - b[1], b[2] - b[0]
    r = np.zeros_like(mask)
    r[b[1]:min(b[3] + int(0.3 * hh) + 1, H), max(b[0] - ww // 2, 0):min(b[2] + ww // 2 + 1, W)] = True
    return r & ~binary_dilation(mask, iterations=3)


def metrics(a):
    """Per option: ground-contact offset, shadow, colour, flicker, and untouched-region PSNR against the log."""
    import csv
    from scipy.ndimage import binary_dilation
    key = a.key
    rd = run_dir(key)
    g = np.load(rd / "geom.npz")
    frames, K, C2W = g["frames"], g["K"], g["c2w"]
    real = np.stack([load_rgb(opt_frames(key, "real") / jpg(int(t))) for t in frames])
    rows = []
    # colour reference for the pedestrian: the donor in the log at its original place (x+ vs x- in the insert run)
    ref_lab, ref_ratio = [], []
    for t in frames:
        pl = load_rgb(src_dir(key) / "plus/cams/front" / jpg(int(t))).astype(int)
        mi = load_rgb(src_dir(key) / "minus/cams/front" / jpg(int(t))).astype(int)
        m = np.abs(pl - mi).max(-1) > 15
        if m.sum() >= 150:
            rl = real[list(frames).index(t)]
            ref_lab.append(_lab(rl)[m].mean(0))
            ref_ratio.append(np.median(_lum(rl)[m]) / max(np.median(_lum(mi)[_ring(m)]), 1.0))
    ref = {"ped": (np.mean(ref_lab, 0) if ref_lab else None, float(np.median(ref_ratio)) if ref_ratio else float("nan"))}
    for opt in sorted(p.name for p in (rd / "opts").iterdir()):
        cls = opt.split("_")[0]
        if cls not in ("ped", "veh") or opt.endswith("_base") or opt.endswith("_raw"):
            continue
        _, regs, boxes = _regions(key, cls)
        base = np.stack([load_rgb(opt_frames(key, f"{cls}_base") / jpg(int(t))) for t in frames])
        im = np.stack([load_rgb(opt_frames(key, opt) / jpg(int(t))) for t in frames])
        rawd = opt_frames(key, opt + "_raw")
        raw = np.stack([load_rgb(rawd / jpg(int(t))) for t in frames]) if rawd.exists() else im
        M, _ = load_mask(rd, opt)
        if cls == "ped":
            cp = g["ins3f_contact_px"] if opt == "ped_fix" else g["ins3_contact_px"]
        else:
            cp = g["veh_contact_px"]
        foot_px, foot_m, shadow, dark, lum_a, lum_s, ratio, labs = [], [], [], [], [], [], [], []
        outside = np.ones((len(frames), H, W), bool)
        for i, t in enumerate(frames):
            r = regs[i]
            outside[i, r[1]:r[3], r[0]:r[2]] = False
            m = M[i]
            if m.sum() < 30:
                for L_ in (foot_px, foot_m, shadow, dark, lum_a, lum_s, ratio):
                    L_.append(np.nan)
                labs.append(np.full(3, np.nan))
                continue
            low = np.nonzero(m.any(1))[0].max() + 0.5
            u = float(np.nonzero(m[int(low)])[0].mean())
            # contact row: the actor's ground point in the image; its depth converts pixels to metres
            v_g = cp[i][1]
            foot_px.append(v_g - low)                            # > 0: the lowest actor pixel is above the contact line
            if cls == "ped":
                P = g["ins3f_pos"][i] if opt == "ped_fix" else g["ins3_pos"][i]
                zc = project(K, C2W[i], P[None])[1][0]
            else:
                zc = project(K, C2W[i], g["veh_pose"][None, :3])[1][0]
            foot_m.append((v_g - low) * zc / K[1, 1])
            reg = np.zeros((H, W), bool)
            reg[r[1]:r[3], r[0]:r[2]] = True
            ring = reg & ~binary_dilation(m, iterations=3)
            ring[: int(low - 0.3 * (low - np.nonzero(m.any(1))[0].min()))] = False     # the lower part of the region only
            q = _lum(im[i]) / np.maximum(_lum(base[i]), 1.0)
            sh = ring & (q < 0.8)
            shadow.append(sh.sum() / m.sum())
            dark.append(float(1 - np.median(q[sh])) if sh.sum() >= 10 else 0.0)
            lum_a.append(_lum(im[i])[m].mean())
            lum_s.append(_lum(im[i])[ring].mean() - _lum(base[i])[ring].mean())
            ratio.append(np.median(_lum(im[i])[m]) / max(np.median(_lum(base[i])[_ring(m)]), 1.0))
            labs.append(_lab(im[i])[m].mean(0))
        la, ls = np.array(lum_a), np.array(lum_s)
        ok = np.isfinite(la)
        labs = np.array(labs)
        row = {"option": opt, "label": LABELS.get(opt, opt), "frames_masked": int(ok.sum()),
               "foot_offset_px_median": float(np.nanmedian(foot_px)), "foot_offset_m_median": float(np.nanmedian(foot_m)),
               "foot_offset_m_abs_p90": float(np.nanpercentile(np.abs(foot_m), 90)),
               "shadow_area_ratio_median": float(np.nanmedian(shadow)), "shadow_darkening_median": float(np.nanmedian(dark)),
               "shadow_frames_pct": float(100 * np.nanmean(np.array(shadow)[ok] >= 0.1)),
               "actor_ratio_median": float(np.nanmedian(ratio)),
               "flicker_actor_pct": float(100 * np.nanstd(np.diff(la[ok])) / np.nanmean(la[ok])) if ok.sum() > 3 else np.nan,
               "flicker_shadow_zone": float(np.nanstd(np.diff(ls[ok]))) if ok.sum() > 3 else np.nan,
               "psnr_outside_vs_log": _psnr(im, real, outside), "psnr_outside_raw_vs_log": _psnr(raw, real, outside),
               "psnr_outside_raw_vs_base": _psnr(raw, base, outside), "psnr_outside_base_vs_log": _psnr(base, real, outside)}
        if cls == "ped" and ref["ped"][0] is not None:
            row["ratio_err_vs_log_donor"] = float(abs(np.log(row["actor_ratio_median"] / ref["ped"][1])))
            row["deltaE_vs_log_donor"] = float(np.linalg.norm(np.nanmean(labs[ok], 0) - ref["ped"][0]))
        rows.append(row)
    keys_ = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(rd / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_)
        w.writeheader()
        w.writerows(rows)
    (rd / "metrics.json").write_text(json.dumps({"rows": rows, "ped_ref_ratio": ref["ped"][1]}, indent=1, default=float))
    for r in rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))


# ------------------------------------------------------------------------------------------------ clips (CPU)

def clip(a):
    """One row per option: the full front view (left) and a 2.5x crop around the actor (right); same crop and caption
    for every option; 5 fps WebP for the doc and a 10 fps MP4."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    key, cls = a.key, a.cls
    rd = run_dir(key)
    frames, regs, boxes = _regions(key, cls)
    opts = a.opts.split(",")
    meta = json.loads((src_dir(key) / "meta.json").read_text())
    ts = meta["t_star"]
    ttf = next(Path(p) for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                 str(DATA / "envs/drivestudio/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"))
               if Path(p).exists())
    f1 = ImageFont.truetype(str(ttf), 17)
    tw, th_ = 480, 320
    cw_, ch_ = 192, 128                                          # crop (x2.5 to 480 x 320)
    cxs = [(b[0] + b[2]) / 2 for b in boxes if b[2] > b[0]]
    cys = [b[3] for b in boxes if b[2] > b[0]]
    wr = imageio.get_writer(rd / f"{cls}.mp4", fps=HZ, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    small = []
    for i, t in enumerate(frames):
        b = boxes[i]
        cx = (b[0] + b[2]) / 2 if b[2] > b[0] else np.median(cxs)
        cy = b[3] - 0.35 * (b[3] - b[1]) if b[2] > b[0] else np.median(cys)
        x0, y0 = int(np.clip(cx - cw_ / 2, 0, W - cw_)), int(np.clip(cy - ch_ / 2, 0, H - ch_))
        G = Image.new("RGB", (2 * tw, len(opts) * th_ + 30), (20, 20, 20))
        d = ImageDraw.Draw(G)
        for k, opt in enumerate(opts):
            im = Image.open(opt_frames(key, opt) / jpg(int(t)))
            G.paste(im.resize((tw, th_), Image.LANCZOS), (0, k * th_))
            G.paste(im.crop((x0, y0, x0 + cw_, y0 + ch_)).resize((tw, th_), Image.LANCZOS), (tw, k * th_))
            d.rectangle([x0 / 2, k * th_ + y0 / 2, (x0 + cw_) / 2, k * th_ + (y0 + ch_) / 2], outline=(255, 210, 0))
            d.text((8, k * th_ + 6), LABELS.get(opt, opt), fill=(255, 255, 255), font=f1, stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((8, len(opts) * th_ + 6), f"{key}  {'pedestrian' if cls == 'ped' else 'vehicle'} insertion, TTR 3 s   "
               f"t - t* = {(int(t) - ts) / HZ:+.1f} s   right: x2.5 crop (yellow box)", fill=(230, 230, 230), font=f1)
        arr = np.asarray(G)
        wr.append_data(arr)
        if i % 2 == 0:
            small.append(G.resize((int(G.width * a.scale), int(G.height * a.scale)), Image.LANCZOS))
    wr.close()
    out = Path(a.webp) if a.webp else rd / f"{cls}.webp"
    small[0].save(out, save_all=True, append_images=small[1:], duration=200, loop=0, quality=a.quality, method=6)
    print(out, round(out.stat().st_size / 2 ** 20, 2), "MB")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("geom", "veh", "r3d2", "vace", "masks", "metrics", "clip"):
        p = sp.add_parser(n)
        p.add_argument("--key", default="p3_001")
        if n == "veh":
            p.add_argument("--asset", default=None)
            p.add_argument("--flip", action="store_true", help="turn the asset around (front towards the ego)")
        if n in ("r3d2", "vace"):
            p.add_argument("--src", default="ped_omni")
        if n == "r3d2":
            p.add_argument("--model", default="R3D2")
            p.add_argument("--fp16", action="store_true")
        if n == "vace":
            p.add_argument("--steps", type=int, default=50)
            p.add_argument("--shift", type=float, default=16.0)
            p.add_argument("--seed", type=int, default=2025)
        if n == "masks":
            p.add_argument("--opts", nargs="*")
        if n == "clip":
            p.add_argument("--cls", default="ped")
            p.add_argument("--opts", default="ped_omni,ped_fix,ped_r3d2,ped_vace")
            p.add_argument("--webp", default=None)
            p.add_argument("--scale", type=float, default=0.8)
            p.add_argument("--quality", type=int, default=70)
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
