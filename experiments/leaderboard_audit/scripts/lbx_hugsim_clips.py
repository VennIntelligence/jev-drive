#!/usr/bin/env python
"""Loss-budget examples, HUGSIM 64: paired clips of the shipped Cinque (cinque-fixed, exam run) and the best driver (it_dw3-s0 + sel3)
on the same scenario, from what the recorded runs left on disk. No simulator, no model, CPU only.

Per step (sim step 0.25 s; one GIF frame per step = real time unless marked) and per arm, one panel:
  left   BEV reconstruction (NOT a render): the run's own ground.ply / scene.ply points (grey; dark = scene points in the ego
         height band, i.e. buildings, poles, parked geometry), the recorded route (green), actor boxes from infos.pkl (red),
         the ego box and trail (blue), the openpilot plan of that step (cyan, model_pos from zs_steps.jsonl);
  right  openpilot's road (top) and wide (bottom) model frames, rebuilt from the run's video.mp4 with the exact gather of
         jevdrive.hugsim_zs.OpenpilotFrames (same rays, same source cameras, nearest neighbour), shown in colour (the model gets
         the same pixels as BT.601 YUV); only mp4 compression differs from what the model saw (decision-era replay: 0.02 m
         mean plan difference). The logged plan (model_pos) is drawn on a flat ground at the camera height.

    CUDA_VISIBLE_DEVICES= $DATA_DIR/envs/hugsim/bin/python experiments/leaderboard_audit/scripts/lbx_hugsim_clips.py scan <scenario> ...
    CUDA_VISIBLE_DEVICES= $DATA_DIR/envs/hugsim/bin/python experiments/leaderboard_audit/scripts/lbx_hugsim_clips.py make <out_dir>
"""
import csv
import io
import json
import os
import pickle
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
INP = ROOT / "experiments/leaderboard_audit/results/loss_budget/hugsim_inputs"
CAMYAML = D / "third_party/HUGSIM/configs/sim/{}_camera.yaml"
ARMS = (("shipped cinque-fixed", INP / "exam_scored_op.csv", "cinque-fixed"),
        ("best it_dw3 + sel3", INP / "it_dw3-s0_sel3.csv", None))
DT, PPM, BW, BH = 0.25, 6.0, 384, 512          # sim step s; BEV px per metre, panel size; ego at 75 % height
T_POS = np.array([0.156, 0.625, 1.406, 2.5, 3.906, 5.625, 10.0])   # times of the 7 logged model_pos points (T_IDXS 4..32)

# (file stem, scenario, class label shown on the frame)
CASES = [
    ("hugsim_spin_0013m", "scene-0013-medium-00", "spin"),
    ("hugsim_spin_0528m", "scene-0528-medium-00", "spin"),
    ("hugsim_spin_1522m", "scene-152217047339-medium-00", "spin"),
    ("hugsim_stop_032m02", "scene-032-medium-02", "stopped / max_steps"),
    ("hugsim_stop_1137e", "scene-113792265837-easy-00", "stopped / max_steps"),
    ("hugsim_stop_0411m", "scene-0411-medium-00", "stopped / max_steps"),
    ("hugsim_fg_0138x", "scene-0138-extreme-00", "fg collision"),
    ("hugsim_fg_3400x", "scene-3400_3600-extreme-00", "fg collision"),
    ("hugsim_fg_034h", "scene-034-hard-00", "fg collision"),
]


def run_dirs(scenario):
    out = []
    for label, f, tag in ARMS:
        r = next(r for r in csv.DictReader(open(f)) if r["scenario"] == scenario and (tag is None or r["tag"] == tag))
        out.append((label, Path(r["run_dir"]), r))
    return out


def route(ds, scene):
    p = D / "datasets/hugsim/scenes" / ds / scene / "ground_param.pkl"
    if p.exists():
        cam = pickle.load(open(p, "rb"))[0]
    else:
        with zipfile.ZipFile(D / "datasets/hugsim/scenes" / ds / f"{scene}.zip") as z:
            cam = pickle.load(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("ground_param.pkl")))))[0]
    rt = np.asarray(cam)[:, :3, 3]
    return np.stack([rt[:, 2], -rt[:, 0]], 1), rt          # BEV (fwd, left), raw OpenCV


def load(d):
    infos = pickle.load(open(d / "infos.pkl", "rb"))
    steps = [json.loads(x) for x in open(d / "zs_steps.jsonl")]
    setup, steps = steps[0], {s["step"]: s for s in steps[1:] if "step" in s}
    txt = (d / "sim.log").read_text(errors="replace")
    end = next((e for k, e in (("Collision with background", "bg_collision"), ("Collision with foreground", "fg_collision"),
                               ("Far from preset trajectory", "off_route"), ("Complete", "complete")) if k in txt),
               "max_steps" if len(infos) >= 400 else "other")
    return infos, steps, setup, end


def stop_step(infos, v_eps=0.2, hold=8):
    v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in infos])
    for k in range(len(v) - hold):
        if (v[k:k + hold] < v_eps).all() and k > 4:
            return k
    return None


def ahead(info):
    """Nearest actor in the ego frame: (forward m, left m) of the closest box centre within 3 m laterally and ahead."""
    e = np.asarray(info["ego_box"], float)
    c, s = np.cos(e[6]), np.sin(e[6])
    best = None
    for b in info["obj_boxes"]:
        b = np.asarray(b, float).ravel()
        dx, dy = b[0] - e[0], b[1] - e[1]
        f, l = c * dx + s * dy, -s * dx + c * dy
        if f > 0 and abs(l) < 3 and (best is None or f < best[0]):
            best = (round(f, 1), round(l, 1))
    return best


def scan(scens):
    for sc in scens:
        for label, d, r in run_dirs(sc):
            infos, steps, setup, end = load(d)
            k = stop_step(infos)
            col = next((i for i, x in enumerate(infos) if bool(x.get("collision"))), None)
            v = [round(float(np.ravel(i["ego_velo"])[0]), 1) for i in infos]
            print(sc, label, "steps", len(infos), "end", end, "hd", round(float(r["hdscore"]), 3), "stop@", k,
                  "ahead@stop", ahead(infos[k]) if k else None, "collision@", col, "ahead@end", ahead(infos[-1]),
                  "n_obj", len(infos[0]["obj_boxes"]), "v[::8]", v[::8][:14], flush=True)


# ---------------------------------------------------------------- drawing

def topdown(d, rt_cv, paths, cell=1 / PPM):
    """World raster (fwd, left) in BEV pixels: ground points light, scene points within the ego height band dark."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    allp = np.concatenate(paths)
    lo, hi = allp.min(0) - 60, allp.max(0) + 60
    shape = np.ceil((hi - lo) / cell).astype(int)
    img = np.full((shape[1], shape[0]), 255, np.uint8)          # rows = left axis, cols = fwd axis
    tree = cKDTree(rt_cv[:, [0, 2]])
    for name, val in (("ground", 225), ("scene", 110)):
        p = np.asarray(o3d.io.read_point_cloud(str(d / f"{name}.ply")).points)
        f = np.stack([p[:, 2], -p[:, 0]], 1)
        m = ((f >= lo) & (f < hi)).all(1)
        p, f = p[m], f[m]
        if name == "scene":
            ycam = rt_cv[tree.query(p[:, [0, 2]], workers=4)[1], 1]
            keep = (p[:, 1] > ycam) & (p[:, 1] < ycam + 1.5)
            f = f[keep]
        ij = ((f - lo) / cell).astype(int)
        h = np.zeros(img.shape, np.uint16)
        np.add.at(h, (ij[:, 1], ij[:, 0]), 1)
        img[h >= (1 if name == "ground" else 3)] = val
    return lo, img


def bev_affine(lo, pos, yaw):
    """2x3 map from world raster pixels to the ego-up BEV panel (ego at (BW/2, 0.75 BH), heading up)."""
    c, s = np.cos(yaw), np.sin(yaw)
    # world (f, l) -> ego (x fwd, y left): R^T (w - pos); panel u = BW/2 - y*PPM, v = 0.75 BH - x*PPM
    A = np.array([[s, -c], [-c, -s]]) * PPM             # rows: u, v from (f, l)
    b = np.array([BW / 2, 0.75 * BH]) - A @ pos
    # raster px (i = (f - lo_f) PPM, j = (l - lo_l) PPM) -> world: f = lo + i / PPM
    M = np.zeros((2, 3))
    M[:, :2] = A / PPM
    M[:, 2] = A @ lo + b
    return M, A, b


def poly(img, pts, A, b, col, th=2, closed=False):
    q = (np.asarray(pts, float) @ A.T + b).round().astype(np.int32)
    cv2.polylines(img, [q], closed, col, th, cv2.LINE_AA)


def box_pts(bx):
    bx = np.asarray(bx, float).ravel()
    c, s = np.cos(bx[6]), np.sin(bx[6])
    L, W = bx[4] / 2, bx[3] / 2
    loc = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])
    return bx[:2] + loc @ np.array([[c, s], [-s, c]])


def put(img, txt, xy, scale=0.5, col=(255, 255, 255), bg=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = xy
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 4), bg, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


class Arm:
    def __init__(self, label, d, row, ds):
        self.label, self.d, self.row = label, d, row
        self.infos, self.steps, self.setup, self.end = load(d)
        cap, self.vid = cv2.VideoCapture(str(d / "video.mp4")), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[0] // 2, f.shape[1] // 3
            self.vid.append({"CAM_FRONT_LEFT": f[:h, :w], "CAM_FRONT": f[:h, w:2 * w], "CAM_FRONT_RIGHT": f[:h, 2 * w:]})  # BGR
        cal = Z.calibs(self.infos[0]["cam_params"], Z.rect_matrix(str(CAMYAML).format(ds)))
        self.op = Z.OpenpilotFrames(cal)
        t = np.asarray(self.infos[0]["cam_params"]["CAM_FRONT"]["v2c"], float)[:3, 3]
        h = -(G.OPENCV_TO_VEHICLE @ t)[2]
        self.cam_h = float(h) if 1.0 < h < 2.5 else 1.5
        self.n = len(self.infos)
        self.pos = np.array([np.asarray(i["ego_box"], float)[:2] for i in self.infos])

    def model_frames(self, k):
        rgb = self.vid[min(k, len(self.vid) - 1)]
        cat = np.concatenate([rgb[c].reshape(-1, 3) for c in self.op.cams] + [np.zeros((1, 3), np.uint8)])
        out = {}
        for m in ("road", "wide"):
            im = np.ascontiguousarray(cat[self.op.idx[m]].reshape(G.OP_H, G.OP_W, 3))
            mp = self.steps.get(k, {}).get("model_pos")
            if mp:
                p = np.asarray(mp, float)
                p = np.concatenate([[[0.0, 0.0]], p[p[:, 0] > 0.5]])
                xyz = np.stack([-p[:, 1], np.full(len(p), self.cam_h), np.maximum(p[:, 0], 0.5)], 1)   # OpenCV view frame
                uv = (xyz @ G.OP_K[m].T)
                uv = (uv[:, :2] / uv[:, 2:]).round().astype(np.int32)
                cv2.polylines(im, [uv], False, (255, 255, 0), 2, cv2.LINE_AA)
                for q in uv[1:]:
                    cv2.circle(im, tuple(int(x) for x in q), 3, (255, 255, 0), -1, cv2.LINE_AA)
            put(im, f"openpilot {m} input", (6, 18), 0.45)
            out[m] = im
        return out

    def panel(self, k, lo, world, rt_bev, cls):
        done = k >= self.n
        kk = min(k, self.n - 1)
        I, S = self.infos[kk], self.steps.get(kk, {})
        e = np.asarray(I["ego_box"], float)
        M, A, b = bev_affine(lo, e[:2], e[6])
        bev = cv2.warpAffine(world, M, (BW, BH), flags=cv2.INTER_NEAREST, borderValue=255)
        bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        poly(bev, rt_bev, A, b, (60, 170, 60), 2)
        poly(bev, self.pos[:kk + 1], A, b, (200, 120, 40), 2)
        for bx in I["obj_boxes"]:
            poly(bev, box_pts(bx), A, b, (40, 40, 220), 2, True)
        poly(bev, box_pts(e), A, b, (200, 80, 0), 3, True)
        if S.get("model_pos"):
            p = np.concatenate([[[0.0, 0.0]], np.asarray(S["model_pos"], float)])
            c, s = np.cos(e[6]), np.sin(e[6])
            w = e[:2] + p @ np.array([[c, s], [-s, c]])
            poly(bev, w, A, b, (200, 200, 0), 2)
        cv2.line(bev, (6, BH - 12), (6 + int(10 * PPM), BH - 12), (0, 0, 0), 2)
        put(bev, "10 m", (8, BH - 18), 0.4, (0, 0, 0), (255, 255, 255))
        put(bev, "BEV reconstruction (not a render)", (6, 16), 0.42)
        v = float(np.ravel(I["ego_velo"])[0])
        put(bev, f"{self.label}", (6, 38), 0.5, (255, 255, 0))
        put(bev, f"t {kk * DT:5.2f} s  v {v:4.1f} m/s  rc {float(I.get('rc', 0.0)):.2f}", (6, 58), 0.45)
        if S:
            put(bev, f"cmd {['right', 'left', 'straight'][int(I['command'])]}  lead_prob {S.get('lead_prob', 0):.2f}", (6, 78), 0.45)
        if S.get("derot", {}).get("used"):                 # sel3 took the plan of the derotated-history rollout this step
            put(bev, "sel3: derotated-history plan taken", (6, 122), 0.45, (0, 0, 0), (0, 200, 255))
        hd = float(self.row["hdscore"])
        if bool(I.get("collision")) or done:
            put(bev, f"END {self.end}  HD {hd:.2f}" if done or kk == self.n - 1 else "COLLISION flag", (6, 100), 0.55,
                (255, 255, 255), (0, 0, 200))
        mf = self.model_frames(kk)
        right = np.concatenate([mf["road"], mf["wide"]], 0)
        if done:
            right = (right * 0.45).astype(np.uint8)
            bev = (bev * 0.7 + 76).astype(np.uint8)
            put(bev, f"episode ended at t {(self.n - 1) * DT:.2f} s: {self.end}, HD {hd:.2f}", (6, 100), 0.45,
                (255, 255, 255), (0, 0, 200))
        return np.concatenate([bev, right], 1)


def timeline(arms, ff=8):
    """Step list with speed labels: real time while any running arm moves or within 12 steps of an arm's end / stop onset; xff
    when every running arm stands (v < 0.2) or has ended. Returns [(step, label)]."""
    n = max(a.n for a in arms)
    keep = np.zeros(n + 4, bool)
    for a in arms:
        v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in a.infos])
        keep[:a.n] |= v > 0.2
        for k in (a.n - 1, stop_step(a.infos) or 0):
            keep[max(0, k - 12):k + 13] = True
    keep[:8] = True
    out, k = [], 0
    while k < n + 4:
        if keep[k] or k >= n:
            out.append((k, "real time"))
            k += 1
        else:
            out.append((k, f"x{ff} speed"))
            k += ff
    return out


def make_case(args):
    stem, sc, cls, out_dir = args
    arms_raw = run_dirs(sc)
    ds = arms_raw[0][2]["dataset"]
    scene = Path(arms_raw[0][1]).name.rsplit("_", 2)[0]
    arms = [Arm(l, d, r, ds) for l, d, r in arms_raw]
    rt_bev, rt_cv = route(ds, scene)
    lo, world = topdown(arms[0].d, rt_cv, [a.pos for a in arms] + [rt_bev])
    tl = timeline(arms)
    frames = []
    for k, lab in tl:
        rows = [a.panel(k, lo, world, rt_bev, cls) for a in arms]
        top = np.full((30, rows[0].shape[1], 3), 30, np.uint8)
        put(top, f"HUGSIM {sc}  |  class (shipped): {cls}  |  sim step 0.25 s  |  {lab}", (8, 21), 0.55,
            (255, 255, 255) if lab == "real time" else (0, 255, 255), (30, 30, 30))
        sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
        frames.append(np.concatenate([top, rows[0], sep, rows[1]], 0))
    from PIL import Image
    out = Path(out_dir) / f"{stem}.gif"
    for scale, ncol in ((0.62, 128), (0.55, 96), (0.48, 80), (0.42, 64), (0.36, 64)):     # shrink until < 3 MB
        ims = [Image.fromarray(cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
               for f in frames]
        pal = ims[len(ims) // 2].quantize(colors=ncol, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        q[0].save(out, save_all=True, append_images=q[1:], duration=250, loop=0, optimize=True)
        if out.stat().st_size < 3.0e6:
            break
    # still: the frame at the shipped arm's event (end or stop onset)
    a0 = arms[0]
    ev = stop_step(a0.infos) if a0.end == "max_steps" else a0.n - 1
    i = min(range(len(tl)), key=lambda j: abs(tl[j][0] - ev))
    cv2.imwrite(str(Path(out_dir) / f"{stem}.jpg"), cv2.resize(frames[i], None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    meta = dict(stem=stem, scenario=sc, cls=cls, frames=len(frames), still_step=tl[i][0], gif_scale=scale, timeline_ff=sum(l != "real time" for _, l in tl),
                arms=[dict(label=a.label, run_dir=str(a.d), end=a.end, steps=a.n, hd=float(a.row["hdscore"]),
                           rc=float(a.row["rc"]), stop_step=stop_step(a.infos), cam_h=a.cam_h,
                           ahead_end=ahead(a.infos[-1]), ahead_stop=ahead(a.infos[stop_step(a.infos)]) if stop_step(a.infos) else None)
                      for a in arms])
    json.dump(meta, open(Path(out_dir) / f"{stem}.json", "w"), indent=1)
    return stem, out.stat().st_size, len(frames)


def main():
    cv2.setNumThreads(2)
    if sys.argv[1] == "scan":
        return scan(sys.argv[2:])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[3:])
    jobs = [(s, sc, c, out) for s, sc, c in CASES if not only or s in only]
    with ProcessPoolExecutor(min(len(jobs), 9)) as ex:
        for r in ex.map(make_case, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
