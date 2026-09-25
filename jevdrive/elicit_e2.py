"""Elicitation E2: real-frame counterfactual edit pairs (todos/2026-09-26-elicitation-program.md, E2 and deviation-log
entry [E2] 00:40, written before any pair was built).

  candidates  (project venv, CPU) navtrain tokens whose t0 frame has a GT pedestrian / bicycle in the corridor
              (logged 4 s path polyline continued along its last heading to 30 m, +-1.5 m, ahead, <= 30 m, >= 20 px tall in CAM_F0); one actor per token (the
              nearest), one token per (log, actor); for every image the features read (CAM_F0 / L0 / R0 x the 4 agent
              frames at 2 Hz): the actor's projected 3D box, the other agents' boxes, and the placebo shift (the actor's
              box moved onto a fixed world point of the ego path with no agent under it)
  build       (envs/sam3, GPU) SAM 3.1 pedestrian / cyclist masks matched to the projected actor box (IoU >= 0.3),
              falling back to the projected box hull; dilated; LaMa (big-lama TorchScript, as released) on a crop
              around the mask, pasted back inside the mask only. x- = actor erased; placebo = the same mask shape moved
              onto empty road and inpainted. Chunks claimed with O_EXCL lock files, resumable.
  validate    (envs/ultralytics, GPU) YOLO26x-seg COCO person detections on x+ / x- / placebo: residual rate inside the
              erased actor's box, retention of the other GT agents
  fig         (project venv) the 16-pair figure
"""
import json
import os
import pickle
import socket
import time
from pathlib import Path

import numpy as np

from .common import data_dir, get_logger

log = get_logger(__name__)
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0")
W_IMG, H_IMG = 1920, 1080
CORRIDOR, RANGE, MIN_PX = 1.5, 30.0, 20.0
SAM_PROMPTS = ("pedestrian", "cyclist")
SAM_SCORE, MATCH_IOU = 0.5, 0.3
CHUNK = 64


def out_root(*p) -> Path:
    d = data_dir() / "processed" / "elicit_e2" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- geometry

def corners(box: np.ndarray) -> np.ndarray:
    """(x, y, z, l, w, h, yaw) -> (8, 3) corners in the same frame."""
    x, y, z, l, w, h, yaw = box[:7]
    dx, dy, dz = np.meshgrid([-l / 2, l / 2], [-w / 2, w / 2], [-h / 2, h / 2], indexing="ij")
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], 1)
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1], z + p[:, 2]], 1)


def project(pts_ego: np.ndarray, cam: dict):
    """Ego points (n, 3) -> pixels (n, 2) and a front/sane mask (image bounds not applied)."""
    from .navsim_zs import project_nuplan
    rays = pts_ego - np.asarray(cam["t"], np.float64)
    uv, _, _ = project_nuplan(rays, cam)
    r = rays @ np.asarray(cam["R"], np.float64)
    ok = (r[:, 2] > 0.5) & (np.abs(r[:, 0]) < 2.5 * r[:, 2]) & (np.abs(r[:, 1]) < 2.5 * r[:, 2])
    return uv.astype(np.float64), ok


def box2d(box3: np.ndarray, cam: dict):
    """Clipped 2D box [u0, v0, u1, v1] of a 3D box, or None when it is not (fully) in front of the camera."""
    uv, ok = project(corners(box3), cam)
    if not ok.all():
        return None
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    b = np.array([max(u0, 0), max(v0, 0), min(u1, W_IMG - 1), min(v1, H_IMG - 1)])
    return b if (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2 else None


def seg_dist(p: np.ndarray, poly: np.ndarray) -> float:
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())


def overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- candidates (CPU)

_TOK2FUT: dict = {}


def _log_candidates(log_path):
    tok2fut = _TOK2FUT
    from .fusion_q2b import extend
    from .navsim_zs import cams_of
    frames = pickle.load(open(log_path, "rb"))
    sensor = data_dir() / "datasets" / "navsim" / "sensor_blobs" / "trainval"
    at = {f["token"]: i for i, f in enumerate(frames)}
    out, seen = [], set()
    for tok in sorted((t for t in tok2fut if t in at), key=lambda t: frames[at[t]]["timestamp"]):
        i = at[tok]
        if i < 3:
            continue
        hist = frames[i - 3:i + 1]
        dts = np.diff([f["timestamp"] for f in hist]) / 1e6
        if not np.allclose(dts, 0.5, atol=0.06):
            continue
        a = hist[-1]["anns"]
        poly = extend(np.vstack([[0.0, 0.0], tok2fut[tok][:, :2]]), RANGE)   # deviation [E2] 00:50
        best = None
        for j, (nm, bx) in enumerate(zip(a["gt_names"], a["gt_boxes"])):
            if nm not in ("pedestrian", "bicycle") or bx[0] <= 0:
                continue
            d = float(np.hypot(bx[0], bx[1]))
            if d > RANGE or seg_dist(bx[:2], poly) > CORRIDOR:
                continue
            if best is None or d < best[0]:
                best = (d, j)
        if best is None:
            continue
        d, j = best
        track = a["track_tokens"][j]
        if track in seen:
            continue
        cams = [cams_of(f["cams"], sensor) for f in hist]
        b0 = box2d(a["gt_boxes"][j], cams[-1]["CAM_F0"])
        if b0 is None or b0[3] - b0[1] < MIN_PX:
            continue
        seen.add(track)
        E = [np.asarray(f["ego2global"], np.float64) for f in hist]
        # placebo world points: the ego path densified at 0.5 m, 5-30 m ahead, nearest range to the actor first
        seg = np.vstack([np.linspace(poly[k], poly[k + 1], max(2, int(np.linalg.norm(poly[k + 1] - poly[k]) / 0.5) + 1))
                         for k in range(len(poly) - 1)])
        rng_ = np.hypot(seg[:, 0], seg[:, 1])
        cand_pts = seg[(rng_ >= 5) & (rng_ <= RANGE)]
        cand_pts = cand_pts[np.argsort(np.abs(np.hypot(cand_pts[:, 0], cand_pts[:, 1]) - d))]
        imgs = []
        for k, (f, cm) in enumerate(zip(hist, cams)):
            fa = f["anns"]
            idx = np.flatnonzero(np.asarray(fa["track_tokens"]) == track)
            for c in CAMS:
                ab = box2d(fa["gt_boxes"][idx[0]], cm[c]) if len(idx) else None
                others = []
                for nm, bx, tt in zip(fa["gt_names"], fa["gt_boxes"], fa["track_tokens"]):
                    if tt == track or np.hypot(bx[0], bx[1]) > 60:
                        continue
                    ob = box2d(bx, cm[c])
                    if ob is not None:
                        others.append((str(nm), ob.tolist(), float(np.hypot(bx[0], bx[1]))))
                imgs.append({"k": k, "cam": c, "path": cm[c]["path"], "actor": None if ab is None else ab.tolist(),
                             "others": others, "cam_calib": {q: np.asarray(cm[c][q]).tolist() for q in ("R", "t", "K", "D")}})
        # choose the placebo point: the moved actor box must stay inside the image and touch no agent in any image
        placebo = None
        for p in cand_pts[:80]:
            pw = E[-1] @ np.r_[p, 0.0, 1.0]
            shifts, good = [], True
            for im in imgs:
                if im["actor"] is None:
                    shifts.append(None)
                    continue
                pk = np.linalg.solve(E[im["k"]], pw)[:3]
                uv, ok = project(pk[None], {q: np.asarray(v) for q, v in im["cam_calib"].items()})
                if not ok[0]:
                    shifts.append(None)
                    continue
                ab = np.asarray(im["actor"])
                du, dv = uv[0, 0] - (ab[0] + ab[2]) / 2, uv[0, 1] - ab[3]
                mb = ab + [du, dv, du, dv]
                if mb[0] < 0 or mb[1] < 0 or mb[2] > W_IMG - 1 or mb[3] > H_IMG - 1:
                    shifts.append(None)             # the point is not in this image: no placebo edit here
                    continue
                if overlap(mb, ab) or any(overlap(mb, o[1]) for o in im["others"]):
                    good = False
                    break
                shifts.append([float(du), float(dv)])
            if good and shifts[-3] is not None:     # at least the t0 CAM_F0 image carries the placebo
                placebo = {"point_t0": p.tolist(), "shifts": shifts}
                break
        out.append({"token": tok, "log": hist[-1]["log_name"], "track": str(track), "cls": str(a["gt_names"][j]),
                    "dist": d, "box3": np.asarray(a["gt_boxes"][j]).tolist(), "h_px_f0": float(b0[3] - b0[1]),
                    "images": imgs, "placebo": placebo})
    return out


def candidates(workers: int = 12) -> Path:
    from multiprocessing import Pool
    z = np.load(data_dir() / "runs" / "navsim_zs" / "index" / "navtrain_future.npz")
    _TOK2FUT.update(zip(z["tokens"].tolist(), z["poses"]))       # inherited by the forked workers
    logs = sorted((data_dir() / "datasets" / "navsim" / "navsim_logs" / "trainval").glob("*.pkl"))
    res = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_log_candidates, logs, chunksize=4)):
            res += r
            if i % 100 == 0:
                log.info("%d / %d logs, %d candidates", i, len(logs), len(res))
    res.sort(key=lambda c: (c["log"], c["token"]))
    dst = out_root("navtrain") / "candidates.pkl"
    pickle.dump(res, open(dst, "wb"), protocol=4)
    n_img = sum(im["actor"] is not None for c in res for im in c["images"])
    log.info("navtrain: %d candidates (%s), %d actor images, placebo for %d", len(res),
             dict(zip(*np.unique([c["cls"] for c in res], return_counts=True))), n_img,
             sum(c["placebo"] is not None for c in res))
    return dst


# ---------------------------------------------------------------- build (GPU, envs/sam3)

def _claim(lock: Path) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            o = json.loads(lock.read_text())
            os.kill(o["pid"], 0)
            return False
        except ProcessLookupError:
            lock.unlink(missing_ok=True)
            return _claim(lock)
        except (OSError, ValueError, KeyError):
            return False
    os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}).encode())
    os.close(fd)
    return True


class Lama:
    """big-lama TorchScript as released: (1, 3, H, W) in [0, 1] and (1, 1, H, W) {0, 1}, H and W multiples of 8."""

    def __init__(self, device="cuda"):
        import torch
        self.m = torch.jit.load(str(data_dir() / "models" / "lama" / "big-lama.pt"), map_location=device).eval()
        self.dev = device

    def __call__(self, img, mask):
        """img uint8 (3, H, W) GPU tensor, mask bool (H, W) GPU -> img with the mask region inpainted (crop around it)."""
        import torch
        import torch.nn.functional as F
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if not len(ys):
            return img
        H, W = mask.shape
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        side = max(384, int(2.5 * max(y1 - y0, x1 - x0)))
        hh, ww = min(side, H), min(side, W)
        top = int(np.clip(cy - hh / 2, 0, H - hh))
        left = int(np.clip(cx - ww / 2, 0, W - ww))
        crop = img[:, top:top + hh, left:left + ww].float()[None] / 255
        m = mask[top:top + hh, left:left + ww].float()[None, None]
        ph, pw = (-hh) % 8, (-ww) % 8
        with torch.inference_mode():
            out = self.m(F.pad(crop, (0, pw, 0, ph), mode="reflect"), F.pad(m, (0, pw, 0, ph)))[..., :hh, :ww]
        out = (out.clamp(0, 1) * 255).round().to(torch.uint8)[0]
        res = img.clone()
        sub = res[:, top:top + hh, left:left + ww]
        mm = m[0, 0].bool()
        sub[:, mm] = out[:, mm]
        return res


def _dilate(mask, px: int):
    import torch.nn.functional as F
    return F.max_pool2d(mask[None, None].float(), 2 * px + 1, 1, px)[0, 0] > 0


def _box_mask(box, H, W, dev):
    import torch
    m = torch.zeros((H, W), dtype=torch.bool, device=dev)
    u0, v0, u1, v1 = [int(round(x)) for x in box]
    m[max(v0, 0):v1 + 1, max(u0, 0):u1 + 1] = True
    return m


def _iou(a, b) -> float:
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def build(dataset: str = "navtrain", limit: int | None = None, tag: str = "main"):
    import torch
    from PIL import Image
    from tqdm import tqdm
    from . import sam_detect as S
    from .runlog import RunLog
    rl = RunLog("elicitation", "e2-build")
    cands = pickle.load(open(out_root(dataset) / "candidates.pkl", "rb"))
    if limit:
        cands = cands[:: max(1, len(cands) // limit)][:limit]
    dst = out_root(dataset, tag)
    model, rep = S.build()
    det = S.Detector(model, prompts=SAM_PROMPTS, mode="exact")
    lama = Lama()
    rl.event("start", dataset=dataset, tag=tag, n=len(cands), sam=rep)
    chunks = [cands[i:i + CHUNK] for i in range(0, len(cands), CHUNK)]
    for ci, chunk in enumerate(chunks):
        cdir = dst / f"c{ci:05d}"
        if (cdir / "meta.json").exists() or not _claim(dst / f"c{ci:05d}.lock"):
            continue
        cdir.mkdir(exist_ok=True)
        t0, metas, n_img = time.time(), [], 0
        for c in tqdm(chunk, desc=cdir.name, mininterval=10):
            td = cdir / c["token"]
            td.mkdir(exist_ok=True)
            rec = {k: c[k] for k in ("token", "log", "track", "cls", "dist", "h_px_f0")}
            rec["images"] = []
            for ii, im in enumerate(c["images"]):
                if im["actor"] is None:
                    continue
                img = S.decode(Path(im["path"]).read_bytes()).cuda()
                H, W = img.shape[-2:]
                ab = im["actor"]
                res = det([img], keep=SAM_SCORE)[0]
                cand = [(p, s, b, m) for p, d in zip(SAM_PROMPTS, res)
                        for s, b, m in zip(d["scores"].tolist(), d["boxes"].tolist(), d["masks"])]
                ious = [_iou(b, ab) for _, _, b, _ in cand]
                if c["cls"] == "bicycle":
                    sel = [m for (_, _, b, m), iou in zip(cand, ious) if iou >= MATCH_IOU]
                else:
                    sel = [cand[int(np.argmax(ious))][3]] if ious and max(ious) >= MATCH_IOU else []
                src = "sam" if sel else "box"
                mask = torch.stack(sel).any(0) if sel else _box_mask(ab, H, W, img.device)
                mask = _dilate(mask, max(7, int(0.08 * (ab[3] - ab[1]))))
                out = lama(img, mask)
                name = f"{im['cam']}_{im['k']}"
                Image.fromarray(out.permute(1, 2, 0).cpu().numpy()).save(td / f"{name}_minus.jpg", quality=95)
                r = {"i": ii, "cam": im["cam"], "k": im["k"], "path": im["path"], "actor": ab, "mask_src": src,
                     "mask_px": int(mask.sum()), "sam_best_iou": float(max(ious)) if ious else 0.0, "placebo": False}
                sh = c["placebo"]["shifts"][ii] if c["placebo"] else None
                if sh is not None:
                    du, dv = int(round(sh[0])), int(round(sh[1]))
                    pm = torch.roll(mask, shifts=(dv, du), dims=(0, 1))
                    pout = lama(img, pm)
                    Image.fromarray(pout.permute(1, 2, 0).cpu().numpy()).save(td / f"{name}_placebo.jpg", quality=95)
                    r.update(placebo=True, placebo_shift=[du, dv])
                rec["images"].append(r)
                n_img += 1
            metas.append(rec)
        dt = time.time() - t0
        (cdir / "meta.json").write_text(json.dumps(metas))
        (dst / f"c{ci:05d}.lock").unlink(missing_ok=True)
        rl.event("chunk", chunk=ci, pairs=len(chunk), images=n_img, seconds=dt)
        rl.scalar("e2/s_per_image", dt / max(n_img, 1), ci)
        rl.info(f"{cdir.name}: {len(chunk)} pairs, {n_img} images, {dt / max(n_img, 1):.2f} s/image")
    if all((dst / f"c{ci:05d}" / "meta.json").exists() for ci in range(len(chunks))):
        (dst / "done.json").write_text(json.dumps({"pairs": len(cands), "chunks": len(chunks)}))
    rl.event("end")
    rl.close()


# ---------------------------------------------------------------- validate (envs/ultralytics)

def validate(dataset: str = "navtrain", tag: str = "val64", conf: float = 0.25, iou_gate: float = 0.3):
    """Residual person detections inside the erased actor's box on x- (vs x+), retention of other GT agents."""
    import pandas as pd
    from ultralytics import YOLO
    from .runlog import RunLog
    rl = RunLog("elicitation", "e2-validate")
    model = YOLO(str(data_dir() / "models" / "ultralytics" / "yolo26x-seg.pt"))
    cands = {c["token"]: c for c in pickle.load(open(out_root(dataset) / "candidates.pkl", "rb"))}
    root = out_root(dataset, tag)
    rows, oth = [], []
    coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    gt_map = {"pedestrian": {"person"}, "bicycle": {"bicycle", "person", "motorcycle"}, "vehicle": {"car", "bus", "truck", "motorcycle"}}

    def dets(path):
        r = model.predict(str(path), conf=conf, verbose=False)[0]
        b = r.boxes
        return [(coco.get(int(c)), x) for c, x in zip(b.cls.tolist(), b.xyxy.tolist()) if int(c) in coco]

    for meta in sorted(root.glob("c*/meta.json")):
        for rec in json.loads(meta.read_text()):
            c = cands[rec["token"]]
            for r in rec["images"]:
                im = c["images"][r["i"]]
                name = f"{r['cam']}_{r['k']}"
                dp, dm = dets(r["path"]), dets(meta.parent / rec["token"] / f"{name}_minus.jpg")
                hit = lambda ds: max([_iou(x, r["actor"]) for cl, x in ds if cl == "person"], default=0.0)  # noqa: E731
                rows.append({"token": rec["token"], "cls": rec["cls"], "img": name, "mask_src": r["mask_src"],
                             "h_px": r["actor"][3] - r["actor"][1], "iou_plus": hit(dp), "iou_minus": hit(dm)})
                for nm, ob, dist in im["others"]:
                    if nm not in gt_map or ob[3] - ob[1] < MIN_PX or dist > 40:
                        continue
                    mb = [x for x in r["actor"]]
                    ib = max(0, min(ob[2], mb[2]) - max(ob[0], mb[0])) * max(0, min(ob[3], mb[3]) - max(ob[1], mb[1]))
                    cov = ib / max((ob[2] - ob[0]) * (ob[3] - ob[1]), 1e-9)
                    f = lambda ds: max([_iou(x, ob) for cl, x in ds if cl in gt_map[nm]], default=0.0)  # noqa: E731
                    oth.append({"token": rec["token"], "img": name, "cls": nm, "covered_by_actor_box": cov,
                                "det_plus": f(dp) >= iou_gate, "det_minus": f(dm) >= iou_gate})
    t, o = pd.DataFrame(rows), pd.DataFrame(oth)
    t.to_csv(rl.dir / "residual.csv", index=False)
    o.to_csv(rl.dir / "others.csv", index=False)
    base = t[t.iou_plus >= iou_gate]
    ok_o = o[o.det_plus & (o.covered_by_actor_box <= 0.2)]
    summ = {"images": len(t), "pairs": t.token.nunique(), "yolo_person_on_plus": len(base),
            "residual_rate": float((base.iou_minus >= iou_gate).mean()) if len(base) else None,
            "residual_rate_sam_masks": float((base[base.mask_src == "sam"].iou_minus >= iou_gate).mean()) if len(base) else None,
            "mask_src_sam": float((t.mask_src == "sam").mean()),
            "others_eval": len(ok_o), "others_retained": float(ok_o.det_minus.mean()) if len(ok_o) else None,
            "others_covered_gt20pct": int((o.covered_by_actor_box > 0.2).sum())}
    (rl.dir / "summary.json").write_text(json.dumps(summ, indent=1))
    rl.info(json.dumps(summ))
    rl.close()
    return summ


# ---------------------------------------------------------------- figure (project venv)

def fig16(dataset: str = "navtrain", tag: str = "val64", n: int = 16, seed: int = 0) -> Path:
    """16 random pairs (seed 0): t0 CAM_F0 crops around the actor, x+ | x- | placebo (the placebo crop is taken
    around the moved box). PNG + PDF in the run dir; the PNG is also written to research/figs/."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import plots
    from .runlog import RunLog
    plt.rcParams.update(plots.STYLE)
    rl = RunLog("elicitation", "e2-fig")
    root = out_root(dataset, tag)
    recs = [(m.parent, r) for m in sorted(root.glob("c*/meta.json")) for r in json.loads(m.read_text())]
    recs = [(d, r, im) for d, r in recs for im in r["images"] if im["cam"] == "CAM_F0" and im["k"] == 3]
    pick = np.random.default_rng(seed).choice(len(recs), min(n, len(recs)), replace=False)

    def crop(path, box, W=480, H=270):
        b = np.asarray(box, float)
        cx, cy, s = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(3 * (b[3] - b[1]), 200)
        w, h = s * 16 / 9, s
        x0, y0 = int(np.clip(cx - w / 2, 0, W_IMG - w)), int(np.clip(cy - h / 2, 0, H_IMG - h))
        return np.asarray(Image.open(path).convert("RGB").crop((x0, y0, x0 + int(w), y0 + int(h))).resize((W, H)))

    fig, axes = plt.subplots(8, 6, figsize=(6.875, 6.875 * 8 * 270 / (6 * 480) + 0.3))
    for j, i in enumerate(pick):
        d, r, im = recs[i]
        name = f"{im['cam']}_{im['k']}"
        panels = [crop(im["path"], im["actor"]), crop(d / r["token"] / f"{name}_minus.jpg", im["actor"])]
        if im.get("placebo"):
            du, dv = im["placebo_shift"]
            pb = [im["actor"][0] + du, im["actor"][1] + dv, im["actor"][2] + du, im["actor"][3] + dv]
            panels.append(crop(d / r["token"] / f"{name}_placebo.jpg", pb))
        row, col = j // 2, (j % 2) * 3
        for q in range(3):
            ax = axes[row, col + q]
            ax.axis("off")
            if q < len(panels):
                ax.imshow(panels[q])
            if row == 0:
                ax.set_title(("$x^+$", "$x^-$", "placebo")[q], fontsize=8, pad=2)
        axes[row, col].text(4, 20, f"{j + 1}: {r['cls'][:3]} {r['dist']:.0f} m ({im['mask_src']})", color="w", fontsize=6)
    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    plots.save(fig, rl.dir, "elicit-e2-pairs16")
    rl.close()
    return rl.dir


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("candidates", "build", "validate", "fig"))
    ap.add_argument("--dataset", default="navtrain")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "candidates":
        candidates(a.workers)
    elif a.cmd == "build":
        build(a.dataset, a.limit, a.tag)
    elif a.cmd == "fig":
        print(fig16(a.dataset, a.tag))
    else:
        print(validate(a.dataset, a.tag))


if __name__ == "__main__":
    main()
