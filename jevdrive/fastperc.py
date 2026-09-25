"""Fast perception: SAM 3.1 speed-ups and faster detectors, latency x recall on the fusion Q4 frames
(pre-registration: todos/2026-09-26-fast-perception.md).

GPU part (run inside each model's own env; only numpy / pandas / torch + that model's package are imported):
  latency  batch 1, 1 camera and 3 cameras, p50 / p95 ms and peak VRAM (the Q4d protocol, same image list)
  detect   an image list -> part-*.parquet in the sam_detect schema (key, prompt, score, box, area, cu, cv, H, W)
CPU part (envs/jevdrive):
  subset   the fixed frame subset S (P5 hazard frames x 3 cameras, every 3rd nuScenes scene x 3 front cameras)
  eval     fusion_q4's own evaluate / hazard_reading on S for one detection dir; summary row + tables
  summary  latency + recall + paired deltas against SAM 3.1 -> small csv files (research/results/fast-perception)
  depth    optional side reading: metric depth (YOLO26-depth) at each detection's contact pixel (envs/ultralytics)

Backend specs (`--backend`), all returning per image, per prompt: scores (n,), boxes xyxy px (n, 4), masks (n, H, W)
bool at the original resolution (None for box-only models):
  sam31:<mode>:<prompts>:<res>:<compile>:<amp>   SAM 3.1 detector via sam_detect.Detector (mode exact|batched,
                                                 prompts all|pv|p, compile none|max-autotune|default)
  esam3:<tinyvit|repvit|efficientvit>:<mode>:<amp>  EfficientSAM3 Stage 3 through the same Detector (their Sam3Image)
  yoloe:<weights>:<imgsz>:<half>[:words]         Ultralytics YOLOE, text prompts = the six Q4 prompts (words: COCO-style
                                                 class words, YOLOE_WORDS, mapped back onto the Q4 classes)
  yolo:<weights>:<imgsz>:<half>                  Ultralytics COCO closed set, mapped onto the Q4 classes
  gdino:<box_thr>:<text_thr>                     Grounding DINO tiny (HF transformers), boxes only
"""
import json
import os
import time
from pathlib import Path

import numpy as np

PROMPTS = ("pedestrian", "cyclist", "vehicle", "cone", "debris", "emergency vehicle")
PROMPT_SETS = {"all": PROMPTS, "pv": ("pedestrian", "vehicle"), "p": ("pedestrian",)}
COCO_MAP = {"person": "pedestrian", "bicycle": "cyclist", "car": "vehicle", "motorcycle": "vehicle", "bus": "vehicle",
            "truck": "vehicle"}
# YOLOE side variant (added after a 6-image smoke test, before any recall number): COCO-style class words as the text
# prompts, mapped back onto the Q4 classes
YOLOE_WORDS = {**COCO_MAP, "traffic cone": "cone", "debris": "debris", "ambulance": "emergency vehicle",
               "fire truck": "emergency vehicle", "police car": "emergency vehicle"}
ESAM3 = {"tinyvit": ("tinyvit", "11m"), "repvit": ("repvit", "m1.1"), "efficientvit": ("efficientvit", "b1")}


def models_dir() -> Path:
    return Path(os.environ["DATA_DIR"]) / "models"


# ================================================================ backends (GPU)

class SamFamily:
    """SAM 3.1 or EfficientSAM3 behind jevdrive.sam_detect.Detector. Input: uint8 CHW tensors on the GPU."""
    kind = "chw"

    def __init__(self, spec: str, keep: float):
        import torch
        from . import sam_detect as sd
        f = spec.split(":")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        if f[0] == "sam31":
            mode, ps, res, comp, amp = f[1:6]
            model, self.report = sd.build(compile_mode=None if comp == "none" else comp)
            self.det = sd.Detector(model, PROMPT_SETS[ps], mode=mode, res=int(res), amp=amp)
        else:
            from sam3.model_builder import build_efficientsam3_image_model
            bb, name = ESAM3[f[1]]
            mode, amp = f[2], f[3]
            model = build_efficientsam3_image_model(
                checkpoint_path=str(models_dir() / "efficientsam3" / f"efficientsam3_{f[1]}.pt"), backbone_type=bb,
                model_name=name, text_encoder_type="MobileCLIP-S0", text_encoder_context_length=16, load_from_HF=False)
            self.report = {"ckpt": f"efficientsam3_{f[1]}.pt"}
            self.det = sd.Detector(model, PROMPTS, mode=mode, image_kwargs={}, amp=amp)
        self.keep = keep

    def __call__(self, imgs):
        return [[(d["scores"], d["boxes"], d["masks"]) for d in per] for per in self.det(imgs, keep=self.keep)], \
            self.det.prompts


class Ultra:
    """Ultralytics YOLOE (text prompts) or YOLO (COCO). Input: HWC uint8 BGR numpy (Ultralytics' array convention)."""
    kind = "bgr"

    def __init__(self, spec: str, keep: float):
        f = spec.split(":")
        self.imgsz, self.half, self.keep = int(f[2]), f[3] == "half", keep
        w = str(models_dir() / "ultralytics" / f[1])
        cwd = os.getcwd()
        os.chdir(models_dir() / "ultralytics")          # the text encoder (mobileclip2_b.ts) is looked up in the cwd
        try:
            words = f[0] == "yoloe" and len(f) > 4 and f[4] == "words"
            if f[0] == "yoloe":
                from ultralytics import YOLOE
                self.m = YOLOE(w)
                self.m.set_classes(list(YOLOE_WORDS) if words else list(PROMPTS))
            else:
                from ultralytics import YOLO
                self.m = YOLO(w)
        finally:
            os.chdir(cwd)
        names = self.m.names
        self.map = {i: (YOLOE_WORDS.get(n) if words else n if f[0] == "yoloe" else COCO_MAP.get(n)) for i, n in names.items()}
        self.prompts = list(PROMPTS) if f[0] == "yoloe" else ["pedestrian", "cyclist", "vehicle"]
        self.report = {"weights": f[1], "names": len(names)}

    def __call__(self, imgs):
        import torch
        rs = self.m.predict(imgs, imgsz=self.imgsz, conf=self.keep, half=self.half, retina_masks=True, verbose=False)
        out = []
        for r in rs:
            cls = r.boxes.cls.int().tolist()
            lab = np.array([self.map.get(c) or "" for c in cls])
            per = []
            for p in self.prompts:
                k = torch.as_tensor(lab == p, device=r.boxes.data.device)
                m = r.masks.data[k] > 0.5 if r.masks is not None and k.any() else \
                    torch.zeros((0, *r.orig_shape), dtype=torch.bool, device=k.device)
                per.append((r.boxes.conf[k], r.boxes.xyxy[k], m))
            out.append(per)
        return out, self.prompts


class GDino:
    """Grounding DINO tiny, HF transformers, the model card's usage. Input: HWC uint8 RGB numpy. Boxes only."""
    kind = "rgb"

    def __init__(self, spec: str, keep: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        f = spec.split(":")
        self.box_thr, self.text_thr = min(float(f[1]), keep), float(f[2])
        d = models_dir() / "grounding-dino-tiny"
        self.proc = AutoProcessor.from_pretrained(d)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(d).cuda().eval()
        self.text = ". ".join(PROMPTS) + "."
        self.prompts, self.torch = list(PROMPTS), torch
        self.report = {"weights": "IDEA-Research/grounding-dino-tiny"}

    def __call__(self, imgs):
        torch = self.torch
        with torch.inference_mode():
            x = self.proc(images=imgs, text=[self.text] * len(imgs), return_tensors="pt").to("cuda")
            o = self.m(**x)
            rs = self.proc.post_process_grounded_object_detection(
                o, x.input_ids, threshold=self.box_thr, text_threshold=self.text_thr,
                target_sizes=[i.shape[:2] for i in imgs])
        out = []
        for r in rs:
            # transformers returns labels [''] for an image with no boxes; keep one label per box
            lab = np.array(list(r.get("text_labels", r["labels"]))[:len(r["scores"])], dtype=object)
            per = []
            for p in self.prompts:           # a phrase maps to the longest prompt it contains ("emergency vehicle")
                hit = np.array([_phrase(l) == p for l in lab], bool)
                k = torch.as_tensor(hit, device=r["scores"].device)
                per.append((r["scores"][k], r["boxes"][k], None))
            out.append(per)
        return out, self.prompts


def _phrase(label: str) -> str | None:
    c = [p for p in PROMPTS if p in str(label)]
    return max(c, key=len) if c else None


def backend(spec: str, keep: float):
    kind = spec.split(":")[0]
    return {"sam31": SamFamily, "esam3": SamFamily, "yoloe": Ultra, "yolo": Ultra, "gdino": GDino}[kind](spec, keep)


def to_input(chw, kind: str):
    """Decoded uint8 CHW RGB tensor (CPU) -> the backend's input form (outside the timed region)."""
    if kind == "chw":
        return chw.pin_memory()
    hwc = chw.permute(1, 2, 0).numpy()
    return np.ascontiguousarray(hwc[:, :, ::-1]) if kind == "bgr" else np.ascontiguousarray(hwc)


def feed(host, kind: str):
    return [h.to("cuda", non_blocking=True) for h in host] if kind == "chw" else list(host)


# ================================================================ latency and detection (GPU)

def latency(be, image_list: str, n: int = 200, warm: int = 20) -> dict:
    """The Q4d protocol: batch 1, 1 camera (one image per call) and 3 cameras (three images in one call), timed from
    decoded images on the host to per-instance outputs on the GPU (torch.cuda.synchronize)."""
    import pandas as pd
    import torch
    from .sam_detect import _reader, decode
    rows = pd.read_parquet(image_list).iloc[:n + warm].to_dict("records")
    host = [to_input(decode(_reader(r)), be.kind) for r in rows]
    out = {}
    for label, k in (("1 camera", 1), ("3 cameras", 3)):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for i in range(0, len(host) - k, k):
            torch.cuda.synchronize()
            a = time.perf_counter()
            be(feed(host[i:i + k], be.kind))
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - a))
        ts = np.array(ts[warm // k:])
        out[label] = {"n": len(ts), "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
                      "mean_ms": float(ts.mean()), "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    return out


def detect(be, image_list: str, out_dir: str, keys=None, batch: int = 8, shard: int = 2000, workers: int = 8, rl=None):
    """Image list (optionally restricted to `keys`) -> out_dir/part-<k>.parquet, resumable per shard."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from .sam_detect import _collate, _Images, contact
    t = pd.read_parquet(image_list)
    if keys is not None:
        t = t[t.key.isin(set(keys))].reset_index(drop=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0, n_img = time.time(), 0
    for s0 in range(0, len(t), shard):
        dst = out / f"part-{s0 // shard:04d}.parquet"
        if dst.exists():
            continue
        rows = t.iloc[s0:s0 + shard].to_dict("records")
        dl = DataLoader(_Images(rows), batch_size=batch, num_workers=workers, collate_fn=_collate, prefetch_factor=4)
        recs, ts = [], time.time()
        for idx, ims in tqdm(dl, desc=dst.name, mininterval=10):
            res, prompts = be(feed([to_input(x, be.kind) for x in ims], be.kind))
            for i, x, per in zip(idx, ims, res):
                H, W = x.shape[-2:]
                for p, (sc, bx, mk) in zip(prompts, per):
                    if not len(sc):
                        continue
                    bx = bx.float().cpu().numpy()
                    if mk is not None:
                        u, v, area = contact(mk.bool())
                    else:                                   # box-only: bottom-centre of the box
                        u, v, area = (bx[:, 0] + bx[:, 2]) / 2, bx[:, 3], np.full(len(bx), np.nan)
                    recs.append(pd.DataFrame({"key": rows[i]["key"], "prompt": p, "score": sc.float().cpu().numpy(),
                                              "x0": bx[:, 0], "y0": bx[:, 1], "x1": bx[:, 2], "y1": bx[:, 3],
                                              "area": area, "cu": u, "cv": v, "H": int(H), "W": int(W)}))
            n_img += len(idx)
        df = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(
            columns=["key", "prompt", "score", "x0", "y0", "x1", "y1", "area", "cu", "cv", "H", "W"])
        df.to_parquet(dst.with_suffix(".tmp"), index=False)
        dst.with_suffix(".tmp").rename(dst)
        dt = time.time() - ts
        if rl:
            rl.info(f"{dst.name}: {len(rows)} images, {len(df)} instances, {1000 * dt / len(rows):.1f} ms/image")
            rl.event("shard", part=dst.name, images=len(rows), instances=len(df), seconds=dt)
    info = {"images": len(t), "new_images": n_img, "seconds": time.time() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    (out / "done.json").write_text(json.dumps(info, indent=1))
    return info


def depth_sample(det_dir: str, out_dir: str, weights: str = "yolo26x-depth.pt", batch: int = 8, workers: int = 8,
                 min_score: float = 0.05, rl=None) -> dict:
    """Optional side reading D-depth: a monocular metric depth model (Ultralytics YOLO26-depth, as shipped: no camera
    intrinsics) on every S image that has detections; per detection, the median predicted depth in a 5 x 5 window just
    above its ground-contact pixel. Writes out_dir/{p5,nusc}/part-*.parquet = the detection rows + a `depth` column."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from ultralytics import YOLO
    from .sam_detect import _collate, _Images
    L = Path(os.environ["DATA_DIR"]) / "processed/fusion_diag/lists"
    m = YOLO(str(models_dir() / "ultralytics" / weights))
    info = {}
    for ds, lst in (("p5", "p5.parquet"), ("nusc", "nusc.parquet")):
        parts = sorted((Path(det_dir) / ds).glob("part-*.parquet"))
        d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        S = json.loads((Path(os.environ["DATA_DIR"]) / "processed/fastperc/subset.json").read_text())[ds]
        d = d[(d.score > min_score) & d.key.isin(set(S))].reset_index(drop=True)
        t = pd.read_parquet(L / lst)
        rows = t[t.key.isin(set(d.key))].to_dict("records")
        gi = d.groupby("key").indices
        dep = np.full(len(d), np.nan, np.float32)
        dl = DataLoader(_Images(rows), batch_size=batch, num_workers=workers, collate_fn=_collate, prefetch_factor=4)
        for idx, ims in tqdm(dl, desc=f"depth {ds}", mininterval=10):
            rs = m.predict([to_input(x, "bgr") for x in ims], verbose=False)
            for i, r in zip(idx, rs):
                D = r.depth.data.float()
                H, W = D.shape
                j = gi[rows[i]["key"]]
                u = np.clip(np.nan_to_num(d.cu.to_numpy()[j]).round().astype(int), 2, W - 3)
                v = np.clip(np.nan_to_num(d.cv.to_numpy()[j]).round().astype(int) - 3, 2, H - 3)
                win = torch.stack([D[vv - 2:vv + 3, uu - 2:uu + 3].reshape(-1) for uu, vv in zip(u, v)])
                dep[j] = win.median(1).values.cpu().numpy()
        o = Path(out_dir) / ds
        o.mkdir(parents=True, exist_ok=True)
        d.assign(depth=dep).to_parquet(o / "part-0000.parquet", index=False)
        info[ds] = {"detections": len(d), "images": len(rows)}
        if rl:
            rl.info(f"depth {ds}: {info[ds]}")
    return info


# ================================================================ the subset and the evaluation (CPU)

def subset() -> dict:
    """The pre-registered frame subset S: keys of the P5 and nuScenes lists."""
    import pandas as pd
    from . import fusion_q4 as Q
    L = Q.root("lists")
    p5, gt = pd.read_parquet(L / "p5.parquet"), pd.read_parquet(L / "p5_gt.parquet")
    h = gt[gt.hazard & (gt.cam == "front") & (gt.px_front >= Q.FACTOR_PX) & (gt.dz.abs() < 8)]
    h = h[h.key.map(p5.set_index("key").world) == "plus"]
    frames = set(h.key.str.split("|").str[0])
    nl = pd.read_parquet(L / "nusc.parquet")
    scenes = set(sorted(nl.scene.unique())[::3])
    return {"p5": p5.key[p5.frame_name.isin(frames)].tolist(), "nusc": nl.key[nl.scene.isin(scenes)].tolist()}


def _depth_place(d):
    """Replace the flat-ground lift by camera centre + depth x ray (rays from fusion_q4.lift have unit optical-axis
    component, so `depth` is the optical-axis depth); lift_ok = finite depth within 80 m."""
    gx = d.ox + d.depth * d.rx
    gy = d.oy + d.depth * d.ry
    gd = np.hypot(gx, gy)
    return d.assign(gx=gx, gy=gy, gdist=gd, lift_ok=np.isfinite(gd) & (d.depth > 0) & (gd <= 80))


def _pick(t, reading: str, cls: str, col: str, val):
    r = t[(t.reading == reading) & (t.cls == cls) & (t[col].astype(str) == str(val))]
    return r.iloc[0] if len(r) else None


def evaluate(det_dir: str, name: str, score: float, out: Path, contact: str = "mask", sweep: bool = True) -> dict:
    """fusion_q4's evaluation, unchanged, on S. contact "box": ground contact = box bottom centre (side reading)."""
    import pandas as pd
    from . import fusion_q4 as Q
    L = Q.root("lists")
    S = subset()
    p5, gt = pd.read_parquet(L / "p5.parquet"), pd.read_parquet(L / "p5_gt.parquet")
    p5, gt = p5[p5.key.isin(S["p5"])], gt[gt.key.isin(S["p5"])]
    nl, ng = pd.read_parquet(L / "nusc.parquet"), pd.read_parquet(L / "nusc_gt.parquet")
    nl, ng = nl[nl.key.isin(S["nusc"])], ng[ng.key.isin(S["nusc"])]
    ncal = json.loads((L / "nusc_calib.json").read_text())
    thr = min(score, 0.05) if sweep else score                # det_dir holds p5/ and nusc/ part directories
    d_all = pd.concat([Q.load_dets(Path(det_dir) / s, thr) for s in ("p5", "nusc") if (Path(det_dir) / s).exists()],
                      ignore_index=True)
    if contact == "box":
        d_all = d_all.assign(cu=(d_all.x0 + d_all.x1) / 2, cv=d_all.y1)
    if contact == "depth":                                    # the depth_sample output: rows carry `depth`
        d_all = pd.concat([pd.read_parquet(p) for s in ("p5", "nusc") for p in sorted((Path(det_dir) / s).glob("part-*.parquet"))],
                          ignore_index=True)
        d_all = d_all[d_all.score > thr].reset_index(drop=True)
    dp = d_all[d_all.key.isin(S["p5"])].reset_index(drop=True)
    dp = Q.lift_dets(dp, dp.key.str.split("|").str[1].to_numpy(), Q.p5_calib())
    dn = d_all[d_all.key.isin(S["nusc"])].reset_index(drop=True)
    dn = Q.lift_dets(dn, dn.key.to_numpy(), ncal)
    if contact == "depth":   # place the contact point on its camera ray at the predicted (optical-axis) depth
        dp, dn = _depth_place(dp), _depth_place(dn)
    wx = p5.set_index("key")
    w5 = pd.Series(np.where(wx.sun_altitude < 0, "night", np.where(wx.precipitation > 30, "rain", "day")), index=wx.index)
    nx = nl.set_index("key")
    wn = pd.Series(np.where(nx.night, "night", np.where(nx.rain, "rain", "day")), index=nx.index)
    ng = ng.assign(eval=ng.vis >= 3)
    dp5, dnu = dp[dp.score > score], dn[dn.score > score]
    ev5 = Q.evaluate(p5, gt, dp5, Q.GT_CLASSES_P5, w5)
    haz, per = Q.hazard_reading(p5, gt, dp5)
    per["base_id"] = per.key.map(wx.base_id)
    evn = Q.evaluate(nl, ng, dnu, ("pedestrian", "vehicle", "cyclist", "cone"), wn)
    out.mkdir(parents=True, exist_ok=True)
    for k, t in {**{f"p5_{a}": b for a, b in ev5.items()}, **{f"nusc_{a}": b for a, b in evn.items()},
                 "p5_hazard": haz}.items():
        t.to_csv(out / f"{k}.csv", index=False)
    per[["key", "id", "cls", "dist", "base_id", "hit", "hit_oracle", "err"]].to_parquet(out / "hazard_rows.parquet")

    R2 = "recall (ii), <= 40 m, in image"
    RO = "side: recall (ii) with the oracle-height lift"
    row = {"name": name, "score": score, "contact": contact}
    for cls in ("pedestrian", "vehicle"):
        H = haz[haz.cls == cls].set_index("scope")
        for sc_, tag in (("all", "all"), ("<= 20 m", "le20"), ("<= 30 m, day", "le30day")):
            row[f"p5_haz_{cls[:3]}_{tag}"] = H.recall.get(sc_)
            row[f"p5_haz_{cls[:3]}_{tag}_oracle"] = H.recall_oracle_height.get(sc_)
        row[f"p5_haz_{cls[:3]}_le20_bev_med"] = H.bev_err_med.get("<= 20 m")
        for ds, ev in (("p5", ev5), ("nusc", evn)):
            rec = ev["recall"]
            for b in ("0-10 m", "10-20 m", "20-40 m"):
                r = _pick(rec, R2, cls, "dbin", b)
                row[f"{ds}_{cls[:3]}_{b.replace(' m', '')}"] = None if r is None else r.recall
            r, o = _pick(rec, R2, cls, "all", "all"), _pick(rec, RO, cls, "all", "all")
            row[f"{ds}_{cls[:3]}_le40"] = None if r is None else r.recall
            row[f"{ds}_{cls[:3]}_le40_oracle"] = None if o is None else o.recall
            pr = _pick(ev["precision"], "precision, lifted <= 80 m", cls, "all", "all")
            row[f"{ds}_{cls[:3]}_prec"] = None if pr is None else pr.precision
    if sweep:                                             # nuScenes pedestrian recall / precision over the threshold
        sw = []
        for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
            e = Q.evaluate(nl, ng, dn[dn.score > thr], ("pedestrian",), wn)
            r, p = _pick(e["recall"], R2, "pedestrian", "all", "all"), _pick(e["precision"], "precision, lifted <= 80 m",
                                                                              "pedestrian", "all", "all")
            sw.append({"thr": thr, "recall": None if r is None else r.recall, "precision": None if p is None else p.precision})
        pd.DataFrame(sw).to_csv(out / "nusc_ped_sweep.csv", index=False)
    (out / "row.json").write_text(json.dumps(row, indent=1, default=float))
    return row


def paired_boot(a: "pd.DataFrame", b: "pd.DataFrame", cls: str, maxd: float | None = None, n: int = 2000, seed: int = 0):
    """Hazard rows of two runs (same GT rows): recall(a) - recall(b) with a base-route bootstrap 95% CI."""
    m = a.merge(b, on=["key", "id"], suffixes=("_a", "_b"))
    m = m[m.cls_a == cls]
    if maxd is not None:
        m = m[m.dist_a <= maxd]
    codes, uniq = np.unique(m.base_id_a.to_numpy(), return_inverse=True)
    diff = m.hit_a.astype(float).to_numpy() - m.hit_b.astype(float).to_numpy()
    num, den = np.bincount(uniq, diff, len(codes)), np.bincount(uniq, minlength=len(codes)).astype(float)
    idx = np.random.default_rng(seed).integers(len(codes), size=(n, len(codes)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(diff.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summary(out: Path, base: str = "sam31-orig") -> dict:
    """Latency (GPU 4 runs, the latest per backend), recall rows, paired hazard-pedestrian deltas against SAM 3.1 and
    nuScenes pedestrian recall at SAM 3.1's precision -> out/{latency,recall}.csv (small, committed)."""
    import pandas as pd
    R = Path(os.environ["DATA_DIR"]) / "runs/fastperc"
    lat = []
    for f in sorted((R / "latency").glob("*/*/latency.json")):
        d = json.loads(f.read_text())
        for cam in ("1 camera", "3 cameras"):
            lat.append({"backend": d["backend"], "cams": int(cam[0]), "run": f.parent.name, **d[cam]})
    lat = pd.DataFrame(lat).sort_values("run").groupby(["backend", "cams"]).last().reset_index().drop(columns="run")
    runs = {}
    for f in sorted((R / "eval").glob("*/*/row.json")):
        runs[f.parent.parent.name] = f.parent                     # the latest run per tag
    rows = {t: json.loads((d / "row.json").read_text()) for t, d in runs.items()}
    b_rows = pd.read_parquet(runs[base] / "hazard_rows.parquet")
    target = rows[base]["nusc_ped_prec"]
    for t, d in runs.items():
        a = pd.read_parquet(d / "hazard_rows.parquet")
        for tag, mx in (("all", None), ("le20", 20.0)):
            m, lo, hi = paired_boot(a, b_rows, "pedestrian", mx)
            rows[t] |= {f"d_haz_ped_{tag}": m, f"d_haz_ped_{tag}_lo": lo, f"d_haz_ped_{tag}_hi": hi}
        sw = d / "nusc_ped_sweep.csv"
        if sw.exists():
            w = pd.read_csv(sw).dropna()
            ok = w[w.precision >= target]
            rows[t]["nusc_ped_recall_at_base_prec"] = float(ok.recall.max()) if len(ok) else np.nan
    rec = pd.DataFrame(rows.values())
    out.mkdir(parents=True, exist_ok=True)
    lat.to_csv(out / "latency.csv", index=False, float_format="%.4g")
    rec.to_csv(out / "recall.csv", index=False, float_format="%.4g")
    return {"latency": len(lat), "recall": len(rec)}


# ================================================================ CLI

def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("latency", "detect", "subset", "eval", "depth", "summary"))
    ap.add_argument("--backend", help="backend spec, see the module docstring")
    ap.add_argument("--list", help="image list parquet (latency / detect)")
    ap.add_argument("--dataset", choices=("p5", "nusc"), help="detect: restrict to that part of S")
    ap.add_argument("--full", action="store_true", help="detect: the whole list instead of S")
    ap.add_argument("--out", help="detect: output dir; eval: detection dir")
    ap.add_argument("--keep", type=float, default=0.05, help="lowest score stored (detect) / used (latency: default thr)")
    ap.add_argument("--score", type=float, default=0.5, help="eval: the model's default threshold")
    ap.add_argument("--contact", default="mask", choices=("mask", "box", "depth"))
    ap.add_argument("--depth-out", help="depth: output dir (detections + depth column)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    rl = RunLog("fastperc", a.step, a.tag)
    rl.event("start", args=vars(a))
    if a.step in ("latency", "detect"):
        be = backend(a.backend, a.keep)
        rl.info(f"backend {a.backend}: {be.report}")
        if a.step == "latency":
            r = latency(be, a.list)
            (rl.dir / "latency.json").write_text(json.dumps({"backend": a.backend, "keep": a.keep, **r}, indent=1))
            rl.info(f"latency {a.backend}: {r}")
        else:
            keys = None
            if not a.full:
                keys = json.loads((Path(os.environ["DATA_DIR"]) / "processed/fastperc/subset.json").read_text())[a.dataset]
            info = detect(be, a.list, a.out, keys, a.batch, workers=a.workers, rl=rl)
            rl.info(f"detect: {info}")
    elif a.step == "depth":
        info = depth_sample(a.out, a.depth_out, workers=a.workers, rl=rl)
        rl.info(f"depth: {info}")
    elif a.step == "summary":
        rl.info(f"summary: {summary(Path(a.out))}")
    elif a.step == "subset":
        S = subset()
        d = Path(os.environ["DATA_DIR"]) / "processed/fastperc"
        d.mkdir(parents=True, exist_ok=True)
        (d / "subset.json").write_text(json.dumps(S))
        rl.info(f"subset: p5 {len(S['p5'])} images, nusc {len(S['nusc'])} images")
    elif a.step == "eval":
        row = evaluate(a.out, a.tag, a.score, rl.dir, a.contact)
        rl.info(f"eval {a.tag}: {json.dumps(row, default=float)}")
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
