"""SAM 3.1 open-vocabulary detection over image lists (fusion diagnostics Q4, todos/2026-09-25-fusion-diagnostics.md).

Runs in envs/sam3 (Python 3.12, facebookresearch/sam3 at $DATA_DIR/third_party/sam3, scripts/sam3_setup.sh).
The model is SAM 3.1's detector: the image-grounding half of the multiplex checkpoint (`detector.*` keys), built
exactly as `sam3.model_builder.build_sam3_multiplex_video_predictor` builds it and run through the same
preprocessing and postprocessing as the repo's `Sam3Processor` (1008 x 1008 resize, mean/std 0.5, score =
sigmoid(logit) x sigmoid(presence), mask = bilinear upsample to the original size, sigmoid > 0.5), under bf16
autocast with TF32 as the repo's own examples do. What we add is batching only: B images x P text prompts go through
one `forward_grounding` call (one find query per (image, prompt) pair, the images encoded once), and
`check_batched` verifies that against the single-image `Sam3Processor` path.

  detect   an image list (parquet: key, and either path or shard/off/len into a WOD shard) -> shards of instance
           rows (key, prompt, score, box, mask area, ground-contact pixel, COCO RLE), resumable per shard
  check    batched vs single-image equivalence on a few images (pre-registered check (c))
  latency  batch-1 image-mode latency, 1 and 3 cameras x 6 prompts (Q4d)

Ground-contact pixel: the lowest mask row (the object's footprint in a forward camera), u = mean column of the mask
pixels in the lowest CONTACT_ROWS rows, v = that lowest row + 0.5.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

PROMPTS = ("pedestrian", "cyclist", "vehicle", "cone", "debris", "emergency vehicle")
RES = 1008
KEEP_SCORE = 0.3          # rows stored; the analysis reads the repo's default 0.5 and uses the rest for sweeps
CONTACT_ROWS = 3
AMP = os.environ.get("SAM_AMP", "bf16")   # "bf16" (the repo examples) or "fp32" (autocast off; TF32 matmuls stay on)


def _amp(amp: str | None = None):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp or AMP) == "bf16")


def ckpt_path() -> Path:
    d = Path(os.environ["DATA_DIR"]) / "models"
    for p in (d / "sam3.1" / "sam3.1_multiplex.pt", d / "sam3.1-fast" / "sam3.1_multiplex.pt"):
        if p.exists():
            return p
    raise FileNotFoundError("sam3.1_multiplex.pt not found under $DATA_DIR/models/sam3.1{,-fast}")


def build(device: str = "cuda", compile_mode=None):
    """SAM 3.1's detector with its checkpoint weights; returns (model, load report). compile_mode: the repo's own
    backbone compile switch (None = off, as in the fusion runs)."""
    import pkg_resources
    import torch
    from sam3 import model_builder as mb
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri
    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    fa3 = False                                         # flash-attn-3 is an optional extra we do not install
    backbone = SAM3VLBackboneTri(scalp=0, visual=mb._create_multiplex_tri_backbone(compile_mode, fa3, False),
                                 text=mb._create_text_encoder(bpe))
    model = Sam3MultiplexDetector(
        num_feature_levels=1, backbone=backbone, transformer=mb._create_sam3_transformer(use_fa3=fa3),
        segmentation_head=mb._create_segmentation_head(use_fa3=fa3), semantic_segmentation_head=None,
        input_geometry_encoder=mb._create_geometry_encoder(), use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=mb._create_dot_product_scoring(), supervise_joint_box_scores=True, is_multiplex=True)
    ck = torch.load(ckpt_path(), map_location="cpu", weights_only=True)
    ck = ck["model"] if isinstance(ck.get("model"), dict) else ck
    if any(k.startswith("sam3_model.") for k in ck):
        ck = {("detector." + k[len("sam3_model."):] if k.startswith("sam3_model.") else k): v for k, v in ck.items()}
    det = {k[len("detector."):]: v for k, v in ck.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(det, strict=False)
    report = {"ckpt": str(ckpt_path()), "detector_keys": len(det), "missing": len(missing), "unexpected": len(unexpected),
              "missing_head": missing[:8], "unexpected_head": unexpected[:8]}
    return model.to(device).eval(), report


class Detector:
    """Images: uint8 CHW tensors on the GPU (any size) -> per image, per prompt: scores, boxes, masks.

    mode "exact" (default): one image and one prompt per `forward_grounding` call, each prompt's text encoded on its
    own -- exactly the repo's `Sam3Processor` path, bit-identical to it (the image encoding is shared across the
    prompts, as in the processor). mode "batched": B images x P prompts in one call. Under bf16 autocast (the model does
    not run without it) every change of batch shape changes the numerics: scores of kept instances move by up to
    ~4e-3 and mask IoU drops to ~0.988 against the processor, which fails the pre-registered check (c), so the batch
    runs "exact" (fusion todo, deviation log)."""

    def __init__(self, model, prompts=PROMPTS, device: str = "cuda", mode: str = "exact", res: int = RES,
                 image_kwargs=None, amp: str | None = None):
        """res: square input size (the processor's `resolution`). image_kwargs: extra `backbone.forward_image`
        arguments; the multiplex detector's defaults, {} for a plain Sam3Image (e.g. EfficientSAM3)."""
        import torch
        from sam3.model.data_misc import FindStage
        self.m, self.prompts, self.dev, self.FindStage, self.mode = model, list(prompts), device, FindStage, mode
        self.res, self.amp = res, amp or AMP
        self.ikw = dict(need_interactive_out=False, need_propagation_out=False) if image_kwargs is None else image_kwargs
        with torch.inference_mode(), _amp(self.amp):
            self.text = model.backbone.forward_text(self.prompts, device=device)
            self.text1 = [model.backbone.forward_text([p], device=device) for p in self.prompts]

    def _prep(self, img):
        """Sam3Processor.transform: uint8 -> Resize(1008, 1008) -> float [0, 1] -> Normalize(0.5, 0.5)."""
        from torchvision.transforms import v2
        x = v2.functional.resize(img, [self.res, self.res])
        return (x.float() / 255.0 - 0.5) / 0.5

    def _ground(self, bo, text, img_ids, txt_ids):
        import torch
        fs = self.FindStage(img_ids=img_ids, text_ids=txt_ids, input_boxes=None, input_boxes_mask=None,
                            input_boxes_label=None, input_points=None, input_points_mask=None)
        bo.update(text)
        out = self.m.forward_grounding(backbone_out=bo, find_input=fs, geometric_prompt=self.m._get_dummy_prompt(len(img_ids)),
                                       find_target=None)
        prob = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1).float()
        return prob, out["pred_boxes"].float(), out["pred_masks"]

    def __call__(self, imgs: list, keep: float = KEEP_SCORE) -> list[list[dict]]:
        """Per image, per prompt: dict of scores (n,), boxes xyxy px (n, 4), masks bool (n, H, W) on the GPU."""
        import torch
        import torch.nn.functional as F
        from sam3.model import box_ops
        B, P = len(imgs), len(self.prompts)
        raw = {}                                               # (b, p) -> (prob (200,), cxcywh (200, 4), mask logits)
        with torch.inference_mode(), _amp(self.amp):
            if self.mode == "batched":
                x = torch.stack([self._prep(i) for i in imgs])
                bo = self.m.backbone.forward_image(x, **self.ikw)
                prob, bxs, ml = self._ground(bo, self.text, torch.arange(B, device=self.dev).repeat_interleave(P),
                                             torch.arange(P, device=self.dev).repeat(B))
                raw = {(b, p): (prob[b * P + p], bxs[b * P + p], ml[b * P + p]) for b in range(B) for p in range(P)}
            else:
                zero = torch.zeros(1, dtype=torch.long, device=self.dev)
                for b in range(B):
                    bo = self.m.backbone.forward_image(self._prep(imgs[b])[None], **self.ikw)
                    for p in range(P):
                        prob, bxs, ml = self._ground(bo, self.text1[p], zero, zero)
                        raw[b, p] = (prob[0], bxs[0], ml[0])
        res = []
        for b in range(B):
            H, W = imgs[b].shape[-2:]
            per = []
            for p in range(P):
                prob, bxs, ml = raw[b, p]
                k = prob > keep
                sc = prob[k]
                bx = box_ops.box_cxcywh_to_xyxy(bxs[k]) * torch.tensor([W, H, W, H], device=self.dev)
                if k.any():
                    mk = F.interpolate(ml[k].unsqueeze(1).float(), (H, W), mode="bilinear", align_corners=False)
                    mk = mk.squeeze(1).sigmoid() > 0.5
                else:
                    mk = torch.zeros((0, H, W), dtype=torch.bool, device=self.dev)
                per.append({"scores": sc, "boxes": bx, "masks": mk})
            res.append(per)
        return res


def contact(masks):
    """(n, H, W) bool on the GPU -> (u, v, area, lowest row) per mask; NaN for empty masks."""
    import torch
    n, H, W = masks.shape
    if n == 0:
        z = np.zeros(0)
        return z, z, z.astype(np.int64)
    rows = masks.any(2)                                           # (n, H)
    ridx = torch.arange(H, device=masks.device)
    low = torch.where(rows, ridx, -1).max(1).values               # lowest row with mask pixels
    band = (ridx[None] > (low[:, None] - CONTACT_ROWS)) & (ridx[None] <= low[:, None])
    m = masks & band[:, :, None]
    cols = torch.arange(W, device=masks.device, dtype=torch.float32)
    cnt = m.sum((1, 2)).float()
    u = (m.float() * cols).sum((1, 2)) / cnt.clamp(min=1)
    u = torch.where(cnt > 0, u, torch.nan)
    v = torch.where(low >= 0, low.float() + 0.5, torch.nan)
    return u.cpu().numpy(), v.cpu().numpy(), masks.sum((1, 2)).cpu().numpy()


# ---------------------------------------------------------------- image lists and reading

def _reader(row) -> bytes:
    if isinstance(row.get("path"), str) and row["path"]:
        return Path(row["path"]).read_bytes()
    with open(row["shard"], "rb") as f:
        f.seek(int(row["off"]))
        return f.read(int(row["len"]))


def decode(blob: bytes):
    """JPEG -> uint8 CHW tensor with PIL, the decoder the repo's Sam3Processor is documented with (nvjpeg on the GPU
    moves scores of kept instances by up to 0.04 against it, so it is not used)."""
    import torch
    from io import BytesIO
    from PIL import Image
    return torch.from_numpy(np.array(Image.open(BytesIO(blob)).convert("RGB"))).permute(2, 0, 1).contiguous()


class _Images:
    """torch Dataset over the image list: decoded uint8 CHW tensors (PIL, in the loader workers)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return i, decode(_reader(self.rows[i]))


def _collate(b):
    return [x[0] for x in b], [x[1] for x in b]


def _rle(masks_cpu: np.ndarray) -> list[str]:
    from pycocotools import mask as mu
    if not len(masks_cpu):
        return []
    r = mu.encode(np.asfortranarray(masks_cpu.transpose(1, 2, 0).astype(np.uint8)))
    return [x["counts"].decode() for x in r]


def detect(image_list: str, out_dir: str, batch: int = 8, shard_size: int = 2000, workers: int = 8, rle: bool = True,
           limit: int | None = None, rl=None, mode: str = "exact", part: str = "0/1") -> dict:
    """Run over an image list, writing out_dir/part-<k>.parquet per `shard_size` images (skipped if present).
    part "i/n": this process takes the shards k with k % n == i; part "claim": processes (on one card or several) claim
    shards one at a time with an atomic mkdir of part-<k>.claim, so faster cards take more shards. Each image's result
    does not depend on which process or card ran it (exact mode is one image per call)."""
    import pandas as pd
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    claim = part == "claim"
    pi, pn = (0, 1) if claim else map(int, part.split("/"))
    t = pd.read_parquet(image_list)
    if limit:
        t = t.iloc[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, rep = build()
    if rl:
        rl.event("model", **rep)
        rl.info(f"model: {rep}")
    det = Detector(model, mode=mode)
    pool = ThreadPoolExecutor(max(1, workers // 2))
    n_img, t0, done = 0, time.time(), 0
    for s0 in range(0, len(t), shard_size):
        dst = out / f"part-{s0 // shard_size:04d}.parquet"
        if (s0 // shard_size) % pn != pi:
            continue
        if dst.exists():
            done += 1
            continue
        if claim:
            try:
                dst.with_suffix(".claim").mkdir()
            except FileExistsError:
                continue
        rows = t.iloc[s0:s0 + shard_size].to_dict("records")
        dl = DataLoader(_Images(rows), batch_size=batch, num_workers=workers, collate_fn=_collate, prefetch_factor=4,
                        persistent_workers=False, pin_memory=True)
        recs, futs = [], []
        ts = time.time()
        for idx, ims in tqdm(dl, desc=dst.name, mininterval=10):
            imgs = [x.to("cuda", non_blocking=True) for x in ims]
            res = det(imgs)
            for b, (i, per) in enumerate(zip(idx, res)):
                key = rows[i]["key"]
                for p, d in zip(det.prompts, per):
                    if not len(d["scores"]):
                        continue
                    u, v, area = contact(d["masks"])
                    bx = d["boxes"].cpu().numpy()
                    sc = d["scores"].cpu().numpy()
                    f = pool.submit(_rle, d["masks"].cpu().numpy()) if rle else None
                    futs.append(f)
                    recs.append({"key": key, "prompt": p, "score": sc, "x0": bx[:, 0], "y0": bx[:, 1], "x1": bx[:, 2],
                                 "y1": bx[:, 3], "area": area, "cu": u, "cv": v,
                                 "H": int(imgs[b].shape[-2]), "W": int(imgs[b].shape[-1])})
            n_img += len(idx)
        cols = []
        for r, f in zip(recs, futs):
            n = len(r["score"])
            d = {k: (np.repeat(v, n) if np.isscalar(v) or isinstance(v, str) else v) for k, v in r.items()}
            d["rle"] = f.result() if f is not None else [""] * n
            cols.append(pd.DataFrame(d))
        df = pd.concat(cols, ignore_index=True) if cols else pd.DataFrame()
        df.to_parquet(dst.with_suffix(".tmp"), index=False)
        dst.with_suffix(".tmp").rename(dst)
        dt = time.time() - ts
        if rl:
            rl.event("shard", part=dst.name, images=len(rows), instances=len(df), seconds=dt)
            rl.scalar("sam/ms_per_image", 1000 * dt / len(rows), s0 // shard_size)
            rl.info(f"{dst.name}: {len(rows)} images, {len(df)} instances, {1000 * dt / len(rows):.1f} ms/image")
    info = {"images": len(t), "new_images": n_img, "skipped_shards": done, "seconds": time.time() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    n_parts = (len(t) + shard_size - 1) // shard_size
    if len(list(out.glob("part-*.parquet"))) == n_parts:
        (out / "done.json").write_text(json.dumps(info, indent=1))
    return info


def check_batched(image_list: str, n: int = 16, batch: int = 8, mode: str = "exact") -> "pd.DataFrame":
    """Pre-registered check (c): the batched path vs the repo's single-image Sam3Processor, per (image, prompt):
    mask IoU of matched instances (Hungarian on box IoU) and score difference."""
    import pandas as pd
    import torch
    from PIL import Image
    from io import BytesIO
    from scipy.optimize import linear_sum_assignment
    from sam3.model.sam3_image_processor import Sam3Processor
    from torchvision.ops import box_iou
    t = pd.read_parquet(image_list).iloc[:n].to_dict("records")
    model, _ = build()
    det, proc = Detector(model, mode=mode), Sam3Processor(model, confidence_threshold=0.5)
    blobs = [_reader(r) for r in t]
    rows = []
    for s in range(0, n, batch):
        imgs = [decode(b).to("cuda") for b in blobs[s:s + batch]]
        res = det(imgs, keep=0.5)
        for j, per in enumerate(res):
            # the processor gets the same decoded tensor the detector got (PIL decode, as documented)
            for src, im in (("same input", imgs[j]),):
              with _amp():
                st = proc.set_image(im)
                for p, d in zip(det.prompts, per):
                    o = proc.set_text_prompt(prompt=p, state=st)
                    a, b = d, {"scores": o["scores"].float(), "boxes": o["boxes"].float(), "masks": o["masks"].squeeze(1)}
                    na, nb = len(a["scores"]), len(b["scores"])
                    row = {"key": t[s + j]["key"], "single": src, "prompt": p, "n_batched": na, "n_single": nb}
                    if na and nb:
                        iou = box_iou(a["boxes"], b["boxes"]).cpu().numpy()
                        ia, ib = linear_sum_assignment(-iou)
                        ma, mb = a["masks"][ia], b["masks"][ib]
                        inter = (ma & mb).sum((1, 2)).float()
                        uni = (ma | mb).sum((1, 2)).float().clamp(min=1)
                        miou = (inter / uni).cpu().numpy()
                        row |= {"mask_iou_min": float(miou.min()), "mask_iou_mean": float(miou.mean()),
                                "score_absdiff_max": float((a["scores"][ia] - b["scores"][ib]).abs().max())}
                    rows.append(row)
    return pd.DataFrame(rows)


def latency(image_list: str, n: int = 200, warm: int = 20, mode: str = "exact") -> dict:
    """Q4d: batch-1 image mode, 6 prompts, 1 camera and 3 cameras (3 images in one call); p50 / p95 ms, peak VRAM.
    Timed from decoded uint8 images in pinned host memory to masks on the GPU (upload + resize + model + mask
    upsampling); JPEG decoding runs in loader workers in the batch and is not on this path."""
    import pandas as pd
    import torch
    t = pd.read_parquet(image_list).iloc[:n + warm].to_dict("records")
    model, _ = build()
    det = Detector(model, mode=mode)
    host = [decode(_reader(r)).pin_memory() for r in t]
    out = {}
    for label, k in (("1 camera", 1), ("3 cameras", 3)):
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for i in range(0, len(host) - k, k):
            torch.cuda.synchronize()
            a = time.perf_counter()
            det([h.to("cuda", non_blocking=True) for h in host[i:i + k]])
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - a))
        ts = np.array(ts[warm // k:])
        out[label] = {"n": len(ts), "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
                      "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    return out


def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("detect", "check", "latency", "load"))
    ap.add_argument("--list", help="image list parquet")
    ap.add_argument("--out", help="detect: output directory")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-rle", action="store_true")
    ap.add_argument("--tag", default="sam")
    ap.add_argument("--mode", default="exact", choices=("exact", "batched"))
    ap.add_argument("--part", default="0/1", help="detect: shards k with k %% n == i")
    ap.add_argument("--shard-size", type=int, default=2000)
    a = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rl = RunLog("fusion_diag", "q4", a.tag)
    rl.event("start", args=vars(a))
    if a.step == "load":
        _, rep = build()
        rl.info(f"load: {rep}")
    elif a.step == "detect":
        info = detect(a.list, a.out, a.batch, a.shard_size, workers=a.workers, rle=not a.no_rle, limit=a.limit, rl=rl, mode=a.mode, part=a.part)
        rl.info(f"detect: {info}")
        rl.event("detect", **info)
    elif a.step == "check":
        r = check_batched(a.list, a.limit or 16, a.batch, a.mode)
        r.to_csv(rl.dir / "check_batched.csv", index=False)
        rl.info("check (c):\n" + r.to_string())
    elif a.step == "latency":
        r = latency(a.list, mode=a.mode)
        (rl.dir / "latency.json").write_text(json.dumps(r, indent=1))
        rl.info(f"latency: {r}")
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
