"""Night queue 2, N4: the fast channel without the lift -- image-plane detection tokens (todos/2026-09-26-night-queue-2.md,
N4 and the [B] 10:12 entry, written before any N4 number).

  detect   (envs/ultralytics) YOLO26x-seg 640 fp16 on E5's image list with a forward pre-hook on the Segment head:
           per kept detection the RoIAlign (1 x 1) of the three neck maps, concatenated (384 + 768 + 768 = 1920)
  tokens   PCA-16 of the detection features (fitted on role == train frames), then per camera the first k = 8
           detections by box bottom (y1, lowest first): camera one-hot, (u, v, w, h) / (W, H), class one-hot, score,
           PCA-16, mask bit -> 3 x 8 x 28 = 672 per row
  fit      E5's student (elicit_e5.train_student, pair-difference loss, no teacher) on op (+) B and op (+) B (+) A,
           seeds 0-2, both models; A = E5's stored student A; one p5_exam.exam over A / B / C
  latency  B's head side (RoIAlign + PCA on the GPU, token build on the CPU, MLP) at batch 1, three cameras

    envs/ultralytics/bin/python -m jevdrive.night2_n4 detect --part 0/6
    .venv/bin/python -m jevdrive.night2_n4 tokens | fit | latency
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
K_TOK, PCA_D, SCORE = 8, 16, 0.25
CAMS = ("front", "front_left", "front_right")
CLASSES = ("pedestrian", "cyclist", "vehicle")
TOK_D = 3 + 4 + 3 + 1 + PCA_D + 1
E5_FIT = "runs/elicitation/e5-fit/20260926-021421"
WEIGHTS = "yolo26x-seg.pt"


def out(*p) -> Path:
    d = data_dir() / "processed" / "night2" / "n4"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


# ---------------------------------------------------------------- detection with per-detection features

class Detector:
    """Ultralytics YOLO (COCO -> three classes, fastperc.COCO_MAP) plus the Segment head's input maps; __call__ returns
    per image (labels, scores, xyxy in original pixels, RoI features (m, 1920) fp16)."""

    def __init__(self, weights: str = WEIGHTS, imgsz: int = 640):
        import torch
        from ultralytics import YOLO
        from .fastperc import COCO_MAP, models_dir
        self.m = YOLO(str(models_dir() / "ultralytics" / weights))
        self.imgsz = imgsz
        head = self.m.model.model[-1]
        self.strides = [float(s) for s in head.stride]
        self._x, self._in, self.checked = None, None, False
        head.register_forward_pre_hook(lambda mod, a: setattr(self, "_x", [t.detach() for t in a[0]]))
        self.m.model.register_forward_pre_hook(lambda mod, a: setattr(self, "_in", a[0].shape))
        self.cmap = {i: COCO_MAP.get(n) for i, n in self.m.names.items()}
        self.torch = torch

    def __call__(self, imgs):
        from torchvision.ops import roi_align
        torch = self.torch
        rs = self.m.predict(imgs, imgsz=self.imgsz, conf=SCORE, half=True, retina_masks=True, verbose=False)
        _, _, hi, wi = self._in
        out_ = []
        for b, r in enumerate(rs):
            lab = np.array([self.cmap.get(c) or "" for c in r.boxes.cls.int().tolist()])
            keep = torch.as_tensor(np.isin(lab, CLASSES), device=r.boxes.data.device)
            xyxy, sc = r.boxes.xyxy[keep].float(), r.boxes.conf[keep].float()
            H, W = r.orig_shape
            g = min(self.imgsz / H, self.imgsz / W)
            nw, nh = int(round(W * g)), int(round(H * g))
            left, top = round((wi - nw) / 2 - 0.1), round((hi - nh) / 2 - 0.1)
            bi = xyxy * g + torch.tensor([left, top, left, top], device=xyxy.device, dtype=xyxy.dtype)
            if not self.checked and len(bi):         # the inverse of Ultralytics' own input -> original mapping
                from ultralytics.utils.ops import scale_boxes
                back = scale_boxes((hi, wi), bi.clone(), (H, W))
                assert float((back - xyxy).abs().max()) < 1.0, "letterbox mapping does not invert"
                self.checked = True
            feats = []
            for x, s in zip(self._x, self.strides):
                if not len(bi):
                    feats.append(torch.zeros((0, x.shape[1]), device=x.device))
                    continue
                f = roi_align(x[b:b + 1].float(), [bi], output_size=1, spatial_scale=1.0 / s, sampling_ratio=2, aligned=True)
                feats.append(f.flatten(1))
            out_.append((lab[np.isin(lab, CLASSES)], sc.cpu().numpy(), xyxy.cpu().numpy(),
                         torch.cat(feats, 1).half().cpu().numpy(), (H, W)))
        return out_


def detect(part: str, batch: int = 12, shard: int = 3000, workers: int = 3):
    """Slice `part` = i/n of E5's image list (interleaved by shard) -> out/dets/part-<k>.parquet + feat-<k>.npy."""
    import torch
    from torch.utils.data import DataLoader
    from .sam_detect import _collate, _Images
    i, n = map(int, part.split("/"))
    t = pd.read_parquet(data_dir() / "processed/elicit_e5/images.parquet")
    det = Detector()
    d = out("dets")
    d.mkdir(exist_ok=True)
    for k, s0 in enumerate(range(0, len(t), shard)):
        if k % n != i or (d / f"part-{k:04d}.parquet").exists():
            continue
        rows = t.iloc[s0:s0 + shard].to_dict("records")
        dl = DataLoader(_Images(rows), batch_size=batch, num_workers=workers, collate_fn=_collate, prefetch_factor=4)
        recs, feats, ts = [], [], time.time()
        for idx, ims in dl:
            res = det([np.ascontiguousarray(x.permute(1, 2, 0).numpy()[:, :, ::-1]) for x in ims])
            for j, (lab, sc, bx, f, (H, W)) in zip(idx, res):
                if len(lab):
                    recs.append(pd.DataFrame({"key": rows[j]["key"], "prompt": lab, "score": sc, "x0": bx[:, 0], "y0": bx[:, 1],
                                              "x1": bx[:, 2], "y1": bx[:, 3], "H": H, "W": W}))
                    feats.append(f)
        df = pd.concat(recs, ignore_index=True)
        np.save(d / f"feat-{k:04d}.npy", np.concatenate(feats))
        df.to_parquet(d / f"part-{k:04d}.tmp", index=False)
        (d / f"part-{k:04d}.tmp").rename(d / f"part-{k:04d}.parquet")
        log.info(f"part {k}: {len(rows)} images, {len(df)} detections, {1000 * (time.time() - ts) / len(rows):.1f} ms/image")
    torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("detect",))
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    if a.step == "detect":
        detect(a.part)


if __name__ == "__main__":
    main()
