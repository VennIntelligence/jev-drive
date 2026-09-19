"""Step 2: frozen single-frame features for every CAM_FRONT keyframe.

qwen    Qwen3-VL-4B-Instruct, BF16. Input is the chat template with the image only (no text prompt) and the
        assistant generation prompt, so the last token is where the answer would start. Features:
          vit_mean     ViT output before the patch merger, mean over patches
          vis_mean     merger output (the visual tokens the LLM sees), mean over image tokens
          Lxx_mean     residual stream after decoder layer xx, mean over image tokens
          Lxx_last     same, last token
dinov2  DINOv2 ViT-B/14 on the full frame resized to 252x448 (no center crop): cls, patch_mean

Layout: processed/nuscenes/<version>/features/<backbone>/{index.parquet, <name>.npy (n, d) float16, meta.json}.
Images are decoded and preprocessed in DataLoader workers; pinned batches overlap H2D copies with compute.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import dataroot, get_logger, processed_dir

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from scripts/download_models.sh; the hub hangs from the box
log = get_logger(__name__)
QWEN, DINO = "Qwen/Qwen3-VL-4B-Instruct", "facebook/dinov2-base"
DEV = "cuda"


class Frames(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(dataroot() / self.paths[i]).convert("RGB"))


class QwenFeatures:
    def __init__(self, n_layer_probes: int = 4, width: int | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(QWEN)
        model = AutoModelForImageTextToText.from_pretrained(QWEN, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model = model.model.to(DEV).eval()  # skip lm_head: we only need hidden states
        self.image_token_id = model.config.image_token_id
        self.prompt = self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}],
                                                    add_generation_prompt=True, tokenize=False)
        self.width = width
        n = model.config.text_config.num_hidden_layers
        self.layers = np.linspace(0, n, n_layer_probes + 1).round().astype(int)[1:].tolist()  # e.g. 9 18 27 36
        self.model.visual.register_forward_hook(self._visual_hook)
        for k in self.layers:
            self.model.language_model.layers[k - 1].register_forward_hook(self._layer_hook(k))

    def transform(self, img):
        if self.width:
            img = img.resize((self.width, round(img.height * self.width / img.width)), Image.BICUBIC)
        b = self.proc(text=[self.prompt], images=[img], return_tensors="pt")
        return b["input_ids"][0], b["pixel_values"], b["image_grid_thw"][0]

    @staticmethod
    def collate(items):
        ids, pv, grid = zip(*items)
        return torch.stack(ids), torch.cat(pv), torch.stack(grid)  # all frames share one size: no padding

    def _visual_hook(self, _, __, out):
        b = len(self.mask)
        self.out["vit_mean"] = out.last_hidden_state.view(b, -1, out.last_hidden_state.shape[-1]).float().mean(1)
        self.out["vis_mean"] = out.pooler_output.view(b, -1, out.pooler_output.shape[-1]).float().mean(1)

    def _layer_hook(self, k):
        def hook(_, __, h):
            h = h[0] if isinstance(h, tuple) else h
            m = self.mask.unsqueeze(-1).to(h.dtype)
            self.out[f"L{k:02d}_mean"] = ((h * m).sum(1, dtype=torch.float32) / m.sum(1)).float()
            self.out[f"L{k:02d}_last"] = h[:, -1].float()
        return hook

    @torch.inference_mode()
    def __call__(self, batch):
        ids, pv, grid = (t.to(DEV, non_blocking=True) for t in batch)
        self.mask, self.out = ids == self.image_token_id, {}
        self.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv, image_grid_thw=grid,
                   use_cache=False)
        return self.out


class DinoFeatures:
    def __init__(self, size=(252, 448)):
        from torchvision.transforms import v2
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(DINO, dtype=torch.bfloat16).to(DEV).eval()
        self.tf = v2.Compose([v2.PILToTensor(), v2.Resize(size, antialias=True), v2.ToDtype(torch.float32, scale=True),
                              v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def transform(self, img):
        return self.tf(img)

    collate = staticmethod(torch.stack)

    @torch.inference_mode()
    def __call__(self, x):
        h = self.model(pixel_values=x.to(DEV, torch.bfloat16, non_blocking=True)).last_hidden_state.float()
        return {"cls": h[:, 0], "patch_mean": h[:, 1:].mean(1)}


BACKBONES = {"qwen": QwenFeatures, "dinov2": DinoFeatures}


def extract(fx, paths, batch_size: int, workers: int, out_dir: Path | None = None) -> dict:
    """Run `fx` over all frames; write float16 arrays to out_dir (None = benchmark only). Returns timing stats."""
    loader = DataLoader(Frames(paths, fx.transform), batch_size=batch_size, num_workers=workers, collate_fn=fx.collate,
                        pin_memory=workers > 0, prefetch_factor=4 if workers else None)
    arrays, i, t_first = {}, 0, None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for batch in loader:
        feats = {k: v.to(torch.float16).cpu().numpy() for k, v in fx(batch).items()}
        n = len(next(iter(feats.values())))
        if out_dir is not None:
            for k, v in feats.items():
                if k not in arrays:
                    arrays[k] = np.lib.format.open_memmap(out_dir / f"{k}.npy", "w+", np.float16, (len(paths), v.shape[1]))
                arrays[k][i:i + n] = v
        i += n
        if t_first is None:
            t_first = time.perf_counter()
            n_first = n
    wall = time.perf_counter() - t0
    for a in arrays.values():
        a.flush()
    steady = (time.perf_counter() - t_first) / max(i - n_first, 1)
    return {"n": i, "batch_size": batch_size, "workers": workers, "wall_s": wall, "ms_per_frame": 1e3 * steady,
            "frames_per_s": 1 / steady, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "bytes_per_sample": sum(a.dtype.itemsize * a.shape[1] for a in arrays.values())}


def run(version: str, backbone: str, batch_size: int, workers: int, force: bool = False, **kw) -> dict:
    out = processed_dir(version) / "features" / backbone
    if (out / "meta.json").exists() and not force:
        log.info("%s features exist at %s, skipping (use --force to redo)", backbone, out)
        return json.loads((out / "meta.json").read_text())
    kf = pd.read_parquet(processed_dir(version) / "keyframes.parquet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)

    t0 = time.perf_counter()
    fx = BACKBONES[backbone](**kw)
    load_s = time.perf_counter() - t0
    stats = extract(fx, kf.path, batch_size, workers, out)
    kf[["sample_token", "sd_token"]].to_parquet(out / "index.parquet")
    meta = {"backbone": backbone, "model": QWEN if backbone == "qwen" else DINO, "load_s": load_s, **stats,
            "features": sorted(p.stem for p in out.glob("*.npy")), **{k: v for k, v in kw.items() if v is not None}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: %d frames, %.1f ms/frame, peak VRAM %.2f GB, %.1f KB/sample -> %s", backbone, stats["n"],
             stats["ms_per_frame"], stats["peak_vram_gb"], stats["bytes_per_sample"] / 1024, out)
    del fx
    torch.cuda.empty_cache()
    return meta


def bench(version: str, backbone: str, configs: list[tuple[int, int]], n: int = 96, **kw) -> list[dict]:
    """Extraction speed and peak VRAM for (batch_size, workers) configs on the first n frames; nothing is saved."""
    paths = pd.read_parquet(processed_dir(version) / "keyframes.parquet").path[:n]
    fx = BACKBONES[backbone](**kw)
    rows = []
    for bs, w in configs:
        r = {"backbone": backbone, **extract(fx, paths, bs, w)}
        log.info("bench %s bs=%d workers=%d: %.1f ms/frame, peak VRAM %.2f GB", backbone, bs, w, r["ms_per_frame"],
                 r["peak_vram_gb"])
        rows.append(r)
    del fx
    torch.cuda.empty_cache()
    return rows
