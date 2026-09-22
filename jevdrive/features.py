"""Step 2: frozen single-frame features for every CAM_FRONT keyframe.

qwen    Qwen3-VL-4B-Instruct, BF16. Input is the chat template with the image only (no text prompt) and the
        assistant generation prompt, so the last token is where the answer would start. Features:
          vit_mean     ViT output before the patch merger, mean over patches
          vis_mean     merger output (the visual tokens the LLM sees), mean over image tokens
          Lxx_mean     residual stream after decoder layer xx, mean over image tokens
                       (layers 1-3: before that layer's DeepStack visual injection is added)
          Lxx_last     same, last token
dinov2  DINOv2 ViT-B/14 on the full frame resized to 252x448 (no center crop): cls, patch_mean
siglip2 SigLIP2 so400m/14 at its native 384x384: pooled (the attention-pooling head the model is trained
        with) and patch_mean. The frame is squashed to a square, as SigLIP2 was pretrained; keeping the
        aspect would mean padding, which is not what it saw.
vjepa2  V-JEPA 2 ViT-L, the one control that takes a *clip*: CLIP_FRAMES frames spanning CLIP_SPAN seconds
        of history, ending at the keyframe, read from sweeps/CAM_FRONT at ~12 Hz (Clips). Pooled over all
        tokens, and over the last frame's tokens alone. Its row is confounded with history by construction
        (decisions 12): it sees time that the single-frame backbones do not, so it is evidence about
        feeding time, not a backbone ranking.

Layout: processed/nuscenes/<version>/features/<set>/{index.parquet, <name>.npy (n, d) float16, meta.json},
where <set> is the backbone name plus a suffix for non-default input size (e.g. qwen_w800).
Images are decoded and preprocessed in DataLoader workers; pinned batches overlap H2D copies with compute, and
results go back to the host asynchronously and are written by a separate thread, so the GPU never waits on I/O.

QwenFeatures also takes several images per forward (`n_images`), which jevdrive.waymo uses for the three
Waymo front cameras; `extract` takes the Dataset class, so another dataset only has to supply items and a reader.
"""
import gc
import json
import os
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import dataroot, get_logger, processed_dir

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from scripts/download_models.sh; the hub hangs from the box
log = get_logger(__name__)
QWEN, DINO = "Qwen/Qwen3-VL-4B-Instruct", "facebook/dinov2-base"
SIGLIP, VJEPA = "google/siglip2-so400m-patch14-384", "facebook/vjepa2-vitl-fpc64-256"
CLIP_FRAMES, CLIP_SPAN = 64, 5.25  # V-JEPA 2's native frames per clip, and the history they span at ~12 Hz
DEV = "cuda"


class Frames(Dataset):
    """nuScenes: one image file per sample, read from `dataroot()`."""

    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(dataroot() / self.paths[i]).convert("RGB"))


class Clips(Dataset):
    """nuScenes: one clip of image paths per sample, for backbones that take a video."""

    def __init__(self, clips, transform):
        self.clips, self.transform = [list(c) for c in clips], transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.transform([Image.open(dataroot() / p).convert("RGB") for p in self.clips[i]])


def clip_paths(kf: pd.DataFrame, cam: pd.DataFrame, frames: int = CLIP_FRAMES, span: float = CLIP_SPAN,
               tol: float | None = None):
    """For every keyframe, `frames` CAM_FRONT image paths evenly spaced over the `span` seconds of history
    ending at it, each taken as the nearest actual frame to the requested time.

    A clip counts as full when its whole span lies inside the scene and every slot landed within `tol` of the
    time asked for (default one native frame step: at ~12 Hz the requested spacing is itself about one step,
    so the phase drifts across the clip and half a step is fractionally too tight to ever hold).
    Slots before the start of the scene clamp to its first frame -- the usual "repeat the oldest" padding.
    Returns (paths (n, frames) object array, full (n,) bool). Rows are kept either way and stay aligned with
    `kf`, so a padded clip is marked, never dropped.
    """
    step = float(np.median(np.diff(np.sort(cam.timestamp.to_numpy() * 1e-6))))
    tol = step if tol is None else tol
    want = np.arange(frames) * (span / (frames - 1)) - span  # -span .. 0
    out = np.empty((len(kf), frames), object)
    full, err = np.zeros(len(kf), bool), np.zeros(len(kf))
    by_scene = dict(tuple(cam.groupby("scene")))
    for scene, g in kf.groupby("scene"):
        c = by_scene[scene]
        ts, paths = c.timestamp.to_numpy() * 1e-6, c.path.to_numpy()
        t = kf.timestamp.to_numpy()[g.index][:, None] * 1e-6 + want
        j = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - t) <= np.abs(ts[j] - t), j - 1, j)
        d = np.abs(ts[j] - t)
        out[g.index], err[g.index] = paths[j], d.max(1)
        full[g.index] = (d.max(1) <= tol) & (t[:, 0] >= ts[0] - tol)
    log.info("clips: %d frames over %.2f s (native step %.0f ms), %d / %d keyframes cover the whole span "
             "within %.0f ms; worst slot error %.0f ms, median %.0f ms", frames, span, step * 1e3,
             int(full.sum()), len(kf), tol * 1e3, err.max() * 1e3, float(np.median(err)) * 1e3)
    return out, full


def _vision_attention(self, hidden_states, cu_seqlens, position_embeddings, **_):
    """Qwen3VLVisionAttention.forward for a batch of same-size images: one batched SDPA call. The stock sdpa path
    splits per image with cu_seqlens.tolist(), a host sync in each of the 24 blocks."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    L, b = hidden_states.shape[0], cu_seqlens.numel() - 1
    q, k, v = self.qkv(hidden_states).reshape(L, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q, k, v = (t.reshape(b, L // b, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scaling)
    return self.proj(o.transpose(1, 2).reshape(L, -1))


class QwenFeatures:
    """Runs the vision tower and decoder layers directly (same modules and math as Qwen3VLModel.forward) so that
    per-layer pooling happens inline and nothing forces a host sync. Every frame has the same size and prompt,
    so position ids, rotary tables and the image-token span are computed once per batch shape."""

    def __init__(self, n_layer_probes: int = 4, width: int | None = None, long_side: int | None = None,
                 n_images: int = 1, compile: bool = True, layers: list[int] | None = None,
                 model_id: str = QWEN, grid_hw: tuple[int, int] | None = None, pooled: bool = True,
                 device_map: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model_id, self.grid_hw, self.pooled = model_id, grid_hw, pooled
        # device_map streams the shards straight onto the card. The 32B checkpoint is 64 GB in bf16, and
        # loading it to host memory first would blow past the RAM this box can spare while another job runs.
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map=device_map)
        self.model = model.model.to(DEV).eval()  # skip lm_head: we only need hidden states
        self.image_token_id = model.config.image_token_id
        self.prompt = self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}] * n_images}], add_generation_prompt=True, tokenize=False)
        self.width, self.long_side, self.n_images, self.n_image_tokens = width, long_side, n_images, 0
        n = model.config.text_config.num_hidden_layers
        # `layers` names the decoder layers to read explicitly, otherwise they are evenly spaced (e.g. 9 18 27 36).
        # The forward stops after the deepest one either way, so layers=[L] is a real early exit at layer L.
        self.layers = list(layers or np.linspace(0, n, n_layer_probes + 1).round().astype(int)[1:].tolist())
        vis, lm = self.model.visual, self.model.language_model
        for blk in vis.blocks:
            blk.attn.forward = types.MethodType(_vision_attention, blk.attn)
        # compile one block / layer at a time: fuses the norm, rotary and activation elementwise ops;
        # all blocks share one graph (weights are graph inputs), so it compiles once per batch shape
        wrap = torch.compile if compile else (lambda m: m)
        self.vis_blocks, self.lm_layers = [wrap(b) for b in vis.blocks], [wrap(l) for l in lm.layers[:max(self.layers)]]
        self.static = {}

    def _resize(self, img):
        """`long_side` fixes the longer side (portrait or landscape), `width` the width. Neither: native pixels."""
        s = self.long_side / max(img.size) if self.long_side else self.width / img.width if self.width else None
        return img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC) if s else img

    def transform(self, img):
        imgs = [self._resize(i) for i in (img if isinstance(img, (list, tuple)) else [img])]
        assert len(imgs) == self.n_images, f"{len(imgs)} images but the prompt has {self.n_images}"
        b = self.proc(text=[self.prompt], images=imgs, return_tensors="pt")
        return b["input_ids"][0], b["mm_token_type_ids"][0], b["pixel_values"], b["image_grid_thw"]

    @staticmethod
    def collate(items):
        ids, mm, pv, grid = zip(*items)
        return torch.stack(ids), torch.stack(mm), torch.cat(pv), torch.cat(grid)  # one frame size: no padding

    def _static(self, key, ids, mm, grid):
        """Batch-shape constants, from the same transformers helpers the stock forward calls on every batch."""
        from transformers.vision_utils import (get_vision_attention_seqlens,
                                               get_vision_interpolation_indices_and_weights, get_vision_position_ids)
        if key not in self.static:
            vis, cfg = self.model.visual, self.model.visual.config
            idx, w = get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=vis.num_grid_per_side, mode=vis.interpolation_mode,
                align_corners=vis.interpolation_align_corners, spatial_merge_size=cfg.spatial_merge_size)
            pos_embeds = (vis.pos_embed(idx) * w[:, :, None]).sum(1).to(torch.bfloat16)
            rot = vis.rotary_pos_emb(pos_embeds, get_vision_position_ids(grid, vis.spatial_merge_size))
            cu_seqlens, _ = get_vision_attention_seqlens(grid, cfg)
            ipos = (ids[0] == self.image_token_id).nonzero().squeeze(1)  # not contiguous for n_images > 1
            assert (ids == ids[0]).all(), "frames must share one prompt layout"
            self.n_image_tokens = int(ipos.numel())
            emb = self.model.get_input_embeddings()(ids)
            pos = self.model.compute_3d_position_ids(input_ids=ids, inputs_embeds=emb, image_grid_thw=grid,
                                                     attention_mask=torch.ones_like(ids), mm_token_type_ids=mm)
            lm_rot = self.model.language_model.rotary_emb(emb, pos)
            self.static[key] = pos_embeds, rot, cu_seqlens, ipos, lm_rot
        return self.static[key]

    @torch.inference_mode()
    def __call__(self, batch) -> dict[str, torch.Tensor]:
        key = (*batch[0].shape, *batch[3].shape, *batch[3][0].tolist())  # from the host copy: no sync
        ids, mm, pv, grid = (t.to(DEV, non_blocking=True) for t in batch)
        pos_embeds, rot, cu_seqlens, ipos, lm_rot = self._static(key, ids, mm, grid)
        vis, lm, b = self.model.visual, self.model.language_model, len(ids)

        x = vis.patch_embed(pv) + pos_embeds
        deepstack = []
        for i, blk in enumerate(self.vis_blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=rot)
            if i in vis.deepstack_visual_indexes:
                deepstack.append(vis.deepstack_merger_list[vis.deepstack_visual_indexes.index(i)](x).view(b, -1, lm.config.hidden_size))
        img = vis.merger(x).view(b, -1, lm.config.hidden_size)
        out = {"vit_mean": x.view(b, -1, x.shape[-1]).float().mean(1), "vis_mean": img.float().mean(1)}

        if not self.pooled:
            out = {}
        h = lm.embed_tokens(ids)
        h[:, ipos] = img
        for i, layer in enumerate(self.lm_layers):
            h = layer(h, position_embeddings=lm_rot)  # attention_mask=None -> causal SDPA, as for an all-ones mask
            if i + 1 in self.layers:
                if self.pooled:
                    out[f"L{i + 1:02d}_mean"] = h[:, ipos].sum(1, dtype=torch.float32) / ipos.numel()
                    out[f"L{i + 1:02d}_last"] = h[:, -1].float()
                if self.grid_hw is not None:
                    out[f"L{i + 1:02d}_grid"] = self._grid(h[:, ipos], grid)
            if i < len(deepstack):  # DeepStack: early ViT features are added to the image tokens after layers 1-3
                h[:, ipos] += deepstack[i]
        return out

    def _grid(self, tok: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Per-camera token map average-pooled to `grid_hw`, flattened to one row.

        `tok` is (b, n_images * gh * gw, d) with the cameras laid out one after another in the order the
        prompt lists them, and `grid` holds each image's patch grid, which the merger halves in both axes.
        Pooling on the GPU keeps the stored feature at 144 x 2560 per frame instead of 3060 x 2560: the same
        spatial layout at 1/21 of the bytes, which is what makes a 20k-frame grid cache 15 GB instead of 314.
        """
        b, d, (H, W) = len(tok), tok.shape[-1], self.grid_hw
        gh, gw = int(grid[0, 1]) // 2, int(grid[0, 2]) // 2
        x = tok.view(b * self.n_images, gh, gw, d).permute(0, 3, 1, 2).float()
        return torch.nn.functional.adaptive_avg_pool2d(x, (H, W)).permute(0, 2, 3, 1).reshape(b, -1)


class QwenVideoFeatures(QwenFeatures):
    """Qwen3-VL over a *clip*, through its own video path, instead of over independent frames.

    P3(d'') exists to separate two things that arm (d) confounds: V-JEPA 2 beats the single-frame arms and it
    differs from them in two ways at once -- it is trained with a JEPA objective **and** it is shown time. Put
    the same clip through Qwen3-VL's video path and the objective is held fixed while the input changes, so a
    move on pre-onset attributable to time alone would show up here too.

    The difference from the image path is not cosmetic. Qwen3-VL's ViT folds `temporal_patch_size` frames
    into the patch embedding, so a T-frame clip costs T times the patches but only T/2 the LLM tokens of T
    independent images: the video path is the model's own notion of a moving scene, not T pictures.
    `grid_thw` carries t > 1 and the placeholder is the video token, and everything else -- the merger, the
    DeepStack injections, the per-layer pooling -- is the same code as the image path, so `Lxx_mean` from this
    class is the same statistic over the same residual stream as `qwen_front3`'s.
    """

    def __init__(self, frames: int = 4, n_videos: int = 3, **kw):
        self.frames, self.n_videos = frames, n_videos
        super().__init__(n_images=n_videos, **kw)
        self.prompt = self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}] * n_videos}], add_generation_prompt=True,
            tokenize=False)
        cfg = self.model.config if hasattr(self.model, "config") else None
        self.image_token_id = getattr(cfg, "video_token_id", None) or self.image_token_id

    def transform(self, clips):
        """`clips` is one list of PIL frames per camera, oldest first, as `waymo.Shards` hands them over."""
        vids = [[self._resize(f) for f in c] for c in clips]
        assert len(vids) == self.n_videos and all(len(v) == self.frames for v in vids), \
            f"expected {self.n_videos} clips of {self.frames} frames, got {[len(v) for v in vids]}"
        b = self.proc(text=[self.prompt], videos=vids, return_tensors="pt")
        return b["input_ids"][0], b["mm_token_type_ids"][0], b["pixel_values_videos"], b["video_grid_thw"]

    def _static(self, key, ids, mm, grid):
        """Same constants as the image path, but the 3D position ids are told these are videos."""
        if key in self.static:
            return self.static[key]
        real, self.model.compute_3d_position_ids = self.model.compute_3d_position_ids, None

        def as_video(input_ids, inputs_embeds, image_grid_thw, attention_mask, mm_token_type_ids):
            return real(input_ids=input_ids, inputs_embeds=inputs_embeds, video_grid_thw=image_grid_thw,
                        attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids)

        self.model.compute_3d_position_ids = lambda **kw: as_video(**kw)
        try:
            out = super()._static(key, ids, mm, grid)
        finally:
            self.model.compute_3d_position_ids = real
        return out


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


class SiglipFeatures:
    """SigLIP2 vision tower at its native square input. Separates language alignment from pure vision."""

    def __init__(self, size: int = 384):
        from torchvision.transforms import v2
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(SIGLIP, dtype=torch.bfloat16).vision_model.to(DEV).eval()
        self.tf = v2.Compose([v2.PILToTensor(), v2.Resize((size, size), antialias=True),
                              v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)])

    def transform(self, img):
        return self.tf(img)

    collate = staticmethod(torch.stack)

    @torch.inference_mode()
    def __call__(self, x):
        o = self.model(pixel_values=x.to(DEV, torch.bfloat16, non_blocking=True))
        return {"pooled": o.pooler_output.float(), "patch_mean": o.last_hidden_state.float().mean(1)}


class VJepaFeatures:
    """V-JEPA 2 over a clip of CLIP_FRAMES frames. `transform` receives the clip as a list of PIL images."""

    def __init__(self, size: int = 256, frames: int = CLIP_FRAMES, model_id: str = VJEPA):
        from torchvision.transforms import v2
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        self.model_id, self.frames = model_id, frames
        self.tf = v2.Compose([v2.PILToTensor(), v2.Resize((size, size), antialias=True),
                              v2.ToDtype(torch.float32, scale=True),
                              v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def transform(self, imgs):
        assert len(imgs) == self.frames, f"{len(imgs)} frames but the model wants {self.frames}"
        return torch.stack([self.tf(i) for i in imgs])  # (T, 3, H, W)

    collate = staticmethod(torch.stack)

    @torch.inference_mode()
    def __call__(self, x):
        h = self.model.get_vision_features(x.to(DEV, torch.bfloat16, non_blocking=True)).float()
        per_frame = h.shape[1] // (self.frames // 2)  # tubelets of 2 frames, so the last tubelet is "now"
        return {"mean": h.mean(1), "last_mean": h[:, -per_frame:].mean(1)}


BACKBONES = {"qwen": QwenFeatures, "dinov2": DinoFeatures, "siglip2": SiglipFeatures, "vjepa2": VJepaFeatures}


def free_gpu():
    """Call after dropping a backbone: collect reference cycles (compiled graphs, bound methods) and cached blocks."""
    gc.collect()
    torch.cuda.empty_cache()


def extract(fx, items, batch_size: int, workers: int, out_dir: Path | None = None, rl=None, tag: str = "",
            dataset=Frames) -> dict:
    """Run `fx` over all items of `dataset`; write float16 arrays to out_dir (None = benchmark only).
    With a RunLog `rl`, per-batch throughput goes to TensorBoard / events.jsonl under `tag`.
    Each batch's features are packed into one float16 tensor and copied to pinned host memory without blocking;
    a writer thread waits for that copy and fills the memmaps while the GPU already runs the next batches."""
    items = list(items)
    loader = DataLoader(dataset(items, fx.transform), batch_size=batch_size, num_workers=workers,
                        collate_fn=fx.collate, pin_memory=workers > 0, prefetch_factor=4 if workers else None)
    arrays, cols, i, t_first, n_first = {}, {}, 0, None, 0

    def write(host, done, i, cols):
        done.synchronize()
        if out_dir is None:
            return
        for k, (a, b) in cols.items():
            if k not in arrays:
                arrays[k] = np.lib.format.open_memmap(out_dir / f"{k}.npy", "w+", np.float16, (len(items), b - a))
            arrays[k][i:i + len(host)] = host[:, a:b].numpy()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = t_prev = time.perf_counter()
    bar = tqdm(total=len(items), desc=tag or "extract", unit="frame", dynamic_ncols=True)
    with ThreadPoolExecutor(1) as writer:
        pending = []
        for batch in loader:
            feats = fx(batch)
            dims = [v.shape[1] for v in feats.values()]
            cols = {k: (e - d, e) for k, d, e in zip(feats, dims, np.cumsum(dims))}
            host = torch.cat([v.to(torch.float16) for v in feats.values()], 1).to("cpu", non_blocking=True)  # pinned
            done = torch.cuda.Event()
            done.record()
            n = len(host)
            pending.append(writer.submit(write, host, done, i, cols))
            while len(pending) > 4:  # bound how far the host runs ahead (and the pinned memory it holds)
                pending.pop(0).result()
            bar.update(n)
            if rl is not None:
                t_now = time.perf_counter()
                rl.scalar(f"{tag}/frames_per_s", n / (t_now - t_prev), i + n)
                t_prev = t_now
            i += n
            if t_first is None:
                pending.pop(0).result()  # warm-up (compile) done: start the steady-state clock
                t_first, n_first = time.perf_counter(), n
        for f in pending:
            f.result()
    wall = time.perf_counter() - t0
    bar.close()
    for a in arrays.values():
        a.flush()
    steady = (time.perf_counter() - t_first) / max(i - n_first, 1)
    return {"n": i, "batch_size": batch_size, "workers": workers, "wall_s": wall, "ms_per_frame": 1e3 * steady,
            "frames_per_s": 1 / steady, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "bytes_per_sample": 2 * sum(b - a for a, b in cols.values())}  # float16, whether or not it is stored


def set_name(backbone: str, width: int | None = None, long_side: int | None = None, **_) -> str:
    return backbone + (f"_w{width}" if width else "") + (f"_l{long_side}" if long_side else "")


def run(version: str, backbone: str, batch_size: int, workers: int, force: bool = False, rl=None, **kw) -> dict:
    kw = {k: v for k, v in kw.items() if v is not None}
    out = processed_dir(version) / "features" / set_name(backbone, **kw)
    if (out / "meta.json").exists() and not force:
        meta = json.loads((out / "meta.json").read_text())
        if all(meta.get(k) == v for k, v in kw.items()):
            log.info("%s features exist at %s, skipping (use --force to redo)", out.name, out)
            return meta
        log.info("%s features at %s were made with other args, recomputing", out.name, out)
    kf = pd.read_parquet(processed_dir(version) / "keyframes.parquet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)

    idx, items, ds, args = kf[["sample_token", "sd_token"]], kf.path, Frames, dict(kw)
    if backbone == "vjepa2":  # a clip per keyframe, from sweeps/CAM_FRONT
        cam = pd.read_parquet(processed_dir(version) / "cam_front.parquet")
        items, full = clip_paths(kf, cam, kw.get("frames", CLIP_FRAMES), kw.get("span", CLIP_SPAN))
        idx, ds = idx.assign(clip_full=full), Clips
        args.pop("span", None)  # clip_paths' argument, not the model's; `kw` still carries it into meta

    t0 = time.perf_counter()
    fx = BACKBONES[backbone](**args)
    load_s = time.perf_counter() - t0
    stats = extract(fx, items, batch_size, workers, out, rl, f"features/{out.name}", dataset=ds)
    idx.to_parquet(out / "index.parquet")
    meta = {"backbone": backbone, "model": {"qwen": QWEN, "dinov2": DINO, "siglip2": SIGLIP,
                                            "vjepa2": VJEPA}[backbone], "load_s": load_s, **stats,
            "features": sorted(p.stem for p in out.glob("*.npy")), **kw}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: %d frames, %.1f ms/frame, peak VRAM %.2f GB, %.1f KB/sample -> %s", out.name, stats["n"],
             stats["ms_per_frame"], stats["peak_vram_gb"], stats["bytes_per_sample"] / 1024, out)
    del fx
    free_gpu()
    return meta


def bench(version: str, backbone: str, configs: list[tuple[int, int]], n: int = 96, rl=None, **kw) -> list[dict]:
    """Extraction speed and peak VRAM for (batch_size, workers) configs on the first n frames; nothing is saved."""
    paths = pd.read_parquet(processed_dir(version) / "keyframes.parquet").path[:n]
    fx = BACKBONES[backbone](**kw)
    rows = []
    for bs, w in configs:
        r = {"backbone": backbone, **extract(fx, paths, bs, w, rl=rl, tag=f"bench/{backbone}_bs{bs}_w{w}")}
        log.info("bench %s bs=%d workers=%d: %.1f ms/frame, peak VRAM %.2f GB", backbone, bs, w, r["ms_per_frame"],
                 r["peak_vram_gb"])
        rows.append(r)
    del fx
    free_gpu()
    return rows
