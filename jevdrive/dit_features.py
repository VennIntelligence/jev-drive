"""P3(b) and (c): frozen features read out of a video-generation DiT, one forward, no sampling loop.

The recipe is DriveLaW's (2512.23421, decisions 21 and the round-4 survey), reduced to what a feature
extractor needs: VAE-encode the frame, add flow-matching noise at a fixed level, run the transformer **once**
and hook two blocks on the way through. Nothing is denoised and nothing is generated, so the cost is one
forward rather than the 5-15 s a clip takes to sample.

Three choices this file makes, all of which have to travel with the numbers:

**One camera at a time, three forwards per frame.** The alternative -- tiling the three cameras into one
16:9 canvas -- squashes a 3.3:1 strip into a 1.8:1 frame, so every object is horizontally compressed by
nearly half and the model sees a geometry it was never trained on. Three forwards keep each camera at its
own aspect ratio and keep the arm looking at exactly what `qwen_front3` looks at; the per-camera pooled
vectors are concatenated, so one frame is still one feature row.

**Two noise levels, swept, because the literature is ambiguous about which end "high noise" is.** DriveLaW
reports its best at denoise step t = 1 of its own schedule (the noisiest step it takes) and catastrophic
collapse by t = 10; 2502.07001's rule of thumb is quoted as "timestep about 200 of 1000". In Wan's
flow-matching parameterisation sigma = t / 1000, so t = 200 is 20 % noise -- nearly clean pixels, the end
DriveLaW says collapses. Rather than pick a reading, both are extracted: sigma 0.2 and sigma 0.8.

**Taps at about 1/2 and 2/3 of the depth**, as the survey's tap rule says, read as block outputs and
mean-pooled over tokens.
"""
import json
import os

import numpy as np
import torch

from .common import data_dir, get_logger

# The weights are on the data disk and the box reaches the hub only through a proxy this process does not
# set. Without this, `from_pretrained` HEAD-requests huggingface.co for files it already has, and on a
# window with no proxy that is five retries and a hard failure -- which is exactly how the Wan probe died
# and stalled the whole P3 queue on 2026-09-23.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
log = get_logger(__name__)
DEV = "cuda"
WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NEUTRAL_PROMPT = "a driving scene recorded from a vehicle's forward-facing camera"
SIGMAS = (0.2, 0.8)          # flow-matching noise levels: the two readings of "tap at high noise"
TAPS = (0.5, 2 / 3)          # fraction of the transformer's depth


def prompt_cache(repo: str, prompt: str = NEUTRAL_PROMPT):
    """The text conditioner's state for one fixed prompt, encoded once and reused.

    The prompt never varies -- these are features, not generations -- so the 11 GB text encoder is loaded
    once in its own process-lifetime, its output is written next to the other processed data, and every
    extraction afterwards reads the tensor. This is the single largest saving in the recipe.
    """
    p = data_dir() / "processed" / "dit_prompts"
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{repo.replace('/', '_')}.pt"
    if f.exists():
        return torch.load(f, map_location="cpu")
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", local_files_only=True)
    enc = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", dtype=torch.bfloat16,
                                        local_files_only=True).to(DEV).eval()
    b = tok([prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(DEV)
    with torch.inference_mode():
        h = enc(b.input_ids, attention_mask=b.attention_mask).last_hidden_state
    h = (h * b.attention_mask.unsqueeze(-1)).float().cpu()          # padding zeroed, as the pipeline does
    torch.save(h, f)
    del enc
    torch.cuda.empty_cache()
    log.info("prompt state for %r cached at %s: %s", prompt, f, tuple(h.shape))
    return h


class WanDiTFeatures:
    """Wan2.2-TI2V-5B's transformer as a frozen encoder: one forward per camera per noise level."""

    def __init__(self, repo: str = WAN, size=(704, 1280), n_images: int = 3, sigmas=SIGMAS, taps=TAPS,
                 prompt: str = NEUTRAL_PROMPT, seed: int = 0):
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from torchvision.transforms import v2
        self.model_id, self.n_images, self.sigmas, self.seed = repo, n_images, tuple(sigmas), seed
        self.n_image_tokens = 0
        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32,
                                                    local_files_only=True).to(DEV).eval()
        self.tr = WanTransformer3DModel.from_pretrained(repo, subfolder="transformer",
                                                        torch_dtype=torch.bfloat16,
                                                        local_files_only=True).to(DEV).eval()
        n = len(self.tr.blocks)
        self.taps = sorted({min(int(round(t * n)), n - 1) for t in taps})
        self.buf = {}
        for i in self.taps:
            self.tr.blocks[i].register_forward_hook(self._hook(i))
        z = getattr(self.vae.config, "z_dim", None) or self.tr.config.in_channels
        m, sd = self.vae.config.get("latents_mean"), self.vae.config.get("latents_std")
        self.mean = torch.tensor(m, device=DEV).view(1, z, 1, 1, 1) if m else torch.zeros(1, device=DEV)
        self.inv_std = 1.0 / torch.tensor(sd, device=DEV).view(1, z, 1, 1, 1) if sd else torch.ones(1, device=DEV)
        self.text = prompt_cache(repo, prompt).to(DEV, torch.bfloat16)
        self.tf = v2.Compose([v2.PILToTensor(), v2.Resize(size, antialias=True),
                              v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)])
        log.info("%s: %d blocks, tapping %s; sigmas %s; latent z_dim %d", repo, n, self.taps, self.sigmas, z)

    def _hook(self, i):
        def f(_m, _in, out):
            self.buf[i] = (out[0] if isinstance(out, tuple) else out).float().mean(1)   # mean over tokens
        return f

    def transform(self, img):
        imgs = img if isinstance(img, (list, tuple)) else [img]
        assert len(imgs) == self.n_images, f"{len(imgs)} images but this set expects {self.n_images}"
        return torch.stack([self.tf(i) for i in imgs])          # (n_images, 3, H, W)

    collate = staticmethod(torch.stack)

    @torch.inference_mode()
    def __call__(self, batch) -> dict[str, torch.Tensor]:
        b = len(batch)
        x = batch.to(DEV, torch.float32, non_blocking=True).flatten(0, 1).unsqueeze(2)   # (b*cams, 3, 1, H, W)
        z = self.vae.encode(x).latent_dist.mode()
        z = ((z - self.mean) * self.inv_std).to(torch.bfloat16)
        self.n_image_tokens = int(np.prod(z.shape[2:]) // (self.tr.config.patch_size[1] ** 2))
        g = torch.Generator(device=DEV).manual_seed(self.seed)
        noise = torch.randn(z.shape, generator=g, device=DEV, dtype=z.dtype)
        txt = self.text.expand(len(z), -1, -1)
        out = {}
        for s in self.sigmas:
            zt = (1 - s) * z + s * noise                         # flow matching between data and noise
            t = torch.full((len(z),), s * 1000, device=DEV, dtype=torch.float32)
            self.buf.clear()
            self.tr(hidden_states=zt, timestep=t, encoder_hidden_states=txt, return_dict=False)
            for i in self.taps:                                  # cameras concatenated: one row per frame
                out[f"s{s:g}_b{i:02d}_mean"] = self.buf[i].view(b, -1)
        return out


def probe(repo: str = WAN, n: int = 4, **kw) -> dict:
    """Load the model and time one batch, before anything is written. Prints the feature names and widths."""
    import time
    from PIL import Image
    t0 = time.perf_counter()
    fx = WanDiTFeatures(repo, **kw)
    load_s = time.perf_counter() - t0
    img = [Image.new("RGB", (1079, 972)) for _ in range(fx.n_images)]
    batch = fx.collate([fx.transform(img) for _ in range(n)])
    for i in range(2):
        t = time.perf_counter()
        o = fx(batch)
        torch.cuda.synchronize()
        log.info("pass %d: %.0f ms for %d frames (%.0f ms/frame)", i, 1e3 * (time.perf_counter() - t), n,
                 1e3 * (time.perf_counter() - t) / n)
    return {"load_s": load_s, "features": {k: tuple(v.shape) for k, v in o.items()},
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(json.dumps(probe(), indent=2, default=str))
