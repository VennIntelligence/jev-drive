"""Load check for our feature backbones in the project venv (no benchmark): one forward on the demo frames.

  cd ~/data/jev-drive && UV_PROJECT_ENVIRONMENT=$DATA_DIR/envs/jevdrive HF_HUB_OFFLINE=1 \
    uv run --no-sync python scripts/bench_baselines/load_backbones.py Qwen/Qwen3-VL-8B-Instruct facebook/dinov2-base

Vision-language models get a short caption; plain encoders (DINOv2, SigLIP2) get one image, video encoders
(V-JEPA 2) get the scene's four front frames. Prints load time, output shape and peak VRAM.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageTextToText, AutoProcessor

FRAMES = Path(os.environ["DATA_DIR"]) / "third_party/qwen-drive/data/demo/frames"
front = [Image.open(FRAMES / f"scene1_{i}.jpg").convert("RGB") for i in range(4)]  # 4 front frames at 2 Hz


def caption(repo):
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(repo)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene in one sentence."}]}]
    inputs = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True), images=[front[-1]],
                  return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    return model, repr(proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def encode(repo, video, frames):
    """Plain encoders: report how the input is built, what comes out, and what one forward costs."""
    model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
    if video:  # V-JEPA 2 and friends take a clip [B, T, C, H, W]
        from transformers import AutoVideoProcessor

        proc = AutoVideoProcessor.from_pretrained(repo)
        clip = [np.asarray(f) for f in front]
        inputs = proc(videos=[clip], return_tensors="pt").to("cuda", torch.bfloat16)
    else:
        proc = AutoImageProcessor.from_pretrained(repo)
        inputs = proc(images=front[-1], return_tensors="pt").to("cuda", torch.bfloat16)
    shapes = {k: tuple(v.shape) for k, v in inputs.items() if torch.is_tensor(v)}
    norm = {k: getattr(proc, k, None) for k in ("image_mean", "image_std", "size", "crop_size", "do_normalize")}

    def forward():
        with torch.no_grad():
            if hasattr(model, "get_image_features") and not video:  # dual encoders (SigLIP2)
                return model.vision_model(**inputs), model.get_image_features(**inputs)
            return (model(**inputs),)

    for _ in range(3):
        forward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = forward()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1e3
    heads = []
    for o in out:
        if torch.is_tensor(o):
            heads.append(f"pooled {tuple(o.shape)}")
        else:
            for name in ("last_hidden_state", "pooler_output", "image_embeds"):
                t = getattr(o, name, None)
                if t is not None:
                    heads.append(f"{name} {tuple(t.shape)}")
    return model, (f"inputs {shapes}, {ms:.1f} ms per forward ({ms / frames:.1f} ms/frame), "
                   f"{', '.join(heads)}, preprocessing {norm}")


for repo in sys.argv[1:]:
    t0 = time.time()
    config = AutoConfig.from_pretrained(repo)
    kinds = {config.model_type, *(getattr(config, "architectures", None) or [])}
    if any("ImageTextToText" in k or "ConditionalGeneration" in k for k in kinds):
        model, what = caption(repo)
    else:
        video = "vjepa" in config.model_type or hasattr(config, "frames_per_clip")
        model, what = encode(repo, video, frames=len(front) if video else 1)
    print(f"{repo} ({config.model_type}): ok in {time.time() - t0:.0f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB: {what}", flush=True)
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
