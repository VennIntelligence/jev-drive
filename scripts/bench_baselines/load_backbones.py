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


def encode(repo, video):
    model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
    if video:  # V-JEPA 2 and friends take a clip [B, T, C, H, W]
        from transformers import AutoVideoProcessor

        proc = AutoVideoProcessor.from_pretrained(repo)
        inputs = proc(videos=[[torch.from_numpy(__import__("numpy").asarray(f)).permute(2, 0, 1) for f in front]],
                      return_tensors="pt").to("cuda", torch.bfloat16)
    else:
        proc = AutoImageProcessor.from_pretrained(repo)
        inputs = proc(images=front[-1], return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model(**inputs)
    state = getattr(out, "last_hidden_state", None)
    if state is None:  # dual encoders return one embedding per tower
        state = getattr(out, "image_embeds", None)
    return model, f"features {tuple(state.shape)}" if state is not None else f"output {type(out).__name__}"


for repo in sys.argv[1:]:
    t0 = time.time()
    config = AutoConfig.from_pretrained(repo)
    kinds = {config.model_type, *(getattr(config, "architectures", None) or [])}
    if any("ImageTextToText" in k or "ConditionalGeneration" in k for k in kinds):
        model, what = caption(repo)
    else:
        model, what = encode(repo, video="vjepa" in config.model_type or hasattr(config, "frames_per_clip"))
    print(f"{repo} ({config.model_type}): ok in {time.time() - t0:.0f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB: {what}", flush=True)
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
