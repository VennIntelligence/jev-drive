"""Load check for our feature backbones in the project venv (no benchmark): one forward on one camera frame.

  cd ~/data/jev-drive && uv run python scripts/bench_baselines/load_backbones.py Qwen/Qwen3-VL-8B-Instruct ...
"""
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

img = Image.open(Path(os.environ["DATA_DIR"]) / "third_party/qwen-drive/data/demo/frames/scene1_3.jpg")
for repo in sys.argv[1:]:
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(repo)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene in one sentence."}]}]
    inputs = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True), images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    text = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"{repo}: loaded+generated in {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB: {text!r}")
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
