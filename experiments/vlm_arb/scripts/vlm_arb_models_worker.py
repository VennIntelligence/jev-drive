"""One model of the model comparison (vlm_arb_models.py), run in the env that can load it (Python 3.8-compatible).

  <env python> vlm_arb_models_worker.py --kind KIND --path PATH --sample sample.jsonl --out models/<name>.jsonl

Every request is the two saved frames (wide, road) plus the one fixed multiple-choice prompt PROMPT; the raw reply and
its wall-clock latency are appended to --out (one line per request, flushed), so a rerun skips what is there. Requests
that raised are retried on the next run. <out stem>.meta.json holds load time, peak VRAM and the load error, if any.
Parsing (parse_reply) is applied by the caller, not here.
"""
import argparse
import base64
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from vlm_protocol import QUESTIONS_SCHEMA  # noqa: E402

MAX_NEW_TOKENS = 64            # the reply is one short JSON object (~40 tokens); also the dgemma canvas length
LETTERS = "ABCD"
OPTIONS = {q: list(s["criteria"]) if s["type"] == "choice" else ["yes", "no"] for q, s in QUESTIONS_SCHEMA.items()}


def build_prompt():
    L = ["You are given two images taken at the same instant by the front cameras of a car (the ego vehicle): image 1 "
         "is the wide-angle camera, image 2 is the narrow road camera. Answer the four multiple-choice questions."]
    for q, s in QUESTIONS_SCHEMA.items():
        L.append("%s: %s" % (q, s["instructions"]))
        crit = s.get("criteria") or {"yes": "yes", "no": "no"}
        L += ["  %s. %s%s" % (LETTERS[i], k, "" if k == v else ": " + v) for i, (k, v) in enumerate(crit.items())]
    L.append("Reply with one JSON object and nothing else, each value being the chosen option name: "
             + "{" + ", ".join('"%s": "<option>"' % q for q in QUESTIONS_SCHEMA) + "}")
    return "\n".join(L)


PROMPT = build_prompt()


def _option(q, v):
    v = str(v).strip().strip("\"'.").strip()
    if v.lower() in OPTIONS[q]:
        return v.lower()
    head = re.match(r"^([A-Da-d])(?:[.):\s]|$)", v)            # "B" or "B. red_or_yellow_for_ego"
    if head and LETTERS.index(head.group(1).upper()) < len(OPTIONS[q]):
        return OPTIONS[q][LETTERS.index(head.group(1).upper())]
    return None


def parse_reply(text):
    """{question: option or None}. Strict JSON first, else `Q_x: <option or letter>` pairs; anything else is None."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    obj, i, j = {}, text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(text[i:j + 1])
        except ValueError:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    out = {}
    for q in OPTIONS:
        v = obj.get(q)
        if v is None:
            m = re.search(q + r"\"?\s*[:=]\s*\"?([A-Za-z_]+)", text)
            v = m.group(1) if m else ""
        out[q] = _option(q, v)
    return out


# ---- loaders: each returns ask(wide_path, road_path) -> reply text ----

def load_openai(path):
    """A running OpenAI-compatible server (the lane's vLLM); `path` = "<url>|<served model name>"."""
    import urllib.request
    url, model = path.split("|")

    def ask(wide, road):
        imgs = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                                    base64.b64encode(Path(p).read_bytes()).decode("ascii")}}
                for p in (wide, road)]
        # no temperature: the diffusion model's server rejects sampling parameters
        body = {"model": model, "max_tokens": MAX_NEW_TOKENS,
                "messages": [{"role": "user", "content": imgs + [{"type": "text", "text": PROMPT}]}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60.0) as r:
            msg = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    return ask


def load_hf_chat(path):
    """transformers image-text-to-text chat model (Qwen3-VL, Qwen2.5-VL family), greedy decoding."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def ask(wide, road):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(wide).convert("RGB")},
                                             {"type": "image", "image": Image.open(road).convert("RGB")},
                                             {"type": "text", "text": PROMPT}]}]
        x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt").to("cuda")
        y = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return ask


def load_qwen_drive(path):
    """Qwen-Drive's VQA mode as it ships (scripts/run_vqa.py): the VLM alone, its released decode parameters."""
    import torch
    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    @torch.no_grad()
    def ask(wide, road):
        return model.generate_text([wide, road], PROMPT, max_new_tokens=MAX_NEW_TOKENS).text
    return ask


def load_internvl(path):
    """InternVL2 remote-code chat model: dynamic 448 px tiles (at most 6 per image plus a thumbnail), greedy."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    grids = sorted(((c, r) for c in range(1, 7) for r in range(1, 7) if c * r <= 6), key=lambda g: g[0] * g[1])

    def tiles(p, S=448):
        img = Image.open(p).convert("RGB")
        c, r = min(grids, key=lambda g: abs(img.width / img.height - g[0] / g[1]))
        big = img.resize((S * c, S * r), Image.BICUBIC)
        parts = [big.crop((i * S, j * S, (i + 1) * S, (j + 1) * S)) for j in range(r) for i in range(c)]
        if len(parts) > 1:
            parts.append(img.resize((S, S), Image.BICUBIC))
        import numpy as np
        return torch.stack([(torch.from_numpy(np.asarray(t).copy()).permute(2, 0, 1).float() / 255.0 - mean) / std
                            for t in parts])

    @torch.no_grad()
    def ask(wide, road):
        px = [tiles(wide), tiles(road)]
        return model.chat(tok, torch.cat(px).to("cuda", torch.bfloat16), "Image-1: <image>\nImage-2: <image>\n" + PROMPT,
                          dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False), num_patches_list=[len(t) for t in px])
    return ask


LOADERS = dict(openai=load_openai, hf_chat=load_hf_chat, qwen_drive=load_qwen_drive, internvl=load_internvl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=sorted(LOADERS))
    ap.add_argument("--path", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load", type=int, default=1, help="stamped on every row: 0 = no CARLA server on the box at start")
    a = ap.parse_args()
    out, meta_path = Path(a.out), Path(a.out).with_suffix(".meta.json")
    sample = [json.loads(line) for line in open(a.sample)]
    kept = [json.loads(line) for line in open(out)] if out.exists() else []
    kept = [r for r in kept if not r.get("err")]
    out.write_text("".join(json.dumps(r) + "\n" for r in kept))
    done = {r["id"] for r in kept}
    todo = [s for s in sample if s["id"] not in done]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.pop("error", None)
    print("%s: %d cached, %d to run" % (out.stem, len(done), len(todo)), flush=True)
    if not todo:
        meta_path.write_text(json.dumps(meta))
        return 0
    t0 = time.time()
    try:
        ask = LOADERS[a.kind](a.path)
    except Exception:  # noqa: BLE001 - a model that does not load is a row of the table
        meta["error"] = "load failed: " + traceback.format_exc().strip().splitlines()[-1][:300]
        meta_path.write_text(json.dumps(meta))
        traceback.print_exc()
        return 1
    meta["load_s"] = round(time.time() - t0, 1)
    print("loaded in %.0f s" % meta["load_s"], flush=True)
    n_err, last_err = 0, ""
    with open(out, "a") as f:
        for i, s in enumerate(todo):
            t1 = time.perf_counter()
            row = {"id": s["id"], "load": a.load}
            try:
                row["raw"] = ask(s["wide"], s["road"])
            except Exception as e:  # noqa: BLE001
                row["err"], last_err = ("%s: %s" % (type(e).__name__, e))[:300], traceback.format_exc()
                n_err += 1
            row["lat_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i == 0 or (i + 1) % 100 == 0:
                print("%d / %d  %.0f ms  %s" % (i + 1, len(todo), row["lat_ms"], (row.get("raw") or row.get("err"))[:160]),
                      flush=True)
            if n_err == i + 1 and n_err >= 5:
                meta["error"] = "first 5 requests failed: " + row["err"]
                print(last_err, flush=True)
                break
    if a.kind != "openai":
        import torch
        meta["peak_vram_mb"] = max(meta.get("peak_vram_mb", 0), int(torch.cuda.max_memory_reserved() / 2 ** 20))
    meta_path.write_text(json.dumps(meta))
    return 1 if "error" in meta else 0


if __name__ == "__main__":
    sys.exit(main())
