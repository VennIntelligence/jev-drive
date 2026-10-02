"""Shared pieces of the vlm_thin study: frames, splits, labels, the Qwen3-VL wrapper (GPU preprocessing, truncated forward,
option scoring), light readouts. Plan: experiments/vlm_arb/plans/2026-10-02-vlm-thin.md (pre-registration).

Question: how much of Qwen3-VL-4B's ego-light reading survives when only the vision tower and the first N language layers
run and a thin head reads the hidden state at the answer position, and what latency does each cut give?
"""
import glob
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_arb_common import DATA, REPO, RUN, boot_ratio  # noqa: E402

sys.path.insert(0, str(REPO))
OUT_ROOT = DATA / "runs/vlm_thin"
CACHE = OUT_ROOT / "cache"
UNITS = ["v2-dbg-shadow-s0-light", "v2-drive-s0-tgt", "v2-drive-s1-dev", "v2-drive-s1-tgt"]
SWEEP_FRAMES = RUN / "results/lightsweep/frames.csv"       # the 233-frame junction-approach subset of the earlier sweep

# Registered in the plan before any result was read ------------------------------------------------------------------
SPLIT = dict(
    test="27043 15483 16529 9196 28147 17280 2520 19324".split(),
    val="15612 27297 334 19832".split(),
    train="15102 16390 16508 24944 27787 27870 22535 24497 37969".split())
RES = {"r4573": 16777216, "r2335": 1200000, "r1153": 600000, "r559": 300000}   # name (visual tokens, 2 images) -> max_pixels
CUTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36]
NL = 36                                                                         # language layers of Qwen3-VL-4B
LIGHTS = ["no_light", "red_or_yellow_for_ego", "green_for_ego", "light_for_other_lane"]
LIGHT3 = ["red", "green", "none"]                                               # head classes
ANS3 = {"red": "red_or_yellow_for_ego", "green": "green_for_ego", "none": "no_light"}
RED = "red_or_yellow_for_ego"
CAMS2 = ("The two images were taken at the same instant by the front cameras of a car (the ego vehicle): "
         "image 1 is the wide-angle camera, image 2 is the narrow road camera.")
Q_FOUR = ("Traffic light status controlling the ego vehicle lane ahead. Options:\n"
          "  no_light: no traffic light controlling ego lane ahead\n"
          "  red_or_yellow_for_ego: red or yellow traffic light controlling ego lane ahead\n"
          "  green_for_ego: green traffic light controlling ego lane ahead\n"
          "  light_for_other_lane: traffic light visible but controlling another lane, not ego lane")
FOUR_PROMPT = ("%s\n%s\nEnd your reply with one line of the form `ANSWER: <option>`, <option> being one of: %s. "
               "Reply with that line only." % (CAMS2, Q_FOUR, ", ".join(LIGHTS)))      # == vlm_arb_lightsweep.prompt_text("four")
SCORE_PREFIX = "ANSWER:"                                                        # forced assistant prefix of the one-pass scoring


def qwen_path():
    return glob.glob(str(DATA / "cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*"))[0]


# ---------------------------------------------------------------------------------------------------- frames and labels
def load_frames(units=UNITS):
    """One row per answered request of the shadow units with both frames on disk, with the Phase A truth columns."""
    rows = []
    for u in units:
        udir = RUN / "arms" / u
        for done in sorted(udir.glob("done/*.json")):
            a = udir / "attempts" / done.stem / str(json.loads(done.read_text()).get("attempt", 1))
            if not (a / "vlm_decisions.jsonl").exists():
                continue
            for line in open(a / "vlm_decisions.jsonl"):
                d = json.loads(line)
                if d.get("k") != "a" or not d["ans"].get("ok"):
                    continue
                f = {c: a / "vlm_frames" / ("%08.2f_%s.jpg" % (d["t_q"], c)) for c in ("wide", "road")}
                if not all(p.exists() for p in f.values()):
                    continue
                g, tl = d["gt"], d["gt"].get("tl")
                others = [x for x in g.get("lights", []) if x[0] != g.get("tl_id")]
                rows.append(dict(
                    id="%s/%s/%.2f" % (u, done.stem, d["t_q"]), src="new", unit=u, route=done.stem, t=d["t_q"],
                    tl=-1 if tl is None else tl, tl_dist=g["tl_dist"] if tl is not None else np.nan,
                    other_red=float(any(x[1] == 2 for x in others)), any_light=float(bool(g.get("lights"))),
                    other_green=float(any(x[1] == 0 for x in others)),
                    stop_dist=np.nan if g.get("stop_dist") is None else g["stop_dist"], has_sign_label=True,
                    block=g["block"], side=g["side"], wide=str(f["wide"]), road=str(f["road"]),
                    oj_light=d["ans"].get("Q_light", ""), oj_lat=d["ans"].get("latency_ms", np.nan)))
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    sweep = set(pd.read_csv(SWEEP_FRAMES).id) if SWEEP_FRAMES.exists() else set()
    df["sweep"] = df.id.isin(sweep)
    df["part"] = df.route.map({r: p for p, rs in SPLIT.items() for r in rs}).fillna("none")
    df["y"] = light_label(df)
    return df


def light_label(df):
    """Head target: 0 red (ego red / yellow, -5 <= dist < 50 m), 1 green (same range), 2 none (no ego light), -1 = ignored
    (an ego light outside that range: the registered readouts do not score those frames)."""
    near = (df.tl_dist >= -5) & (df.tl_dist < 50)
    y = np.full(len(df), -1)
    y[(df.tl == -1).to_numpy()] = 2
    y[(df.tl.isin([1, 2]) & near).to_numpy()] = 0
    y[((df.tl == 0) & near).to_numpy()] = 1
    return y


def define_splits():
    """Register the route-level splits (jevdrive.data.splits); idempotent."""
    from jevdrive.data import splits
    out = {}
    for part, routes in SPLIT.items():
        out[part] = splits.define(
            "b2d", "vlm-thin-" + part, routes, unit="route", status="frozen", used_by="experiments/vlm_arb vlm_thin",
            origin="hand-assigned before any model answer on these frames was read, to cover every Phase A readout "
                   "in test (plan 2026-10-02-vlm-thin.md section 2); routes of the vlm_arb shadow units with saved frames")
    splits.check_disjoint(*out.values())
    return out


# ---------------------------------------------------------------------------------------------------- the model
class Thin:
    """Qwen3-VL-4B-Instruct with a GPU preprocessing path and a truncated forward.

    prep(jpgs, max_px)   two JPEG byte strings -> model inputs (nvjpeg decode, GPU resize / patchify, ids built from the
                         fixed prompt); the same path serves extraction, the bench and the Phase A serve run
    run(x, N, ...)       vision tower + the first N language layers (N = NL: the whole model); returns the hidden state at
                         the answer position (the last token, `SCORE_PREFIX` forced) and, if asked, the mean of the image
                         tokens per image at layer N
    """

    def __init__(self, dev="cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor
        self.torch, self.dev = torch, dev
        torch.set_grad_enabled(False)                                           # inference only, everywhere
        p = qwen_path()
        self.proc = AutoProcessor.from_pretrained(p)
        self.ip = AutoImageProcessor.from_pretrained(p, backend="torchvision")
        self.m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16).to(dev).eval()
        self.lm = self.m.model.language_model
        self.layers, self.norm = list(self.lm.layers), self.lm.norm
        self.img_id = self.m.config.image_token_id
        tok = self.proc.tokenizer
        self.tok = tok
        assert len(self.layers) == NL
        self.set_prompt(FOUR_PROMPT)
        pre = tok(SCORE_PREFIX, add_special_tokens=False).input_ids
        first = []
        for o in LIGHTS:
            full = tok(SCORE_PREFIX + " " + o, add_special_tokens=False).input_ids
            assert full[:len(pre)] == pre
            first.append(full[len(pre)])
        assert len(set(first)) == 4, first
        self.opt_ids = first
        self.W = self.m.lm_head.weight[first].float()                            # [4, 2560]

    def chat_text(self, prompt, prefix=""):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix

    def set_prompt(self, prompt):
        """Pre-tokenise the three text pieces around the two image placeholders: `pieces` end with the forced assistant
        prefix `SCORE_PREFIX` (one-pass scoring), `pieces_gen` end at the generation prompt (free generation)."""
        t = self.torch
        mk = lambda prefix: [t.tensor(self.tok(x, add_special_tokens=False).input_ids, device=self.dev)   # noqa: E731
                             for x in self.chat_text(prompt, prefix).split("<|image_pad|>")]
        self.pieces, self.pieces_gen = mk(SCORE_PREFIX), mk("")
        assert len(self.pieces) == 3

    def prep(self, jpgs, max_px, gen=False):
        t = self.torch
        from torchvision.io import ImageReadMode, decode_jpeg
        imgs = decode_jpeg([t.frombuffer(bytearray(b), dtype=t.uint8) for b in jpgs], device=self.dev, mode=ImageReadMode.RGB)
        o = self.ip(images=imgs, device=self.dev, size={"shortest_edge": 65536, "longest_edge": max_px}, return_tensors="pt")
        grid = o["image_grid_thw"].to(self.dev)
        n = (grid.prod(-1) // 4).tolist()
        pad = lambda k: t.full((k,), self.img_id, device=self.dev)   # noqa: E731
        pc = self.pieces_gen if gen else self.pieces
        ids = t.cat([pc[0], pad(n[0]), pc[1], pad(n[1]), pc[2]])
        mm = (ids == self.img_id).to(t.int64)
        return dict(input_ids=ids[None], mm_token_type_ids=mm[None], pixel_values=o["pixel_values"].to(self.dev, t.bfloat16),
                    image_grid_thw=grid, attention_mask=t.ones_like(ids)[None], n_img=n)

    def prep_ref(self, paths, prompt=None, prefix="", max_px=None):
        """The reference path: PIL decode, the HF processor on CPU (what the earlier sweep ran)."""
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in paths]
        msgs = [{"role": "user", "content": [{"type": "image", "image": i} for i in imgs]
                 + [{"type": "text", "text": prompt or FOUR_PROMPT}]}]
        x = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        return x.to(self.dev)

    def set_cut(self, N):
        """Run only the first N language layers; for N < NL the final norm is skipped so the output is the raw layer-N
        residual stream (== hidden_states[N] of the full model)."""
        self.lm.layers = self.torch.nn.ModuleList(self.layers[:N])
        self.lm.norm = self.norm if N == NL else self.torch.nn.Identity()
        self.cut = N

    def vision_only(self, x):
        """The vision tower alone: merged tokens per image [(n_i, 2560)] (the layer-0 image tokens of the language model)."""
        emb = self.m.model.get_image_features(x["pixel_values"], x["image_grid_thw"])
        emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb
        return [e for e in (emb if isinstance(emb, (list, tuple)) else emb.split(x["n_img"]))]

    def run(self, x, N=NL, pool=False):
        """Hidden state at the answer position after N layers (float32 [2560]); pool=True also returns the mean of each
        image's tokens at layer N ([2, 2560])."""
        t = self.torch
        if getattr(self, "cut", None) != N:
            self.set_cut(N)
        out = self.m.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=x["pixel_values"],
                           image_grid_thw=x["image_grid_thw"], mm_token_type_ids=x["mm_token_type_ids"], use_cache=False)
        h = out.last_hidden_state[0]
        last = h[-1].float()
        if not pool:
            return last
        n0, n1 = x["n_img"]
        s0 = len(self.pieces[0])
        s1 = s0 + n0 + len(self.pieces[1])
        return last, t.stack([h[s0:s0 + n0].float().mean(0), h[s1:s1 + n1].float().mean(0)])

    def option_logits(self, last_normed):
        return self.W @ last_normed.float()

    def restore(self):
        self.set_cut(NL)


def sync(torch):
    torch.cuda.synchronize()


def parse_option(text, options):
    """Last `ANSWER: <option>` of a reply (a bare option also counts); "" when none of `options` was named."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.findall(r"ANSWER\s*[:=]\s*[\"'`<*\s]*([A-Za-z_]+)", text, flags=re.I)
    w = (m[-1] if m else text.strip("\"'`.*<> \n")).lower()
    return w if w in options else ""


# ---------------------------------------------------------------------------------------------------- readouts
def light_masks(df):
    red = df.tl.isin([1, 2]) & (df.tl_dist >= -5) & (df.tl_dist < 50)
    green = (df.tl == 0) & (df.tl_dist >= -5) & (df.tl_dist < 50)
    nolight = (df.tl == -1) & (df.any_light == 0)
    other = (df.other_red == 1) & ~(df.tl.isin([1, 2]) & (df.tl_dist < 50))
    return dict(red=red, green=green, nolight=nolight, other=other)


def light_metrics(df, ans):
    """Phase A light readouts of answers `ans` (strings, one per row of df): estimate [95% route-cluster CI] each, plus
    S = red recall - red answered green + green recall - no-light false alarm (the sweep's score)."""
    df, ans = df.reset_index(drop=True), pd.Series(np.asarray(ans), index=range(len(df)))
    m = light_masks(df)
    R = {}
    for k, (mk, hit) in dict(red_recall=("red", ans == RED), red_as_green=("red", ans == "green_for_ego"),
                             green_recall=("green", ans == "green_for_ego"), nolight_fp=("nolight", ans == RED),
                             other_fp=("other", ans == RED)).items():
        R[k] = boot_ratio((hit & m[mk]).astype(float), m[mk].astype(float), df.route)
    R["S"] = float(R["red_recall"]["est"] - R["red_as_green"]["est"] + R["green_recall"]["est"] - R["nolight_fp"]["est"]) \
        if all(np.isfinite(R[k]["est"]) for k in ("red_recall", "red_as_green", "green_recall", "nolight_fp")) else float("nan")
    return R


def cellp(r):
    return "n/a" if not r["n"] else "%.0f%% [%.0f, %.0f] n=%d" % (100 * r["est"], 100 * r["lo"], 100 * r["hi"], r["n"])


def percentiles(ms):
    ms = np.asarray(ms, float)
    return dict(n=int(len(ms)), p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)),
                p99=float(np.percentile(ms, 99)), mean=float(ms.mean()))


def jwrite(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)) + "\n")
    tmp.replace(path)


def read_jpgs(row):
    return [Path(row.wide).read_bytes(), Path(row.road).read_bytes()]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
