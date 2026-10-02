"""Qwen3-VL (4B or 8B) reader of the vlm_cmp study: GPU preprocessing, one prefill per frame set, many question suffixes.

  python vlm_cmp_model.py prompts      print every prompt verbatim (pasted into the pre-registration)

Why a shared prefix: every prompt puts its text AFTER the images, so the key / value cache of "images + <|vision_end|>" is
the same for every question about the same frames and the hidden states of the image tokens do not depend on the question
(causal attention). One prefill therefore serves the separate questions, the directive prompt and the pooled image features;
each question costs only its own suffix (about 150-250 tokens). The suffix is scored like the earlier path (`vlm_thin_common`,
`fwd_<res>`): the assistant prefix `ANSWER:` is forced and the log-probability of the first token of each option is read at
that position; options that share a first token (pass_left / pass_right) add the log-probability of their second token.
"""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_arb_common import DATA  # noqa: E402
from vlm_cmp_frames import DIRECTIVES  # noqa: E402

MODELS = {"4b": "Qwen3-VL-4B-Instruct", "8b": "Qwen3-VL-8B-Instruct"}
RES = {"r4573": 16777216, "r2335": 1200000, "r1153": 600000, "r559": 300000}   # name (visual tokens, two images, 4B) -> max_pixels
SCORE_PREFIX = "ANSWER:"

# ---------------------------------------------------------------------------------------------------- prompts (pre-registered)
CAMS2 = ("The two images were taken at the same instant by the front cameras of a car (the ego vehicle): "
         "image 1 is the wide-angle camera, image 2 is the narrow road camera.")
CAMS4 = ("Images 1 and 2 were taken about 1 second ago by the front cameras of a car (the ego vehicle): image 1 is the "
         "wide-angle camera, image 2 is the narrow road camera. Images 3 and 4 were taken now by the same two cameras, in "
         "the same order. Compare the two moments: a traffic light that changed, a vehicle ahead that moved or stayed still.")


def _end(opts):
    return ("End your reply with one line of the form `ANSWER: <option>`, <option> being one of: %s. "
            "Reply with that line only." % ", ".join(opts))


LIGHT_OPTS = ["no_light", "red_or_yellow_for_ego", "green_for_ego", "light_for_other_lane"]
Q_LIGHT = ("Traffic light status controlling the ego vehicle lane ahead. Options:\n"
           "  no_light: no traffic light controlling ego lane ahead\n"
           "  red_or_yellow_for_ego: red or yellow traffic light controlling ego lane ahead\n"
           "  green_for_ego: green traffic light controlling ego lane ahead\n"
           "  light_for_other_lane: traffic light visible but controlling another lane, not ego lane")
SIGN_OPTS = ["yes", "no"]
Q_SIGN = "Is there a stop sign controlling the ego vehicle ahead at the intersection?"
BLOCK_OPTS = ["clear", "moving_lead", "static_block"]
Q_BLOCK = ("Obstacle status in the ego vehicle travel lane ahead. Options:\n"
           "  clear: clear path ahead in ego lane\n"
           "  moving_lead: moving vehicle or dynamic object moving ahead in ego lane\n"
           "  static_block: stationary obstacle, construction cones, or stopped broken vehicle blocking ego lane")
SIDE_OPTS = ["left_free", "right_free", "none_free"]
Q_SIDE = ("If there is a blockage ahead, which adjacent lane is clear and safe to bypass? Options:\n"
          "  left_free: left adjacent lane is clear and available for bypass\n"
          "  right_free: right adjacent lane is clear and available for bypass\n"
          "  none_free: neither adjacent lane is clear, or no adjacent lane exists")
CONVENTIONS = ("You are the slow decision channel of a driving stack. Driving conventions:\n"
               "- Approach a junction slowly.\n"
               "- The traffic light that faces you across the junction governs you: red or yellow means stop before the "
               "line, green means go. Ignore lights that face other directions.\n"
               "- A stop sign means a full stop at the line, then go.\n"
               "- A stationary object blocking your lane is passed on a free side; if neither side is free, wait.")
DIRECTIVE_LIST = ("Choose exactly one directive:\n"
                  "  proceed: keep driving normally\n"
                  "  slow: reduce speed and approach carefully\n"
                  "  stop_at_line: stop before the line ahead (light or sign) and stay there\n"
                  "  go_now: start moving or drive through now, the way is open\n"
                  "  pass_left: pass the blocking object on the left\n"
                  "  pass_right: pass the blocking object on the right\n"
                  "  wait: stay stopped, you are blocked and cannot pass now")
HOLD_RULE = "If your previous directive was stop_at_line for a light, keep it until the light facing you is clearly green."


def speed_line(v, now=False):
    return "Ego speed%s: %.1f m/s%s." % (" now" if now else "", v, " (stopped)" if v < 0.5 else "")


def prompt_text(name, v=0.0, prev=None):
    """Prompt body of a question. `name`: light, sign, block, side, dir; the suffixes `_2f` mean the two-moment variant
    (CAMS4 instead of CAMS2; `dir_2f` also states the speed now, the previous directive and HOLD_RULE)."""
    two = name.endswith("_2f")
    base = name[:-3] if two else name
    cams = CAMS4 if two else CAMS2
    if base == "light":
        return "%s\n%s\n%s" % (cams, Q_LIGHT, _end(LIGHT_OPTS))
    if base == "sign":
        return "%s\n%s\n%s" % (cams, Q_SIGN, _end(SIGN_OPTS))
    if base == "block":
        return "%s\n%s\n%s" % (cams, Q_BLOCK, _end(BLOCK_OPTS))
    if base == "side":
        return "%s\n%s\n%s" % (cams, Q_SIDE, _end(SIDE_OPTS))
    if base == "dir":
        mid = [speed_line(v, now=two)]
        if two:
            mid += ["Your previous directive, about 1 second ago, was: %s." % prev, HOLD_RULE]
        return "%s\n%s\n%s\n%s\n%s" % (cams, CONVENTIONS, "\n".join(mid), DIRECTIVE_LIST, _end(DIRECTIVES))
    raise ValueError(name)


OPTIONS = dict(light=LIGHT_OPTS, sign=SIGN_OPTS, block=BLOCK_OPTS, side=SIDE_OPTS, dir=DIRECTIVES)


def model_path(key):
    g = glob.glob(str(DATA / "cache/huggingface/hub") + "/models--Qwen--%s/snapshots/*" % MODELS[key])
    if not g:
        raise FileNotFoundError("model %s not on disk" % MODELS[key])
    return g[0]


# ---------------------------------------------------------------------------------------------------- the model
class VL:
    """prep(jpgs, max_px) -> frame set x; prefill(x, pooled) -> state; ask(state, name, v, prev) -> (log-probs of the options,
    answer-position hidden states per kept layer)."""

    def __init__(self, key, dev="cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor
        self.torch, self.dev, self.key = torch, dev, key
        torch.set_grad_enabled(False)
        p = model_path(key)
        self.proc = AutoProcessor.from_pretrained(p)
        self.ip = AutoImageProcessor.from_pretrained(p, backend="torchvision")
        self.m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16).to(dev).eval()
        self.lm = self.m.model.language_model
        self.layers, self.norm = list(self.lm.layers), self.lm.norm
        self.NL = len(self.layers)
        self.d = self.m.config.text_config.hidden_size
        self.img_id = self.m.config.image_token_id
        self.Wf = self.m.lm_head.weight.float()
        self.tok = self.proc.tokenizer
        self.keep = list(range(2, self.NL + 1, 2))                   # layers whose answer-position state is kept (raw residual; NL = normed)
        self.pool_layers = [0] + self.keep
        self._pieces, self._tails, self._opt = {}, {}, {}
        pre = self.tok(SCORE_PREFIX, add_special_tokens=False).input_ids
        for name, opts in OPTIONS.items():
            seqs = []
            for o in opts:
                full = self.tok(SCORE_PREFIX + " " + o, add_special_tokens=False).input_ids
                assert full[:len(pre)] == pre
                seqs.append(full[len(pre):])
            first = [s[0] for s in seqs]
            groups = {}
            for i, f in enumerate(first):
                groups.setdefault(f, []).append(i)
            for f, ix in groups.items():                             # options that share a first token must differ at the second
                if len(ix) > 1:
                    assert all(len(seqs[i]) > 1 for i in ix) and len({seqs[i][1] for i in ix}) == len(ix), (name, opts, seqs)
            self._opt[name] = (seqs, groups)

    # ---- text pieces
    def _template(self, k, text, prefix=""):
        msgs = [{"role": "user", "content": [{"type": "image"}] * k + [{"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix

    def split_prompt(self, k, text):
        """(head token pieces around the k image slots, tail token ids): the head ends with <|vision_end|> of the last image,
        the tail is the question text up to and including the forced `ANSWER:`."""
        if (k, text) not in self._tails:
            s = self._template(k, text, SCORE_PREFIX)
            cut = s.rindex("<|vision_end|>") + len("<|vision_end|>")
            head, tail = s[:cut], s[cut:]
            t = self.torch
            pieces = [t.tensor(self.tok(x, add_special_tokens=False).input_ids, device=self.dev) for x in head.split("<|image_pad|>")]
            assert len(pieces) == k + 1
            if len(self._tails) > 4000:
                self._tails.clear()
            self._tails[(k, text)] = (pieces, t.tensor(self.tok(tail, add_special_tokens=False).input_ids, device=self.dev))
        return self._tails[(k, text)]

    def head_pieces(self, k):
        if k not in self._pieces:
            self._pieces[k] = self.split_prompt(k, prompt_text("light" if k == 2 else "light_2f"))[0]
        return self._pieces[k]

    # ---- images
    def prep(self, jpgs, max_px):
        """k JPEG byte strings -> pixel values, grid, tokens per image, input ids of the prefix (images included)."""
        t = self.torch
        from torchvision.io import ImageReadMode, decode_jpeg
        imgs = decode_jpeg([t.frombuffer(bytearray(b), dtype=t.uint8) for b in jpgs], device=self.dev, mode=ImageReadMode.RGB)
        o = self.ip(images=imgs, device=self.dev, size={"shortest_edge": 65536, "longest_edge": max_px}, return_tensors="pt")
        grid = o["image_grid_thw"].to(self.dev)
        n = (grid.prod(-1) // 4).tolist()
        pc = self.head_pieces(len(jpgs))
        parts = [pc[0]]
        for i, k in enumerate(n):
            parts += [t.full((k,), self.img_id, device=self.dev), pc[i + 1]]
        ids = t.cat(parts)
        return dict(ids=ids, pixel_values=o["pixel_values"].to(self.dev, t.bfloat16), grid=grid, n_img=n,
                    k=len(jpgs), mm=(ids == self.img_id).to(t.int64))

    # ---- forward
    def prefill(self, x, pooled=False):
        """Run images + prefix once; keeps the cache for the suffixes. pooled: also the per-image mean of the image tokens at
        every kept layer (and layer 0), [len(pool_layers), k, d] float32 on the GPU."""
        t = self.torch
        out = self.m.model(input_ids=x["ids"][None], attention_mask=t.ones_like(x["ids"])[None], pixel_values=x["pixel_values"],
                           image_grid_thw=x["grid"], mm_token_type_ids=x["mm"][None], use_cache=True, output_hidden_states=pooled)
        st = dict(cache=out.past_key_values, P=int(x["ids"].numel()), rope_deltas=self.m.model.rope_deltas.clone())
        if pooled:
            starts, pos = [], len(self.head_pieces(x["k"])[0])
            for i, n in enumerate(x["n_img"]):
                starts.append(pos)
                pos += n + len(self.head_pieces(x["k"])[i + 1])
            st["pool"] = t.stack([t.stack([out.hidden_states[L][0, s:s + n].float().mean(0) for s, n in zip(starts, x["n_img"])])
                                  for L in self.pool_layers])
        return st

    def _suffix(self, st, ids, hidden=False):
        """Feed `ids` after the cached prefix; returns (logits at the last position [V] float32, hidden states [len(keep), d]
        at the last position or None). The cache is cropped back by the caller."""
        t = self.torch
        P0 = st["cache"].get_seq_length()
        S = int(ids.numel())
        pos = (t.arange(P0, P0 + S, device=self.dev)[None, None, :] + st["rope_deltas"].view(1, 1, 1)).expand(3, 1, S)
        out = self.m.model(input_ids=ids[None], attention_mask=t.ones(1, P0 + S, dtype=t.int64, device=self.dev),
                           past_key_values=st["cache"], position_ids=pos, cache_position=t.arange(P0, P0 + S, device=self.dev),
                           use_cache=True, output_hidden_states=hidden)
        last = out.last_hidden_state[0, -1]
        logits = self.Wf @ last.float()                              # float32 head: bf16 logits are quantised to 0.12-0.25 here
        hs = None
        if hidden:
            hs = t.stack([out.hidden_states[L][0, -1].float() for L in self.keep])
        return logits, hs

    def ask(self, st, k, text, name, hidden=False):
        """Option log-probabilities (numpy [n_options]) of question `name` with prompt body `text` about the frames of `st`
        (k images); hidden: also the answer-position hidden states per kept layer ([len(keep), d] numpy float16)."""
        t = self.torch
        pieces, tail = self.split_prompt(k, text)
        P = st["P"]
        logits, hs = self._suffix(st, tail, hidden)
        lp = t.log_softmax(logits, -1)
        seqs, groups = self._opt[name]
        out = np.zeros(len(seqs), np.float32)
        for f, ix in groups.items():
            for i in ix:
                out[i] = float(lp[f])
        for f, ix in groups.items():
            if len(ix) > 1:
                st["cache"].crop(P + int(tail.numel()))
                l2, _ = self._suffix(st, t.tensor([f], device=self.dev))
                lp2 = t.log_softmax(l2, -1)
                for i in ix:
                    out[i] += float(lp2[seqs[i][1]])
        st["cache"].crop(P)
        return out, (None if hs is None else hs.half().cpu().numpy())

    # ---- full reference forward without cache (equivalence checks, cut latency)
    def full_logits(self, x, text):
        """Whole sequence in one forward, no cache: (option logits of the first token over the vocabulary [V], hidden states
        at the answer position per kept layer)."""
        t = self.torch
        pieces, tail = self.split_prompt(x["k"], text)
        ids = t.cat([x["ids"], tail])
        mm = (ids == self.img_id).to(t.int64)
        self.set_cut(self.NL)
        out = self.m.model(input_ids=ids[None], attention_mask=t.ones_like(ids)[None], pixel_values=x["pixel_values"],
                           image_grid_thw=x["grid"], mm_token_type_ids=mm[None], use_cache=False, output_hidden_states=True)
        logits = self.Wf @ out.last_hidden_state[0, -1].float()
        return logits, t.stack([out.hidden_states[L][0, -1].float() for L in self.keep])

    def set_cut(self, N):
        self.lm.layers = self.torch.nn.ModuleList(self.layers[:N])
        self.lm.norm = self.norm if N == self.NL else self.torch.nn.Identity()
        self.cut = N

    def cut_state(self, x, text, N):
        """Latency path of a truncated model: whole sequence, only the first N layers, hidden state at the answer position
        (float32 [d]) after layer N."""
        t = self.torch
        pieces, tail = self.split_prompt(x["k"], text)
        ids = t.cat([x["ids"], tail])
        mm = (ids == self.img_id).to(t.int64)
        if getattr(self, "cut", None) != N:
            self.set_cut(N)
        out = self.m.model(input_ids=ids[None], attention_mask=t.ones_like(ids)[None], pixel_values=x["pixel_values"],
                           image_grid_thw=x["grid"], mm_token_type_ids=mm[None], use_cache=False)
        return out.last_hidden_state[0, -1].float()


def sync(torch):
    torch.cuda.synchronize()


if __name__ == "__main__":
    if sys.argv[1] == "prompts":
        for n in ("light", "sign", "block", "side", "dir"):
            print("=== %s ===\n%s\n" % (n, prompt_text(n, v=0.0)))
        print("=== dir_2f (previous directive stop_at_line) ===\n%s\n" % prompt_text("dir_2f", v=0.0, prev="stop_at_line"))
        for n in ("light_2f", "block_2f"):
            print("=== %s ===\n%s\n" % (n, prompt_text(n)))
