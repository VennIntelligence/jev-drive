"""small_vlm: zero-shot ego-light reading of small VLMs on the d85 sweep frames (233 frames, 21 routes), same prompt and scoring
as vlm_thin (`FOUR_PROMPT`, `ANSWER:`-forced option scoring, free `generate` with 512 new tokens: the new models reason aloud before the ANSWER line), one generic HF path for every model so the
Qwen3-VL-4B control and the new models share a harness.

  small_vlm_eval.py run  --model q3vl4b|q35_2b|q35_4b|gemma4_e2b [--limit K]   answers + latency (restartable, jsonl append)
  small_vlm_eval.py report                                                      results/small_vlm.md from the jsonl files

Configs: gen_native (HF generate, native resolution, = d85 `gen_ref`), fwd_r4573 / fwd_r1153 / fwd_r559 (one forward, option
scoring; visual-token budgets as in d85, for Gemma 4 the closest soft-token budgets 1120 / 560 / 280 per image).
Latency: `lat` = decode + preprocess on CPU + forward + argmax (the bench path of this harness), `fwd` = forward only.
Output: $DATA_DIR/runs/small_vlm/eval/<model>/{answers.jsonl,meta.json,DONE,ERROR,STATUS}.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_thin_common import FOUR_PROMPT, LIGHTS, RED, SCORE_PREFIX, log, parse_option  # noqa: E402
from vlm_arb_common import DATA, REPO  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402

THIN_RUN = DATA / "runs/vlm_thin/20261002-114036"           # frames.csv with the `sweep` column == the d85 233-frame set
OUT = DATA / "runs/small_vlm/eval"
MODELS = {                                                  # name -> (path or HF id, family)
    "q3vl4b": ("hf:Qwen/Qwen3-VL-4B-Instruct", "qwen"),
    "q35_2b": ("models/Qwen3.5-2B", "qwen"),
    "q35_4b": ("models/Qwen3.5-4B", "qwen"),
    "gemma4_e2b": ("models/gemma-4-E2B-it", "gemma"),
}
CFGS = ["gen_native", "fwd_r4573", "fwd_r1153", "fwd_r559"]
QWEN_PX = {"r4573": 16777216, "r1153": 600000, "r559": 300000}          # max_pixels, as vlm_thin RES
GEMMA_TOK = {"r4573": 1120, "r1153": 560, "r559": 280}                  # soft tokens per image
N_BENCH, N_WARM = 100, 5


def model_path(name):
    p, _ = MODELS[name]
    if p.startswith("hf:"):
        return glob.glob(str(DATA / ("cache/huggingface/hub/models--%s/snapshots/*" % p[3:].replace("/", "--"))))[0]
    return str(DATA / p)


def sweep_frames():
    df = pd.read_csv(THIN_RUN / "frames.csv", dtype={"route": str})
    df = df[df["sweep"]].reset_index(drop=True)
    sw = pd.read_csv(DATA / "runs/vlm_arb/results/lightsweep/frames.csv", usecols=["id", "truth"])   # the sweep's own labels (d85 table)
    df = df.merge(sw, on="id", how="left")
    assert df.truth.notna().all() and len(df) == 233, len(df)
    return df


class Reader:
    def __init__(self, name, dev="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.name, self.dev, self.fam = torch, name, dev, MODELS[name][1]
        torch.set_grad_enabled(False)
        p = model_path(name)
        self.proc = AutoProcessor.from_pretrained(p)
        self.m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16).to(dev).eval()
        self.tok = self.proc.tokenizer
        pre = self.tok(SCORE_PREFIX, add_special_tokens=False).input_ids
        first = []
        for o in LIGHTS:
            full = self.tok(SCORE_PREFIX + " " + o, add_special_tokens=False).input_ids
            assert full[:len(pre)] == pre, (o, full, pre)
            first.append(full[len(pre)])
        assert len(set(first)) == 4, first
        self.opt_ids = first

    def text(self, prefix=""):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": FOUR_PROMPT}]}]
        kw = dict(enable_thinking=False) if self.fam == "qwen" and self.name != "q3vl4b" else {}
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + prefix

    def prep(self, imgs, res, prefix=""):
        if self.fam == "qwen":
            kw = dict(size={"shortest_edge": 65536, "longest_edge": QWEN_PX[res]})
        else:
            kw = dict(max_soft_tokens=GEMMA_TOK[res])
        x = self.proc(text=[self.text(prefix)], images=imgs, return_tensors="pt", **kw)
        return x.to(self.dev)

    def n_vis(self, x):
        t = x["input_ids"]
        ids = [getattr(self.m.config, k, None) for k in ("image_token_id", "image_token_index")]
        return int(sum((t == i).sum() for i in ids if i is not None))

    def score(self, x):
        out = self.m(**x, use_cache=False, logits_to_keep=1)
        return out.logits[0, -1, self.opt_ids].float().cpu().numpy()

    def generate(self, x):
        y = self.m.generate(**x, max_new_tokens=512, do_sample=False)
        return self.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def load_imgs(r):
    from PIL import Image
    return [Image.open(r.wide).convert("RGB"), Image.open(r.road).convert("RGB")]


def run(a):
    import torch
    d = OUT / a.model
    d.mkdir(parents=True, exist_ok=True)
    (d / "ERROR").unlink(missing_ok=True)
    if (d / "DONE").exists() and not a.limit:
        return
    (d / "STATUS").write_text("start %s\n" % time.strftime("%FT%T"))
    try:
        _run(a, d)
    except BaseException as e:                              # noqa: BLE001
        (d / "ERROR").write_text("%s: %s\n" % (type(e).__name__, e))
        raise


def _run(a, d):
    import torch
    df = sweep_frames()
    if a.limit:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, a.limit).astype(int))].reset_index(drop=True)
    rd = Reader(a.model)
    log("loaded %s, %.2f GB" % (a.model, torch.cuda.memory_allocated() / 2 ** 30))
    out = d / ("answers.jsonl" if not a.limit else "answers_limit.jsonl")
    have = {(j["cfg"], j["id"]) for j in map(json.loads, open(out))} if out.exists() else set()
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() and not a.limit else {}
    meta.setdefault("peak_gb", {}), meta.setdefault("n_vis", {})
    with open(out, "a") as f:
        for cfg in CFGS:
            torch.cuda.reset_peak_memory_stats()
            for k, r in enumerate(df.itertuples()):
                if (cfg, r.id) in have:
                    continue
                imgs = load_imgs(r)
                if cfg == "gen_native":
                    x = rd.prep(imgs, "r4573")
                    txt = rd.generate(x)
                    row = dict(raw=txt, ans=parse_option(txt, LIGHTS))
                else:
                    res = cfg[4:]
                    x = rd.prep(imgs, res, SCORE_PREFIX)
                    lg = rd.score(x)
                    row = dict(ans=LIGHTS[int(lg.argmax())], logits=[round(float(v), 3) for v in lg])
                    meta["n_vis"].setdefault(cfg, rd.n_vis(x))
                f.write(json.dumps(dict(cfg=cfg, id=r.id, **row)) + "\n")
                f.flush()
                if k % 40 == 0:
                    (d / "STATUS").write_text("%s %s %d/%d %s\n" % (a.model, cfg, k, len(df), time.strftime("%T")))
            meta["peak_gb"][cfg] = torch.cuda.max_memory_allocated() / 2 ** 30
    # latency bench at r1153 on N_BENCH evenly spaced frames, batch 1, files already decoded from disk each time
    if "lat" not in meta or a.limit:
        torch.cuda.reset_peak_memory_stats()
        b = df.iloc[np.unique(np.linspace(0, len(df) - 1, min(N_BENCH, len(df))).astype(int))]
        lat, fwd = [], []
        for k, r in enumerate(list(b.itertuples())[:N_WARM] + list(b.itertuples())):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = rd.prep(load_imgs(r), "r1153", SCORE_PREFIX)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rd.score(x)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if k >= N_WARM:
                lat.append(1e3 * (t2 - t0)), fwd.append(1e3 * (t2 - t1))
        pc = lambda v: dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)))   # noqa: E731
        meta["lat"], meta["fwd"], meta["bench_peak_gb"] = pc(lat), pc(fwd), torch.cuda.max_memory_allocated() / 2 ** 30
    meta["params_b"] = sum(p.numel() for p in rd.m.parameters()) / 1e9
    if not a.limit:
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        (d / "DONE").write_text(time.strftime("%FT%T\n"))
    (d / "STATUS").write_text("done %s\n" % time.strftime("%T"))
    log("done %s %s" % (a.model, json.dumps(meta)))


# ---------------------------------------------------------------------------------------------------- report
def answers(name):
    p = OUT / name / "answers.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(p)]) if p.exists() else pd.DataFrame()


def cell(r):
    return "%.1f%% [%.1f, %.1f]" % (100 * r["mean"], 100 * r["lo"], 100 * r["hi"])


def metrics(df, ans):
    """Per-cfg metrics: red recall / red-as-green / green recall / no-light FP (route-cluster CI via jevdrive.stats) and red precision."""
    m = dict(red=df.truth == "red", green=df.truth == "green", nolight=df.truth == "none")
    out = {}
    for k, (mk, hit) in dict(red_recall=("red", ans == RED), red_as_green=("red", ans == "green_for_ego"),
                             green_recall=("green", ans == "green_for_ego"), nolight_fp=("nolight", ans == RED)).items():
        sel = m[mk].to_numpy()
        out[k] = dict(stats.bootstrap(hit.to_numpy()[sel].astype(float), groups=df.route.to_numpy()[sel]), hits=int(hit.to_numpy()[sel].sum()))
    tp = int((ans == RED).to_numpy()[m["red"].to_numpy()].sum())
    fp = int((ans == RED).to_numpy()[(m["green"] | m["nolight"]).to_numpy()].sum())
    out["red_precision"] = dict(tp=tp, fp=fp, est=tp / max(tp + fp, 1))
    return out


def report(a):
    df = sweep_frames().set_index("id")
    rows, lines = [], []
    arch = {}
    for l in open(THIN_RUN / "baseline/shard-0-of-3.jsonl"):
        pass
    for s in glob.glob(str(THIN_RUN / "baseline/shard-*.jsonl")):
        for l in open(s):
            j = json.loads(l)
            arch[(j["cfg"], j["id"])] = j["ans"]
    ids = list(df.index)
    for name in MODELS:
        A = answers(name)
        if A.empty:
            continue
        meta = json.loads((OUT / name / "meta.json").read_text()) if (OUT / name / "meta.json").exists() else {}
        for cfg in CFGS:
            S = A[A.cfg == cfg].set_index("id").ans
            if len(S) < len(ids):
                continue
            sub = df.loc[ids].reset_index()
            M = metrics(sub, S.reindex(ids).reset_index(drop=True))
            rows.append(dict(model=name, cfg=cfg, n_vis=meta.get("n_vis", {}).get(cfg, ""), red=M["red_recall"], red_green=M["red_as_green"],
                             green=M["green_recall"], nolight=M["nolight_fp"], prec=M["red_precision"],
                             peak=meta.get("peak_gb", {}).get(cfg, np.nan)))
    head = "| model | config | visual tokens | ego red: red | ego red: green | ego green: green | no light: red | red precision | peak VRAM GiB |\n|:--|:--|--:|:--|:--|:--|:--|:--|--:|"
    body = ["| %s | %s | %s | %s (%d/85) | %s (%d) | %s (%d/68) | %s (%d/80) | %.1f%% (%d/%d) | %.1f |" % (
        r["model"], r["cfg"], r["n_vis"], cell(r["red"]), r["red"]["hits"], cell(r["red_green"]), r["red_green"]["hits"], cell(r["green"]),
        r["green"]["hits"], cell(r["nolight"]), r["nolight"]["hits"], 100 * r["prec"]["est"], r["prec"]["tp"], r["prec"]["tp"] + r["prec"]["fp"],
        r["peak"]) for r in rows]
    lat = ["| model | params (B) | latency p50 / p95 ms (decode + preprocess + forward) | forward only p50 / p95 ms | peak VRAM GiB (bench) |\n|:--|--:|:--|:--|--:|"]
    for name in MODELS:
        p = OUT / name / "meta.json"
        if p.exists():
            m = json.loads(p.read_text())
            lat.append("| %s | %.2f | %.0f / %.0f | %.0f / %.0f | %.1f |" % (name, m["params_b"], m["lat"]["p50"], m["lat"]["p95"], m["fwd"]["p50"], m["fwd"]["p95"], m["bench_peak_gb"]))
    # control vs archive
    ctl = []
    A = answers("q3vl4b")
    if not A.empty:
        for cfg, acfg in [("gen_native", "gen_ref"), ("fwd_r1153", "fwd_r1153"), ("fwd_r4573", "fwd_r4573"), ("fwd_r559", "fwd_r559")]:
            S = A[A.cfg == cfg].set_index("id").ans
            if len(S) < len(ids):
                continue
            agree = np.mean([S[i] == arch[(acfg, i)] for i in ids])
            a_ans = pd.Series([arch[(acfg, i)] for i in ids])
            M = metrics(df.loc[ids].reset_index(), a_ans)
            ctl.append("| %s | %d/85 vs archived %d/85 | answers identical to archive on %.1f%% of frames |" % (
                cfg, next(r["red"]["hits"] for r in rows if r["model"] == "q3vl4b" and r["cfg"] == cfg), M["red_recall"]["hits"], 100 * agree))
    txt = "## Accuracy (233 sweep frames, 85 ego red / 68 ego green / 80 no light; cell = estimate [95%% route-cluster CI])\n\n%s\n%s\n\n## Latency (r1153, batch 1, 100 frames)\n\n%s\n\n## Control vs archive\n\n| config | ego red recall | note |\n|:--|:--|:--|\n%s\n" % (
        head, "\n".join(body), "\n".join(lat), "\n".join(ctl))
    p = Path(a.out)
    p.write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--limit", type=int, default=0)
    q = sub.add_parser("report")
    q.add_argument("--out", default=str(DATA / "runs/small_vlm/eval/report_body.md"))
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
