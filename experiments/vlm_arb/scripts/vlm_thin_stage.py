"""GPU stages of the vlm_thin study, one process per card (the chain, vlm_thin_chain.py, launches them).

  vlm_thin_stage.py selftest                       preprocessing / truncation equivalence checks on a few frames
  vlm_thin_stage.py baseline --shard i/n           part 1: accuracy of the training-free variants on the 233 sweep frames
  vlm_thin_stage.py extract  --shard i/n           hidden states of every frame at every resolution (feature cache)
  vlm_thin_stage.py bench    --group G             batch-1 latency per variant on 100 test frames (quiet card)
  vlm_thin_stage.py gen      --shard i/n           Phase A (a): full zero-shot generate on the test frames
  vlm_thin_stage.py serve    --shard i/n           Phase A (b): the selected truncated variant + thin head on the test frames

Every stage reads/writes under --run (the chain's run dir) and skips what is already there.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_thin_common import (CACHE, CUTS, FOUR_PROMPT, LIGHTS, NL, RES, Thin, jwrite, load_frames, log, parse_option,  # noqa: E402
                             percentiles, read_jpgs, sync)
from vlm_thin_fit import frames_tag  # noqa: E402
from vlm_arb_models_worker import PROMPT as JOINT_PROMPT, parse_reply  # noqa: E402
from vlm_arb_common import REPO  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive import cache, par  # noqa: E402

POOL_LAYERS = [0] + CUTS                    # layers whose per-image mean of the image tokens is cached
N_BENCH = 100                               # frames per latency measurement (evenly spaced over the test frames)
N_WARM = 5


def shard_arg(s):
    i, n = s.split("/")
    return int(i), int(n)


def frames_for(run, part=None):
    df = pd.read_csv(Path(run) / "frames.csv", dtype={"route": str})
    return df if part is None else df[df.part == part].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------- selftest
def selftest(a):
    import torch
    th = Thin()
    df = frames_for(a.run)
    t = df[df.part == "test"]
    rows = t.iloc[np.unique(np.linspace(0, len(t) - 1, 3).astype(int))]
    res = dict(checks=[])
    for r in rows.itertuples():
        ref = th.prep_ref([r.wide, r.road])
        xg = th.prep(read_jpgs(r), RES["r4573"], gen=True)
        x = th.prep(read_jpgs(r), RES["r4573"])
        c = dict(id=r.id, ids_equal=bool(torch.equal(ref["input_ids"][0], xg["input_ids"][0])),
                 mm_equal=bool(torch.equal(ref["mm_token_type_ids"][0].to(torch.int64), xg["mm_token_type_ids"][0])) if "mm_token_type_ids" in ref else None,
                 pix_abs_mean_diff=float((ref["pixel_values"].float() - xg["pixel_values"].float()).abs().mean()),
                 grid_equal=bool(torch.equal(ref["image_grid_thw"], xg["image_grid_thw"])))
        with torch.no_grad():
            th.restore()
            lg = th.m(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=x["pixel_values"],
                      image_grid_thw=x["image_grid_thw"], mm_token_type_ids=x["mm_token_type_ids"], logits_to_keep=1).logits[0, -1].float()
            out = th.m.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=x["pixel_values"],
                             image_grid_thw=x["image_grid_thw"], mm_token_type_ids=x["mm_token_type_ids"], use_cache=False,
                             output_hidden_states=True)
            lg2 = th.m.lm_head(out.last_hidden_state[0, -1:]).float()[0]
        c["opt_logits_model_forward"], c["opt_logits_lmhead_of_hidden"] = lg[th.opt_ids].tolist(), lg2[th.opt_ids].tolist()
        c["opt_logits_W"] = th.option_logits(out.last_hidden_state[0, -1]).tolist()
        c["hs_last_is_normed"] = float((out.hidden_states[NL][0, -1].float() - out.last_hidden_state[0, -1].float()).abs().max())
        # truncated forward == hidden_states[N] of the full forward
        c["trunc_vs_full_maxabs_rel"] = {}
        for N in (2, 12, 20, 36):
            h = th.run(x, N)
            ref_h = out.hidden_states[N][0, -1].float()
            c["trunc_vs_full_maxabs_rel"][N] = float((h - ref_h).abs().max() / ref_h.abs().max())
        th.restore()
        vo = th.vision_only(x)
        c["vision_only"] = [list(v.shape) for v in vo]
        s0 = len(th.pieces[0])
        c["vision_vs_layer0"] = float((vo[0].float() - out.hidden_states[0][0, s0:s0 + vo[0].shape[0]].float()).abs().max())
        # generate: the reference path and the GPU-prep path answer the same?
        with torch.no_grad():
            g1 = th.m.generate(**ref, max_new_tokens=160, do_sample=False)
            t1 = th.proc.batch_decode(g1[:, ref["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            g2 = th.m.generate(input_ids=xg["input_ids"], attention_mask=xg["attention_mask"], pixel_values=xg["pixel_values"],
                               image_grid_thw=xg["image_grid_thw"], mm_token_type_ids=xg["mm_token_type_ids"], max_new_tokens=160,
                               do_sample=False)
            t2 = th.proc.batch_decode(g2[:, xg["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        c["gen_ref"], c["gen_gpuprep"] = t1, t2
        res["checks"].append(c)
        log(json.dumps(c))
    jwrite(Path(a.run) / "selftest.json", res)


# ---------------------------------------------------------------------------------------------------- baseline (part 1)
def score_answer(th, x):
    h = th.run(x, NL)
    return th.option_logits(h)


def baseline(a):
    import torch
    th = Thin()
    i, n = shard_arg(a.shard)
    df = frames_for(a.run)
    df = df[df["sweep"]].reset_index(drop=True)
    mine = par.shards(list(df.index), n, i)
    out = Path(a.run) / "baseline" / ("shard-%d-of-%d.jsonl" % (i, n))
    out.parent.mkdir(parents=True, exist_ok=True)
    have = {(json.loads(l)["cfg"], json.loads(l)["id"]) for l in open(out)} if out.exists() else set()
    cfgs = ["gen_ref"] + ["fwd_" + r for r in RES]
    with open(out, "a") as f, torch.no_grad():
        for k in mine:
            r = df.iloc[k]
            for cfg in cfgs:
                if (cfg, r.id) in have:
                    continue
                if cfg == "gen_ref":
                    x = th.prep_ref([r.wide, r.road])
                    th.restore()
                    y = th.m.generate(**x, max_new_tokens=160, do_sample=False)
                    txt = th.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                    row = dict(raw=txt, ans=parse_option(txt, LIGHTS))
                else:
                    x = th.prep(read_jpgs(r), RES[cfg[4:]])
                    lg = score_answer(th, x)
                    row = dict(ans=LIGHTS[int(lg.argmax())], logits=[round(float(v), 3) for v in lg])
                f.write(json.dumps(dict(cfg=cfg, id=r.id, **row)) + "\n")
                f.flush()
    log("baseline shard %d/%d done" % (i, n))


# ---------------------------------------------------------------------------------------------------- extract
def extract_fn(th, ids, df, res):
    import torch
    H, P, VM, ZS = [], [], [], []
    with torch.no_grad():
        for k, fid in enumerate(ids):
            r = df.loc[df.id == fid].iloc[0]
            x = th.prep(read_jpgs(r), RES[res])
            th.restore()
            out = th.m.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=x["pixel_values"],
                             image_grid_thw=x["image_grid_thw"], mm_token_type_ids=x["mm_token_type_ids"], use_cache=False,
                             output_hidden_states=True)
            hs = out.hidden_states
            s0 = len(th.pieces[0])
            n0, n1 = x["n_img"]
            s1 = s0 + n0 + len(th.pieces[1])
            H.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
            pool = []
            for L in POOL_LAYERS:
                h = hs[L][0]
                pool.append(torch.stack([h[s0:s0 + n0].float().mean(0), h[s1:s1 + n1].float().mean(0)]))
            P.append(torch.stack(pool).cpu().numpy())
            h0 = hs[0][0]
            VM.append(torch.stack([h0[s0:s0 + n0].float().amax(0), h0[s1:s1 + n1].float().amax(0)]).cpu().numpy())
            ZS.append(th.option_logits(out.last_hidden_state[0, -1]).cpu().numpy())
            if (k + 1) % 100 == 0:
                log("extract %s: %d / %d" % (res, k + 1, len(ids)))
    return dict(ids=np.array(ids), hid=np.stack(H), pool=np.stack(P), vismax=np.stack(VM), zs=np.stack(ZS))


def extract(a):
    th = Thin()
    i, n = shard_arg(a.shard)
    df = frames_for(a.run)
    ids = par.shards(sorted(df.id), n, i)
    for res in a.res.split(","):
        path = CACHE / "features" / frames_tag(df) / res / ("shard-%d-of-%d.npz" % (i, n))
        k = cache.key(dict(res=res, max_px=RES[res], pool_layers=POOL_LAYERS, ids=len(ids), shard=[i, n], prompt=FOUR_PROMPT),
                      code=[extract_fn, Thin], version="v1")
        cache.cached(path, k, lambda: extract_fn(th, ids, df, res))
        log("extract %s shard %d/%d ready" % (res, i, n))


# ---------------------------------------------------------------------------------------------------- bench
def bench_frames(run):
    df = frames_for(run, "test")
    idx = np.unique(np.round(np.linspace(0, len(df) - 1, N_BENCH)).astype(int))
    return df.iloc[idx].reset_index(drop=True)


def timed(fn, frames, warm=N_WARM):
    """Per-frame wall ms of fn(frame jpgs) -> (answer idx); the result is read on the host inside the timed region."""
    import torch
    for k in range(warm):
        fn(frames[k % len(frames)])
    sync(torch)
    ms = []
    for jp in frames:
        t = time.perf_counter()
        fn(jp)
        ms.append((time.perf_counter() - t) * 1000)
    return ms


def make_fn(th, cfg, torch):
    """cfg -> fn(jpgs) that does the whole request and returns a host int. Names: gen_ref, gen_gpu, fwd_<res>,
    fwd_<res>_compile, cut_<res>_<N>, vis_<res>."""
    head = lambda d: (torch.nn.Linear(d, 3).cuda(), torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 3)).cuda())  # noqa: E731
    dummy_lin, dummy_mlp = head(5120)
    vis_lin, vis_mlp = head(10240)
    part = cfg.split("_")
    if cfg == "gen_ref":
        def fn(jp):
            import io
            from PIL import Image
            imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jp]
            msgs = [{"role": "user", "content": [{"type": "image", "image": i} for i in imgs] + [{"type": "text", "text": FOUR_PROMPT}]}]
            x = th.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            y = th.m.generate(**x, max_new_tokens=160, do_sample=False)
            return len(th.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return fn
    if cfg == "gen_gpu":
        def fn(jp):
            x = th.prep(jp, RES["r4573"], gen=True)
            y = th.m.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=x["pixel_values"],
                              image_grid_thw=x["image_grid_thw"], mm_token_type_ids=x["mm_token_type_ids"], max_new_tokens=160,
                              do_sample=False)
            return len(th.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return fn
    res = part[1]
    if part[0] == "fwd":
        def fn(jp):
            x = th.prep(jp, RES[res])
            return int(score_answer(th, x).argmax())
        return fn
    if part[0] == "cut":
        N = int(part[2])

        def fn(jp):
            x = th.prep(jp, RES[res])
            last, pool = th.run(x, N, pool=True)
            f = torch.cat([pool.flatten()])                        # the pooled feature (upper bound of the head cost)
            return int(dummy_mlp(f).argmax()) + int(dummy_lin(f).argmax())
        return fn
    if part[0] == "vis":
        def fn(jp):
            x = th.prep(jp, RES[res])
            vo = th.vision_only(x)
            f = torch.cat([torch.cat([v.float().mean(0) for v in vo]), torch.cat([v.float().amax(0) for v in vo])])
            return int(vis_mlp(f).argmax()) + int(vis_lin(f).argmax())
        return fn
    raise ValueError(cfg)


def breakdown(th, res, torch, frames):
    """Mean ms of the stages of one full-model request at resolution `res`: decode + patchify (prep), vision tower, language
    layers (everything after the vision tower, incl. the one-token readout)."""
    th.restore()
    P, V, T = [], [], []
    for jp in frames[:20] + frames[:20]:
        sync(torch)
        t0 = time.perf_counter()
        x = th.prep(jp, RES[res])
        sync(torch)
        t1 = time.perf_counter()
        th.vision_only(x)
        sync(torch)
        t2 = time.perf_counter()
        score_answer(th, x)
        sync(torch)
        t3 = time.perf_counter()
        P.append(t1 - t0)
        V.append(t2 - t1)
        T.append(t3 - t2)
    P, V, T = (np.mean(v[20:]) * 1000 for v in (P, V, T))
    return dict(prep_ms=P, vision_ms=V, full_model_ms=T, llm_ms=T - V)


def bench(a):
    import torch
    th = Thin()
    fr = bench_frames(a.run)
    jp = [read_jpgs(r) for r in fr.itertuples()]
    outdir = Path(a.run) / "bench"
    outdir.mkdir(exist_ok=True)
    for cfg in a.group.split(","):
        path = outdir / (cfg + ".json")
        if path.exists():
            continue
        th.restore()
        err = None
        try:
            if cfg.endswith("_compile"):
                t0 = time.time()
                th.m.model.language_model.compile()
                th.m.model.visual.compile()
                fn = make_fn(th, cfg.replace("_compile", ""), torch)
                fn(jp[0])
                sync(torch)
                comp_s = time.time() - t0
            else:
                fn = make_fn(th, cfg, torch)
                comp_s = None
            with torch.no_grad():
                ms = timed(fn, jp)
        except Exception as e:  # noqa: BLE001
            import traceback
            err = traceback.format_exc()[-1500:]
            ms = []
        row = dict(cfg=cfg, frames=len(jp), error=err, compile_s=comp_s if err is None else None)
        if ms:
            row.update(percentiles(ms), ms=[round(v, 1) for v in ms])
        if cfg.startswith("fwd_") and not cfg.endswith("_compile") and err is None:
            row["breakdown"] = breakdown(th, cfg[4:], torch, jp)
        jwrite(path, row)
        log("bench %s: %s" % (cfg, ("p50 %.0f p95 %.0f ms" % (row["p50"], row["p95"])) if ms else "FAILED " + str(err)[-200:]))


# ---------------------------------------------------------------------------------------------------- Phase A (a)
def gen(a):
    import torch
    th = Thin()
    i, n = shard_arg(a.shard)
    df = frames_for(a.run, "test")
    mine = par.shards(list(df.index), n, i)
    out = Path(a.run) / "phase_a" / ("gen-%d-of-%d.jsonl" % (i, n))
    out.parent.mkdir(parents=True, exist_ok=True)
    have = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    f = open(out, "a")
    with torch.no_grad():
        for k in mine:
            r = df.iloc[k]
            if r.id in have:
                continue
            row = dict(id=r.id)
            for tag, prompt, max_new in (("light", FOUR_PROMPT, 160), ("joint", JOINT_PROMPT, 64)):
                sync(torch)
                t = time.perf_counter()
                x = th.prep_ref([r.wide, r.road], prompt=prompt)
                y = th.m.generate(**x, max_new_tokens=max_new, do_sample=False)
                txt = th.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                row[tag + "_ms"] = round((time.perf_counter() - t) * 1000, 1)
                row[tag + "_raw"] = txt
            row["light"] = parse_option(row["light_raw"], LIGHTS)
            row["joint"] = parse_reply(row["joint_raw"])
            f.write(json.dumps(row) + "\n")
            f.flush()
    log("gen shard %d/%d done" % (i, n))


# ---------------------------------------------------------------------------------------------------- Phase A (b)
def load_head(path):
    import torch
    from vlm_thin_fit import probs
    d = torch.load(path, map_location="cuda")
    return d, lambda f: probs(torch, d, ((f.float() - d["mu"]) / d["sd"])[None])[0]


def serve(a):
    import torch
    from vlm_thin_common import ANS3, LIGHT3
    sel = json.loads((Path(a.run) / a.sel).read_text())["chosen"]
    th = Thin()
    N, res, feat = sel["N"], sel["res"], sel["feat"]
    fwd = None if feat == "zs" else load_head(Path(a.run) / sel["head_file"])[1]
    i, n = shard_arg(a.shard)
    df = frames_for(a.run, "test")
    mine = par.shards(list(df.index), n, i)
    out = Path(a.run) / "phase_a" / ("%s-%d-of-%d.jsonl" % ("serve" if a.sel == "selection.json" else "serve-trunc", i, n))
    out.parent.mkdir(exist_ok=True)
    have = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    f = open(out, "a")
    th.restore()

    def request(jp):
        """The whole request from JPEG bytes to (answer string, probabilities) on the host."""
        x = th.prep(jp, RES[res])
        if feat == "zs":
            lg = score_answer(th, x)
            return LIGHTS[int(lg.argmax())], torch.softmax(lg, -1).cpu().numpy()
        if feat == "vis":
            vo = th.vision_only(x)
            ft = torch.cat([torch.cat([v.float().mean(0) for v in vo]), torch.cat([v.float().amax(0) for v in vo])])
        elif feat == "pool":
            ft = th.run(x, N, pool=True)[1].flatten()
        else:
            ft = th.run(x, N)
        p = fwd(ft).cpu().numpy()
        return ANS3[LIGHT3[int(p.argmax())]], p
    with torch.no_grad():
        for k in range(N_WARM):
            request(read_jpgs(df.iloc[mine[k % len(mine)]]))
        for k in mine:
            r = df.iloc[k]
            if r.id in have:
                continue
            jp = read_jpgs(r)
            sync(torch)
            t = time.perf_counter()
            ans, p = request(jp)
            ms = (time.perf_counter() - t) * 1000
            f.write(json.dumps(dict(id=r.id, ans=ans, p=[round(float(v), 5) for v in p], ms=round(ms, 1))) + "\n")
            f.flush()
    log("serve shard %d/%d done" % (i, n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "baseline", "extract", "bench", "gen", "serve"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--res", default=",".join(RES))
    ap.add_argument("--group", default="")
    ap.add_argument("--sel", default="selection.json")
    a = ap.parse_args()
    globals()[a.cmd](a)
