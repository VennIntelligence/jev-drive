"""GPU stages of the vlm_cmp study (Qwen3-VL-4B vs 8B), one process per card; the chain (vlm_cmp_chain.py) launches them.

  vlm_cmp_stage.py selftest --model 4b                    cache path == full forward; 4B == the earlier vlm_thin reader
  vlm_cmp_stage.py extract  --model M --shard i/n         single frame: option log-probs of 5 questions + hidden states, 4 resolutions
  vlm_cmp_stage.py twoframe --model M --shard i/n         previous + current frame (4 images), directive / light / block
  vlm_cmp_stage.py bench    --model M --group G           batch-1 latency and peak memory per variant (quiet card)

All stages read frames.csv of the run dir (--run) and skip what is already there.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_cmp_frames import DIRECTIVES  # noqa: E402
from vlm_cmp_model import OPTIONS, RES, VL, prompt_text, sync  # noqa: E402
from vlm_arb_common import REPO  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive import cache, par  # noqa: E402

CHUNK = 40                                       # frames per cached chunk file
HID = ("light", "sign", "block", "dir")          # questions whose answer-position hidden states are kept
SINGLE = ("light", "sign", "block", "side", "dir")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def shard_arg(s):
    i, n = s.split("/")
    return int(i), int(n)


def frames(run):
    return pd.read_csv(Path(run) / "frames.csv", dtype={"route": str, "attempt": str}, keep_default_na=False,
                       na_values=[""])


def jpgs(*paths):
    return [Path(p).read_bytes() for p in paths]


def frames_tag(df):
    import hashlib
    return "%d-%s" % (len(df), hashlib.sha1("\n".join(df.id).encode()).hexdigest()[:8])


# ---------------------------------------------------------------------------------------------------- selftest
def selftest(a):
    import torch
    vl = VL(a.model)
    df = frames(a.run)
    rows = df.iloc[np.unique(np.linspace(0, len(df) - 1, 4).astype(int))]
    res = dict(model=a.model, NL=vl.NL, d=vl.d, checks=[])
    for r in rows.itertuples():
        for rn in ("r559", "r1153"):
            x = vl.prep(jpgs(r.wide, r.road), RES[rn])
            c = dict(id=r.id, res=rn, n_tokens=int(x["ids"].numel()))
            for name in ("light", "dir"):
                text = prompt_text(name, v=float(r.v))
                st = vl.prefill(x, pooled=False)
                lp, h = vl.ask(st, 2, text, name, hidden=True)
                lg, hf = vl.full_logits(x, text)
                seqs, groups = vl._opt[name]
                firsts = [s[0] for s in seqs]
                lpf = torch.log_softmax(lg, -1)[firsts].cpu().numpy()
                c[name] = dict(cache_lp=lp.tolist(), full_first_lp=lpf.tolist(),
                               max_abs_lp_diff_unique_first=float(max(abs(lp[i] - lpf[i]) for ix in groups.values() if len(ix) == 1 for i in ix)),
                               hid_rel_diff=float(np.abs(h.astype(np.float32) - hf.cpu().numpy()).max() / np.abs(hf.cpu().numpy()).max()),
                               same_argmax_cache_vs_full=bool(int(np.argmax(lp)) == int(np.argmax(lpf)) or len(set(firsts)) < len(firsts)))
                # a second ask on the same state must give the same numbers (the cache was cropped back)
                lp2, _ = vl.ask(st, 2, text, name, hidden=False)
                c[name]["repeat_equal"] = bool(np.array_equal(lp, lp2))
            res["checks"].append(c)
            log(json.dumps(c))
    (Path(a.run) / ("selftest-%s.json" % a.model)).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------------- extract
def extract_chunk(vl, ids, df, rn):
    import torch
    out = {"lp_" + n: [] for n in SINGLE}
    out.update({"h_" + n: [] for n in HID})
    out.update(pool=[], ntok=[])
    by_id = df.set_index("id")
    t0 = time.time()
    with torch.no_grad():
        for k, fid in enumerate(ids):
            r = by_id.loc[fid]
            x = vl.prep(jpgs(r.wide, r.road), RES[rn])
            st = vl.prefill(x, pooled=True)
            for name in SINGLE:
                lp, h = vl.ask(st, 2, prompt_text(name, v=float(r.v)), name, hidden=name in HID)
                out["lp_" + name].append(lp)
                if h is not None:
                    out["h_" + name].append(h)
            out["pool"].append(st["pool"].half().cpu().numpy())
            out["ntok"].append(int(x["ids"].numel()))
            del st
    log("chunk of %d frames %s: %.1f s per frame" % (len(ids), rn, (time.time() - t0) / max(len(ids), 1)))
    res = {k: np.stack(v) for k, v in out.items()}
    res["ids"] = np.array(ids)
    return res


def extract(a):
    vl = VL(a.model)
    i, n = shard_arg(a.shard)
    df = frames(a.run)
    mine = par.shards(sorted(df.id), n, i)
    chunks = [mine[j:j + CHUNK] for j in range(0, len(mine), CHUNK)]
    root = Path(a.run) / "features" / a.model
    for rn in a.res.split(","):
        for c, ids in enumerate(chunks):
            path = root / rn / ("shard%d-%d-chunk%03d.npz" % (i, n, c))
            k = cache.key(dict(model=a.model, res=rn, max_px=RES[rn], ids=ids, prompts={q: prompt_text(q, v=0.0) for q in SINGLE}),
                          code=[extract_chunk, VL], version="v1")
            cache.cached(path, k, lambda: extract_chunk(vl, ids, df, rn))
        log("extract %s %s shard %d/%d done (%d frames)" % (a.model, rn, i, n, len(mine)))


# ---------------------------------------------------------------------------------------------------- two frames
def twoframe_chunk(vl, ids, df, rn):
    import torch
    out = dict(lp_dir_prev=[], lp_dir=[], lp_light=[], lp_block=[], prev_dir=[])
    by_id = df.set_index("id")
    with torch.no_grad():
        for fid in ids:
            r = by_id.loc[fid]
            # the model's own directive on the earlier frame (single frame), as the "previous directive" of the stack
            x1 = vl.prep(jpgs(r.prev_wide, r.prev_road), RES[rn])
            st1 = vl.prefill(x1)
            lp0, _ = vl.ask(st1, 2, prompt_text("dir", v=float(r.prev_v)), "dir")
            prev = DIRECTIVES[int(np.argmax(lp0))]
            x2 = vl.prep(jpgs(r.prev_wide, r.prev_road, r.wide, r.road), RES[rn])
            st2 = vl.prefill(x2)
            out["lp_dir_prev"].append(lp0)
            out["prev_dir"].append(prev)
            out["lp_dir"].append(vl.ask(st2, 4, prompt_text("dir_2f", v=float(r.v), prev=prev), "dir")[0])
            out["lp_light"].append(vl.ask(st2, 4, prompt_text("light_2f"), "light")[0])
            out["lp_block"].append(vl.ask(st2, 4, prompt_text("block_2f"), "block")[0])
            del st1, st2
    res = {k: np.stack(v) for k, v in out.items() if k != "prev_dir"}
    res["prev_dir"] = np.array(out["prev_dir"])
    res["ids"] = np.array(ids)
    return res


def twoframe(a):
    vl = VL(a.model)
    i, n = shard_arg(a.shard)
    df = frames(a.run)
    df = df[df.prev_t.notna()].reset_index(drop=True)
    mine = par.shards(sorted(df.id), n, i)
    chunks = [mine[j:j + CHUNK] for j in range(0, len(mine), CHUNK)]
    root = Path(a.run) / "twoframe" / a.model
    for rn in a.res.split(","):
        for c, ids in enumerate(chunks):
            path = root / rn / ("shard%d-%d-chunk%03d.npz" % (i, n, c))
            k = cache.key(dict(model=a.model, res=rn, ids=ids, p=[prompt_text("dir_2f", prev="X"), prompt_text("light_2f"), prompt_text("block_2f")]),
                          code=[twoframe_chunk, VL], version="v1")
            t0 = time.time()
            cache.cached(path, k, lambda: twoframe_chunk(vl, ids, df, rn))
            log("twoframe %s %s chunk %d/%d %.1f s" % (a.model, rn, c + 1, len(chunks), time.time() - t0))


# ---------------------------------------------------------------------------------------------------- bench
N_BENCH, N_WARM = 100, 5


def bench_rows(run):
    df = frames(run)
    df = df[df.prev_t.notna()].sort_values(["route", "id"]).reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(df) - 1, N_BENCH)).astype(int))
    return df.iloc[idx].reset_index(drop=True)


def make_fn(vl, cfg, torch):
    """cfg -> fn(row_jpgs) doing the whole request from JPEG bytes to the answer on the host.
    q1_<res>        one question (light), one forward pass with no cache: the path the closed loop served
    dir_<res>       directive question: prefill + suffix (+ second step if the pass group needs it)
    four_<res>      the four separate questions on one prefill (light, sign, block, side)
    two_<res>       two-moment directive: 4 images prefill + suffix
    cut_<res>_<N>   truncated model, first N layers, light prompt, answer-position state (head cost excluded: a linear layer)
    """
    kind, rn = cfg.split("_")[0], cfg.split("_")[1]
    mx = RES[rn]
    if kind == "q1":
        def fn(jp):
            x = vl.prep(jp[:2], mx)
            vl.set_cut(vl.NL)
            t = torch
            pieces, tail = vl.split_prompt(2, prompt_text("light"))
            ids = t.cat([x["ids"], tail])
            out = vl.m.model(input_ids=ids[None], attention_mask=t.ones_like(ids)[None], pixel_values=x["pixel_values"],
                             image_grid_thw=x["grid"], mm_token_type_ids=(ids == vl.img_id).to(t.int64)[None], use_cache=False)
            return int((vl.Wf @ out.last_hidden_state[0, -1].float()).argmax())
        return fn
    if kind == "dir":
        def fn(jp):
            x = vl.prep(jp[:2], mx)
            st = vl.prefill(x)
            lp, _ = vl.ask(st, 2, prompt_text("dir", v=0.0), "dir")
            return int(np.argmax(lp))
        return fn
    if kind == "four":
        def fn(jp):
            x = vl.prep(jp[:2], mx)
            st = vl.prefill(x)
            return sum(int(np.argmax(vl.ask(st, 2, prompt_text(q, v=0.0), q)[0])) for q in ("light", "sign", "block", "side"))
        return fn
    if kind == "two":
        def fn(jp):
            x = vl.prep(jp[:4], mx)
            st = vl.prefill(x)
            lp, _ = vl.ask(st, 4, prompt_text("dir_2f", v=0.0, prev="proceed"), "dir")
            return int(np.argmax(lp))
        return fn
    if kind == "cut":
        N = int(cfg.split("_")[2])

        def fn(jp):
            x = vl.prep(jp[:2], mx)
            return int(vl.cut_state(x, prompt_text("light"), N)[0] > 0)
        return fn
    raise ValueError(cfg)


def percentiles(ms):
    ms = np.asarray(ms, float)
    return dict(n=int(len(ms)), p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)),
                p99=float(np.percentile(ms, 99)), mean=float(ms.mean()))


def bench(a):
    import torch
    vl = VL(a.model)
    rows = bench_rows(a.run)
    # JPEG bytes in host memory: [prev wide, prev road, wide, road]
    jp = [jpgs(r.prev_wide, r.prev_road, r.wide, r.road) for r in rows.itertuples()]
    outdir = Path(a.run) / "bench" / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    weights_gb = torch.cuda.memory_allocated() / 2 ** 30
    for cfg in a.group.split(","):
        path = outdir / (cfg + ".json")
        if path.exists():
            continue
        vl.set_cut(vl.NL)
        fn = make_fn(vl, cfg, torch)
        err, ms = None, []
        try:
            with torch.no_grad():
                for k in range(N_WARM):
                    fn(jp[k % len(jp)])
                sync(torch)
                torch.cuda.reset_peak_memory_stats()
                for j in jp:
                    t = time.perf_counter()
                    fn(j)
                    sync(torch)
                    ms.append((time.perf_counter() - t) * 1000)
        except Exception:  # noqa: BLE001
            import traceback
            err = traceback.format_exc()[-1500:]
        row = dict(cfg=cfg, model=a.model, frames=len(jp), error=err, weights_gb=weights_gb, wf_gb=vl.Wf.numel() * 4 / 2 ** 30,
                   peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        if ms:
            row.update(percentiles(ms), ms=[round(v, 1) for v in ms])
        (outdir / (cfg + ".json")).write_text(json.dumps(row, indent=1))
        log("bench %s %s: %s" % (a.model, cfg, ("p50 %.0f p95 %.0f ms peak %.1f GiB" % (row["p50"], row["p95"], row["peak_gb"])) if ms else "FAILED " + str(err)[-300:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "extract", "twoframe", "bench"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True, choices=["4b", "8b"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--res", default=",".join(RES))
    ap.add_argument("--group", default="")
    a = ap.parse_args()
    globals()[a.cmd](a)
