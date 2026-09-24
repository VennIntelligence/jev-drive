#!/usr/bin/env python
"""Alpamayo 1.5 zero-shot on NAVSIM (todos/2026-09-24-zeroshot-exam/navsim.md). Runs in third_party/alpamayo1.5/.venv.

  adapt   adapter-loss ablation on PhysicalAI-AV clips (native 10 Hz data with ground truth): native inputs vs the
          NAVSIM compromises (2 Hz frames: repeat t0 or 0.5 s-spaced; 2 Hz egomotion; nuPlan camera round trip)
  viz     model images + history of a few NAVSIM tokens, for eyeballing the reprojection
  run     inference over a split (--shard i/n), batched by prompt; appends one JSON line per token and variant
  bench   throughput vs batch size on a fixed token subset (the optimization pass)

    PY=$DATA_DIR/third_party/alpamayo1.5/.venv/bin/python
    scripts/tmux_run.sh nzs-a0 $PY scripts/navsim_zs_alpamayo.py run --split navtest --shard 0/2
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

RIG = Z.rigs()
VARIANTS = ("nav", "nonav")
FRAMES = {"repeat": [3, 3, 3, 3], "2hz": [0, 1, 2, 3]}   # which of the 4 NAVSIM history frames fill the 4 slots


# ---------------------------------------------------------------- inputs

class Inputs:
    """CPU side, thread-safe: JPEG decode (DCT-scaled for the 120-degree views), reprojection, egomotion."""

    def __init__(self, frames: str):
        self.slots = FRAMES[frames]
        self.maps, self.lock = {}, threading.Lock()

    def maps_for(self, cams):
        k = Z.calib_key(cams)
        with self.lock:
            m = self.maps.get(k)
        if m is None:
            m = Z.AlpamayoMaps(cams, RIG)
            with self.lock:
                self.maps[k] = m
        return m

    @staticmethod
    def decode(path: str, scale: int) -> np.ndarray:
        import simplejpeg
        buf = Path(path).read_bytes()
        if scale == 1:
            return simplejpeg.decode_jpeg(buf, colorspace="RGB")
        w, h = Z.NUPLAN_WH
        return simplejpeg.decode_jpeg(buf, colorspace="RGB", min_width=w // scale, min_height=h // scale)

    def __call__(self, e: dict) -> dict:
        m = self.maps_for(e["cams"][-1])
        need = sorted(set(self.slots))
        views = {}
        for f in need:
            imgs = {(c, s): self.decode(e["cams"][f][c]["path"], s) for c, s in m.sources()}
            views[f] = m.render(imgs)                                 # (4 cams, 3, 320, 576)
        frames = np.stack([views[f] for f in self.slots], 1)          # (4 cams, 4 slots, 3, 320, 576)
        xyz, rot = Z.alpamayo_history(e["pose"], e["vel"])
        return {"token": e["token"], "frames": frames, "cam_idx": np.asarray(m.cam_idx), "xyz": xyz, "rot": rot,
                "nav": Z.nav_text(e["cmd"][-1])}


def tokenize(proc, s: dict, nav: str | None):
    import torch
    from alpamayo1_5 import helper
    frames = torch.from_numpy(s["frames"])
    msgs = helper.create_message(frames=frames.flatten(0, 1), camera_indices=torch.from_numpy(s["cam_idx"]),
                                 nav_text=nav)
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                    return_dict=True, return_tensors="pt")


def collate(toks: list, xyz: list, rot: list) -> dict:
    """Same-length prompts only (grouped by nav text): stack ids/masks, concatenate the flattened image patches."""
    import torch
    L = {t["input_ids"].shape[1] for t in toks}
    assert len(L) == 1, f"mixed prompt lengths {L}"
    td = {k: torch.cat([t[k] for t in toks]) for k in toks[0].keys()}   # per-sequence rows and flattened patches
    return {"tokenized_data": {k: v.cuda(non_blocking=True) for k, v in td.items()},
            "ego_history_xyz": torch.from_numpy(np.stack(xyz))[:, None].cuda(),
            "ego_history_rot": torch.from_numpy(np.stack(rot))[:, None].cuda()}


# ---------------------------------------------------------------- model

class Model:
    def __init__(self, attn="sdpa", compile_=("visual", "expert"), flow_steps=10):
        import torch
        from jevdrive.alpamayo import infer as I
        self.torch, self.I = torch, I
        self.cfg = I.Config(name="navsim", attn=attn, compile=tuple(compile_), flow_steps=flow_steps)
        self.model, self.proc = I.load(attn)
        I.apply(self.model, self.cfg)

    def __call__(self, batch: dict, seed: int, n: int = 1) -> dict:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            t0 = time.perf_counter()
            xyz, rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=batch, top_p=self.cfg.top_p, temperature=self.cfg.temperature, num_traj_samples=n,
                num_traj_sets=1, max_generation_length=self.cfg.max_gen, return_extra=True,
                diffusion_kwargs={"inference_step": self.cfg.flow_steps})
            torch.cuda.synchronize()
        if n > 1:
            return {"xyz": xyz[:, 0].float().cpu().numpy(), "wall": time.perf_counter() - t0}
        return {"xyz": xyz[:, 0, 0].float().cpu().numpy(), "rot": rot[:, 0, 0].float().cpu().numpy(),
                "cot": [str(c) for c in np.asarray(extra["cot"]).reshape(len(xyz), -1)[:, 0]],
                "wall": time.perf_counter() - t0}


def seed_of(token: str) -> int:
    return int(token[:8], 16) % (2 ** 31)


def run_stream(model, samples, variants, B, emit, log=None):
    """samples: iterator of Inputs outputs. Buckets by (variant, nav text) so each forward has equal-length
    prompts, runs a bucket when it has B samples, flushes the rest at the end."""
    buckets = {}

    def fire(key):
        items = buckets.pop(key)
        toks = [it[1] for it in items]
        batch = collate(toks, [it[0]["xyz"] for it in items], [it[0]["rot"] for it in items])
        o = model(batch, seed_of(items[0][0]["token"]))
        for i, (s, _) in enumerate(items):
            emit({"token": s["token"], "variant": key[0], "nav_text": key[1], "B": len(items),
                  "ms": 1e3 * o["wall"] / len(items), "poses": Z.alpamayo_to_navsim(o["xyz"][i], o["rot"][i]).tolist(),
                  "xyz": np.round(o["xyz"][i], 3).tolist(), "cot": o["cot"][i]})

    for s in samples:
        for v in variants:
            nav = s["nav"] if v == "nav" else None
            key = (v, nav)
            buckets.setdefault(key, []).append((s, s["tok"][nav]))
            if len(buckets[key]) >= B:
                fire(key)
    for key in list(buckets):
        fire(key)


def prefetch(fn, items, workers: int, ahead: int):
    """Ordered map with bounded lookahead on a thread pool."""
    with ThreadPoolExecutor(workers) as ex:
        futs = []
        it = iter(items)
        for x in it:
            futs.append(ex.submit(fn, x))
            if len(futs) >= ahead:
                break
        while futs:
            f = futs.pop(0)
            nxt = next(it, None)
            if nxt is not None:
                futs.append(ex.submit(fn, nxt))
            yield f.result()


def prepared(model, inputs, entries, variants, workers, ahead=32):
    """Inputs + tokenization for every variant, in worker threads."""
    def one(e):
        s = inputs(e)
        navs = {s["nav"] if v == "nav" else None for v in variants}
        s["tok"] = {n: tokenize(model.proc, s, n) for n in navs}
        return s
    return prefetch(one, entries, workers, ahead)


# ---------------------------------------------------------------- commands

def cmd_run(a, log):
    idx = Z.load_index(a.split)
    i, n = map(int, a.shard.split("/"))
    entries = [e for k, e in enumerate(idx) if k % n == i]
    out = Z.root("alpamayo", a.split) / f"{a.frames}_{a.tag}_shard{i}of{n}.jsonl"
    done = set()
    if out.exists():
        done = {(r["token"], r["variant"]) for r in map(json.loads, out.read_text().splitlines())}
    variants = a.variants.split(",")
    todo = [e for e in entries if any((e["token"], v) not in done for v in variants)]
    if a.limit:
        todo = todo[:a.limit]
    log.info(f"{a.split} shard {i}/{n}: {len(entries)} tokens, {len(todo)} to do, variants {variants}, B {a.batch} -> {out}")
    model = Model(a.attn, a.compile.split(",") if a.compile else (), a.flow_steps)
    log.event("start", n=len(todo), out=str(out), cfg=vars(model.cfg), versions=model.I.versions())
    inputs = Inputs(a.frames)
    f = open(out, "a", buffering=1)
    t0, cnt = time.time(), [0]
    from tqdm import tqdm
    bar = tqdm(total=len(todo) * len(variants), desc=f"alp {a.split} {i}/{n}", smoothing=0.05)

    def emit(r):
        if (r["token"], r["variant"]) in done:
            return
        f.write(json.dumps(r) + "\n")
        cnt[0] += 1
        bar.update(1)
        if cnt[0] % 500 == 0:
            rate = cnt[0] / (time.time() - t0)
            log.info(f"{cnt[0]} calls, {rate:.2f}/s")
            log.scalar("throughput/calls_per_s", rate, cnt[0])

    run_stream(model, prepared(model, inputs, todo, variants, a.workers), variants, a.batch, emit, log)
    bar.close()
    log.event("end_run", n=cnt[0], seconds=time.time() - t0)
    log.info(f"done {cnt[0]} calls in {time.time() - t0:.0f} s")


def cmd_bench(a, log):
    """Throughput per batch size on the same tokens (fresh buckets per row); one warm-up pass first."""
    idx = Z.load_index(a.split)
    rng = np.random.default_rng(0)
    entries = [idx[k] for k in rng.choice(len(idx), a.n, replace=False)]
    model = Model(a.attn, a.compile.split(",") if a.compile else (), a.flow_steps)
    inputs = Inputs(a.frames)
    t0 = time.time()
    samples = list(prepared(model, inputs, entries, ("nav",), a.workers))
    prep_s = time.time() - t0
    log.info(f"prepared {len(samples)} samples in {prep_s:.1f} s ({1e3 * prep_s / len(samples):.0f} ms/sample wall, "
             f"{a.workers} threads)")
    import torch
    rows = []
    for B in [int(b) for b in a.batches.split(",")]:
        for rep in range(2):  # rep 0 = warm-up (compile for this batch shape)
            recs = []
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            run_stream(model, iter(samples), ("nav",), B, recs.append)
            wall = time.time() - t0
            if rep:
                rows.append({"B": B, "n": len(recs), "wall_s": wall, "ms_per_sample": 1e3 * wall / len(recs),
                             "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
                log.info(json.dumps(rows[-1]))
                np.save(log.dir / f"poses_B{B}.npy", np.array([r["poses"] for r in recs]))
    import pandas as pd
    pd.DataFrame(rows).to_csv(log.dir / "bench.csv", index=False)
    json.dump({"prep_ms_per_sample": 1e3 * prep_s / len(samples), "workers": a.workers, "rows": rows},
              open(log.dir / "bench.json", "w"), indent=1)


def cmd_viz(a, log):
    """Save the 4 model images (current frame) and the egomotion of a few tokens as PNG/npz for inspection."""
    import cv2
    idx = Z.load_index(a.split)
    rng = np.random.default_rng(a.seed)
    inputs = Inputs("2hz")
    for k in rng.choice(len(idx), a.n, replace=False):
        e = idx[k]
        s = inputs(e)
        m = inputs.maps_for(e["cams"][-1])
        top = np.concatenate([s["frames"][0, 3], s["frames"][1, 3], s["frames"][2, 3]], 2)     # CL, FW, CR
        bot = np.concatenate([s["frames"][3, 0], s["frames"][3, 3], s["frames"][1, 0]], 2)     # tele t-1.5, tele t0, FW t-1.5
        img = np.concatenate([top, bot], 1).transpose(1, 2, 0)
        cv2.imwrite(str(log.dir / f"{e['token']}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        np.savez(log.dir / f"{e['token']}.npz", xyz=s["xyz"], rot=s["rot"], pose=e["pose"], vel=e["vel"], cmd=e["cmd"])
        log.info(f"{e['token']} nav={s['nav']!r} coverage={np.round(m.coverage, 3).tolist()} "
                 f"sources={[[c for c, *_ in v] for v in m.views]}")


# ---------------------------------------------------------------- adapter ablation on PhysicalAI-AV

def _nuplan_like_sources(native: dict, cams: dict, sources) -> dict:
    """Round trip: render nuPlan camera images (their pinhole + distortion + rig yaw) from the native PhysicalAI-AV
    f-theta images, at the scales AlpamayoMaps samples. native: {model cam name: (H, W, 3) RGB, 1920x1080}."""
    import cv2
    out = {}
    wide = [c for c in RIG.ALPAMAYO_CAMERAS if "tele" not in c[0]]
    for c, s in sources:
        cal = cams[c]
        w, h = Z.NUPLAN_WH[0] // s, Z.NUPLAN_WH[1] // s
        uu, vv = np.meshgrid((np.arange(w) + .5) * s - .5, (np.arange(h) + .5) * s - .5)
        K, (k1, k2, p1, p2, k3) = np.asarray(cal["K"], np.float64), np.asarray(cal["D"], np.float64)
        xd, yd = (uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]
        x, y = xd.copy(), yd.copy()
        for _ in range(30):   # fixed-point inversion of Brown-Conrady (the image lies in its monotonic region)
            r2 = x * x + y * y
            rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            x = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / rad
            y = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / rad
        und = np.stack([x, y], -1)
        rays = np.concatenate([und, np.ones((h, w, 1))], -1) @ Z.cam_to_ego(cal).T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        img = np.zeros((h, w, 3), np.uint8)
        best = np.full((h, w), -1.0)
        src = {n: im if s == 1 else cv2.resize(im, (im.shape[1] // s, im.shape[0] // s), interpolation=cv2.INTER_AREA)
               for n, im in native.items()}
        for cam in wide:
            name, _, _, (yaw, pitch), cx, cy, fw, _ = cam
            rc = rays @ Z._yaw_pitch_to_R(yaw, pitch)                     # rig -> camera axes
            th = np.arccos(np.clip(rc[..., 2], -1, 1))
            rr = np.polynomial.polynomial.polyval(th, fw)
            rho = np.maximum(np.hypot(rc[..., 0], rc[..., 1]), 1e-9)
            mx, my = cx + rr * rc[..., 0] / rho, cy + rr * rc[..., 1] / rho
            ok = (th < np.radians(62)) & (mx >= 0) & (mx <= 1919) & (my >= 0) & (my <= 1079) & (rc[..., 2] > best)
            best[ok] = rc[..., 2][ok]
            mx, my = ((mx + .5) / s - .5).astype(np.float32), ((my + .5) / s - .5).astype(np.float32)
            img[ok] = cv2.remap(src[name], mx, my, cv2.INTER_LINEAR)[ok]
        out[(c, s)] = img
    return out


def cmd_adapt(a, log):
    """Per clip and config: n=6 samples at fixed seeds; ADE / minADE over 6.4 s and over NAVSIM's 4 s."""
    import pandas as pd
    import torch
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from jevdrive.alpamayo import data as D
    avdi = D.interface()
    clips = json.loads((D.cache_dir() / "clips.json").read_text())[:a.n]
    import os
    import pickle
    base = Path(os.environ.get("OPENSCENE_DATA_ROOT", Z.data_dir() / "datasets/navsim"))
    log0 = sorted((base / "navsim_logs/test").glob("*.pkl"))[0]
    nup = Z.cams_of(pickle.load(open(log0, "rb"))[0]["cams"], base / "sensor_blobs/test")  # a real nuPlan calibration
    maps = Z.AlpamayoMaps(nup, RIG)
    model = Model(a.attn, a.compile.split(",") if a.compile else (), a.flow_steps)
    I = model.I
    cfgs = a.configs.split(",")
    rows = []
    for c in clips:
        d = load_physical_aiavdataset(c, t0_us=5_100_000, avdi=avdi, maybe_stream=False, num_frames=16)
        fr = d["image_frames"]                                               # (4, 16, 3, 1080, 1920), idx order 0,1,2,6
        gt = d["ego_future_xyz"][0, 0, :, :2].numpy()
        hx, hr = d["ego_history_xyz"][0, 0].numpy(), d["ego_history_rot"][0, 0].numpy()
        # NAVSIM-like egomotion: the 4 samples at -1.5/-1.0/-0.5/0 s with body-frame velocity, then our interpolation
        k4 = [0, 5, 10, 15]
        yaw = np.arctan2(hr[:, 1, 0], hr[:, 0, 0])
        v_w = np.gradient(hx[:, :2], 0.1, axis=0)
        v_b = np.stack([np.cos(yaw) * v_w[:, 0] + np.sin(yaw) * v_w[:, 1], -np.sin(yaw) * v_w[:, 0] + np.cos(yaw) * v_w[:, 1]], -1)
        ex, er = Z.alpamayo_history(np.c_[hx[k4, :2], yaw[k4]], v_b[k4])
        names = [cam[0] for cam in sorted(RIG.ALPAMAYO_CAMERAS, key=lambda q: q[1])]
        rt_cache = {}
        for cfg in cfgs:
            tsel = {"native": [12, 13, 14, 15], "repeat": [15] * 4, "2hz": k4}[cfg.split("+")[0]]
            frames = fr[:, tsel]
            if "rt" in cfg.split("+"):   # nuPlan camera round trip, per slot
                for t in set(tsel) - set(rt_cache):
                    native = {n: np.ascontiguousarray(fr[j, t].permute(1, 2, 0).numpy()) for j, n in enumerate(names)}
                    rt_cache[t] = maps.render(_nuplan_like_sources(native, nup, maps.sources()))
                frames = np.stack([rt_cache[t] for t in tsel], 1)
            xyz, rot = (ex, er) if "ego2hz" in cfg.split("+") else (hx, hr)
            s = {"frames": frames.numpy() if torch.is_tensor(frames) else frames, "cam_idx": d["camera_indices"].numpy()}
            tok = tokenize(model.proc, s, None)
            o = model(collate([tok], [xyz.astype(np.float32)], [rot.astype(np.float32)]), 7, n=a.seeds)
            for k, p in enumerate(o["xyz"][0, :, :, :2]):             # (n, 64, 2)
                e64 = np.linalg.norm(p - gt, axis=-1)
                rows.append({"clip": c, "config": cfg, "sample": k, "ade64": e64.mean(), "ade40": e64[:40].mean(),
                             "fde40": e64[39]})
        log.info(f"{c}: " + "  ".join(f"{g} {np.mean([r['ade40'] for r in rows if r['clip'] == c and r['config'] == g]):.2f}"
                                      for g in cfgs))
    df = pd.DataFrame(rows)
    df.to_csv(log.dir / "adapt.csv", index=False)
    per = df.groupby(["config", "clip"]).agg(ade40=("ade40", "mean"), minade40=("ade40", "min"), ade64=("ade64", "mean"))
    summ = per.groupby("config").mean()
    log.info("\n" + summ.to_string())
    summ.to_csv(log.dir / "adapt_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--attn", default="sdpa")
    common.add_argument("--compile", default="visual,expert")
    common.add_argument("--flow-steps", type=int, default=10)
    common.add_argument("--frames", default="repeat", choices=list(FRAMES))
    common.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--split", default="navtest")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--variants", default="nav,nonav")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--tag", default="main")
    r.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("bench", parents=[common])
    b.add_argument("--split", default="navtest")
    b.add_argument("--n", type=int, default=64)
    b.add_argument("--batches", default="1,4,8,16")
    v = sub.add_parser("viz", parents=[common])
    v.add_argument("--split", default="navtest")
    v.add_argument("--n", type=int, default=6)
    v.add_argument("--seed", type=int, default=0)
    ad = sub.add_parser("adapt", parents=[common])
    ad.add_argument("--n", type=int, default=31)
    ad.add_argument("--seeds", type=int, default=6)
    ad.add_argument("--configs", default="native,native+ego2hz,repeat,2hz,native+rt,repeat+rt+ego2hz,2hz+rt+ego2hz")
    a = ap.parse_args()
    log = RunLog("navsim_zs", "alpamayo_" + a.cmd)
    log.info(f"args {vars(a)} -> {log.dir}")
    {"run": cmd_run, "bench": cmd_bench, "viz": cmd_viz, "adapt": cmd_adapt}[a.cmd](a, log)
    log.event("end")
    log.close()
