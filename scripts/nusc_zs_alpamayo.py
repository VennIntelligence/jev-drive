#!/usr/bin/env python
"""Alpamayo 1.5 zero-shot on nuScenes open-loop planning (todos/2026-09-24-zeroshot-exam/nuscenes-physicalai.md).
Runs in third_party/alpamayo1.5/.venv; model wrapper, tokenization and prompt batching from the NAVSIM runner.

Inputs per keyframe t0: 4 virtual f-theta cameras (the shared rig of scripts/zeroshot_rigs.py) rendered at the
576x320 model resolution by rotation-only reprojection of CAM_FRONT / FRONT_LEFT / FRONT_RIGHT / BACK_LEFT /
BACK_RIGHT (per pixel the camera that sees the ray closest to its optical axis; unseen = black); 4 frames per
camera at t0 - 0.3 ... t0 (each camera's frame nearest in time, ~12 Hz sweeps); egomotion 16 x 10 Hz from the ego
poses (xyz + full rotation, t0 rear-axle frame); nav text from the VAD driving command or none.

  run    --set main|quarter|half --variants nav,nonav  (one JSON line per sample and variant, resumable)
  bench  throughput vs batch size on tail keyframes (not in any evaluation set)
  viz    model images vs native nuScenes images for a few tail keyframes (adapter validation)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import navsim_zs_alpamayo as NA  # noqa: E402  (Model, tokenize, collate, prefetch)
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import navsim_zs as NZ  # noqa: E402
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.common import dataroot  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

RIG = NZ.rigs()
SRC = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
WIDE_SCALE = 2          # sources of the 120-degree views decoded at 1/2 (633 px/rad vs 278 needed at their centre)
ALP_T = np.arange(1, 65) * 0.1


class Maps:
    """Per scene calibration: for each Alpamayo camera, its source cameras and bilinear sample maps."""

    def __init__(self, scene: dict):
        (w, h), (mw, mh) = RIG.ALPAMAYO_NATIVE_WH, RIG.ALPAMAYO_MODEL_WH
        u, v = np.meshgrid((np.arange(mw) + .5) * w / mw - .5, (np.arange(mh) + .5) * h / mh - .5)
        cal = {c: Z.cam_calib(scene, c) for c in SRC}
        self.views, self.cam_idx, self.coverage = [], [], []
        for cam in RIG.ALPAMAYO_CAMERAS:
            name, idx, _, (yaw, pitch) = cam[:4]
            rays = RIG._ftheta_rays(cam, u, v) @ NZ._yaw_pitch_to_R(yaw, pitch).T
            src, U, V = G.choose_sources(np, rays, cal)
            s = 1 if "tele" in name else WIDE_SCALE
            parts = [(SRC[j], s, ((U + .5) / s - .5).astype(np.float32), ((V + .5) / s - .5).astype(np.float32), src == j)
                     for j in range(len(SRC)) if (src == j).any()]
            self.views.append(parts)
            self.cam_idx.append(idx)
            self.coverage.append(float((src >= 0).mean()))

    def render(self, imgs: dict) -> np.ndarray:
        import cv2
        out = []
        for view in self.views:
            o = np.zeros(view[0][2].shape + (3,), np.uint8)
            for c, s, mx, my, m in view:
                o[m] = cv2.remap(imgs[(c, s)], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[m]
            out.append(o.transpose(2, 0, 1))
        return np.stack(out)                                          # (4, 3, 320, 576)


class Inputs:
    def __init__(self, idx: dict):
        self.idx, self.maps = idx, {}

    def maps_for(self, name):
        if name not in self.maps:
            self.maps[name] = Maps(self.idx["scenes"][name])
        return self.maps[name]

    @staticmethod
    def decode(path, scale):
        import simplejpeg
        buf = (dataroot() / path).read_bytes()
        if scale == 1:
            return simplejpeg.decode_jpeg(buf, colorspace="RGB")
        return simplejpeg.decode_jpeg(buf, colorspace="RGB", min_width=1600 // scale, min_height=900 // scale)

    def __call__(self, e: dict) -> dict:
        sc = self.idx["scenes"][e["scene"]]
        m = self.maps_for(e["scene"])
        need = {(c, s) for v in m.views for c, s, *_ in v}
        slots = []
        for k in range(4):
            t = e["t0"] - (3 - k) * 100_000
            imgs = {(c, s): self.decode(sc["cams"][c]["path"][Z.frame_at(sc, c, t)], s) for c, s in need}
            slots.append(m.render(imgs))
        xyz, rot = Z.alpamayo_history(sc, e["t0"])
        return {"token": e["token"], "frames": np.stack(slots, 1), "cam_idx": np.asarray(m.cam_idx), "xyz": xyz,
                "rot": rot, "nav": Z.NAV_TEXT.get(e.get("cmd"))}


def run_batches(model, samples, variants, B, emit):
    buckets = {}

    def fire(key):
        items = buckets.pop(key)
        batch = NA.collate([it[1] for it in items], [it[0]["xyz"] for it in items], [it[0]["rot"] for it in items])
        o = model(batch, NA.seed_of(items[0][0]["token"]))
        for i, (s, _) in enumerate(items):
            emit({"token": s["token"], "variant": key[0], "nav_text": key[1], "B": len(items),
                  "ms": 1e3 * o["wall"] / len(items), "xyz": np.round(o["xyz"][i], 3).tolist(),
                  "yaw": np.round(Z.yaw_of(o["rot"][i]), 4).tolist(), "cot": o["cot"][i]})

    for s in samples:
        for v in variants:
            nav = s["nav"] if v == "nav" else None
            buckets.setdefault((v, nav), []).append((s, s["tok"][nav]))
            if len(buckets[(v, nav)]) >= B:
                fire((v, nav))
    for key in list(buckets):
        fire(key)


def prepared(model, inputs, entries, variants, workers):
    def one(e):
        s = inputs(e)
        s["tok"] = {n: NA.tokenize(model.proc, s, n) for n in {s["nav"] if v == "nav" else None for v in variants}}
        return s
    return NA.prefetch(one, entries, workers, 32)


def entries_of(idx, set_name):
    sets = json.loads(Z.index_path().with_name("sets.json").read_text())
    by = {e["token"]: e for e in idx["samples"]}
    return [by[t] for t in sets[set_name]]


def cmd_run(a, log):
    idx = Z.load_index()
    variants = a.variants.split(",")
    out = Z.root("alpamayo") / f"{a.set}_{'-'.join(variants)}.jsonl"
    done = set()
    if out.exists():
        done = {(r["token"], r["variant"]) for r in NA._read_lines(out, log)}
    todo = [e for e in entries_of(idx, a.set) if any((e["token"], v) not in done for v in variants)][:a.limit or None]
    log.info(f"{a.set}: {len(todo)} samples to do, variants {variants}, B {a.batch} -> {out}")
    model = NA.Model(a.attn, a.compile.split(",") if a.compile else (), a.flow_steps)
    log.event("start", n=len(todo), cfg=vars(model.cfg), versions=model.I.versions())
    f = open(out, "a", buffering=1)
    t0, cnt = time.time(), [0]
    from tqdm import tqdm
    bar = tqdm(total=len(todo) * len(variants), desc=f"alp {a.set}", smoothing=0.05)

    def emit(r):
        if (r["token"], r["variant"]) in done:
            return
        f.write(json.dumps(r) + "\n")
        cnt[0] += 1
        bar.update(1)
        if cnt[0] % 200 == 0:
            log.scalar("throughput/samples_per_s", cnt[0] / (time.time() - t0), cnt[0])
    run_batches(model, prepared(model, Inputs(idx), todo, variants, a.workers), variants, a.batch, emit)
    bar.close()
    log.event("end_run", n=cnt[0], seconds=time.time() - t0)
    log.info(f"done {cnt[0]} in {time.time() - t0:.0f} s ({(time.time() - t0) / max(cnt[0], 1):.2f} s/sample)")


def cmd_bench(a, log):
    """Throughput per batch size on tail keyframes (no 3 s future, so in no evaluation set); warm-up first."""
    import torch
    idx = Z.load_index()
    tail = [e for e in idx["samples"] if not e["valid"]]
    rng = np.random.default_rng(0)
    ents = [tail[k] for k in rng.choice(len(tail), a.n, replace=False)]
    for e in ents:
        e.setdefault("cmd", "straight")
    model = NA.Model(a.attn, a.compile.split(",") if a.compile else (), a.flow_steps)
    t0 = time.time()
    samples = list(prepared(model, Inputs(idx), ents, ("nav",), a.workers))
    log.info(f"prepared {len(samples)} in {time.time() - t0:.1f} s")
    for B in map(int, a.batches.split(",")):
        run_batches(model, iter(samples[:B * 2]), ("nav",), B, lambda r: None)       # warm-up / compile
        torch.cuda.reset_peak_memory_stats()
        t1, recs = time.time(), []
        run_batches(model, iter(samples), ("nav",), B, recs.append)
        dt = time.time() - t1
        log.info(json.dumps({"B": B, "n": len(recs), "s_per_sample": dt / len(recs),
                             "peak_gb": torch.cuda.max_memory_allocated() / 1e9}))
        with open(log.dir / f"tail_B{B}.jsonl", "w") as f:     # predictions on tail keyframes, for the BEV check
            f.writelines(json.dumps(r) + "\n" for r in recs)


def cmd_viz(a, log):
    """Adapter check on tail keyframes: the 4 model images at t0 next to the native nuScenes cameras."""
    import cv2
    idx = Z.load_index()
    tail = [e for e in idx["samples"] if not e["valid"]]
    rng = np.random.default_rng(a.seed)
    inp = Inputs(idx)
    for k in rng.choice(len(tail), a.n, replace=False):
        e = dict(tail[k], cmd="straight")
        s = inp(e)
        m = inp.maps_for(e["scene"])
        sc = idx["scenes"][e["scene"]]
        f = s["frames"][:, 3].transpose(0, 2, 3, 1)                                  # (4, 320, 576, 3) at t0
        top = np.concatenate([f[0], f[1], f[2]], 1)
        nat = {c: cv2.resize(inp.decode(sc["cams"][c]["path"][Z.frame_at(sc, c, e["t0"])], 1), (576, 324))[2:322]
               for c in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")}
        mid = np.concatenate([nat["CAM_FRONT_LEFT"], nat["CAM_FRONT"], nat["CAM_FRONT_RIGHT"]], 1)
        bot = np.concatenate([f[3], s["frames"][1, 0].transpose(1, 2, 0), np.zeros_like(f[3])], 1)
        cv2.imwrite(str(log.dir / f"{e['token']}.jpg"), np.concatenate([top, mid, bot], 0)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        np.savez(log.dir / f"{e['token']}.npz", frames=s["frames"], xyz=s["xyz"], rot=s["rot"])
        log.info(f"{e['token']} {e['scene']} {sc['location']} coverage={np.round(m.coverage, 3).tolist()} "
                 f"sources={[[c for c, *_ in v] for v in m.views]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--attn", default="sdpa")
    common.add_argument("--compile", default="visual,expert")
    common.add_argument("--flow-steps", type=int, default=10)
    common.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--set", default="main")
    r.add_argument("--variants", default="nav")
    r.add_argument("--batch", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("bench", parents=[common])
    b.add_argument("--n", type=int, default=48)
    b.add_argument("--batches", default="1,4")
    v = sub.add_parser("viz", parents=[common])
    v.add_argument("--n", type=int, default=6)
    v.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    log = RunLog("nusc_zs", "alpamayo_" + a.cmd)
    log.info(f"args {vars(a)} -> {log.dir}")
    {"run": cmd_run, "bench": cmd_bench, "viz": cmd_viz}[a.cmd](a, log)
    log.event("end")
