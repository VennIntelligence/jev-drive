#!/usr/bin/env python
"""Night queue 3, lane C, Q1: Alpamayo 1.5 as an examinee on the P6 v0 exam frames, configuration "nav, E[1 sample]"
(todos/2026-09-26-night-queue-3.md, Q1 and general rules 7 / 8). Runs in the alpamayo1.5 venv.

The model runs as it ships: I.load() (bf16, FlashAttention-2), the shipped sampling call with its defaults
(one reasoning rollout and one trajectory, top-p 0.98, temperature 0.6, <= 256 reasoning tokens, 10 flow steps),
batch 1. Only the input adapter is ours, and it reuses the WOD exam's pieces:
  images   P6's three Waymo-calibrated cameras (jevdrive.p5_openpilot.carla_calib) -> Alpamayo's four f-theta views at
           native 1920x1080 by rotation-only reprojection, uncovered pixels black (the WOD exam's GPU renderer with
           the grids cached, scripts/drive_backbones_alpamayo.Renderer, bit-identical to wod_zeroshot_alpamayo.render);
           the shipped processor resizes them. The four 10 Hz slots t-0.3 / -0.2 / -0.1 / 0 s take the latest 5 Hz
           frame at or before the slot (sample-and-hold): clip frames t-0.4, -0.2, -0.2, 0.
  history  rear-axle poses from the world's 20 Hz pose.jsonl at t-1.5 ... 0 s (every 2nd tick), in the t0 rig frame,
           z = 0, yaw-only rotations; ticks before the spawn take the spawn pose (standstill), as the P4 / P6 index.
  nav      the WOD exam's pre-registered text for P6's `intent` column (GO_STRAIGHT -> "Continue straight").
  seed     crc32("<base_id>-<seed>-<k>"): identical for every world of one case at one tick, so the paired x10 / x00 /
           null frames see the same random stream.
Outputs: $DATA_DIR/processed/carla_p6/nq3_alpamayo.npz (frame_name, grid (n, 20, 2) = the sample at 0.25 ... 5 s,
rear-axle ego frame at the frame's tick, x forward, y left, m; xyz (n, 64, 3) raw 10 Hz output; seed) and
nq3_alpamayo_cot.parquet (frame_name, cot, meta_action, answer, cot_nudge, cot_nudge_side, nav_text). Parts under
nq3_alpamayo_parts/ make the run resumable; the two files are rebuilt from them after every part.

  check   rule 8 on --n frames: inputs of the pipelined path vs a direct construction through the precedent functions
          (wod_zeroshot_alpamayo.render, I.build_inputs, the B2D agent's history), trajectory + text vs I.run with
          the same seed; profile of the direct (baseline) and pipelined paths
  run     the batch: priority 0 in full, then whole (priority, base_id, seed) units while they fit before --deadline
"""
import argparse
import io
import json
import queue
import re
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.alpamayo import infer as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.p5_openpilot import carla_calib  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

P6 = data_dir() / "processed" / "carla_p6"
OUT_NPZ, OUT_COT, PARTS = P6 / "nq3_alpamayo.npz", P6 / "nq3_alpamayo_cot.parquet", P6 / "nq3_alpamayo_parts"
CAMS = (1, 2, 3)                          # front, front_left, front_right (index `files` order, 4 clip frames each)
SLOTS = (1, 2, 2, 3)                      # clip frames (t-0.6, -0.4, -0.2, 0 s) filling the 10 Hz slots, sample-and-hold
REAR_AXLE_X = -1.388633220                # jevdrive.p4_carla.REAR_AXLE_X (MKZ actor origin -> rear axle)
HIST_TICKS = 2 * np.arange(-15, 1)        # 16 steps at 10 Hz on the 20 Hz tick grid
CFG = I.Config(name="nq3_p6")             # shipped defaults: FA2, 1 sample, 256 tokens, 10 flow steps
NUDGE = re.compile(r"\bnudg\w*\b(?:\s+(?:to\s+)?(?:the\s+)?(left|right))?", re.I)


def seed_of(base_id, seed, k) -> int:
    return zlib.crc32(f"{base_id}-{seed}-{k}".encode()) & 0x7FFFFFFF


def cot_nudge(cot: str) -> tuple[bool, str | None]:
    """Decision 47's CoT reading (behavior-layer-instruments section 2.2): the reasoning says "nudge"; plus its side."""
    m = NUDGE.search(cot or "")
    return (m is not None, (m.group(1) or "").lower() or None if m else None)


# ---------------------------------------------------------------- work list

def frames() -> pd.DataFrame:
    """Unique exam frames joined with the P6 index, each at its lowest priority, in run order."""
    f = pd.read_parquet(P6 / "nq3_exam_frames.parquet")
    f = f.groupby("frame_name", as_index=False).agg(priority=("priority", "min"))
    t = pd.read_parquet(P6 / "index.parquet", columns=["frame_name", "route_id", "frame", "k", "files", "intent",
                                                       "base_id", "seed", "world"])
    t = f.merge(t, on="frame_name", how="left", validate="1:1")
    assert t.files.notna().all(), "exam frames missing from the index"
    return t.sort_values(["priority", "base_id", "seed", "world", "k"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- inputs

class Poses:
    """Rear-axle pose per 20 Hz CARLA frame of a world (right-handed world, yaw left-positive), cached per world."""

    def __init__(self):
        self.cache, self.lock = {}, threading.Lock()

    def __call__(self, adir: Path):
        with self.lock:
            p = self.cache.get(adir)
        if p is None:
            d = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame").sort_index()
            th = -np.radians(d.yaw.to_numpy())
            ra = np.stack([d.x.to_numpy(), -d.y.to_numpy()], -1) + REAR_AXLE_X * np.stack([np.cos(th), np.sin(th)], -1)
            p = (d.index.to_numpy(), ra, np.unwrap(th))
            with self.lock:
                if len(self.cache) > 64:
                    self.cache.clear()
                self.cache[adir] = p
        return p


def history(poses, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Alpamayo egomotion at CARLA frame `frame`: xyz (16, 3) and rot (16, 3, 3) in the t0 rig frame (x forward,
    y left), oldest first; frames before the first pose take the first pose. Rounded like the B2D agent + policy
    server (x, y, yaw as float32, rotations built from those), so both constructions give the same float32 tensors."""
    f, ra, th = poses
    i0 = int(np.searchsorted(f, frame))
    assert f[i0] == frame and (np.diff(f) == 1).all(), "pose ticks missing"
    i = np.maximum(i0 + HIST_TICKS, 0)
    d = ra[i] - ra[i0]
    c, s = np.cos(th[i0]), np.sin(th[i0])
    h = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], th[i] - th[i0]], -1)
    h = h.astype(np.float32).astype(np.float64)
    xyz = np.c_[h[:, :2], np.zeros(16)]
    c, s = np.cos(h[:, 2]), np.sin(h[:, 2])
    rot = np.zeros((16, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    return xyz.astype(np.float32), rot.astype(np.float32)


def jpeg_bytes(files) -> dict:
    """{camera id: [4 slot JPEGs as uint8 tensors]} from the index's 12 paths (camera-major, oldest first)."""
    return {c: [torch.frombuffer(bytearray(Path(files[4 * j + s]).read_bytes()), dtype=torch.uint8) for s in SLOTS]
            for j, c in enumerate(CAMS)}


def calib() -> dict:
    return {c: {"intrinsic": np.asarray(v["intrinsic"], np.float64), "extrinsic": np.asarray(v["extrinsic"], np.float64),
                "width": int(v["width"]), "height": int(v["height"])} for c, v in zip(CAMS, carla_calib().values())}


def tokenize(frames: torch.Tensor, xyz, rot, nav, processor) -> dict:
    """I.build_inputs without the device move (it runs in a prep thread): the shipped message and chat template."""
    from alpamayo1_5 import helper
    msgs = helper.create_message(frames=frames.flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]), nav_text=nav)
    tok = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                        return_dict=True, return_tensors="pt")
    return {"tokenized_data": tok, "ego_history_xyz": torch.from_numpy(xyz)[None, None],
            "ego_history_rot": torch.from_numpy(rot)[None, None]}


class Prep:
    """CPU side of one frame (JPEG read, GPU decode + reprojection on a side stream, history, processor)."""

    def __init__(self, processor):
        from drive_backbones_alpamayo import Renderer
        self.processor, self.cal, self.poses = processor, calib(), Poses()
        self.rend, self.lock, self.stream = Renderer(), threading.Lock(), torch.cuda.Stream()

    def __call__(self, r) -> dict:
        t0 = time.perf_counter()
        jp = jpeg_bytes(r.files)
        t1 = time.perf_counter()
        with self.lock, torch.cuda.stream(self.stream):
            frames = self.rend("p6", self.cal, jp)                  # (4 views, 4 slots, 3, 1080, 1920) uint8 CPU
        t2 = time.perf_counter()
        xyz, rot = history(self.poses(Path(r.files[0]).parents[2]), int(r.frame))
        nav = Z.NAV_TEXT.get(int(r.intent))
        inp = tokenize(frames, xyz, rot, nav, self.processor)
        t3 = time.perf_counter()
        return {"row": r, "inputs": inp, "nav": nav, "seed": seed_of(r.base_id, r.seed, r.k),
                "t_read": t1 - t0, "t_render": t2 - t1, "t_proc": t3 - t2}


def to_cuda(inp: dict) -> dict:
    from alpamayo1_5 import helper
    return helper.to_device(inp, "cuda")


def infer(model, inputs: dict, seed: int) -> dict:
    """I.run's call (same seeding, autocast and arguments), keeping every text field the model emits."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        t0 = time.perf_counter()
        xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs, top_p=CFG.top_p, temperature=CFG.temperature, num_traj_samples=CFG.n_samples,
            num_traj_sets=CFG.n_sets, max_generation_length=CFG.max_gen, return_extra=True,
            diffusion_kwargs={"inference_step": CFG.flow_steps})
        torch.cuda.synchronize()
    txt = {k: str(np.asarray(v).ravel()[0]) for k, v in extra.items()}
    return {"xyz": xyz[0, 0, 0].float().cpu().numpy(), "wall": time.perf_counter() - t0, **txt}


def prefetch(fn, items, workers: int, ahead: int):
    """Ordered map with bounded lookahead on a thread pool (navsim_zs_alpamayo.prefetch)."""
    with ThreadPoolExecutor(workers) as ex:
        futs, it = [], iter(items)
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


# ---------------------------------------------------------------- outputs

def record(p: dict, o: dict) -> dict:
    nudge, side = cot_nudge(o.get("cot", ""))
    r = p["row"]
    return {"frame_name": r.frame_name, "seed": p["seed"], "nav_text": p["nav"] or "", "xyz": o["xyz"],
            "cot": o.get("cot", ""), "meta_action": o.get("meta_action", ""), "answer": o.get("answer", ""),
            "cot_nudge": nudge, "cot_nudge_side": side, "wall": o["wall"]}


def write_part(recs: list, j: int):
    PARTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "xyz"} for r in recs])
    tmp = PARTS / f"part_{j:05d}.tmp.npz"
    np.savez(tmp, frame_name=df.frame_name.to_numpy().astype(str), xyz=np.stack([r["xyz"] for r in recs]),
             meta=np.array(df.to_json(orient="records")))
    tmp.replace(PARTS / f"part_{j:05d}.npz")


def read_parts():
    xs, ms = [], []
    for p in sorted(PARTS.glob("part_?????.npz")):
        z = np.load(p)
        xs.append(z["xyz"])
        ms.append(pd.read_json(io.StringIO(str(z["meta"])), orient="records", dtype=False))
    if not xs:
        return None, None
    return np.concatenate(xs), pd.concat(ms, ignore_index=True)


def consolidate() -> int:
    xyz, meta = read_parts()
    if xyz is None:
        return 0
    keep = ~meta.frame_name.duplicated().to_numpy()
    xyz, meta = xyz[keep], meta[keep].reset_index(drop=True)
    grid = Z.resample(Z.ALP_T, xyz[..., :2]).astype(np.float32)               # (n, 20, 2) at 0.25 ... 5 s
    tmp = OUT_NPZ.with_suffix(".tmp.npz")
    np.savez(tmp, frame_name=meta.frame_name.to_numpy().astype(str), grid=grid, xyz=xyz.astype(np.float32),
             seed=meta.seed.to_numpy())
    tmp.replace(OUT_NPZ)
    meta.drop(columns=["wall"]).to_parquet(OUT_COT.with_suffix(".tmp"), index=False)
    OUT_COT.with_suffix(".tmp").replace(OUT_COT)
    return len(meta)


# ---------------------------------------------------------------- check (rule 8) and profile

def direct_inputs(r, processor, cal) -> tuple[dict, torch.Tensor, np.ndarray, np.ndarray]:
    """One frame through the precedent functions only: an in-memory WOD-exam package -> wod_zeroshot_alpamayo.render,
    the B2D agent's _history on the raw 20 Hz poses + the policy server's rotation build, I.build_inputs."""
    import wod_zeroshot_alpamayo as WA
    z = {f"jpg_{k}_{c}": np.frombuffer(Path(r.files[4 * j + s]).read_bytes(), np.uint8)
         for j, c in enumerate(CAMS) for k, s in enumerate(SLOTS)}
    for c, v in cal.items():
        z[f"cal_{c}_intrinsic"], z[f"cal_{c}_extrinsic"] = v["intrinsic"], v["extrinsic"]
        z[f"cal_{c}_wh"] = np.array([v["width"], v["height"]])
    src, Z.ALP_SRC = Z.ALP_SRC, CAMS
    try:
        frames = WA.render(z)
    finally:
        Z.ALP_SRC = src
    # scripts/b2d_zeroshot_agent.py ZeroShotAgent._history, verbatim on (t, rear-axle xy, CARLA yaw) of pose.jsonl
    adir = Path(r.files[0]).parents[2]
    d = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame").sort_index()
    yaw_c = np.radians(d.yaw.to_numpy())
    xy = np.stack([d.x.to_numpy(), d.y.to_numpy()], -1) + REAR_AXLE_X * np.stack([np.cos(yaw_c), np.sin(yaw_c)], -1)
    t = d.t.to_numpy()
    t0 = float(d.t.loc[int(r.frame)])
    yaw = np.unwrap(yaw_c)
    ts = np.clip(t0 + np.arange(-15, 1) * 0.1, t[0], t[-1])
    hx, hy, hyaw = (np.interp(ts, t, v) for v in (xy[:, 0], xy[:, 1], yaw))
    c0, s0 = np.cos(hyaw[-1]), np.sin(hyaw[-1])
    dx, dy = hx - hx[-1], hy - hy[-1]
    h = np.stack([c0 * dx + s0 * dy, s0 * dx - c0 * dy, hyaw[-1] - hyaw], -1).astype(np.float32).astype(np.float64)
    # scripts/zeroshot_policy_server.py prepare()
    xyz = np.c_[h[:, :2], np.zeros(len(h))]
    c, s = np.cos(h[:, 2]), np.sin(h[:, 2])
    rot = np.zeros((len(h), 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    data = {"image_frames": frames, "camera_indices": torch.tensor([0, 1, 2, 6]),
            "ego_history_xyz": torch.from_numpy(xyz).float()[None, None],
            "ego_history_rot": torch.from_numpy(rot).float()[None, None]}
    return I.build_inputs(data, processor, nav_text=Z.NAV_TEXT.get(int(r.intent))), frames, xyz, rot


def cmd_check(a, log):
    t = frames()
    rng = np.random.default_rng(0)
    sub = t.iloc[np.sort(rng.choice(len(t), a.n, replace=False))].reset_index(drop=True)
    # the P6 index's 4 Hz past (p4_carla.route_rows) at -0.5 / -1.0 / -1.5 s for an independent history check
    idx = pd.read_parquet(P6 / "index.parquet", columns=["frame_name"])
    past = np.load(P6 / "past.npy", mmap_mode="r")
    prow = pd.Series(np.arange(len(idx)), index=idx.frame_name)
    model, processor = I.load(CFG.attn)
    I.apply(model, CFG)
    log.event("start", n=a.n, cfg=vars(CFG), versions=I.versions())
    prep, cal = Prep(processor), calib()
    timer = I.StageTimer(model)
    rows, base_t, fast = [], [], []
    # warm-up (kernels, allocator) outside every timing
    w = prep(sub.iloc[0])
    infer(model, to_cuda(w["inputs"]), 0)
    # 1) baseline: the direct path, sequential, per stage
    ref = {}
    for r in sub.itertuples():
        t0 = time.perf_counter()
        inp, fr, hxyz, _ = direct_inputs(r, processor, cal)
        t1 = time.perf_counter()
        o = I.run(model, inp, CFG, timer, seed=seed_of(r.base_id, r.seed, r.k))
        ref[r.frame_name] = (inp, hxyz, o)
        base_t.append({"prep": t1 - t0, **{k: o[k] for k in ("wall", "vision", "prefill", "decode", "flow", "other",
                                                                 "n_decode")}})
    # 2) the pipelined path: prep threads overlap the GPU, batch 1, same seeds
    t0 = time.perf_counter()
    for p in prefetch(prep, list(sub.itertuples()), a.workers, 4 * a.workers):
        o = infer(model, to_cuda(p["inputs"]), p["seed"])
        fast.append((p, o))
    wall_fast = time.perf_counter() - t0
    for p, o in fast:
        r = p["row"]
        inp, hxyz, ro = ref[r.frame_name]
        a_td, b_td = p["inputs"]["tokenized_data"], inp["tokenized_data"]
        tok_same = all(torch.equal(a_td[k], b_td[k].cpu()) for k in b_td.keys())
        hist_d = float(max((p["inputs"]["ego_history_xyz"] - inp["ego_history_xyz"].cpu()).abs().max(),
                           (p["inputs"]["ego_history_rot"] - inp["ego_history_rot"].cpu()).abs().max()))
        pi = int(prow[r.frame_name])
        past_d = float(np.abs(p["inputs"]["ego_history_xyz"][0, 0, [10, 5, 0], :2].numpy() - past[pi, [13, 11, 9], :2]).max())
        traj_d = float(np.abs(o["xyz"] - ro["xyz"][0]).max())
        v = np.linalg.norm(np.diff(np.r_[[[0, 0]], o["xyz"][:, :2]], axis=0), axis=1) / 0.1
        rows.append({"frame_name": r.frame_name, "world": r.world, "tokens_identical": tok_same,
                     "history_max_abs": hist_d, "history_vs_index_past_m": past_d, "traj_max_abs_m": traj_d,
                     "cot_identical": o["cot"] == ro["cot"][0], "x5_m": float(o["xyz"][49, 0]),
                     "y5_m": float(o["xyz"][49, 1]), "v_max": float(v.max()), "cot": o["cot"],
                     "meta_action": o.get("meta_action", ""), "n_prompt": int(b_td["input_ids"].shape[1])})
    df = pd.DataFrame(rows)
    df.to_csv(log.dir / "check.csv", index=False)
    bt = pd.DataFrame(base_t)
    prof = {"n": a.n, "workers": a.workers,
            "baseline_s_per_frame": float(bt.prep.mean() + bt.wall.mean()),
            "baseline_prep_s": float(bt.prep.mean()), **{f"baseline_{k}_s": float(bt[k].mean()) for k in
                                                         ("wall", "vision", "prefill", "decode", "flow", "other")},
            "decode_tokens_mean": float(bt.n_decode.mean()),
            "pipelined_s_per_frame": wall_fast / a.n,
            "pipelined_gpu_s": float(np.mean([o["wall"] for _, o in fast])),
            **{f"prep_{k}_s": float(np.mean([p[f"t_{k}"] for p, _ in fast])) for k in ("read", "render", "proc")},
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    json.dump(prof, open(log.dir / "profile.json", "w"), indent=1)
    ok = bool(df.tokens_identical.all() and (df.history_max_abs < 1e-5).all() and (df.traj_max_abs_m <= 1e-3).all()
              and df.cot_identical.all())
    summ = {"equivalent": ok, "tokens_identical": int(df.tokens_identical.sum()), "cot_identical": int(df.cot_identical.sum()),
            "traj_max_abs_m": float(df.traj_max_abs_m.max()), "history_max_abs": float(df.history_max_abs.max()),
            "history_vs_index_past_m": float(df.history_vs_index_past_m.max()),
            "x5_range": [float(df.x5_m.min()), float(df.x5_m.max())], "v_max": float(df.v_max.max()),
            "cot_nudge": int(sum(cot_nudge(c)[0] for c in df.cot))}
    log.info("profile " + json.dumps(prof))
    log.info("check " + json.dumps(summ))
    log.info("\n" + df[["frame_name", "world", "traj_max_abs_m", "x5_m", "y5_m", "v_max", "cot"]].head(12).to_string())
    log.event("check", **summ, profile=prof)
    if not ok:
        raise SystemExit("rule 8 check failed")


def cmd_bench(a, log):
    """Exactness-preserving GPU knobs (I.Config: the repo's expert CUDA graphs, static KV cache) against the shipped
    default on the same frames and seeds, configs interleaved per frame (the card is shared, so its load drifts);
    batch B > 1 for reference (a batched sampler draws differently, so it is never bit-identical)."""
    import copy
    t = frames()
    sub = t.iloc[np.sort(np.random.default_rng(1).choice(len(t), a.n, replace=False))].reset_index(drop=True)
    model, processor = I.load(CFG.attn)
    prep = Prep(processor)
    items = list(prefetch(prep, list(sub.itertuples()), a.workers, 4 * a.workers))
    names = a.configs.split(",")
    cfgs = {"default": CFG, "graph": I.Config(name="graph", expert_graph=True),
            "static": I.Config(name="static", static_cache=True),
            "graph_static": I.Config(name="graph_static", expert_graph=True, static_cache=True)}
    I.apply(model, CFG)
    infer(model, to_cuda(items[0]["inputs"]), 0)
    res = {c: [] for c in names}
    for p in items:
        for c in names:
            I.apply(model, cfgs[c])
            if c != "default":
                infer(model, to_cuda(copy.deepcopy(p["inputs"])), 12345)       # warm-up / capture at this shape
            res[c].append(infer(model, to_cuda(copy.deepcopy(p["inputs"])), p["seed"]))
    I.apply(model, CFG)
    rows = []
    for c in names:
        d = [float(np.abs(o["xyz"] - r["xyz"]).max()) for o, r in zip(res[c], res["default"])]
        same = [o["cot"] == r["cot"] for o, r in zip(res[c], res["default"])]
        rows.append({"config": c, "s_per_frame": float(np.mean([o["wall"] for o in res[c]])),
                     "traj_max_abs_m": max(d), "cot_identical": int(sum(same)), "n": len(items)})
    for B in [int(b) for b in a.batches.split(",") if b]:
        for k in range(0, len(items) - B + 1, B):
            grp = items[k:k + B]
            if k == 0:
                infer_b(model, to_cuda(stack(grp)), 0)
            o = infer_b(model, to_cuda(stack(grp)), grp[0]["seed"])
            rows.append({"config": f"batch{B}", "s_per_frame": o[0]["wall"], "n": B,
                         "traj_max_abs_m": float(max(np.abs(o[i]["xyz"] - res["default"][k + i]["xyz"]).max()
                                                     for i in range(B))),
                         "cot_identical": int(sum(o[i]["cot"] == res["default"][k + i]["cot"] for i in range(B)))})
            log.info(json.dumps(rows[-1]))
    df = pd.DataFrame(rows)
    df.to_csv(log.dir / "bench.csv", index=False)
    log.info("\n" + df.to_string())
    log.event("bench", rows=rows)


def stack(items: list) -> dict:
    """Same-length prompts (one nav text, same images count) into one batch, as navsim_zs_alpamayo.collate."""
    tok = [g["inputs"]["tokenized_data"] for g in items]
    return {"tokenized_data": {k: torch.cat([x[k] for x in tok]) for k in tok[0].keys()},
            "ego_history_xyz": torch.cat([g["inputs"]["ego_history_xyz"] for g in items]),
            "ego_history_rot": torch.cat([g["inputs"]["ego_history_rot"] for g in items])}


def infer_b(model, inputs: dict, seed: int) -> list:
    """A batch of B frames in one shipped call (per-row draws differ from batch 1: never bit-identical)."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        t0 = time.perf_counter()
        xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs, top_p=CFG.top_p, temperature=CFG.temperature, num_traj_samples=1, num_traj_sets=1,
            max_generation_length=CFG.max_gen, return_extra=True, diffusion_kwargs={"inference_step": CFG.flow_steps})
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / len(xyz)
    txt = {k: [str(c) for c in np.asarray(v).reshape(-1)] for k, v in extra.items()}
    xyz = xyz[:, 0, 0].float().cpu().numpy()
    return [{"xyz": xyz[i], "wall": wall, **{k: v[i] for k, v in txt.items()}} for i in range(len(xyz))]

# ---------------------------------------------------------------- run

def cmd_run(a, log):
    t = frames()
    PARTS.mkdir(parents=True, exist_ok=True)
    _, meta = read_parts()
    done = set() if meta is None else set(meta.frame_name)
    j0 = len(list(PARTS.glob("part_?????.npz")))
    units = [(k, g) for k, g in t.groupby(["priority", "base_id", "seed"], sort=False)]
    todo = [(k, g[~g.frame_name.isin(done)]) for k, g in units]
    todo = [(k, g) for k, g in todo if len(g)]
    deadline = datetime.combine(datetime.now().date(), datetime.strptime(a.deadline, "%H:%M").time()).timestamp()
    n_todo = sum(len(g) for _, g in todo)
    log.info(f"{len(t)} exam frames, {len(done)} done, {n_todo} to do in {len(todo)} units; deadline {a.deadline} "
             f"(priority >= 1 units start only if they fit)")
    model, processor = I.load(CFG.attn)
    cfg = I.Config(name="nq3_p6_graph", expert_graph=True)     # bench: bit-identical to the shipped default
    I.apply(model, cfg)
    log.event("start", batch=a.batch, todo=n_todo, done=len(done), deadline=a.deadline, cfg=vars(cfg), versions=I.versions())
    prep = Prep(processor)
    t_start, n, gpu_s, recs, j = time.time(), 0, 0.0, [], j0
    rate = a.est_s                                                # s/frame, measured as the run goes
    stopped, ui = None, 0
    from tqdm import tqdm
    bar = tqdm(total=n_todo, desc="alpamayo p6", smoothing=0.05)
    while ui < len(todo) and stopped is None:
        # one part = whole units up to >= a.part_frames frames; a priority >= 1 unit joins only if it ends before
        # the deadline at the measured rate, and the first one that does not ends the run (units are in priority order)
        part, nf = [], 0
        while ui < len(todo) and nf < a.part_frames:
            (pri, _, _), g = todo[ui]
            if pri > 0 and time.time() + (nf + len(g)) * rate > deadline:
                stopped = todo[ui][0]
                break
            part.append(todo[ui])
            nf += len(g)
            ui += 1
        if not part:
            break
        rows, t_part = [r for _, g in part for r in g.itertuples()], time.time()
        buf = []
        for p in prefetch(prep, rows, a.workers, 4 * a.workers):
            buf.append(p)
            if len(buf) < a.batch and len(buf) + len(recs) < len(rows):
                continue
            if a.batch == 1:
                outs = [infer(model, to_cuda(buf[0]["inputs"]), buf[0]["seed"])]
            else:
                outs = infer_b(model, to_cuda(stack(buf)), buf[0]["seed"])
            recs += [dict(record(q, o), batch=a.batch) for q, o in zip(buf, outs)]
            gpu_s += sum(o["wall"] for o in outs)
            n += len(buf)
            bar.update(len(buf))
            buf = []
        write_part(recs, j)
        j, recs = j + 1, []
        tot = consolidate()
        rate = (time.time() - t_part) / nf                        # the last part's rate decides the next units
        log.info(f"{n}/{n_todo} frames this run ({tot} in the outputs), {rate:.2f} s/frame wall (last part), "
                 f"{gpu_s / n:.2f} s/frame GPU, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
                 f"ETA all {(n_todo - n) * rate / 60:.0f} min")
        log.scalar("s_per_frame", rate, n)
        log.event("part", part=j - 1, frames=n, total=tot, s_per_frame=rate, gpu_s_per_frame=gpu_s / n)
    bar.close()
    tot = consolidate()
    cov = frames().assign(done=lambda d: d.frame_name.isin(set(np.load(OUT_NPZ)["frame_name"]) if OUT_NPZ.exists() else set()))
    by_p = {int(k): [int(v.done.sum()), int(len(v))] for k, v in cov.groupby("priority")}
    log.info(f"covered per priority [done, all]: {by_p}; stopped before unit {stopped}")
    summary = {"wall_s": time.time() - t_start, "frames_this_run": n, "frames_total": tot, "per_priority": by_p,
               "stopped_before_unit": None if stopped is None else [int(stopped[0]), str(stopped[1]), int(stopped[2])],
               "s_per_frame": (time.time() - t_start) / max(n, 1), "gpu_s_per_frame": gpu_s / max(n, 1),
               "outputs": [str(OUT_NPZ), str(OUT_COT)]}
    json.dump(summary, open(log.dir / "summary.json", "w"), indent=1)
    if a.summary:
        json.dump(summary, open(a.summary, "w"), indent=1)
    log.event("end", **summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=50)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--threads", type=int, default=1, help="torch intra-op threads (the processor)")
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=8)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--configs", default="default,graph,static,graph_static")
    b.add_argument("--batches", default="4")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--threads", type=int, default=1, help="torch intra-op threads (the processor)")
    r.add_argument("--batch", type=int, default=1, help="1 = bit-identical to the shipped path (rule 8); B > 1 is "
                   "the same sampler on a batch, a different random draw per row (not bit-identical)")
    r.add_argument("--part-frames", type=int, default=300, help="frames per part file (whole units)")
    r.add_argument("--deadline", default="23:30", help="box clock; priority >= 1 units start only if they fit")
    r.add_argument("--est-s", type=float, default=1.5, help="s/frame assumed before the first part is measured")
    r.add_argument("--summary", default="", help="also write the summary JSON here")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    log = RunLog("nq3", "c", "q1_alp", a.cmd)
    log.info(f"args {vars(a)} -> {log.dir}")
    {"check": cmd_check, "bench": cmd_bench, "run": cmd_run}[a.cmd](a, log)
    log.close()
