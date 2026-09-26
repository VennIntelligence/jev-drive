#!/usr/bin/env python
"""openpilot zero-shot on NAVSIM (todos/2026-09-24-zeroshot-exam/navsim.md). Runs in envs/openpilot.

  frames  CAM_F0 of the 4 history frames -> packed road/wide model frames, cached once per split as a uint8
          memmap (N, 4, 2, 6, 128, 256) that every model and variant reads
  run     one model, one desire variant: per token, a zero-state rollout over the 1.5 s of history on the 20 Hz
          modeld clock with each 2 Hz frame held until the next one arrives (small / Cinque step 31 times,
          Lebowski steps its 8 context-rate phases, exact at the output step); the plan at t0 -> NAVSIM 8 poses

    PY=$DATA_DIR/envs/openpilot/bin/python
    $PY scripts/navsim_zs_openpilot.py frames --split navtest
    $PY scripts/navsim_zs_openpilot.py run --split navtest --model cinque --desire none

  feat    the same rollout (desire none) for several models at once, also reading the `temporal` tap at t0: the
          frozen feature of the NAVSIM thin heads (todos/2026-09-25-openpilot-openloop-comparison.md, G3). Splits with
          a `frames` cache read it; others (navtrain, ~100k tokens, a 160 GB cache) render CAM_F0 on the fly in a
          process pool. Chunks of 1000 tokens, claimed by --shard i/n, resumable:
          openpilot/<split>/feat/<model>/chunk_NNNN.npz (tokens, temporal, poses); `feat --merge` joins them.

    $PY scripts/navsim_zs_openpilot.py feat --split navtrain --models cinque lebowski --shard 0/6 --workers 3
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive.common import n_cpus  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

MODELS = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}   # backends of the openpilot smoke run
ACTION_T = (0.275, 0.525)
DT = 0.05                                                            # modeld clock
LHT_MAPS = {"sg-one-north"}                                          # left-hand traffic -> traffic_convention [0, 1]
DESIRE = {0: 1, 2: 2}                                                # command left/right -> desire turnLeft/turnRight


def frame_paths(split):
    return Z.root("openpilot", split) / "frames.npy", Z.root("openpilot", split) / "frames.json"


def cmd_frames(a, log):
    idx = Z.load_index(a.split)
    fpath, jpath = frame_paths(a.split)
    mm = np.lib.format.open_memmap(fpath, "w+", np.uint8, (len(idx), 4, 2, 6, 128, 256))
    maps, cov = {}, []

    def one(k):
        e = idx[k]
        cams = e["cams"][-1]
        key = Z.calib_key({"CAM_F0": cams["CAM_F0"]})
        m = maps.get(key) or maps.setdefault(key, Z.OpenpilotMaps(cams["CAM_F0"]))
        for f in range(4):
            mm[k, f] = m(m.decode(e["cams"][f]["CAM_F0"]["path"]))
        return min(m.coverage)

    t0 = time.time()
    from tqdm import tqdm
    with ThreadPoolExecutor(a.workers or n_cpus()) as ex:
        cov = list(tqdm(ex.map(one, range(len(idx))), total=len(idx), desc="op frames"))
    mm.flush()
    json.dump({"tokens": [e["token"] for e in idx], "min_coverage": float(min(cov))}, open(jpath, "w"))
    log.info(f"{len(idx)} tokens in {time.time() - t0:.0f} s, min model-frame coverage {min(cov):.3f}")


def schedule(context_rate: bool):
    """History frame slot (0..3, the latest 2 Hz frame at or before t) per step, oldest first; the last step is t0.
    20 Hz: t = -1.5 ... 0;
    context rate: t = -1.4, -1.2, ..., 0 (the output phase)."""
    ts = np.round(np.arange(-7, 1) * 0.2, 3) if context_rate else np.round(np.arange(-30, 1) * DT, 3)
    return [int(np.searchsorted(Z.T_HIST2, t + 1e-6) - 1) for t in ts]


def cmd_run(a, log):
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    idx = Z.load_index(a.split)
    fpath, jpath = frame_paths(a.split)
    frames = np.load(fpath, mmap_mode="r")
    assert json.load(open(jpath))["tokens"] == [e["token"] for e in idx]
    m = OPModel(a.model, MODELS[a.model], context_rate=(a.model == "lebowski"))
    sched = schedule(a.model == "lebowski")
    log.event("start", model=a.model, backend=MODELS[a.model], desire=a.desire, n=len(idx), steps=len(sched),
              action_t=ACTION_T)
    out = Z.root("openpilot", a.split) / f"{a.model}_{a.desire}.npz"
    poses = np.zeros((len(idx), 8, 3), np.float32)
    plans = np.zeros((len(idx), 33, 4), np.float32)       # x, y, z, yaw (calib frame at the camera)
    t0, tm = time.time(), 0.0
    from tqdm import tqdm
    for k in tqdm(range(len(idx)), desc=f"op {a.model} {a.desire}", smoothing=0.05):
        e = idx[k]
        fr = np.ascontiguousarray(frames[k])
        tc = (0, 1) if e["map"] in LHT_MAPS else (1, 0)
        t = time.perf_counter()
        m.reset()
        for slot in sched:
            desire = np.zeros(8, np.float32)
            if a.desire == "cmd":
                desire[DESIRE.get(int(np.argmax(e["cmd"][slot])), 0)] = 1
            raw = m.step(fr[slot], desire=desire, traffic=tc, action_t=ACTION_T)
        d = decode(raw, m.slices, float(np.linalg.norm(e["vel"][-1])), ACTION_T)
        tm += time.perf_counter() - t
        dev = e["cams"][-1]["CAM_F0"]["t"][:2]
        poses[k] = Z.openpilot_to_navsim(d["plan_pos"], d["plan_yaw"], T_IDXS, dev)
        plans[k] = np.c_[d["plan_pos"], d["plan_yaw"]]
    np.savez(out, tokens=np.array([e["token"] for e in idx]), poses=poses, plans=plans)
    s = {"n": len(idx), "wall_s": time.time() - t0, "gpu_ms_per_token": 1e3 * tm / len(idx)}
    log.event("end_run", **s)
    log.info(json.dumps(s) + f" -> {out}")


_maps = {}


def render_token(e) -> np.ndarray:
    """(4, 2, 6, 128, 256) packed model frames of one index entry, exactly as cmd_frames (calibration of the t0 frame)."""
    cams = e["cams"][-1]
    key = Z.calib_key({"CAM_F0": cams["CAM_F0"]})
    m = _maps.get(key) or _maps.setdefault(key, Z.OpenpilotMaps(cams["CAM_F0"]))
    return np.stack([m(m.decode(e["cams"][f]["CAM_F0"]["path"])) for f in range(4)])


def rollout(m, fr, sched, tc, desire=None):
    m.reset()
    for slot in sched:
        raw = m.step(fr[slot], desire=np.zeros(8, np.float32) if desire is None else desire(slot), traffic=tc,
                     action_t=ACTION_T)
    return raw


def cmd_feat(a, log):
    from concurrent.futures import ProcessPoolExecutor
    from jevdrive.drive_backbones import OP_TAPS
    idx = Z.load_index(a.split, slim=True)
    toks = np.array([e["token"] for e in idx])
    base = Z.root("openpilot", a.split, "feat_lead" if a.lead else "feat")
    if a.tokens:                  # a subset (real-data transfer G1: lead outputs on sampled navtrain rows)
        want = set(Path(a.tokens).read_text().split())
        idx = [e for e in idx if e["token"] in want]
        toks = np.array([e["token"] for e in idx])
    if a.merge:
        for mn in a.models:
            parts = [np.load(p) for p in sorted((base / mn).glob("chunk_*.npz"))]
            got = np.concatenate([q["tokens"] for q in parts])
            assert len(got) == len(toks) and (np.sort(got) == np.sort(toks)).all(), f"{mn}: {len(got)}/{len(toks)} tokens"
            extra = {k: np.concatenate([q[k] for q in parts]) for k in ("lead", "lead_prob") if a.lead}
            np.savez(Z.root("openpilot", a.split) / f"{mn}_temporal{'_lead' if a.lead else ''}.npz", tokens=got,
                     temporal=np.concatenate([q["temporal"] for q in parts]).astype(np.float16),
                     poses=np.concatenate([q["poses"] for q in parts]), **extra)
            log.info(f"{mn}: merged {len(parts)} chunks, {len(got)} tokens")
        return
    si, sn = map(int, a.shard.split("/"))
    chunks = [c for c in range(0, len(idx), a.chunk)
              if (c // a.chunk) % sn == si and not all((base / mn / f"chunk_{c // a.chunk:04d}.npz").exists() for mn in a.models)]
    chunks = chunks[: a.limit or None]
    fpath, jpath = frame_paths(a.split)
    cached = fpath.exists() and json.load(open(jpath))["tokens"] == toks.tolist()
    frames = np.load(fpath, mmap_mode="r") if cached else None
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    ex = None if cached else ProcessPoolExecutor(a.workers)
    if ex is not None:
        list(ex.map(int, range(a.workers)))        # fork the renderers before the TensorRT sessions exist
    models = {mn: OPModel(mn, MODELS[mn], context_rate=(mn == "lebowski"), taps=[OP_TAPS[mn]["temporal"]]) for mn in a.models}
    sched = {mn: schedule(mn == "lebowski") for mn in a.models}
    log.info(f"{a.split}: {len(chunks)} chunks of {a.chunk} (shard {a.shard}), frames {'cached' if cached else 'rendered'}")
    t0, n, tg = time.time(), 0, {mn: 0.0 for mn in a.models}
    for c in chunks:
        rows = range(c, min(c + a.chunk, len(idx)))
        it = (frames[k] for k in rows) if cached else ex.map(render_token, [idx[k] for k in rows], chunksize=4)
        out = {mn: {"temporal": [], "poses": [], "lead": [], "lead_prob": []} for mn in a.models}
        for k, fr in zip(rows, it):
            e = idx[k]
            fr = np.ascontiguousarray(fr)
            tc = (0, 1) if e["map"] in LHT_MAPS else (1, 0)
            dev = e["cams"][-1]["CAM_F0"]["t"][:2]
            for mn, m in models.items():
                t = time.perf_counter()
                raw = rollout(m, fr, sched[mn], tc)
                tg[mn] += time.perf_counter() - t
                d = decode(raw, m.slices, float(np.linalg.norm(e["vel"][-1])), ACTION_T)
                out[mn]["temporal"].append(m.tap_values[OP_TAPS[mn]["temporal"]].astype(np.float16))
                out[mn]["poses"].append(Z.openpilot_to_navsim(d["plan_pos"], d["plan_yaw"], T_IDXS, dev))
                if a.lead:            # raw output slices, decoded as fusion_q4c.outputs
                    out[mn]["lead"].append(raw[m.slices["lead"]].astype(np.float32))
                    out[mn]["lead_prob"].append(raw[m.slices["lead_prob"]].astype(np.float32))
            n += 1
        for mn in a.models:
            (base / mn).mkdir(parents=True, exist_ok=True)
            tmp = base / mn / f"chunk_{c // a.chunk:04d}.tmp.npz"
            extra = {k: np.stack(out[mn][k]) for k in ("lead", "lead_prob") if a.lead}
            np.savez(tmp, tokens=toks[list(rows)], temporal=np.stack(out[mn]["temporal"]),
                     poses=np.stack(out[mn]["poses"]).astype(np.float32), **extra)
            tmp.replace(base / mn / f"chunk_{c // a.chunk:04d}.npz")
        el = time.time() - t0
        log.info(f"chunk {c // a.chunk}: {n} tokens, {1e3 * el / n:.1f} ms/token wall; GPU ms/token "
                 + ", ".join(f"{mn} {1e3 * v / n:.1f}" for mn, v in tg.items()))
        log.event("progress", tokens=n, wall_s=el, **{f"gpu_s_{mn}": v for mn, v in tg.items()})
    if ex is not None:
        ex.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--split", default="navtest")
    f.add_argument("--workers", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--split", default="navtest")
    r.add_argument("--model", choices=list(MODELS), required=True)
    r.add_argument("--desire", choices=("none", "cmd"), default="none")
    f = sub.add_parser("feat")
    f.add_argument("--split", default="navtrain")
    f.add_argument("--models", nargs="+", default=["cinque", "lebowski"])
    f.add_argument("--shard", default="0/1")
    f.add_argument("--chunk", type=int, default=1000)
    f.add_argument("--workers", type=int, default=3)
    f.add_argument("--limit", type=int, default=0, help="chunks")
    f.add_argument("--merge", action="store_true")
    f.add_argument("--lead", action="store_true", help="also store the raw lead / lead_prob slices (feat_lead/)")
    f.add_argument("--tokens", default="", help="a file of tokens: run only those")
    a = ap.parse_args()
    log = RunLog("navsim_zs", "openpilot_" + a.cmd + (f"_{a.model}_{a.desire}" if a.cmd == "run" else ""))
    log.info(f"args {vars(a)} -> {log.dir}")
    {"frames": cmd_frames, "run": cmd_run, "feat": cmd_feat}[a.cmd](a, log)
    log.event("end")
    log.close()
