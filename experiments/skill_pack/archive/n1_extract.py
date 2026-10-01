#!/usr/bin/env python
"""Skill pack N1 (fc65452:todos/2026-09-29-n1-scorer.md): GIMM-interpolated openpilot Cinque features for the scorer retrain.
Thin wrapper over scripts/op_lb.py (its caches, keyframe reader, step schedule and GIMM synthesis are reused as shipped;
nothing of op_lb.py is edited, its lane's data dirs are only read).

  prep     data dir runs/op_lb/lb_n1train: E6's 20 000 navtrain sub-score tokens, in a fixed random order (the 224 first
           tokens are lane tokens, so the pilot can be checked against the lane's plans), 4 keyframes rendered  (envs/openpilot)
  synth    GIMM frames of lb_n1train (the op_lb worker, envs/vfi): --gpu 6, chunks of 32 claimed from a shared queue
  extract  Cinque rollout on the GIMM frames (op_lb's step schedule, zero state, no desire) with the `temporal` tap at t0
           -> feat/<data>/blk_NNNNN.npz (64 tokens each; tokens, temporal, plan_pos / yaw / vel / lead_prob, native poses
           via op_interp's base adapter); resumable; --shard i/K runs blocks b % K == i  (envs/openpilot)
  merge    joins the blocks of the first --n tokens into feat/<data>.npz

    PY=$DATA_DIR/envs/openpilot/bin/python; $PY experiments/skill_pack/archive/n1_extract.py prep
    $DATA_DIR/envs/vfi/bin/python experiments/skill_pack/archive/n1_extract.py synth --gpu 6 --limit-chunks 6
    $PY experiments/skill_pack/archive/n1_extract.py extract --data lb_n1train --n 192 --procs 4
    $PY experiments/skill_pack/archive/n1_extract.py merge --data lb_n1train --n 192
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/op_openloop/lib", "scripts",)]
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import op_lb as L  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DATA = "lb_n1train"
E6_TOKENS = "runs/elicitation/e6-prep/20260926-003758/tokens.txt"
LANE_TOKENS = "runs/op_lb/lb_navtrain/tokens.txt"
BLK = 64
OUT = "runs/skill_pack/n1/feat"
MODEL = "cinque"


def order() -> list[str]:
    """E6's 20 000 tokens: 224 random lane tokens first (pilot), then all the rest shuffled (stage 2 / full run)."""
    e6 = (data_dir() / E6_TOKENS).read_text().split()
    lane = set((data_dir() / LANE_TOKENS).read_text().split())
    rng = np.random.default_rng(0)
    ov = [t for t in e6 if t in lane]
    ov = [ov[i] for i in rng.permutation(len(ov))]
    head, rest = ov[:224], ov[224:] + [t for t in e6 if t not in lane]
    return head + [rest[i] for i in rng.permutation(len(rest))]


def cmd_prep(a):
    from jevdrive import navsim_zs as Z
    toks = Path(a.tokens).read_text().split() if a.tokens else order()
    assert len(toks) == len(set(toks)) and (a.tokens or len(toks) == 20000)
    full = Z.load_index("navtrain", slim=True)
    by = {e["token"]: e for e in full}
    idx = [by[t] for t in toks]
    Z.load_index = lambda split, slim=False: idx          # op_lb.cmd_prep reads the index and the subset through Z
    Z.nonav_subset = lambda idx_, per_cmd, seed: set(toks)
    L.SPLITS[a.data] = "navtrain"
    L.cmd_prep(argparse.Namespace(data=a.data, per_cmd=0, seed=0, workers=a.workers))
    mt = L.meta(a.data)
    assert mt["names"] == toks, "prep did not keep the requested order"


def cmd_synth(a):
    L.cmd_synth(argparse.Namespace(data=[a.data], method="gimm", gpu=a.gpu, vram_gb=a.vram_gb, cap_gb=a.cap_gb, chunk=32,
                                   batch=8, workers=1, limit_chunks=a.limit_chunks))


def _need(q: Path, n: int):
    miss = [c for c in range(0, n, 32) if not (q / f"{c // 32:05d}.done").exists()]
    assert not miss, f"{len(miss)} GIMM chunks of the first {n} tokens are not done (first: {miss[:3]})"


ALT = {"lcL": 3, "lcR": 4, "tL": 1, "tR": 2}     # forced desire held from ALT_T on, whatever the NAVSIM command
ALT_T = -1.0


def cmd_extract(a):
    fdir = data_dir() / OUT / (a.data + ("__alt" if a.alt else ""))
    fdir.mkdir(parents=True, exist_ok=True)
    mt = L.meta(a.data)
    n = min(a.n, len(mt["names"]))
    blocks = [b for b in range(0, n, BLK) if not (fdir / f"blk_{b // BLK:05d}.npz").exists()]
    if a.procs > 1 and not a.shard:
        argv = [sys.executable, __file__] + sys.argv[1:]
        ps = [subprocess.Popen(argv + ["--shard", f"{k}/{a.procs}"]) for k in range(a.procs)]
        for k, p in enumerate(ps):
            if p.wait() != 0:
                print(f"shard {k} failed (rc {p.returncode}); retrying once")
                assert subprocess.call(argv + ["--shard", f"{k}/{a.procs}"]) == 0, f"shard {k} failed twice"
        left = [b for b in range(0, n, BLK) if not (fdir / f"blk_{b // BLK:05d}.npz").exists()]
        assert not left, f"{len(left)} blocks missing"
        return
    from jevdrive.drive_backbones import OP_TAPS
    from jevdrive.openpilot.model import OPModel, decode
    from jevdrive.runlog import RunLog
    import op_interp as OPI
    k, K = map(int, (a.shard or "0/1").split("/"))
    mine = [b for i, b in enumerate(blocks) if i % K == k]
    if not mine:
        return
    log = RunLog("skill_pack", "n1", f"extract_{a.data}{'_alt' if a.alt else ''}_{k}of{K}")
    _need(L.root(a.data, "gimm.chunks"), n)
    keys = L.Keys(a.data)
    syn = np.load(L.root(a.data) / "gimm.npy", mmap_mode="r")
    m = OPModel(MODEL, L.BACKENDS[MODEL], context_rate=False, taps=[OP_TAPS[MODEL]["temporal"]])
    ts, src = L._steps(0.0, False)
    alt = {k: L.desire_arr(np.where(ts >= ALT_T - 1e-9, v, 0), len(ts)) for k, v in ALT.items()}
    assert len(ts) == 31 and ts[-1] == 0
    tap = OP_TAPS[MODEL]["temporal"]
    from jevdrive import navsim_zs as Z
    t_out = Z.T_OUT
    t0, tg, cnt = time.time(), 0.0, 0
    for b in mine:
        rows = range(b, min(b + BLK, n))
        o = {q: [] for q in (("temporal", "plan_pos", "plan_vel", "plan_yaw", "lead_prob", "native") if not a.alt
                             else [f"native_{k}" for k in ALT])}
        for i in rows:
            kf, sf = keys[i], np.asarray(syn[i])
            frames = [np.ascontiguousarray(kf[j] if s == "k" else sf[j]) for s, j in src]
            tc = (0, 1) if mt["lht"][i] else (1, 0)
            if a.alt:
                for k, des in alt.items():
                    t = time.perf_counter()
                    m.reset()
                    for f, dv in zip(frames, des):
                        raw = m.step(f, desire=dv, traffic=tc, action_t=L.ACTION_T)
                    tg += time.perf_counter() - t
                    d = decode(raw, m.slices, float(mt["speed"][i]), L.ACTION_T)
                    one = {q: d[q][None] for q in ("plan_pos", "plan_yaw", "plan_vel")}
                    o[f"native_{k}"].append(OPI.adapt(one, 0, {"cam": [mt["cam"][i]], "speed": [mt["speed"][i]]}, t_out,
                                                      **OPI.ADAPTERS["base"])[0])
                cnt += 1
                continue
            t = time.perf_counter()
            m.reset()
            for f in frames:
                raw = m.step(f, desire=np.zeros(8, np.float32), traffic=tc, action_t=L.ACTION_T)
            tg += time.perf_counter() - t
            d = decode(raw, m.slices, float(mt["speed"][i]), L.ACTION_T)
            o["temporal"].append(m.tap_values[tap])
            for q in ("plan_pos", "plan_vel", "plan_yaw", "lead_prob"):
                o[q].append(d[q])
            one = {q: d[q][None] for q in ("plan_pos", "plan_yaw", "plan_vel")}
            o["native"].append(OPI.adapt(one, 0, {"cam": [mt["cam"][i]], "speed": [mt["speed"][i]]}, t_out, **OPI.ADAPTERS["base"])[0])
            cnt += 1
        tmp = fdir / f"blk_{b // BLK:05d}.tmp.npz"
        np.savez(tmp, tokens=np.array(mt["names"])[list(rows)], **{q: np.stack(v) for q, v in o.items()})
        tmp.replace(fdir / f"blk_{b // BLK:05d}.npz")
        log.info(f"block {b // BLK}: {cnt} tokens here, {1e3 * (time.time() - t0) / cnt:.0f} ms/token wall, "
                 f"{1e3 * tg / cnt:.0f} ms/token in the model")
        log.event("progress", tokens=cnt, wall_s=time.time() - t0, model_s=tg)
    log.event("end")
    log.close()
    sys.stdout.flush()
    os._exit(0)                              # TensorRT / ORT teardown can hang after the outputs are written


def cmd_merge(a):
    fdir = data_dir() / OUT / (a.data + ("__alt" if a.alt else ""))
    mt = L.meta(a.data)
    n = min(a.n, len(mt["names"]))
    parts = [np.load(fdir / f"blk_{b // BLK:05d}.npz") for b in range(0, n, BLK)]
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    assert z["tokens"].tolist() == mt["names"][:n], "tokens out of order"
    tag = a.data + ("__alt" if a.alt else "")
    np.savez(data_dir() / OUT / f"{tag}_n{n}.npz", **z)
    print(f"{tag}: {n} tokens; arrays {[(k, v.shape) for k, v in z.items()]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data", default=DATA)
    p.add_argument("--tokens", default="", help="token list (fixed order) for a data dir other than lb_n1train")
    p = sp.add_parser("synth")
    p.add_argument("--data", default=DATA)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--vram-gb", type=float, default=12.0)
    p.add_argument("--cap-gb", type=float, default=78.0)
    p.add_argument("--limit-chunks", type=int, default=0)
    for name in ("extract", "merge"):
        p = sp.add_parser(name)
        p.add_argument("--data", default=DATA)
        p.add_argument("--n", type=int, required=True)
        p.add_argument("--alt", action="store_true", help="forced-desire alternative plans (ALT) instead of the none rollout")
        if name == "extract":
            p.add_argument("--procs", type=int, default=4)
            p.add_argument("--shard", default="")
    a = ap.parse_args()
    {"prep": cmd_prep, "synth": cmd_synth, "extract": cmd_extract, "merge": cmd_merge}[a.cmd](a)
