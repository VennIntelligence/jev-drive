"""op-adapt H: a fresh navtrain token pool on op_lb's exact frame protocol (4 keys + 6 GIMM context frames).

The pool is disjoint from op_lb's lb_navtrain (3 000 tokens; the decision-92 probe's nav samples come from it), stratified
by t0 speed with the low-speed bins over-weighted (the history-yaw failure lives there), at most 4 tokens per log.
op_lb's own prep / synth do the rendering; this script only picks the tokens and registers the data name `lb_h1train`.

  select  (envs/openpilot)  pick tokens, register splits navsim/op-adapt-h-nav-{train,dev}, write
                            runs/op_lb/lb_h1train/{meta.json,tokens.txt,keys.npy} via op_lb.cmd_prep
  synth   (envs/vfi)        GIMM context frames -> runs/op_lb/lb_h1train/gimm.npy (op_lb.cmd_synth; several workers may
                            share one card: chunks are claimed atomically)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import op_lb as B  # noqa: E402

DATA = "lb_h1train"
B.SPLITS[DATA] = "navtrain"
QUOTA = {"stop": 700, "low": 800, "mid": 600, "high": 300}     # t0 speed bins of op_common_cause (0.5 / 3 / 8 m/s)
PER_LOG = 4


def speed_bin(v):
    return np.select([v < 0.5, v < 3.0, v < 8.0], ["stop", "low", "mid"], "high")


def cmd_select(a):
    from jevdrive import navsim_zs as Z
    from jevdrive.data import splits
    idx = Z.load_index("navtrain", slim=True)
    taken = set(B.meta("lb_navtrain")["names"])
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        has_fut = set(f["tokens"].tolist())
    rng = np.random.default_rng(a.seed)
    v = np.array([float(np.linalg.norm(e["vel"][-1])) for e in idx])
    b = speed_bin(v)
    ok = np.array([e["token"] not in taken and e["token"] in has_fut for e in idx])
    logs = np.array([e["log_name"] for e in idx])
    keep, per_log = [], {}
    for name, q in QUOTA.items():
        got = 0
        for k in rng.permutation(np.flatnonzero(ok & (b == name))):
            if got >= q:
                break
            if per_log.get(logs[k], 0) >= PER_LOG:
                continue
            per_log[logs[k]] = per_log.get(logs[k], 0) + 1
            keep.append(k)
            got += 1
        print(f"{name}: {got} / {q}")
    keep = np.sort(np.array(keep))
    toks = [idx[k]["token"] for k in keep]
    ulogs = np.array(sorted(set(logs[keep])))
    dev_logs = set(np.random.default_rng(20261004).permutation(ulogs)[: int(round(0.12 * len(ulogs)))])
    tr = [t for t, k in zip(toks, keep) if logs[k] not in dev_logs]
    dv = [t for t, k in zip(toks, keep) if logs[k] in dev_logs]
    origin = (f"navtrain tokens not in runs/op_lb/lb_navtrain, stratified by t0 speed {QUOTA}, <= {PER_LOG} per log, "
              f"seed {a.seed}; dev = 12% of the logs (seed 20261004)")
    for nm, mem in (("op-adapt-h-nav-train", tr), ("op-adapt-h-nav-dev", dv)):
        sp = splits.define("navsim", nm, mem, unit="token", origin=origin, status="frozen", used_by=["op_adapt_h"],
                           notes="op_adapt_h layer-3 pilot pool (train / dev by log); never navtest / navhard")
        print(sp.id, len(mem))
    assert not set(tr) & set(dv)
    sel = set(toks)
    Z.nonav_subset = lambda idx_, per_command=0, seed=0: sel                       # op_lb.cmd_prep's navtrain draw -> this pool
    B.cmd_prep(argparse.Namespace(data=DATA, per_cmd=0, seed=0, workers=a.workers))


def cmd_synth(a):
    B.cmd_synth(argparse.Namespace(data=[DATA], method="gimm", gpu=a.gpu, vram_gb=a.vram_gb, cap_gb=a.cap_gb, chunk=32,
                                   batch=8, workers=1, limit_chunks=a.limit_chunks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=40)
    p = sp.add_parser("synth")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--vram-gb", type=float, default=14.0)
    p.add_argument("--cap-gb", type=float, default=80.0)
    p.add_argument("--limit-chunks", type=int, default=0)
    a = ap.parse_args()
    {"select": cmd_select, "synth": cmd_synth}[a.cmd](a)
