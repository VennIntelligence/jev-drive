"""P1 level (b) on WA-JEPA: drop the deepest V-JEPA encoder blocks / prune context tokens, unchanged planner, navtest.

Arms (plans/2026-10-22-p1b-prereg.md):
  drop<d>   skip the last d of the encoder's 24 blocks (residual blocks -> identity). The planner reads the concatenation of
            the normalised outputs of blocks 5 / 11 / 17 / 23 (hierarchical_layers), so a skipped block's slot carries the
            stream of the last kept block through that slot's own norm.
  keep<f>   keep 1/f of the 16 x 32 patch positions of each view and tubelet (2-D stride; same positions for all views):
            the context tokens and their position indices are pruned together; the planner attends over fewer tokens.
The encoder runs once per token for the off / keep arms (cached feature, lib/usage_probe.Probe.sweep with callable arms) and
once more per drop arm.

  sweep --shard I N  -> $P1_OUT/sweepb/sIofN/c*.npz ;  merge -> bench pred files PROBE-wajepa-b-<arm>, evb.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_wajepa as W  # noqa: E402
from p1_wajepa import UP  # noqa: E402

DROPS = [2, 6, 12]
KEEPS = {2: (1, 2), 4: (2, 2), 8: (2, 4)}          # 1/f of the positions: (row stride, column stride) on the 16 x 32 grid
GRID_W, PER_STEP = 32, 512
TAG = "wajepa-b"


def keep_sel(n_hist: int, f: int, device) -> torch.Tensor:
    """Indices (into the n_hist history tokens of one view: tubelet-major, then row, column) of the kept positions."""
    sh, sw = KEEPS[f]
    s = torch.arange(n_hist, device=device) % PER_STEP
    return torch.nonzero(((s // GRID_W) % sh == 0) & ((s % GRID_W) % sw == 0)).squeeze(1)


def prune_ctx(f: int, views: int):
    def fn(x):                                     # [B, V * N, D] -> [B, V * N', D]
        b, vn, d = x.shape
        sel = keep_sel(vn // views, f, x.device)
        return x.reshape(b, views, vn // views, d).index_select(2, sel).reshape(b, -1, d)
    return fn


def prune_idx(f: int):
    def fn(idx):                                   # [B, N] -> [B, N']
        assert torch.equal(idx[0], torch.arange(idx.shape[1], device=idx.device)), "history indices are not 0..N-1"
        return idx.index_select(1, keep_sel(idx.shape[1], f, idx.device))
    return fn


def cmd_sweep(a):
    from jevdrive import par
    A = W.agent()
    m = A.model
    L, toks = W.loader("navtest", A.get_sensor_config())
    mine = par.shards(toks, a.shard[1], a.shard[0])
    d = W.OUT / "sweepb" / f"s{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    blocks = m.encoder.blocks
    nb = len(blocks)
    keeps = [(f"keep{f}", prune_ctx(f, m.num_cameras)) for f in KEEPS]
    names = ["off"] + [k for k, _ in keeps] + [f"drop{x}" for x in DROPS]
    chunks = [(c0, mine[c0:c0 + a.chunk]) for c0 in range(0, len(mine), a.chunk) if not (d / f"c{c0:06d}.npz").exists()]
    todo = [t for _, c in chunks for t in c]
    t0, n = time.time(), 0
    with UP.Probe(m, W.SITE) as p, UP.Probe(m, "_history_indices_from_masks") as q:
        def on_arm(key):
            q.apply(prune_idx(int(key[4:]))) if isinstance(key, str) else q.off()
        it = W.batches(A, L, todo, W.workers())
        for c0, c in chunks:
            T, P = [], []
            for _ in c:
                t, f = next(it)
                r = p.sweep(lambda: W.plan(A, f), [None] + keeps, on_arm=on_arm)
                for x in DROPS:
                    with UP.skip_blocks(blocks, range(nb - x, nb), passthrough=lambda h, *aa, **kw: (h, None)):
                        r[f"drop{x}"] = W.plan(A, f)
                r["off"] = r.pop(None)
                T.append(t)
                P.append(np.stack([r[k] for k in names]))
            n += len(T)
            np.savez(d / "tmp.npz", tokens=np.array(T), poses=np.stack(P, 1), ks=np.array(names))
            os.replace(d / "tmp.npz", d / f"c{c0:06d}.npz")
            print(f"[sweepb] chunk {c0}: {n}/{len(todo)} to do, {(time.time() - t0) / max(n, 1):.2f} s/token", flush=True)
    (d / "DONE").write_text(json.dumps(dict(tokens=len(mine), wall_s=time.time() - t0)))


def cmd_merge(a):
    from jevdrive.bench.navsim import pred_file
    from jevdrive.bench.models import resolve
    import yaml
    toks = yaml.safe_load(open(W.TP / "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))["tokens"]
    zs = [np.load(f) for f in sorted((W.OUT / "sweepb").glob("s*of*/c*.npz"))]
    T = np.concatenate([z["tokens"] for z in zs])
    P = np.concatenate([z["poses"] for z in zs], 1)
    names = [str(x) for x in zs[0]["ks"]]
    assert len(set(T)) == len(T) == len(toks) and set(T) == set(toks), f"{len(T)} plans for {len(toks)} navtest tokens"
    a_off = np.load(pred_file(resolve(f"PROBE-{W.TAG}-off"), "navtest"))      # level (a)'s off arm: must be the same plans
    o = dict(zip(a_off["tokens"].tolist(), a_off["poses"]))
    off = P[names.index("off")]
    same = all(np.array_equal(off[j], o[t]) for j, t in enumerate(T))
    for i, nm in enumerate(names):
        if nm == "off":
            continue
        pf = pred_file(resolve(f"PROBE-{TAG}-{nm}"), "navtest")
        pf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pf, tokens=T, poses=P[i])
    ev = dict(tokens=len(T), off_identical_to_level_a_off=bool(same),
              plan_shift_vs_off_m={nm: float(np.linalg.norm(P[i][..., :2] - off[..., :2], axis=-1).mean()) for i, nm in enumerate(names)})
    (W.OUT / "evb.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))
    if not same:
        sys.exit("level (b) off arm differs from level (a) off arm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "merge"])
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    os.chdir(W.WJ)
    from jevdrive.run import Run
    from jevdrive.data import splits
    with Run("wm_surplus", f"p1b-{a.cmd}" + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd == "sweep" else ""), config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
