"""factor_wm G1 rollouts with any policy, plane engine (G0a: depth not adopted), op-train env, one card + CPU warp workers.

  collect  DAgger collection on the WOD train clips: kind closed (S3: SPEC lateral + longitudinal, 8 s) or closedlat (S2: lateral only,
           logged speed, 2 s = decision 132's engine); arms free, kick +-U(0.5..3 | 1..4) deg, swerve +-U(0.2..0.8) m (launch / stay: kicks only)
  eval     held-out readouts: g0b (launch / turn / cruise) and g1s (stay) with kind closed, arms free, kick +-2 deg, swerve +-0.5 m
Output: $DATA_DIR/runs/factor_wm/roll/<tag>/<set>-<cmd>-<i>of<n>.npz (fw_g0.save layout).

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_roll.py collect --model shipped --kind closed
      --tag s3r1 --seed 1 --shard 0/3 [--limit 4]
  ... fw_roll.py eval --model <ckpt|shipped> --tag eval-S3 --sets g0b,g1s --shard 0/1
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402
import fw_g0 as G  # noqa: E402
import dg_roll as DR  # noqa: E402


def collect_specs(S, seed, kind):
    rng = np.random.default_rng([seed, 7])
    sp = []
    for c in range(S.n):
        cat = str(S.t["cat"][c])
        slow = cat in ("launch", "stay") or S.t["v"][c][C.T0] < 3.0
        lo, hi = (0.5, 3.0) if slow else (1.0, 4.0)
        arms = [("free", None)]
        for sg in (1, -1):
            arms.append((f"kick{'+' if sg > 0 else '-'}", C.kick(S.kl, sg * rng.uniform(lo, hi))))
        for sg in (1, -1):
            e = None if slow else C.swerve(S.kl, sg * rng.uniform(0.2, 0.8), S.t["v"][c])
            arms.append((f"swerve{'+' if sg > 0 else '-'}", e) if e is not None else (f"kick2{'+' if sg > 0 else '-'}", C.kick(S.kl, sg * rng.uniform(lo, hi))))
        sp += [dict(c=c, kind=kind, engine="plane", src="log", arm=a, exo=x) for a, x in arms]
    return sp


def eval_specs(S):
    sp = []
    for c in range(S.n):
        arms = [("free", None)] + [(f"kick{d:+g}", C.kick(S.kl, d)) for d in (2.0, -2.0)]
        if S.t["cat"][c] not in ("launch", "stay"):
            for o in (0.5, -0.5):
                e = C.swerve(S.kl, o, S.t["v"][c])
                if e is not None:
                    arms.append((f"swerve{o:+g}", e))
        sp += [dict(c=c, kind="closed", engine="plane", src="log", arm=a, exo=x) for a, x in arms]
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "eval"])
    ap.add_argument("--model", default="shipped")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kind", default="closed", choices=["closed", "closedlat"])
    ap.add_argument("--kl", type=int, default=0, help="steps per rollout (0 = the clip's 40; S2 uses 10)")
    ap.add_argument("--sets", default="g0b,g1s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    workers = a.workers or max(1, len(os.sched_getaffinity(0)) - 2)
    R = G.Runner(a.model, "cuda", workers)
    t0 = time.time()
    i, n = map(int, a.shard.split("/"))
    sets = ["train"] if a.cmd == "collect" else a.sets.split(",")
    for name in sets:
        S = C.Clips(name)
        sp = collect_specs(S, a.seed, a.kind) if a.cmd == "collect" else eval_specs(S)
        if a.limit:
            sp = [s for s in sp if s["c"] < a.limit]
        cs = set(sorted({s["c"] for s in sp})[i::n])
        sp = [s for s in sp if s["c"] in cs]
        out = C.root("roll", a.tag) / f"{name}-{a.cmd}{'-smoke' if a.limit else ''}-{i}of{n}.npz"
        if out.exists():
            print("exists", out)
            continue
        print(f"{a.tag} {name}: {len(sp)} rollouts, {len(cs)} clips, kl {a.kl or S.kl}, {workers} warp workers, model {a.model}", flush=True)
        G.save(out, sp, R.run(S, sp, a.batch, kl=a.kl or None), S)
        print(f"{a.tag} {name}: done at {time.time() - t0:.0f} s -> {out}", flush=True)
    DR.bye()


if __name__ == "__main__":
    main()
