"""op_probe per-token navtest scoring of arbitrary plans with the devkit's own pdm_score: thin wrapper over jevdrive.bench.poses
(the code moved there on 2026-10-07, scores identical). In-process on this process's cores, for a job that already holds a pool
lease; to score on the whole box use the bench primitive instead (docs/bench.md, "Pose scoring"):

  .venv/bin/python -m jevdrive.bench score-poses --poses f.npz [--keys a b] [--tokens t.txt] --out o.csv --wait
  $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_score.py --poses f.npz [--keys a b] [--tokens t.txt] --out o.csv

f.npz: `tokens` (N,) and one or more (N, 8, 3) pose arrays (rear axle at t0, 0.5 .. 4 s); every key is scored on the selected tokens.
Columns: key, token, the v2 sub-scores, `score` (per-token EPDMS without EC), raw_out / raw_depth, lqr_out / out_depth (jevdrive.bench.poses).
--procs defaults to the cores granted to this process (the pool's taskset), not a constant.
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.bench import poses as PS  # noqa: E402

for _k, _v in PS.devkit_env().items():                           # before numpy / the devkit load BLAS
    os.environ.setdefault(_k, _v)
SUBS = PS.SUBS


def main():
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--tokens", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=0, help="worker processes (default: the cores granted to this process)")
    a = ap.parse_args()
    keys, toks = PS.check_inputs(a.poses, a.keys, PS.read_tokens(a.tokens) if a.tokens else None)
    procs = a.procs or PS.granted_cores()
    with Run("op_probe", f"score-{Path(a.out).stem}", config=vars(a) | {"keys": keys, "n": len(toks), "procs": procs}) as run:
        t0 = time.time()
        df, cost = PS.score_local(a.poses, keys, toks, a.out, procs, status=run.status)
        g = df.groupby("key")[["drivable_area_compliance", "score", "raw_out", "lqr_out"]].mean()
        run.summary.update(n=len(toks), keys=keys, wall_s_scoring=time.time() - t0, means=g.to_dict(), cost=cost)
        run.info(f"{len(toks)} tokens x {len(keys)} keys in {time.time() - t0:.0f} s on {procs} workers\n{g}")


if __name__ == "__main__":
    main()
