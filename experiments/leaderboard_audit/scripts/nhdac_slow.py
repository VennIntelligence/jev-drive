"""navhard DAC attribution, exploratory check (not pre-registered): is DAC failure speed-driven?

For every official DAC failure of each arm, the same plan path is replayed (scorer LQR) at k x its own distance-time curve,
k in KS, and the geometric DAC is recorded. A failure that passes at k = 0.85 needed only a 15% slower plan on the same path.

  navsim2 env:  nhdac_slow.py [--procs 100]  -> $DATA_DIR/runs/leaderboard_audit/navhard_dac/slow.pkl
"""
import argparse
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import nhdac_feat as F  # noqa: E402

L, R = F.L, F.R
KS = [0.5, 0.7, 0.85, 1.0]


def work(job):
    token, arms = job
    W = F._W
    mc = L.load_cache(W["cp"][token])
    g = R.geoms(mc)
    out = {}
    for a in arms:
        d = L.dense_from_poses(W["P"][a][token])
        sd = F.dist_profile(d)
        out[a] = {k: bool(F.inside_of(mc, g, L.simulate(W["sim"], mc, L.poses_from_dense(F.retime(d, np.c_[np.interp(k * sd, sd, d[:, 0]), np.interp(k * sd, sd, d[:, 1]), d[:, 2]]))))[1].all())
                  for k in KS}
    return token, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=100)
    a = ap.parse_args()
    S = pickle.load(open(F.OUTD / "scores.pkl", "rb"))
    jobs = {}
    for arm in F.ARMS:
        for t in S[arm].index[S[arm].drivable_area_compliance < 1]:
            jobs.setdefault(t, []).append(arm)
    with mp.get_context("fork").Pool(a.procs, initializer=F._init) as pool:
        res = dict(pool.imap_unordered(work, list(jobs.items()), chunksize=4))
    pickle.dump(res, open(F.OUTD / "slow.pkl", "wb"))
    for arm in F.ARMS:
        v = [res[t][arm] for t in res if arm in res[t]]
        print(arm, len(v), {k: round(float(np.mean([x[k] for x in v])), 3) for k in KS}, flush=True)


if __name__ == "__main__":
    main()
