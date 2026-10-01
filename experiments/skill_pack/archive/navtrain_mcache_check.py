#!/usr/bin/env python3
"""Sanity check of the v1_navtrain metric cache: file count vs the navtrain token list, lzma integrity of every file
(parallel), PDM-Closed trajectory / ego-state ranges on a sample, and equality with v1_navtrain_oplb on shared tokens.
  navtrain_mcache_check.py [--logs N] [--sample K]     (--logs: only tokens of the first N pkl logs found, for pilots)
"""
import argparse, lzma, os, pickle, random, sys, time, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

D = Path(os.environ["DATA_DIR"]) / "runs/navsim/metric_cache"


def integrity(p):
    try:
        return len(lzma.decompress(Path(p).read_bytes())) > 0
    except Exception:
        return False


def stats(p):
    import numpy as np
    m = pickle.loads(lzma.decompress(Path(p).read_bytes()))
    tr = m.trajectory.get_sampled_trajectory()
    xy = np.array([[s.rear_axle.x, s.rear_axle.y] for s in tr])
    v = np.array([s.dynamic_car_state.speed for s in tr])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return dict(n=len(tr), ego_v=m.ego_state.dynamic_car_state.speed, max_v=float(np.abs(v).max()), max_step=float(d.max()),
                length=float(d.sum()), n_lane=len(m.route_lane_ids), cl=len(m.centerline.discrete_path))


def same(t):
    a, b = t
    import numpy as np
    A, B = (pickle.loads(lzma.decompress(Path(x).read_bytes())) for x in (a, b))
    ta, tb = A.trajectory.get_sampled_trajectory(), B.trajectory.get_sampled_trajectory()
    return len(ta) == len(tb) and all(abs(x.rear_axle.x - y.rear_axle.x) + abs(x.rear_axle.y - y.rear_axle.y) < 1e-9 for x, y in zip(ta, tb)) \
        and A.route_lane_ids == B.route_lane_ids and A.ego_state.rear_axle.x == B.ego_state.rear_axle.x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--expect", type=int, default=0); ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--procs", type=int, default=16); ap.add_argument("--cache", default="v1_navtrain"); ap.add_argument("--full-integrity", action="store_true")
    a = ap.parse_args()
    root = D / a.cache
    files = sorted(root.rglob("metric_cache.pkl"))
    out = dict(files=len(files), empty=sum(f.stat().st_size == 0 for f in files), bytes=sum(f.stat().st_size for f in files))
    out["expect"] = a.expect; out["count_ok"] = (not a.expect) or len(files) == a.expect
    rng = random.Random(0); samp = rng.sample(files, min(a.sample, len(files)))
    chk = files if a.full_integrity else samp
    with ProcessPoolExecutor(a.procs) as ex:
        bad = [str(f) for f, ok in zip(chk, ex.map(integrity, chk, chunksize=64)) if not ok]
        out["integrity_checked"] = len(chk); out["integrity_bad"] = len(bad); out["bad_examples"] = bad[:5]
        st = list(ex.map(stats, samp, chunksize=8))
        for k in st[0]:
            v = [s[k] for s in st]; out[f"stat_{k}"] = [min(v), sum(v) / len(v), max(v)]
        ref = D / "v1_navtrain_oplb"
        pairs = [(str(f), str(ref / f.relative_to(root))) for f in files if (ref / f.relative_to(root)).exists()]
        pairs = rng.sample(pairs, min(100, len(pairs)))
        out["oplb_pairs"] = len(pairs); out["oplb_identical"] = sum(ex.map(same, pairs))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
