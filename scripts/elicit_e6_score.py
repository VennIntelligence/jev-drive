"""Elicitation E6 (b): per-anchor PDM sub-scores on navtrain tokens with the NAVSIM v1.1 devkit's simulator and scorer
as shipped (todos/2026-09-26-elicitation-program.md, deviation [E6] 00:34 (2)).

Each token is scored once with [PDM-Closed] + K anchors. From the scorer's per-proposal arrays every anchor gets the
sub-scores `pdm_score()` would give it scored alone (paired with PDM-Closed only): NC, DAC, TTC and C do not depend on
the other proposals; EP is re-normalised against PDM-Closed alone. `check` verifies that equality on a few tokens.

Runs in envs/navsim1 (Python 3.10), OPENBLAS_CORETYPE=Haswell:
    python scripts/elicit_e6_score.py check <cache_dir> <anchors.npz>
    python scripts/elicit_e6_score.py run <cache_dir> <anchors.npz> <tokens.txt> <out_dir> [--procs 16]
Output: <out_dir>/chunk_<i>.npz with tokens (n,), sub (n, K, 5) = NC, DAC, EP, TTC, C, and pdms (n, K); resumable.
"""
import argparse
import glob
import lzma
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CHUNK = 50
_SIM = _SCORER = _ANCHORS = None


def _init(anchor_path):
    global _SIM, _SCORER, _ANCHORS
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    dk = os.environ["NAVSIM_DEVKIT_ROOT"]
    with initialize_config_dir(config_dir=dk + "/navsim/planning/script/config/pdm_scoring", version_base=None):
        cfg = compose("default_scoring_parameters")
    _SIM, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    _ANCHORS = np.load(anchor_path)["anchors"].astype(np.float64)


def score_token(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(K, 5) sub-scores and (K,) PDMS of every anchor, each as if scored alone against PDM-Closed."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as M, WeightedMetricIndex as W
    with lzma.open(path, "rb") as f:
        mc = pickle.load(f)
    ego, samp = mc.ego_state, _SIM.proposal_sampling
    st = [get_trajectory_as_array(mc.trajectory, samp, ego.time_point)]
    st += [get_trajectory_as_array(transform_trajectory(Trajectory(a), ego), samp, ego.time_point) for a in _ANCHORS]
    sims = _SIM.simulate_proposals(np.stack(st), ego)
    sc = _SCORER
    sc.score_proposals(sims, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
    nc, dac = sc._multi_metrics[M.NO_COLLISION], sc._multi_metrics[M.DRIVABLE_AREA]
    mult = sc._multi_metrics.prod(0)
    raw = sc._progress_raw * mult
    top = np.maximum(raw[0], raw)                       # pairwise with PDM-Closed (index 0) only
    thr = sc._config.progress_distance_threshold
    ep = np.where(top > thr, raw / np.where(top > 0, top, 1), (mult > 0).astype(float))
    wm = sc._weighted_metrics.copy()
    wm[W.PROGRESS] = ep
    w = sc._config.weighted_metrics_array
    pdms = mult * (wm * w[:, None]).sum(0) / w.sum()
    sub = np.stack([nc, dac, ep, wm[W.TTC], wm[W.COMFORTABLE]], 1)
    return sub[1:].astype(np.float32), pdms[1:].astype(np.float32)


def _cache_paths(cache_dir: str) -> dict:
    return {Path(p).parent.name: p for p in glob.glob(f"{cache_dir}/*/*/*/metric_cache.pkl")}


def _chunk(args):
    i, toks, paths, out = args
    dst = Path(out) / f"chunk_{i:05d}.npz"
    if dst.exists():
        return i, 0.0
    t0 = time.time()
    subs, ps, ok = [], [], []
    for t in toks:
        if t not in paths:
            continue
        s, p = score_token(paths[t])
        subs.append(s), ps.append(p), ok.append(t)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, tokens=np.array(ok), sub=np.stack(subs) if subs else np.zeros((0, len(_ANCHORS), 5)),
             pdms=np.stack(ps) if ps else np.zeros((0, len(_ANCHORS))))
    os.replace(tmp, dst)
    return i, time.time() - t0


def check(cache_dir: str, anchor_path: str, n_tok: int = 5, n_anc: int = 20):
    """Per-anchor numbers against the devkit's own pdm_score() for single trajectories."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    _init(anchor_path)
    paths = sorted(_cache_paths(cache_dir).values())[:n_tok]
    rng = np.random.default_rng(0)
    worst = 0.0
    for p in paths:
        sub, pdms = score_token(p)
        with lzma.open(p, "rb") as f:
            mc = pickle.load(f)
        for k in rng.choice(len(_ANCHORS), n_anc, replace=False):
            r = pdm_score(mc, Trajectory(_ANCHORS[k]), _SIM.proposal_sampling, _SIM, _SCORER)
            ref = np.array([r.no_at_fault_collisions, r.drivable_area_compliance, r.ego_progress,
                            r.time_to_collision_within_bound, r.comfort, r.score])
            worst = max(worst, float(np.abs(np.r_[sub[k], pdms[k]] - ref).max()))
    print(f"max |per-anchor - pdm_score()| over {len(paths)} tokens x {n_anc} anchors: {worst:.3e}")
    assert worst < 1e-6, "per-anchor scores do not reproduce pdm_score()"


def run(cache_dir, anchor_path, token_file, out, procs):
    Path(out).mkdir(parents=True, exist_ok=True)
    toks = [t.strip() for t in open(token_file) if t.strip()]
    paths = _cache_paths(cache_dir)
    print(f"{len(toks)} tokens, {sum(t in paths for t in toks)} with a metric cache", flush=True)
    jobs = [(i, toks[j:j + CHUNK], paths, out) for i, j in enumerate(range(0, len(toks), CHUNK))]
    t0, done = time.time(), 0
    with Pool(procs, initializer=_init, initargs=(anchor_path,)) as pool:
        for i, dt in pool.imap_unordered(_chunk, jobs):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} chunks, {el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run"))
    ap.add_argument("cache_dir")
    ap.add_argument("anchors")
    ap.add_argument("tokens", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.cache_dir, a.anchors)
    else:
        run(a.cache_dir, a.anchors, a.tokens, a.out, a.procs)
