#!/usr/bin/env python
"""Skill pack N1 (fc65452:todos/2026-09-29-n1-scorer.md): per-candidate PDM sub-scores of the native-plan candidates on the
20 000-token navtrain sub-score set, with the NAVSIM v1.1 devkit's simulator and scorer as shipped, on E6's v1_e6sub
metric cache. Same per-candidate definition as experiments/elicitation/archive/elicit_e6_score.py (each candidate scored as if alone against
PDM-Closed): NC, DAC, EP, TTC, C and PDMS.

Runs in envs/navsim1, OPENBLAS_CORETYPE=Haswell:
    python experiments/skill_pack/archive/n1_score_native.py check <cache_dir> <cands.npz>
    python experiments/skill_pack/archive/n1_score_native.py run <cache_dir> <cands.npz> <out_dir> [--procs 8]
cands.npz: tokens (n,), cands (n, S, 8, 3). Output <out_dir>/chunk_<i>.npz: tokens, sub (n, S, 5), pdms (n, S); resumable.
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
_SIM = _SCORER = None


def _init():
    global _SIM, _SCORER
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    dk = os.environ["NAVSIM_DEVKIT_ROOT"]
    with initialize_config_dir(config_dir=dk + "/navsim/planning/script/config/pdm_scoring", version_base=None):
        cfg = compose("default_scoring_parameters")
    _SIM, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)


def score_token(path: str, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, 5) sub-scores and (S,) PDMS of the S candidate trajectories (S, 8, 3), each as if scored alone against PDM-Closed."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as M, WeightedMetricIndex as W
    with lzma.open(path, "rb") as f:
        mc = pickle.load(f)
    ego, samp = mc.ego_state, _SIM.proposal_sampling
    st = [get_trajectory_as_array(mc.trajectory, samp, ego.time_point)]
    st += [get_trajectory_as_array(transform_trajectory(Trajectory(c.astype(np.float64)), ego), samp, ego.time_point) for c in cands]
    sims = _SIM.simulate_proposals(np.stack(st), ego)
    sc = _SCORER
    sc.score_proposals(sims, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
    nc, dac = sc._multi_metrics[M.NO_COLLISION], sc._multi_metrics[M.DRIVABLE_AREA]
    mult = sc._multi_metrics.prod(0)
    raw = sc._progress_raw * mult
    top = np.maximum(raw[0], raw)
    thr = sc._config.progress_distance_threshold
    ep = np.where(top > thr, raw / np.where(top > 0, top, 1), (mult > 0).astype(float))
    wm = sc._weighted_metrics.copy()
    wm[W.PROGRESS] = ep
    w = sc._config.weighted_metrics_array
    pdms = mult * (wm * w[:, None]).sum(0) / w.sum()
    sub = np.stack([nc, dac, ep, wm[W.TTC], wm[W.COMFORTABLE]], 1)
    return sub[1:].astype(np.float32), pdms[1:].astype(np.float32)


def _paths(cache_dir: str) -> dict:
    return {Path(p).parent.name: p for p in glob.glob(f"{cache_dir}/*/*/*/metric_cache.pkl")}


def _chunk(args):
    i, toks, cands, paths, out = args
    dst = Path(out) / f"chunk_{i:05d}.npz"
    if dst.exists():
        return i
    subs, ps, ok = [], [], []
    for t, c in zip(toks, cands):
        if t in paths:
            s, p = score_token(paths[t], c)
            subs.append(s), ps.append(p), ok.append(t)
    S = cands.shape[1]
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, tokens=np.array(ok), sub=np.stack(subs) if subs else np.zeros((0, S, 5), np.float32),
             pdms=np.stack(ps) if ps else np.zeros((0, S), np.float32))
    os.replace(tmp, dst)
    return i


def check(cache_dir, cands_path, n_tok=5):
    """Against the devkit's own pdm_score() for single trajectories."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    _init()
    z = np.load(cands_path)
    paths = _paths(cache_dir)
    worst = 0.0
    for t, c in list(zip(z["tokens"], z["cands"]))[:n_tok]:
        sub, pdms = score_token(paths[t], c)
        with lzma.open(paths[t], "rb") as f:
            mc = pickle.load(f)
        for k in range(len(c)):
            r = pdm_score(mc, Trajectory(c[k].astype(np.float64)), _SIM.proposal_sampling, _SIM, _SCORER)
            ref = np.array([r.no_at_fault_collisions, r.drivable_area_compliance, r.ego_progress,
                            r.time_to_collision_within_bound, r.comfort, r.score])
            worst = max(worst, float(np.abs(np.r_[sub[k], pdms[k]] - ref).max()))
    print(f"max |per-candidate - pdm_score()| over {n_tok} tokens: {worst:.3e}")
    assert worst < 1e-6


def run(cache_dir, cands_path, out, procs):
    Path(out).mkdir(parents=True, exist_ok=True)
    z = np.load(cands_path)
    toks, cands = z["tokens"], z["cands"]
    paths = _paths(cache_dir)
    print(f"{len(toks)} tokens, {sum(t in paths for t in toks)} with a metric cache", flush=True)
    jobs = [(i, toks[j:j + CHUNK], cands[j:j + CHUNK], paths, out) for i, j in enumerate(range(0, len(toks), CHUNK))]
    t0, done = time.time(), 0
    with Pool(procs, initializer=_init) as pool:
        for _ in pool.imap_unordered(_chunk, jobs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} chunks, {el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run"))
    ap.add_argument("cache_dir")
    ap.add_argument("cands")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    check(a.cache_dir, a.cands) if a.cmd == "check" else run(a.cache_dir, a.cands, a.out, a.procs)
