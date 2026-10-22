"""Run a third-party NAVSIM agent as it ships and write its trajectories as a prediction file (tokens, poses (N, 8, 3)).

Runs inside the model's own env (envs/<agent env>, its navsim fork on PYTHONPATH); imports nothing from jevdrive, so it works
in any env that has the fork's dependencies. It does what the fork's run_pdm_score.py does before scoring: compose the fork's
own hydra scoring config with the overrides of its published eval script, `instantiate(cfg.agent)`, `agent.initialize()`, a
SceneLoader on the split's scene filter (restricted to this chunk's logs / tokens) with `agent.get_sensor_config()`, then
`agent.compute_trajectory(agent_input[, scene])` per token. Scoring is not done here: the bench scores the poses with the v1.1
devkit (PDMS, the published metric of every NAVSIM v1 paper) and the v2 devkit (EPDMS, our board) through PrecomputedAgent.

    python jevdrive/bench/agent_poses.py --repo DIR --config-name default_run_pdm_score \
        --tokens chunk.tsv --out chunk.npz [--seed 0] -- agent=diffusiondrive_agent agent.checkpoint_path=...

chunk.tsv: one `token<TAB>log` per line. Output written atomically; re-running with a finished --out exits at once.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the fork's checkout (its navsim package is imported from here)")
    ap.add_argument("--config-dir", default="navsim/planning/script/config/pdm_scoring", help="relative to --repo")
    ap.add_argument("--config-name", default="default_run_pdm_score")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="torch / numpy seed (the forks' eval scripts set none)")
    ap.add_argument("--limit", type=int, default=0, help="first N tokens only (setup checks)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists, nothing to do", flush=True)
        return
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    rows = [ln.split("\t") for ln in Path(a.tokens).read_text().split("\n") if ln.strip()]
    if a.limit:
        rows = rows[: a.limit]
    tokens, logs = [r[0] for r in rows], sorted({r[1] for r in rows})

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    with initialize_config_dir(config_dir=str(repo / a.config_dir), version_base=None):
        cfg = compose(config_name=a.config_name, overrides=list(a.overrides))
    agent = instantiate(cfg.agent)
    agent.initialize()
    if hasattr(agent, "eval"):
        agent.eval()
    sf = instantiate(cfg.train_test_split.scene_filter) if "train_test_split" in cfg else instantiate(cfg.scene_filter)
    sf.log_names = logs
    sf.tokens = tokens
    loader = SceneLoader(sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path), scene_filter=sf,
                         sensor_config=agent.get_sensor_config())
    missing = sorted(set(tokens) - set(loader.tokens))
    assert not missing, f"{len(missing)} tokens not found by the SceneLoader, e.g. {missing[:3]}"
    poses = np.full((len(tokens), 8, 3), np.nan, np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            inp = loader.get_agent_input_from_token(tok)
            if getattr(agent, "requires_scene", False):
                traj = agent.compute_trajectory(inp, loader.get_scene_from_token(tok))
            else:
                traj = agent.compute_trajectory(inp)
            p = np.asarray(traj.poses, np.float32)
            assert p.shape == (8, 3), f"{tok}: trajectory shape {p.shape}, expected (8, 3) at 0.5 .. 4 s"
            poses[i] = p
            if i % 100 == 0 or i == len(tokens) - 1:
                el = time.time() - t0
                print(f"{time.strftime('%T')} {i + 1}/{len(tokens)} tokens, {el / (i + 1):.3f} s/token", flush=True)
    assert np.isfinite(poses).all(), "non-finite poses"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
    np.savez(tmp, tokens=np.asarray(tokens), poses=poses, wall_s=time.time() - t0)
    os.replace(tmp, out)
    print(f"wrote {out}: {len(tokens)} tokens in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
