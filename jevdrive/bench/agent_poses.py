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
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Run as a script, this file's directory (jevdrive/bench, which holds a navsim.py) would shadow the fork's `navsim` package.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != _HERE]


def build(repo: Path, config_dir: str, config_name: str, overrides) -> tuple:
    """(cfg, agent): the fork's scoring config composed with its eval overrides, agent instantiated, initialized, eval mode."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    with initialize_config_dir(config_dir=str(Path(repo) / config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    agent = instantiate(cfg.agent)
    agent.initialize()
    if hasattr(agent, "eval"):
        agent.eval()
    return cfg, agent


def sensor_config(agent, overrides: dict = None):
    """agent.get_sensor_config() with field overrides (e.g. {"lidar_pc": False} for an agent that requests LiDAR but never
    reads it; PD2, recorded per model in jevdrive/bench/agents/<model>.json)."""
    sc = agent.get_sensor_config()
    for k, v in (overrides or {}).items():
        assert hasattr(sc, k), f"SensorConfig has no field {k}"
        setattr(sc, k, v)
    return sc


def scene_loader(cfg, agent, tokens, logs, split_filter=None, blobs=None, data=None, sensors: dict = None, loader_kw: dict = None):
    """SceneLoader as run_pdm_score.py builds it (the split's scene filter, restricted to these logs / tokens). split_filter /
    blobs / data override the config's filter and paths (P1 fits on navtrain with the navtest-configured agent)."""
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    sf = split_filter if split_filter is not None else (instantiate(cfg.train_test_split.scene_filter) if "train_test_split" in cfg
                                                       else instantiate(cfg.scene_filter))
    sf.log_names = list(logs)
    sf.tokens = list(tokens)
    L = SceneLoader(sensor_blobs_path=Path(blobs or cfg.sensor_blobs_path), data_path=Path(data or cfg.navsim_log_path), scene_filter=sf,
                    sensor_config=sensor_config(agent, sensors), **(loader_kw or {}))
    missing = sorted(set(tokens) - set(L.tokens))
    assert not missing, f"{len(missing)} tokens not found by the SceneLoader, e.g. {missing[:3]}"
    return L


def trajectory(agent, loader, tok) -> np.ndarray:
    """agent.compute_trajectory as run_pdm_score.py calls it -> (8, 3) float32."""
    inp = loader.get_agent_input_from_token(tok)
    if getattr(agent, "requires_scene", False):
        traj = agent.compute_trajectory(inp, loader.get_scene_from_token(tok))
    else:
        traj = agent.compute_trajectory(inp)
    p = np.asarray(traj.poses, np.float32)
    assert p.shape == (8, 3), f"{tok}: trajectory shape {p.shape}, expected (8, 3) at 0.5 .. 4 s"
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the fork's checkout (its navsim package is imported from here)")
    ap.add_argument("--config-dir", default="navsim/planning/script/config/pdm_scoring", help="relative to --repo")
    ap.add_argument("--config-name", default="default_run_pdm_score")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="torch / numpy seed (the forks' eval scripts set none)")
    ap.add_argument("--limit", type=int, default=0, help="first N tokens only (setup checks)")
    ap.add_argument("--sensors", default="{}", help="JSON SensorConfig field overrides, e.g. {\"lidar_pc\": false}")
    ap.add_argument("--loader-kw", default="{}", help="JSON extra SceneLoader kwargs as the fork's scorer passes them")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists, nothing to do", flush=True)
        return
    repo = Path(a.repo).resolve()
    rows = [ln.split("\t") for ln in Path(a.tokens).read_text().split("\n") if ln.strip()]
    if a.limit:
        rows = rows[: a.limit]
    tokens, logs = [r[0] for r in rows], sorted({r[1] for r in rows})

    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg, agent = build(repo, a.config_dir, a.config_name, a.overrides)
    loader = scene_loader(cfg, agent, tokens, logs, sensors=json.loads(a.sensors), loader_kw=json.loads(a.loader_kw))
    poses = np.full((len(tokens), 8, 3), np.nan, np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            p = trajectory(agent, loader, tok)
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
