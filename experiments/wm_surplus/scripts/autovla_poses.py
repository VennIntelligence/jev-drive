"""AutoVLA (ucla-mobility/AutoVLA @ ba34eed) navtest poses for jevdrive.bench (runner `cmd`, jevdrive/bench/agents/AutoVLA.json).

Their NAVSIM eval (navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh -> run_pdm_score_cot.py) is not the
generic per-token loop: the agent's input is the per-token JSON written by their preprocessing
(scripts/run_nuplan_preprocessing.sh, INCLUDE_COT=false), not a NAVSIM AgentInput, and compute_trajectory returns
(trajectory, chain-of-thought text). This adapter does exactly what one worker of run_pdm_score_cot.py does before scoring:
compose their pdm_scoring hydra config with the eval script's overrides, `instantiate(cfg.agent)`, `agent.initialize()`, then
per token `json.load(<json_dir>/<token>.json)` -> `agent.compute_trajectory(...)`. Scoring is the bench's (v1.1 PDMS + v2 EPDMS
through PrecomputedAgent). Their trajectory is 10 poses at 0.5 s (5 s); the first 8 (0.5 .. 4.0 s, what the PDM scorer
reads) are stored.

The training config the eval script points at (config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml) is copied next to --out
with only `model.pretrained_model_path` set to --base-model (the path the README tells users to set); the repo is untouched.
Runs inside envs/autovla with cwd = their repo (codebook_cache path is repo-relative). Imports nothing from jevdrive.

    python autovla_poses.py --repo DIR --tokens chunk.tsv --out chunk.npz --json-dir DIR --base-model DIR -- <hydra overrides>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tokens", required=True, help="TSV token<TAB>log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-dir", required=True, help="their preprocessing output (one <token>.json per navtest token)")
    ap.add_argument("--base-model", required=True, help="Qwen2.5-VL-3B-Instruct dir (config model.pretrained_model_path)")
    ap.add_argument("--train-config", default="config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml", help="relative to --repo")
    ap.add_argument("--seed", type=int, default=0, help="torch / numpy seed (their eval sets none; generate() samples at T=0.01)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists():
        print(f"{out} exists, nothing to do", flush=True)
        return
    repo = Path(a.repo).resolve()
    rows = [ln.split("\t") for ln in Path(a.tokens).read_text().split("\n") if ln.strip()]
    tokens = [r[0] for r in rows][: a.limit or None]

    cfg_y = yaml.safe_load((repo / a.train_config).read_text())
    cfg_y["model"]["pretrained_model_path"] = str(Path(a.base_model).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    tcfg = out.with_name(f".{out.stem}.train_config.yaml")
    tcfg.write_text(yaml.safe_dump(cfg_y, sort_keys=False))

    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "navsim")]
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    with initialize_config_dir(config_dir=str(repo / "navsim/navsim/planning/script/config/pdm_scoring"), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[*a.overrides, f"agent.config_path={tcfg}"])
    agent = instantiate(cfg.agent)
    agent.initialize()
    print(f"{time.strftime('%T')} agent ready, {torch.cuda.max_memory_allocated() / 2**30:.2f} GB allocated", flush=True)

    poses = np.full((len(tokens), 8, 3), np.nan, np.float32)
    cot, dt = [], np.zeros(len(tokens), np.float32)
    t0 = time.time()
    for i, tok in enumerate(tokens):
        t1 = time.time()
        with open(Path(a.json_dir) / f"{tok}.json") as f:
            agent_input = json.load(f)
        traj, cot_text = agent.compute_trajectory(agent_input)
        p = np.asarray(traj.poses, np.float32)
        assert p.ndim == 2 and p.shape[0] >= 8 and p.shape[1] == 3, f"{tok}: trajectory shape {p.shape}"
        poses[i] = p[:8]
        cot.append(cot_text)
        dt[i] = time.time() - t1
        if i % 50 == 0 or i == len(tokens) - 1:
            print(f"{time.strftime('%T')} {i + 1}/{len(tokens)} tokens, {(time.time() - t0) / (i + 1):.2f} s/token, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    assert np.isfinite(poses).all(), "non-finite poses"
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
    np.savez(tmp, tokens=np.asarray(tokens), poses=poses, wall_s=time.time() - t0, token_s=dt, cot=np.asarray(cot),
             peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    os.replace(tmp, out)
    tcfg.unlink(missing_ok=True)
    print(f"wrote {out}: {len(tokens)} tokens in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
