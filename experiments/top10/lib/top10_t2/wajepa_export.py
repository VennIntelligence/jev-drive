"""Resumable version of WA-JEPA's navtest trajectory export (eval/navsim_export_trajectory_cache.py, _run_worker).

Same scene filter, same log sharding, same agent (eval.navsim_agent.WorldModelNavsimAgent.compute_trajectory) and the
same output format; the only addition is that each rank checkpoints its trajectories every CKPT_EVERY tokens and
skips tokens it already has, so a killed run continues where it stopped. To change the number of ranks mid-run:
--consolidate the finished tokens into done_union.pkl, then start the new ranks with --seed done_union.pkl. The model seeds its flow sampler per call
(MultiViewCausalFutureMaskedJEPA._make_inference_generator), so a token's trajectory does not depend on call order.

Run from the WA-JEPA checkout with PYTHONPATH=<wajepa>:<navsim devkit>:
  python .../wajepa_export.py --rank R --num-shards N --out-dir D --config C --checkpoint K --navsim-root NR --openscene-root O
  python .../wajepa_export.py --merge --num-shards N --out-dir D --navsim-root NR     (writes D/navtest_trajectories.pkl)
"""
import argparse
import os
import pickle
import time
from pathlib import Path

from eval.navsim_export_trajectory_cache import _load_scene_filter, _merge_rank_pickles, _scene_filter_path, _shard_logs

CKPT_EVERY = 100


def _dump(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def export(a):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from eval.navsim_agent import WorldModelNavsimAgent

    log_names, _ = _load_scene_filter(a.navsim_root)
    scene_filter = instantiate(OmegaConf.load(_scene_filter_path(a.navsim_root)))
    scene_filter.log_names = _shard_logs(log_names, a.num_shards)[a.rank]
    part = a.out_dir / (f"rank_{a.rank:02d}.partial{'' if a.seed is None else f'.n{a.num_shards}'}"
                        f"{'' if a.split is None else f'.split{a.split[0]}of{a.split[1]}'}.pkl")
    done = pickle.load(open(part, "rb"))["trajectories"] if part.exists() else {}
    agent = WorldModelNavsimAgent(config_path=str(a.config), checkpoint_path=str(a.checkpoint), device="cuda")
    agent.initialize()
    loader = SceneLoader(original_sensor_path=a.openscene_root / "sensor_blobs/test",
                         data_path=a.openscene_root / "navsim_logs/test", scene_filter=scene_filter,
                         sensor_config=agent.get_sensor_config())
    if a.seed and a.seed.exists():            # trajectories of an earlier sharding (see --consolidate), own tokens only
        own = {str(t) for t in loader.tokens}
        done |= {k: v for k, v in pickle.load(open(a.seed, "rb"))["trajectories"].items() if k in own and k not in done}
    todo = [t for t in loader.tokens if str(t) not in done]
    if a.split is not None:                    # a helper takes every M-th remaining token of this rank (--split j M)
        todo = todo[a.split[0]::a.split[1]]
    print(f"[export] rank={a.rank} {len(loader.tokens)} tokens, {len(done)} already done, {len(todo)} to go", flush=True)
    t0 = time.time()
    for i, token in enumerate(todo, 1):
        done[str(token)] = agent.compute_trajectory(loader.get_agent_input_from_token(token)).poses.astype("float32")
        if i % CKPT_EVERY == 0 or i == len(todo):
            _dump({"trajectories": done}, part)
            print(f"[export] rank={a.rank} {len(done)}/{len(loader.tokens)} ({(time.time() - t0) / i:.2f} s/token)",
                  flush=True)
    if a.split is not None:                    # helpers only checkpoint; a final plain run with --seed writes rank_XX.pkl
        return
    own = [str(t) for t in loader.tokens]         # the partial may hold another sharding's tokens; merge wants disjoint ranks
    _dump({"trajectories": {k: done[k] for k in own}}, a.out_dir / f"rank_{a.rank:02d}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--navsim-root", type=Path, required=True)
    ap.add_argument("--openscene-root", type=Path)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=Path, help="union pickle from --consolidate; lets a run change --num-shards")
    ap.add_argument("--split", type=int, nargs=2, metavar=("J", "M"),
                    help="helper for a slow rank: only its remaining tokens j::M, checkpoint only (then --consolidate)")
    ap.add_argument("--consolidate", action="store_true",
                    help="union of every rank_*.pkl / rank_*.partial.pkl in --out-dir -> done_union.pkl")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.consolidate:
        u = {}
        for f in sorted(a.out_dir.glob("rank_*.pkl")):
            u |= pickle.load(open(f, "rb"))["trajectories"]
        _dump({"trajectories": u}, a.out_dir / "done_union.pkl")
        print(f"[export] consolidated {len(u)} tokens")
    elif a.merge:
        _, tokens = _load_scene_filter(a.navsim_root)
        _merge_rank_pickles([a.out_dir / f"rank_{r:02d}.pkl" for r in range(a.num_shards)],
                            a.out_dir / "navtest_trajectories.pkl", tokens)
    else:
        export(a)


if __name__ == "__main__":
    main()
