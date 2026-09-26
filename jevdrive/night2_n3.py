"""Night queue 2, N3: leaderboard head x reaction (todos/2026-09-26-night-queue-2.md, N3 and the [B] entries, each
written before the numbers it affects).

  prep       Hydra seed s > 0: the CPU k-means vocabulary with seed s over the E6 navtrain rows; the scored subset
             (E6's 20 000 tokens) and its metric cache stay fixed, so only anchors.npz is new
  fit        elicit_e6's fit unchanged, plus the heads' weights and the navtrain statistics saved, so the same heads
             can be applied to any feature matrix (P5 / I3 zero-shot)

    python -m jevdrive.night2_n3 prep --seed 1
    python -m jevdrive.night2_n3 fit <prep dir> <score dir> --seed 1 --model cinque
"""
import json
from pathlib import Path

import numpy as np
import torch

from . import elicit_e6 as E6, navsim_heads as H, traj
from .common import data_dir, get_logger

log = get_logger(__name__)
E6_PREP = "runs/elicitation/e6-prep/20260926-003758"


def prep(rl, seed: int):
    """E6.prep's vocabulary step with k-means seed `seed` (CPU, deterministic); tokens and cache are E6's."""
    tr = E6._navtrain(False)
    fut = tr["fut"]
    F = torch.as_tensor(fut[..., :2].reshape(len(fut), -1))
    A = traj.kmeans(F, H.K, seed=seed)
    ids = traj.nearest(F, A, 1)[0][:, 0]
    s_, c_ = np.zeros((H.K, 8)), np.zeros((H.K, 8))
    np.add.at(s_, ids, np.sin(fut[..., 2]))
    np.add.at(c_, ids, np.cos(fut[..., 2]))
    anchors = np.concatenate([A.reshape(H.K, 8, 2).numpy(), np.arctan2(s_, c_)[..., None]], -1).astype(np.float32)
    np.savez(rl.dir / "anchors.npz", anchors=anchors, ids=ids)
    src = data_dir() / E6_PREP
    (rl.dir / "tokens.txt").write_text((src / "tokens.txt").read_text())
    (rl.dir / "seed.json").write_text(json.dumps({"kmeans_seed": seed, "holdout_seed": seed + 1, "subset": str(src)}))
    rl.log.info(f"seed {seed}: vocabulary K={H.K}, oracle (x, y) ADE "
                f"{np.linalg.norm(A.numpy()[ids].reshape(-1, 8, 2) - fut[..., :2], axis=-1).mean():.3f} m -> {rl.dir}")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep",))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rl = RunLog("night2", f"n3-{a.step}-s{a.seed}")
    if a.step == "prep":
        prep(rl, a.seed)
    rl.close()


if __name__ == "__main__":
    main()
