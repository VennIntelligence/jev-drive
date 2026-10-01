"""End-to-end example of the shared libraries (docs/lib.md) on synthetic data: Run + splits + cached + pmap + stats.

    .venv/bin/python scripts/lib_demo.py [--seed 0] [--force] [--resume RUN_DIR]
"""
import argparse

import numpy as np

from jevdrive import cache, par, stats
from jevdrive.data import splits
from jevdrive.run import Run, cli_args


def score(scene: str) -> tuple:
    """One unit of work: two arms' scores on one scene (stands in for a model evaluation)."""
    r = np.random.default_rng(int(scene.split("-")[1]))
    return r.normal(0.60, 0.1), r.normal(0.55, 0.1)


def main(a):
    with Run("lib-demo", "synthetic", seed=a.seed, config=vars(a), resume=a.resume) as run:
        val = splits.load("nuscenes/val")
        run.use_split(val)
        k = cache.key(dict(seed=a.seed), code=score, version=val.id)
        res = cache.cached(run.path("scores.npy"), k, force=a.force,
                           fn=lambda: np.array(par.pmap(score, val.members, run=run, desc="scenes").values))
        r = stats.paired(res[:, 0], res[:, 1])
        stats.write_table([dict(contrast="A - B", unit="scene", **r)], run.path("results"))
        run.summary.update(delta=r["mean"], lo=r["lo"], hi=r["hi"], split=val.id)
        run.info("A - B = %s over %d scenes", stats.fmt(r), r["n"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    cli_args(ap)
    main(ap.parse_args())
