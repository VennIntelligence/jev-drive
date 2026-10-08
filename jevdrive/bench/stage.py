"""Stage entry points run inside pool jobs: `python -m jevdrive.bench.stage <name> args...` (cwd = the repo).

Each runs in the env its code needs (runner.stage_cmd picks it): parity-plans and ts-select in envs/op-train, hugsim-collect in envs/hugsim,
poses-score / poses-collect in envs/navsim2, the rest in the repo venv. Imports stay inside the functions so that every env can
import this module.
"""
from __future__ import annotations

import sys


def main(argv: list) -> None:
    name, args = argv[0], argv[1:]
    if name == "parity-plans":
        from .navsim import parity_plans
        parity_plans(*args)
    elif name == "onnx-plans":
        from .navsim import onnx_plans
        onnx_plans(*args)
    elif name == "unfreeze-plans":
        from .navsim import unfreeze_plans
        unfreeze_plans(*args)
    elif name == "ts-select":
        from .navsim import select_stage
        select_stage(*args)
    elif name == "navsim-score":
        from .navsim import score_shard
        score_shard(*args)
    elif name == "navsim-collect":
        from .navsim import collect
        collect(*args)
    elif name == "poses-score":
        from .poses import worker
        worker(*args)
    elif name == "poses-collect":
        from .poses import collect
        collect(*args)
    elif name == "hugsim-onnx":
        from .hugsim import build_onnx
        build_onnx(*args)
    elif name == "hugsim-worker":
        from .hugsim import worker
        worker(*args)
    elif name == "hugsim-collect":
        from .hugsim import collect
        collect(*args)
    elif name == "b2d-collect":
        from .b2d import collect
        collect(*args)
    else:
        raise SystemExit(f"unknown stage {name}")


if __name__ == "__main__":
    main(sys.argv[1:])
