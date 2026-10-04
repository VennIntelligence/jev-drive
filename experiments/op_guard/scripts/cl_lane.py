"""Lane file of the closed-loop guard lines (cllib.py): HUGSIM spec scenes, B2D junction turns (zones off), B2D 19-route DS.
Run through a line script (line_hugsim.py / line_b2d_turns.py / line_b2d_ds.py), which leases, sets --root and the args:
  .venv/bin/python -m jevdrive.cl run experiments/op_guard/scripts/cl_lane.py --lane <leased lane> --root <run_dir>/cl \
      --arg candidate=<name> --arg mode=subset|full --arg lines=hugsim,b2d_turns,b2d_ds [--arg stage=smoke] [--arg force=1]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cllib  # noqa: E402

NAME, ROOT = "op-guard-cl", "op_guard/cl"


def jobs(args):
    return cllib.lane_jobs(args["candidate"], args.get("mode", "subset"), args.get("lines", "hugsim,b2d_turns,b2d_ds").split(","),
                           args.get("stage", "all"), args.get("force") == "1")
