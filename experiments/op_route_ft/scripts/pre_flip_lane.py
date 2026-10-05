"""Lane file: command-flip control for rc-bear-pre-s0 on the 13 routes that hold a choice turn (B2D, zones off, desire off, seed 2).
The agent mirrors the navigation polyline it sends to the route adapter (cfg "route_flip": left <-> right command). A model that follows the
command goes to the wrong side or loses the turn; rc-ctl-pre has no side input and is unaffected by construction.
Units: $DATA_DIR/runs/op_route_ft/pre/flip/<arm>/b2d/turns-s2-k<K>. Run on a lease:
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/pre_flip_lane.py --lane <lane> --root $DATA_DIR/runs/op_route_ft/pre/flip/cl
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

NAME, ROOT = "rft-pre-flip", "op_route_ft/pre/flip/cl"
OUT = G.data_dir() / "runs/op_route_ft/pre/flip"
CHOICE_ROUTES = "10255 15102 24758 24944 26872 27297 28008 28147 334 34183 5423 6999 9196".split()   # routes with a choice (forced 0) turn


def jobs(args):
    out = []
    for arm in args.get("arms", "rc-bear-pre-s0").split(","):
        c = G.resolve(arm)
        for k in range(0, len(CHOICE_ROUTES), 3):
            j = C.b2d_unit("flip-%s-s%d-k%d" % (arm, C.TURN_SEED, k // 3), CHOICE_ROUTES[k:k + 3], OUT / arm / "b2d" / ("turns-s%d-k%d" % (C.TURN_SEED, k // 3)),
                           C.TURN_SEED, True, c.get("onnx"), priority=1, route_adapter=c.get("route_adapter"))
            j.env["DESIRE"] = "false"
            j.env["TOP_ARGS"] += ', "route_flip": true'
            out.append(j)
    return out
