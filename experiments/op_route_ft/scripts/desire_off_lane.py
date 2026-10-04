"""Lane file: B2D 25-turn set (guard b2d_turns unit config: zones off, aligned camera, seed 2) with the route turn desire OFF
(agent cfg "desire": false -> DESIRE_NONE every step; DESIRE=false in the unit env) for rc-bear-s0, rc-ctl-s0, shipped.
Units: $DATA_DIR/runs/op_route_ft/desire_off/<arm>/b2d/turns-s2-k<K>. Run (leased lane):
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/desire_off_lane.py --lane <lane> --root $DATA_DIR/runs/op_route_ft/desire_off/cl [--arg arms=rc-bear-s0,...]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

NAME, ROOT = "op-route-ft-desire-off", "op_route_ft/desire_off/cl"
OUT = G.data_dir() / "runs/op_route_ft/desire_off"


def jobs(args):
    out = []
    for arm in args.get("arms", "rc-bear-s0,rc-ctl-s0,shipped").split(","):
        c = G.resolve(arm)
        for k, ids in C.shards(C.TURN_ROUTES, "all"):
            j = C.b2d_unit("doff-%s-s%d-k%d" % (arm, C.TURN_SEED, k), ids, OUT / arm / "b2d" / ("turns-s%d-k%d" % (C.TURN_SEED, k)),
                           C.TURN_SEED, True, c.get("onnx"), priority=2, route_adapter=c.get("route_adapter"))
            j.env["DESIRE"] = "false"
            out.append(j)
    return out
