"""Lane file: rc-bear-pre-s0 / rc-ctl-pre-s0 on the B2D 25-turn set (guard b2d_turns unit config: zones off, aligned camera, seed 2), desire off and on
(plans/2026-10-05-route-ft-prereg.md, 2026-10-06 section).
  desire off: $DATA_DIR/runs/op_route_ft/desire_off/<arm>/b2d/turns-s2-k<K> (agent DESIRE=false, as desire_off_lane.py)
  desire on:  $DATA_DIR/runs/op_guard/<arm>/subset/b2d/turns-s2-k<K> (the guard's own b2d_turns units, reused by guard.py)
Run on the lane's lease:
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/pre_lane.py --lane rft-pre --root $DATA_DIR/runs/op_route_ft/pre/cl \
      [--arg arms=rc-bear-pre-s0,rc-ctl-pre-s0] [--arg desire=off,on]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

NAME, ROOT = "rft-pre", "op_route_ft/pre/cl"
DOFF = G.data_dir() / "runs/op_route_ft/desire_off"


def jobs(args):
    out = []
    for de in args.get("desire", "off,on").split(","):
        for arm in args.get("arms", "rc-bear-pre-s0,rc-ctl-pre-s0").split(","):
            c = G.resolve(arm)
            base = DOFF / arm if de == "off" else G.run_dir(arm, "subset")
            for k, ids in C.shards(C.TURN_ROUTES, "all"):
                j = C.b2d_unit("pre-%s-%s-s%d-k%d" % (de, arm, C.TURN_SEED, k), ids, base / "b2d" / ("turns-s%d-k%d" % (C.TURN_SEED, k)),
                               C.TURN_SEED, True, c.get("onnx"), priority=3 if de == "off" else 2, route_adapter=c.get("route_adapter"))
                j.env["DESIRE"] = "false" if de == "off" else "true"
                out.append(j)
    return out
