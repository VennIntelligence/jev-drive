"""Lane file: B2D 25-turn set (guard b2d_turns unit config: zones off, aligned camera `spec`, seed 2, route turn desire OFF) with the lateral
executor switched from the action head's curvature (curv, the spec default) to P7 tracking the model's PLAN path (LAT_EXEC=p7), or the hybrid
(hyb: action curvature below 3 m/s, plan tracking above; lib/op_arb_agent.py). Longitudinal unchanged.
Arms: shipped, rc-ctl-s0, rc-bear-s0, rc-poly-s0 (p7) and shipped-hyb (hyb). Units: $DATA_DIR/runs/op_route_ft/plan_track/<arm>/b2d/turns-s2-k<K>.
`part=i/n` keeps every n-th job (job list = arms in the order given, shards inside), so n lanes on n cards each run one slice. Run (leased lane):
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/plan_track_lane.py --lane <lane> --root $DATA_DIR/runs/op_route_ft/plan_track/cl-<i> \
      [--arg arms=rc-bear-s0,...] [--arg part=0/3] [--arg stage=smoke]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

NAME, ROOT = "op-route-ft-plan-track", "op_route_ft/plan_track/cl"
OUT = G.data_dir() / "runs/op_route_ft/plan_track"
ARMS = "shipped,rc-bear-s0,rc-ctl-s0,shipped-hyb,rc-poly-s0"


def jobs(args):
    out = []
    for arm in args.get("arms", ARMS).split(","):
        hyb = arm.endswith("-hyb")
        c = G.resolve(arm[:-4] if hyb else arm)
        for k, ids in C.shards(C.TURN_ROUTES, args.get("stage", "all")):
            j = C.b2d_unit("pt-%s-s%d-k%d" % (arm, C.TURN_SEED, k), ids, OUT / arm / "b2d" / ("turns-s%d-k%d" % (C.TURN_SEED, k)),
                           C.TURN_SEED, True, c.get("onnx"), priority=2, route_adapter=c.get("route_adapter"))
            j.env["DESIRE"] = "false"
            j.env["LAT_EXEC"] = "hyb" if hyb else "p7"
            out.append(j)
    if "part" in args:
        i, n = map(int, args["part"].split("/"))
        out = out[i::n]
    return out
