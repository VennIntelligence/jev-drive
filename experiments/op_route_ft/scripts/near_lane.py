"""Lane file: rc-bear-near-s0 / rc-bear-fix-s0 (/ rc-ctl-near-s0) on the B2D 25-turn set (guard b2d_turns unit config: zones off, aligned camera `spec`, seed 2),
route turn desire OFF, lateral executor per `exec` (curv = action curvature, the spec default; p7 = plan tracking; hyb; lib/op_arb_agent.py).
Units land where the existing readers look, so pre_report.py (`<arm>@doff`) and plan_track_report.py read them unchanged:
  curv -> $DATA_DIR/runs/op_route_ft/desire_off/<arm>/b2d/turns-s2-k<K>;  p7 / hyb -> $DATA_DIR/runs/op_route_ft/plan_track/<arm>[-hyb]/b2d/turns-s2-k<K>
Run on a lease:
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/near_lane.py --lane rft-near --root $DATA_DIR/runs/op_route_ft/near/cl \
      [--arg arms=rc-bear-near-s0,rc-bear-fix-s0,rc-ctl-near-s0] [--arg exec=curv,p7] [--arg stage=smoke] [--arg routes=small]
routes=small: the staged first read (user rule 2026-10-05: stop early on a clear negative), 8 routes / 9 of the 25 turns, mixed choice / forced,
tight / wide, and the turns rc-bear-s0 (desire off) entered late: SMALL below; the units go to turns-small-s2-k<K> (not mixed with the 25-turn dirs).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

NAME, ROOT = "rft-near", "op_route_ft/near/cl"
DATA = G.data_dir() / "runs/op_route_ft"
SMALL = "10255 5423 15102 34183 25051 28180 27994 28008".split()   # rc-bear-s0 desire off: late x 6, crawl x 1, took x 2 (28008 holds 2 turns)


def jobs(args):
    out = []
    for ex in args.get("exec", "curv").split(","):
        for arm in args.get("arms", "rc-bear-near-s0,rc-bear-fix-s0").split(","):
            c = G.resolve(arm)
            base = DATA / "desire_off" / arm if ex == "curv" else DATA / "plan_track" / (arm + ("-hyb" if ex == "hyb" else ""))
            small = args.get("routes") == "small"
            for k, ids in C.shards(SMALL if small else C.TURN_ROUTES, args.get("stage", "all")):
                sub = "turns-small" if small else "turns"
                j = C.b2d_unit("near-%s-%s-%s-s%d-k%d" % (ex, arm, sub, C.TURN_SEED, k), ids, base / "b2d" / ("%s-s%d-k%d" % (sub, C.TURN_SEED, k)),
                               C.TURN_SEED, True, c.get("onnx"), priority=3 if arm.startswith("rc-bear") else 2, route_adapter=c.get("route_adapter"))
                j.env["DESIRE"] = "false"
                if ex != "curv":
                    j.env["LAT_EXEC"] = ex
                out.append(j)
    return out
