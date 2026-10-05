"""Lane file: shipped, B2D `spec` preset, zones off, seed 2, routes 24944 10255 26153 with arb.plan_vmin = 0 (crawl.md, smallest test of the harness rule).
Unit: $DATA_DIR/runs/op_route_ft/vmin0/b2d/turns-s2-k0. Run (leased lane):
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/vmin_lane.py --lane <lane> --root $DATA_DIR/runs/op_route_ft/vmin0/cl
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

NAME, ROOT = "op-route-ft-vmin0", "op_route_ft/vmin0/cl"
OUT = G.data_dir() / "runs/op_route_ft/vmin0"
ROUTES = ["24944", "10255", "26153"]


def jobs(args):
    j = C.b2d_unit("vmin0-s%d" % C.TURN_SEED, ROUTES, OUT / "b2d" / ("turns-s%d-k0" % C.TURN_SEED), C.TURN_SEED, True, None, priority=2)
    j.env["DRIVE_ARGS"] = C.NOZ + ', "plan_vmin": 0'
    return [j]
