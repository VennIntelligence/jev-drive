"""Closed-loop junction turns of openpilot's action head with nothing else steering (results/junction_closed_loop.md).

  scripts/tmux_run.sh jcl .venv/bin/python -m jevdrive.cl run experiments/op_closed_loop/scripts/junction_cl_lane.py \
      --lane jcl --workers-per-card 6 --arg stage=smoke|all [--arg seeds=2] [--arg shards=6]

Arms (all the shipped Cinque `drive` agent, lib/op_arb_agent.py, no OP_CTRL, never "resume": "nored"):
  jclA  as shipped: the dense route steers in the junction zones (DRIVE_ZONES) and on divergence
  jclD  "zones": false and "div_m": 1e9: the action-head curvature steers everywhere (no zone, no divergence hand-over)
  jclE  jclD plus "zone_gain": 1.95 on the curvature inside LEFT / RIGHT command runs +- 3 m
Routes: the 36 B2D routes holding the 38 turns >= 25 deg of turn_calibration_options.py (list in ROUTES). Units v2-jcl<arm>-s<seed>-q<k>
under $DATA_DIR/runs/vlm_arb/arms; hand-offs in $DATA_DIR/runs/jcl.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402  (sets SLOT_WORKERS / SLOT_CORES = 3, 9)
import vmerge_chain as vm  # noqa: E402

NAME, ROOT = "jcl", "jcl"
ROUTES = ("10255 10364 15102 24944 25051 27043 27297 27870 28147 28180 34183 34391 35330 4104 4183 4721 5423 6999 7157 7616 7841 8859 "
          "9102 9196 9218 9646 35243 334 17280 15612 16390 15483 16508 16529 26872 27787").split()
ARMS = {"jclA": "", "jclD": '"zones": false, "div_m": 1e9', "jclE": '"zones": false, "div_m": 1e9, "zone_gain": 1.95'}
SMOKE_ROUTE = "17280"


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "2").split(",")]
    n = int(args.get("shards", "6"))
    base_env = dict(OP_DET_HEAD=vm.DET)
    env = lambda a: dict(base_env, **({"DRIVE_ARGS": ARMS[a]} if ARMS[a] else {}))  # noqa: E731
    smoke = [base.unit("dbg-" + a, 2, SMOKE_ROUTE, [SMOKE_ROUTE], env(a), "", 0, base="drive") for a in ("jclD", "jclE")]
    out = list(smoke)
    if args.get("stage", "smoke") == "all":
        for s in seeds:
            for k in range(n):
                out += [base.unit(a, s, "q%d" % k, ROUTES[k::n], env(a), "", 1, deps=[j.name for j in smoke], base="drive") for a in ARMS]
    return out
