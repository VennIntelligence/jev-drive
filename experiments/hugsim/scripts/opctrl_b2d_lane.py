"""openpilot's own lateral path on B2D (plans/2026-10-04-op-control-stack-b2d-prereg.md): the `drive` arm with env OP_CTRL
(scripts/b2d_zeroshot_agent.py _curvature_steer routes the action curvature through lib/op_ctrl.OpLateral: modeld hold, controlsd
latActive + clip_curvature, 0.2 s lateralDelay; the curvature -> CARLA steer geometry is the shipped one).

  scripts/tmux_run.sh opc-b2d .venv/bin/python -m jevdrive.cl run experiments/hugsim/scripts/opctrl_b2d_lane.py \
      --lane opc-b2d --workers-per-card 6 --arg stage=smoke|all [--arg seeds=2,3]

Arm `opc` units v2-opc-s<seed>-q<k> under $DATA_DIR/runs/vlm_arb/arms; baseline = vmerge2's drive reruns v2-drive-s<seed>-q<k>.
Hand-offs in $DATA_DIR/runs/opc_b2d.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
import vmerge_chain as vm  # noqa: E402  (sets SLOT_WORKERS / SLOT_CORES = 3, 9, as the vmerge2 drive reruns)

NAME, ROOT = "opc-b2d", "opc_b2d"
RULE = {"delay": 0.2}                      # seconds; everything else is openpilot's own (lib/op_ctrl.DEFAULT)
SMOKE_ROUTE = "17280"


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "2,3").split(",")]
    env = dict(OP_DET_HEAD=vm.DET, OP_CTRL=json.dumps(RULE))
    smoke = [base.unit("dbg-opc", 2, SMOKE_ROUTE, [SMOKE_ROUTE], env, "", 0, base="drive")]
    out = list(smoke)
    if args.get("stage", "smoke") == "all":
        sh = vc.shards()
        for i, s in enumerate(seeds):
            out += [base.unit("opc", s, k, sh[k], env, "", 1 + i, deps=[j.name for j in smoke], base="drive") for k in vc.SHARDS]
    return out
