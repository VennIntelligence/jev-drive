"""Low-speed lateral transfer limit on B2D (plans/2026-10-04-lowspeed-ctrl-prereg.md, section B): the unchanged `drive` arm with env
LOWSPEED_CTRL (scripts/b2d_zeroshot_agent.py applies lib/lowspeed_ctrl.B2DFilter to the 20 Hz steer command).

  scripts/tmux_run.sh lsc-b2d .venv/bin/python -m jevdrive.cl run experiments/hugsim/scripts/lowspeed_b2d_lane.py \
      --lane lsc-b2d --workers-per-card 6 --arg stage=smoke|all [--arg seeds=2,3]

Arm `lsc` units v2-lsc-s<seed>-q<k> under $DATA_DIR/runs/vlm_arb/arms; baseline = vmerge2's drive reruns v2-drive-s<seed>-q<k> (same code
path, the hook is inert without the env). Hand-offs in $DATA_DIR/runs/lsc_b2d.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
import vmerge_chain as vm  # noqa: E402  (sets SLOT_WORKERS / SLOT_CORES = 3, 9, as the vmerge2 drive reruns)

NAME, ROOT = "lsc-b2d", "lsc_b2d"
RULE = {"jerk": 5.0, "tau0": 3.0, "v0": 2.5, "v1": 3.5}
SMOKE_ROUTE = "17280"


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "2,3").split(",")]
    env = dict(OP_DET_HEAD=vm.DET, LOWSPEED_CTRL=json.dumps(RULE))
    smoke = [base.unit("dbg-lsc", 2, SMOKE_ROUTE, [SMOKE_ROUTE], env, "", 0, base="drive")]
    out = list(smoke)
    if args.get("stage", "smoke") == "all":
        sh = vc.shards()
        for i, s in enumerate(seeds):
            out += [base.unit("lsc", s, k, sh[k], env, "", 1 + i, deps=[j.name for j in smoke], base="drive") for k in vc.SHARDS]
    return out
