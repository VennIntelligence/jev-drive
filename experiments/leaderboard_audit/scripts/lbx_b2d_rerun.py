"""Loss-budget example reruns with a chase camera AND openpilot input dumps (jevdrive.cl lane file).

  python -m jevdrive.cl lease lbx-b2d --gpus 1 --explicit        (the card: --gpus 1 --explicit picks card 1 only if it is the one named)
  scripts/tmux_run.sh lbx-b2d .venv/bin/python -m jevdrive.cl run experiments/leaderboard_audit/scripts/lbx_b2d_rerun.py --lane lbx-b2d \
      --workers-per-card 4 [--arg only=drive-24944-s0,... --arg tag=b]
  vmerge2 units need a Qwen server on the card: $DATA_DIR/envs/jevdrive/bin/python experiments/vlm_arb/scripts/vlm_qwen_server.py supervise --cards 1 --run $DATA_DIR/runs/lbx_b2d/qwen

One unit per (arm, seed, route), the same arm / route / traffic seed / configuration as the original example. Each unit has its own openpilot
server (lbx_dump_server.py: LBX_DUMP=$DATA_DIR/runs/lbx_b2d/dump/<unit>) and the chase recorder (vlm_arb_record_agent.py). Outputs:
$DATA_DIR/runs/vlm_arb/arms/v2-lbx-<arm>-s<seed>-<route>/attempts/<route>/<n>/{chase_raw.mp4, ...}; dumps in runs/lbx_b2d/dump/<unit>/.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "vlm_arb" / "scripts"
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vmerge_chain as vm  # noqa: E402  (sets base.SLOT_WORKERS / SLOT_CORES)

NAME, ROOT = "lbx-b2d", "lbx_b2d"
base.SLOT_WORKERS, base.SLOT_CORES = 1, 4
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
AGENT = "experiments/vlm_arb/scripts/vlm_arb_record_agent.py"
SRV = "experiments/leaderboard_audit/scripts/lbx_dump_server.py"
ENV2 = dict(vm.env(), **vm.J_ENV)               # vmerge2 = vmerge + VLM_R2_TARGET=junction + VM_R5_TMAX=50 (vmerge2_chain.py)
# (arm, base arm, seed, route, env, example)
UNITS = [
    ("drive", "drive", 0, "24944", {}, "R1 drive"), ("drive", "drive", 0, "19324", {}, "B1"), ("drive", "drive", 0, "9196", {}, "B2"),
    ("drive", "drive", 0, "27043", {}, "C4 drive"),
    ("pbyp2ng", "pbyp2ng", 0, "19832", {}, "C1"), ("pbyp2ng", "pbyp2ng", 0, "2520", {}, "C2"),
    ("vmerge2", "vmerge", 1, "15612", ENV2, "B3"), ("vmerge2", "vmerge", 1, "19832", ENV2, "C3"),
    ("vmerge2", "vmerge", 0, "27043", ENV2, "C4 vmerge2"), ("vmerge2", "vmerge", 2, "17280", ENV2, "C5"),
]


def jobs(args):
    only = set(args["only"].split(",")) if "only" in args else None
    tag = args.get("tag", "")                       # --arg tag=b: a further attempt of the same unit (unit / dump names get the suffix)
    out = []
    for arm, real, seed, route, env, _ in UNITS:
        key = "%s-%s-s%d" % (arm, route, seed)
        if only and key not in only:
            continue
        key += tag
        e = dict(env, OP_ARB_AGENT=AGENT, SRV_PY=SRV, OP_LANES="1", LBX_DUMP=str(DATA / "runs/lbx_b2d/dump" / key))
        if real in ("drive", "vmerge"):
            e["GIF_BASE"] = "vlm"
        j = base.unit("lbx-" + arm + tag, seed, route, [route], e, "", 0, base=real)
        j.ok = None
        out.append(j)
    return out
