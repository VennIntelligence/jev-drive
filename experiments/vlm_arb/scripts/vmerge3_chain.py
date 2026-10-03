"""The vmerge3 lane: vmerge2 with the non-privileged bypass and the cross-traffic release check (plan 2026-10-04-vmerge3.md).

  scripts/tmux_run.sh vmerge3 .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vmerge3_chain.py \
      --lane vlm-vmerge3 --workers-per-card 6 --arg stage=pre|all [--arg seeds=0,1]

Arms (all = vmerge2's env: VLM_ARM=vmerge, VLM_R2_TARGET=junction, VM_R5_TMAX=50, plus):
  vmerge3   VM3_BYP=perc + VM3_REL=1      both changes
  vm3priv   VM3_REL=1                     ablation of the bypass change: vmerge2's privileged pbyp3 bypass + the release check
  vm3norel  VM3_BYP=perc                  ablation of the release check
Stages: pre = smokes `dbg-vm3-s0-25169` (obstacle debug route) and `dbg-vm3-s0-334` (red-light debug route), outside the 19;
all = per seed the three arms (vmerge3 first), `report` last (vmerge3_report.py). Units `<arm>-s<seed>-q0..q2` under
$DATA_DIR/runs/vlm_arb/arms/v2-<unit>. Two Qwen servers on the one card (ports 8202 / 8212, `vlm_qwen_server.py supervise --cards 2
--port0 8200|8210`); each route process takes a free slot on one of them (VLM_PORTS / VLM_SLOT_DIR, lib/vlm_arb_agent._qwen_port),
so two units of 3 routes share the card at the vmerge serving load of 3 routes per server.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
import vmerge_chain as vm  # noqa: E402  (sets base.SLOT_WORKERS / SLOT_CORES)
from vlm_arb_common import RUN  # noqa: E402

NAME, ROOT = "vlm-vmerge3", "vlm_arb_vmerge3"
PORTS = "8202,8212"
ARMS = {"vmerge3": dict(VM3_BYP="perc", VM3_REL="1", OP_LANES="1"),
        "vm3priv": dict(VM3_REL="1"),
        "vm3norel": dict(VM3_BYP="perc", OP_LANES="1")}


def env(arm):
    e = dict(vm.env(), **vm.J_ENV)
    e.update(ARMS[arm], VLM_PORTS=PORTS, VLM_SLOT_DIR=str(RUN.parent / ROOT / "qwen_slots"))
    return e


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "0,1").split(",")]
    tag = args.get("smoke", "dbg-vm3")              # smoke rounds after a code change: dbg2-vm3, dbg3-vm3, ... (execution log)
    only = args.get("smoke_routes", "25169,334").split(",")
    smoke = [base.unit(tag, 0, r, [r], env("vmerge3"), "", 0, base="vmerge") for r in only]
    if args.get("stage", "pre") == "pre":
        return smoke
    sh = vc.shards()
    out, names = [], []
    for i, s in enumerate(seeds):
        for j, arm in enumerate(ARMS):
            u = [base.unit(arm, s, k, sh[k], env(arm), "", 3 * i + j, deps=[x.name for x in smoke]) for k in vc.SHARDS]
            out += u
            names += [x.name for x in u]
    return smoke + out + [base.tool("report", "vmerge3_report.py", "report", deps=names, prio=99)]
