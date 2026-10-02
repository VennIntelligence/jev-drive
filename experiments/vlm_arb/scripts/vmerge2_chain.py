"""The vmerge2 lane: vmerge + VLM_R2_TARGET=junction + VM_R5_TMAX=50 on all 19 routes (plan 2026-10-03-vmerge.md, section "vmerge2").

  scripts/tmux_run.sh vmerge2 .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vmerge2_chain.py \
      --lane vlm-vmerge2 --workers-per-card 3 --arg seeds=0,1,2,3

Minimal extension of vmerge_chain.py (its code paths are unchanged; this file only declares units). Order: smoke `dbg-vmerge2-s0-15102`,
then vmerge2 seeds 0, 1 (priority 0, 1), then per extra seed: vmerge2 and the unchanged `drive` arm (priority 2 + 2 * k), `report` last.
Units are `vmerge2-s<seed>-q0..q2` and `drive-s<seed>-q0..q2` (extra seeds only), under $DATA_DIR/runs/vlm_arb/arms/v2-<unit>.
Hand-offs in $DATA_DIR/runs/vlm_arb_vmerge2: STATUS, DONE / ERROR, DRAIN (touch: no new unit starts), lane/<ts>/log.txt.
Qwen servers: one supervisor, `vlm_qwen_server.py supervise --cards 0,2 --run $DATA_DIR/runs/vlm_arb_vmerge2/qwen` (STOP file ends it).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
import vmerge_chain as vm  # noqa: E402  (sets base.SLOT_WORKERS / SLOT_CORES)

NAME, ROOT = "vlm-vmerge2", "vlm_arb_vmerge2"
ENV2 = dict(vm.env(), **vm.J_ENV)
ENV_DRIVE = dict(OP_DET_HEAD=vm.DET)               # the unchanged drive arm: no VLM, same trigger-head file path as the other vmerge units


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "0,1").split(",")]
    sh = vc.shards()
    smoke = base.unit("dbg-vmerge2", 0, "15102", ["15102"], ENV2, "", 0, base="vmerge")
    out, deps = [smoke], []
    for i, s in enumerate(seeds):
        v = [base.unit("vmerge2", s, k, sh[k], ENV2, "", 2 * i, deps=[smoke.name], base="vmerge") for k in vc.SHARDS]
        out += v
        deps += [j.name for j in v]
        if s >= 2:                                    # repeat-noise replicates of the unchanged drive arm, right behind vmerge2 of the same seed
            d = [base.unit("drive", s, k, sh[k], ENV_DRIVE, "", 2 * i + 1, deps=[smoke.name]) for k in vc.SHARDS]
            out += d
            deps += [j.name for j in d]
    return out + [base.tool("report", "vmerge2_report.py", "report", deps=deps, prio=99)]
