"""OD2 B2D lane: the d101 driver (it_dw3-s0 adapted Cinque, with and without the uncertainty selector) in the vlm_arb `drive`
arm on the 19-route diagnostic set (plan experiments/op_adapt_h/plans/2026-10-04-od2-prereg.md, section 3).

  scripts/tmux_run.sh od2-b2d .venv/bin/python -m jevdrive.cl run experiments/op_adapt_h/scripts/od2_b2d_lane.py \
      --lane od2-b2d --workers-per-card 6 --arg stage=smoke|all [--arg seeds=2,3] [--arg ratio=0.6]

Arms (agent = the unchanged `drive` arm, lib/vlm_arb_agent.py VLM_ARM=drive, same env as vmerge2's drive reruns):
  od2it   SRV_ONNX = it_dw3-s0 serving ONNX, no selector
  od2sel  + OP_SEL=<ratio>: op_arb_server.py's selector (second session on the heading-aligned 5 s history, below 3 m/s)
The shipped `drive` arm is not rerun: seeds 2, 3 of vmerge2 (v2-drive-s<seed>-q<k>) are the same code path (the server and agent
changes are inert without OP_SEL / SRV_ONNX). Units v2-od2<arm>-s<seed>-q<k> under $DATA_DIR/runs/vlm_arb/arms, 3 workers each.
Stages: smoke = `dbg3-od2sel` on one route (`dbg-od2sel`, `dbg2-od2sel`: earlier smokes, 0.1 deg steps / black out-of-image pixels) (the replay-fidelity check, OP_SEL_CHECK, was run off-CARLA: the unrotated 100-frame
replay reproduces the native plan exactly, max position difference 0.0 m over 30 steps); all = smoke + the batch (od2it and od2sel
per seed), each batch unit after the smoke.
Hand-offs in $DATA_DIR/runs/od2_b2d: STATUS, DONE / ERROR, lane/<ts>/log.txt.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
import vmerge_chain as vm  # noqa: E402  (sets base.SLOT_WORKERS / SLOT_CORES = 3, 9, as the vmerge2 drive reruns)

NAME, ROOT = "od2-b2d", "od2_b2d"
ONNX = str(base.DATA / "runs/op_adapt_H/onnx/it_dw3-s0.onnx")
SMOKE_ROUTE = "17280"          # short (~330 ticks), stop sign + junction turn: low-speed steps after yaw


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "2,3").split(",")]
    ratio = args.get("ratio", "0.6")
    e_it = dict(OP_DET_HEAD=vm.DET, SRV_ONNX=ONNX)
    e_sel = dict(e_it, OP_SEL=ratio)
    smoke = [base.unit("dbg3-od2sel", 2, SMOKE_ROUTE, [SMOKE_ROUTE], e_sel, "", 0, base="drive")]
    if args.get("stage") == "chk":                 # selector diagnosis on the smoke route: unrotated incremental (2), constant 0.01 deg (3)
        n = {"OP_SEL_N": args["n"]} if "n" in args else {}
        return [base.unit("dbg%s-od2chk%s" % (args.get("try", ""), c), 2, SMOKE_ROUTE, [SMOKE_ROUTE], dict(e_sel, OP_SEL_CHECK=c, **n), "", 0,
                          base="drive") for c in args.get("chk", "2,3").split(",")]
    out = list(smoke)
    if args.get("stage", "smoke") == "all":
        sh = vc.shards()
        for i, s in enumerate(seeds):
            for arm, e in (("od2it", e_it), ("od2sel", e_sel)):
                out += [base.unit(arm, s, k, sh[k], e, "", 1 + i, deps=[j.name for j in smoke], base="drive") for k in vc.SHARDS]
    for j in out:
        if "OP_SEL" in j.env:
            j.vram_gb = 9.5            # a second openpilot session per worker
    return out
