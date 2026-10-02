"""The vmerge lane: the merged Bench2Drive arm (plan 2026-10-03-vmerge.md) on the 19-route x 2-seed diagnostic set.

  scripts/tmux_run.sh vmerge .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vmerge_chain.py \
      --lane vlm-vmerge --workers-per-card 3 --arg stage=pre|all

Rerunning the same command resumes (state.json; finished units are skipped, b2d_run skips finished routes inside a unit).
Hand-offs in $DATA_DIR/runs/vlm_arb_vmerge: STATUS, status.json, DONE / ERROR / ERROR.<job>, util.csv, lane/<ts>/log.txt;
unit outputs under $DATA_DIR/runs/vlm_arb/arms/v2-<unit>; readouts/<unit>/{routes.csv, checks.json}; gates/vmerge_*.json.
The per-card Qwen servers (with the /sign question) are a separate process (vlm_qwen_server.py supervise, window vm-srv2b / vm-srv0,
run dirs $DATA_DIR/runs/vlm_arb_vmerge/qwen<card>) and are not stopped by the lane. The trigger heads: OP_DET_HEAD
($DATA_DIR/runs/vlm_arb_vmerge/det_head.npz, vmerge_det_fit.py), loaded by each unit's openpilot server.

Stages (docs/long-runs.md):
  pre  `dbg-vmerge-334` (debug red-light route) and `dbg-vmerge-25169` (debug obstacle route), outside the 19
  all  `vmerge-s0-q0..q2`, then `few-vmerge` (vmerge_report.py few: no crash, latency lines), `vmerge-s1-q0..q2`, `report`
  full all + abl (the ablations start after `report`)
  abl  (--arg abl=nobyp,nocusum,...) one ablation arm `vm<abl>` per name (VM_ABL, one component off), both seeds, after `report`;
       then `report-abl` (vmerge_report.py report <abl names>)
Units: 3 CARLA workers + their own openpilot server, one unit per card (3 routes per card's Qwen server, as in vred).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
from vlm_arb_common import DATA, RUN  # noqa: E402

NAME, ROOT = "vlm-vmerge", "vlm_arb_vmerge"
base.SLOT_WORKERS, base.SLOT_CORES = 3, 9
SHARDS = vc.SHARDS
L_REGISTERED = 0.35                                  # s, the vred value (same server, same concurrency)
DET = str(DATA / "runs/vlm_arb_vmerge/det_head.npz")


def env():
    return dict(VLM_BACKEND="qwen", VLM_GPU="{gpu}", VLM_BASE_PORT="8200", VLM_L="%.2f" % L_REGISTERED, VLM_SAVE_FRAMES="1",
                OP_DET_HEAD=DET)


def gate(name):
    p = RUN / "gates" / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def tool(name, *args, deps=(), prio=0, ok=None):
    j = base.tool(name, "vmerge_report.py", *args, deps=deps, prio=prio)
    j.ok = ok
    return j


def pre_jobs():
    # dbg-vmerge-334 (first round) answered stop_sign_for_ego in front of a green light; the sign question now needs "no light for the ego"
    return [base.unit("dbg2-vmerge", 0, "334", ["334"], env(), "", 0, base="vmerge"),
            base.unit("dbg-vmerge", 0, "25169", ["25169"], env(), "", 0, base="vmerge")]


def abl_jobs(names, deps=()):
    sh = vc.shards()
    units = [base.unit("vm" + a, s, k, sh[k], dict(env(), VM_ABL=a), "", 6 + i, deps=deps, base="vmerge")
             for i, a in enumerate(names) for s in (0, 1) for k in SHARDS]
    return units + [tool("report-abl", "report", *names, deps=[j.name for j in units], prio=9)]


def jobs(args):
    stage = args.get("stage", "pre")
    pre = pre_jobs()
    if stage == "pre":
        return pre
    if stage == "abl":
        return abl_jobs(args["abl"].split(","))
    sh = vc.shards()
    v0 = [base.unit("vmerge", 0, k, sh[k], env(), "", 4, deps=[j.name for j in pre]) for k in SHARDS]
    few = tool("few-vmerge", "few", deps=[j.name for j in v0], prio=4, ok=lambda j: bool((gate("vmerge_few") or {}).get("passed")))
    v1 = [base.unit("vmerge", 1, k, sh[k], env(), "", 5, deps=[few.name]) for k in SHARDS]
    rep = tool("report", "report", deps=[j.name for j in v1], prio=9)
    out = pre + v0 + [few] + v1 + [rep]
    if stage == "full":                              # all, then the ablation arms (--arg abl=...) once the main report exists
        out += abl_jobs(args["abl"].split(","), deps=[rep.name])
    return out
