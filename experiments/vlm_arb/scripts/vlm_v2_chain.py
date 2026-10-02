"""The v2 lane: closed-loop arms `pbyp2` (privileged bypass, fixed) and `vred2` (R2 stops at the light's stop line)
(plan 2026-10-02-pbyp2-vred2.md).

  scripts/tmux_run.sh vlm-v2 .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vlm_v2_chain.py \
      --lane vlm-v2 --workers-per-card 6 --arg stage=pre|all

Rerunning the same command resumes (state.json; finished units are skipped, b2d_run skips finished routes inside a unit).
Hand-offs in $DATA_DIR/runs/vlm_arb_v2: STATUS, status.json, DONE / ERROR / ERROR.<job>, util.csv, lane/<ts>/log.txt;
unit outputs under $DATA_DIR/runs/vlm_arb/arms/v2-<unit>; readouts/<unit>/{routes.csv, checks.json}; gates/v2_*.json.
The per-card Qwen servers are a separate process (vlm_qwen_server.py supervise, window vq2-srv, run dir
$DATA_DIR/runs/vlm_arb_v2/qwen) and are not stopped by the lane.

Stages (docs/long-runs.md: 1 unit -> a few -> all):
  pre   three single-route units at once, one per card: `dbg2-pbyp2-25169`, `dbg2-pbyp2-24955` (second round after the one adjustment of plans section 9; the first round is `dbg-pbyp2-*`) (debug obstacle routes, `pbyp2dbg`
        checks: obstacle detected, shifted path, valid shift, returned) and `dbg-vred2-334` (the real VLM drives R2 + R5 with the
        stop-line target on the debug red-light route 334, `red_stop2`).
  v     (after `all` stopped at the failed calibration, plan section 9) `cal3-s1-q0..q2` alone at 3 workers per card -> `calibrate3` (gates/v2_cal3.json),
        `vred2-s0-q0..q2`, `few-vred2`, `vred2-s1-q0..q2`, `report`; run with --workers-per-card 3 after the pbyp2 units are done
  ng    (plan section 10, after stage `v`) `pbyp2ng-s<seed>-{a,b,c}`: pbyp2 without any gap check on the four obstacle routes x 2 seeds, then `report2`
  all   4  `cal2-s1-q0..q2` (shadow `drive`, Qwen servers) and `pbyp2-s0-q0..q2` at the same time (6 workers per card): `calibrate`
           reads the in-loop latency and passes only with p95 <= L = 0.35 s; `few-pbyp2` checks the pbyp2 seed-0 units
        5  `vred2-s0-q0..q2` and `pbyp2-s1-q0..q2` at the same time (the load of the calibration), then `few-vred2`
        6  `vred2-s1-q0..q2`, then `report`.
Units: 3 CARLA workers + their own openpilot server; the shards are the vred shards (gates/vred_shards.json).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
from vlm_arb_common import RUN, SEEDS  # noqa: E402

NAME, ROOT = "vlm-v2", "vlm_arb_v2"
base.SLOT_WORKERS, base.SLOT_CORES = 3, 9
SHARDS = vc.SHARDS
L_REGISTERED = 0.35                              # s, registered in the plan (the vred value, re-measured here)
STOPLINE = dict(VLM_R2_TARGET="stopline")
WAIVED = ("dbg2-pbyp2-s0-24955",)


def gate(name):
    p = RUN / "gates" / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def passed(name):
    return lambda j: bool((gate(name) or {}).get("passed"))


def tool(name, *args, deps=(), prio=0, ok=None):
    j = base.tool(name, "vlm_v2_report.py", *args, deps=deps, prio=prio)
    j.ok = ok
    return j


def pre_jobs():
    return [base.unit("dbg2-pbyp2", 0, "25169", ["25169"], None, "pbyp2dbg", 0, base="pbyp2"),
            base.unit("dbg2-pbyp2", 0, "24955", ["24955"], None, "pbyp2dbg", 0, base="pbyp2"),
            base.unit("dbg-vred2", 0, "334", ["334"], dict(vc.qwen_env(L_REGISTERED), **STOPLINE), "red_stop2", 0, base="vred")]


def batch_jobs(pre):
    sh = vc.shards()
    pre = [j.name for j in pre]
    qe = dict(vc.qwen_env(L_REGISTERED), **STOPLINE)
    p0 = [base.unit("pbyp2", 0, k, sh[k], None, "pbyp2", 3, deps=pre) for k in SHARDS]
    cal = [base.unit("cal2", 1, k, sh[k], vc.qwen_env(L_REGISTERED, True), "shadow", 3, deps=pre, base="drive") for k in SHARDS]
    calib = tool("calibrate", "calibrate", deps=[j.name for j in cal], prio=2, ok=passed("v2_cal"))
    few_p = tool("few-pbyp2", "few-pbyp2", deps=[j.name for j in p0], prio=3, ok=passed("v2_few_pbyp2"))
    v0 = [base.unit("vred2", 0, k, sh[k], qe, "", 4, deps=[calib.name], base="vred") for k in SHARDS]
    p1 = [base.unit("pbyp2", 1, k, sh[k], None, "pbyp2", 4, deps=[few_p.name]) for k in SHARDS]
    few_v = tool("few-vred2", "few-vred2", deps=[j.name for j in v0], prio=4, ok=passed("v2_few_vred2"))
    v1 = [base.unit("vred2", 1, k, sh[k], qe, "", 5, deps=[few_v.name], base="vred") for k in SHARDS]
    rep = tool("report", "report", deps=[j.name for j in p1 + v1], prio=9)
    return p0 + cal + [calib, few_p] + v0 + p1 + [few_v] + v1 + [rep]


def vred2_jobs():
    """Stage `v`: the registered rule after the calibration at the load of stage `all` failed (p95 446 ms > 350 ms, gates/v2_cal.json): concurrency reduced to the
    `vred` batch's own (3 workers per card, one unit per card, nothing else on the box), latency re-measured at that load, then the vred2 batch."""
    sh = vc.shards()
    qe = dict(vc.qwen_env(L_REGISTERED), **STOPLINE)
    cal = [base.unit("cal3", 1, k, sh[k], vc.qwen_env(L_REGISTERED, True), "shadow", 2, base="drive") for k in SHARDS]
    calib = tool("calibrate3", "calibrate3", deps=[j.name for j in cal], prio=2, ok=passed("v2_cal3"))
    v0 = [base.unit("vred2", 0, k, sh[k], qe, "", 4, deps=[calib.name], base="vred") for k in SHARDS]
    few_v = tool("few-vred2", "few-vred2", deps=[j.name for j in v0], prio=4, ok=passed("v2_few_vred2"))
    v1 = [base.unit("vred2", 1, k, sh[k], qe, "", 5, deps=[few_v.name], base="vred") for k in SHARDS]
    rep = tool("report", "report", deps=[j.name for j in v1], prio=9)
    return cal + [calib] + v0 + [few_v] + v1 + [rep]


def ng_jobs():
    """Stage `ng` (plan section 10): the diagnostic ablation pbyp2ng (no gap check) on the four obstacle routes x 2 seeds, after the vred2 batch has finished."""
    ids = {"a": ["19324", "24497"], "b": ["2520"], "c": ["19832"]}
    units = [base.unit("pbyp2ng", s, k, v, None, "pbyp2", 3) for s in SEEDS for k, v in ids.items()]
    return units + [tool("report2", "report", deps=[j.name for j in units], prio=9)]


def jobs(args):
    stage = args.get("stage", "pre")
    if stage == "v":
        return vred2_jobs()
    if stage == "ng":
        return ng_jobs()
    RUN.mkdir(parents=True, exist_ok=True)
    pre = pre_jobs()
    if stage == "pre":
        return pre
    # dbg2-pbyp2-s0-24955 stalls under the frozen gap rule (plan section 9: a stream of traffic every 3 s, the gap opens for 0.2-0.8 s at a time); it is
    # kept as evidence, not as a dependency, so the batch does not wait on it
    pre = [j for j in pre if j.name not in WAIVED]
    return pre + batch_jobs(pre)
