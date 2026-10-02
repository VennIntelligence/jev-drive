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
  pre   three single-route units at once, one per card: `dbg-pbyp2-25169`, `dbg-pbyp2-24955` (debug obstacle routes, `pbyp2dbg`
        checks: obstacle detected, shifted path, valid shift, returned) and `dbg-vred2-334` (the real VLM drives R2 + R5 with the
        stop-line target on the debug red-light route 334, `red_stop2`).
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
from vlm_arb_common import RUN  # noqa: E402

NAME, ROOT = "vlm-v2", "vlm_arb_v2"
base.SLOT_WORKERS, base.SLOT_CORES = 3, 9
SHARDS = vc.SHARDS
L_REGISTERED = 0.35                              # s, registered in the plan (the vred value, re-measured here)
STOPLINE = dict(VLM_R2_TARGET="stopline")


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
    return [base.unit("dbg-pbyp2", 0, "25169", ["25169"], None, "pbyp2dbg", 0, base="pbyp2"),
            base.unit("dbg-pbyp2", 0, "24955", ["24955"], None, "pbyp2dbg", 0, base="pbyp2"),
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


def jobs(args):
    stage = args.get("stage", "pre")
    RUN.mkdir(parents=True, exist_ok=True)
    pre = pre_jobs()
    return pre if stage == "pre" else pre + batch_jobs(pre)
