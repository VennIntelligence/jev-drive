"""The vred lane: closed-loop arm `vred` with zero-shot Qwen3-VL-4B reading the light (plan 2026-10-02-vlm-vred.md).

  scripts/tmux_run.sh vlm-vred .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vlm_vred_chain.py \
      --lane vlm-arb --workers-per-card 3 --arg stage=pre|all

Rerunning the same command resumes (state.json; finished units are skipped, b2d_run skips finished routes inside a unit).
Hand-offs in $DATA_DIR/runs/vlm_arb_vred: STATUS, status.json, DONE / ERROR / ERROR.<job>, util.csv, lane/<ts>/log.txt;
unit outputs under $DATA_DIR/runs/vlm_arb/arms/v2-<unit>; readouts/<unit>/{routes.csv, checks.json}; gates/vred_*.json.
The per-card Qwen servers are a separate process (vlm_qwen_server.py supervise, window vq-srv) and are not stopped by the lane.

Stages (docs/long-runs.md: 1 unit -> a few -> all):
  pre   1  `dbg-vred-qwen-334`: the real VLM drives R2 + R5 on the debug red-light route 334 (checklist: red_stop), provisional L
        2  `cal-s1-q0..q2`: shadow `drive` (no control) with the same server, 3 workers on each of the three cards at once =
           the concurrency of the batch, all 19 routes of seed 1; `calibrate` reads the in-loop latency and writes
           gates/vred_cal.json (L = measured p95, 0.05 s grid). Calibration only: its outcomes are not read.
  all   needs the registered L (gates/vred_L_registered, written by hand after the plan was pushed with it)
        3  `vred-s0-q0..q2`: the batch, seed 0, then `few`: checklist over those three units (a stop driven by a VLM answer,
           a release after green, no crash); only then
        4  `vred-s1-q0..q2` (seed 1), then `report`.
Units: 3 CARLA workers + their own openpilot server, one unit per card at a time (--workers-per-card 3).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
from jevdrive.cl import Job  # noqa: E402
from vlm_arb_common import DATA, RUN, ROUTES, SEEDS, drive_dir, route_row, write_json  # noqa: E402

NAME, ROOT = "vlm-arb", "vlm_arb_vred"
base.SLOT_WORKERS, base.SLOT_CORES = 3, 9
SHARDS = ("q0", "q1", "q2")
L_PROVISIONAL = 0.5
ROWS = "R2,R5"                                  # R3 off (no fast path for the stop-sign question), R1 / R4 not part of vred


def shards():
    """Routes dealt over three shards by decreasing drive game time (LPT), fixed once in gates/vred_shards.json."""
    p = RUN / "gates/vred_shards.json"
    if not p.exists():
        t = {}
        for rid in ROUTES:
            v = [r["game_s"] for r in (route_row(drive_dir(rid, s), rid) for s in SEEDS) if r]
            t[rid] = sum(v) / len(v) if v else 60.0
        out, load = {k: [] for k in SHARDS}, {k: 0.0 for k in SHARDS}
        for rid in sorted(ROUTES, key=lambda r: (-t[r], r)):
            k = min(SHARDS, key=lambda k: (load[k], k))
            out[k].append(rid)
            load[k] += t[rid]
        write_json(p, out)
    return json.loads(p.read_text())


def qwen_env(L, shadow=False):
    e = dict(VLM_BACKEND="qwen", VLM_GPU="{gpu}", VLM_BASE_PORT="8200", VLM_L="%.2f" % L, VLM_ROWS=ROWS, VLM_SAVE_FRAMES="1")
    if shadow:
        e["VLM_SHADOW"] = "1"
    return e


def gate(name):
    p = RUN / "gates" / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def tool(name, *args, deps=(), prio=0, ok=None):
    j = base.tool(name, "vlm_vred_report.py", *args, deps=deps, prio=prio)
    j.ok = ok
    return j


def pre_jobs():
    sh = shards()
    dbg = base.unit("dbg-vred-qwen", 0, "334", ["334"], qwen_env(L_PROVISIONAL), "red_stop", 0, base="vred")
    cal = [base.unit("cal", 1, k, sh[k], qwen_env(L_PROVISIONAL, True), "shadow", 1, deps=[dbg.name], base="drive") for k in SHARDS]
    return [dbg] + cal + [tool("calibrate", "calibrate", deps=[j.name for j in cal], prio=2,
                                ok=lambda j: bool((gate("vred_cal") or {}).get("done")))]


def batch_jobs(L):
    sh = shards()
    s0 = [base.unit("vred", 0, k, sh[k], qwen_env(L), "", 3) for k in SHARDS]
    few = tool("few", "few", deps=[j.name for j in s0], prio=3, ok=lambda j: bool((gate("vred_few") or {}).get("passed")))
    s1 = [base.unit("vred", 1, k, sh[k], qwen_env(L), "", 4, deps=[few.name]) for k in SHARDS]
    rep = tool("report", "report", deps=[j.name for j in s1], prio=9)
    return s0 + [few] + s1 + [rep]


def jobs(args):
    stage = args.get("stage", "pre")
    RUN.mkdir(parents=True, exist_ok=True)
    out = pre_jobs()
    if stage == "pre":
        return out
    reg, cal = (RUN / "gates/vred_L_registered"), gate("vred_cal")
    if not reg.exists() or not cal:
        raise SystemExit("stage all needs gates/vred_cal.json and gates/vred_L_registered (the plan with L pushed first)")
    L = float(reg.read_text().split()[0])
    return out + batch_jobs(L)
