"""The vlm_arb chain: one jevdrive.cl lane file that runs the whole experiment and advances itself.

  scripts/tmux_run.sh vlm-chain .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vlm_arb_chain.py [--arg stage=1|2|all]

Rerunning the same command resumes: finished jobs are skipped (state.json), live ones are adopted, and b2d_run skips
finished routes inside a unit. Hand-offs in $DATA_DIR/runs/vlm_arb: STATUS, status.json, DONE / ERROR / ERROR.<job>,
util.csv, lane/<ts>/log.txt, jobs/<job>/log.<k>.txt, readouts/<unit>/{routes.csv, checks.json}, gates/*.json,
results/{phase_a.md, lightsweep.md, report.md, *.png}.

Stages (docs/long-runs.md): 1 = one unit (the table driven by truth answers on debug route 334); 2 = the ten
single-route checklist units on debug routes; all = the batch. A stage's jobs stay in state.json, so "all" does not
repeat them.

The batch (plan deviation D7: 19 routes x 2 traffic seeds; a unit = arm x seed x shard, 4 workers):
  at once     drive (seed 1 everywhere, seed 0 on the six added red-light / stop-sign routes; shadow VLM + frames),
              the debug-route shadow unit, phase A on the old frames, jslow, pred, pbyp
  phase A     after the drive units: gates/phase_a.json decides the VLM arms; nothing is relaxed here, the lines are
              the registered ones in vlm_arb_phase_a.LINES
  vbyp        if Q-block and the latency line pass
  dslow       after jslow and drive of the same seed: per-route set speed = 8 x v_jslow / v_drive (cruise_by_route)
  vred, vall  if Q-light and the latency line pass (R3 only if Q-sign passed, R4 only if Q-block passed)
  sweep       offline (deviations D15, D16), as soon as the shadow units are done, whatever the gates say: the
              Q-light prompt sweep on junction-approach frames (vlm_arb_lightsweep.py). sweep-select picks the
              frames, sweep-<model> runs one model alone on a card, sweep writes results/lightsweep.md. It unlocks
              no closed-loop arm.
  report      when nothing else of the closed loop is left
Packing: every unit takes 4 CARLA workers, 12 cores and its own openpilot server; the lane places as many as the
card's VRAM and the per-card worker cap allow (card 0 also holds the VLM server), and pulls the next job whenever a
slot frees.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jevdrive.cl import Job  # noqa: E402
from vlm_arb_checks import unit_ok  # noqa: E402
from vlm_arb_common import DATA, DEV, OBS, RED, REPO, RUN, SEEDS, SHARDS, STOP, drive_dir, route_row, unit_dir, unit_name, write_json  # noqa: E402

NAME, ROOT = "vlm-arb", "vlm_arb"
SLOT_WORKERS, SLOT_CORES = 4, 12
WORKERS_PER_CARD = 8                  # two slots; the VRAM check leaves card 0 (VLM server) with one
VRAM_GB = 7.5                         # per worker: CARLA server ~5.5 GB + its share of the openpilot server
PY = str(DATA / "envs/jevdrive/bin/python")
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"
PRIO = {"drive": 0, "dbg-shadow": 0, "jslow": 1, "pred": 2, "pbyp": 2, "vbyp": 3, "dslow": 4, "vred": 5, "vall": 5}
STATE = {"final": False}
SERVER_MODELS = ("openjev", "dgemma-chat")           # served by the VLM server on card 0
SWEEP_MODELS = SERVER_MODELS + ("qwen3-vl-4b", "cosmos-reason1-7b", "qwen-drive-1.0-4b", "internvl2-1b")   # vlm_arb_models.MODELS


def unit(arm, seed, shard, ids, env=None, kind="", prio=None, deps=(), base=None):
    """One closed-loop unit through op_arb.sh. `base`: the arm name op_arb.sh and the agent see (debug units)."""
    name, out, real = unit_name(arm, seed, shard), unit_dir(arm, seed, shard), base or arm
    priv = real in ("pred", "pbyp", "pbyp2")
    e = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{workers}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
             OP_ARB_ARMS=str(RUN / "arms"), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", VLM_ARM=real,
             OP_ARB_AGENT="lib/op_arb_agent.py" if priv else "lib/vlm_arb_agent.py")
    if priv:
        e["PC_ENABLE"] = "1"
    e.update(env or {})
    ids = list(ids)
    cfg_arm = real
    if "CRUISE_BY_ROUTE" in e:        # op_arb.sh's own dslow case mis-nests a set CRUISE_BY_ROUTE; DRIVE_ARGS is clean
        cfg_arm, e["DRIVE_ARGS"] = "drive", '"cruise_by_route": ' + e.pop("CRUISE_BY_ROUTE")
    return Job(name, ["bash", OP_ARB, "arm", cfg_arm, ",".join(ids), str(out)], workers=min(SLOT_WORKERS, len(ids)),
               vram_gb=VRAM_GB, cores=SLOT_CORES if len(ids) > 1 else 4, tries=2, retry_check=True, env=e,
               priority=PRIO.get(arm, 1) if prio is None else prio, deps=tuple(deps), out=str(out),
               ok=lambda j: unit_ok(out, real, seed, ids, kind, "v2-" + name), meta=dict(arm=real, seed=seed, routes=len(ids)))


def tool(name, script, *args, deps=(), prio=0, ready=None):
    """An analysis step: no CARLA worker, no VRAM claim."""
    return Job(name, [PY, str(HERE / script)] + list(args), workers=0, vram_gb=0.0, deps=tuple(deps), tries=2,
               priority=prio, cuda=False, ready=ready)


def stage_jobs(stage):
    """Staged-launch units on debug routes; truth answers (VLM_ORACLE) drive the table so its execution is tested
    independently of what the VLM sees."""
    orc = dict(VLM_ORACLE="truth")
    one = [unit("dbg-vred-oracle", 0, "334", ["334"], orc, "red_stop", 0, base="vred")]
    if stage == "1":
        return one
    return one + [
        unit("dbg-vred-stuck", 0, "334b", ["334"], dict(VLM_ORACLE="stuckred"), "r5", 0, base="vred"),
        unit("dbg-vbyp-oracle", 0, "25169", ["25169"], orc, "bypass", 0, base="vbyp"),
        unit("dbg-vbyp-oracle", 0, "24955", ["24955"], orc, "bypass", 0, base="vbyp"),
        unit("dbg-vall-vlm", 0, "334", ["334"], None, "", 0, base="vall"),
        unit("dbg-vbyp-vlm", 0, "25169", ["25169"], None, "", 0, base="vbyp"),
        unit("dbg-jslow", 0, "26872", ["26872"], None, "r1", 0, base="jslow"),
        unit("dbg-dslow", 0, "26872", ["26872"], dict(CRUISE_BY_ROUTE='{"26872": 5.0}'), "cruise", 0, base="dslow"),
        unit("dbg-pred", 0, "334", ["334"], None, "pred", 0, base="pred"),
        unit("dbg-pbyp", 0, "25169", ["25169"], None, "pbyp", 0, base="pbyp"),
        unit("dbg-shadow", 0, "light", ["334", "27787"], dict(VLM_SHADOW="1", VLM_SAVE_FRAMES="1"), "shadow", 0, base="drive"),
    ]


def batch(arm, env=None, kind=""):
    return [unit(arm, s, sh, ids, env, kind) for s in SEEDS for sh, ids in SHARDS.items()]


def jobs(args):
    stage = args.get("stage", "all")
    STATE["stage"] = stage
    RUN.mkdir(parents=True, exist_ok=True)
    out = stage_jobs("2" if stage != "1" else "1")
    if stage != "all":
        return out
    shadow = dict(VLM_SHADOW="1", VLM_SAVE_FRAMES="1")
    drive = [unit("drive", 0, "tgt", RED + STOP, shadow, "shadow"),                       # seed 0: dev + obstacle routes are reused
             unit("drive", 1, "dev", DEV, shadow, "shadow"), unit("drive", 1, "tgt", RED + STOP + OBS, shadow, "shadow")]
    out += drive + batch("jslow") + batch("pred") + batch("pbyp")
    out += [tool("phaseA-old", "vlm_arb_phase_a.py", "--stage", "old"),
            tool("phaseA", "vlm_arb_phase_a.py", "--stage", "final", deps=[j.name for j in drive]),
            tool("report", "vlm_arb_report.py", prio=9, ready=lambda j: STATE["final"])]
    # The offline Q-light sweep (deviations D15, D16): no gate decides whether it runs. It waits for the shadow units
    # because those need the VLM server at its closed-loop latency. Each model is alone on its card (latency); the two
    # served by the VLM server take card 0 in turn. `ready` instead of deps: a failed step fails the next, no deadlock.
    shadow = tuple(j.name for j in drive) + (unit_name("dbg-shadow", 0, "light"),)
    sweep = [PY, str(HERE / "vlm_arb_lightsweep.py")]
    out.append(tool("sweep-select", "vlm_arb_lightsweep.py", "select", deps=shadow, prio=3))
    for m in SWEEP_MODELS:
        srv = m in SERVER_MODELS
        out.append(Job("sweep-" + m, sweep + ["run", "--model", m], workers=1, vram_gb=0.0 if srv else 24.0, cores=8,
                       tries=1, priority=8, exclusive=True, cuda=not srv, gpus=(0,) if srv else (),
                       ready=lambda j: STATE.get("select_end", False)))
    out.append(Job("sweep", sweep + ["report"], workers=0, vram_gb=0.0, tries=2, priority=8, cuda=False,
                   ready=lambda j: STATE.get("sweep_end", False)))
    return out


def gate(name):
    p = RUN / "gates" / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def cruise_by_route(seed):
    """Speed match of dslow to jslow, per route: set speed = 8 x v_jslow / v_drive, clipped to [0.5, 8]. Computed once."""
    p = RUN / "gates" / ("cruise-s%d.json" % seed)
    if not p.exists():
        m = {}
        for sh, ids in SHARDS.items():
            for rid in ids:
                a, b = route_row(unit_dir("jslow", seed, sh), rid), route_row(drive_dir(rid, seed), rid)
                if a and b and a["v_mean"] == a["v_mean"] and b["v_mean"] > 1e-3:
                    m[rid] = round(min(max(8.0 * a["v_mean"] / b["v_mean"], 0.5), 8.0), 2)
        write_json(p, m)
    return json.loads(p.read_text())


def more(lane, args):
    """Called every round: the jobs that results unlock. Idempotent; the lane keeps one job per name."""
    if STATE.get("stage") != "all":
        return []
    st = lambda n: lane.st["jobs"].get(n, {}).get("state")      # noqa: E731
    end = lambda names: all(st(n) in ("done", "failed") for n in names)   # noqa: E731
    done = lambda names: all(st(n) == "done" for n in names)    # noqa: E731
    new, settled = [], True
    STATE["select_end"] = st("sweep-select") in ("done", "failed")
    STATE["sweep_end"] = all(st("sweep-" + m) in ("done", "failed") for m in SWEEP_MODELS)
    for s in SEEDS:                                             # dslow, matched to jslow of the same seed
        need = [unit_name("jslow", s, sh) for sh in SHARDS] + [unit_name("drive", s, "tgt")] + ([unit_name("drive", s, "dev")] if s else [])
        if done(need):
            cb = json.dumps(cruise_by_route(s))
            new += [unit("dslow", s, sh, ids, dict(CRUISE_BY_ROUTE=cb)) for sh, ids in SHARDS.items()]
        elif not end(need):
            settled = False
    g = gate("phase_a")
    if g is None:
        settled = settled and st("phaseA") == "failed"
    else:
        vlm = dict(VLM_L="%.2f" % g["L_s"])
        if g["q_block"] and g["latency"]:
            new += batch("vbyp", vlm)
        variant = "base" if g["q_light"] else None            # D16: the sweep's prompts unlock nothing
        if variant and g["latency"]:
            rows = ["R2", "R5"] + (["R3"] if g["q_sign"] else [])
            env = dict(vlm, VLM_LIGHT_VARIANT=variant)
            new += batch("vred", dict(env, VLM_ROWS=",".join(rows)))
            new += batch("vall", dict(env, VLM_ROWS=",".join(rows + ["R1"] + (["R4"] if g["q_block"] else []))))
    others = [n for n in list(lane.jobs) + [j.name for j in new] if n != "report"]       # the report is last
    STATE["final"] = settled and end(others)
    return new
