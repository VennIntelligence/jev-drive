"""B2D small-step checks of the unified openpilot interface (docs/openpilot-interface.md, results/unified_interface.md).

  python -m jevdrive.cl lease uni-b2d --gpus 1 ...
  scripts/tmux_run.sh uni-b2d .venv/bin/python -m jevdrive.cl run experiments/leaderboard_audit/scripts/unified_b2d_lane.py \
      --lane uni-b2d [--arg stage=smoke|height|nored|all] [--arg seeds=2]

Units (experiments/op_closed_loop/archive/op_arb.sh arm, lib/op_arb_agent.py, shipped Cinque, one openpilot server per unit):
  height   6 val routes (2 forced turns 28180 / 24944 + 4 choice-turn routes, the jfa "few" set), seed(s) `seeds`:
           legacy    preset drive (raw action curvature, no clip / delay, 1.433 m), the reference every earlier B2D number used
           s143      preset spec (action -> clip_curvature -> 0.2 s lateralDelay), 1.433 m windshield top
           s122      preset spec, 1.22 m at the front bumper line (x 3.8 m: no hood in view, outside the tinted glass;
                     precedent: zeroshot_b2d_opfix.sh shadow-h122)
           s143nz / s122nz  the same with the dense-route zones off ("zones": false, "div_m": 1e9): action-only reading
  nored    the decision-102 smoke's drive arm (img_cl_lane.py: 4 routes x seeds 0, 1) with "resume": "timer" instead of nored
Units land in $DATA_DIR/runs/unified/b2d/arms/<arm>-s<seed>; hand-offs in $DATA_DIR/runs/unified/b2d.
"""
import os
from pathlib import Path

from jevdrive.cl import Job

NAME, ROOT = "uni-b2d", "unified/b2d"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / ROOT
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"
HEIGHT_ROUTES = ["28180", "24944", "27297", "9196", "6999", "34183"]
NORED_ROUTES = ["27297", "27043", "9196", "24944"]                     # img_cl_lane.py ROUTES
NOZ = '"zones": false, "div_m": 1e9'
H122 = '"op_mount": [3.8, 0.0, 1.22]'
ARMS = {"legacy": ("drive", "", ""), "s143": ("spec", "", ""), "s122": ("spec", "", H122),
        "s143nz": ("spec", NOZ, ""), "s122nz": ("spec", NOZ, H122), "drivetimer": ("drive", '"resume": "timer"', "")}


def unit(arm, seed, ids, tag=""):
    cfg_arm, drive_args, top = ARMS[arm]
    out = RUN / "arms" / f"{arm}{tag}-s{seed}"
    e = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{workers}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
             OP_ARB_ARMS=str(RUN / "arms"), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", OP_ARB_AGENT="lib/op_arb_agent.py",
             DESIRE="true")
    if cfg_arm == "drive":                                             # exactly img_cl_lane.py's drive unit
        e.update(LAT_EXEC="curv", RESUME_S="5")
    if drive_args:
        e["DRIVE_ARGS"] = drive_args
    if top:
        e["TOP_ARGS"] = top
    return Job(f"{arm}{tag}-s{seed}", ["bash", OP_ARB, "arm", cfg_arm, ",".join(ids), str(out)],
               workers=len(ids), vram_gb=7.5, cores=12, tries=2, env=e, out=str(out),
               ok=lambda j, out=out, n=len(ids): len(list((out / "done").glob("*.json"))) == n)


def jobs(args):
    seeds = [int(s) for s in args.get("seeds", "2").split(",")]
    stage = args.get("stage", "smoke")
    smoke = [unit("s122", 2, ["28180"], tag="dbg")]
    if stage == "smoke":
        return smoke
    out = []
    if stage in ("height", "all"):
        out += [unit(a, s, HEIGHT_ROUTES) for s in seeds for a in ("legacy", "s143", "s122", "s143nz", "s122nz")]
    if stage in ("nored", "all"):
        out += [unit("drivetimer", s, NORED_ROUTES) for s in (0, 1)]
    return out
