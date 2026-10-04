"""Junction turns at the openpilot camera height: B2D `spec` preset at 1.22 m (bumper line) with and without the dense-route zones (results/junction_rig122.md).

  python -m jevdrive.cl lease rig122 --gpus 2
  scripts/tmux_run.sh rig122 .venv/bin/python -m jevdrive.cl run experiments/op_closed_loop/scripts/junction_rig122_lane.py \
      --lane rig122 [--arg stage=smoke|all|gif] [--arg gif=<arm>:<route>[,...]]

Arms (op_arb.sh `spec`, shipped Cinque, seed 2, one run per cell, turn desire on as in decisions 121 / 122):
  s122    zones on (reference), camera (3.8, 0, 1.22)
  s122nz  zones off (`"zones": false, "div_m": 1e9`): the action head steers everywhere, 1.22 m
  s143nz  zones off, 1.433 m: the height-only control for s122nz (same op-path execution, so only the camera differs from s122nz)
  ol      preset spec as shipped after the 2026-10-05 user decision: the open-loop-aligned camera (1.59, 0, 1.86) (interface.B2D_MOUNTS["openloop"])
  olnz    ol with the zones off
(stage=ol runs ol / olnz on the 20 routes; stages all / smoke are the 1.22 m side row, stopped by the change of plan: s122nz / s143nz k2 never ran.)
Routes: 14 val routes not in the unified-interface run (6 choice-turn + 8 forced-turn routes, all in junction_cl_per_turn.csv / junction_forced_per_turn.csv);
the 6 routes of unified_b2d_lane.py (28180 24944 27297 9196 6999 34183) already have all three arms there (`$DATA_DIR/runs/unified/b2d/arms/<arm>-s2`).
Units `<arm>-s2-k<k>` under `$DATA_DIR/runs/rig122/arms`. stage=gif: chase camera + every model input frame of one run (record agent + img_cl_server, no arrow).
"""
import os
from dataclasses import replace
from pathlib import Path

from jevdrive.cl import Job

NAME, ROOT = "rig122", "rig122"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / ROOT
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"
NEW = "10255 15102 28147 5423 334 26872 25051 27994 26153 26723 26365 24758 28008 24416".split()
NOZ = '"zones": false, "div_m": 1e9'
H122 = '"op_mount": [3.8, 0.0, 1.22]'
ARMS = {"s122": ("", H122), "s122nz": (NOZ, H122), "s143nz": (NOZ, ""), "ol": ("", ""), "olnz": (NOZ, "")}
OLD6 = "28180 24944 27297 9196 6999 34183".split()
SRV = "experiments/op_img_cmd/scripts/img_cl_server.py"


def unit(arm, ids, tag, seed=2, extra=None, workers=None):
    drive_args, top = ARMS[arm]
    out = RUN / "arms" / f"{arm}-s{seed}-{tag}"
    e = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{workers}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
             OP_ARB_ARMS=str(RUN / "arms"), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", OP_ARB_AGENT="lib/op_arb_agent.py", DESIRE="true")
    if drive_args:
        e["DRIVE_ARGS"] = drive_args
    if top:
        e["TOP_ARGS"] = top
    e.update(extra or {})
    return Job(f"{arm}-s{seed}-{tag}", ["bash", OP_ARB, "arm", "spec", ",".join(ids), str(out)], workers=workers or len(ids), vram_gb=7.5, cores=12, tries=2,
               env=e, out=str(out), ok=lambda j, out=out, n=len(ids): len(list((out / "done").glob("*.json"))) == n)


def jobs(args):
    stage = args.get("stage", "smoke")
    if stage in ("olsmoke", "ol"):
        dbg = unit("olnz", ["10255"], "dbg")
        allr = NEW + OLD6
        if stage == "olsmoke":
            return [dbg]
        return [dbg] + [replace(unit(a, allr[k::4], "k%d" % k), deps=(dbg.name,)) for k in range(4) for a in ("ol", "olnz")]
    smoke = [unit("s122nz", ["10255"], "dbg")]
    if stage == "smoke":
        return smoke
    if stage == "gif":                                       # gif=<arm>:<route>[,...]
        out = []
        for g in args["gif"].split(","):
            arm, rid = g.split(":")
            out.append(unit(arm, [rid], "g" + rid, extra=dict(OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm", VLM_ARM="drive", SRV_PY=SRV,
                                                          IMG_CL_DUMP=str(RUN / "dump" / f"{arm}_{rid}"), IMG_CL_DUMP_EVERY="1")))
        return out
    return smoke + [replace(unit(a, NEW[k::3], "k%d" % k), deps=("s122nz-s2-dbg",)) for k in range(3) for a in ARMS]
