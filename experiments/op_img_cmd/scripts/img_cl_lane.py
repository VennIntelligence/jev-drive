"""op_img_cmd closed-loop smoke (GPU-pool submitter): the sky-arrow fine-tune served with the arrow drawn from the official route,
against the shipped model's drive arm, on the four turn_agree-failure routes, 2 traffic seeds.
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md (runs only if the negatives round passed on 3 seeds).

  .venv/bin/python experiments/op_img_cmd/scripts/img_cl_lane.py [--stage 1|all] [--model q3NA-s0[,q3SA-s0]] [--wait] [--dry-run]

Units (op_arb.sh arm, one openpilot server per unit, 4 CARLA workers = one pool job: 30 GB, 4 CARLA, 12 cores; log dir
$DATA_DIR/runs/op_img_cmd/cl/pool/<unit>/; finished units are skipped, units still in the pool reused):
  drive    shipped Cinque (base ONNX), route desire on, resume timer (decision 102's smoke ran nored = decision 81's drive arm; refused since 2026-10-05)
  imgsky   the fine-tune (ONNX in $DATA_DIR/runs/op_img_cmd/cl/onnx/<model>.onnx), desire off, the agent sends [command, distance]
           of the next route command (lib/op_arb_agent.py img_cmd) and img_cl_server.py draws the sky arrow on every frame
stage 1 = imgsky seed 0 on 27043 alone with frame dumps (IMG_CL_DUMP); all = the 2 x 2 units.
Readout: img_cl_report.py -> results/cl_smoke.md.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/op_guard/scripts"), str(REPO)]
import cllib as C  # noqa: E402

ROOT = "op_img_cmd/cl"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / ROOT
ROUTES = ["27297", "27043", "9196", "24944"]
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"
R1 = '"resume": "timer"'   # was nored (ground-truth light) in the decision-102 smoke; nored is refused now (docs/openpilot-interface.md)


def unit(arm, seed, ids, model, extra=None):
    out = RUN / "arms" / f"{arm}-s{seed}" if len(ids) > 1 else RUN / "stage1" / f"{arm}-s{seed}-{ids[0]}"
    e = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{workers}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
             OP_ARB_ARMS=str(RUN / "arms"), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", OP_ARB_AGENT="lib/op_arb_agent.py",
             LAT_EXEC="curv", RESUME_S="5")
    if arm == "drive":
        e.update(DESIRE="true", DRIVE_ARGS=R1)
    else:
        e.update(DESIRE="false", DRIVE_ARGS=f'{R1}, "img_cmd": true', SRV_ONNX=str(RUN / "onnx" / f"{model}.onnx"),
                 SRV_PY="experiments/op_img_cmd/scripts/img_cl_server.py", IMG_CL_DUMP=str(RUN / "dump" / arm))
    e.update(extra or {})
    done = lambda out=out, n=len(ids): len(list((out / "done").glob("*.json"))) == n  # noqa: E731
    check = f'test $(ls {shlex.quote(str(out))}/done/*.json | wc -l) -eq {len(ids)}'
    return dict(name=f"{arm}-s{seed}" + ("" if len(ids) > 1 else f"-{ids[0]}"), cmd=shlex.join(["bash", OP_ARB, "arm", "drive", ",".join(ids), str(out)]) + " && " + check,
                env=e, vram_gb=7.5 * len(ids), carla=len(ids), cpu=12, tries=2, priority=0, out=str(out), done=done)


def units(stage="all", model="q3NA-s0"):
    models = model.split(",")                                   # "q3NA-s0,q3SA-s0": one sky arm per model (skyNA, skySA), one shared drive arm
    if stage == "1":
        return [unit("imgsky", 0, ["27043"], models[0], dict(IMG_CL_DUMP=str(RUN / "stage1" / "dump")))]
    sky = lambda m: [unit("sky" + m[2:-3], s, ROUTES, m) for s in (0, 1)]        # noqa: E731
    return sky(models[0]) + [unit("drive", s, ROUTES, models[0]) for s in (0, 1)] + [j for m in models[1:] for j in sky(m)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("1", "all"), default="all")
    ap.add_argument("--model", default="q3NA-s0")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ids = C.submit_units(units(a.stage, a.model), RUN / "pool", "op_img_cmd cl", dry_run=a.dry_run)
    if a.wait and not a.dry_run:
        bad = C.wait_units(ids)
        print("units not done: %s" % bad if bad else "all units done")
        sys.exit(1 if bad else 0)
