"""op_img_cmd closed-loop smoke (jevdrive.cl lane file): the sky-arrow fine-tune served with the arrow drawn from the official route,
against the shipped model's drive arm, on the four turn_agree-failure routes, 2 traffic seeds.
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md (runs only if the negatives round passed on 3 seeds).

  python -m jevdrive.cl lease img2 --gpus 1 ...
  scripts/tmux_run.sh img-cl .venv/bin/python -m jevdrive.cl run experiments/op_img_cmd/scripts/img_cl_lane.py [--arg stage=1|all] [--arg model=q3NA-s0]

Units (op_arb.sh arm, one openpilot server per unit, 4 CARLA workers):
  drive    shipped Cinque (base ONNX), route desire on, resume timer (decision 102's smoke ran nored = decision 81's drive arm; refused since 2026-10-05)
  imgsky   the fine-tune (ONNX in $DATA_DIR/runs/op_img_cmd/cl/onnx/<model>.onnx), desire off, the agent sends [command, distance]
           of the next route command (lib/op_arb_agent.py img_cmd) and img_cl_server.py draws the sky arrow on every frame
stage 1 = imgsky seed 0 on 27043 alone with frame dumps (IMG_CL_DUMP); all = the 2 x 2 units.
Readout: img_cl_report.py -> results/cl_smoke.md.
"""
import os
from pathlib import Path

from jevdrive.cl import Job

NAME, ROOT = "img-cl", "op_img_cmd/cl"
REPO = Path(__file__).resolve().parents[3]
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
    return Job(f"{arm}-s{seed}" + ("" if len(ids) > 1 else f"-{ids[0]}"), ["bash", OP_ARB, "arm", "drive", ",".join(ids), str(out)],
               workers=len(ids), vram_gb=7.5, cores=12, tries=2, env=e, out=str(out),
               ok=lambda j, out=out, n=len(ids): len(list((out / "done").glob("*.json"))) == n)


def jobs(args):
    models = args.get("model", "q3NA-s0").split(",")          # "q3NA-s0,q3SA-s0": one sky arm per model (skyNA, skySA), one shared drive arm
    if args.get("stage", "all") == "1":
        return [unit("imgsky", 0, ["27043"], models[0], dict(IMG_CL_DUMP=str(RUN / "stage1" / "dump")))]
    sky = lambda m: [unit("sky" + m[2:-3], s, ROUTES, m) for s in (0, 1)]        # noqa: E731
    return sky(models[0]) + [unit("drive", s, ROUTES, models[0]) for s in (0, 1)] + [j for m in models[1:] for j in sky(m)]
