"""One recorded B2D run per (arm, route) for a GIF: chase camera + every model input frame, op_route_ft candidates (rig122's stage=gif unit
with the candidate's ONNX and route adapter; the arm config is decision 127's olnz: spec, zones off, open-loop camera, desire on, seed 2).

  python -m jevdrive.cl lease rft-gif --gpus 1
  scripts/tmux_run.sh rft-gif .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/rft_gif_lane.py --lane rft-gif --arg gif=rc-bear-s0:10255
  then experiments/op_closed_loop/scripts/junction_forced_gif.py <attempt> <dump> <out.gif> --route-turn <mi>
Units under $DATA_DIR/runs/op_route_ft/gif/<arm>-<route>/, frames under .../dump/<arm>_<route>/.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/op_closed_loop/scripts"))
import junction_rig122_lane as J  # noqa: E402

NAME, ROOT = "rft-gif", "op_route_ft"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / "op_route_ft" / "gif"


def jobs(args):
    from dataclasses import replace
    out = []
    for g in args["gif"].split(","):
        arm, rid = g.split(":")
        onnx = DATA / "runs/op_route_ft/onnx" / f"{arm}.onnx"
        ad = onnx.with_suffix(".adapter.npz")
        extra = dict(OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm", VLM_ARM="drive", SRV_PY=J.SRV,
                     IMG_CL_DUMP=str(RUN / "dump" / f"{arm}_{rid}"), IMG_CL_DUMP_EVERY="1")
        if arm != "shipped":
            extra["SRV_ONNX"] = str(onnx)
            if ad.exists():
                extra["TOP_ARGS"] = '"route_adapter": "%s"' % ad
        u = J.unit("olnz", [rid], "g" + rid, extra=extra)
        o = RUN / f"{arm}-{rid}"
        u = replace(u, name=f"{arm}-g{rid}", out=str(o), cmd=u.cmd[:-1] + [str(o)],
                    ok=lambda j, o=o: len(list((o / "done").glob("*.json"))) == 1)
        out.append(u)
    return out
