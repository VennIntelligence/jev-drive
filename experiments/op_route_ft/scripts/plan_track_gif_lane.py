"""One recorded B2D run per (arm, route) with plan tracking (LAT_EXEC=p7; arm ...-hyb = hybrid) for a GIF: chase camera + every model input frame.
Same unit as rft_gif_lane.py (rig122 olnz: spec, zones off, aligned camera, seed 2) but desire OFF and the lateral executor of plan_track_lane.py.
  .venv/bin/python -m jevdrive.cl run experiments/op_route_ft/scripts/plan_track_gif_lane.py --lane <lane> --root $DATA_DIR/runs/op_route_ft/plan_track/gif-cl --arg gif=rc-bear-s0:10255
then experiments/op_closed_loop/scripts/junction_forced_gif.py <attempt> <dump> <out.gif> --route-turn <mi>
Units under $DATA_DIR/runs/op_route_ft/plan_track_gif/<arm>-p7-<route>/, frames under .../dump/.
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/op_closed_loop/scripts"))
import junction_rig122_lane as J  # noqa: E402

NAME, ROOT = "rft-pt-gif", "op_route_ft"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / "op_route_ft" / "plan_track_gif"


def jobs(args):
    out = []
    for g in args["gif"].split(","):
        arm, rid = g.split(":")
        hyb = arm.endswith("-hyb")
        base = arm[:-4] if hyb else arm
        onnx = DATA / "runs/op_route_ft/onnx" / f"{base}.onnx"
        ad = onnx.with_suffix(".adapter.npz")
        tag = "%s-%s-%s" % (arm, "hyb" if hyb else "p7", rid)
        extra = dict(OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm", VLM_ARM="drive", SRV_PY=J.SRV, DESIRE="false",
                     LAT_EXEC="hyb" if hyb else "p7", IMG_CL_DUMP=str(RUN / "dump" / tag), IMG_CL_DUMP_EVERY="1")
        if base != "shipped":
            extra["SRV_ONNX"] = str(onnx)
            if ad.exists():
                extra["TOP_ARGS"] = '"route_adapter": "%s"' % ad
        u = J.unit("olnz", [rid], "g" + rid, extra=extra)
        o = RUN / tag
        out.append(replace(u, name=tag, out=str(o), cmd=u.cmd[:-1] + [str(o)], ok=lambda j, o=o: len(list((o / "done").glob("*.json"))) == 1))
    return out
