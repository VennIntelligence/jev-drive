"""One recorded B2D run per (arm, route) for the op_wide_ft GIF: chase camera + every model input frame as the network got it (road + wide at
the arm's wide FOV), op_route_ft's GIF unit (rft_gif_lane.gif_unit: record agent, img_cl_server frame dump) with the arm's ONNX + route adapter,
desire off and OP_WIDE_FOCAL for the 116 deg arm (the same settings as wide_lane.py's turn units). One GPU-pool job per run.

  .venv/bin/python experiments/op_wide_ft/scripts/wide_gif.py --gif wf-w58-s0@58:10255,wf-w116-s0@116:10255 [--wait] [--dry-run]
  then experiments/op_closed_loop/scripts/junction_forced_gif.py <attempt> <dump> <out.gif> --route-turn <mi>
Units under $DATA_DIR/runs/op_wide_ft/gif/<arm>-<route>/, frames under .../dump/<arm>_<route>/.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/op_route_ft/scripts"), str(REPO / "experiments/op_guard/scripts"), str(REPO)]
import cllib as C  # noqa: E402
import rft_gif_lane as GL  # noqa: E402
from wide_lane import FOCAL, W  # noqa: E402

RUN = W / "gif"


def units(gif):
    out = []
    for g in gif.split(","):
        arm, rid = g.rsplit(":", 1)
        tag, fov = arm.split("@")
        onnx = W / "onnx" / f"{tag}.onnx"
        ad = onnx.with_suffix(".adapter.npz")
        name = "g-%s@%s-%s" % (tag.replace("wf-", "").replace("-s0", ""), fov, rid)
        u = C.b2d_unit(name, [rid], RUN / f"{arm}-{rid}", 2, True, str(onnx), route_adapter=str(ad) if ad.exists() else None)
        u["env"].update(OP_ARB_ARMS=str(GL.RIG122_ARMS), OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm", VLM_ARM="drive",
                        SRV_PY=GL.SRV, IMG_CL_DUMP=str(RUN / "dump" / f"{arm}_{rid}"), IMG_CL_DUMP_EVERY="1", DESIRE="false")
        if FOCAL[fov]:
            u["env"]["OP_WIDE_FOCAL"] = "%g" % FOCAL[fov]
        out.append(u)
    return out


if __name__ == "__main__":
    GL.main(units, RUN / "pool", __doc__)
