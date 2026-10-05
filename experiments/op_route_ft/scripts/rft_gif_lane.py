"""One recorded B2D run per (arm, route) for a GIF: chase camera + every model input frame, op_route_ft candidates (rig122's stage=gif unit
with the candidate's ONNX and route adapter; the arm config is decision 127's olnz: spec, zones off, open-loop camera, desire on, seed 2).
One GPU-pool job per run (1 CARLA worker, 7.5 GB, 12 cores); pool logs in <root>/<arm>-g<route>/; finished runs are skipped.

  .venv/bin/python experiments/op_route_ft/scripts/rft_gif_lane.py --gif rc-bear-s0:10255[,...] [--wait] [--dry-run]
  then experiments/op_closed_loop/scripts/junction_forced_gif.py <attempt> <dump> <out.gif> --route-turn <mi>
Units under $DATA_DIR/runs/op_route_ft/gif/<arm>-<route>/, frames under .../dump/<arm>_<route>/.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/op_guard/scripts"), str(REPO)]
import cllib as C  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / "op_route_ft" / "gif"
RIG122_ARMS = DATA / "runs/rig122/arms"                     # OP_ARB_ARMS of the rig122 unit these runs copy
SRV = "experiments/op_img_cmd/scripts/img_cl_server.py"


def gif_unit(name, arm, rid, out, dump, **env):
    """rig122 `olnz` unit on one route with the record agent + img_cl_server frame dump; `arm` = candidate base name (shipped: base ONNX)."""
    onnx = DATA / "runs/op_route_ft/onnx" / f"{arm}.onnx"
    ad = onnx.with_suffix(".adapter.npz")
    u = C.b2d_unit(name, [rid], out, 2, True, None if arm == "shipped" else str(onnx), route_adapter=str(ad) if arm != "shipped" and ad.exists() else None)
    u["env"].update(OP_ARB_ARMS=str(RIG122_ARMS), OP_ARB_AGENT="experiments/vlm_arb/scripts/vlm_arb_record_agent.py", GIF_BASE="vlm", VLM_ARM="drive",
                    SRV_PY=SRV, IMG_CL_DUMP=str(dump), IMG_CL_DUMP_EVERY="1", **env)
    return u


def main(units_of, root, doc):
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--gif", required=True, help="<arm>:<route>[,...]")
    ap.add_argument("--root", default=str(root), help="pool log dirs <root>/<unit>/")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ids = C.submit_units(units_of(a.gif), a.root, "op_route_ft gif", dry_run=a.dry_run)
    if a.wait and not a.dry_run:
        bad = C.wait_units(ids)
        print("units not done: %s" % bad if bad else "all units done")
        sys.exit(1 if bad else 0)


def units(gif):
    out = []
    for g in gif.split(","):
        arm, rid = g.split(":")
        out.append(gif_unit(f"{arm}-g{rid}", arm, rid, RUN / f"{arm}-{rid}", RUN / "dump" / f"{arm}_{rid}"))
    return out


if __name__ == "__main__":
    main(units, RUN / "pool", __doc__)
