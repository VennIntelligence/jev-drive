"""One recorded B2D run per (arm, route) with plan tracking (LAT_EXEC=p7; arm ...-hyb = hybrid) for a GIF: chase camera + every model input frame.
Same unit as rft_gif_lane.py (rig122 olnz: spec, zones off, aligned camera, seed 2) but desire OFF and the lateral executor p7 / hyb.
One GPU-pool job per run; pool logs in <root>/<tag>/ (default $DATA_DIR/runs/op_route_ft/plan_track/gif-cl); finished runs are skipped.
  .venv/bin/python experiments/op_route_ft/scripts/plan_track_gif_lane.py --gif rc-bear-s0:10255[,rc-bear-s0-hyb:10255] [--wait] [--dry-run]
then experiments/op_closed_loop/scripts/junction_forced_gif.py <attempt> <dump> <out.gif> --route-turn <mi>
Units under $DATA_DIR/runs/op_route_ft/plan_track_gif/<arm>-p7-<route>/, frames under .../dump/.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rft_gif_lane as R  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs" / "op_route_ft" / "plan_track_gif"


def units(gif):
    out = []
    for g in gif.split(","):
        arm, rid = g.split(":")
        hyb = arm.endswith("-hyb")
        base = arm[:-4] if hyb else arm
        tag = "%s-%s-%s" % (arm, "hyb" if hyb else "p7", rid)
        out.append(R.gif_unit(tag, base, rid, RUN / tag, RUN / "dump" / tag, DESIRE="false", LAT_EXEC="hyb" if hyb else "p7"))
    return out


if __name__ == "__main__":
    R.main(units, DATA / "runs/op_route_ft/plan_track/gif-cl", __doc__)
