"""Submitter of the op_wide_ft B2D units (plans/2026-10-05-wide-ft-prereg.md): fine-tuned arm x wide FOV of its input on the B2D 25-turn set
(guard b2d_turns: zones off, aligned camera `spec`, seed 2), route turn desire OFF, action-curvature executor (curv, op-path), route adapter on.
One GPU-pool job per unit (cllib.b2d_unit: 3 routes, 22.5 GB, 3 CARLA, 12 cores). The wide FOV is the server's warp focal (env OP_WIDE_FOCAL; the
CARLA wide sensor is the same 118.9 deg camera in every arm), recorded in interface.json as rig.wide = sensor-f160.

Arms: `<tag>@<fov>`, tag = a run under $DATA_DIR/runs/op_wide_ft/onnx/<tag>.onnx (+ .adapter.npz), or `shipped`; fov 58 (focal 455) or 116 (160).
Units -> $DATA_DIR/runs/op_wide_ft/b2d/<arm>/b2d/turns[-small]-s2-k<K>; pool logs in <root>/<unit>/.
  .venv/bin/python experiments/op_wide_ft/scripts/wide_lane.py --arms wf-w58-s0@58,wf-w116-s0@116 [--routes small|all] [--stage smoke] [--wait] [--dry-run]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_route_ft/scripts"), str(REPO)]
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402
from near_lane import SMALL  # noqa: E402  (9 turns / 8 routes: the op_route_ft staged small set)

W = G.data_dir() / "runs/op_wide_ft"
FOCAL = {"58": None, "116": 160.0}


def arm_dir(arm):
    return W / "b2d" / arm / "b2d"


def units(arms, routes="all", stage="all"):
    out = []
    for arm in arms:
        tag, fov = arm.split("@")
        onnx = None if tag == "shipped" else W / "onnx" / f"{tag}.onnx"
        if onnx is not None and not onnx.exists():
            raise SystemExit(f"{arm}: {onnx} missing")
        ad = onnx.with_suffix(".adapter.npz") if onnx is not None else None
        small = routes == "small"
        for k, ids in C.shards(SMALL if small else C.TURN_ROUTES, stage):
            name = "%s-%s@%s-k%d" % ("s" if small else "a", tag.replace("wf-", "").replace("-s0", ""), fov, k)
            u = C.b2d_unit(name, ids, arm_dir(arm) / ("%s-s%d-k%d" % ("turns-small" if small else "turns", C.TURN_SEED, k)), C.TURN_SEED, True,
                           str(onnx) if onnx else None, priority=3, route_adapter=str(ad) if ad is not None and ad.exists() else None)
            u["env"]["DESIRE"] = "false"
            if FOCAL[fov]:
                u["env"]["OP_WIDE_FOCAL"] = "%g" % FOCAL[fov]
            out.append(u)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(W / "cl"), help="pool log dirs <root>/<unit>/")
    ap.add_argument("--arms", default="wf-w58-s0@58,wf-w116-s0@116")
    ap.add_argument("--routes", choices=("all", "small"), default="small")
    ap.add_argument("--stage", choices=("smoke", "all"), default="all")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ids = C.submit_units(units(a.arms.split(","), a.routes, a.stage), a.root, "op_wide_ft", dry_run=a.dry_run)
    if a.wait and not a.dry_run:
        bad = C.wait_units(ids)
        print("units not done: %s" % bad if bad else "all units done", flush=True)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
