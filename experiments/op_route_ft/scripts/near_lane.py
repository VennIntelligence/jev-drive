"""Submitter of the rc-*-near B2D units: rc-bear-near-s0 / rc-bear-fix-s0 (/ rc-ctl-near-s0) on the B2D 25-turn set (guard b2d_turns unit
config: zones off, aligned camera `spec`, seed 2), route turn desire OFF, lateral executor per --exec (curv = action curvature, the spec
default; p7 = plan tracking; hyb; lib/op_arb_agent.py). One GPU-pool job per unit (cllib.b2d_unit: 3 routes, 22.5 GB, 3 CARLA, 12 cores).
Units land where the existing readers look, so pre_report.py (`<arm>@doff`) and plan_track_report.py read them unchanged:
  curv -> $DATA_DIR/runs/op_route_ft/desire_off/<arm>/b2d/turns-s2-k<K>;  p7 / hyb -> $DATA_DIR/runs/op_route_ft/plan_track/<arm>[-hyb]/b2d/turns-s2-k<K>
Pool logs (log.txt, STATUS, DONE / ERROR, pool_id) in <root>/<unit>/. Resumable: finished units are skipped, units still in the pool reused.
  .venv/bin/python experiments/op_route_ft/scripts/near_lane.py [--root $DATA_DIR/runs/op_route_ft/near/cl] \
      [--arms rc-bear-near-s0,rc-bear-fix-s0,rc-ctl-near-s0] [--exec curv,p7] [--stage smoke] [--routes small] [--wait] [--dry-run]
--routes small: the staged first read (user rule 2026-10-05: stop early on a clear negative), 8 routes / 9 of the 25 turns, mixed choice /
forced, tight / wide, and the turns rc-bear-s0 (desire off) entered late: SMALL below; the units go to turns-small-s2-k<K> (not mixed with the
25-turn dirs). --wait blocks until every unit is final and exits 1 if any did not finish.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/op_guard/scripts"))
sys.path.insert(0, str(REPO))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

DATA = G.data_dir() / "runs/op_route_ft"
SMALL = "10255 5423 15102 34183 25051 28180 27994 28008".split()   # rc-bear-s0 desire off: late x 6, crawl x 1, took x 2 (28008 holds 2 turns)


def units(arms, execs, stage="all", small=False):
    out = []
    for ex in execs:
        for arm in arms:
            c = G.resolve(arm)
            base = DATA / "desire_off" / arm if ex == "curv" else DATA / "plan_track" / (arm + ("-hyb" if ex == "hyb" else ""))
            sub = "turns-small" if small else "turns"
            for k, ids in C.shards(SMALL if small else C.TURN_ROUTES, stage):
                u = C.b2d_unit("near-%s-%s-%s-s%d-k%d" % (ex, arm, sub, C.TURN_SEED, k), ids, base / "b2d" / ("%s-s%d-k%d" % (sub, C.TURN_SEED, k)),
                               C.TURN_SEED, True, c.get("onnx"), priority=3 if arm.startswith("rc-bear") else 2, route_adapter=c.get("route_adapter"))
                u["env"]["DESIRE"] = "false"
                if ex != "curv":
                    u["env"]["LAT_EXEC"] = ex
                out.append(u)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(DATA / "near/cl"), help="pool log dirs <root>/<unit>/")
    ap.add_argument("--arms", default="rc-bear-near-s0,rc-bear-fix-s0")
    ap.add_argument("--exec", default="curv")
    ap.add_argument("--stage", choices=("smoke", "all"), default="all")
    ap.add_argument("--routes", choices=("all", "small"), default="all")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    us = units(a.arms.split(","), a.exec.split(","), a.stage, a.routes == "small")
    ids = C.submit_units(us, a.root, "op_route_ft near", dry_run=a.dry_run)
    if a.wait and not a.dry_run:
        bad = C.wait_units(ids)
        print("units not done: %s" % bad if bad else "all units done", flush=True)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
