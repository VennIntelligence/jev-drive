"""CARLA half of the image-command test, step 1 (envs/carla, CPU, no server): the recording plan.

For every lane-D traversal with a map alternative (op_common_cause/results/carla_traversals.json, 179 routes, one
traversal each), re-trace the route as pair_inv_carla does (GlobalRoutePlanner at 1 m) and keep the junction entry
(last waypoint before the junction), the exit (first after it) and the connecting road / lane the route takes.
Writes $DATA_DIR/runs/op_img_cmd/carla/plan/{routes.xml, stops.json, agent_config.json}: the routes (both B2D route files merged, ordered
round-robin over the taken class so any prefix is mixed) and per route the CARLA-world entry / exit points the
recording agent stops on (img_carla_agent.py); the agent config is P4's with cameras at 5 Hz (sensor_tick 0.2).

  $DATA_DIR/envs/carla/bin/python experiments/op_img_cmd/scripts/img_carla_prep.py --workers 16
"""
import argparse, json, os, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "op_common_cause" / "scripts"))
import pair_inv_carla as PI  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
OUT = DATA / "runs" / "op_img_cmd" / "carla" / "plan"


def trace(job):
    import carla
    t, pts = job
    grp = PI.get_grp(t["town"])
    locs = [carla.Location(*p) for p in pts]
    wps = [w for a, b in zip(locs[:-1], locs[1:]) for w, _ in grp.trace_route(a, b)]
    i = t["i_entry"] + 1
    rj = PI.TAB[t["town"]][0]
    assert rj.get(wps[i].road_id, -1) == t["junction"], (t["route"], "entry index does not match")
    j = i
    while rj.get(wps[j + 1].road_id, -1) != -1:
        j += 1
    xy = lambda w: [w.transform.location.x, w.transform.location.y]  # noqa: E731
    return t["route"], dict(entry=xy(wps[i - 1]), exit=xy(wps[j + 1]), conn=[wps[i].road_id, wps[i].lane_id],
                            town=t["town"], junction=t["junction"], taken=t["taken"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    T = json.load(open(REPO / "experiments" / "op_common_cause" / "results" / "carla_traversals.json"))
    T = [t for t in T if {b[0] for b in t["branches"]} - {t["taken"]}]
    els = {}
    for x in PI.ROUTE_XMLS:
        for e in ET.parse(PI.ROUTE_DIR + x).getroot().iter("route"):
            els.setdefault(e.get("id"), e)
    pts = {rid: [(float(p.get("x")), float(p.get("y")), float(p.get("z"))) for p in e.find("waypoints")] for rid, e in els.items()}
    for town in {t["town"] for t in T}:
        PI.TAB[town] = PI.xodr_tables(town)
    T.sort(key=lambda t: t["town"])
    with Pool(a.workers) as pool:
        stops = dict(pool.map(trace, [(t, pts[t["route"]]) for t in T], chunksize=2))
    by = {c: sorted([t for t in T if t["taken"] == c], key=lambda t: t["route"]) for c in ("straight", "right", "left")}
    order = []
    while any(by.values()):
        for c in by:
            if by[c]:
                order.append(by[c].pop(0)["route"])
    root = ET.Element("routes")
    for rid in order:
        root.append(els[rid])
    OUT.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUT / "routes.xml")
    (OUT / "stops.json").write_text(json.dumps({r: stops[r] for r in order}, indent=0))
    cfg = json.load(open(REPO / "experiments" / "prediag" / "results" / "p4-carla-gap" / "agent_config.json"))
    cfg.update(sensor_tick=0.2, max_sim_s=150.0, stuck_s=60.0, stops=str(OUT / "stops.json"))
    (OUT / "agent_config.json").write_text(json.dumps(cfg, indent=0))
    print(len(order), "routes;", order[:12])


if __name__ == "__main__":
    main()
