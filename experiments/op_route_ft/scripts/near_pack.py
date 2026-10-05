"""rc-*-near: pack the rendered near poses (experiments/op_route_cmd/scripts/carla_pairs_pack.py unchanged) and add the near columns.

  python near_pack.py --plan <poses.pkl> --render <render dir> --root <packed root> [--register]
  route.npz += dense (rows, 80, 2) f32 / dmask (rows, 80): the lane-centre path at 1 m in the t0 ego frame (near_plan.polyline_dense), taken (bool),
               where (app / in); tab.npz += where, n_exits_road.
  --register: junction-level splits b2d/route-carla-near-{train,dev} (same sha1 dev rule as route-carla-*), checked disjoint from each other and
              from b2d/route-carla-heldout (the 172 B2D junctions).
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_route_cmd/scripts")]
import carla_pairs_pack as PK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    sys.argv = [sys.argv[0], "--plan", a.plan, "--render", a.render, "--root", a.root]
    PK.main()
    root = Path(a.root)
    P = [p for p in pickle.load(open(a.plan, "rb")) if (Path(a.render) / "frames" / p["town"] / (p["id"] + ".npz")).exists()]
    E = [(p, e) for p in P for e in p["exits"]]
    with np.load(root / "route.npz", allow_pickle=True) as z:
        route = {k: z[k] for k in z.files}
    assert len(route["id"]) == len(E) and all(r == "%s-x%d" % (p["id"], e["index_from_left"]) for r, (p, e) in zip(route["id"], E))
    route.update(dense=np.stack([e["dense"] for _, e in E]).astype(np.float32), dmask=np.stack([e["dmask"] for _, e in E]),
                 taken=np.array([bool(e["taken"]) for _, e in E]), where=np.array([p["where"] for p, _ in E]))
    np.savez(root / "route.npz", **route)
    tp = root / "samples/route_carla/tab.npz"
    with np.load(tp, allow_pickle=True) as z:
        tab = {k: z[k] for k in z.files}
    assert (tab["id"] == np.array([p["id"] for p in P])).all()
    tab.update(where=np.array([p["where"] for p in P]), n_exits_road=np.array([p["n_exits_road"] for p in P], np.int16))
    np.savez(tp, **tab)
    print("near columns added:", len(E), "rows,", len(P), "poses")
    if a.register:
        from jevdrive.data import splits
        J = json.load(open(root / "junctions.json"))
        held = set(splits.load("b2d/route-carla-heldout").members)
        assert not (set(J["train"]) | set(J["dev"])) & held, "near material touches a held-out junction"
        src = "op_route_ft near CARLA poses (near_plan.py: 0-10 m before / 0.5-6 m into the junction, 0-3 m/s), junctions of the Bench2Drive towns minus route-carla-heldout"
        kw = dict(unit="junction", used_by=["op_route_ft"], status="frozen")
        tr = splits.define("b2d", "route-carla-near-train", J["train"], origin=src, notes="junction-level, sha1 dev rule of carla_pairs_plan.py", **kw)
        dv = splits.define("b2d", "route-carla-near-dev", J["dev"], origin=src, notes="junction-level, 10% of sampled junctions by sha1 hash", **kw)
        splits.check_disjoint(tr, dv)
        print(tr.id, dv.id)


if __name__ == "__main__":
    main()
