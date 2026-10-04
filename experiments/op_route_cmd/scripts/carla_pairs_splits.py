"""Register the junction-level splits of the CARLA route-pair material (pure Python, run anywhere with the repo).

  python carla_pairs_splits.py --junctions <packed>/junctions.json --holdout <carla_topo run>/holdout.json --tag s2000
  -> jevdrive/data/splits/defs/b2d/route-carla-{train,dev,heldout}.v1.json (unit "junction", members "Town12:J1234")

train / dev: the junctions that have poses in the packed material (dev = 10% of the sampled junctions by hash). heldout: every junction that any
Bench2Drive route of the two route files crosses (the strict hold-out of carla_pairs_plan.py); the material has none of them.
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.data import splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--junctions", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    J = json.load(open(a.junctions))
    held = sorted("%s:J%d" % (t, j) for t, j in json.load(open(a.holdout))["b2d_any"])
    assert not (set(J["train"]) | set(J["dev"])) & set(held), "material touches a held-out junction"
    src = "op_route_cmd CARLA route pairs (%s): approach roads sampled from the Bench2Drive towns' junctions (carla_pairs_plan.py)" % a.tag
    kw = dict(unit="junction", used_by=["op_route_cmd"], status="frozen")
    tr = splits.define("b2d", "route-carla-train", J["train"], origin=src + "; train junctions with poses", notes="junction-level, 90% of sampled junctions", **kw)
    dv = splits.define("b2d", "route-carla-dev", J["dev"], origin=src + "; dev junctions with poses", notes="junction-level, 10% of sampled junctions by sha1 hash", **kw)
    ho = splits.define("b2d", "route-carla-heldout", held, origin="every junction (town, carla Junction.id) that any Bench2Drive route of the two route files crosses "
                       "(experiments/op_common_cause/results/carla_traversals.json, 226 traversals); never used for fine-tune material", 
                       notes="strict hold-out: neighbouring entries of a held-out junction cannot leak", **kw)
    splits.check_disjoint(tr, dv)
    print(tr.id, dv.id, ho.id, len(J["train"]), len(J["dev"]), len(held))


if __name__ == "__main__":
    main()
