"""Register the route-grouped 5-fold split of the stop-position probe (plan 2026-10-03-stoppos-probe.md, section Split).

  .venv/bin/python experiments/vlm_arb/scripts/stoppos_split.py     (reads results/stoppos_routes.csv, written by stoppos_labels.py build)

Members are route ids prefixed by their set (`cl-9196`, `p4-10857`). Folds are stratified by (set, junction kind) with seed 0:
each stratum is shuffled and dealt round-robin. `b2d/stoppos-cv5-fold<k>` holds the test routes of fold k (train = every other
route); `b2d/stoppos-all` holds every route. Idempotent.
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive.data import splits  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "results/stoppos_routes.csv"
ORIGIN = "results/stoppos_routes.csv (stoppos_labels.py build, 2026-10-03), strata (set, kind), seed 0, round-robin over shuffled strata; written before any feature or probe number"


def folds(rows, k=5, seed=0):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["src"], r["kind"])].append(r["id"])
    rng, out, i = random.Random(seed), [[] for _ in range(k)], 0
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for x in ids:
            out[i % k].append(x)
            i += 1
    return out


def main():
    rows = list(csv.DictReader(open(CSV)))
    fs = folds(rows)
    for j, f in enumerate(fs):
        s = splits.define("b2d", f"stoppos-cv5-fold{j}", f, unit="route", origin=ORIGIN, status="frozen", used_by="experiments/vlm_arb stoppos",
                          notes="test routes of fold %d; the training routes are all other routes of b2d/stoppos-all" % j)
        print(s.id, len(s))
    s = splits.define("b2d", "stoppos-all", [r["id"] for r in rows], unit="route", origin=ORIGIN, status="frozen", used_by="experiments/vlm_arb stoppos")
    print(s.id, len(s))
    splits.check_disjoint(*[splits.load(f"b2d/stoppos-cv5-fold{j}") for j in range(5)])


if __name__ == "__main__":
    main()
