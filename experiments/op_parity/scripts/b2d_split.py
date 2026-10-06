"""Register the B2D P2 train / val split by route: b2d/b2dc-v2-train and b2d/b2dc-v2-val (jevdrive.data.splits).

Val = about 5% of the 998 collected routes (b2dc-train@v2 clips with a complete recording), stratified by scenario type: every type gives
max(1, round(0.05 n)) routes, picked at evenly spaced positions of the type's routes ordered by (turn side, hashed route id), so left / right /
no-turn routes of a type are all represented. Deterministic (no rng state).

  python experiments/op_parity/scripts/b2d_split.py <index.csv>      # index.csv of $DATA_DIR/runs/b2d_collect/data/all
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive.data import splits  # noqa: E402


def pick_val(rows, frac=0.05):
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    val = []
    for t, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: (r["turn"], hashlib.sha256(r["route_id"].encode()).hexdigest()))
        k = max(1, round(frac * len(rs)))
        val += [rs[int((j + 0.5) * len(rs) / k)]["route_id"] for j in range(k)]
    return sorted(val)


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    allr = sorted(r["route_id"] for r in rows)
    val = pick_val(rows)
    train = sorted(set(allr) - set(val))
    src = ("experiments/op_parity/scripts/b2d_split.py on $DATA_DIR/runs/b2d_collect/data/all/index.csv: the %d routes with a complete clip of "
           "b2d/b2dc-train@v2; val = stratified by scenario type, max(1, round(5%% n)) per type" % len(allr))
    sv = splits.define("b2d", "b2dc-v2-val", val, "route", src, used_by="op_parity b2d P2 (b2d_prep.py, pp_train.py)", status="frozen")
    st = splits.define("b2d", "b2dc-v2-train", train, "route", src, used_by="op_parity b2d P2 (b2d_prep.py, pp_train.py)", status="frozen")
    for ev in ("b2d/bench2drive220", "b2d/bench2drive-0.0.4-val"):
        for x in (st, sv):
            splits.check_disjoint(x, splits.load(ev))
    splits.check_disjoint(st, sv)
    types = {r["route_id"]: r for r in rows}
    n_turn = sum(types[v]["turn"] != "" for v in val)
    print(f"train {len(train)} val {len(val)} ({n_turn} turn routes: left {sum(types[v]['turn'] == 'left' for v in val)}, "
          f"right {sum(types[v]['turn'] == 'right' for v in val)}); {sv.id} {st.id}")
