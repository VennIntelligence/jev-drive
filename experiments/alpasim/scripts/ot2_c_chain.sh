#!/usr/bin/env bash
# Lane OT2 piece C (plans/2026-10-09-ot2-ensemble-prereg.md, decision 211): the adapter ensemble driver, closed loop on the fixed 700 scenes.
#   scripts/tmux_run.sh ot2-c experiments/alpasim/scripts/ot2_c_chain.sh <label> <tag+tag> <single-driver label of the first tag in the c0b / b manifest>
#   e.g.  ot2_c_chain.sh ENS-OT30 OT30-F-s0+OT30-F-s1 OT30-F-s0
# Gates before the full run: the one-member ensemble reproduces its single driver scene for scene on the 8 pilot scenes (deterministic simulator),
# the two-member driver is healthy on them; single-step latency (same process, same card). Then 700 scenes in three chunk jobs.
# State: $DATA_DIR/runs/alpasim/ot2/chain-c-<label>/{STATUS, DONE, ERROR, log.txt}; runs under .../ot2/c-<label>{,-pilot}/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
LABEL=$1 TAGS=$2 REF=$3
O=$DATA_DIR/runs/alpasim/ot2
D=$O/chain-c-$LABEL; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
S=experiments/alpasim/scripts
status() { echo "$(date '+%F %T') ot2-c $LABEL: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
status "latency job; pilot: one member and the ensemble on 8 scenes"
LB=$D/bench
[[ -f $LB/DONE ]] || $VPY -m jevdrive.cl submit --owner alpasim-ot2 --priority 12 --name ot2-ens-bench --vram 8 --cpu 10 --ram 12 --log-dir "$LB" -- \
    "$DATA_DIR/envs/op-train/bin/python" experiments/alpasim/lib/ens_driver.py bench "$TAGS" 100 > /dev/null || die "submit bench"
python3 $S/ot2_loop.py "c-$LABEL-pilot" "ONE:ens:ENS_TAGS=${TAGS%%+*}::pilot8" "$LABEL:ens:ENS_TAGS=$TAGS::pilot8" || die "pilot"
$VPY - "$O/c-$LABEL-pilot/manifest.json" "$REF" "$O" <<'PYEOF' || die "identity gate: the one-member ensemble does not reproduce its single driver"
import json, sys
from pathlib import Path
man, ref, O = json.load(open(sys.argv[1])), sys.argv[2], Path(sys.argv[3])
single = {}
for f in (O.parent / "c0b/manifest.json", O / "b/manifest.json"):
    if f.exists():
        for d in json.load(open(f)).get(ref, []):
            single |= {r["clipgt_id"]: r["score"] for r in json.load(open(Path(d) / "aggregate/results-summary.json"))["rollouts"]}
one = {r["clipgt_id"]: r["score"] for d in man["ONE"] for r in json.load(open(Path(d) / "aggregate/results-summary.json"))["rollouts"]}
bad = [s for s in one if s not in single or abs(one[s] - single[s]) > 1e-9]
print(f"identity gate: {len(one)} pilot scenes, {len(bad)} differ from {ref}: {[(s, one[s], single.get(s)) for s in bad]}")
sys.exit(1 if bad or not one else 0)
PYEOF
until [[ -f $LB/DONE || -f $LB/ERROR ]]; do sleep 20; done
[[ -f $LB/ERROR ]] && die "latency job failed"
sed -n '/^{/,/^}/p' $LB/log.txt | tee "$D/latency.json"
status "700 scenes"
python3 $S/ot2_loop.py "c-$LABEL" "$LABEL:ens:ENS_TAGS=$TAGS" || die "full run"
status "done"; date > "$D/DONE"
