#!/usr/bin/env bash
# P1 level (b) lane on WA-JEPA (plans/2026-10-22-p1b-prereg.md): submitted at once, chained after level (a)'s read-out.
#   gate (CPU: level (a) arm off within 0.3 of 91.71 and k512 within 0.1 of off) -> sweepb shards (GPU, one per card)
#   -> mergeb (CPU; its off arm must equal level (a)'s) -> readout (bench scoring, paired vs PROBE-wajepa-ctx-off)
# Usage (on the box, repo root): AFTER=<level (a) readout job id> experiments/wm_surplus/scripts/p1_lane_b.sh
set -euo pipefail
: "${DATA_DIR:?}" "${AFTER:?level (a) readout job id}"
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/wm_surplus/p1_usage
PY=$DATA_DIR/envs/wajepa/bin/python
ME=experiments/wm_surplus/scripts/p1_wajepa_b.py
NS=${NS:-$(nvidia-smi -L | wc -l)}; PRIO=${PRIO:-5}
sub() { local n=$1; shift; .venv/bin/python -m jevdrive.cl submit --owner p1-usage --priority "$PRIO" --log-dir "$O/pool/b-$n" --name "p1b-$n" "$@" | awk '{print $NF}'; }
join() { local IFS=,; echo "$*"; }
G=$(sub gate --after "$AFTER" --vram 0.1 --cpu 1 --ram 2 -- .venv/bin/python -c "
import json, sys
r = json.load(open('$O/results/p1_readout.json')); import pandas as pd
t = pd.read_csv('$O/results/p1_readout.csv').set_index('arm')
off, d512 = r['ref_EPDMS'], float(t.loc['k512', 'd'])
ok = abs(off - 91.71) <= 0.3 and abs(d512) <= 0.1
print(f'level (a): off {off:.2f} (91.71 +- 0.3), k512 - off {d512:+.3f} (+- 0.1) -> ' + ('pass' if ok else 'STOP'))
sys.exit(0 if ok else 1)")
S=(); for i in $(seq 0 $((NS - 1))); do
  S+=($(sub sweep-s$i --after "$G" --vram 45 --cpu 12 --ram 34 -- "$PY" $ME sweep --shard "$i" "$NS"))
done
M=$(sub merge --after "$(join "${S[@]}")" --vram 0.1 --cpu 2 --ram 16 -- "$PY" $ME merge)
R=$(sub readout --after "$M" --vram 0.1 --cpu 1 --ram 8 -- .venv/bin/python experiments/wm_surplus/scripts/p1_readout.py \
  --tag wajepa-b --arms off keep2 keep4 keep8 drop2 drop6 drop12 --ref off --ref-spec PROBE-wajepa-ctx-off \
  --out "$O/results_b" --priority "$PRIO")
printf 'gate %s\nsweep %s\nmerge %s\nreadout %s\n' "$G" "$(join "${S[@]}")" "$M" "$R" | tee "$O/jobs_b.txt"
