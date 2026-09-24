#!/usr/bin/env bash
# One lane of the TFv6 rules x interface batch (todos/2026-09-25-tfv6-rules-interface/README.md, "估计与分批").
# Usage: scripts/tfv6_rules_batch.sh <lane X|Y> <workers> <gpu>
# Two lanes run side by side; steps that share an out dir split its routes through b2d_run's claims.
# Every step gets its own server-index block (a traffic-manager port outlives its server, docs/carla.md).
set -uo pipefail
cd "$(dirname "$0")/.."
lane=$1 w=$2 gpu=$3
R=$DATA_DIR/runs/tfv6_rules
t1=$(cat $R/ids_t1.txt) rest=$(cat $R/ids_rest.txt) all=$(cat $R/ids_all.txt)
if [[ $lane == X ]]; then base=120; steps=("A on $R/A1.rep0 $t1" "A off $R/A0.rep0 $t1" "A on $R/A1.rep0 $rest"
  "A on $R/A1.rep1 $all" "A off $R/A0.rep0 $rest" "B on $R/B1.rep0 $rest")
else base=170; steps=("B on $R/B1.rep0 $t1" "B off $R/B0.rep0 $t1" "A on $R/A1.rep0 $rest"
  "A on $R/A1.rep1 $all" "B off $R/B0.rep0 $rest" "B on $R/B1.rep0 $rest"); fi
i=0
for s in "${steps[@]}"; do
  read -r arm rules out ids <<<"$s"
  echo "$(date +%F\ %T) lane $lane step $i: $arm $rules -> $out"
  scripts/tfv6_rules_run.sh $arm $rules $out $w $((base + 8 * i)) $gpu $ids || echo "step $i exited $?"
  i=$((i + 1))
done
echo "$(date +%F\ %T) lane $lane done"
