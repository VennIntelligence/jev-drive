#!/usr/bin/env bash
# The TFv6 rules x interface batch (todos/2026-09-25-tfv6-rules-interface/README.md, "估计与分批"), in tier order:
# T1 pairs (A1, A0, B1, B0) -> T2 A1 rest of B2D-209 -> T3 A1 repeat -> T4 A0, B1, B0 rest. Stop it when the
# slot's window closes; whatever finished is reported. Launched through scripts/slot_run.sh (slot tfv6-exp).
# Usage: scripts/tfv6_rules_batch.sh <workers> <gpu>
# Every step gets its own 10-wide server-index block inside 200-319, so RPC ports (12000-17950) and TM ports
# (18000-23950) never overlap; b2d_run moves a worker by +workers on every server restart, and a traffic-manager
# port outlives its server (docs/carla.md).
set -uo pipefail
cd "$(dirname "$0")/.."
w=$1 gpu=$2
R=$DATA_DIR/runs/tfv6_rules
# the smoke (one worker, jev:tfr-smoke) must be finished so the slot's worker cap holds
for _ in $(seq 240); do [[ -f $R/smoke/B0/done/2751520.json ]] && break; sleep 15; done
t1=$(cat $R/ids_t1.txt) rest=$(cat $R/ids_rest.txt) all=$(cat $R/ids_all.txt)
steps=("A on $R/A1.rep0 $t1" "A off $R/A0.rep0 $t1" "B on $R/B1.rep0 $t1" "B off $R/B0.rep0 $t1"
       "A on $R/A1.rep0 $rest" "A on $R/A1.rep1 $all"
       "A off $R/A0.rep0 $rest" "B on $R/B1.rep0 $rest" "B off $R/B0.rep0 $rest")
i=0
for s in "${steps[@]}"; do
  read -r arm rules out ids <<<"$s"
  echo "$(date +%F\ %T) step $i: $arm $rules -> $out"
  scripts/tfv6_rules_run.sh $arm $rules $out $w $((200 + 10 * i)) $gpu $ids || echo "step $i exited $?"
  i=$((i + 1))
done
echo "$(date +%F\ %T) batch done"
