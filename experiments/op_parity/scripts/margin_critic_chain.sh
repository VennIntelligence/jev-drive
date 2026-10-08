#!/usr/bin/env bash
# op_parity margin critic (plans/2026-10-08-margin-critic-prereg.md). One self-advancing chain in tmux jev:
#   scripts/tmux_run.sh margin-critic experiments/op_parity/scripts/margin_critic_chain.sh            (TAG=smoke for the smoke)
# Stages: hidden gate -> extract (pool job per shard) + bank (CPU) -> train (pool job per arm) -> select -> report. Every GPU job through the pool.
# State: $DATA_DIR/runs/op_parity/margin_critic/<tag>/chain/{STATUS, DONE, ERROR, log.txt, jobs.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
TAG=${TAG:-full}
O=$DATA_DIR/runs/op_parity/margin_critic/$TAG
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts/margin_critic.py
L=$O/pool
status() { echo "$(date '+%F %T') op_parity margin-critic[$TAG]: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
if [[ $TAG == smoke ]]; then SH="0"; LIM="--limit 600"; ARMS=${ARMS:-"MC MC-E R-WA"}; else SH=$(seq 0 11); LIM=""; ARMS=${ARMS:-"MC MC-30 MC-10 MC-3 MC-V MC-E R-WA R-C1"}; fi

status "hidden gate"
$PY $S hidden --tag $TAG || die hidden

status "extract (pool) + bank"
ext=()
for i in $SH; do sub mc-ext-$TAG-s$i $L/ext-$i --vram 12 --cpu 3 --ram 12 -- $PY $S extract --tag $TAG --shard $i $LIM; ext+=($L/ext-$i); done
$PY $S bank --tag $TAG $LIM || die bank
waitdirs "${ext[@]}"

status "train (pool): $ARMS"
tr=()
for arm in $ARMS; do
  vram=$($PY $S arms | awk -v a="$arm" '$1 == a {print $2}')
  [[ $TAG == smoke ]] && vram=12
  [[ -f $O/pred/$arm.npz ]] || sub mc-train-$TAG-$arm $L/train-$arm --train --vram "$vram" --cpu 4 --ram 48 -- $PY $S train --tag $TAG --arm $arm
  [[ -f $O/pred/$arm.npz ]] || tr+=($L/train-$arm)
done
(( ${#tr[@]} )) && waitdirs "${tr[@]}"

status "select"
$PY $S select --tag $TAG || die select
status "report"
if [[ $TAG == smoke ]]; then $PY $S report --tag $TAG --out $O/report --figs $O/report/figs || die report; else $PY $S report --tag $TAG || die report; fi
status "done"; date > "$D/DONE"
