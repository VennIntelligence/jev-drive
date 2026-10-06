#!/usr/bin/env bash
# op_parity navhard readout under protocol G (GIMM frames; results/navhard.md "Protocol G"): P0 (harness check vs op_guard `shipped`)
# and the full-run P2 checkpoints. The G front tokens of lb_navhard are already cached by pp_navhard_chain.sh's first prep call.
# State: $DATA_DIR/runs/op_parity/navhard_gimm/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/navhard_gimm; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
NAV2=$DATA_DIR/envs/navsim2/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
MODELS="P0 P2-F-s0 P2-F-s1"
status() { echo "$(date '+%F %T') op_parity navhard_gimm: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
aft() { [[ $1 == done ]] && echo "" || echo "--after $1"; }

status "plans"
j_plans=$(sub ppG-plans $L/plans --gpus 1,2 --vram 30 --cpu 8 --ram 24 -- \
  $PY $S/pp_eval.py --data lb_navhard --frames gimm plans --models $MODELS --tag navhard_gimm)
stems=$(for m in $MODELS; do echo -n "gimm@cinque_PP$m "; done)
j_exp=$(sub ppG-export $L/export --gpus 1,2 --vram 1 --cpu 4 --ram 16 --env OPI_ROOT=op_lb $(aft $j_plans) -- \
  $JEV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navhard --adapters base --plans $stems)
for m in $MODELS; do
  sub ppG-h-$m $L/h-$m --gpus 1,2 --vram 1 --cpu 10 --ram 24 $(aft $j_exp) -- \
    $NAV2 experiments/op_guard/scripts/nav_harness.py --poses $DATA_DIR/runs/op_lb/lb_navhard/preds/gimm-cinque_PP${m}__base.npz \
    --out $D/harness/$m --procs 10 >/dev/null
done
waitdirs $L/plans $L/export $(for m in $MODELS; do echo $L/h-$m; done)
status "scoring done; report"
git pull -q 2>/dev/null; $PY $S/pp_navhard.py report-gimm || die "report"
status "done"
date > "$D/DONE"
