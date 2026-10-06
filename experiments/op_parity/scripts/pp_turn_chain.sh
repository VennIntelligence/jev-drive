#!/usr/bin/env bash
# op_parity turn-training lane (plans/2026-10-06-turn-train-prereg.md), one self-advancing chain in tmux jev (scripts/tmux_run.sh).
#   train H / T1 / T2 (/ T3 when VERDICT=shrinkage) x 2 seeds at pilot scale (frozen vision, W frames) -> navtest plans + scoring
#   -> gate (pp_turn_report.py gate) -> HUGSIM 64 exam + spec for H and every arm that passes the gate -> reports.
# Every GPU job goes through the pool. State: $DATA_DIR/runs/op_parity/turn/{STATUS, DONE, ERROR, log.txt, jobs.txt, gate.json}; rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/turn; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
HPY=$DATA_DIR/envs/hugsim/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
SH="navtrain_full.s0of12 navtrain_full.s1of12"
SPLIT=navsim/op-parity-full STEPS=${STEPS:-3000} BATCH=${BATCH:-64} LAM=${LAM:-10}
BAL=${BAL:-0.35,0.15,0.25,0.25}
VERDICT=${VERDICT:?set VERDICT (shrinkage|mixed) from experiments/op_probe/results/turn-gain.md}
LIST=experiments/hugsim/scripts/derot_all64.txt
ARMS="HP T1P T2P"; [[ $VERDICT == shrinkage ]] && ARMS="$ARMS T3P"
declare -A XF=([HP]="" [T1P]="--turn-bal $BAL" [T2P]="--turn-bal $BAL --anchor-off-turn" [T3P]="--turn-bal $BAL --anchor-off-turn --late-lat-w 2.0")
status() { echo "$(date '+%F %T') op_parity turn: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

# ---------------------------------------------------------------- 1. training: all arms x seeds, the pool places them on the 3 cards
status "training $ARMS (verdict $VERDICT)"
for a in $ARMS; do for s in 0 1; do
  smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s0of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 --hinge-lam $LAM ${XF[$a]} --tag smoke-$a"
  sub ppT-t-$a-s$s $L/t-$a-s$s --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
      --data $SH --split $SPLIT --steps $STEPS --batch $BATCH --warmup 100 --eval-every 1000 --hinge-lam $LAM ${XF[$a]} --tag $a-F-s$s >/dev/null
done; done
waitdirs $(for a in $ARMS; do for s in 0 1; do echo $L/t-$a-s$s; done; done)
for a in $ARMS; do for s in 0 1; do $PY $S/pp_full_check.py train --tag $a-F-s$s || die "training sanity $a-F-s$s"; done; done

# ---------------------------------------------------------------- 2. navtest plans + scoring
status "navtest plans + scoring"
for a in $ARMS; do
  sub ppT-plans-$a $L/plans-$a --vram 30 --cpu 8 --ram 24 -- $PY $S/pp_eval.py --data lb_navtest --frames warp plans --models $a-F-s0 $a-F-s1 --tag turn >/dev/null
done
for a in $ARMS; do
  waitdirs $L/plans-$a
  sub ppT-score-$a $L/score-$a --vram 1 --cpu 24 --ram 48 --env NAVSIM_THREADS=22 -- $PY $S/pp_eval.py --data lb_navtest --frames warp score --models $a-F-s0 $a-F-s1 >/dev/null
done
waitdirs $(for a in $ARMS; do echo $L/score-$a; done)
T_ARMS=$(echo $ARMS | sed 's/HP //')
$PY $S/pp_turn_report.py gate --arms $T_ARMS || die "gate"
$PY $S/pp_turn_report.py report --arms $T_ARMS || die "navtest report"
status "gate: $(tr -d '\n ' < $D/gate.json | cut -c1-400)"

# ---------------------------------------------------------------- 3. HUGSIM 64 (exam + spec, both seeds) for H and every arm passing the gate
PASS=$($PY - <<PYEOF
import json; g = json.load(open("$D/gate.json")); print(" ".join(a for a, r in g.items() if r["hugsim"]))
PYEOF
)
if [[ -n $PASS ]]; then
  status "HUGSIM for HP $PASS"
  hug() { sub ppT-h$2-$1 $L/h-$2-$1 --vram 40 --cpu 14 --ram 45 --env PRESET=$2 -- bash $S/pp_hugsim.sh arm $1 $LIST 6 >/dev/null; }
  for a in HP $PASS; do for s in 0 1; do for pr in exam spec; do hug $a-F-s$s $pr; done; done; done
  waitdirs $(for a in HP $PASS; do for s in 0 1; do for pr in exam spec; do echo $L/h-$pr-$a-F-s$s; done; done; done)
  export FULL_TAGS="P0,$(for a in HP $PASS; do echo -n "$a-F-s0,$a-F-s1,"; done | sed 's/,$//')" FULL_OUT=$PWD/experiments/op_parity/results/hugsim_turn
  $HPY $S/pp_hugsim_report.py extract full || die "hugsim extract"
  $PY $S/pp_turn_report.py report --arms $T_ARMS || die "report with hugsim"
else
  status "no arm passes the gate (guard + closure >= half): no HUGSIM"
fi
status "done"
date > "$D/DONE"
