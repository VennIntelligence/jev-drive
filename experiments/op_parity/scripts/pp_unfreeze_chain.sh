#!/usr/bin/env bash
# op_parity vision-unfreeze PILOT (plans/2026-10-06-unfreeze-prereg.md), one-shot chain in tmux jev (scripts/tmux_run.sh).
#   caches (x4: U1 / U1L, vh140: V) on shards s0, s1 -> 5 variants x 2 seeds (3000 steps, batch 64) -> navtest plans from pixels (W; V on
#   the 1.40 m virtual camera) with the equivalence checks -> v2 EPDMS scoring (CPU) -> report + gate (pp_unfreeze_report.py).
# Cards: GPUS (default 2) until the navhard chain writes navhard/GPU_DONE; the U2 runs (online rendering, the heavy ones) may then use any
# card (GPUS_LATE). State: $DATA_DIR/runs/op_parity/unfreeze/{STATUS, DONE, ERROR, log.txt, jobs.txt}; rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/unfreeze; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$DATA_DIR/runs/op_parity/pool/unfreeze
G=${GPUS:-2} GL=${GPUS_LATE:-0,1,2}
FREE=$DATA_DIR/runs/op_parity/navhard/GPU_DONE
SH="navtrain_full.s0of12 navtrain_full.s1of12"
STEPS=${STEPS:-3000} BATCH=${BATCH:-64}
status() { echo "$(date '+%F %T') op_parity unfreeze: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

# ---------------------------------------------------------------- 1. caches
status "caches"
for s in 0 1; do
  for w in x4 vh140; do
    sub ppU-$w-s$s $L/$w-s$s --gpus $G --vram 20 --cpu 20 --ram 60 -- $PY $S/pp_unfreeze.py prep --data navtrain_full.s${s}of12 --what $w --workers 18 >/dev/null
  done
done

# ---------------------------------------------------------------- 2. training: F / U1 / U1L / V on card G as their caches land, U2 anywhere once navhard frees its cards
status "training"
for v in F U1 U1L V U2; do
  for s in 0 1; do
    case $v in
      U1|U1L) when="--when-exists $L/x4-s1/DONE" gp="--gpus $G" vr=24 cpu=6 ;;
      V) when="--when-exists $L/vh140-s1/DONE" gp="--gpus $G" vr=20 cpu=6 ;;
      F) when="" gp="--gpus $G" vr=20 cpu=6 ;;
      U2) when="--when-exists $FREE" gp="--gpus $GL" vr=60 cpu=${U2_CPU:-28} ;;
    esac
    pf=(); [[ $s == 0 ]] && pf=(--preflight "$PY $S/pp_unfreeze.py train --var $v --steps 3 --batch 8 --eval-every 3 --data navtrain_full.s0of12 --workers 8 --tag smoke-$v")
    sub ppU-t-$v-s$s $L/t-$v-s$s --train --vram $vr --cpu $cpu --ram 40 $gp $when "${pf[@]}" -- \
      $PY $S/pp_unfreeze.py train --var $v --seed $s --steps $STEPS --batch $BATCH --data $SH --workers $((cpu - 2)) --tag UF-$v-s$s >/dev/null
  done
done
waitdirs $(for v in F U1 U1L V U2; do for s in 0 1; do echo $L/t-$v-s$s; done; done)

# ---------------------------------------------------------------- 3. navtest plans from pixels (+ equivalence), scoring
status "navtest plans + scoring"
W_MODELS="P0 P2-F-s0 $(for v in F U1 U1L U2; do for s in 0 1; do echo -n "UF-$v-s$s "; done; done)"
V_MODELS="P0 UF-V-s0 UF-V-s1"
sub ppU-plans-w $L/plans-w --gpus $GL --vram 40 --cpu 24 --ram 60 -- $PY $S/pp_unfreeze.py plans --frames warp --models $W_MODELS --workers 22 >/dev/null
sub ppU-plans-v $L/plans-v --gpus $GL --vram 30 --cpu 20 --ram 60 -- $PY $S/pp_unfreeze.py plans --frames vh140 --models $V_MODELS --workers 18 >/dev/null
waitdirs $L/plans-w
$PY $S/pp_unfreeze_report.py equiv || die "pixel-path equivalence check failed (no scoring)"
SC_W=$(echo $W_MODELS | tr ' ' '\n' | grep -v -x 'P0\|P2-F-s0' | tr '\n' ' ')
sub ppU-score-w $L/score-w --gpus $GL --vram 1 --cpu 24 --ram 48 --env NAVSIM_THREADS=22 -- $PY $S/pp_eval.py --data lb_navtest --frames warp score --models $SC_W >/dev/null
waitdirs $L/plans-v
sub ppU-score-v $L/score-v --gpus $GL --vram 1 --cpu 16 --ram 32 --env NAVSIM_THREADS=14 -- $PY $S/pp_eval.py --data lb_navtest --frames vh140 score --models $V_MODELS >/dev/null
waitdirs $L/score-w $L/score-v
$PY $S/pp_unfreeze_report.py pilot || die "report"
status "done"
date > "$D/DONE"
