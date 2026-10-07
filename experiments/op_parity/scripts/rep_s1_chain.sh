#!/usr/bin/env bash
# op_parity representation fix, stage 1 policy pilot (plans/2026-10-07-representation-design.md 6.2): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh rep-s1 experiments/op_parity/scripts/rep_s1_chain.sh MW [MV]); arms after H0 are the ones stage 0 admits (MV only if
# its gate passed). H0 = P2H pilot recipe on navsim/op-parity-s234 (s2-s4, 3000 steps x 64, warmup 100, hinge lambda 10, W frames);
# MW / MV = H0 + pp_train --mem wa_cf / vj21. Every GPU job through the pool, every navtest read through jevdrive.bench.
#   memory banks -> seed 0 (H0, arms) -> navtest (+ :noside = memory off) -> early stop (rep.py s1gate) -> seed 1 (H0 + continuing arms)
#   -> navtest -> rep.py s1report. STOP before stage 2. State: $DATA_DIR/runs/op_parity/rep/s1/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/rep/s1; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
L=$D/pool
ARMS=("$@"); (( ${#ARMS[@]} )) || ARMS=(MW)
DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
SPLIT=navsim/op-parity-s234
declare -A TAG=([H0]=RH0 [MW]=RMW [MV]=RMV) MEM=([H0]="" [MW]=wa_cf [MV]=vj21)
status() { echo "$(date '+%F %T') op_parity rep s1: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
memflag() { [[ -n ${MEM[$1]} ]] && echo "--mem ${MEM[$1]}"; }
train() {  # arm seed
  local arm=$1 s=$2 t=${TAG[$1]}-F-s$2
  local smoke="$PY $S/pp_train.py --arm P2 $(memflag $arm) --frames warp --host --data navtrain_full.s2of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 --hinge-lam 10 --tag smoke-rep"
  sub rep-t-$t $L/t-$t --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 $(memflag $arm) --seed $s --frames warp --host \
      --data $DATA --split $SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 --hinge-lam 10 --tag $t >/dev/null
}
read_navtest() {  # arm seed -> bench navtest of the arm (and its memory-off read), submitted (no wait)
  local t=${TAG[$1]}-F-s$2
  "${B[@]}" run --model $t --bench navtest || die "bench navtest $t"
  [[ $1 != H0 ]] && { "${B[@]}" run --model $t:noside --bench navtest || die "bench navtest $t:noside"; }
  return 0
}
wait_navtest() {  # arm seed
  local t=${TAG[$1]}-F-s$2 specs=(${TAG[$1]}-F-s$2); [[ $1 != H0 ]] && specs+=($t:noside)
  "${B[@]}" status --model "${specs[@]}" --bench navtest --wait || die "navtest $t"
}

# ---------------------------------------------------------------- 0. memory banks (CPU)
status "memory banks: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
  k=${MEM[$arm]}
  if [[ ! -f $DATA_DIR/runs/op_parity/mem/$k/lb_navtest.npy ]]; then
    sub rep-mem-$k $L/mem-$k --vram 0.5 --cpu 4 --ram 48 -- $PY $S/rep.py mem --kind $k >/dev/null; waitdirs $L/mem-$k
  fi
done

# ---------------------------------------------------------------- 1. seed 0
status "seed 0: train H0 ${ARMS[*]}"
for arm in H0 "${ARMS[@]}"; do train $arm 0; done
for arm in H0 "${ARMS[@]}"; do
  waitdirs $L/t-${TAG[$arm]}-F-s0
  $PY $S/pp_full_check.py train --tag ${TAG[$arm]}-F-s0 || die "training sanity ${TAG[$arm]}-F-s0"
  read_navtest $arm 0
done
status "seed 0: navtest"
for arm in H0 "${ARMS[@]}"; do wait_navtest $arm 0; done
KEEP=$($PY $S/rep.py s1gate --arms "${ARMS[@]}"); rc=$?
if (( rc != 0 )); then
  status "EARLY STOP: no arm passes the seed-0 read (see $D/gate-s0.json); seed 1 not run"
  cp "$D/gate-s0.json" "$D/GATE_STOP"
  $PY $S/rep.py s1report --arms "${ARMS[@]}" --seeds 0 --out stage1_s0 > $D/report-s0.txt || die "report"
  exit 0
fi
read -ra KEEP <<< "$KEEP"
status "seed 0 passed for ${KEEP[*]}; seed 1: H0 ${KEEP[*]}"

# ---------------------------------------------------------------- 2. seed 1
for arm in H0 "${KEEP[@]}"; do train $arm 1; done
for arm in H0 "${KEEP[@]}"; do
  waitdirs $L/t-${TAG[$arm]}-F-s1
  $PY $S/pp_full_check.py train --tag ${TAG[$arm]}-F-s1 || die "training sanity ${TAG[$arm]}-F-s1"
  read_navtest $arm 1
done
for arm in H0 "${KEEP[@]}"; do wait_navtest $arm 1; done

# ---------------------------------------------------------------- 3. report (stage 2 needs review: not launched here)
status "report"
$PY $S/rep.py s1report --arms "${KEEP[@]}" --seeds 0 1 > $D/report.txt || die "report"
$PY $S/rep.py s1report --arms "${ARMS[@]}" --seeds 0 --out stage1_s0 > $D/report-s0.txt || die "report s0"
status "done (stage 2 not launched: review)"
date > "$D/DONE"
