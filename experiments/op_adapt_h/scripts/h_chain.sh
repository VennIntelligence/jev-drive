#!/usr/bin/env bash
# op-adapt H: one self-advancing chain for a set of arms on one card (plans/2026-10-04-op-adapt-H-prereg.md).
#   1 train every arm (two at a time on the card)          4 navtest PDMS (op_adapt_l_readout navtest, port O alongside)
#   2 dev metrics of O, probe (a) of O + every arm, tables  5 WOD val capture (op_adapt_l_readout eval / read)
#   3 link the runs into op_adapt_L/runs/H-<tag>            6 navhard two-stage EPDMS        7 HUGSIM spin set (h_hugsim.sh)
# Training waits for the train banks ($H/bank/<dom>/var.npz). Every step is skipped when its output exists, so re-running the same command resumes. Steps 6-7 only with EXAMS=1.
# Usage (box, tmux):  GPU=0 CPUS=0-49 SCPUS=100-149 [EXAMS=1] experiments/op_adapt_h/scripts/h_chain.sh <chain name> <arm> [<arm> ...]
# Files: $DATA_DIR/runs/op_adapt_H/chain/<name>/{STATUS,DONE,ERROR,log.txt}
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${CPUS:?}" "${SCPUS:?}"
cd "$(dirname "$0")/../../.."
NAME=$1; shift
ARMS=("$@")
H=$DATA_DIR/runs/op_adapt_H
LR=$DATA_DIR/runs/op_adapt_L
C=$H/chain/$NAME
mkdir -p "$C"
rm -f "$C/DONE" "$C/ERROR"
PY=$DATA_DIR/envs/op-train/bin/python
T=experiments/op_adapt_h/scripts/h_train.py
RO=experiments/op_adapt_l/scripts/op_adapt_l_readout.py
log() { echo "$(date '+%F %T') $*" | tee -a "$C/log.txt"; }
st() { echo "$(date '+%F %T') $*" > "$C/STATUS"; log "$*"; }
fail() { log "ERROR $*"; echo "$(date '+%F %T') $*" > "$C/ERROR"; exit 1; }
g() { CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" "$@" >> "$C/log.txt" 2>&1; }
TAGS=(); for a in "${ARMS[@]}"; do TAGS+=("$a-s0"); done

until [[ -f $H/bank/nav/var.npz && -f $H/bank/wod/var.npz && -f $H/bank/carla/var.npz ]]; do sleep 30; done
st "1 train ${ARMS[*]}"
pids=()
for a in "${ARMS[@]}"; do
  [[ -f $H/runs/$a-s0/ckpt-final.pt ]] && continue
  g $PY $T train --arm "$a" --no-eval & pids+=($!)
  if (( ${#pids[@]} >= 2 )); then wait "${pids[0]}" || fail "train"; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait "$p" || fail "train"; done
for a in "${ARMS[@]}"; do [[ -f $H/runs/$a-s0/ckpt-final.pt ]] || fail "no checkpoint for $a"; done

st "2 dev, probe (waits for the fixed trunk banks)"
until [[ -f $H/fixbank/dev_carla/jobs.json ]]; do sleep 30; done
g $PY $T dev --models O "${TAGS[@]}" || fail "dev"
g $PY $T probe --models O "${TAGS[@]}" || fail "probe"
for t in "${TAGS[@]}"; do g $PY $T probe-table --model "$t" || fail "probe table $t"; done
if (( ${#TAGS[@]} >= 2 )); then g $PY $T probe-table --model "${TAGS[0]}" --ref "${TAGS[1]}" || fail "probe table pair"; fi

st "3 link"
g $PY $T link --models "${TAGS[@]}" || fail "link"
HT=(); for t in "${TAGS[@]}"; do HT+=("H-$t"); done

st "4 navtest"
todo=(); for t in "${HT[@]}"; do [[ -f $LR/readout/$t/navtest.json ]] || todo+=("$t"); done
(( ${#todo[@]} )) && { g $PY $RO navtest --models "${todo[@]}" --cpus "$SCPUS" || fail "navtest"; }

st "5 WOD val capture"
for t in "${HT[@]}"; do
  [[ -f $LR/readout/$t/summary.json ]] && continue
  g $PY $RO eval --model "$t" || fail "eval $t"
  g $PY $RO read --model "$t" || fail "read $t"
done

if [[ ${EXAMS:-0} == 1 ]]; then
  st "6 navhard"
  todo=(); for t in "${TAGS[@]}"; do [[ -f $H/readout/$t/navhard.json ]] || todo+=("$t"); done
  (( ${#todo[@]} )) && { g $PY $T navhard --models "${todo[@]}" --cpus "$SCPUS" || fail "navhard"; }
  st "7 HUGSIM"
  for t in "${TAGS[@]}"; do
    [[ -f $H/hugsim/$t/DONE ]] && continue
    GPU=$GPU experiments/op_adapt_h/scripts/h_hugsim.sh "$t" >> "$C/log.txt" 2>&1 || fail "hugsim $t"
  done
fi
st "done"
touch "$C/DONE"
