#!/usr/bin/env bash
# op-adapt H round 2 (plans/2026-10-04-launch-pairs-prereg.md): one self-advancing, resumable chain on one card.
#   0 launch samples lwod / lcarla      4 train the arms (two at a time)           8 report (a) (b) + launch dev
#   1 teachers of lwod / lcarla         5 launch dev of the arms                   9 h_chain.sh (dev, 10 deg/s probe, navtest, WOD capture)
#   2 bank2 on all five domains         6 (b) small-rate probe                    10 (d)(f) HUGSIM all 64, only for arms whose (a) ratio <= GATE
#   3 launch dev fixbanks + O           7 (a) ONNX + lean_probe replay / local
# Usage (box, tmux): GPU=2 CPUS=150-199 SCPUS=150-199 [GATE=0.75] [BANK3=1 for arms with M rows] experiments/op_adapt_h/scripts/h_lchain.sh <name> <arm> [<arm> ...]
# Files: $DATA_DIR/runs/op_adapt_H/chain/<name>/{STATUS,DONE,ERROR,log.txt}; report in $H/report/<name>/.
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${CPUS:?}" "${SCPUS:?}"
cd "$(dirname "$0")/../../.."
NAME=$1; shift
ARMS=("$@")
H=$DATA_DIR/runs/op_adapt_H
C=$H/chain/$NAME
mkdir -p "$C" "$H/report/$NAME"
rm -f "$C/DONE" "$C/ERROR"
PY=$DATA_DIR/envs/op-train/bin/python
OPY=$DATA_DIR/envs/openpilot/bin/python
S=experiments/op_adapt_h/scripts
T=$S/h_train.py
log() { echo "$(date '+%F %T') $*" | tee -a "$C/log.txt"; }
st() { echo "$(date '+%F %T') $*" > "$C/STATUS"; log "$*"; }
fail() { log "ERROR $*"; echo "$(date '+%F %T') $*" > "$C/ERROR"; exit 1; }
g() { CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" "$@" >> "$C/log.txt" 2>&1; }
TAGS=(); for a in "${ARMS[@]}"; do TAGS+=("$a-s0"); done
DOMS=(nav wod carla lwod lcarla)

st "0 launch samples"
for d in lwod lcarla; do
  [[ -f $H/samples/$d/tab.npz ]] || { CUDA_VISIBLE_DEVICES= taskset -c "$CPUS" $PY $S/h_prep.py $d >> "$C/log.txt" 2>&1 || fail "prep $d"; }
done
st "1 teachers"
g $PY $T teacher --domains lwod lcarla || fail "teacher"
st "2 bank2"
g $PY $T bank --name bank2 --domains "${DOMS[@]}" --workers 40 || fail "bank2"
if [[ ${BANK3:-0} == 1 ]]; then g $PY $T bank --name bank3 --domains nav wod carla --workers 40 || fail "bank3"; fi
st "3 launch dev of O"
[[ -f $H/dev/O_ldev.json ]] || { g $PY $T ldev --models O || fail "ldev O"; }

st "4 train ${ARMS[*]}"
pids=()
for a in "${ARMS[@]}"; do
  [[ -f $H/runs/$a-s0/ckpt-final.pt ]] && continue
  g $PY $T train --arm "$a" --no-eval & pids+=($!)
  if (( ${#pids[@]} >= 2 )); then wait "${pids[0]}" || fail "train"; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait "$p" || fail "train"; done

st "5 launch dev"
todo=(); for t in "${TAGS[@]}"; do [[ -f $H/runs/$t/ldev.json ]] || todo+=("$t"); done
(( ${#todo[@]} )) && { g $PY $T ldev --models "${todo[@]}" || fail "ldev"; }

st "6 (b) small-rate probe"
g $PY $S/h_rate_probe.py run --models O "${TAGS[@]}" || fail "rate probe"
g $PY $S/h_rate_probe.py table --models O it_dw3-s0 "${TAGS[@]}" --suffix _launch || fail "rate table"

st "7 (a) ONNX, replay, local"
for t in "${TAGS[@]}"; do
  O=$H/launch/$t; mkdir -p "$O"
  if [[ ! -f $H/onnx/$t.onnx ]]; then
    g $PY experiments/op_adapt_l/scripts/op_l_onnx.py build --ckpt "$H/runs/$t/ckpt-final.pt" --out "$H/onnx/$t.onnx" --no-adapter || fail "onnx $t"
  fi
  if [[ ! -f $O/onnx_check.txt ]]; then
    g $PY experiments/op_adapt_l/scripts/op_l_onnx.py ref --ckpt "$H/runs/$t/ckpt-final.pt" --out "$O/onnx_ref.npz" || fail "onnx ref $t"
    CUDA_VISIBLE_DEVICES=$GPU $OPY experiments/op_adapt_l/scripts/op_l_onnx.py check --onnx "$H/onnx/$t.onnx" --ref "$O/onnx_ref.npz" > "$O/onnx_check.tmp" 2>&1 || fail "onnx check $t"
    grep "^stream" "$O/onnx_check.tmp" | awk '{ if ($16 + 0 > 0.5) bad = 1 } END { exit bad }' || fail "onnx differs from the port $t"
    mv "$O/onnx_check.tmp" "$O/onnx_check.txt"
  fi
  [[ -f $O/replay.json ]] || { g $OPY experiments/hugsim/scripts/lean_probe.py replay "$DATA_DIR/runs/hugsim-lean/jobs_replay.json" --models "$t" --steps 20 --out "$O/replay.json" & p1=$!; }
  [[ -f $O/local.json ]] || { g $OPY experiments/hugsim/scripts/lean_probe.py local "$DATA_DIR/runs/hugsim-lean/jobs_replay.json" --models "$t" --steps 11 --out "$O/local.json" & p2=$!; }
  wait ${p1:-} 2>/dev/null; wait ${p2:-} 2>/dev/null; unset p1 p2
  [[ -f $O/replay.json && -f $O/local.json ]] || fail "lean_probe $t"
done

st "8 report (a) (b)"
g $PY $S/h_launch_report.py "$H/report/$NAME" "${TAGS[@]}" || fail "report ab"
touch "$C/AB_DONE"

st "9 h_chain (dev, probe, navtest, capture)"
GPU=$GPU CPUS=$CPUS SCPUS=$SCPUS EXAMS=0 $S/h_chain.sh "$NAME-x" "${ARMS[@]}" >> "$C/log.txt" 2>&1 || fail "h_chain"
g $PY $S/h_report.py "$NAME-x" "${TAGS[@]}" || log "h_report failed (non-fatal)"

st "10 HUGSIM all 64 (gate ${GATE:-0.75})"
for t in "${TAGS[@]}"; do
  r=$($PY -c "import json; print(json.load(open('$H/report/$NAME/launch_ab.json'))['a']['$t']['ratio'])")
  if $PY -c "import sys; sys.exit(0 if $r <= ${GATE:-0.75} else 1)"; then
    [[ -f $H/hugsim/${t}_all64/DONE ]] && continue
    SCEN=experiments/hugsim/scripts/derot_all64.txt SUFFIX=_all64 WORKERS=${WORKERS:-4} REPORT=$S/h_hugsim64_report.py GPU=$GPU \
      $S/h_hugsim.sh "$t" >> "$C/log.txt" 2>&1 || fail "hugsim $t"
  else
    log "$t: (a) ratio $r > ${GATE:-0.75}, HUGSIM skipped"
  fi
done
st "done"
touch "$C/DONE"
