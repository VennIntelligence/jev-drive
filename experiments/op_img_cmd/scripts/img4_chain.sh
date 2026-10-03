#!/usr/bin/env bash
# Image-command Q3, negatives round (sky arrow + trained disc / wrong-arrow / straight-arrow negatives) on one card: smoke -> NA / NB
# in parallel -> dev reports -> relative-guard selection -> seeds 1, 2 of the selected config -> eval plans + reports.
# Plan: experiments/op_img_cmd/plans/2026-10-04-img-cmd-ft2-prereg.md (addendum "负样本一轮"). Needs the sky banks and teacher2.
# Writes $DATA_DIR/runs/op_img_cmd/ft/chain4/{STATUS,DONE,ERROR,log.txt}; reports in ft/report3 (select_neg.json).
#   GPU=1 CPUS=100-149 scripts/tmux_run.sh img4-chain experiments/op_img_cmd/scripts/img4_chain.sh
set -uo pipefail
GPU=${GPU:-1}; CPUS=${CPUS:-100-149}
S=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
FT=$DATA_DIR/runs/op_img_cmd/ft
C=$FT/chain4; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
export IMG_SETS=sky
PY="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 taskset -c $CPUS $DATA_DIR/envs/op-train/bin/python"
status() { echo "$(date '+%F %T') img4-chain: $*" | tee "$C/STATUS" >> "$C/log.txt"; }
fail() { status "FAILED: $*"; echo "$(date '+%F %T') $*" > "$C/ERROR"; tail -40 "$C/log.txt" >> "$C/ERROR"; exit 1; }
step() { status "$1"; shift; "$@" >> "$C/log.txt" 2>&1 || fail "$*"; }
ck() { [ -f "$FT/runs/$1/ckpt-final.pt" ]; }
train_par() {
  local seed=$1; shift; local pids=() arm
  for arm in "$@"; do
    ck "q3$arm-s$seed" && continue
    $PY "$S/img3_train.py" train --arm "$arm" --seed "$seed" > "$C/train_q3$arm-s$seed.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "training ($*, seed $seed); see $C/train_*.log"; done
  for arm in "$@"; do ck "q3$arm-s$seed" || fail "no checkpoint q3$arm-s$seed"; done
}

ck q3nsmoke-s0 || step "trainer smoke" $PY "$S/img3_train.py" train --arm nsmoke
status "training NA / NB (seed 0) in parallel"
train_par 0 NA NB
M0="q3NA-s0 q3NB-s0"
step "plans (dev banks)" $PY "$S/img2_eval.py" plans --models $M0 --banks skytrain skycarla dist
for m in $M0; do step "dev report $m" $PY "$S/img2_eval.py" report --model $m --sets navdev carladev; done
step "dev selection (relative guard)" $PY "$S/img2_eval.py" select --models $M0 --rule rel --tag _neg
SEL=$($DATA_DIR/envs/op-train/bin/python -c "import json; print(json.load(open('$FT/report3/select_neg.json'))['selected'] or '')")
MODELS="$M0"
if [ -n "$SEL" ]; then
  arm=${SEL#q3}; arm=${arm%-s0}
  status "selected $SEL -> seeds 1, 2 in parallel"
  train_par 1 "$arm" &
  P1=$!
  train_par 2 "$arm" || fail "seed 2"
  wait $P1 || fail "seed 1"
  MODELS="$M0 q3$arm-s1 q3$arm-s2"
  step "plans (dev banks, seeds 1-2)" $PY "$S/img2_eval.py" plans --models "q3$arm-s1" "q3$arm-s2" --banks skytrain skycarla dist
else
  status "no config passed the dev rule"
fi
step "plans (eval bank)" $PY "$S/img2_eval.py" plans --models $MODELS --banks skyeval
pids=()
for m in $MODELS; do
  $PY "$S/img2_eval.py" report --model $m --sets naveval carlatest navdev carladev > "$C/report_$m.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail "test reports; see $C/report_*.log"; done
status "done: selected '${SEL}'; reports in $FT/report3"
date '+%F %T' > "$C/DONE"
