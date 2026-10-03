#!/usr/bin/env bash
# Image-command fine-tune Q3 (drift-free) chain on one card: banks -> teacher -> smoke -> configs A / B / C in parallel -> dev
# reports -> dev selection (plan rule) -> seeds 1, 2 of the selected config (if any) -> eval plans + reports (each model once).
# Plan: experiments/op_img_cmd/plans/2026-10-04-img-cmd-ft2-prereg.md. Resumes from disk (every step skips what exists).
# Writes $DATA_DIR/runs/op_img_cmd/ft/chain2/{STATUS,DONE,ERROR,log.txt}.
#   GPU=1 CPUS=100-149 scripts/tmux_run.sh img2-chain experiments/op_img_cmd/scripts/img2_chain.sh
set -uo pipefail
GPU=${GPU:-1}; CPUS=${CPUS:-100-149}
S=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
FT=$DATA_DIR/runs/op_img_cmd/ft
C=$FT/chain2; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
PY="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 taskset -c $CPUS $DATA_DIR/envs/op-train/bin/python"
status() { echo "$(date '+%F %T') img2-chain: $*" | tee "$C/STATUS" >> "$C/log.txt"; }
fail() { status "FAILED: $*"; echo "$(date '+%F %T') $*" > "$C/ERROR"; tail -40 "$C/log.txt" >> "$C/ERROR"; exit 1; }
step() { status "$1"; shift; "$@" >> "$C/log.txt" 2>&1 || fail "$*"; }
ck() { [ -f "$FT/runs/$1/ckpt-final.pt" ]; }
train_par() {   # train_par <seed> <arm>... : the arms in parallel on the card, each its own log
  local seed=$1; shift; local pids=() arm
  for arm in "$@"; do
    ck "q3$arm-s$seed" && continue
    $PY "$S/img2_train.py" train --arm "$arm" --seed "$seed" > "$C/train_q3$arm-s$seed.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "training ($*, seed $seed); see $C/train_*.log"; done
  for arm in "$@"; do ck "q3$arm-s$seed" || fail "no checkpoint q3$arm-s$seed"; done
}

step "banks (split, dist, carla, nba)" $PY "$S/img2_bank.py" split dist carla nba
step "teacher" $PY "$S/img2_train.py" teacher
ck q3smoke-s0 || step "trainer smoke (40 steps)" $PY "$S/img2_train.py" train --arm smoke
status "training A / B / C (seed 0) in parallel"
train_par 0 A B C
M0="q3A-s0 q3B-s0 q3C-s0"
step "plans (dev banks)" $PY "$S/img2_eval.py" plans --models O $M0 --banks train carla dist
for m in $M0; do step "dev report $m" $PY "$S/img2_eval.py" report --model $m --sets navdev carladev; done
step "dev selection" $PY "$S/img2_eval.py" select --models $M0
SEL=$($DATA_DIR/envs/op-train/bin/python -c "import json; print(json.load(open('$FT/report2/select.json'))['selected'] or '')")
MODELS="$M0"
if [ -n "$SEL" ]; then
  arm=${SEL#q3}; arm=${arm%-s0}
  status "selected $SEL -> seeds 1, 2"
  train_par 1 "$arm"
  train_par 2 "$arm"
  MODELS="$M0 q3$arm-s1 q3$arm-s2"
  step "plans (dev banks, seeds 1-2)" $PY "$S/img2_eval.py" plans --models "q3$arm-s1" "q3$arm-s2" --banks train carla dist
else
  status "no config passed the dev rule: test read once per config for the report, no further configs"
fi
step "plans (eval bank)" $PY "$S/img2_eval.py" plans --models $MODELS --banks eval
for m in $MODELS; do step "test report $m" $PY "$S/img2_eval.py" report --model $m --sets naveval carlatest navdev carladev; done
status "done: selected '${SEL}'; reports in $FT/report2"
date '+%F %T' > "$C/DONE"
