#!/usr/bin/env bash
# Image-command Q3, sky + green-line arm (sg) on one card: waits for the sky chain's banks, lane lines of the original -> sg banks ->
# teacher -> waits until the sky chain has finished training -> GA / GB / GC in parallel -> zero-shot / dev reports -> selection ->
# seeds 1, 2 of the selected config -> eval plans + reports. Plan: plans/2026-10-04-img-cmd-ft2-prereg.md (addendum, sky + green).
# Writes $DATA_DIR/runs/op_img_cmd/ft/chain3g/{STATUS,DONE,ERROR,log.txt}; reports ft/report3/*_sg.*
#   GPU=1 CPUS=100-149 scripts/tmux_run.sh img3g-chain experiments/op_img_cmd/scripts/img3g_chain.sh
set -uo pipefail
GPU=${GPU:-1}; CPUS=${CPUS:-100-149}
S=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
FT=$DATA_DIR/runs/op_img_cmd/ft
C=$FT/chain3g; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
export IMG_SETS=sg IMG_ARM=sg
PY="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 taskset -c $CPUS $DATA_DIR/envs/op-train/bin/python"
status() { echo "$(date '+%F %T') img3g-chain: $*" | tee "$C/STATUS" >> "$C/log.txt"; }
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

status "waiting for the sky banks"
until [ -f "$FT/bank/skycarla/var.npz" ]; do sleep 30; done
step "lane lines of the original" $PY "$S/img3_bank.py" lanes
step "sg banks" $PY "$S/img3_bank.py" sgtrain sgeval sgcarla
step "teacher" $PY "$S/img3_train.py" teacher
step "zero-shot plans (original)" $PY "$S/img2_eval.py" plans --models O --banks sgtrain sgeval sgcarla
step "zero-shot report (original)" $PY "$S/img2_eval.py" report --model O --sets naveval carlatest
status "waiting for the sky chain to finish training (card memory)"
until grep -q "plans (dev banks)\|dev report\|dev selection\|selected\|no config\|done\|FAILED" "$FT/chain3/STATUS" && \
      ! grep -q "seeds 1, 2" "$FT/chain3/STATUS"; do sleep 60; done
ck q3gsmoke-s0 || step "trainer smoke" $PY "$S/img3_train.py" train --arm gsmoke
status "training GA / GB / GC (seed 0) in parallel"
train_par 0 GA GB GC
M0="q3GA-s0 q3GB-s0 q3GC-s0"
step "plans (dev banks)" $PY "$S/img2_eval.py" plans --models $M0 --banks sgtrain sgcarla dist
for m in $M0; do step "dev report $m" $PY "$S/img2_eval.py" report --model $m --sets navdev carladev; done
step "dev selection" $PY "$S/img2_eval.py" select --models $M0 --family sg --tag _sg
SEL=$($DATA_DIR/envs/op-train/bin/python -c "import json; print(json.load(open('$FT/report3/select_sg.json'))['selected'] or '')")
MODELS="$M0"
if [ -n "$SEL" ]; then
  arm=${SEL#q3}; arm=${arm%-s0}
  status "selected $SEL -> seeds 1, 2 (after the sky chain's seeds)"
  until [ -f "$FT/chain3/DONE" ] || [ -f "$FT/chain3/ERROR" ]; do sleep 60; done
  train_par 1 "$arm"
  train_par 2 "$arm"
  MODELS="$M0 q3$arm-s1 q3$arm-s2"
  step "plans (dev banks, seeds 1-2)" $PY "$S/img2_eval.py" plans --models "q3$arm-s1" "q3$arm-s2" --banks sgtrain sgcarla dist
else
  status "no config passed the dev rule"
fi
step "plans (eval bank)" $PY "$S/img2_eval.py" plans --models $MODELS --banks sgeval
for m in $MODELS; do step "test report $m" $PY "$S/img2_eval.py" report --model $m --sets naveval carlatest navdev carladev; done
status "done: selected '${SEL}'; reports in $FT/report3 (*_sg)"
date '+%F %T' > "$C/DONE"
