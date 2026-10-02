#!/usr/bin/env bash
# Image-command fine-tune (Q2) chain on one card: wait for lb_imgtrain's GIMM frames -> train bank -> teacher -> trainer smoke ->
# ft -> dev check (prereg iteration rule: ft2 when both trained families have dev uptake < 0.30) -> eval plans -> reports.
# Plan: experiments/op_img_cmd/plans/2026-10-04-img-cmd-ft-prereg.md. Resumes from disk (every step skips what exists).
# Writes $DATA_DIR/runs/op_img_cmd/ft/chain/{STATUS,DONE,ERROR,log.txt}.
#   scripts/tmux_run.sh imgft-chain experiments/op_img_cmd/scripts/img_ft_chain.sh   (env GPU=1 CPUS=100-149 SCRIPTS=<dir>)
set -uo pipefail
GPU=${GPU:-1}; CPUS=${CPUS:-100-149}
S=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
FT=$DATA_DIR/runs/op_img_cmd/ft
C=$FT/chain; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
PY="env CUDA_VISIBLE_DEVICES=$GPU taskset -c $CPUS $DATA_DIR/envs/op-train/bin/python"
status() { echo "$(date '+%F %T') imgft-chain: $*" | tee "$C/STATUS" >> "$C/log.txt"; }
step() { status "$1"; shift; "$@" >> "$C/log.txt" 2>&1 || { status "FAILED: $*"; echo "$(date '+%F %T') $*" > "$C/ERROR"; tail -30 "$C/log.txt" >> "$C/ERROR"; exit 1; }; }

need=$(( (1800 + 31) / 32 ))
status "waiting for $need GIMM chunks of lb_imgtrain"
until [ "$(ls "$DATA_DIR/runs/op_lb/lb_imgtrain/gimm.chunks/" 2>/dev/null | grep -c '\.done$')" -ge "$need" ]; do sleep 60; done
step "bank: train pool" $PY "$S/img_ft_bank.py" train
[ -f "$FT/teacher/train.npz" ] || step "teacher" $PY "$S/img_ft_train.py" teacher
[ -f "$FT/runs/smoke-s0/ckpt-final.pt" ] || step "trainer smoke (40 steps)" $PY "$S/img_ft_train.py" train --arm smoke
[ -f "$FT/runs/ft-s0/ckpt-final.pt" ] || step "train ft (2000 steps)" $PY "$S/img_ft_train.py" train --arm ft
step "plans on the train pool (O, ft-s0)" $PY "$S/img_ft_eval.py" plans --models O ft-s0 --pool train
step "dev report ft-s0" $PY "$S/img_ft_eval.py" report --model ft-s0 --pool dev
models="O ft-s0"
if $PY - "$FT/report/ft-s0_vs_O_dev.csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]).set_index("fam")
u = d.loc[["band", "barrier"], "uptake_a"]
print("dev uptake", u.to_dict())
sys.exit(0 if (u < 0.30).all() else 1)
EOF
then
  status "dev rule: both trained families < 0.30 -> iteration ft2 (lr 1e-4, 3000 steps)"
  echo "- $(date '+%F %H:%M') chain: dev uptake of ft-s0 below 0.30 on band and barrier -> ft2 (lr 1e-4, 3000 steps) per the iteration rule" >> "$C/deviations.txt"
  [ -f "$FT/runs/ft2-s0/ckpt-final.pt" ] || step "train ft2" $PY "$S/img_ft_train.py" train --arm ft2 --steps 3000 --override '{"lr": 1e-4}'
  step "plans on the train pool (ft2)" $PY "$S/img_ft_eval.py" plans --models ft2-s0 --pool train
  step "dev report ft2" $PY "$S/img_ft_eval.py" report --model ft2-s0 --pool dev
  models="$models ft2-s0"
fi
step "eval plans ($models)" $PY "$S/img_ft_eval.py" plans --models $models --pool eval
[ -f "$FT/report/port-O/nav_effects.md" ] || step "port-O zero-shot table (img_report)" $PY "$S/img_report.py" --domain nav --glob port-O.npz --out "$FT/report/port-O"
for m in $models; do
  [ "$m" = O ] && continue
  step "eval report $m" $PY "$S/img_ft_eval.py" report --model $m --pool eval --trt
done
status "done: reports in $FT/report"
date '+%F %T' > "$C/DONE"
