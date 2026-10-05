#!/usr/bin/env bash
# op_parity Stage B (plans/2026-10-06-parity-prereg.md, addendum 1): the whole chain into the GPU pool at once, on the box from the repo root.
#   caches (warp / keys fronts; real-frame subset) -> 12 trainings {N, W, G} x {P1, P2} x {s0, s1} -> plans per protocol -> CPU scoring -> report
# CPU scoring of one protocol runs while the other protocols still train. Job ids -> $DATA_DIR/runs/op_parity/stageB_jobs.txt
set -euo pipefail
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl submit --owner op_parity"
S=experiments/op_parity/scripts
L=$DATA_DIR/runs/op_parity/pool/stageB
J=$DATA_DIR/runs/op_parity/stageB_jobs.txt
: > "$J"
sub() { local id; id=$($CL "$@"); echo "$id $2" >> "$J"; echo "$id"; }

# ---- caches: one GPU encoder + a CPU pool each
smoke_c="$PY $S/pp_prep.py --data lb_navtrain --limit 32 --frames keys --workers 8 && $PY $S/pp_prep.py --data lb_navtrain --limit 32 --frames warp --workers 8"
declare -A C
for d in lb_navtrain lb_h1train lb_navtest; do
  for f in warp keys; do
    C[$d.$f]=$(sub --name ppB-c-$d-$f --vram 24 --cpu 20 --ram 24 --preflight "$smoke_c" --log-dir $L/c-$d-$f -- $PY $S/pp_prep.py --data $d --frames $f --workers 18)
  done
done
for f in real keys; do
  C[hq.$f]=$(sub --name ppB-c-hq-$f --vram 24 --cpu 12 --ram 16 --preflight "$smoke_c" --log-dir $L/c-hq-$f -- $PY $S/pp_prep.py --data lb_hq_navtestX --frames $f --no-side --workers 10)
done

# ---- trainings
declare -A FR=([N]=keys [W]=warp [G]=gimm) RF=([N]=keys [W]=real [G]=real)
smoke_t="$PY $S/pp_train.py --arm P2 --frames gimm --steps 5 --tag smoke-stageB"
declare -A T
for p in N W G; do
  dep=""
  [[ $p != G ]] && dep="--after ${C[lb_navtrain.${FR[$p]}]},${C[lb_h1train.${FR[$p]}]}"
  ids=()
  for arm in P1 P2; do for s in 0 1; do
    ids+=("$(sub --name ppB-t-$arm-$p-s$s --train --vram 40 --cpu 4 --ram 24 $dep --preflight "$smoke_t" --log-dir $L/t-$arm-$p-s$s -- \
             $PY $S/pp_train.py --arm $arm --seed $s --frames ${FR[$p]} --steps 600 --tag $arm-$p-s$s)")
  done; done
  T[$p]=$(IFS=,; echo "${ids[*]}")
done

# ---- plans (GPU, minutes) and CPU scoring per protocol, matched navtest and the real-frame subset
score_ids=()
for p in N W G; do
  M="P0 P1-$p-s0 P1-$p-s1 P2-$p-s0 P2-$p-s1"
  a1=${T[$p]}; [[ $p != G ]] && a1+=",${C[lb_navtest.${FR[$p]}]}"
  pm=$(sub --name ppB-p-$p --vram 30 --cpu 8 --ram 24 --after $a1 --log-dir $L/p-$p -- $PY $S/pp_eval.py --data lb_navtest --frames ${FR[$p]} plans --models $M --tag stageB)
  ph=$(sub --name ppB-ph-$p --vram 16 --cpu 8 --ram 16 --after ${T[$p]},${C[hq.${RF[$p]}]} --log-dir $L/ph-$p -- \
           $PY $S/pp_eval.py --data lb_hq_navtestX --frames ${RF[$p]} plans --models $M --tag stageB)
  score_ids+=("$(sub --name ppB-s-$p --vram 1 --cpu 24 --ram 48 --after $pm,$ph --env NAVSIM_THREADS=22 --log-dir $L/s-$p -- bash -c \
     "$PY $S/pp_eval.py --data lb_navtest --frames ${FR[$p]} score --models $M && $PY $S/pp_eval.py --data lb_hq_navtestX --frames ${RF[$p]} score --models $M")")
done
sub --name ppB-report --vram 1 --cpu 4 --ram 16 --after "$(IFS=,; echo "${score_ids[*]}")" --log-dir $L/report -- $PY $S/pp_stageB_report.py
cat "$J"
