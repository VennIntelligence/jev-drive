#!/usr/bin/env bash
# torch.compile validation for rm_train (plans/2026-10-17-compile-prereg.md). Box, repo root; every job through the GPU pool, --after chained;
# ids in $O/jobs.txt. Existing read bodies only (wt.py devplans, wc.py / led.py / dc.py plans, rm_chain.sh read, pai_box.sh).
#   compile_val.sh submit   the whole chain at once: RSGE-L18-s0 (eager, today's code, = nx.sh `train rsg`) and RSGC-L18-s0 / s1 (the same
#                           command + --compile), each alone on a card (cards 1-4, comparable wall time) -> bitcmp RSGE vs RSG-L18-s0 (--hash);
#                           open-loop reads; held 460 @18 (6 chunks per compiled tag); the PAI submitter (after both compiled trainings)
#   compile_val.sh pai      job body: PAI 281 in the submission config of RSGC-L18-s0 / s1, RSG-L18-s1 and an RSG-L18-s0 re-read, then the
#                           summary job (compile_val.py summary) after every read
# Out: $DATA_DIR/runs/cl_infra/compile/ (pool logs, pai/<tag>/, summary.{json,md}); repo experiments/cl_infra/results/compile_val.{json,md}
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/cl_infra/compile; CK=$DATA_DIR/runs/op_parity/runs; ME=experiments/cl_infra/scripts/compile_val.sh
NX=experiments/wait_cap/scripts/nx.sh; NXP=experiments/wait_cap/scripts/nx.py; WT=experiments/wait_cap/scripts/wt.py
WC=experiments/wait_cap/scripts/wc.py; DC=experiments/desire_cap/scripts/dc.py; LED=experiments/cap_ledger/scripts/led.py
SUM=experiments/cl_infra/scripts/compile_val.py; TR=$DATA_DIR/envs/op-train/bin/python; RT=experiments/route_mem/scripts/rm_train.py
RSG="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 \
--on-split alpasim_nuplan/onpol-train --lr-scale 0.1 --on on1_L18RB0 --on-k 13 --on-mode self"                # = nx.sh $RSG (RSG-L18-s0's command)
E=(RSG-L18-s0 RSG-L18-s1); C=(RSGC-L18-s0 RSGC-L18-s1); R=RSGE-L18-s0; HASH60=ea8cacc3e451c577
PAICH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks; PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; PAIWIZ=controller.gains.rel_front_steering_angle_weight=10
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner cl-compile --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $O/jobs.txt; }
csv() { local IFS=,; echo "$*"; }
TRN=(--train --exclusive --gpus 1,2,3,4 --pin-strict --vram 24 --cpu 6 --ram 70 --timeout-h 2)
case ${1:-} in
submit)
  for T in $R ${C[@]}; do [[ ! -e $CK/$T ]] || { echo "$T exists" >&2; exit 1; }; done
  [[ ! -e $O/jobs.txt ]] || { echo "$O/jobs.txt exists (already submitted)" >&2; exit 1; }
  te=$(sub cmp-train-$R "${TRN[@]}" -- $TR $RT train --tag $R $RSG --steps 2500 --seed 0) || exit 1
  tc=(); for s in 0 1; do tc+=($(sub cmp-train-${C[$s]} "${TRN[@]}" -- $TR $RT train --tag ${C[$s]} $RSG --steps 2500 --seed $s --compile)) || exit 1; done
  b=$(sub cmp-bitcmp-eager --after $te --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash -c "$TR $NXP bitcmp --a RSG-L18-s0 --b $R --out compile_eager_identity \
      --hash $HASH60 --note 'RSG-L18 recipe 2 500 steps, seed 0: stored RSG-L18-s0 vs today code ($(git rev-parse --short HEAD)) without --compile'; true") || exit 1
  T=$(csv ${tc[@]})
  r1=$(sub cmp-dev --after $T --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $WT devplans --tags ${E[1]} ${C[@]}) || exit 1
  r2=$(sub cmp-wc --after $T --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags ${C[@]}) || exit 1
  r3=$(sub cmp-led --after $T --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $LED plans --tags ${C[@]} --conds F) || exit 1
  r4=$(sub cmp-dc --after $T --vram 16 --cpu 4 --ram 40 --timeout-h 1 -- $TR $DC plans --tags ${C[@]} --srcs nav pai) || exit 1
  hh=(); for s in 0 1; do for i in 0 1 2 3 4 5; do hh+=($(sub cmp-held-${C[$s]}-$i --after ${tc[$s]} --vram 20 --cpu 6 --ram 40 --timeout-h 2 \
      --env CL_HIST_KEY=alpasim-read4 -- bash experiments/route_mem/scripts/rm_chain.sh read ${C[$s]} $i 6)) || exit 1; done; done
  p=$(sub cmp-pai-submit --after $T --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME pai) || exit 1
  echo "submit: trainings $te ${tc[*]} bitcmp $b open-loop $r1 $r2 $r3 $r4 held $(csv ${hh[@]}) pai-submitter $p" | tee -a $O/stages.txt ;;
pai)   # job body (after both compiled trainings): the four PAI reads, then the summary after everything
  pp=()
  for T in ${C[@]} ${E[1]} ${E[0]}; do
    CHUNKS_FROM=$PAICH SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/pai/$T $T > $O/pai_$T.submit.txt || exit 1
    q=$(awk '{print $1}' $O/pai_$T.submit.txt | grep -E '^[0-9]{4}-' | paste -sd,); [[ -n $q ]] || { echo "no PAI job for $T" >&2; exit 1; }
    pp+=($q); echo "$q pai-$T" >> $O/jobs.txt
  done
  NN=(cmp-bitcmp-eager cmp-dev cmp-wc cmp-led cmp-dc); for t in ${C[@]}; do for i in 0 1 2 3 4 5; do NN+=(cmp-held-$t-$i); done; done; ww=()
  for n in ${NN[@]}; do ww+=($(jid $n)) || { echo "no job $n" >&2; exit 1; }; done; W=$(csv ${ww[@]})
  s=$(sub cmp-summary --after "$W,$(csv ${pp[@]})" --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $SUM summary) || exit 1
  echo "pai: $(csv ${pp[@]}) summary $s" | tee -a $O/stages.txt ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
