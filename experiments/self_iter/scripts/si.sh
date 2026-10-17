#!/usr/bin/env bash
# SELF-ITER chain (plans/2026-10-17-self-iter-prereg.md). Box, repo root; every job through the GPU pool, --after chained; ids in $O/jobs.txt.
#   si.sh launch <rev>     one call: 60-step identity (RSG-L18 recipe, code <rev> vs this code, flags off) + select -> the driver job `rows`
#   si.sh rows             (job body, after select) per new batch index -> pp_prep -> route memory + teacher; pack (merged pool on1_$POOL) ->
#                          w0 identity (merged pool, new rows weight 0, mixed) -> bitcmp; scan of RSG-L18-s0 (all rows) -> report; dev scans of
#                          RSG-L18-s0 / WTCP-L18-s0; smoke SISMOKE (40 steps); pilot SIU1P / SIM3P / SIM6P (1 000 steps, mult 1 / 3 / 6, after both
#                          identities + smoke) -> dev scans -> gate `pilot` (C WTCP-L18-s0) -> driver `after pilot`
#   si.sh after <pilot|long>   (job body, after a gate) every passing arm -> `cl`; after pilot: the long run L (25 000 steps, 10 x the RSG-L18
#                          continuation's 2 500) of the passing arms (best two by the top-decile drop) -> dev scan -> gate `long` (C RSG-L18-s0) -> `after long`
#   si.sh cl <tag>         held 460 @18 (6), static scene @10, PAI stop17, arrival replay, public 1491 @10 (10) -> rf_follow -> si.py clsum
#   si.sh train <mult|off|w0|smoke> <seed> <steps> <tag>   job body
# Full scale (the whole trainval download): SFX=2 si.sh launch-full -> select adds batches for the new logs, pack over every batch into
# on1_L18RBSI2, then the same rows chain with tags SI*2-L18-s0. Env: PRIO (default 9), SFX (pool / tag suffix, default '').
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/self_iter; CK=$DATA_DIR/runs/op_parity/runs; CR=$DATA_DIR/runs/op_parity/cache; ME=experiments/self_iter/scripts/si.sh; PY=experiments/self_iter/scripts/si.py
TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RT=experiments/route_mem/scripts/rm_train.py; OL=experiments/onpol_long/scripts/ol_chain.sh; NXP=experiments/wait_cap/scripts/nx.py
SFX=${SFX:-}; POOL=L18RBSI$SFX; SCAN=runs/self_iter/scan/RSG-L18-s0$SFX.npz
RSG="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 \
--on-split alpasim_nuplan/onpol-train --lr-scale 0.1"                                                   # wt.sh REC + --lr-scale 0.1 = RSG-L18-s0's command
RSGP="--on on1_L18RB0 --on-k 13 --on-mode self"
PILOT=(SIU1P$SFX-L18-s0 SIM3P$SFX-L18-s0 SIM6P$SFX-L18-s0); MULT=(1 3 6); LSTEPS=25000
PAICH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks; PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; PAIWIZ=controller.gains.rel_front_steering_angle_weight=10
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner self-iter --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
FW=(--train --vram 28 --cpu 6 --ram 24 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)   # measured: RSG recipe 19 GB VRAM, 3 GB RSS; one pool history key for every rm_train arm
csv() { local IFS=,; echo "$*"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $O/jobs.txt; }
fresh() { for T in "$@"; do [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }; done; }
dscan() {  # dscan <after> <tag>: the dev-set scan of a tag -> id
  sub si-dscan-$2 --after $1 --vram 22 --cpu 6 --ram 16 --timeout-h 1 -- $TR $PY scan --policy $2 --pool on1_$POOL --sets dev --name $2$SFX; }
case ${1:-} in
launch|launch-full)
  if [[ $1 == launch ]]; then
    rev=${2:?rev}; W=$DATA_DIR/tmp/si_old_$rev; [[ -d $W ]] || git worktree add --detach $W $rev > /dev/null || exit 1; fresh SIGG-old SIGG-new
    a=$(sub si-gg-old "${FW[@]}" --timeout-h 1 -- $TR $W/$RT train --tag SIGG-old $RSG $RSGP --steps 60 --seed 0) || exit 1
    b=$(sub si-gg-new "${FW[@]}" --timeout-h 1 -- bash $ME train off 0 60 SIGG-new) || exit 1
    c=$(sub si-bitcmp-code --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a SIGG-old --b SIGG-new --out identity_code \
        --note "RSG-L18 recipe 60 steps: code $rev vs this code ($(git rev-parse --short HEAD)) with --si-scan off") || exit 1
  fi
  s=$(sub si-select$SFX --vram 0.1 --cpu 40 --ram 16 --timeout-h 3 -- $NS $PY select --workers 40) || exit 1
  r=$(sub si-rows$SFX --after $s --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env SFX=$SFX --env PRIO=${PRIO:-9} -- bash $ME rows) || exit 1
  echo "$1: identity ${a:-} ${b:-} ${c:-}; select $s rows-driver $r" | tee -a $O/stages.txt ;;
rows)   # job body
  BB=$(ls $O/batches | sort | paste -sd,); af=()
  for B in ${BB//,/ }; do D=idx_selfiter_tr$B
    if [[ -f $CR/$D@warp/teacher.npz && -f $DATA_DIR/runs/route_mem/feat/$D.npz ]]; then continue; fi       # an earlier round's batch
    i=$(sub si-index-$B --vram 0.1 --cpu 4 --ram 24 --timeout-h 2 -- $NS $PY index --batch $B) || exit 1
    p=$(sub si-prep-$B --after $i --vram 6 --cpu 16 --ram 40 --timeout-h 4 -- $TR experiments/op_parity/scripts/pp_prep.py --data $D --frames warp --no-side --synth cpu --workers 16) || exit 1
    m=$(sub si-rmnav-$B --after $p --vram 0.1 --cpu 16 --ram 16 --timeout-h 2 -- $AP experiments/route_mem/scripts/rm_build.py nav --data $D --workers 16) || exit 1
    t=$(sub si-teacher-$B --after $p --vram 10 --cpu 4 --ram 12 --timeout-h 2 -- $TR $PY teacher --batch $B) || exit 1
    af+=($m $t); echo "batch $B: index $i prep $p rmnav $m teacher $t" | tee -a $O/stages.txt
  done
  k=$(sub si-pack-$POOL ${af:+--after $(csv ${af[@]})} --vram 0.1 --cpu 4 --ram 48 --timeout-h 3 -- $TR $PY pack --batches $BB --name SITV$SFX --merged $POOL) || exit 1
  gn=$(jid si-gg-new) || { echo "no SIGG-new job" >&2; exit 1; }; gc=$(jid si-bitcmp-code)
  w=$(sub si-gg-w0$SFX "${FW[@]}" --after $k --timeout-h 1 -- bash $ME train w0 0 60 SIGG-w0$SFX) || exit 1
  wb=$(sub si-bitcmp-w0$SFX --after $w,$gn --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a SIGG-new --b SIGG-w0$SFX --out identity_w0$SFX \
       --note "RSG-L18 recipe vs merged pool on1_$POOL with the new rows at weight 0, --on-mode mixed, --si-scan off") || exit 1
  sc=$(sub si-scan-RSG$SFX --after $k --vram 24 --cpu 8 --ram 24 --timeout-h 3 -- $TR $PY scan --policy RSG-L18-s0 --pool on1_$POOL --name RSG-L18-s0$SFX) || exit 1
  rp=$(sub si-report-RSG$SFX --after $sc --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY report --scan RSG-L18-s0$SFX) || exit 1
  d1=$(dscan $k RSG-L18-s0) || exit 1; d2=$(dscan $k WTCP-L18-s0) || exit 1
  fresh SISMOKE$SFX-L18-s0 ${PILOT[@]}
  sm=$(sub si-smoke$SFX "${FW[@]}" --after $sc,$gc --timeout-h 1 -- bash $ME train 3 0 40 SISMOKE$SFX-L18-s0) || exit 1
  tt=(); dd=()
  for j in 0 1 2; do T=${PILOT[$j]}
    tt+=($(sub si-train-$T "${FW[@]}" --after $sm,$wb,$gc --timeout-h 2 -- bash $ME train ${MULT[$j]} 0 1000 $T)) || exit 1
    dd+=($(dscan ${tt[-1]} $T)) || exit 1; done
  g=$(sub si-gate-pilot$SFX --after "$(csv ${dd[@]}),$d1,$d2,$sc" --vram 0.1 --cpu 4 --ram 12 --timeout-h 1 -- \
      $TR $PY gate --j ${PILOT[@]} --c WTCP-L18-s0 --pol RSG-L18-s0$SFX --sfx "$SFX" --name pilot$SFX) || exit 1
  n=$(sub si-after-pilot$SFX --after $g --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env SFX=$SFX --env PRIO=${PRIO:-9} -- bash $ME after pilot) || exit 1
  echo "rows: pack $k w0 $w bitcmp $wb scan $sc report $rp dscan $d1 $d2 smoke $sm pilot ${tt[*]} dscans ${dd[*]} gate $g after $n" | tee -a $O/stages.txt ;;
after)   # job body: after <pilot|long>
  S=$2; read -ra PP < $O/pass_$S$SFX.txt; echo "after $S: passing ${PP[*]:-none}" | tee -a $O/stages.txt
  for T in "${PP[@]}"; do bash $ME cl $T || exit 1; done
  if [[ $S == pilot && ${#PP[@]} -gt 0 ]]; then LL=(); dd=(); tt=()
    for T in "${PP[@]:0:2}"; do m=$(echo $T | sed -E 's/^SI[UM]([0-9]+)P.*/\1/'); L=${T/P$SFX-L18/L$SFX-L18}; fresh $L; LL+=($L)
      tt+=($(sub si-train-$L "${FW[@]}" --timeout-h 5 -- bash $ME train $m 0 $LSTEPS $L)) || exit 1; dd+=($(dscan ${tt[-1]} $L)) || exit 1; done
    g=$(sub si-gate-long$SFX --after "$(csv ${dd[@]})" --vram 0.1 --cpu 4 --ram 12 --timeout-h 1 -- $TR $PY gate --j ${LL[@]} --c RSG-L18-s0 --pol RSG-L18-s0$SFX --sfx "$SFX" --name long$SFX) || exit 1
    n=$(sub si-after-long$SFX --after $g --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env SFX=$SFX --env PRIO=${PRIO:-9} -- bash $ME after long) || exit 1
    echo "after pilot: long ${LL[*]} trainings ${tt[*]} dscans ${dd[*]} gate $g after $n" | tee -a $O/stages.txt
  fi ;;
cl)   # cl <tag>: closed-loop reads against RSG-L18-s0 (nx.sh cont-reads' instruments + public 1491 @10)
  T=$2; D=$O/cl/$T; mkdir -p $D; X=$DATA_DIR/runs/wait_cap/next/cont; hh=(); pp=()
  for i in 0 1 2 3 4 5; do hh+=($(jid si-held-$T-$i || sub si-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  st=$(jid si-static-$T || sub si-static-$T --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_DUMP=1 \
      -- bash $OL readlist $T $X/static.txt 10 $D/static/$T/run) || exit 1
  CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $D/paisub/$T $T $X/stop17.txt > $D/paisub.submit.txt || exit 1
  ps=$(awk '{print $1}' $D/paisub.submit.txt | grep -E '^[0-9]{4}-' | paste -sd,); [[ -n $ps ]] || { echo "no PAI stop17 job" >&2; exit 1; }
  ar=$(jid si-arr-$T || sub si-arr-$T --vram 12 --cpu 4 --ram 40 --timeout-h 1 -- $TR $NXP arrival --tags $T) || exit 1
  for i in $(seq 0 9); do pp+=($(jid si-pub-$T-$i || sub si-pub-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash $OL readlist $T $DATA_DIR/runs/onpol_long/lists/all_n10/c$i.txt 10 $D/pub/$T/b$i)) || exit 1; done
  f=$(sub si-follow-$T --after "$(csv ${hh[@]})" --vram 0.1 --cpu 16 --ram 32 --timeout-h 2 -- bash experiments/wait_cap/scripts/wt.sh follow $T RSG-L18-s0,WTC-L18-s0,RSG-L18-s1,WTC-L18-s1 $D) || exit 1
  c=$(sub si-clsum-$T --after "$f,$st,$ps,$ar,$(csv ${pp[@]})" --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY clsum --j $T) || exit 1
  echo "cl $T: held $(csv ${hh[@]}) static $st stop17 $ps arrival $ar public $(csv ${pp[@]}) follow $f summary $c" | tee -a $O/stages.txt ;;
train)   # train <mult|off|w0> <seed> <steps> <tag>
  case $2 in
    off) F="$RSGP" ;;
    w0) F="$(cat $O/flags_${POOL}_w0.txt)" ;;
    *) F="$(cat $O/flags_$POOL.txt) --si-scan $SCAN --si-mode top --si-top 0.1 --si-mult $2" ;;
  esac
  exec $TR $RT train --tag $5 $RSG $F --steps $4 --seed $3 ;;
*) sed -n 2,14p $0; exit 2 ;;
esac
