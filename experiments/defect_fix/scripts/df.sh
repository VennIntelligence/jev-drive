#!/usr/bin/env bash
# NR1 chain (plans/2026-10-17-nr1-prereg.md). Box, repo root; every job through the GPU pool, --after chained; ids in $O/jobs.txt.
# NR1 trainings leave JEV_RM_EAGER out (rm_train compiles, the default for new lanes); only the flags-off identity runs are eager.
#   df.sh data              (CPU) trainval reasons -> pool on1_NR1TV -> the three base draw-weight files; merged L18 pool on1_L18NR1
#   df.sh codecheck <rev>   60-step identity, eager: V2's base command (RTB1AR-F-s0) and the RSG-L18 command, code <rev> vs this code with every
#                           NR1 flag off -> df.py bitcmp (results/identity_base.json, identity_rsg.json; the RSG one also = RSG-L18-s0's on_hash_60)
#   df.sh lat               lateral rasters of on1_L18NR1 (AlpaSim, CPU) + V2's plans on its rows (GPU) -> label gate (results/lat_gate) -> $O/lat_pick.env
#   df.sh base              (after codecheck) bases NR1B05-F-s0 / NR1B10-F-s0 / NR1BALL-F-s0 (10 000 steps from the shipped weights) -> wc / led F /
#                           hist / arrival -> df.py candgate per base -> `cont` (job body) for each candidate; df.py basesum after everything
#   df.sh cont <base>       job body: V2's L18 command from <base> (2 500 steps, old pool) -> held 460 @18 (6), public 1491 @10 (10), static, wc,
#                           the waiting set (every base checkpoint and every read below is also read on alpasim_nuplan/waitset@v1)
#   df.sh l18pilot          job body after the label gate: NR1H0P / NR1H15P / NR1H30P-L18-s0 (1 000 steps from RTB1AR-F-s0, merged pool, noise on;
#                           lateral terms at 0 / 0.15 / 0.3 as the gate allows) -> held (6), public (10), static, wc each -> df.py pilotsum
#   df.sh lat2              prereg amendment 1: the opposite-lane region rebuilt (junction mask + 6 m, rear axle only) -> gate -> NR1H15O-L18-s0
#   df.sh full <base> <w> [noac]   full chain: L18 2 500 steps from <base>, lateral weight <w> (0 = none), seeds 0 / 1 -> held, public, static,
#                           stop17, wc per seed, PAI 281 (seed 0, submission config) -> df.py fullsum; noac: without the history-speed noise.
#                           Env: TAGP tag prefix (NR1F), SEEDS ("0 1"), NOPUB=1 no public chunks, NOPAI=1 no PAI submitter (stop17 + PAI 281);
#                           prereg supplement 3: SEEDS=0 NOPUB=1 NOPAI=1 TAGP=NR1FB10 df.sh full NR1B10-F-s0 0.15v2
#   df.sh reads <tag> [pai] job body: the closed-loop reads of one L18 checkpoint (held 6, public 10, static, wc; + stop17 / PAI 281 with pai)
#   df.sh wait [submitter ..] blocks until every NR1 pool job ended (run once in the background); df.sh fullsum: results/full_chain.{md,json} + _regress
#   df.sh train <arm> <seed> <steps> <tag>   job body; arms below
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/defect_fix; CK=$DATA_DIR/runs/op_parity/runs; CR=$DATA_DIR/runs/op_parity/cache; ME=experiments/defect_fix/scripts/df.sh
PY=experiments/defect_fix/scripts/df.py; LT=experiments/defect_fix/scripts/df_lat.py; TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RT=experiments/route_mem/scripts/rm_train.py
NX=$DATA_DIR/runs/wait_cap/next; NXP=experiments/wait_cap/scripts/nx.py; WC=experiments/wait_cap/scripts/wc.py; LED=experiments/cap_ledger/scripts/led.py
OL=experiments/onpol_long/scripts/ol_chain.sh; OLW=$DATA_DIR/runs/onpol_long; LAT=$O/lat/L18NR1; LAT2=$O/lat/L18NR1v2; WS=experiments/alpasim_proxy/scripts/wait_set.sh
RB0="--yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"             # RB0-F-s0's command (nx.sh RB0)
RB0I="--yr-mass 0.1 --label-bank --shape --ho-w 3 --rm-lr 3e-4"        # the same as it ran: no pool -> agent hinge off (rm_train's rule), so a pool can join
KD="--kd-lam 1 --d2-lam 1 --kd-groups AR:1 --kd-nav runs/wait_cap/next/approach_nav.npz --kd-nav-k 8 --d2-teacher $DATA_DIR/runs/desire_cap/train/teacher"
RSG="--warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train \
--lr-scale 0.1 --on-k 13 --on-mode self"                                                                  # RSG-L18-s0's command minus --init / --on
TV="--on on1_NR1TV --on-k 52 --on-mode imit --on-normal --on-split navsim/selfiter_tv"                     # the base stage's trainval rows (prereg 2.3)
AC="--ego-scale-speed 0.5 --ego-scale-max 0.2"; V2B=RTB1AR-F-s0; V2=RTB1ARO-L18-s0; RSG_HASH60=ea8cacc3e451c577
STATIC=$NX/cont/static.txt; STOP17=$NX/cont/stop17.txt; PAICH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; PAIWIZ=controller.gains.rel_front_steering_angle_weight=10
BASES=("0.05/0.05 b05 NR1B05-F-s0" "0.10/0.10 b10 NR1B10-F-s0" "natural ball NR1BALL-F-s0")
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $O/jobs.txt; }
fresh() { for T in "$@"; do [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }; done; }
FT=(--train --vram 32 --cpu 6 --ram 70 --env CL_HIST_KEY=rm-train)                       # compiled (no JEV_RM_EAGER); kd / d2 second forwards
FL=(--train --vram 44 --cpu 6 --ram 90 --env CL_HIST_KEY=rm-train)                       # + the lateral rasters on the card (~12 GB)
FE=(--train --vram 28 --cpu 6 --ram 70 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)  # identity runs only
lat_flags() {  # lat_flags <w> [v2]: the lateral terms at weight w with the gate's pick ($O/lat_pick.env; v2: amendment 1's $O/lat_pick_v2.env and raster)
  [[ $1 == 0 ]] && return 0; local env=$O/lat_pick.env lf=$LAT; [[ ${2:-} == v2 ]] && { env=$O/lat_pick_v2.env; lf=$LAT2; }; source $env
  local f=""; [[ $ONC == 1 ]] && f="$f --lat-onc $1"; [[ $SOFT == 1 ]] && f="$f --lat-soft $(python3 -c "print($1 / 5)")"; [[ $SIDE == 1 ]] && f="$f --lat-side $1"
  [[ -n $f ]] && echo "$f --lat-file $lf --lat-depth $DEPTH --lat-soft-depth $SDEPTH --lat-buf $BUF"; }
wsread() {  # wsread <tag> [after]: the closed-loop waiting set alpasim_nuplan/waitset@v1 (wait_set.sh's `run` job body = its `read`, whose own
           # submit insists on an existing checkpoint) -> $DATA_DIR/runs/alpasim_proxy/wait_set/reads/<tag>/{run, table/<tag>.csv}; id on stdout
  sub df-ws-$1 ${2:+--after $2} --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $WS run $1 $DATA_DIR/runs/alpasim_proxy/wait_set/reads/$1/run; }
reads() {  # reads <tag> <after> [pai]: held 460 @18 (6) + public 1491 @10 (10) + static (+ stop17, PAI 281) + wc plans -> ids on stdout (space separated)
  local T=$1 A=$2 ids=() i
  for i in 0 1 2 3 4 5; do ids+=($(sub df-held-$T-$i --after $A --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || return 1; done
  [[ -n ${NOPUB:-} ]] || for i in $(seq 0 9); do ids+=($(sub df-pub-$T-$i --after $A --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash $OL readlist $T $OLW/lists/all_n10/c$i.txt 10 $O/pub/$T/b$i)) || return 1; done          # NOPUB=1: amendment 2 (pilot reads without public)
  ids+=($(sub df-static-$T --after $A --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_DUMP=1 \
      -- bash $OL readlist $T $STATIC 10 $O/static/$T/run)) || return 1
  ids+=($(sub df-wc-$T --after $A --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags $T)) || return 1
  ids+=($(wsread $T $A)) || return 1
  if [[ -n ${3:-} ]]; then                                            # pai: stop17 + PAI 281; stop17: stop17 only (job body after the training)
    ids+=($(sub df-paisub-$T --after $A --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME paisub $T $3)) || return 1; fi
  echo "${ids[*]}"; }
case ${1:-} in
data)
  [[ ! -e $CR/on1_NR1TV/tab.npz && ! -e $CR/on1_L18NR1/tab.npz ]] || { echo "a pool exists" >&2; exit 1; }
  r=$(sub df-reasons --vram 0.1 --cpu 48 --ram 64 --timeout-h 2 -- $NS $PY reasons --workers 48) || exit 1
  p=$(sub df-pack --after $r --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- $TR $PY pack) || exit 1
  w=$(sub df-weights --after $p --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- bash -c "$TR $PY weights --wait 0.05 --launch 0.05 --out D05 && \
      $TR $PY weights --wait 0.10 --launch 0.10 --out D10 && $TR $PY weights --rule all --out DALL") || exit 1
  m=$(sub df-merge-L18NR1 --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_pait.py merge \
      --pools on1_L18RB0:rb0 on1_L18RTB1:rtb1 --out L18NR1) || exit 1
  echo "data: reasons $r pack $p weights $w merge $m" | tee -a $O/stages.txt ;;
codecheck)
  rev=${2:?rev}; W=$DATA_DIR/tmp/df_old_$rev; [[ -d $W ]] || git worktree add --detach $W $rev > /dev/null || exit 1
  fresh NR1GG-B-old NR1GG-B-new NR1GG-R-old NR1GG-R-new
  a=$(sub df-gg-b-old "${FE[@]}" --timeout-h 1 -- $TR $W/$RT train --tag NR1GG-B-old $RB0 $KD --steps 60 --seed 0) || exit 1
  b=$(sub df-gg-b-new "${FE[@]}" --timeout-h 1 -- bash $ME train v2base 0 60 NR1GG-B-new) || exit 1
  c=$(sub df-gg-r-old "${FE[@]}" --timeout-h 1 -- $TR $W/$RT train --tag NR1GG-R-old --init RB0-F-s0 $RSG --on on1_L18RB0 --steps 60 --seed 0) || exit 1
  d=$(sub df-gg-r-new "${FE[@]}" --timeout-h 1 -- bash $ME train rsg 0 60 NR1GG-R-new) || exit 1
  e=$(sub df-bitcmp-base --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a NR1GG-B-old --b NR1GG-B-new --out identity_base \
      --note "V2 base command (RTB1AR-F-s0) 60 steps from the shipped weights, eager: code $rev vs this code ($(git rev-parse --short HEAD)) with every NR1 flag off") || exit 1
  f=$(sub df-bitcmp-rsg --after $c,$d --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a NR1GG-R-old --b NR1GG-R-new --out identity_rsg \
      --hash $RSG_HASH60 --note "RSG-L18 command 60 steps, eager: code $rev vs this code ($(git rev-parse --short HEAD)) with every NR1 flag off; on_hash_60 = RSG-L18-s0's") || exit 1
  echo "codecheck $rev: base $a $b bitcmp $e; rsg $c $d bitcmp $f" | tee -a $O/stages.txt ;;
lat)
  [[ ! -e $LAT/meta.npz ]] || { echo "$LAT exists" >&2; exit 1; }
  b=$(sub df-lat-build --vram 0.1 --cpu 64 --ram 160 --timeout-h 2 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
      RAYON_NUM_THREADS=1 $AP $LT build --out $LAT --workers 64) || exit 1
  v=$(sub df-v2plans --vram 24 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY v2plans --tag $V2 --pool on1_L18NR1) || exit 1
  g=$(sub df-lat-gate --after $b,$v --vram 0.1 --cpu 16 --ram 200 --timeout-h 2 -- $TR $LT gate --lat $LAT --v2 $O/v2plans/${V2}_on1_L18NR1.npz --out lat_gate) || exit 1
  echo "lat: build $b v2plans $v gate $g" | tee -a $O/stages.txt ;;
lat2)   # prereg amendment 1: the opposite-lane region with the junction mask grown 6 m and the rear axle only -> gate (each hard term alone) ->
        # `l18pilot2` (job body): NR1H15O-L18-s0 if the opposite-lane term passes alone
  [[ ! -e $LAT2/meta.npz ]] || { echo "$LAT2 exists" >&2; exit 1; }
  b=$(sub df-lat2-build --vram 0.1 --cpu 64 --ram 160 --timeout-h 2 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
      RAYON_NUM_THREADS=1 $AP $LT build --out $LAT2 --workers 64 --dilate 6 --points ra) || exit 1
  g=$(sub df-lat2-gate --after $b --vram 0.1 --cpu 16 --ram 200 --timeout-h 2 -- $TR $LT gate --lat $LAT2 --v2 $O/v2plans/${V2}_on1_L18NR1.npz --out lat_gate_v2 \
      --env-out lat_pick_v2.env --alone) || exit 1
  q=$(sub df-l18sub2 --after $g --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME l18pilot2) || exit 1
  echo "lat2: build $b gate $g submitter $q" | tee -a $O/stages.txt ;;
l18pilot2)   # job body after the amendment-1 gate
  source $O/lat_pick_v2.env; [[ $ONC == 1 ]] || { echo "amendment 1: the opposite-lane term does not pass alone; no extra arm" | tee -a $O/stages.txt; exit 0; }
  e=$(jid df-bitcmp-rsg) || exit 1; T=NR1H15O-L18-s0; fresh $T
  t=$(sub df-train-$T "${FL[@]}" --after $e --timeout-h 2 -- bash $ME train h2=0.15 0 1000 $T) || exit 1
  ids=$(reads $T $t) || exit 1; s0=$(jid df-pilotsum) || exit 1
  s=$(sub df-pilotsum2 --after "$(csv $ids),$s0" --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY pilotsum --j NR1H15P-L18-s0 NR1H30P-L18-s0 $T \
      --control NR1H0P-L18-s0 --out l18_pilot_a1) || exit 1
  echo "l18pilot2 $T: train $t reads $ids summary $s" | tee -a $O/stages.txt ;;
base)
  e=$(jid df-bitcmp-base) && f=$(jid df-bitcmp-rsg) || { echo "no identity jobs (df.sh codecheck)" >&2; exit 1; }; sums=()
  [[ -d $DATA_DIR/runs/alpasim_proxy/wait_set/reads/$V2B ]] || { v=$(wsread $V2B) || exit 1; echo "base: waiting set of $V2B $v" | tee -a $O/stages.txt; }
  for x in "${BASES[@]}"; do read -r DOSE ARM B <<< "$x"; fresh $B; C=${B/-F-s0/O-L18-s0}
    t=$(sub df-train-$B "${FT[@]}" --after $e,$f --timeout-h 3 -- bash $ME train $ARM 0 10000 $B) || exit 1
    w=$(sub df-wc-$B --after $t --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags $B) || exit 1
    l=$(sub df-led-$B --after $t --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $LED plans --tags $B --conds F) || exit 1
    h=$(sub df-hist-$B --after $t --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR $NXP hist --tags $B) || exit 1
    r=$(sub df-arr-$B --after $t --vram 12 --cpu 4 --ram 40 --timeout-h 1 -- $TR $NXP arrival --tags $B) || exit 1
    q=$(wsread $B $t) || exit 1
    g=$(sub df-candgate-$B --after $w,$l,$h,$r --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $TR $PY candgate --j $B) || exit 1
    c=$(sub df-contsub-$B --after $g --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont $B) || exit 1
    sums+=("$DOSE:$B:$C"); echo "base $B ($DOSE): train $t wc $w led $l hist $h arrival $r waiting set $q candgate $g cont-submitter $c" | tee -a $O/stages.txt
  done; echo "${sums[*]}" > $O/bases.txt ;;
cont)   # job body after a passing candgate
  B=$2; C=${B/-F-s0/O-L18-s0}; fresh $C
  t=$(sub df-train-$C "${FT[@]}" --timeout-h 2 -- bash $ME train cont=$B 0 2500 $C) || exit 1
  ids=$(NOPUB=1 reads $C $t) || exit 1; echo "cont $B: train $t reads $ids (no public: amendment 2)" | tee -a $O/stages.txt ;;
regate)   # amendment 2: the bases' led / hist reads cut; candgate (after wc + arrival only) and the cont submitter again, per base
  for x in "${BASES[@]}"; do read -r DOSE ARM B <<< "$x"
    w=$(jid df-wc-$B) && r=$(jid df-arr-$B) || exit 1
    g=$(sub df-candgate-$B --after $w,$r --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $TR $PY candgate --j $B) || exit 1
    c=$(sub df-contsub-$B --after $g --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont $B) || exit 1
    echo "regate $B: candgate $g cont-submitter $c" | tee -a $O/stages.txt; done ;;
repub)   # repub <tag> <chunk ..>: resubmit public 1491 chunks of a tag (partial outputs removed first)
  T=$2; for i in "${@:3}"; do rm -rf $O/pub/$T/b$i
    id=$(sub df-pub-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $OL readlist $T $OLW/lists/all_n10/c$i.txt 10 $O/pub/$T/b$i) || exit 1
    echo "repub $T b$i: $id" | tee -a $O/stages.txt; done ;;
wscollect)   # wscollect <tag ..>: the waiting-set tables of finished reads whose collect step stopped (df_wscollect.py; CPU pool job)
  id=$(sub df-wscollect --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $AP experiments/defect_fix/scripts/df_wscollect.py "${@:2}") || exit 1
  echo "wscollect ${*:2}: $id" | tee -a $O/stages.txt ;;
rews)   # rews <tag>: resubmit a waiting-set read
  id=$(wsread $2) || exit 1; echo "rews $2: $id" | tee -a $O/stages.txt ;;
l18pilot)   # job body after the label gate
  [[ -f $O/lat_pick.env ]] || { echo "no $O/lat_pick.env" >&2; exit 1; }; source $O/lat_pick.env; e=$(jid df-bitcmp-rsg) || exit 1; all=()
  PP=("0 NR1H0P-L18-s0"); [[ $ONC$SOFT$SIDE != 000 ]] && PP+=("0.15 NR1H15P-L18-s0" "0.3 NR1H30P-L18-s0")
  for x in "${PP[@]}"; do read -r Wt T <<< "$x"; fresh $T
    t=$(sub df-train-$T "${FL[@]}" --after $e --timeout-h 2 -- bash $ME train h=$Wt 0 1000 $T) || exit 1
    ids=$(reads $T $t) || exit 1; all+=($ids); echo "l18pilot $T (w $Wt): train $t reads $ids" | tee -a $O/stages.txt; done
  J=$(for x in "${PP[@]:1}"; do echo -n "${x#* } "; done)
  [[ -n $J ]] && { s=$(sub df-pilotsum --after "$(csv ${all[@]})" --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY pilotsum --j $J --control NR1H0P-L18-s0) || exit 1
    echo "l18pilot: summary $s" | tee -a $O/stages.txt; } ;;
full)
  B=${2:?base}; Wt=${3:?lateral weight}; NOAC=${4:-}; all=(); TT=()
  for s in ${SEEDS:-0 1}; do T=${TAGP:-NR1F}-L18-s$s; fresh $T; TT+=($T)                         # TAGP: a second chain's tag prefix; SEEDS: fewer seeds
    t=$(sub df-train-$T "${FL[@]}" --timeout-h 3 -- bash $ME train full=$B:$Wt:$NOAC $s 2500 $T) || exit 1
    ids=$(reads $T $t $([[ -n ${NOPAI:-} ]] || { [[ $s == 0 ]] && echo pai || echo stop17; })) || exit 1; all+=($ids); echo "full $T (base $B, w $Wt, $NOAC): train $t reads $ids" | tee -a $O/stages.txt; done
  echo "full: ${TT[*]} after $(csv ${all[@]})" | tee -a $O/stages.txt ;;
paisub)   # job body: paisub <tag> <pai|stop17>: PAI stop17 (+ PAI 281 with pai), submission config, of an L18 checkpoint
  T=$2
  CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/paisub/$T $T $STOP17 > $O/paisub_$T.submit.txt || exit 1
  [[ ${3:-pai} == pai ]] || exit 0
  CHUNKS_FROM=$PAICH SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/pai/$T $T > $O/pai_$T.submit.txt || exit 1 ;;
wait)   # wait [submitter ...]: blocks until every NR1 pool job ($O/pool/*/STATUS) is terminal, the named submitters ended and every pai_box
        # read root in WAIT_PAI has a DONE per chunk; cap CAP s (7000).
        # Run it ONCE with run_in_background (no polling by hand); prints the jobs that did not end `done`.
  t0=$(date +%s)
  while :; do
    open=0; for f in $O/pool/*/STATUS; do l=$(tail -1 $f); case "${l#*): }" in queued*|running*|inbox*|submitted*|waiting*) open=$((open+1));; esac; done   # the state word after "(id): " ("cancelled: cancelled while queued" is over)
    sub=1; for n in "${@:2}"; do case "$(tail -1 $O/pool/$n/STATUS 2>/dev/null)" in *done*|*failed*|*cancel*|*skip*|*error*) ;; *) sub=0;; esac; done
    pai=1; for r in ${WAIT_PAI:-}; do n=$(ls $r/chunks/*.txt 2>/dev/null | wc -l); d=$(ls -d $r/runs/*_c*/DONE 2>/dev/null | wc -l); (( n > 0 && d == n )) || pai=0; done
    (( open == 0 && sub == 1 && pai == 1 )) && break; (( $(date +%s) - t0 > ${CAP:-7000} )) && { echo "cap reached: $open open, pai $pai"; break; }; sleep 60
  done; date; for f in $O/pool/*/STATUS; do tail -1 $f; done | grep -v ": done" ;;
fullsum)   # after the full chain ended (df.sh wait df-paisub-NR1F-L18-s0 df-paisub-NR1F-L18-s1, then the PAI chunks): the table against the bar
  P=${TAGP:-NR1F}; $TR $PY fullsum --j $P-L18-s0 $P-L18-s1 --pai $P-L18-s0 ;;
train)   # train <arm> <seed> <steps> <tag>
  case $2 in
    v2base) F="$RB0 $KD" ;;                                                                       # identity only (= nx.sh rtb1ar)
    rsg) F="--init RB0-F-s0 $RSG --on on1_L18RB0" ;;                                              # identity only (= nx.sh rsg)
    b05) F="$RB0I $KD $TV --on-wfile runs/defect_fix/w/D05.npz" ;;
    b10) F="$RB0I $KD $TV --on-wfile runs/defect_fix/w/D10.npz" ;;
    ball) F="$RB0I $KD $TV" ;;                                                                    # natural share: the uniform pool draw
    cont=*) F="--init ${2#cont=} --allow-init $RSG --on on1_L18RB0" ;;                           # V2's L18 command from any base
    h=*) F="--init $V2B --allow-init $RSG --on on1_L18NR1 $AC $(lat_flags ${2#h=})" ;;            # L18 pilots
    h2=*) F="--init $V2B --allow-init $RSG --on on1_L18NR1 $AC $(lat_flags ${2#h2=} v2)" ;;       # amendment 1's extra arm
    full=*) IFS=: read -r FB FW FN <<< "${2#full=}"
            F="--init $FB --allow-init $RSG --on on1_L18NR1 $([[ $FN == noac ]] || echo $AC) $(lat_flags ${FW%v2} $([[ $FW == *v2 ]] && echo v2))" ;;   # w = 0.15v2: amendment 1's terms
    *) echo "train: arm $2?" >&2; exit 2 ;;
  esac
  exec $TR $RT train --tag $5 $F --steps $4 --seed $3 ;;
*) sed -n 2,23p $0; exit 2 ;;
esac
