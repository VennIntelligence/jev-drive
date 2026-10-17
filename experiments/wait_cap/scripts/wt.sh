#!/usr/bin/env bash
# WAIT-TRAIN chain (plans/2026-10-16-wait-train-prereg.md). Box, repo root; every job through the GPU pool, --after chained; ids in $O/jobs.txt.
#   wt.sh rows-batch <B>        one log batch (the used logs of the archives with .done that no earlier batch took): select (W / G / S cells + d331's
#                               Y_P / Y_V, leak check, split waitcap_tr<B>) -> index -> pp_prep W -> route memory + teacher (parallel)
#   wt.sh pack <B1,B2,..>       after every batch: pack (cap over the union, on1_WAIT, merged on1_L18RBW, flags) -> the 60-step identity runs WTGG-off
#                               (RSG-L18 recipe) / WTGG-w0 (merged pool, new rows weight 0, --on-lon wc_arc) -> bitcmp
#   wt.sh control               the control arm with the RSG-L18 recipe (Amendment B: bit-identical to the merged pool at weight 0, proven by bitcmp):
#                               WTCP-L18-s0 1 000 steps, WTC-L18-s{0,1} 2 500 steps -> their reads (waitcap / dev / ledger F plans, held 460 @18)
#   wt.sh pilot                 WTJ2P / WTJ5P -L18-s0, 1 000 steps (after bitcmp) -> reads -> gate (writes the picked dose to $O/pilot_pick.txt;
#                               Amendment C: it no longer submits `full`)
#   wt.sh full <j2|j5>          Amendment C fallback only (WT_FALLBACK=1 by hand): WTJ{2,5}-L18-s{0,1}, 2 500 steps -> reads -> held 460 @18 -> clgate held
# Amendment C (the composed WAIT + desire arm WD; experiments/desire_cap/scripts/dt.py, lib/desire_train.py):
#   wt.sh wd-teacher            D2 teacher: shipped Cinque's desire increments on every navtrain shard (12 jobs)
#   wt.sh wd-codecheck <rev>    60-step RSG-L18 recipe with the code of <rev> (before D1 / D2) vs this code -> bitcmp wd_code_identity
#   wt.sh wd-launch <after ids> one pool job waiting on $O/pilot_pick.txt (the WAIT pilot's dose) and the ids -> `wd-pilot`; plus wd-deadline
#                               (one sleeping job: 24 h after the WAIT verdict, the WAIT-only formal unless $O/WD_PASS exists; Amendment C 5)
#   wt.sh wd-pilot              (pilot_pick j2 | j5) D1 rows -> 60-step WDGG-w0 + bitcmp vs WTGG-off -> WD<dose>P{1,3}-L18-s0 (D2 weight 1 / 3,
#                               1 000 steps) -> WAIT reads + dc.py plans -> wd-gate-pilot (wt.py wdgate --submit: wd-full <lam> or the fallback)
#   wt.sh wd-full <lam>         WD<dose>-L18-s{0,1}, 2 500 steps -> reads -> held 460 @18 -> rf_follow (J + C held reads) -> clgate held --follow
#   wt.sh follow J0,J1 C0,C1 D  rf_follow over the held reads of the four tags -> D/follow
# Prereg 补记 D (exploratory reads after the pilot STOP; never writes a registered verdict file):
#   wt.sh explore               devall + held 460 @18 / static scene @10 / PAI stop17 for WTJ2P, WTJ5P, WTCP -> rf_follow -> wx-xheld -> explore-pub
#   wt.sh explore-pub <J,..|-> <C>   public 1491 @10 for the J tags meeting the held lines and C -> wx-xsum ($O/explore/FINISHED)
# Prereg 补记 E (diagnostics after decision 335; no training, no verdict file):
#   wt.sh explore-e             static scene @10 (frame dump) for P0 / P2-F / RB0 / RSG and JEV_VHIST=zero for those + WTCP / WTJ2P / WTJ5P; held 460 @18 and
#                               PAI stop17 for the stages without a stored read; c1_replay of the static scene (plain / zero history); cap_audit collect on
#                               the WTJ2P / WTCP held reads; c1_extract logs + map of the J2P-only offroad zeros -> $O/explore_e/
#   wt.sh xe-replay <tag> <run> | xe-logs | xe-xreplay | xe-heldca | xe-offroad      job bodies; analysis: scripts/wt_xe.py
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/wait_cap/train; CK=$DATA_DIR/runs/op_parity/runs; ME=experiments/wait_cap/scripts/wt.sh; PY=experiments/wait_cap/scripts/wt.py
TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
WC=experiments/wait_cap/scripts/wc.py; OL=$DATA_DIR/runs/onpol_long; D=idx_waitcap_train
REC="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train"
RB="--on on1_L18RB0 --on-k 13 --on-mode self"
PAICH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; PAIWIZ=controller.gains.rel_front_steering_angle_weight=10
STATIC=2021.09.16.15.12.03_veh-42_01037_01434-b5b8d20688385790
STOP17="054b5901 095cf563 127c4f85 2d10a08d 35e082a4 3fe2bac8 44df49fe 4cb7f7ac 4ecc114d 559d4ad6 68c3be44 7d994ed3 9f8acd00 ba40be62 bc309fd6 cbd87234 fc9fde87"
DT=experiments/desire_cap/scripts/dt.py; DC=experiments/desire_cap/scripts/dc.py
TEA=$DATA_DIR/runs/desire_cap/train/teacher; D1R=$DATA_DIR/runs/desire_cap/train/d1_rows.npz; D1K=2
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner wait-train --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
FW=(--train --vram 24 --cpu 6 --ram 70 --env CL_HIST_KEY=rm-train)      # one pool history key for every rm_train arm
csv() { local IFS=,; echo "$*"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $O/jobs.txt; }       # last pool id of a job name
reads() {  # <after ids> <tags..>: waitcap / dev / ledger-F plans -> echo the three ids
  local af=$1; shift
  local a b c
  a=$(sub wt-plans-$1 --after $af --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags "$@") || exit 1
  b=$(sub wt-dev-$1 --after $af --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY devplans --tags "$@") || exit 1
  c=$(sub wt-led-$1 --after $af --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR experiments/cap_ledger/scripts/led.py plans --tags "$@" --conds F) || exit 1
  echo "$a,$b,$c"; }
held() {  # <after id> <tag>: 6 held chunks -> echo ids
  local i ids=(); for i in 0 1 2 3 4 5; do ids+=($(sub wt-held-$2-$i --after $1 --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/route_mem/scripts/rm_chain.sh read $2 $i 6)) || exit 1; done; csv ${ids[@]}; }
case ${1:-} in
rows-batch)
  B=${2:?batch}; D=idx_waitcap_tr$B
  s=$(sub wt-select-$B --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 -- $NS $PY select --batch $B --workers 16) || exit 1
  i=$(sub wt-index-$B --after $s --vram 0.5 --cpu 4 --ram 32 --timeout-h 2 -- $NS $PY index --batch $B) || exit 1
  p=$(sub wt-prep-$B --after $i --vram 10 --cpu 16 --ram 48 --timeout-h 2 -- $TR experiments/op_parity/scripts/pp_prep.py --data $D --frames warp --no-side --synth cpu --workers 16) || exit 1
  r=$(sub wt-rmnav-$B --after $p --vram 0.5 --cpu 16 --ram 40 --timeout-h 1 -- $AP experiments/route_mem/scripts/rm_build.py nav --data $D --workers 16) || exit 1
  t=$(sub wt-teacher-$B --after $p --vram 10 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY teacher --batch $B) || exit 1
  echo "batch $B: select $s index $i prep $p rmnav $r teacher $t" | tee -a $O/rows.txt ;;
waitdl)   # pool-side wait for the 17 archives: one tiny job per missing .done (--when-exists, chained), then `finish`
  S=$DATA_DIR/datasets/navsim/_archives/state; prev=""
  for a in $(cut -f1 experiments/wait_cap/results/trainval_archives.tsv); do [[ -e $S/$a.done ]] && continue
    prev=$(sub wt-dl-${a//[^0-9]/} --vram 0.1 --cpu 1 --ram 0.1 --timeout-h 12 --when-exists $S/$a.done ${prev:+--after $prev} -- true) || exit 1; done
  f=$(sub wt-finish --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 ${prev:+--after $prev} -- bash $ME finish) || exit 1
  echo "waitdl: last wait ${prev:-none} finish $f" | tee -a $O/stages.txt ;;
finish)   # every archive done: the last batch (if any log is new), pack over all batches, pilot
  have=$(ls $O/batches | sort | paste -sd,); B=$(ls $O/batches | sort | tail -1 | tr 'A-Y' 'B-Z')
  bash $ME rows-batch $B || exit 1
  bash $ME pack $have,$B && bash $ME pilot ;;
pack)
  BS=${2:?B1,B2}; af=(); for B in ${BS//,/ }; do af+=($(jid wt-rmnav-$B) $(jid wt-teacher-$B)) || { echo "batch $B not submitted" >&2; exit 1; }; done
  k=$(sub wt-pack --after "$(csv ${af[@]})" --vram 0.5 --cpu 4 --ram 120 --timeout-h 1 -- $TR $PY pack --batches $BS) || exit 1
  g=(); for x in off w0; do g+=($(sub wt-gg-$x "${FW[@]}" --after $k --timeout-h 1 -- bash $ME train $x 0 60 WTGG-$x)) || exit 1; done
  b=$(sub wt-bitcmp --after "$(csv ${g[@]})" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp) || exit 1
  echo "pack $k identity ${g[*]} bitcmp $b" | tee -a $O/rows.txt ;;
control)
  t0=$(sub wt-train-WTCP-L18-s0 "${FW[@]}" --timeout-h 1 -- bash $ME train off 0 1000 WTCP-L18-s0) || exit 1
  r0=$(reads $t0 WTCP-L18-s0) || exit 1
  tt=(); for s in 0 1; do tt+=($(sub wt-train-WTC-L18-s$s "${FW[@]}" --timeout-h 2 -- bash $ME train off $s 2500 WTC-L18-s$s)) || exit 1; done
  r1=$(reads "$(csv ${tt[@]})" WTC-L18-s0 WTC-L18-s1) || exit 1
  h0=$(held ${tt[0]} WTC-L18-s0) || exit 1; h1=$(held ${tt[1]} WTC-L18-s1) || exit 1
  echo "control: pilot C $t0 reads $r0; full C ${tt[*]} reads $r1; held $h0 $h1" | tee -a $O/stages.txt ;;
pilot)
  b=$(jid wt-bitcmp) || { echo "no bitcmp job" >&2; exit 1; }
  ids=(); for q in j2 j5; do T=WT${q^^}P-L18-s0; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    ids+=($(sub wt-train-$T "${FW[@]}" --after $b --timeout-h 1 -- bash $ME train $q 0 1000 $T)) || exit 1; done
  rj=$(reads "$(csv ${ids[@]})" WTJ2P-L18-s0 WTJ5P-L18-s0) || exit 1
  rc="$(jid wt-plans-WTCP-L18-s0),$(jid wt-dev-WTCP-L18-s0),$(jid wt-led-WTCP-L18-s0)" || { echo "control reads not submitted" >&2; exit 1; }
  gg=$(sub wt-gate-pilot --after $rj,$rc,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- \
       $TR $PY gate --j WTJ2P-L18-s0 WTJ5P-L18-s0 --c WTCP-L18-s0 --dose-of j2=WTJ2P-L18-s0,j5=WTJ5P-L18-s0 --submit) || exit 1
  echo "pilot trainings ${ids[*]} reads $rj gate $gg" | tee -a $O/stages.txt ;;
full)    # full <j2|j5>: Amendment C fallback only (the WAIT-only formal); refuses unless WT_FALLBACK=1 is set by hand
  [[ ${WT_FALLBACK:-0} == 1 ]] || { echo "full: held by prereg Amendment C (WAIT-only formal = fallback only; set WT_FALLBACK=1 by hand)" >&2; exit 3; }
  q=${2:?dose}; J=WT${q^^}-L18; JJ=($J-s0 $J-s1); CC=(WTC-L18-s0 WTC-L18-s1); tr=()
  for s in 0 1; do T=$J-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    tr+=($(sub wt-train-$T "${FW[@]}" --timeout-h 2 -- bash $ME train $q $s 2500 $T)) || exit 1; done
  rj=$(reads "$(csv ${tr[@]})" ${JJ[@]}) || exit 1
  hj=$(held ${tr[0]} ${JJ[0]}),$(held ${tr[1]} ${JJ[1]}) || exit 1
  hc=(); for T in ${CC[@]}; do for i in 0 1 2 3 4 5; do hc+=($(jid wt-held-$T-$i)) || { echo "control held reads not submitted" >&2; exit 1; }; done; done
  g=$(sub wt-clgate-held --after "$hj,$(csv ${hc[@]})" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- \
      $TR $PY clgate --stage held --j $(csv ${JJ[@]}) --c $(csv ${CC[@]}) --submit) || exit 1
  echo "full $q: trainings ${tr[*]} reads $rj held $hj gate $g" | tee -a $O/stages.txt ;;
stage-sub)
  JS=${2:?J0,J1}; CS=${3:?C0,C1}; J0=${JS%%,*}; rr=(); mkdir -p $O/sub $O/paisub
  echo $STATIC > $O/sub/static.txt
  for T in ${JS//,/ } ${CS//,/ }; do rr+=($(sub wt-sub-$T --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $O/sub/static.txt 10 $O/sub/$T/run)) || exit 1; done
  cat $PAICH/*.txt | grep -E "^clipgt-($(echo $STOP17 | tr ' ' '|'))" | sort -u > $O/paisub/stop17.txt
  (( $(wc -l < $O/paisub/stop17.txt) == 17 )) || { echo "stop17 list: $(wc -l < $O/paisub/stop17.txt) scenes" >&2; exit 1; }
  CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/paisub/$J0 $J0 $O/paisub/stop17.txt > $O/paisub/submit.txt || exit 1
  pp=$(awk '{print $1}' $O/paisub/submit.txt | grep -E '^[0-9]{4}-' | paste -sd,)
  [[ -n $pp ]] || { echo "no PAI subset job" >&2; exit 1; }
  g=$(sub wt-clgate-sub --after "$(csv ${rr[@]}),$pp" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- $TR $PY clgate --stage sub --j $JS --c $CS --submit) || exit 1
  echo "stage-sub: public static ${rr[*]} PAI stop17 $pp gate $g" | tee -a $O/stages.txt ;;
stage-pub)
  JS=${2:?}; CS=${3:?}; rr=()
  for T in ${JS//,/ } ${CS//,/ }; do [[ ! -e $O/read10/$T ]] || { echo "$O/read10/$T exists" >&2; exit 1; }
    for i in $(seq 0 9); do rr+=($(sub wt-g10-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $OL/lists/all_n10/c$i.txt 10 $O/read10/$T/b$i)) || exit 1; done; done
  g=$(sub wt-clgate-pub --after "$(csv ${rr[@]})" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- $TR $PY clgate --stage pub --j $JS --c $CS --submit) || exit 1
  echo "stage-pub: ${#rr[@]} read jobs gate $g" | tee -a $O/stages.txt ;;
stage-pai)
  JS=${2:?}; CS=${3:?}; J0=${JS%%,*}; [[ ! -e $O/pai/$J0 ]] || { echo "$O/pai/$J0 exists" >&2; exit 1; }
  CHUNKS_FROM=$PAICH SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $O/pai/$J0 $J0 > $O/pai_submit.txt || exit 1
  pp=$(awk '{print $1}' $O/pai_submit.txt | grep -E '^[0-9]{4}-' | paste -sd,)
  [[ -n $pp ]] || { echo "no PAI job" >&2; exit 1; }
  g=$(sub wt-clgate-pai --after $pp --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY clgate --stage pai --j $JS --c $CS) || exit 1
  echo "stage-pai: $(echo $pp | tr , '\n' | wc -l) chunk jobs gate $g" | tee -a $O/stages.txt ;;
explore)   # prereg 补记 D (exploratory, after the pilot STOP; nothing here writes a registered verdict file): devall; held 460 @18, the static public
           # scene @10 and PAI stop17 for the two 1 000-step J pilots and WTCP -> rf_follow -> wx-xheld (public 1491 @10 for J meeting the held lines) -> wx-xsum
  JJ=(WTJ2P-L18-s0 WTJ5P-L18-s0); C=WTCP-L18-s0; X=$O/explore; mkdir -p $X/sub $X/paisub
  [[ ! -e $X/held.json && ! -e $X/jobs.txt ]] || { echo "$X already submitted" >&2; exit 1; }
  echo $STATIC > $X/static.txt
  cat $PAICH/*.txt | grep -E "^clipgt-($(echo $STOP17 | tr ' ' '|'))" | sort -u > $X/stop17.txt
  (( $(wc -l < $X/stop17.txt) == 17 )) || { echo "stop17 list: $(wc -l < $X/stop17.txt) scenes" >&2; exit 1; }
  d=$(sub wx-devall --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR $PY devall --tags ${JJ[@]} $C) || exit 1
  hh=(); ss=()
  for T in ${JJ[@]} $C; do
    for i in 0 1 2 3 4 5; do hh+=($(sub wx-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    ss+=($(sub wx-sub-$T --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $X/static.txt 10 $X/sub/$T/run)) || exit 1
    CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $X/paisub/$T $T $X/stop17.txt > $X/paisub/$T.submit.txt || exit 1
    p=$(awk '{print $1}' $X/paisub/$T.submit.txt | grep -E '^[0-9]{4}-' | paste -sd,); [[ -n $p ]] || { echo "no PAI stop17 job for $T" >&2; exit 1; }; ss+=($p)
  done
  csv ${ss[@]} > $X/sub_ids.txt
  f=$(sub wx-follow --after "$(csv ${hh[@]})" --vram 0.1 --cpu 16 --ram 100 --timeout-h 2 -- bash $ME follow $(csv ${JJ[@]}) $C $X) || exit 1
  g=$(sub wx-xheld --after $f --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- $TR $PY xheld --j $(csv ${JJ[@]}) --c $C --submit) || exit 1
  echo "explore: devall $d held $(csv ${hh[@]}) static+stop17 $(cat $X/sub_ids.txt) follow $f xheld $g" | tee -a $X/jobs.txt $O/stages.txt ;;
explore-pub)   # explore-pub <J,..|-> <C>: 补记 D (c) public 1491 @10 for the J tags meeting the held lines + C, then wx-xsum (after every read)
  X=$O/explore; C=${3:?}; rr=()
  if [[ $2 != - ]]; then for T in ${2//,/ } $C; do [[ ! -e $X/read10/$T ]] || { echo "$X/read10/$T exists" >&2; exit 1; }
    for i in $(seq 0 9); do rr+=($(sub wx-g10-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $OL/lists/all_n10/c$i.txt 10 $X/read10/$T/b$i)) || exit 1; done; done; fi
  JS=$(python3 -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['arms']))" $X/held.json) || exit 1
  s=$(sub wx-xsum --after "$(cat $X/sub_ids.txt)${rr:+,$(csv ${rr[@]})}" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY xsum --j $JS --c $C) || exit 1
  echo "explore-pub ($2): ${#rr[@]} public reads, xsum $s" | tee -a $X/jobs.txt $O/stages.txt ;;
explore-e)   # prereg 补记 E: every closed-loop read is the 补记 D instrument (readlist static @10, rm_chain held @18, pai_box stop17 in the PAI config)
  XE=$O/explore_e; X=$O/explore; mkdir -p $XE/static $XE/paisub; [[ ! -e $XE/jobs.txt ]] || { echo "$XE already submitted" >&2; exit 1; }
  st() {  # st <tag> <label> [after] [env K=V]: the static public scene @10 with the frame dump -> echo id
    local T=$1 L=$2 A=${3:-} e=(); [[ -n ${4:-} ]] && e=(--env $4)
    sub xe-static-$L --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_DUMP=1 "${e[@]}" ${A:+--after $A} \
      -- bash experiments/onpol_long/scripts/ol_chain.sh readlist $T $X/static.txt 10 $XE/static/$L/run; }
  p0=$(st P0 P0) || exit 1; ss=($p0); declare -A sid
  for T in P2-F-s0 RB0-F-s0 RSG-L18-s0; do sid[$T]=$(st $T $T) || exit 1; ss+=(${sid[$T]}); done
  for T in WTCP-L18-s0 WTJ2P-L18-s0 WTJ5P-L18-s0 P2-F-s0 RB0-F-s0 RSG-L18-s0; do ss+=($(st $T $T-vhzero "" JEV_VHIST=zero)) || exit 1; done
  hh=(); for T in P0 P2-F-s0; do A=; [[ $T == P0 ]] && A=$p0
    for i in 0 1 2 3 4 5; do hh+=($(sub xe-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 ${A:+--after $A} \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done; done
  pp=(); for T in P0 P2-F-s0 RB0-F-s0 RSG-L18-s0; do A=; [[ $T == P0 ]] && A=$p0
    AFTER=$A CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $XE/paisub/$T $T $X/stop17.txt > $XE/paisub/$T.submit.txt || exit 1
    p=$(awk '{print $1}' $XE/paisub/$T.submit.txt | grep -E '^[0-9]{4}-' | paste -sd,); [[ -n $p ]] || { echo "no PAI stop17 job for $T" >&2; exit 1; }; pp+=($p); done
  rr=(); for T in WTCP-L18-s0 WTJ2P-L18-s0 WTJ5P-L18-s0; do rr+=($(sub xe-replay-$T --vram 10 --cpu 4 --ram 30 --timeout-h 1 -- bash $ME xe-replay $T $X/sub/$T/run)) || exit 1; done
  for T in RB0-F-s0 RSG-L18-s0; do rr+=($(sub xe-replay-$T --vram 10 --cpu 4 --ram 30 --timeout-h 1 --after ${sid[$T]} -- bash $ME xe-replay $T $XE/static/$T/run)) || exit 1; done
  lg=$(sub xe-logs --after "$(csv ${ss[@]})" --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME xe-logs) || exit 1
  hc=$(sub xe-heldca --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- bash $ME xe-heldca) || exit 1
  of=$(sub xe-offroad --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME xe-offroad) || exit 1
  d=$(sub xe-done --after "$(csv ${ss[@]} ${hh[@]} ${pp[@]} ${rr[@]}),$lg,$hc,$of" --vram 0.1 --cpu 1 --ram 1 --timeout-h 1 -- touch $XE/FINISHED) || exit 1
  echo "explore-e: static $(csv ${ss[@]}) held $(csv ${hh[@]}) stop17 $(csv ${pp[@]}) replay $(csv ${rr[@]}) logs $lg heldca $hc offroad $of done $d" | tee -a $XE/jobs.txt $O/stages.txt ;;
xe-replay)   # xe-replay <tag> <run dir>: the static scene's stored driver messages through the driver class again (c1_replay, open loop on the logged
             # frames and states): JEV_VHIST off (= the reproduction of the logged plans) and zero (the adapter fed a standing car's speed history)
  T=$2; D=$O/explore_e/replay/$T; mkdir -p $D; echo $STATIC > $D/scene.txt; echo "{\"$STATIC\": {\"all\": true, \"frames\": true}}" > $D/spec.json
  $AP experiments/alpasim/scripts/c1_extract.py msgs --run $3 --scenes $D/scene.txt --out $D/msgs --jobs 1 || exit 1
  for m in off zero; do env SH30_TAG=$T JEV_VHIST=$m ALPASIM_SRC=$DATA_DIR/third_party/alpasim OMP_NUM_THREADS=2 \
      $TR experiments/alpasim/scripts/c1_replay.py --driver sh30 --msgs $D/msgs --spec $D/spec.json --out $D/replay_$m.pkl || exit 1; done ;;
xe-logs)     # c1_extract logs + frames of every 补记 D / E static-scene run -> explore_e/logs/<label>/
  for R in $O/explore_e/static/*/run $O/explore/sub/*/run; do L=$(basename $(dirname $R))
    ls $R/rollouts/$STATIC/*/rollout.asl > /dev/null 2>&1 || { echo "$L: rollout.asl pruned (scored > 0), skipped"; continue; }; mkdir -p $O/explore_e/logs/$L
    $AP experiments/alpasim/scripts/c1_extract.py logs --run $R --out $O/explore_e/logs/$L/logs.pkl --jobs 1 || exit 1
    $AP experiments/alpasim/scripts/c1_extract.py frames --run $R --scenes $O/explore/static.txt --out $O/explore_e/logs/$L/frames.pkl --jobs 1 || exit 1; done ;;
xe-xreplay)  # every stage's plan on P0's own closed-loop static-scene inputs (its SH30_DUMP), measured and zeroed speed history -> explore_e/out/xreplay.json
  exec $TR experiments/wait_cap/scripts/wt_xe.py xreplay ;;
xe-heldca)   # cap_audit collect, measures unchanged, on the held 460 @18 reads of WTJ2P / WTCP (YR10m10-F-s0 is the collect's scene-config label)
  CA_EXTRA='{"WTJ2P-L18-s0": ["route_mem/read/WTJ2P-L18-s0/c*/run"], "WTCP-L18-s0": ["route_mem/read/WTCP-L18-s0/c*/run"]}' \
  CA_ONLY=YR10m10-F-s0,WTJ2P-L18-s0,WTCP-L18-s0 exec $AP experiments/cap_audit/scripts/ca.py collect --set held --workers 32 ;;
xe-offroad)  # the J2P-only offroad zeros on held 460 @18 (offroad in J2P, not in WTCP): per J2P chunk, c1_extract logs (zero rollouts keep their asl) and map
  F=$O/explore_e/offroad; mkdir -p $F
  python3 - $DATA_DIR/runs/route_mem/read $F <<'PY' || exit 1
import glob, json, sys
R, F = sys.argv[1], sys.argv[2]
def summ(t):
    o = {}
    for d in sorted(glob.glob(f"{R}/{t}/c*/run")):
        for r in json.load(open(f"{d}/aggregate/results-summary.json"))["rollouts"]:
            o.setdefault(r["clipgt_id"], (d, float((r.get("metrics") or {}).get("offroad") or 0), r.get("score")))
    return o
J, C = summ("WTJ2P-L18-s0"), summ("WTCP-L18-s0")
new = sorted(s for s, v in J.items() if v[1] > 0 and s in C and not C[s][1] > 0)
by = {}
for s in new:
    by.setdefault(J[s][0], []).append(s)
for d, ss in by.items():
    open(f"{F}/{d.split('/')[-2]}.txt", "w").write("\n".join(ss) + "\n")
json.dump(dict(j_only=new, c_only=sorted(s for s, v in C.items() if v[1] > 0 and s in J and not J[s][1] > 0)), open(f"{F}/scenes.json", "w"), indent=1)
print(len(new), "J2P-only offroad scenes in", len(by), "chunks")
PY
  for f in $F/c*.txt; do c=$(basename $f .txt); R=$DATA_DIR/runs/route_mem/read/WTJ2P-L18-s0/$c/run
    $AP experiments/alpasim/scripts/c1_extract.py map --run $R --scenes $f --out $F/map_$c.pkl --radius 120 || exit 1
    $AP experiments/alpasim/scripts/c1_extract.py logs --run $R --out $F/logs_$c.pkl --jobs 8 || exit 1; done ;;
wd-teacher)
  ids=(); for i in $(seq 0 11); do ids+=($(sub wd-teacher-s$i --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $DT teacher --data navtrain_full.s${i}of12)) || exit 1; done
  echo "wd-teacher: $(csv ${ids[@]})" | tee -a $O/stages.txt ;;
wd-codecheck)
  rev=${2:?rev}; W=$DATA_DIR/tmp/wd_old_$rev; [[ -d $W ]] || git worktree add --detach $W $rev > /dev/null || exit 1
  a=$(sub wd-gg-old "${FW[@]}" --timeout-h 1 -- bash $W/experiments/wait_cap/scripts/wt.sh train off 0 60 WDGG-old) || exit 1
  b=$(sub wd-gg-new "${FW[@]}" --timeout-h 1 -- bash $ME train off 0 60 WDGG-new) || exit 1
  c=$(sub wd-codecmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a WDGG-old --b WDGG-new --out wd_code_identity \
      --note "RSG-L18 recipe 60 steps: code $rev (before D1 / D2) vs this code with D1 / D2 off") || exit 1
  echo "wd-codecheck: old $a new $b bitcmp $c" | tee -a $O/stages.txt ;;
wd-launch)
  l=$(sub wd-launch --vram 0.1 --cpu 1 --ram 2 --timeout-h 48 --when-exists $O/pilot_pick.txt ${2:+--after $2} --env PRIO=${PRIO:-9} -- bash $ME wd-pilot) || exit 1
  d=$(sub wd-deadline --vram 0.1 --cpu 1 --ram 1 --timeout-h 26 --when-exists $O/pilot_pick.txt --env PRIO=${PRIO:-9} -- bash $ME wd-deadline) || exit 1
  echo "wd-launch: $l (waits on $O/pilot_pick.txt${2:+, after $2}); deadline job $d" | tee -a $O/stages.txt ;;
wd-deadline)   # Amendment C 5: 24 h after the WAIT pilot verdict (pilot_gate.json's mtime), the WAIT-only formal unless a composed arm passed
  w=$(( $(stat -c %Y $O/pilot_gate.json) + 86400 - $(date +%s) )); (( w > 0 )) && sleep $w
  q=$(cat $O/pilot_pick.txt); [[ $q == j2 || $q == j5 ]] || { echo "deadline: WAIT pilot $q, no formal"; exit 0; }
  [[ -e $O/WD_PASS ]] && { echo "deadline: a composed arm passed ($(cat $O/WD_PASS)), no fallback"; exit 0; }
  [[ -e $O/FALLBACK ]] && { echo "deadline: fallback already run ($(cat $O/FALLBACK))"; exit 0; }
  echo "deadline $(date -Is)" > $O/FALLBACK; echo "deadline: no composed arm passed within 24 h -> WAIT-only formal $q" | tee -a $O/stages.txt
  WT_FALLBACK=1 exec bash $ME full $q ;;
wd-pilot)
  q=$(cat $O/pilot_pick.txt); [[ $q == j2 || $q == j5 ]] || { echo "WAIT pilot verdict $q: no composed pilot, no formal (Amendment C 5)" | tee -a $O/stages.txt; exit 0; }
  r=$(sub wd-rows --vram 0.5 --cpu 4 --ram 64 --timeout-h 24 --when-exists $PWD/lib/route_desire.py -- $TR $DT rows) || exit 1   # the D0 line's rule
  g=$(sub wd-gg-w0 "${FW[@]}" --timeout-h 1 -- bash $ME train wdw0 0 60 WDGG-w0) || exit 1
  b=$(sub wd-bitcmp --after $g --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a WTGG-off --b WDGG-w0 --out wd_identity \
      --note "Amendment C 2: RSG-L18 recipe vs the composed command with WAIT rows at weight 0, D1 0 rows, D2 weight 0") || exit 1
  TT=(WD${q^^}P1-L18-s0 WD${q^^}P3-L18-s0); ids=()
  for L in 1 3; do T=WD${q^^}P$L-L18-s0; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    ids+=($(sub wd-train-$T "${FW[@]}" --after $r,$b --timeout-h 1 -- bash $ME train wd$L 0 1000 $T)) || exit 1; done
  rj=$(reads "$(csv ${ids[@]})" ${TT[@]}) || exit 1
  dd=$(sub wd-dc-${TT[0]} --after "$(csv ${ids[@]})" --vram 16 --cpu 4 --ram 40 --timeout-h 1 -- $TR $DC plans --tags ${TT[@]} --srcs nav pai) || exit 1
  gg=$(sub wd-gate-pilot --after $rj,$dd --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- \
       $TR $PY wdgate --j ${TT[@]} --c WTCP-L18-s0 --lam-of 1=${TT[0]},3=${TT[1]} --submit) || exit 1
  echo "wd-pilot ($q): rows $r identity $g bitcmp $b trainings ${ids[*]} reads $rj dc $dd gate $gg" | tee -a $O/stages.txt ;;
wd-full)   # wd-full <lam>
  L=${2:?lam}; q=$(cat $O/pilot_pick.txt); J=WD${q^^}-L18; JJ=($J-s0 $J-s1); CC=(WTC-L18-s0 WTC-L18-s1); tr=()
  for s in 0 1; do T=$J-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    tr+=($(sub wd-train-$T "${FW[@]}" --timeout-h 2 -- bash $ME train wd$L $s 2500 $T)) || exit 1; done
  rj=$(reads "$(csv ${tr[@]})" ${JJ[@]}) || exit 1
  dd=$(sub wd-dc-$J --after "$(csv ${tr[@]})" --vram 16 --cpu 4 --ram 40 --timeout-h 1 -- $TR $DC plans --tags ${JJ[@]} --srcs nav pai) || exit 1
  hj=$(held ${tr[0]} ${JJ[0]}),$(held ${tr[1]} ${JJ[1]}) || exit 1
  hc=(); for T in ${CC[@]}; do for i in 0 1 2 3 4 5; do hc+=($(jid wt-held-$T-$i)) || { echo "control held reads not submitted" >&2; exit 1; }; done; done
  f=$(sub wd-follow-$J --after "$hj,$(csv ${hc[@]})" --vram 0.1 --cpu 16 --ram 100 --timeout-h 2 -- bash $ME follow $(csv ${JJ[@]}) $(csv ${CC[@]}) $O/wd_heldturn) || exit 1
  g=$(sub wd-clgate-held --after "$hj,$(csv ${hc[@]}),$f" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- \
      $TR $PY clgate --stage held --j $(csv ${JJ[@]}) --c $(csv ${CC[@]}) --follow $O/wd_heldturn --submit) || exit 1
  echo "wd-full (D2 $L, WAIT $q): trainings ${tr[*]} reads $rj dc $dd held $hj follow $f gate $g" | tee -a $O/stages.txt ;;
follow)    # follow J0,J1 C0,C1 DIR: rf_follow on the held 460 @18 reads (rm_chain.sh read dirs) of the four tags
  R=(); for T in ${2//,/ } ${3//,/ }; do R+=("$T@18=$(ls -d $DATA_DIR/runs/route_mem/read/$T/c*/run | paste -sd,)"); done
  mkdir -p $4; exec $AP experiments/route_follow/scripts/rf_follow.py --runs "${R[@]}" --out $4/follow --workers 16 ;;
train)   # train <off|w0|j2|j5|wdw0|wd1|wd3> <seed> <steps> <tag>
  case $2 in
    off) F="$RB" ;;
    wdw0) F="$(cat $O/flags_w0.txt) --d1-k 0 --d2-lam 0" ;;                               # Amendment C 2: D1 / D2 off
    wd*) q=$(cat $O/pilot_pick.txt); f=$O/flags_$q.txt; [[ -s $f && -s $D1R ]] || { echo "no $f or $D1R" >&2; exit 1; }
         F="$(cat $f) --desire-rows $D1R --d1-k $D1K --d2-lam ${2#wd} --d2-teacher $TEA" ;;
    *) f=$O/flags_$2.txt; [[ -s $f ]] || { echo "no $f (wt.py pack)" >&2; exit 1; }; F=$(cat $f) ;;
  esac
  PYTHONPATH=$PWD/experiments/wait_cap/lib${PYTHONPATH:+:$PYTHONPATH} exec $TR experiments/route_mem/scripts/rm_train.py train --tag $5 $REC --lr-scale 0.1 $F --steps $4 --seed $3 ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
