#!/usr/bin/env bash
# RECOVERY-ENVELOPE post-training chain (prereg plans/2026-10-16-recovery-envelope-prereg.md, 全量读数与线). Per arm, after its trainings:
# held 460 @18 (rm_chain.sh read, 6 chunks per seed, rf_onk.sh's way), public 1491 @10 guard (ol_chain.sh readlist, 10 chunks per seed,
# rf_slow.sh's way), PAI 281 for seed 0 on the box (pai_box.sh run per chunk: gate serve = JEV_TRACK unset, gains 20, JEV_VCONT 1.0 JEV_LEAD 1,
# the Tokyo reference r281/rsg's own 8 chunk lists), the PAI extracts, then the lines report (rf_recov_post.py lines). Box, repo root.
# Every job goes through the pool, --after chained; ids in $O/<ARM>/jobs.txt.
#   rf_recov_post.sh fetch          pool network job: the PAI 281 scenes missing from the box scene cache (hf-mirror, pai_fetch.sh) -> $O/post/fetch.id
#   rf_recov_post.sh ref-pai        the same-host PAI reference: RSG-L18-s0 on the 281, identical box path, + its extracts -> $O/post/ref_paix.id
#   rf_recov_post.sh submit <ARM> <TAG_s0> <TAG_s1|-> <train_job_s0> <train_job_s1|->     the whole arm, --after its training jobs
#   rf_recov_post.sh post <ARM> <TAG_s0> <TAG_s1|->    a report job now, no dependencies (rerun after a failure / partial report)
#   rf_recov_post.sh cost           job count and estimated job-hours per arm (+ the one-off reference)
#   job bodies: report <ARM> <TAG_s0> <TAG_s1|-> | paix <PAI read dir> | fetchjob
# Env: KIND (R | L | RL; default from the arm name: RCVRL* -> RL, RCVR* -> R, RCVL* -> L), PRIO (pool priority, 6), DRY=1 (submit / ref-pai /
#   fetch: print the job specs, submit nothing), report: PAI_ARM / PAI_REF (PAI read dirs, default $O/pai/<TAG_s0> / $O/pai/RSG-L18-s0),
#   NOCOPY=1 (no copy into experiments/route_follow/results/recovery_envelope/<ARM>/).
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/recovery_envelope; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long; S=experiments/route_follow/scripts; ME=$S/rf_recov_post.sh
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; PB=experiments/alpasim/scripts/pai_box.sh
CH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks          # the Tokyo reference's 8 chunk lists (36 scenes; the simulator repeats a list)
STRICT=experiments/route_follow/results/pai281/strict36.txt; WL=$O/pai_wl.pkl; REFT=RSG-L18-s0
WIZ=controller.gains.rel_front_steering_angle_weight=20
X=$O/post; mkdir -p $X                               # one-off jobs (fetch, reference) of this script; $O is shared with the training half
kind() { [[ -n ${KIND:-} ]] && { echo $KIND; return; }; case $1 in RCVRL*) echo RL ;; RCVR*) echo R ;; RCVL*) echo L ;; *) echo "arm $1: set KIND" >&2; return 1 ;; esac; }
sub() { local log=$1 name=$2; shift 2; local id
  if [[ ${DRY:-0} == 1 ]]; then echo "DRY $name: cl submit --owner recovery-envelope --name $name --tries ${TRIES:-1} --priority ${PRIO:-6} $*" >&2; echo dry:$name; return; fi
  id=$(.venv/bin/python -m jevdrive.cl submit --owner recovery-envelope --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-6} --no-check \
       --log-dir $(dirname $log)/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }
  echo "$id $name" >> $log; echo "$id"; }
jstate() { .venv/bin/python - "$@" <<'PY'
import sys, time
from jevdrive.cl import pool as P
for _ in range(12):     # the dispatcher unlinks an inbox file before it saves state.json: a fresh id can be in neither for a few s
    s = P.job_states(sys.argv[1:])
    if "unknown" not in s.values():
        break
    time.sleep(5)
for i, v in s.items():
    print(i, v)
PY
}
okafter() {  # ids...: every id must be a known pool job that has not failed / been cancelled
  local bad; bad=$(jstate "$@" | awk '$2 == "unknown" || $2 == "failed" || $2 == "cancelled"') || return 1
  [[ -z $bad ]] || { echo "refused: --after job(s) unknown / failed / cancelled: $bad" >&2; return 1; }; }
fetchdep() {  # the fetch job id when the cache still misses scenes (the PAI jobs wait for it); refuses a failed fetch
  local n; n=$(python3 $S/rf_recov_post.py missing --scenes $DATA_DIR/third_party/alpasim/data/scenes --cache $DATA_DIR/datasets/nurec/all-usdzs --chunks $CH/*.txt)
  (( n == 0 )) && return 0
  [[ -s $X/fetch.id ]] || { echo "$n PAI scenes missing from the cache and no fetch job: run rf_recov_post.sh fetch" >&2; return 1; }
  okafter $(cat $X/fetch.id) || return 1; cat $X/fetch.id; }
held_ok() { local i; for i in 0 1 2 3 4 5; do [[ -f $M/$1/c$i/run/aggregate/results-summary.json ]] || return 1; done; }
hdirs() { ls -d $M/$1/c*/run | paste -sd,; }
pdir() { local r i; for r in $O/read10 $RF/guard/read10 $RF/slow/read10 $RF/allrows/read10; do   # the first root holding all 10 public chunks of the tag
  for i in $(seq 0 9); do [[ -f $r/$1/b$i/aggregate/results-summary.json ]] || continue 2; done; echo $r/$1; return 0; done; return 1; }
pai_ok() { local n; n=$(ls $1/chunks/*.txt 2>/dev/null | wc -l); (( n > 0 )) && (( $(ls $1/runs/*/DONE 2>/dev/null | wc -l) == n )); }
pai_submit() {  # <read dir> <tag> <log> <after ids>: one pool job per chunk (pai_box.sh run; pai_box.sh submit's job spec), echoes the ids
  local R=$1 T=$2 log=$3 af=$4 f c r id ids=() C=$1/chunks
  if [[ ${DRY:-0} == 1 ]]; then C=$CH; else
    mkdir -p $R/chunks $R/runs; ls $R/chunks/*.txt > /dev/null 2>&1 || cp $CH/*.txt $R/chunks/
    printf '%s\n' "SERVE=JEV_VCONT=1.0 JEV_LEAD=1" "WIZ=$WIZ" "CONC=4" "KEEP_VIDEO=0" "KEEP_ASL=0" "code=$(git rev-parse --short HEAD)" "chunks=$CH" > $R/config.txt; fi
  for f in $C/*.txt; do c=$(basename $f .txt); r=$R/runs/${T,,}_$c
    [[ -f $r/DONE ]] && continue
    [[ -f $r/SUBMITTED && ! -f $r/ERROR ]] && { ids+=($(cat $r/SUBMITTED)); continue; }
    id=$(sub $log pai-${T,,}-$c ${af:+--after $af} --vram 28 --cpu 6 --ram 40 --timeout-h 3 --env CL_HIST_KEY=pai-read4 --env CONC=4 --env "WIZ=$WIZ" \
         --env KEEP_VIDEO=0 --env KEEP_ASL=0 --env JEV_VCONT=1.0 --env JEV_LEAD=1 -- bash $PB run $R $T $c) || exit 1
    [[ ${DRY:-0} == 1 ]] || { mkdir -p $r; echo $id > $r/SUBMITTED; }; ids+=($id)
  done
  (IFS=,; echo "${ids[*]}"); }
case ${1:-} in
fetch)
  n=$(python3 $S/rf_recov_post.py missing --scenes $DATA_DIR/third_party/alpasim/data/scenes --cache $DATA_DIR/datasets/nurec/all-usdzs --chunks $CH/*.txt --tsv $X/fetch281.tsv)
  (( n > 0 )) || { echo "every PAI 281 scene is in the cache"; exit 0; }
  if [[ -s $X/fetch.id ]] && jstate $(cat $X/fetch.id) | grep -qE ' (queued|running|inbox)$'; then echo "fetch job $(cat $X/fetch.id) still pending"; exit 0; fi
  id=$(TRIES=2 sub $X/jobs.txt recov-fetch --vram 0.5 --cpu 2 --ram 4 --timeout-h 10 -- bash $ME fetchjob) || exit 1
  [[ ${DRY:-0} == 1 ]] || echo $id > $X/fetch.id; echo "$n scenes ($X/fetch281.tsv), fetch job $id" ;;
fetchjob)    # up to 3 passes of pai_fetch.sh from hf-mirror (direct, no proxy: docs/alpasim.md), rc 0 only when no scene of the 281 is missing
  for pass in 1 2 3; do
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY HFROOT=https://hf-mirror.com \
      bash experiments/alpasim/scripts/pai_fetch.sh $X/fetch281.tsv 1 4
    n=$(python3 $S/rf_recov_post.py missing --scenes $DATA_DIR/third_party/alpasim/data/scenes --cache $DATA_DIR/datasets/nurec/all-usdzs --chunks $CH/*.txt --tsv $X/fetch281.tsv)
    echo "$(date '+%F %T') pass $pass: $n scenes missing"; (( n == 0 )) && exit 0
  done; exit 1 ;;
ref-pai)     # RSG-L18-s0 on the 281, the identical box path; its extracts -> $X/ref_paix.id (each arm's report waits for it)
  [[ -s $WL ]] || { echo "no $WL (the stored PAI wait lines, Mac tmp/tax/pai_wl.pkl): copy it first" >&2; exit 1; }
  af=$(fetchdep) || exit 1; R=$O/pai/$REFT
  ids=$(pai_submit $R $REFT $R/jobs.txt "$af") || exit 1
  x=$(sub $R/jobs.txt paix-${REFT,,} ${ids:+--after $ids} --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME paix $R) || exit 1
  [[ ${DRY:-0} == 1 ]] || echo $x > $X/ref_paix.id; echo "reference PAI read $R: chunks ${ids:-all done}, extracts $x" ;;
submit)
  ARM=${2:?ARM}; T0=${3:?TAG_s0}; T1=${4:?TAG_s1 or -}; J0=${5:?train job s0}; J1=${6:?train job s1 or -}; K=$(kind $ARM) || exit 1
  [[ $T1 == - ]] && J1=-; TAGS=($T0); JOBS=($J0); [[ $T1 != - ]] && { TAGS+=($T1); JOBS+=($J1); }
  [[ $ARM =~ ^[A-Za-z0-9_.-]+$ && $T0 != $T1 ]] || { echo "bad arm / tags" >&2; exit 1; }
  for T in ${TAGS[@]}; do [[ ! -e $M/$T && ! -e $O/read10/$T ]] || { echo "$T: reads exist ($M/$T or $O/read10/$T): never overwritten" >&2; exit 1; }; done
  [[ ! -e $O/pai/$T0 ]] || { echo "$O/pai/$T0 exists: never overwritten" >&2; exit 1; }
  okafter ${JOBS[@]} || exit 1
  [[ -s $WL ]] || { echo "no $WL: copy the stored PAI wait lines first" >&2; exit 1; }
  af=$(fetchdep) || exit 1
  A=$O/$ARM; LOG=$A/jobs.txt; rr=(); [[ ! -e $LOG ]] || { echo "$ARM already submitted ($LOG)" >&2; exit 1; }; [[ ${DRY:-0} == 1 ]] || mkdir -p $A/pool
  for k in ${!TAGS[@]}; do T=${TAGS[$k]}; t=${JOBS[$k]}
    for i in 0 1 2 3 4 5; do
      rr+=($(sub $LOG rcv-held-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
             -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1
    done
    for i in $(seq 0 9); do
      rr+=($(sub $LOG rcv-g10-$T-$i --after $t --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
             -- bash $E/scripts/ol_chain.sh readlist $T $W/lists/all_n10/c$i.txt 10 $O/read10/$T/b$i)) || exit 1
    done
  done
  pids=$(pai_submit $O/pai/$T0 $T0 $LOG "$J0${af:+,$af}") || exit 1
  x=$(sub $LOG paix-${T0,,} --after $pids --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME paix $O/pai/$T0) || exit 1
  dep="$(IFS=,; echo "${rr[*]}"),$x"; [[ -s $X/ref_paix.id ]] && dep="$dep,$(cat $X/ref_paix.id)" || echo "warning: no reference PAI job ($X/ref_paix.id): PAI lines stay pending until ref-pai + post" >&2
  r=$(sub $LOG rcv-report-$ARM --after $dep --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report $ARM $T0 $T1) || exit 1
  echo "$ARM ($K): ${#rr[@]} held / public read jobs, $(echo $pids | tr , '\n' | grep -c .) PAI chunk jobs, extracts $x, report $r"
  [[ ${DRY:-0} == 1 ]] && echo "DRY: job specs above, nothing submitted" || echo "ids in $LOG" ;;
post)
  ARM=${2:?ARM}; T0=${3:?TAG_s0}; T1=${4:-}; kind $ARM > /dev/null || exit 1; mkdir -p $O/$ARM/pool
  sub $O/$ARM/jobs.txt rcv-report-$ARM --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report $ARM $T0 ${T1:--} ;;
paix)        # <PAI read dir>: rf_tax extract (--asl-actors, AlpaSim venv), rf_londiag slim, rf_pai_follow extract (label x), serving path
  R=${2:?read dir}; pai_ok $R || { echo "$R: not every chunk DONE" >&2; exit 1; }
  [[ -s $R/tax.pkl ]] || { $P $S/rf_tax.py extract --runs $R/runs/*/sim --out $R/tax.tmp --asl-actors --workers 16 && mv $R/tax.tmp $R/tax.pkl; } || exit 1
  [[ -s $R/dec.pkl ]] || { $P $S/rf_londiag.py slim --runs $R/runs/* --out $R/dec.tmp --workers 16 && mv $R/dec.tmp $R/dec.pkl; } || exit 1
  [[ -s $R/follow.pkl ]] || { $P $S/rf_pai_follow.py extract --runs x=$R --out $R/follow.tmp --workers 16 && mv $R/follow.tmp $R/follow.pkl; } || exit 1
  python3 $S/rf_recov_post.py track --read $R > $R/track.txt; echo "$R: extracts done, serving path $(cat $R/track.txt)" ;;
report)
  ARM=${2:?ARM}; T0=${3:?TAG_s0}; T1=${4:--}; K=$(kind $ARM) || exit 1; A=$O/$ARM; mkdir -p $A
  ht=(); pt=(); for T in $T0 $T1; do [[ $T == - ]] && continue; held_ok $T && ht+=($T); pdir $T > /dev/null && pt+=($T); done
  echo "held complete: ${ht[*]:-none}; public complete: ${pt[*]:-none}"
  REF=(RSG-L18-s0 RSG-L18-s1); rc=0
  if (( ${#ht[@]} )); then
    H=(); F=(); for T in ${REF[@]} ${ht[@]}; do H+=("$T@held18=$(hdirs $T)"); F+=("$T@18=$(hdirs $T)"); done
    $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --mean "RSG-L18=RSG-L18-s0,RSG-L18-s1" "$ARM=$(IFS=,; echo "${ht[*]}")" \
      --pairs "$ARM:RSG-L18" --out $A/held18 > $A/held18.log 2>&1 || { echo "held confirm failed ($A/held18.log)"; rc=1; }
    $P $S/rf_follow.py --runs "${F[@]}" --out $A/follow --workers 16 > $A/follow.log 2>&1 || { echo "rf_follow failed ($A/follow.log)"; rc=1; }
  fi
  if (( ${#pt[@]} )); then
    AL=$DATA_DIR/runs/alpasim; G=()
    G+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$AL/ot3/a/manifest.json')); b = json.load(open('$AL/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
    for T in ${REF[@]} ${pt[@]}; do r=$(pdir $T) || { echo "no public read of $T"; exit 1; }; G+=("$T@all10=$(ls -d $r/b*/ | sed 's#/$##' | paste -sd,)"); done
    $P $E/scripts/ol_confirm.py --runs "${G[@]}" --base YR10m10-F-s0 --mean "RSG-L18=RSG-L18-s0,RSG-L18-s1" "$ARM=$(IFS=,; echo "${pt[*]}")" \
      --pairs "$ARM:RSG-L18" --out $A/confirm > $A/confirm.log 2>&1 || { echo "guard confirm failed ($A/confirm.log)"; rc=1; }
  fi
  XA=${PAI_ARM:-$O/pai/$T0}; XR=${PAI_REF:-$O/pai/$REFT}
  if [[ -s $XA/tax.pkl && -s $XA/follow.pkl && -s $XR/tax.pkl && -s $XR/follow.pkl ]]; then
    printf '{"arm": "%s", "ref": "%s", "arm_dir": "%s", "ref_dir": "%s"}\n' "$(cat $XA/track.txt)" "$(cat $XR/track.txt)" $XA $XR > $A/pai_track.json
    $P $S/rf_londiag.py pai --tax $XA/tax.pkl --dec $XA/dec.pkl --ref-tax $XR/tax.pkl --ref-dec $XR/dec.pkl --wl $WL --workers 16 --out $A/pai_londiag.json \
      > $A/pai_londiag.txt 2>&1 || { echo "rf_londiag pai failed ($A/pai_londiag.txt)"; rc=1; }
    $P $S/rf_v2pai.py turns --pkl "ref=$XR/follow.pkl:x" "arm=$XA/follow.pkl:x" --scenes $STRICT --out $A/pai_turns.json > $A/pai_turns.log 2>&1 \
      && $P $S/rf_v2pai.py table --turns $A/pai_turns.json --a ref --b arm --strict $STRICT --md $A/pai_turns.md >> $A/pai_turns.log 2>&1 \
      || { echo "rf_v2pai turns failed ($A/pai_turns.log)"; rc=1; }
  else echo "PAI pending: extracts of $XA and / or $XR missing"; fi
  TT=($T0); [[ $T1 != - ]] && TT+=($T1)
  $P $S/rf_recov_post.py lines --arm $ARM --kind $K --tags ${TT[@]} --dir $A --strict $STRICT \
    --note "Held complete: ${ht[*]:-none}; public complete: ${pt[*]:-none}." || exit 1
  if [[ ${NOCOPY:-0} != 1 ]]; then D=experiments/route_follow/results/recovery_envelope/$ARM; mkdir -p $D; cp $A/lines.md $A/lines.json $D/; echo "-> $D"; fi
  exit $rc ;;
cost)
  cat <<'EOF'
Per arm (2 seeds; PAI seed 0), measured unit times: held chunk ~11 min, public chunk ~11 min (SLOW round); PAI 36-scene chunk ~55 min
(box pai_box reads 2026-10-13: 84-106 s per scene incl. start-up, one stack per job, CONC 4; decision-free extracts ~20 min on 16 cores).
  held 460 @18      2 x 6  = 12 GPU jobs x 0.18 h = 2.2 job-h   (--vram 20 --cpu 6)
  public 1491 @10   2 x 10 = 20 GPU jobs x 0.18 h = 3.7 job-h   (--vram 20 --cpu 6)
  PAI 281 seed 0    8 GPU jobs x ~0.9 h            = ~7 job-h    (--vram 28 --cpu 6; 281 scenes x ~90 s)
  PAI extracts      1 CPU job (16 cores)           ~0.3-0.5 h
  report            1 CPU job (16 cores)           ~0.3-0.5 h (ol_confirm x 2, rf_follow on 4 labels x 452, rf_londiag pai, rf_v2pai turns)
  total             42 jobs, ~13 GPU job-h; wall on 4 idle cards (3 read stacks each) ~1.5-2 h after the training
One-off: ref-pai = 8 GPU jobs (~7 job-h, ~1 h wall) + 1 extract job; fetch = 1 network job (118 scenes x ~1.8 GB = ~215 GB, ~3-5 h at 4 parallel).
EOF
  ;;
*) sed -n 2,17p $0; exit 2 ;;
esac
