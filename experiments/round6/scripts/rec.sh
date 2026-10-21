#!/usr/bin/env bash
# ROUND6 line B: perturbation-recovery rows (plans/2026-10-21-recovery-rows.md). Box, repo root; every job through the pool at PRIO 22 (PRIO env).
# State in $O = $DATA_DIR/runs/round6/rec (jobs.txt, stages.txt, eps/, roll/<pass>/<chunk>/, rows/, cache/, select/, fwd/, checks/, flags).
# Passes: A, B, C = three rollouts of every eligible on1_PAIH scene with different episodes (lib/rec_driver.py, REC_EPS = $O/eps/eps_<pass>.json),
# Z = the 11 probe scenes with A's episodes at zero offset (G1). Rollouts as pai_rows' H read (NTV0-L18-s0, A0 serve, log driver, g10, ground
# truth on, KEEP_ASL, CONC 4) -> onp_rows rows / cold -> clean -> onp_cache build -> rec.py select -> rf_pai_pool --select.
#   rec.sh sched            one CPU job: rec.py sched A, B, C -> $O/SCHED (episodes, chunk lists)
#   rec.sh chain            (after SCHED) staged: a000 / z000 (1 scene) -> check -> a001 / z001 (10) -> check -> rows -> probe job (cache, select,
#                           pools PRECAT / PRECZT, fwd + R1, G1 / G2 / G3 -> results/rec/gates_probe.md; PROBE_OK or STOP) -> rest job (the other
#                           chunks of A and all of B / C, window W, then the tail: cache / select / pools RECA-C / fwd + R1 / G2 G3 on the full
#                           pools / merge on1_JTPHREC + xfix / draw files / rows report -> smoke + identity (G4) -> trainings s0 / s1 -> PAI 281
#                           ACC + GAP each -> report + new breaks -> local PCS ruler -> verdict -> DONE)
#   rec.sh resume           after a probe STOP fixed in code: the probe job and the rest job again
#   rec.sh wait <FLAG>      one blocking waiter (run_in_background): returns at $O/<FLAG>, $O/STOP or a failed / cancelled job of this lane
#   job bodies: sched | check <pass> <chunk> <n> <min> | rows <pass> <chunk> | clean <pass> <chunk> | probejob | rest | cache <pass> |
#               select <pass> | pool <pass> <name> | fwd <name> <rows> | r1 <pass> <name> | gfull | merge | prep | smoketrain | identrun <v> |
#               ident | launch | train <seed> | r281 | pcsrep | verdict
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round6/rec; ME=experiments/round6/scripts/rec.sh; PY=experiments/round6/scripts/rec.py; PRPY=experiments/pai_rows/scripts/pr.py
PP=experiments/practice_pai/scripts/pp.py; RF=experiments/route_follow/scripts; R2PY=experiments/round2/scripts/r2.py; R5PY=experiments/round5/scripts/r5.py
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; V=.venv/bin/python; CRC=$DATA_DIR/runs/op_parity/cache
RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts; RES=experiments/round6/results/rec; DFR=experiments/defect_fix/results
T=NTV0-L18-s0; WRONG=RB0-F-s1; TSV=experiments/pai_rows/results/scenes.tsv; W=${W:-8}; PRIO=${PRIO:-22}; N=9500; K=8
A0SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 PAI_DUMP=100000"; G10=controller.gains.rel_front_steering_angle_weight=10
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both"  # = on1_JTPH
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $O/checks $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
stop() { echo "$(date '+%F %T') $1" | tee $O/STOP >> $O/stages.txt; exit 2; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round6-rec --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name.$(date +%m%d%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
chunks() { ls -d $O/roll/$1/${1,,}[0-9][0-9][0-9] 2>/dev/null | xargs -n1 basename; }
roll() {  # roll <pass> <chunk> [after] -> the chunk's job id
  local p=$1 c=$2 r=$O/roll/$1/$2
  SERVE="$A0SERVE DRV_PY=../../round6/lib/rec_driver.py REC_EPS=$O/eps/eps_$p.json" WIZ="$G10 runtime.simulation_config.send_recording_ground_truth=true" \
    KEEP_ASL=1 CONC=4 PRIO=$PRIO AFTER=${3:-} bash experiments/alpasim/scripts/pai_box.sh submit $r $T > $r/submit.txt || die "roll submit $p $c"
  local id; id=$(awk '$2 ~ /^pai-/ {print $1}' $r/submit.txt); [[ -n $id && $(wc -w <<< "$id") == 1 ]] || die "roll $p $c: no single job"
  echo "$id pai-$p-$c" >> $O/jobs.txt; echo "$id"; }
unit() {  # unit <pass> <chunk> <after roll>: roll -> rows -> clean; prints "roll rows clean"
  local r w c; r=$(roll $1 $2 $3) || exit 1
  w=$(sub rec-rows-$1-$2 --after $r --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $1 $2) || exit 1
  c=$(sub rec-clean-$1-$2 --after $w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $1 $2) || exit 1; echo "$r $w $c"; }
argv() {  # argv <variant> <steps> <tag> <seed>: R5LC-L18-s<seed>'s recorded argv (no --save-at); rec = + on1_JTPHREC / R6REC, on-k 72, batch 136,
          # yr-mass 0.0956 (51 navtrain + 13 yaw-rate + 64 pool + 8 recovery rows); rec0 = on1_JTPHREC / R6REC0 (recovery rows at weight 0, identity)
  local M; M=$(ls -d $DATA_DIR/runs/route_mem/train-R5LC-L18-s$4/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  python3 - "$M" "$@" "$O/w" <<'PY'
import json, sys
m, var, steps, tag, seed, w6 = sys.argv[1:7]
a = json.load(open(m))["argv"]
last = lambda k: len(a) - 1 - a[::-1].index(k)  # noqa: E731
assert a[a.index("--init") + 1] == "NTV0-L18-s0" and a[a.index("--tag") + 1] == f"R5LC-L18-s{seed}" and a[last("--on-wfile") + 1].endswith("/R5LC.npz")
assert a[last("--on") + 1] == "on1_JTPH" and a[last("--on-k") + 1] == "64" and a[last("--seed") + 1] == seed and "--batch" not in a and a[last("--yr-mass") + 1] == "0.1"
a[a.index("--tag") + 1] = tag
a[last("--steps") + 1] = steps
i = a.index("--save-at"); del a[i:i + 2]
if var in ("rec", "rec0"):
    a[last("--on") + 1] = "on1_JTPHREC"
    a[last("--on-wfile") + 1] = f"{w6}/{'R6REC' if var == 'rec' else 'R6REC0'}.npz"
if var == "rec":
    a[last("--on-k") + 1] = "72"
    a[last("--yr-mass") + 1] = "0.0956"
    a += ["--batch", "136"]
print("\n".join(a))
PY
}
run() { local L; L=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$L"; }
read281() {  # read281 <tag> [after]: PAI 281 ACC + GAP as round 5's read281 (NTV0's chunk lists, g10, FAST 1, CONC 4) -> "<chunk ids...> <facts id>"
  local t=$1 R=$O/pai281/$1-ag ids x; mkdir -p $O/pai281
  AFTER=${2:-} CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" \
    WIZ=$G10 CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $t > $R.submit.txt || die "pai submit $t"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$t: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  x=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $t-281ag $R | tail -1) && [[ -n $x ]] || die "tax $t"
  echo "$x pai_fix-tax-$t-281ag" >> $O/jobs.txt; echo $ids $x; }
case ${1:-} in
sched)
  if [[ ${2:-} != run ]]; then j=$(sub rec-sched --vram 0.1 --cpu 32 --ram 64 --timeout-h 2 -- bash $ME sched run "${2:-A B C}") || exit 1
    echo "$(date '+%F %T') sched (code $(git rev-parse --short HEAD)): $j" | tee -a $O/stages.txt; exit 0; fi
  for p in ${3:-A B C}; do $TR $PY sched --pass $p --jobs 32 > $O/sched_$p.log 2>&1 || die "sched $p (log $O/sched_$p.log)"; done; date '+%F %T' > $O/SCHED ;;
chain)
  [[ -f $O/SCHED && ! -f $O/STOP ]] || die "no SCHED (or STOP)"; [[ -z $(grep -E ' (pai-A-a000|rec-probe)$' $O/jobs.txt 2>/dev/null) ]] || die "already chained"
  h=$(git rev-parse --short HEAD); echo "$h chain" >> $O/code.txt
  ra=$(roll A a000) || exit 1; rz=$(roll Z z000) || exit 1
  ka=$(sub rec-chk-a000 --after $ra --vram 0.1 --cpu 2 --ram 4 --timeout-h 1 -- bash $ME check A a000 1 1) || exit 1
  kz=$(sub rec-chk-z000 --after $rz --vram 0.1 --cpu 2 --ram 4 --timeout-h 1 -- bash $ME check Z z000 1 1) || exit 1
  wa=$(sub rec-rows-A-a000 --after $ka --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows A a000) || exit 1
  wz=$(sub rec-rows-Z-z000 --after $kz --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows Z z000) || exit 1
  ra1=$(roll A a001 $ka,$kz) || exit 1; rz1=$(roll Z z001 $ka,$kz) || exit 1
  ka1=$(sub rec-chk-a001 --after $ra1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME check A a001 10 9) || exit 1
  kz1=$(sub rec-chk-z001 --after $rz1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME check Z z001 10 10) || exit 1
  wa1=$(sub rec-rows-A-a001 --after $ka1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows A a001) || exit 1
  wz1=$(sub rec-rows-Z-z001 --after $kz1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows Z z001) || exit 1
  cl=(); for x in "A a000 $wa" "Z z000 $wz" "A a001 $wa1" "Z z001 $wz1"; do set -- $x
    cl+=($(sub rec-clean-$1-$2 --after $3 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $1 $2)) || exit 1; done
  pb=$(sub rec-probe --after $wa,$wz,$wa1,$wz1 --vram 20 --cpu 16 --ram 96 --timeout-h 3 -- bash $ME probejob) || exit 1
  rs=$(sub rec-rest --after $pb --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $h): a000 $ra z000 $rz checks $ka $kz rows $wa $wz; a001 $ra1 z001 $rz1 checks $ka1 $kz1 rows $wa1 $wz1; cleans ${cl[*]};" \
       "probe $pb rest $rs" | tee -a $O/stages.txt ;;
check) $V $PRPY check --root $O/roll/$2/$3 --n $4 --min-ok $5 --log --out $O/checks/$3.json || stop "roll check $3 failed ($O/checks/$3.json)" ;;
rows)  # rows <pass> <chunk>: the chunk's rollouts -> $O/rows/<pass>_<chunk>/<scene>.npz, linked into $O/rows/<pass>/
  p=$2; c=$3; P=${p}_$c; G=$DATA_DIR/tmp/rec_stage/$P; R=$O/rows; mkdir -p $G $R/$p
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $P \
      --runs "$O/roll/$p/$c/runs/*" --out $R --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $P --out $R --stage $G --jobs 8 || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass $P --out $R > $R/report_$P.log 2>&1
  rm -rf $G; for f in $R/$P/*.npz; do [[ -e $f ]] && ln -sfn $f $R/$p/$(basename $f); done
  n=$(ls $R/$P/*.npz 2>/dev/null | wc -l); m=$(wc -w < $O/roll/$p/$c/chunks/$c.txt); echo "$p $c: $n / $m scene files" | tee -a $O/stages.txt
  (( n >= m * 9 / 10 )) || die "rows $p $c: $n of $m scenes" ;;
clean)  # this lane's own rollout.asl and frame dumps of the chunk (its rows are built); drive.jsonl stays (route messages)
  r=$O/roll/$2/$3; n=$(find $r/runs -name rollout.asl | wc -l); find $r/runs -name rollout.asl -delete; rm -rf $r/runs/*/sim/driver-logs/dump
  echo "$2 $3: removed $n rollout.asl and the frame dumps; $(du -sh $r | cut -f1) left; free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
cache)  $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $2 --rows $O/rows/$2 --out $O/cache ;;
select) $TR $PY select --pass $2 --jobs 16 ;;
pool)   # pool <pass> <name>
  p=$2; for x in $CRC/on1_$3 $CRC/on1_$3@warp; do [[ ! -e $x ]] || rm -rf $x; done                          # this lane's own pool (a rerun of this job)
  $TR $RF/rf_pai_pool.py --name $3 --rows $O/rows/$p --cache $O/cache/$p --runs "$O/roll/$p/*/runs/*" --split $TSV --fut log_fut \
      --prefix prec${p,,} --src recovery --select $O/select/$p.npz ;;
fwd)    # fwd <pool> <rows>; an empty dev-rec set (no row in the trainer's dev split) is not an error: train-rec then holds every row (R1 checks)
  rm -rf $O/fwd/$2-$3; $TR $RF/rf_recov.py fwd --pool $2 --rows $3 --tags $T,$WRONG --out $O/fwd || [[ $3 == dev-rec ]] ;;
r1)     $TR $PP r1 --pool $3 --cache $O/cache/$2 --fwd $O/fwd --tag $T --control $WRONG --out $O/checks/r1_$3.json; cp $O/checks/r1_$3.json $RES/ ;;
probejob)  # probe gates on the 11 probe scenes (A: perturbed, Z: zero offset)
  [[ ! -f $O/STOP ]] || exit 2; rm -f $O/PROBE_OK
  for p in A Z; do bash $ME cache $p || stop "probe cache $p failed"; bash $ME select $p || stop "probe select $p failed"; done
  bash $ME pool A PRECAT || stop "probe pool A failed"; bash $ME pool Z PRECZT || stop "probe pool Z failed"
  for rs in train-rec dev-rec; do bash $ME fwd PRECAT $rs || stop "probe fwd $rs failed"; done
  bash $ME r1 A PRECAT
  $TR $PY g1 --pool PRECZT > $O/g1.log 2>&1; $TR $PY g2 --pool PRECAT --pass A --jobs 16 > $O/g2p.log 2>&1
  $TR $PY g3 --pool PRECAT --pass A --r1 $O/checks/r1_PRECAT.json --jobs 16 > $O/g3p.log 2>&1
  $TR $PY gates --pool PRECAT > $O/gates.log 2>&1 || stop "probe gates failed ($RES/gates_probe.md)"
  git rev-parse --short HEAD > $O/PROBE_OK; echo "$(date '+%F %T') probe gates passed" | tee -a $O/stages.txt ;;
resume)  # after a probe STOP fixed in code: probe job -> rest job again (rolls / rows / caches are reused)
  [[ -f $O/STOP ]] || die "no STOP"; mv $O/STOP $O/STOP.$(date +%m%d%H%M%S); h=$(git rev-parse --short HEAD); echo "$h resume" >> $O/code.txt
  pb=$(sub rec-probe --vram 20 --cpu 16 --ram 96 --timeout-h 3 -- bash $ME probejob) || exit 1
  rs=$(sub rec-rest --after $pb --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') resume (code $h): probe $pb rest $rs" | tee -a $O/stages.txt ;;
rest)  # the remaining chunks (A after the probe, all of B; window W) and the tail
  [[ -f $O/PROBE_OK && ! -f $O/STOP ]] || exit 2; h=$(git rev-parse --short HEAD); echo "$h rest" >> $O/code.txt
  ch=(); for c in $(chunks A); do [[ $c == a000 || $c == a001 ]] || ch+=("A $c"); done; for p in B C; do for c in $(chunks $p); do ch+=("$p $c"); done; done
  declare -A rw=(); cl=()
  for i in "${!ch[@]}"; do set -- ${ch[$i]}; a=""; (( i >= W )) && a=${cl[$((i - W))]}
    u=($(unit $1 $2 "$a")) || exit 1; cl+=(${u[2]}); rw[$1]="${rw[$1]:-} ${u[1]}"; done
  echo "$(date '+%F %T') rest: ${#ch[@]} chunks, window $W" | tee -a $O/stages.txt
  q=(); f=()
  for p in A B C; do c=$(sub rec-cache-$p --after $(csv ${rw[$p]}) --vram 20 --cpu 8 --ram 64 --timeout-h 6 -- bash $ME cache $p) || exit 1
    q+=($(sub rec-pool-$p --after $c --vram 0.1 --cpu 16 --ram 96 --timeout-h 3 -- bash -c "bash $ME select $p && bash $ME pool $p REC$p")) || exit 1
    for rs in train-rec dev-rec; do f+=($(sub rec-fwd-REC$p-$rs --after ${q[-1]} --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- bash $ME fwd REC$p $rs)) || exit 1; done; done
  g=$(sub rec-gfull --after $(csv ${f[@]}) --vram 0.1 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME gfull) || exit 1
  m=$(sub rec-merge --after $(csv ${q[@]}) --vram 0.1 --cpu 8 --ram 120 --timeout-h 3 -- bash $ME merge) || exit 1
  s=$(sub rec-smoke --after $m "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain) || exit 1
  i1=$(sub rec-ident-lc --after $m "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun lc) || exit 1
  i2=$(sub rec-ident-rec0 --after $m "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun rec0) || exit 1
  i=$(sub rec-ident --after $i1,$i2 --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  l=$(sub rec-launch --after $s,$i,$g --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 --env PRIO=$PRIO -- bash $ME launch) || exit 1
  echo "$(date '+%F %T') tail: select+pool ${q[*]} fwd ${f[*]} gfull $g merge $m smoke $s identity $i1 $i2 bitcmp $i launch $l" | tee -a $O/stages.txt ;;
gfull)  # R1 and G2 / G3 on the full pools (reported; the registered gate is the probe's)
  for p in A B C; do bash $ME r1 $p REC$p; $TR $PY g2 --pool REC$p --pass $p --jobs 16 > $O/g2${p,,}.log 2>&1
    $TR $PY g3 --pool REC$p --pass $p --r1 $O/checks/r1_REC$p.json --jobs 16 > $O/g3${p,,}.log 2>&1; done
  $TR $PY rows --pool RECA,RECB,RECC --pass A,B,C > $O/rows.log 2>&1; true ;;
merge)
  [[ ! -e $CRC/on1_JTPHREC || -f $O/merge_started ]] || die "on1_JTPHREC exists: never overwritten"; touch $O/merge_started
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 150 )) || die "free $free GB < 150 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_RECA:reca:both on1_RECB:recb:both on1_RECC:recc:both --out JTPHREC --nan-keys fut_reg || die merge
  $TR $PRPY xfix --pool JTPHREC --ref JTPH --parts reca,recb,recc --out $O/checks || die "xfix (prefix differs from on1_JTPH: $O/checks/xfix_JTPHREC.json)"
  cp $O/checks/xfix_JTPHREC.json $RES/
  $TR $PY weights --pool on1_JTPHREC --k 0 --out R6REC0 > $O/w0.log 2>&1 || die "R6REC0 weights"
  $TR $PY weights --pool on1_JTPHREC --k $K --out R6REC > $O/w.log 2>&1 || die "R6REC weights" ;;
smoketrain)
  X=R6RECSM-L18-s0; [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; run rec 60 $X 0
  echo "$(date '+%F %T') smoke: ${A[*]} --diag-every 20 --save-at 20,40" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 --save-at 20,40 > $O/smoke.log 2>&1 || die "smoke run (log $O/smoke.log)"
  $TR $PY smoke --tag $X --wfile R6REC --k $K; r=$?
  for k in 20 40; do [[ -s $RUNS/R6RECSMx$k-L18-s0/ckpt-final.pt ]] || { echo "$(date '+%F %T') ERROR smoke: no snapshot x$k" | tee -a $O/stages.txt; r=3; }; done
  rm -rf $RUNS/$X $RUNS/R6RECSMx20-L18-s0 $RUNS/R6RECSMx40-L18-s0; exit $r ;;
identrun)  # identrun lc|rec0: 60 steps eager (pool --env JEV_RM_EAGER=1)
  declare -A IX=([lc]=R6IDLC [rec0]=R6IDR0); X=${IX[$2]}-L18-s0; [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; run $2 60 $X 0
  echo "$(date '+%F %T') identity $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)  # G4
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R6IDLC-L18-s0 --b R6IDR0-L18-s0 --out round6_identity_rec \
      --note "R5LC-L18-s0 argv 60 steps eager: on1_JTPH / R5LC vs on1_JTPHREC / R6REC0 (recovery rows weight 0)"; r=$?
  cp $DFR/round6_identity_rec.json $RES/identity.json || r=4; rm -rf $RUNS/R6IDLC-L18-s0 $RUNS/R6IDR0-L18-s0; exit $r ;;
launch)  # G4 + smoke passed -> the two trainings in parallel, each one's PAI 281 --after it, one report after both facts, then the ruler
  [[ ! -f $O/STOP ]] || exit 2
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/smoke_R6REC.json'))['pass_'] and json.load(open('$RES/identity.json'))['identical'] else 1)" \
    || stop "smoke / identity (G4) not passed"
  for s in 0 1; do for X in R6REC-L18-s$s R6RECx2500-L18-s$s R6RECx5000-L18-s$s R6RECx7500-L18-s$s; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
    [[ -s $F_/R5LC-L18-s$s-281ag.csv ]] || die "no stored R5LC s$s facts"; done
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; x=()
  for s in 0 1; do t=$(sub rec-train-s$s "${FT[@]}" --timeout-h 4 -- bash $ME train $s) || exit 1
    y=($(read281 R6REC-L18-s$s $t)) || exit 1; x+=(${y[-1]})
    echo "$(date '+%F %T') launch s$s (code $h): train $t PAI 281 ag chunks ${y[*]:0:${#y[@]}-1} facts ${y[-1]}" | tee -a $O/stages.txt; done
  r=$(sub rec-r281 --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME r281) || exit 1
  P0=$DATA_DIR/tmp/pcs_r5; PB=$DATA_DIR/tmp/pcs_r5l; PT=$DATA_DIR/tmp/pcs_rec; [[ ! -e $PT ]] || die "$PT exists"
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$P0/site)
  c=$(sub rec-pcs-collect --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only R6REC-s0+AG,R6REC-s1+AG --base $PB/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub rec-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  p=$(sub rec-pcs-report --after "$(csv ${b[@]}),$r" --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME pcsrep) || exit 1
  echo "$(date '+%F %T') launch: report $r ruler collect $c boot ${b[*]} report+verdict $p" | tee -a $O/stages.txt ;;
train)
  run rec $N R6REC-L18-s$2 $2; echo "$(date '+%F %T') train s$2: ${A[*]} --save-at 2500,5000,7500" | tee -a $O/stages.txt
  exec $TR "${A[@]}" --save-at 2500,5000,7500 ;;
r281)
  rd=(NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R5LC-s0-ag=$F_/R5LC-L18-s0-281ag.csv R5LC-s1-ag=$F_/R5LC-L18-s1-281ag.csv
      REC-s0-ag=$F_/R6REC-L18-s0-281ag.csv REC-s1-ag=$F_/R6REC-L18-s1-281ag.csv)
  $TR $R2PY r281 --read "${rd[@]}" --pairs REC-s0-ag:R5LC-s0-ag REC-s1-ag:R5LC-s1-ag REC-s0-ag:NTV0-ag REC-s1-ag:NTV0-ag R5LC-s0-ag:NTV0-ag \
      R5LC-s1-ag:NTV0-ag REC-s1-ag:REC-s0-ag --out $RES/pai281_REC_ag.md \
      --title "PAI 281, ACC + GAP: REC (MV-LC + recovery rows) s0 / s1 vs stored R5LC s0 / s1 (MV-LC) and NTV0-L18-s0" || die "r281"
  $V $R5PY newbreaks --json $RES/pai281_REC_ag.json --md $RES/pai281_REC_ag.md --read "${rd[@]}" || die "newbreaks" ;;
pcsrep)
  PB=$DATA_DIR/tmp/pcs_r5l; PT=$DATA_DIR/tmp/pcs_rec; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  python3 - $PB/out $PT/out <<'PY' || die "bootstrap draws of the old reads differ from pcs_r5l's"
import csv, glob, sys                                            # same seeds, same pool: the old subjects' draws must repeat exactly
rd = lambda d: [r for f in sorted(glob.glob(d + "/boot_*.csv")) for r in csv.DictReader(open(f))]  # noqa: E731
a, b = rd(sys.argv[1]), rd(sys.argv[2])
d = max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in x if x[k] and y.get(k))
print("old draws repeat:", len(a), "draws,", len(a[0]), "subjects, max |diff|", d, "PCS")
assert len(a) == len(b) == 100 and d < 0.5, "boot mismatch"
PY
  sj="R2C-s0+AG=R2C-L18-s0-281ag.csv,R3MV-s0+AG=R3MV-L18-s0-281ag.csv,R5LC-s0+AG=R5LC-L18-s0-281ag.csv,R5LC-s1+AG=R5LC-L18-s1-281ag.csv,R6REC-s0+AG=R6REC-L18-s0-281ag.csv,R6REC-s1+AG=R6REC-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs R6REC-s0+AG:R5LC-s0+AG,R6REC-s1+AG:R5LC-s1+AG,R6REC-s1+AG:R6REC-s0+AG \
      > $PT/out/r5.log 2>&1 || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $RES/pcs_rec.md && cp $PT/out/pai_pcs_local_r5.json $RES/pcs_rec.json || die "copy ruler report"
  $V $R5PY pcsrow --pcs $RES/pcs_rec.json --json $RES/pai281_REC_ag.json --md $RES/pai281_REC_ag.md --subject R5LC-s0+AG R5LC-s1+AG R6REC-s0+AG R6REC-s1+AG \
      --report pcs_rec.md || die "pcsrow"
  $V $PY verdict --json $RES/pai281_REC_ag.json --md $RES/pai281_REC_ag.md --pcs $RES/pcs_rec.json || die "verdict"
  date '+%F %T' > $O/DONE ;;
wait)  # wait <FLAG>: one blocking waiter
  f=${2:-DONE}
  until [[ -f $O/$f || -f $O/STOP ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/STOP 2>/dev/null; [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; tail -4 $O/stages.txt; true ;;
*) sed -n 2,18p "$0"; exit 2 ;;
esac
