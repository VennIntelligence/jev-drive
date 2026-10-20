#!/usr/bin/env bash
# CURATED-PAI (plans/2026-10-20-curated-pilot-prereg.md): NTV0-L18-s0 rolled with the PAI probe serve (frame dumps) on the 570 curated-train NuRec
# scenes on the box -> S rows (onp_rows / onp_cache / rf_pai_pool, the d367 pipeline) -> R1 -> merged into on1_JT (on1_JTCP) -> STOPPT labels and
# the firing count G0 -> 1 000-step pilot (curated rows at half the S draws) and control (same command, curated rows at weight 0) from NTV0-L18-s0
# with --on-stoppt 0.004 -> PAI 281 (NTV0 + ACC + GAP's serve and chunk lists) -> gate. Box, repo root; every job through the GPU pool, PRIO 11.
# Staged rolls (docs/long-runs.md): c000 = 1 scene -> check -> c001 = 10 scenes -> check (sizes the rest to the disk budget) -> c002.. (32 each),
# at most W chunks holding rollout.asl / dumps at once (chunk k + W waits for chunk k's clean). Every chunk: roll -> rows -> clean.
# Ids in $O/jobs.txt, stages in $O/stages.txt; the outcome (PASS / FAIL / STOP ..) in $O/VERDICT.
#   cp.sh chain                     submit stage 1 + 2 and their checks (the stage-2 check submits everything after it)
#   cp.sh dry                       the pilot and control training commands and their diff (only --tag and --on-wfile may differ)
#   cp.sh wait                      blocks until $O/VERDICT or a failed job of this lane (for a run_in_background waiter)
#   job bodies: chk1 | chk2 | rest <after ids> | rows <chunk> | clean <chunk> | link | cache | pool | fwd <rows> | r1 | merge | labels |
#               weights <arm> | train <arm> | tax <arm> | gate
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/practice_pai/curated; ME=experiments/practice_pai/scripts/cp.sh; PY=experiments/practice_pai/scripts/cp.py; PP=experiments/practice_pai/scripts/pp.py
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; V=.venv/bin/python; S=experiments/route_follow/scripts
T=NTV0-L18-s0; WRONG=RB0-F-s1; NP=CPAI570; NM=JTCP; CR=$DATA_DIR/runs/op_parity/cache; DF=$DATA_DIR/runs/defect_fix; PRIO=${PRIO:-11}
TSV=experiments/practice_pai/results/curated/scenes570.tsv; LAT=$DF/lat/L18NR1v2; JT=experiments/defect_fix/scripts/jt.sh; WS=0.004
REF=$DF/pai_accgap/$T                                                                  # the stored NTV0 + ACC + GAP read (8 chunk lists)
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10   # the PAI probe serve (d366)
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"   # = jt.sh / stoppt.sh: on1_JT's parts in order
declare -A TAG=([pilot]=CPAIP1K-L18-s0 [control]=CPAIC1K-L18-s0) WF=([pilot]=CPAIP [control]=CPAIC)
ROWS=$O/rows; STG=$DATA_DIR/tmp/curated_pai_stage; RES=experiments/practice_pai/results/curated
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner curated-pai --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # roll <chunk> [after]: one chunk root $O/roll/<chunk> (its list written before), NTV0 with the probe serve + frame dumps -> the job id
  local c=$1 r=$O/roll/$1; [[ -f $r/chunks/$c.txt ]] || die "no list $c"
  SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ KEEP_ASL=1 CONC=4 PRIO=$PRIO AFTER=${2:-} bash experiments/alpasim/scripts/pai_box.sh submit $r $T > $r/submit.txt \
    || die "roll submit $c"; local id; id=$(awk '$2 ~ /^pai-/ {print $1}' $r/submit.txt); [[ -n $id && $(wc -w <<< "$id") == 1 ]] || die "roll $c: no single job"
  echo "$id pai-$c" >> $O/jobs.txt; echo "$id"; }
unit() {  # unit <chunk> [after]: roll -> rows -> clean; prints "roll rows clean" ids
  local r w c; r=$(roll $1 ${2:-}) || exit 1
  w=$(sub cp-rows-$1 --after $r --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $1) || exit 1
  c=$(sub cp-clean-$1 --after $w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $1) || exit 1; echo "$r $w $c"; }
lists() {  # chunk lists from the scene table (rank order): c000 = 1 scene, c001 = 10, then 32 each up to <n> scenes in total
  python3 - $TSV $O/roll ${1:-570} <<'PY'
import csv, os, sys
t, root, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = [r["scene_id"] for r in csv.DictReader(open(t), delimiter="\t")][:n]
cut = [0, 1, 11] + list(range(43, len(ids), 32)) + [len(ids)]
for i, (a, b) in enumerate(zip(cut, cut[1:])):
    if b > a:
        os.makedirs(f"{root}/c{i:03d}/chunks", exist_ok=True)
        f = f"{root}/c{i:03d}/chunks/c{i:03d}.txt"
        if not os.path.exists(f):
            open(f, "w").write("\n".join(ids[a:b]) + "\n")
PY
}
cmd_train() {  # cmd_train <arm>: NTV0-L18-s0's command (jt.sh train DRY, ONK=32, ARCROWS=brake) on on1_JTCP with the arm's draw, 1 000 steps,
               # warm-up 50 (the 1K pilots'), --init NTV0-L18-s0, + --on-stoppt 0.004 (labels only on the curated rows)
  local C; C=$(DRY=1 ONK=32 ARCROWS=brake WARM=50 bash $JT train ${TAG[$1]} on1_$NM $LAT ${WF[$1]} 1000 0 | tail -1)
  [[ $C == *"--init RB0-F-s0 "* && $C == *"--on-k 32 "* && $C == *"--warmup 50 "* && $C == *"--steps 1000 "* && $C == *"w/${WF[$1]}.npz "* ]] || die "unexpected: $C"
  echo "${C/--init RB0-F-s0 /--init $T } --on-stoppt $WS"; }
case ${1:-} in
dry)
  diff <(cmd_train pilot | tr ' ' '\n') <(cmd_train control | tr ' ' '\n'); echo "pilot:   $(cmd_train pilot)"; echo "control: $(cmd_train control)" ;;
chain)
  for x in $O/roll $CR/on1_$NP $CR/on1_$NM $DATA_DIR/runs/op_parity/runs/${TAG[pilot]} $DATA_DIR/runs/op_parity/runs/${TAG[control]} $O/VERDICT; do
    [[ ! -e $x ]] || die "$x exists: never overwritten"; done
  [[ -f $REF/chunks/c000.txt && $(ls $REF/chunks/*.txt | wc -l) == 8 ]] || die "reference chunk lists"
  git rev-parse --short HEAD > $O/code.txt; lists 570
  # each check reads its stage's rollouts before that chunk's rows / clean may start (rows wait for the check)
  r0=$(roll c000) || exit 1
  k1=$(sub cp-chk1 --after $r0 --vram 0.1 --cpu 2 --ram 4 --timeout-h 1 -- bash $ME chk1) || exit 1
  w0=$(sub cp-rows-c000 --after $k1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows c000) || exit 1
  c0=$(sub cp-clean-c000 --after $w0 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean c000) || exit 1
  r1=$(roll c001 $k1) || exit 1
  k2=$(sub cp-chk2 --after $r1,$w0 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME chk2) || exit 1
  w1=$(sub cp-rows-c001 --after $k2 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows c001) || exit 1
  c1=$(sub cp-clean-c001 --after $w1 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean c001) || exit 1
  rs=$(sub cp-rest --after $k2 --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest $w1,$c1,$c0) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): c000 roll $r0 chk1 $k1 rows $w0 clean $c0; c001 roll $r1 chk2 $k2 rows $w1 clean $c1;" \
       "rest (stage 3 + tail submitter) $rs" | tee -a $O/stages.txt ;;
chk1) $V $PY stage --roots $O/roll/c000 --n 1 --min-ok 1 --out $O/checks/stage1.json ;;
chk2) $V $PY stage --roots $O/roll/c001 --n 10 --min-ok 9 --n3 559 --done 11 --out $O/checks/stage2.json ;;
rest)  # rest <after ids>: stage-3 units (n3 scenes, window W from the stage-2 check), then the tail; submitted by the queue after chk2
  read -r n3 W < $O/checks/stage2.n3; per=$(python3 -c "import json; print(json.load(open('$O/checks/stage2.json'))['gb_per_scene'])")
  python3 - $TSV $O/roll $((11 + n3)) <<'PY'
import csv, glob, os, sys                      # drop the stage-3 lists beyond n3 scenes (lists of stage 3 not yet submitted)
t, root, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
keep = set([r["scene_id"] for r in csv.DictReader(open(t), delimiter="\t")][:n])
for f in sorted(glob.glob(f"{root}/c*/chunks/c*.txt")):
    if int(os.path.basename(f)[1:4]) >= 2:
        ids = [s for s in open(f).read().split() if s in keep]
        if ids:
            open(f, "w").write("\n".join(ids) + "\n")
        else:
            os.remove(f); os.rmdir(os.path.dirname(f)); os.rmdir(os.path.dirname(os.path.dirname(f)))
PY
  ch=($(ls -d $O/roll/c* | xargs -n1 basename | awk '$1 >= "c002"')); cl=(); rw=(${2//,/ })
  for i in "${!ch[@]}"; do a=""; (( i >= W )) && a=${cl[$((i - W))]}
    u=($(unit ${ch[$i]} $a)) || exit 1; rw+=(${u[1]}); cl+=(${u[2]}); done
  echo "$(date '+%F %T') rest: n3 $n3, $per GB / scene, window $W, ${#ch[@]} chunks, cleans ${cl[*]}" | tee -a $O/stages.txt
  l=$(sub cp-link --after $(csv ${rw[@]} ${cl[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME link) || exit 1
  c=$(sub cp-cache --after $l --vram 20 --cpu 8 --ram 64 --timeout-h 3 -- bash $ME cache) || exit 1
  p=$(sub cp-pool --after $c --vram 0.5 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME pool) || exit 1
  f1=$(sub cp-fwd-train --after $p --vram 16 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME fwd train-rec) || exit 1
  f2=$(sub cp-fwd-dev --after $p --vram 16 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME fwd dev-rec) || exit 1
  g=$(sub cp-r1 --after $f1,$f2 --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME r1) || exit 1
  m=$(sub cp-merge --after $g --vram 0.1 --cpu 4 --ram 60 --timeout-h 3 -- bash $ME merge) || exit 1
  lb=$(sub cp-labels --after $m --vram 0.1 --cpu 24 --ram 96 --timeout-h 2 -- bash $ME labels) || exit 1
  t=(); for A in pilot control; do
    w=$(sub cp-w-$A --after $m --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- bash $ME weights $A) || exit 1
    t+=($(sub cp-train-$A --after $lb,$w "${FT[@]}" --timeout-h 2 -- bash $ME train $A)) || exit 1; done
  q=(); x=(); for i in 0 1; do A=$([[ $i == 0 ]] && echo pilot || echo control); R=$O/read/${TAG[$A]}
    CHUNKS_FROM=$REF/chunks SERVE="$SERVE" WIZ=$WIZ CONC=4 PRIO=$PRIO AFTER=${t[$i]} bash experiments/alpasim/scripts/pai_box.sh submit $R ${TAG[$A]} \
      > $O/read_$A.txt || die "read submit $A"
    [[ $(wc -l < $R/jobs.txt) == 8 ]] || die "$A: not 8 read chunks"; awk '{print $1, $2}' $R/jobs.txt >> $O/jobs.txt
    r=$(awk '{print $1}' $R/jobs.txt | paste -sd,); q+=($r)
    x+=($(sub cp-tax-$A --after $r --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME tax $A)) || exit 1; done
  gt=$(sub cp-gate --after $(csv ${q[@]} ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME gate) || exit 1
  echo "$(date '+%F %T') tail: link $l cache $c pool $p fwd $f1 $f2 r1 $g merge $m labels $lb train ${t[*]} reads ${q[*]} tax ${x[*]} gate $gt" | tee -a $O/stages.txt ;;
rows)  # rows <chunk>: that chunk's rollouts -> $ROWS/CP_<chunk>/<scene>.npz (own pass and staging dir: chunks run side by side)
  c=$2; P=CP_$c; G=$STG/$c; mkdir -p $G $ROWS
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $P \
      --runs "$O/roll/$c/runs/*" --out $ROWS --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $P --out $ROWS --stage $G --jobs 8 || exit 1
  rm -rf $G; n=$(ls $ROWS/$P/*.npz 2>/dev/null | wc -l); m=$(wc -w < $O/roll/$c/chunks/$c.txt); echo "$c: $n / $m scene files"
  (( n >= m * 9 / 10 )) || die "rows $c: $n of $m scenes" ;;
clean)  # clean <chunk>: this pilot's own rollout.asl and frame dumps of that chunk (rows are built); videos are never written (KEEP_VIDEO 0)
  c=$2; n=$(find $O/roll/$c/runs -name rollout.asl | wc -l); find $O/roll/$c/runs -name rollout.asl -delete; rm -rf $O/roll/$c/runs/*/sim/driver-logs/dump
  echo "$c: removed $n rollout.asl and the frame dumps; $(du -sh $O/roll/$c | cut -f1) left; free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
link)  # one rows dir for the cache / pool / labels: symlinks to every chunk's scene files
  mkdir -p $ROWS/$NP; for f in $ROWS/CP_c*/*.npz; do ln -sfn $f $ROWS/$NP/$(basename $f); done; echo "$(ls $ROWS/$NP | wc -l) scene files" | tee -a $O/stages.txt ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $NP --rows $ROWS/$NP --out $O/cache ;;
pool)  $TR $S/rf_pai_pool.py --name $NP --rows $ROWS/$NP --cache $O/cache/$NP --runs "$O/roll/c*/runs/*" --split $TSV ;;
fwd)   $TR $S/rf_recov.py fwd --pool $NP --rows $2 --tags $T,$WRONG --out $O/fwd ;;
r1)    $TR $PP r1 --pool $NP --cache $O/cache/$NP --fwd $O/fwd --tag $T --control $WRONG --out $O/checks/r1_$NP.json \
         || { echo "STOP R1 input-path check failed ($O/checks/r1_$NP.json)" > $O/VERDICT; exit 2; } ;;
merge) $TR $S/rf_pait.py merge --pools $PARTS on1_$NP:cp:self --out $NM --nan-keys fut_reg ;;
labels) $TR $PY labels --pool on1_$NM --src-pool on1_$NP --rows $ROWS/$NP --out $O/fire --workers 24 ;;
weights) $TR $PY weights --pool on1_$NM --base JT00S --arm $2 --out ${WF[$2]} ;;
train)
  A=$2; CMD=$(cmd_train $A) || exit 1; [[ -z ${DRY:-} ]] || { echo "$CMD"; exit 0; }
  [[ -f $O/fire/fire.json && -f $O/w_${WF[$A]}.json ]] || die "train $A: no G0 / weights"
  echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $O/stages.txt; exec $CMD ;;
tax)   A=$2; mkdir -p $O/tax; $AP $S/rf_tax.py extract --runs $(ls -d $O/read/${TAG[$A]}/runs/*/sim) --out $O/tax/$A.pkl --workers 16 > $O/tax/$A.log 2>&1 ;;
gate)
  $V $PY gate --pilot $O/read/${TAG[pilot]} --control $O/read/${TAG[control]} --ref $REF --out $O/gate || { echo "STOP gate crashed" > $O/VERDICT; exit 1; }
  $V experiments/defect_fix/scripts/pai281.py --j $O/read/${TAG[pilot]} --jl pilot --j2 $O/read/${TAG[control]} --j2l control --ref $REF --rl NTV0+ACC+GAP \
      --out $O/gate/pai281_full.md > /dev/null || echo "pai281 table failed" >> $O/gate/notes.txt
  $AP experiments/defect_fix/scripts/stoppt.py overrun --tax NTV0=$DATA_DIR/runs/route_follow/pai_input_loop/tax/A0.pkl pilot=$O/tax/pilot.pkl \
      control=$O/tax/control.pkl --out $O/gate/overrun.json || echo "overrun failed" >> $O/gate/notes.txt
  date '+%F %T' > $O/DONE ;;
wait)  # for the main session's single run_in_background waiter: returns when the lane has an outcome or a failed job
  until [[ -f $O/VERDICT ]] || ls $O/pool/*/ERROR $O/roll/*/runs/*/ERROR $O/read/*/runs/*/ERROR > /dev/null 2>&1; do sleep 300; done
  cat $O/VERDICT 2>/dev/null; ls $O/pool/*/ERROR $O/roll/*/runs/*/ERROR $O/read/*/runs/*/ERROR 2>/dev/null ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
