#!/usr/bin/env bash
# S3 outcome rows (round 5; plans/2026-10-21-s3-rows.md): D3's 4 serve perturbations of NTV0-L18-s0 + ACC + GAP (d3.sh SV: A as served, F / S = ACC
# targets x1.15 / x0.85, L = route-seeking TTS choice) on all 719 non-held eligible curated train scenes (results/s3_scenes.tsv; D3's roll reused
# for its 149 scored scenes) -> per-scene winner (best non-A arm with best - A >= 0.1, else A) and outcome type (partial-fixed / zero-rescued /
# A-wins) -> the winners re-rolled with frame dumps -> onp_rows -> onp_cache -> rf_pai_pool --fut poses (label = the winner's served plan) ->
# pool on1_PAIO + tags -> gates R1 (re-roll reproduces), R2 (label = served plan), R3 (zero-copy merge on1_JTPHO, R2J's command 60 steps eager with
# the PAIO rows at weight 0 bit-identical to on1_JTPH) -> report. Every job through the pool at PRIO 8; ids in $O/jobs.txt, stages in $O/stages.txt.
#   s3.sh chain      stage 1 (score only): c000 = 4 new scenes + 2 D3 check scenes x 4 arms -> chk1 -> rest (all chunks x 4) -> sel -> $O/S1DONE
#   s3.sh stage2go   submit the stage-2 submitter behind the sel job (re-roll r000 = 3 check scenes -> R1 -> rest2: every re-roll + rows + tail)
#   s3.sh wait       blocks until $O/DONE / $O/VERDICT / $O/S1DONE (with WAIT_S1=1) or a failed job of this lane (one run_in_background waiter)
#   job bodies: chk1 | rest | sel | stage2 | r1 <chunk|all> | rest2 | rows <arm> <chunk> | clean <arm> <chunk> | link | cache | pool | tag | r2 |
#               fwd | r1in | merge | weights | smoke <a|b> | bitcmp | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round5/s3; D3R=$DATA_DIR/runs/round5/d3; ME=experiments/round5/scripts/s3.sh; PY=experiments/round5/scripts/s3.py; V=.venv/bin/python
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; RF=experiments/route_follow/scripts
PP=experiments/practice_pai/scripts/pp.py; CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs
T=NTV0-L18-s0; WRONG=RB0-F-s1; PRIO=${PRIO:-8}; TSV=experiments/round5/results/s3_scenes.tsv; ARMS="A F S L"; REF=R2J-L18-s0
NP=${S3_NP:-PAIO}; NM=${S3_NM:-JTPHO}; WO=${S3_WO:-R2JFO}; SMX=${S3_SM:-S3SM}   # env overrides: the R3 pre-flight on a test pool
BASE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
declare -A SV=([A]="$BASE" [F]="$BASE JEV_ACC_VSET=1.15 JEV_ACC_ALAT=3.174" [S]="$BASE JEV_ACC_VSET=0.85 JEV_ACC_ALAT=1.734"
               [L]="$BASE JEV_TTS=1 JEV_TTS_W=$PWD/experiments/round5/scripts/d3_tts_route.json")
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both"
declare -A SM=([a]=${SMX}A-L18-s0 [b]=${SMX}B-L18-s0)
FE=(--train --vram 40 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)
mkdir -p $O/pool $O/checks
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-s3 --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # roll <arm> <root> [after] [extra serve env] -> "<id> <chunk>" lines of the chunks submitted now
  local f=$2/submit.$(date +%s%N).txt
  SERVE="${SV[$1]}${4:+ $4}" WIZ=$WIZ CONC=4 PRIO=$PRIO AFTER=${3:-} KEEP_ASL=${KEEP_ASL:-0} bash experiments/alpasim/scripts/pai_box.sh submit $2 $T > $f \
    || die "roll submit $1 $2"
  grep -q 'not submitted' $f && die "roll $1 $2: $(grep 'not submitted' $f)"
  awk -v a=$1 -v p=$(basename $(dirname $2)) '$2 ~ /^pai-/ {print $1, p "-" a "-" substr($2, length($2) - 3)}' $f >> $O/jobs.txt
  awk '$2 ~ /^pai-/ {print $1, substr($2, length($2) - 3)}' $f; }
case ${1:-} in
chain)
  [[ ! -e $O/roll && ! -e $O/DONE ]] || die "$O has a chain: never overwritten"
  [[ -f $RUNS/$T/ckpt-final.pt && -f $D3R/DONE ]] || die "no $T or no D3"
  git rev-parse --short HEAD > $O/code.txt; $V $PY lists --scenes $TSV --root $O || die lists
  s=(); for a in $ARMS; do mkdir -p $O/roll/$a/chunks; cp $O/lists/c000.txt $O/roll/$a/chunks/; s+=($(roll $a $O/roll/$a | awk '{print $1}')) || exit 1; done
  k1=$(sub s3-chk1 --after $(csv ${s[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME chk1) || exit 1
  rs=$(sub s3-rest --after $k1 --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt)): c000 ${s[*]} chk1 $k1 rest $rs" | tee -a $O/stages.txt ;;
chk1) $V $PY check --root $O --d3 $D3R ;;
rest)  # every chunk list into every arm root, one submit per arm (pai_box.sh skips the finished c000), then sel after all
  r=(); for a in $ARMS; do cp $O/lists/c[0-9][0-9][0-9].txt $O/roll/$a/chunks/; r+=($(roll $a $O/roll/$a | awk '{print $1}')) || exit 1; done
  p=$(sub s3-sel --after $(csv ${r[@]}) --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sel) || exit 1
  echo "$(date '+%F %T') rest: ${#r[@]} roll chunks, sel $p" | tee -a $O/stages.txt ;;
sel)
  $V $PY select --root $O --d3 $D3R --scenes $TSV > $O/sel.log 2>&1 || { echo "STOP select crashed ($O/sel.log)" > $O/VERDICT; exit 1; }
  date '+%F %T' > $O/S1DONE ;;
stage2go)  # from the Mac / main session: the stage-2 submitter behind sel (or now, if sel is done)
  ! grep -q ' s3-stage2$' $O/jobs.txt || die "stage 2 already submitted"
  sl=$(awk '$2 == "s3-sel" {print $1}' $O/jobs.txt | tail -1); [[ -n $sl ]] || die "no s3-sel job"
  h=$(git rev-parse --short HEAD); echo "$h stage2" >> $O/code.txt
  x=$(sub s3-stage2 --after $sl --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME stage2) || exit 1
  echo "$(date '+%F %T') stage2go (code $h): submitter $x after sel $sl" | tee -a $O/stages.txt ;;
stage2)  # r000 (3 check scenes, one per kind of winner) re-rolled with frame dumps and asl -> R1 -> rest2
  [[ -f $O/S1DONE && -f $O/reroll/check.json ]] || die "stage 1 not done"
  s=(); for a in $ARMS; do d=$O/reroll/$a; [[ -f $d/lists/r000.txt ]] || continue; mkdir -p $d/chunks; cp $d/lists/r000.txt $d/chunks/
    s+=($(KEEP_ASL=1 roll $a $d "" PAI_DUMP=100000 | awk '{print $1}')) || exit 1; done
  k=$(sub s3-r1-r000 --after $(csv ${s[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME r1 r000) || exit 1
  rs=$(sub s3-rest2 --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest2) || exit 1
  echo "$(date '+%F %T') stage2: r000 ${s[*]} R1 $k rest2 $rs" | tee -a $O/stages.txt ;;
r1) $V $PY r1 --root $O $([[ $2 != all ]] && echo --chunk $2) ;;
rest2)  # every other re-roll chunk; per chunk rows -> clean (r000's at once); then R1 over all, link -> cache -> pool -> tag -> R2 / fwd -> R1in,
        # merge -> mfix -> weights -> smoke a / b -> bitcmp -> report -> DONE
  w=(); c=(); ro=()
  for a in $ARMS; do d=$O/reroll/$a; ls $d/lists/r*.txt > /dev/null 2>&1 || continue; mkdir -p $d/chunks; cp $d/lists/r*.txt $d/chunks/
    out=$(KEEP_ASL=1 roll $a $d "" PAI_DUMP=100000) || exit 1
    while read -r id ch; do [[ -n $id ]] || continue; ro+=($id)
      x=$(sub s3-rows-$a-$ch --after $id --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $a $ch) || exit 1
      y=$(sub s3-clean-$a-$ch --after $x --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $a $ch) || exit 1; w+=($x); c+=($y)
    done <<< "$out"
    if [[ -f $d/chunks/r000.txt ]]; then
      x=$(sub s3-rows-$a-r000 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $a r000) || exit 1
      y=$(sub s3-clean-$a-r000 --after $x --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $a r000) || exit 1; w+=($x); c+=($y); fi; done
  ra=$(sub s3-r1-all ${ro:+--after $(csv ${ro[@]})} --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME r1 all) || exit 1
  l=$(sub s3-link --after $(csv ${w[@]} ${c[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME link) || exit 1
  ca=$(sub s3-cache --after $l --vram 20 --cpu 8 --ram 64 --timeout-h 3 -- bash $ME cache) || exit 1
  po=$(sub s3-pool --after $ca,$ra --vram 0.1 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME pool) || exit 1
  tg=$(sub s3-tag --after $po --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME tag) || exit 1
  g2=$(sub s3-r2 --after $tg --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME r2) || exit 1
  fw=$(sub s3-fwd --after $tg --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- bash $ME fwd) || exit 1
  gi=$(sub s3-r1in --after $fw --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME r1in) || exit 1
  m=$(sub s3-merge --after $g2,$gi --vram 0.1 --cpu 4 --ram 120 --timeout-h 3 -- bash $ME merge) || exit 1
  wt=$(sub s3-weights --after $m --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- bash $ME weights) || exit 1
  sa=$(sub s3-smoke-a --after $wt "${FE[@]}" -- bash $ME smoke a) || exit 1
  sb=$(sub s3-smoke-b --after $wt "${FE[@]}" -- bash $ME smoke b) || exit 1
  bc=$(sub s3-bitcmp --after $sa,$sb --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME bitcmp) || exit 1
  rp=$(sub s3-report --after $bc --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') rest2: re-rolls ${ro[*]} rows ${w[*]} cleans ${c[*]} R1-all $ra link $l cache $ca pool $po tag $tg R2 $g2 fwd $fw R1in $gi" \
       "merge $m weights $wt smoke $sa $sb bitcmp $bc report $rp" | tee -a $O/stages.txt ;;
rows)  # rows <arm> <chunk>: that chunk's re-rolled rollouts -> $O/rows/<arm>_<chunk>/<scene>.npz, linked into $O/rows/PAIO/ (pr.sh's rows)
  a=$2; c=$3; P=${a}_$c; G=$DATA_DIR/tmp/s3_stage/$P; R=$O/rows; mkdir -p $G $R/$NP
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $P \
      --runs "$O/reroll/$a/runs/ntv0-l18-s0_$c" --out $R --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $P --out $R --stage $G --jobs 8 || exit 1
  rm -rf $G; for f in $R/$P/*.npz; do [[ -e $f ]] && ln -sfn $f $R/$NP/$(basename $f); done
  n=$(ls $R/$P/*.npz 2>/dev/null | wc -l); m=$(wc -w < $O/reroll/$a/chunks/$c.txt); echo "$a $c: $n / $m scene files" | tee -a $O/stages.txt
  (( n >= m * 9 / 10 )) || die "rows $a $c: $n of $m scenes" ;;
clean)  # clean <arm> <chunk>: this lane's own re-roll rollout.asl and frame dumps of the chunk (its rows are built); drive.jsonl and scores stay
  r=$O/reroll/$2/runs/ntv0-l18-s0_$3; n=$(find $r -name rollout.asl | wc -l); find $r -name rollout.asl -delete; rm -rf $r/sim/driver-logs/dump
  echo "$2 $3: removed $n rollout.asl and the frame dumps; $(du -sh $r | cut -f1) left; free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
link) n=$(ls $O/rows/$NP/*.npz | wc -l); echo "$(date '+%F %T') link: $n scene files in $O/rows/$NP" | tee -a $O/stages.txt ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $NP --rows $O/rows/$NP --out $O/cache ;;
pool)
  [[ ! -e $CR/on1_$NP ]] || die "on1_$NP exists: never overwritten"
  $TR $RF/rf_pai_pool.py --name $NP --rows $O/rows/$NP --cache $O/cache/$NP --runs "$O/reroll/*/runs/*" --split $O/keep.tsv --fut poses --prefix paio --src outcome ;;
tag)  $V $PY tag --root $O --pool on1_$NP ;;
r2)   $V $PY r2 --root $O --pool on1_$NP ;;
fwd)  $TR $RF/rf_recov.py fwd --pool $NP --rows train-rec --tags $T,$WRONG --out $O/fwd ;;
r1in) $TR $PP r1 --pool $NP --cache $O/cache/$NP --fwd $O/fwd --tag $T --control $WRONG --out $O/checks/r1in_$NP.json \
        || { echo "STOP R1 input-path check failed ($O/checks/r1in_$NP.json)" > $O/VERDICT; exit 2; } ;;
merge)
  [[ ! -e $CR/on1_$NM ]] || die "on1_$NM exists: never overwritten"
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 200 )) || die "free $free GB < 200 GB"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_$NP:po:both --out $NM --nan-keys fut_reg || die merge
  $TR $PY mfix --pool on1_$NM --ref on1_JTPH || die "mfix: the merged prefix differs from on1_JTPH ($O/checks/mfix_on1_$NM.json)" ;;
weights) $TR $PY weights --pool on1_$NM --base R2JF --out $WO ;;
smoke)  # smoke a|b: R2J-L18-s0's recorded argv, 60 steps eager; a = as recorded (on1_JTPH, R2JF), b = on1_JTPHO + R2JFO (PAIO rows at weight 0)
  X=${SM[$2]}; [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; M=$(ls -d $DATA_DIR/runs/route_mem/train-$REF/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  mapfile -t A < <(python3 - "$M" "$X" "$2" "$NM" "$WO" <<'PY'
import json, sys
m, tag, arm, nm, wo = sys.argv[1:]
a = json.load(open(m))["argv"]
a[a.index("--tag") + 1] = tag
a[len(a) - 1 - a[::-1].index("--steps") + 1] = "60"
if arm == "b":
    i = a.index("--on")
    assert a[i + 1] == "on1_JTPH", a[i + 1]
    a[i + 1] = f"on1_{nm}"
    j = a.index("--on-wfile")
    assert a[j + 1].endswith("/R2JF.npz"), a[j + 1]
    a[j + 1] = a[j + 1][:-len("R2JF.npz")] + f"{wo}.npz"
print("\n".join(a))
PY
)
  echo "$(date '+%F %T') smoke $2 $X: ${A[*]}" >> $O/stages.txt
  exec $TR "${A[@]}" ;;
bitcmp)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a ${SM[a]} --b ${SM[b]} --out round5_s3_r3_$NM \
      --note "R2J-L18-s0 command 60 steps eager: on1_JTPH / R2JF vs on1_$NM / $WO (on1_$NP rows appended at weight 0)"
  cp experiments/defect_fix/results/round5_s3_r3_$NM.json $O/checks/r3_$NM.json; exit 0 ;;   # the verdict is in the json (report)   # the verdict is in r3.json (report)
report) $V $PY report --root $O --pool on1_$NP > $O/report/rows.log 2>&1 || { echo "STOP report crashed" > $O/VERDICT; exit 1; }; date '+%F %T' > $O/DONE ;;
wait)  # returns when the lane has an outcome (DONE / VERDICT; S1DONE too with WAIT_S1=1) or one of its jobs failed / was cancelled
  until [[ -f $O/VERDICT || -f $O/DONE || ( -n ${WAIT_S1:-} && -f $O/S1DONE ) ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/VERDICT 2>/dev/null; for f in S1DONE DONE; do [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; done; tail -3 $O/stages.txt; true ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
