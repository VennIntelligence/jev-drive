#!/usr/bin/env bash
# PAI-ROWS (plans/2026-10-20-pai-rows-prereg.md): box, repo root; every GPU job through the pool. Two passes over results/scenes.tsv:
#   H  human-state rows: NTV0-L18-s0 fed and recorded while the ego follows the log (onp_log_driver.py; pai_gap.sh's H read: A0's serve +
#      send_recording_ground_truth) -> onp_rows -> onp_cache build -> rf_pai_pool --fut log_fut (pool on1_PAIH) -> R1 -> pr.py stretch
#      f = 0.5 / 1.5 (on1_PAIH05 / on1_PAIH15) -> label feasibility + row report
#   N  the second batch of human-state rows (2026-10-21; plans/2026-10-21-pai-rows-n.md): H's recipe unchanged on the 64 scenes of results/scenes_n.tsv
#      (each shares a drive with a PAI test281 scene -> tab key t281) into NEW pools on1_PAIHN / PAIHN05 / PAIHN15, then the zero-copy merges
#      on1_PAIHX{,05,15} (= PAIH* + PAIHN*) and on1_JTPHX (= on1_JTPH's parts + PAIH* + PAIHN*), prefix checks, DONE after the merge job
#   S  on-policy S rows on the scenes outside the d402 pool: cp.sh's roll unchanged (NTV0-L18-s0, PAI probe serve) -> onp_rows -> onp_cache
#      build -> rf_pai_pool (on1_CPAIX) -> R1
# Staged (docs/long-runs.md): chunk 0 = 1 scene -> check -> chunk 1 = 10 scenes -> check -> probe (the whole tail on those 11 scenes: cache,
# pool, trainer-forward R1, for H also the f = 1 bit-identity gate, f = 0.5 / 1.5 builds and the feasibility table) -> rest (32 a chunk, at
# most W chunks holding rollout.asl / dumps at once) -> tail. Every chunk: roll -> rows -> clean. Rerunning `chain` after new scenes join
# scenes.tsv lists only the scenes not yet in a chunk (chunks of 32, no staging) and rebuilds the tail (cache / stretch parts are resumed).
#   pr.sh chain H|S|N     submit (ids in $O/jobs.txt, stages in $O/stages.txt)
#   pr.sh resume H|S      after a probe STOP fixed in code: probe -> rest (unsubmitted chunks) [-> chain S]
#   pr.sh wait H|S|N        blocks until $O/<pass>/DONE, $O/<pass>/STOP or a failed job of the pass (for one run_in_background waiter)
#   pr.sh reprobe [H|N]   (GPU job body) the probe's stretch part alone -> $O/H/probe/REPROBE_OK; the full stretch requires it
#   job bodies: check <pass> <chunk> <n> <min> | rows <pass> <chunk> | clean <pass> <chunk> | probe <pass> | rest <pass> | cache <pass> <tag> |
#               pool <pass> <name> | fwd <name> <rows> | r1 <pass> <name> | stretch <f> <pool> [pass] | gate1 | feas | report | merge (N)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/pai_rows; ME=experiments/pai_rows/scripts/pr.sh; PY=experiments/pai_rows/scripts/pr.py; PP=experiments/practice_pai/scripts/pp.py
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; V=.venv/bin/python; RF=experiments/route_follow/scripts
T=NTV0-L18-s0; WRONG=RB0-F-s1; TSV=experiments/pai_rows/results/scenes.tsv; TSVN=experiments/pai_rows/results/scenes_n.tsv; RES=experiments/pai_rows/results; W=${W:-8}
A0SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 PAI_DUMP=100000"; G10=controller.gains.rel_front_steering_angle_weight=10
declare -A SERVE=([H]="$A0SERVE DRV_PY=../../onpol_pai/lib/onp_log_driver.py" [S]="$A0SERVE" [N]="$A0SERVE DRV_PY=../../onpol_pai/lib/onp_log_driver.py")
declare -A WIZ=([H]="$G10 runtime.simulation_config.send_recording_ground_truth=true" [S]="$G10" [N]="$G10 runtime.simulation_config.send_recording_ground_truth=true")
declare -A POOL=([H]=PAIH [S]=CPAIX [N]=PAIHN) PRI=([H]=12 [S]=10 [N]=6) LOGCHK=([H]=--log [S]="" [N]=--log) TSVP=([H]=$TSV [S]=$TSV [N]=$TSVN)
CRC=$DATA_DIR/runs/op_parity/cache; JT_PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"   # = r2.sh PARTS: on1_JT's parts
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
stop() { mkdir -p $O/$1; echo "$(date '+%F %T') $2" | tee $O/$1/STOP >> $O/stages.txt; exit 2; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner pai-rows --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} --no-check \
     --log-dir $O/pool/$name.$(date +%m%d%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
want() {  # want <pass>: the pass's scenes (H: all of scenes.tsv, S: those outside the d402 pool, N: all of scenes_n.tsv)
  awk -F'\t' -v p=$1 'NR > 1 && (p != "S" || $3 == "False") {print $1}' ${TSVP[$1]}; }
lists() {  # lists <pass>: chunk lists for the scenes not yet in a chunk; first ever = 1 scene, second = 10, then 32 each -> new chunk names
  local p=$1 r=$O/roll/$1; mkdir -p $r
  python3 - $r ${p,,} <(want $p) <<'PY'
import glob, os, sys
root, px, wf = sys.argv[1], sys.argv[2], sys.argv[3]
done = set(" ".join(open(f).read() for f in glob.glob(f"{root}/*/chunks/*.txt")).split())
ids = [s for s in open(wf).read().split() if s not in done]
k = len(glob.glob(f"{root}/{px}[0-9][0-9][0-9]"))
new = []
while ids:
    n = 1 if k == 0 else 10 if k == 1 else 32
    c = f"{px}{k:03d}"
    os.makedirs(f"{root}/{c}/chunks")
    open(f"{root}/{c}/chunks/{c}.txt", "w").write("\n".join(ids[:n]) + "\n")
    new.append(c); ids = ids[n:]; k += 1
print(" ".join(new))
PY
}
roll() {  # roll <pass> <chunk> [after] -> the chunk's job id
  local p=$1 c=$2 r=$O/roll/$1/$2
  SERVE="${SERVE[$p]}" WIZ="${WIZ[$p]}" KEEP_ASL=1 CONC=4 PRIO=${PRI[$p]} AFTER=${3:-} bash experiments/alpasim/scripts/pai_box.sh submit $r $T > $r/submit.txt \
    || die "roll submit $p $c"; local id; id=$(awk '$2 ~ /^pai-/ {print $1}' $r/submit.txt); [[ -n $id && $(wc -w <<< "$id") == 1 ]] || die "roll $p $c: no single job"
  echo "$id pai-$p-$c" >> $O/jobs.txt; echo "$id"; }
unit() {  # unit <pass> <chunk> <after roll> <after rows>: roll -> rows -> clean; prints "roll rows clean"
  local r w c; PRIO=${PRI[$1]}; r=$(roll $1 $2 $3) || exit 1
  w=$(sub pr-rows-$1-$2 --after $(csv $r ${4:-}) --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $1 $2) || exit 1
  c=$(sub pr-clean-$1-$2 --after $w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $1 $2) || exit 1; echo "$r $w $c"; }
tail_() {  # tail_ <pass> <after>: cache -> pool -> fwd x 2 -> R1 (-> H: stretch x 2 -> feas + report) -> DONE
  local p=$1 n=${POOL[$1]} a=$2 c q f1 f2 g s5 s15 s10 f x; PRIO=${PRI[$1]}
  c=$(sub pr-cache-$p --after $a --vram 20 --cpu 8 --ram 64 --timeout-h 4 -- bash $ME cache $p $p) || exit 1
  q=$(sub pr-pool-$p --after $c --vram 0.1 --cpu 8 --ram 96 --timeout-h 3 -- bash $ME pool $p $n) || exit 1
  f1=$(sub pr-fwd-train-$p --after $q --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- bash $ME fwd $n train-rec) || exit 1
  f2=$(sub pr-fwd-dev-$p --after $q --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- bash $ME fwd $n dev-rec) || exit 1
  g=$(sub pr-r1-$p --after $f1,$f2 --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME r1 $p $n) || exit 1
  if [[ $p == H ]]; then
    s5=$(sub pr-stretch05-H --after $g --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 0.5 $n) || exit 1
    s15=$(sub pr-stretch15-H --after $g --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 1.5 $n) || exit 1
    x=$(sub pr-final-H --after $s5,$s15 --vram 0.1 --cpu 4 --ram 64 --timeout-h 1 -- bash $ME final H) || exit 1
  elif [[ $p == N ]]; then          # + the f = 1 identity on the full pool, then the merges (DONE is written by the merge job)
    s5=$(sub pr-stretch05-N --after $g --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 0.5 $n N) || exit 1
    s15=$(sub pr-stretch15-N --after $g --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 1.5 $n N) || exit 1
    s10=$(sub pr-stretch10-N --after $g --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 1.0 $n N) || exit 1
    f=$(sub pr-final-N --after $s5,$s15,$s10 --vram 0.1 --cpu 4 --ram 64 --timeout-h 1 -- bash $ME final N) || exit 1
    x=$(sub pr-merge-N --after $f --vram 0.1 --cpu 8 --ram 120 --timeout-h 3 -- bash $ME merge) || exit 1
  else x=$(sub pr-final-S --after $g --vram 0.1 --cpu 4 --ram 64 --timeout-h 1 -- bash $ME final S) || exit 1; fi
  echo "$(date '+%F %T') tail $p: cache $c pool $q fwd $f1 $f2 r1 $g ${s5:+stretch $s5 $s15 ${s10:-} }final ${f:-$x}${f:+ merge $x}" | tee -a $O/stages.txt; }
case ${1:-} in
chain)
  p=$2; [[ $p == H || $p == S || $p == N ]] || die "pass H, S or N"; mkdir -p $O/pool $O/$p; rm -f $O/$p/DONE
  git rev-parse --short HEAD >> $O/$p/code.txt; ch=($(lists $p)); [[ ${#ch[@]} -gt 0 ]] || die "$p: no new scenes"
  if [[ ${ch[0]} == ${p,,}000 ]]; then                                  # first chain: staged 1 -> 10 -> probe -> rest
    PRIO=${PRI[$p]}
    r0=$(roll $p ${ch[0]}) || exit 1
    k1=$(sub pr-chk-$p-${ch[0]} --after $r0 --vram 0.1 --cpu 2 --ram 4 --timeout-h 1 -- bash $ME check $p ${ch[0]} 1 1) || exit 1
    u0=($(PRIO=${PRI[$p]}; w=$(sub pr-rows-$p-${ch[0]} --after $k1 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $p ${ch[0]}) && \
          c=$(sub pr-clean-$p-${ch[0]} --after $w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $p ${ch[0]}) && echo $w $c)) || exit 1
    r1=$(roll $p ${ch[1]} $k1) || exit 1
    k2=$(sub pr-chk-$p-${ch[1]} --after $r1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME check $p ${ch[1]} 10 9) || exit 1
    w1=$(sub pr-rows-$p-${ch[1]} --after $k2 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $p ${ch[1]}) || exit 1
    c1=$(sub pr-clean-$p-${ch[1]} --after $w1 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $p ${ch[1]}) || exit 1
    pb=$(sub pr-probe-$p --after ${u0[0]},$w1 --vram 20 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME probe $p) || exit 1
    rs=$(sub pr-rest-$p --after $pb --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest $p "${ch[*]:2}" ${u0[0]},$w1,${u0[1]},$c1) || exit 1
    echo "$(date '+%F %T') chain $p (code $(git rev-parse --short HEAD)): ${ch[0]} roll $r0 check $k1 rows/clean ${u0[*]}; ${ch[1]} roll $r1 check $k2" \
         "rows $w1 clean $c1; probe $pb; rest $rs (${#ch[@]} chunks)" | tee -a $O/stages.txt
  else bash $ME rest $p "${ch[*]}" ""; fi ;;                          # an append: the new chunks and the tail, no staging
resume)  # resume <pass>: after a probe STOP whose cause is fixed in the code: probe -> rest (the chunks not yet submitted) [-> chain S]; rolls,
         # rows and cache parts are reused, probe outputs are rebuilt
  p=$2; [[ -f $O/$p/STOP ]] || die "$p: no STOP"; mv $O/$p/STOP $O/$p/STOP.$(date +%m%d%H%M%S); git rev-parse --short HEAD >> $O/$p/code.txt
  ch=($(for d in $O/roll/$p/${p,,}[0-9][0-9][0-9]; do [[ -n $(ls $d/runs/*/SUBMITTED 2>/dev/null) ]] || basename $d; done))
  PRIO=${PRI[$p]}; pb=$(sub pr-probe-$p --vram 20 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME probe $p) || exit 1
  rs=$(sub pr-rest-$p --after $pb --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest $p "${ch[*]}" "") || exit 1
  [[ $p == H && ! -d $O/roll/S ]] && cs=$(PRIO=10 sub pr-chainS --after $rs --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME chain S)
  echo "$(date '+%F %T') resume $p (code $(git rev-parse --short HEAD)): probe $pb rest $rs (${#ch[@]} chunks) ${cs:+chain S $cs}" | tee -a $O/stages.txt ;;
rest)  # rest <pass> "<chunks>" <after ids>: the remaining units (window W) and the tail (an in-queue job after the probe, or an append)
  p=$2; [[ ! -f $O/$p/STOP ]] || exit 2; ch=($3); rw=(${4:+${4//,/ }}); cl=()
  for i in "${!ch[@]}"; do a=""; (( i >= W )) && a=${cl[$((i - W))]}
    u=($(unit $p ${ch[$i]} "$a" "")) || exit 1; rw+=(${u[1]} ${u[2]}); cl+=(${u[2]}); done
  echo "$(date '+%F %T') rest $p: ${#ch[@]} chunks, window $W, cleans ${cl[*]}" | tee -a $O/stages.txt
  tail_ $p $(csv ${rw[@]}) ;;
check) p=$2; $V $PY check --root $O/roll/$p/$3 --n $4 --min-ok $5 ${LOGCHK[$p]} --out $O/$p/checks/$3.json || stop $p "roll check $3 failed ($O/$p/checks/$3.json)" ;;
rows)  # rows <pass> <chunk>: the chunk's rollouts -> $O/rows/<pass>_<chunk>/<scene>.npz, linked into $O/rows/<pass>/ (own staging dir: chunks run side by side)
  p=$2; c=$3; P=${p}_$c; G=$DATA_DIR/tmp/pai_rows_stage/$P; R=$O/rows; mkdir -p $G $R/$p
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $P \
      --runs "$O/roll/$p/$c/runs/*" --out $R --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $P --out $R --stage $G --jobs 8 || exit 1
  env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass $P --out $R > $R/report_$P.log 2>&1
  rm -rf $G; for f in $R/$P/*.npz; do [[ -e $f ]] && ln -sfn $f $R/$p/$(basename $f); done
  n=$(ls $R/$P/*.npz 2>/dev/null | wc -l); m=$(wc -w < $O/roll/$p/$c/chunks/$c.txt); echo "$p $c: $n / $m scene files" | tee -a $O/stages.txt
  (( n >= m * 9 / 10 )) || die "rows $p $c: $n of $m scenes" ;;
clean)  # clean <pass> <chunk>: this lane's own rollout.asl and frame dumps of the chunk (its rows are built); drive.jsonl stays (route messages)
  r=$O/roll/$2/$3; n=$(find $r/runs -name rollout.asl | wc -l); find $r/runs -name rollout.asl -delete; rm -rf $r/runs/*/sim/driver-logs/dump
  echo "$2 $3: removed $n rollout.asl and the frame dumps; $(du -sh $r | cut -f1) left; free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $3 --rows $O/rows/$2 --out $O/cache ;;
pool)  # pool <pass> <name>: rf_pai_pool on the pass's cache (H: fut = the log's future poses, names on:paih:..)
  p=$2; extra=(); [[ $p == H ]] && extra=(--fut log_fut --prefix paih --src human); [[ $p == N ]] && extra=(--fut log_fut --prefix paihn --src human)
  $TR $RF/rf_pai_pool.py --name $3 --rows $O/rows/$p --cache $O/cache/$p --runs "$O/roll/$p/*/runs/*" --split ${TSVP[$p]} "${extra[@]}" || exit 1
  [[ $p != N ]] || $TR $PY tag --pool $3 ;;                            # N: the per-row t281 tag (stretch pools inherit it)
fwd)   $TR $RF/rf_recov.py fwd --pool $2 --rows $3 --tags $T,$WRONG --out $O/fwd ;;
r1)    $TR $PP r1 --pool $3 --cache $O/cache/$2 --fwd $O/fwd --tag $T --control $WRONG --out $O/$2/r1_$3.json || stop $2 "R1 failed on $3 ($O/$2/r1_$3.json)" ;;
stretch)  # stretch <f> <plain pool> [pass H|N]; the full pool only after the probe's stretch gate of the current label rule (supplement 2: REPROBE_OK)
  n=$3$(tr -d . <<< $2)                                                # PAIH + 05 / 15 / 10
  [[ $3 != PAIH || -f $O/H/probe/REPROBE_OK ]] || stop H "full stretch $2: no $O/H/probe/REPROBE_OK (supplement 2 probe)"
  $TR $PY stretch --pool $3 --rows $O/rows/${4:-H} --f $2 --name $n ;;
probe)  # probe <pass>: the tail on the first 11 scenes (pools on1_<POOL>T*), every gate before the rest is submitted
  p=$2; n=${POOL[$p]}T; mkdir -p $O/$p/probe
  for x in $DATA_DIR/runs/op_parity/cache/on1_$n $O/fwd/$n-train-rec $O/fwd/$n-dev-rec; do rm -rf $x $x@warp; done   # probe outputs only (a rerun of this job)
  bash $ME cache $p $p || stop $p "probe cache failed"
  bash $ME pool $p $n || stop $p "probe pool failed"
  for rs in train-rec dev-rec; do bash $ME fwd $n $rs || stop $p "probe fwd $rs failed"; done
  bash $ME r1 $p $n || exit 2
  if [[ $p == H || $p == N ]]; then
    bash $ME reprobe $p || exit 2
  else $TR $PY report --pools $n --out $O/S/probe; fi
  echo "$(date '+%F %T') probe $p passed" | tee -a $O/stages.txt ;;                # tables stay in $O (copied to results/ on the Mac)
reprobe)  # supplement 2: the probe's stretch part again (dilated label): f = 1 / 0.5 / 1.5 on on1_<POOL>T, f = 1 gate, feasibility, rows -> REPROBE_OK
  p=${2:-H}; n=${POOL[$p]}T; rm -f $O/$p/probe/REPROBE_OK
  for f in 1.0 0.5 1.5; do rm -rf $DATA_DIR/runs/op_parity/cache/on1_$n${f/./}{,@warp} $O/stretch/$n${f/./}; bash $ME stretch $f $n $p || stop $p "reprobe stretch $f failed"; done
  $TR $PY gate1 --plain $n --f1 ${n}10 --out $O/$p/probe/gate1.json || stop $p "f = 1 identity gate failed ($O/$p/probe/gate1.json)"
  $TR $PY feas --pools $n:1 ${n}05:0.5 ${n}15:1.5 --out $O/$p/probe && $TR $PY report --pools $n ${n}05 ${n}15 --out $O/$p/probe || exit 1
  git rev-parse --short HEAD > $O/$p/probe/REPROBE_OK; echo "$(date '+%F %T') reprobe $p passed" | tee -a $O/stages.txt ;;
final)  # final <pass>: tables of the full pools -> $O/<pass>/ (copied to results/ by hand: the box never commits)
  p=$2
  if [[ $p == H ]]; then $TR $PY feas --pools PAIH:1 PAIH05:0.5 PAIH15:1.5 --out $O/H && $TR $PY report --pools PAIH PAIH05 PAIH15 --out $O/H || exit 1
  elif [[ $p == N ]]; then          # the f = 1 identity on the full pool too (PAIHN10 = f 1.0 of PAIHN); DONE comes after the merge job
    $TR $PY gate1 --plain PAIHN --f1 PAIHN10 --out $O/N/gate1.json || stop N "f = 1 identity gate failed on the full pool ($O/N/gate1.json)"
    $TR $PY feas --pools PAIHN:1 PAIHN05:0.5 PAIHN15:1.5 --out $O/N && $TR $PY report --pools PAIHN PAIHN05 PAIHN15 --out $O/N || exit 1
    exit 0
  else $TR $PY report --pools CPAIX --out $O/S || exit 1; fi
  date '+%F %T' > $O/$p/DONE ;;
merge)  # N: zero-copy merges (rf_pait.py merge, the default; no token file is copied): PAIHX* = PAIH* + PAIHN*; JTPHX = JTPH's parts + PAIH* + PAIHN*
  [[ ! -f $O/N/STOP ]] || exit 2; mkdir -p $O/N/x
  for x in PAIHX PAIHX05 PAIHX15 JTPHX; do [[ ! -e $CRC/on1_$x ]] || stop N "on1_$x exists: never overwritten"; done
  for s in "" 05 15; do
    $TR $RF/rf_pait.py merge --pools on1_PAIH$s:ph$s:both on1_PAIHN$s:phn$s:both --out PAIHX$s || stop N "merge PAIHX$s failed"
    $TR $PY mcheck --pool on1_PAIHX$s --ref on1_PAIH$s --skip-names --out $O/N/x || stop N "prefix check PAIHX$s failed ($O/N/x/mcheck_on1_PAIHX$s.json)"; done
  $TR $RF/rf_pait.py merge --pools $JT_PARTS on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both on1_PAIHN:phn:both on1_PAIHN05:phn05:both on1_PAIHN15:phn15:both \
      --out JTPHX --nan-keys fut_reg || stop N "merge JTPHX failed"
  $TR $PY xfix --pool on1_JTPHX --ref on1_JTPH --parts phn,phn05,phn15 --out $O/N/x || stop N "JTPHX fix failed ($O/N/x/xfix_on1_JTPHX.json)"
  $TR $PY report --pools PAIHX PAIHX05 PAIHX15 --out $O/N/x || exit 1
  date '+%F %T' > $O/N/DONE ;;
wait)  # wait <pass>: one blocking waiter (run_in_background): returns at DONE / STOP or when a job of this lane has failed
  p=$2
  until [[ -f $O/$p/DONE || -f $O/$p/STOP ]] || ! $V - $O/jobs.txt $p <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
p = sys.argv[2]
bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if (n.endswith("-" + p) or f"-{p}-" in n) and jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/$p/STOP 2>/dev/null; [[ -f $O/$p/DONE ]] && echo "DONE $(cat $O/$p/DONE)"; true ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
