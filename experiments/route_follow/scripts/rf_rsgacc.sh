#!/usr/bin/env bash
# RSG-ACC (plans/2026-10-17-rsg-acc-prereg.md): RSG-L18's recipe on on-policy rows rolled under the ACC(+GAP) serve, nuPlan onpol-train
# (18 decisions, sh30) and PAI train160 (pai_box), one joint arm + one control. Box, repo root; every job through the GPU pool; ids in $O/jobs.txt.
#   rf_rsgacc.sh smoke              stage 1: RSG-L18-s0 on 4 nuPlan + 4 PAI train scenes, both row pipelines -> smoke pools L18Asm / PAIAsm
#   rf_rsgacc.sh roll <T..>         rollouts of each checkpoint on both tracks + its row chains (nuPlan: ol_chain's geo / tok / strip, rm_build pool,
#                                   poolgate; PAI: onp_rows rows / cold, onp_cache, rf_pai_pool, R1, asl + dump cleanup); last row job ids -> $O/rows_<T>.txt
#   rf_rsgacc.sh pilot <T..>        merge (after the row chains of these T) -> 5 pilot trainings (1 000 steps) -> forward reads -> pilot gate job; the gate
#                                   job submits `full <cell>` when a cell passes
#   rf_rsgacc.sh full <cell>        arm + control at 2 500 steps -> forward reads -> full gate -> (gate) held 452 reads -> held gate -> (gate) public 1491
#                                   reads -> public gate -> (gate) PAI 281 reads -> PAI report
#   job bodies: roll-nu <K> <T> <i> | rows-nu <K> <T> <N> <step> | rows-pai <K> <T> <N> <step> | merge <T..> | train <cell> <tag> <steps> | gate <stage> [..]
# Env: PRIO (14). Serve of every rollout: $SERVE (+ PAI_DUMP on PAI); the reads: held / public default nuPlan serve, PAI = d330 stage 2.
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
RF=$DATA_DIR/runs/route_follow; O=$RF/rsgacc; W=$DATA_DIR/runs/onpol_long; M=$DATA_DIR/runs/route_mem/read; G=$RF/guard; CR=$DATA_DIR/runs/op_parity/cache
E=experiments/onpol_long; ON=experiments/onpol_nuplan; S=experiments/route_follow/scripts; ME=$S/rf_rsgacc.sh; PY=$S/rf_rsgacc.py; SPL=$ON/results/split
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python
AS=$DATA_DIR/third_party/alpasim; T160=experiments/onpol_pai/results/split/train160.tsv; PRIO=${PRIO:-14}
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
REF_PAI=$RF/acc_gap/read/s2
REC="--init RB0-F-s0 --warmup 1 --lr-scale 0.1 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --seed 0"
X=(); for kv in $SERVE; do X+=(--env "$kv"); done
short() { case $1 in RB0-F-s0) echo a0 ;; RB0-F-s1) echo a1 ;; RSG-L18-s0) echo g0 ;; RSG-L18-s1) echo g1 ;; RSGlr03n5000-L18-s0) echo l0 ;; RSGk51w1-L18-s0) echo k0 ;;
          *) echo "no short name for $1" >&2; exit 1 ;; esac; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner rsgacc --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("runtime.scene_provider.trajdata.cache_location=$DATA_DIR/datasets/alpasim_nuplan_long/after160" "runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=160")

roll_one() {  # roll_one <K> <T> <nuPlan lists dir or chunk count> <PAI list>: submits both tracks' rollouts and row chains; echoes the last row job ids
  local K=$1 T=$2 NL=$3 PL=$4 x N NP nu=() ids=() g k p q
  [[ -f $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || { echo "no $T" >&2; exit 1; }
  [[ $K == smoke ]] && x=sm || x=$(short $T) || exit 1; N=L18A$x; NP=PAIA$x
  [[ ! -e $O/roll/nu/$K/c0 && ! -e $CR/on1_$N && ! -e $CR/on1_$NP ]] || { echo "$K already rolled / pools exist: never overwritten" >&2; exit 1; }
  mkdir -p $O/roll/nu/$K
  for f in $NL/c*.txt; do i=$(basename $f .txt); i=${i#c}
    nu+=($(sub ra-roll-nu-$x-$i --vram 32 --cpu 8 --ram 60 --timeout-h 3 "${X[@]}" -- env LIST=$f bash $ME roll-nu $K $T $i)) || exit 1; done
  g=$(sub ra-geo-$x --after "$(csv "${nu[@]}")" --vram 0.5 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME rows-nu $K $T $N geo) || exit 1
  k=$(sub ra-tok-$x --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME rows-nu $K $T $N tok) || exit 1
  sub ra-strip-$x --after $k --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $ME rows-nu $K $T $N strip > /dev/null || exit 1
  p=$(sub ra-pool-$x --after $k --vram 0.5 --cpu 32 --ram 80 --timeout-h 2 -- bash $ME rows-nu $K $T $N pool) || exit 1
  ids+=($p)
  CHUNK=32 SERVE="$SERVE PAI_DUMP=100000" WIZ=$WIZ KEEP_ASL=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/roll/pai/$K $T $PL > /dev/null || exit 1
  q=$(awk '{print $1}' $O/roll/pai/$K/jobs.txt | paste -sd,)
  g=$(sub ra-prow-$x --after $q --vram 0.5 --cpu 24 --ram 96 --timeout-h 3 -- bash $ME rows-pai $K $T $NP rows) || exit 1
  k=$(sub ra-pcache-$x --after $g --vram 20 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME rows-pai $K $T $NP cache) || exit 1
  p=$(sub ra-ppool-$x --after $k --vram 0.5 --cpu 8 --ram 64 --timeout-h 2 -- bash $ME rows-pai $K $T $NP pool) || exit 1
  q=$(sub ra-pr1-$x --after $p --vram 16 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME rows-pai $K $T $NP r1) || exit 1
  sub ra-pclean-$x --after $q --vram 0.5 --cpu 2 --ram 4 --timeout-h 1 -- bash $ME rows-pai $K $T $NP clean > /dev/null || exit 1
  ids+=($q); echo "${ids[*]}"; }

case ${1:-} in
smoke)
  mkdir -p $O/pool $O/smoke
  head -4 $W/lists/smoke5.txt > $O/smoke/c0.txt
  awk -F'\t' 'NR > 1 {print $1}' $T160 | head -4 > $O/smoke/pai4.txt
  r=$(roll_one smoke RSG-L18-s0 $O/smoke $O/smoke/pai4.txt) || exit 1; echo "$r" > $O/rows_smoke.txt
  echo "smoke: last row jobs $r"; tail -n 14 $O/jobs.txt ;;
roll)
  shift; mkdir -p $O/pool; awk -F'\t' 'NR > 1 {print $1}' $T160 > $O/pai160.txt
  for T in "$@"; do r=$(roll_one $T $T $W/lists $O/pai160.txt) || exit 1; echo "$r" > $O/rows_$T.txt; echo "$T: last row jobs $r"; done ;;
pilot)
  shift; dep=(); for T in "$@"; do dep+=($(cat $O/rows_$T.txt)); done
  m=$(sub ra-merge --after "$(csv "${dep[@]}")" --vram 0.5 --cpu 8 --ram 120 --timeout-h 2 -- bash $ME merge "$@") || exit 1
  t=(); for c in a b c p0k13 p0k26; do t+=($(sub ra-train-pilot-$c --after $m --train --vram 28 --cpu 6 --ram 120 --timeout-h 2 -- bash $ME train $c RSGACCp$c-L18-s0 1000)) || exit 1; done
  TG=$(for c in a b c p0k13 p0k26; do printf RSGACCp%s-L18-s0, $c; done)
  f1=$(sub ra-fwd-pilot-held --after "$(csv "${t[@]}")" --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- $TR $S/rf_recov.py fwd --pool L18RSGallheld --rows held01 --tags RSG-L18-s0,${TG%,}) || exit 1
  f2=$(sub ra-fwd-pilot-dev --after "$(csv "${t[@]}")" --vram 16 --cpu 8 --ram 160 --timeout-h 2 -- $TR $S/rf_recov.py fwd --pool RSGACC --rows dev-rec --tags ${TG%,} --out $O/fwd) || exit 1
  sub ra-gate-pilot --after $f1,$f2 --vram 0.5 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME gate pilot ;;
full)
  c=$2; [[ $c =~ ^[abc]$ ]] || { echo "cell $c" >&2; exit 1; }; k=$( [[ $c == c ]] && echo 26 || echo 13 )
  a=$(sub ra-train-arm --train --vram 28 --cpu 6 --ram 120 --timeout-h 2 -- bash $ME train $c RSGACC-L18-s0 2500) || exit 1
  b=$(sub ra-train-ctl --train --vram 28 --cpu 6 --ram 120 --timeout-h 2 -- bash $ME train p0k$k RSGACC0-L18-s0 2500) || exit 1
  f1=$(sub ra-fwd-full-held --after $a,$b --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- $TR $S/rf_recov.py fwd --pool L18RSGallheld --rows held01 --tags RSG-L18-s0,RSGACC-L18-s0,RSGACC0-L18-s0) || exit 1
  f2=$(sub ra-fwd-full-dev --after $a,$b --vram 16 --cpu 8 --ram 160 --timeout-h 2 -- $TR $S/rf_recov.py fwd --pool RSGACC --rows dev-rec --tags RSGACC-L18-s0,RSGACC0-L18-s0 --out $O/fwd) || exit 1
  echo "$c" > $O/cell.txt
  sub ra-gate-full --after $f1,$f2 --vram 0.5 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME gate full ;;
roll-nu)   # roll-nu <K> <T> <i>: ol_chain roll1 with the serve switches from the job's env and the list $LIST
  K=$2; T=$3; i=$4; D=$O/roll/nu/$K/c$i/run
  [[ -z $(comm -12 <(sort "$LIST") <(sort $SPL/held.txt)) ]] || { echo "$LIST has a held scene" >&2; exit 1; }
  [[ ! -e $D/aggregate/results-summary.json ]] || { echo "$D already ran" >&2; exit 1; }
  echo "serve: JEV_VCONT=${JEV_VCONT:-} JEV_LEAD=${JEV_LEAD:-} JEV_ACC=${JEV_ACC:-} JEV_GAP=${JEV_GAP:-}"; [[ ${JEV_ACC:-} == 1 && -n ${JEV_GAP:-} ]] || exit 1
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$LIST" "${BASE[@]}" "${LONG[@]}" || exit 1
  [[ -f $D/aggregate/results-summary.json ]] || exit 1
  $P experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/onpol_sim.pkl" --jobs 8 || exit 1
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason"):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "roll-nu $K c$i: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
rows-nu)   # rows-nu <K> <T> <N> geo|tok|strip|pool: ol_chain's row stages on $O/roll/nu/<K>
  K=$2; T=$3; N=$4; R=$O/roll/nu/$K
  case $5 in
  geo)
    python3 - "$R" "$T" <<'PY' || exit 1
import glob, json, os, sys
R, T = sys.argv[1], sys.argv[2]
m = {f"{T}@{os.path.basename(c)}": [f"{c}/run"] for c in sorted(glob.glob(f"{R}/c*")) if os.path.exists(f"{c}/run/onpol_sim.pkl")}
n = len(glob.glob(f"{R}/c*")); json.dump(m, open(f"{R}/manifest.json", "w"), indent=1); print(len(m), "of", n, "chunks"); sys.exit(len(m) != n)
PY
    $NAV $ON/scripts/on_rows.py geo --name $N --roll $R --workers 32 || exit 1
    $NAV $E/scripts/ol_labels.py build --name $N --workers 32 ;;
  tok)
    $TR $ON/scripts/on_rows.py tok --name $N --control smoke10eY; rc=$?
    $TR $E/scripts/ol_labels.py gate --name $N --out $W/rows/$N; rc=$((rc + $?)); [[ $K == smoke ]] && exit 0; exit $rc ;;
  strip)
    $TR - "$R" <<'PY'
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
def one(f):
    z = dict(np.load(f))
    if "cur" not in z:
        return 0
    del z["cur"]; np.savez_compressed(f + ".tmp.npz", **z); os.replace(f + ".tmp.npz", f); return 1
fs = sorted(glob.glob(f"{sys.argv[1]}/c*/run/driver-logs/dump/*.npz"))
with ProcessPoolExecutor(16) as ex:
    print("stripped", sum(ex.map(one, fs, chunksize=64)), "of", len(fs))
PY
    ;;
  pool)
    $P experiments/route_mem/scripts/rm_build.py pool --name $N --workers 32 || exit 1
    $P $RF/poolgate.py $N; rc=$?; [[ $K == smoke ]] && exit 0; exit $rc ;;
  esac ;;
rows-pai)  # rows-pai <K> <T> <N> rows|cache|pool|r1|clean: onp_rows -> onp_cache -> rf_pai_pool on $O/roll/pai/<K>, native (rf_selfpool's box backend)
  K=$2; T=$3; N=$4; RR=$O/roll/pai/$K; ROWS=$O/rows/pai; STG=$DATA_DIR/tmp/rsgacc_$N; J=$(( $(nproc) > 24 ? 24 : $(nproc) ))
  case $5 in
  rows)
    mkdir -p $STG
    env $SERVE PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AS/.venv/bin/python experiments/onpol_pai/scripts/onp_rows.py rows --pass $N --runs "$RR/runs/*" \
        --out $ROWS --stage $STG --tag $T --jobs $J || exit 1
    env $SERVE PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $N --out $ROWS --stage $STG --jobs $J || exit 1
    env PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py report --pass $N --out $ROWS > $ROWS/report_$N.log 2>&1; tail -3 $ROWS/report_$N.log ;;
  cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $N --rows $ROWS/$N --out $O/cache ;;
  pool)  $TR $S/rf_pai_pool.py --name $N --rows $ROWS/$N --cache $O/cache/$N --runs "$RR/runs/*" --split $T160 ;;
  r1)
    C=$( [[ $T == *-s0 ]] && echo ${T%-s0}-s1 || echo ${T%-s1}-s0 ); [[ -f $DATA_DIR/runs/op_parity/runs/$C/ckpt-final.pt ]] || C=RB0-F-s1
    for rs in train-rec dev-rec; do $TR $S/rf_recov.py fwd --pool $N --rows $rs --tags $T,$C --out $O/fwd || exit 1; done
    $TR $PY r1 --pool $N --tag $T --control $C; rc=$?; [[ $K == smoke ]] && exit 0; exit $rc ;;
  clean)
    n=$(find $RR/runs -name rollout.asl | wc -l); find $RR/runs -name rollout.asl -delete; rm -rf $RR/runs/*/sim/driver-logs/dump
    echo "removed $n rollout.asl and the dumps of $RR; $(du -sh $RR | cut -f1) left"; rm -rf $STG ;;
  esac ;;
merge)     # merge <T..>: dev tags on the new pools that passed their gates, rf_pait merge (on1_L18RB0 first), the pool table
  shift; $TR $PY merge --tags "$@" ;;
train)     # train <cell> <tag> <steps>: RSG-L18-s0's command on on1_RSGACC, the cell's --on-k / --on-weight (rf_rsgacc.py weights)
  c=$2; T=$3; N=$4; k=$( [[ $c == c || $c == p0k26 ]] && echo 26 || echo 13 ); WT=$($TR $PY weights --cell $c) || exit 1
  echo "cell $c: --on-k $k --on-weight $WT"
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC --on on1_RSGACC --on-k $k --on-weight "$WT" --steps $N --eval-every 500 ;;
gate)      # gate pilot | full | held | pub | pai: the registered lines; on PASS the next stage is submitted from here
  st=$2; $TR $PY gate --stage $st; rc=$?
  (( rc == 0 )) || { echo "gate $st: lines do not hold (rc $rc), the chain stops here"; touch $O/STOP_$st; exit 0; }
  case $st in
  pilot) bash $ME full "$(cat $O/cell.txt)" ;;
  full)
    h=(); for T in RSGACC-L18-s0 RSGACC0-L18-s0; do for i in 0 1 2 3 4 5; do
      h+=($(sub ra-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done; done
    sub ra-gate-held --after "$(csv "${h[@]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME gate held > /dev/null ;;
  held)
    r=(); for T in RSGACC-L18-s0 RSGACC0-L18-s0; do for i in $(seq 0 9); do
      r+=($(sub ra-g10-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1; done; done
    sub ra-gate-pub --after "$(csv "${r[@]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME gate pub > /dev/null ;;
  pub)
    for T in RSGACC-L18-s0 RSGACC0-L18-s0; do
      CHUNKS_FROM=$REF_PAI/chunks SERVE="$SERVE" WIZ=$WIZ PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/pai/$T $T > /dev/null || exit 1; done
    q=$(cat $O/pai/RSGACC-L18-s0/jobs.txt $O/pai/RSGACC0-L18-s0/jobs.txt | awk '{print $1}' | paste -sd,)
    sub ra-gate-pai --after $q --vram 0.5 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME gate pai > /dev/null ;;
  pai) touch $O/ALL_DONE ;;
  esac ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
