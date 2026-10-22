#!/usr/bin/env bash
# ROUND9 NU-PAIBASE (plans/2026-10-22-nu-paibase.md): the nuPlan candidate B2 (STANDFMXH-L18-s0 / s1, decisions 389 / 391) with only its init
# changed to the PAI-mixed bases RB0-P70-s0 / RB0-P30-s0 (decision 449), seeds 0 and 1 (each pairs with B2 of the same seed). Box, repo root;
# every job through the pool, the whole chain in one submit; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = runs/round9/nu_paibase.
#   nupb.sh check      the four arms' commands against B2's stored command of the same seed (= its meta.json argv): only tag + init values differ
#   nupb.sh launch     identity (B2-s0's command 60 steps eager: code 5858f899 from its worktree vs HEAD -> bitcmp) + smoke (P70-s0's command 60
#                      steps eager) -> the four trainings (one cl fanout) -> per arm arena held 18 chunks native + held 460 @18 -> sum -> $O/DONE
#   nupb.sh wait       one run_in_background waiter: $O/DONE or a failed / cancelled job of this lane
#   job bodies: ident <rev> | smoke | train <arm> | sum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/nu_paibase; ME=experiments/round9/scripts/nupb.sh; PY=experiments/round9/scripts/nupb.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RUNS=$DATA_DIR/runs/op_parity/runs
JT=experiments/defect_fix/scripts/jt.sh; RES=experiments/round9/results/nu_paibase; SB=$DATA_DIR/runs/defect_fix/stand_b; REV=5858f899
declare -A B2=([0]=STANDFMXH-L18-s0 [1]=STANDFMXH-L18-s1) CF=([0]=$SB/cmd_FH.txt [1]=$SB/cmd_FHS.txt)
ARMS="70s0 70s1 30s0 30s1"; REFS="STANDFMXH-L18-s0 STANDFMXH-L18-s1 JTN-L18-s1"
TRN=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train); RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <prio> <name> <pool args...>
  local p=$1 name=$2; shift 2; local id
  id=$($V -m jevdrive.cl submit --owner round9-nupb --name "$name" --tries ${TRIES:-1} --priority $p --no-check --log-dir $O/pool/$name "$@" < /dev/null \
       | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
tag() { echo NPB${1%s*}-L18-s${1#*s}; }                 # 70s0 -> NPB70-L18-s0
cmd() {  # cmd <arm|b2s0> [tag steps warm]: B2's stored command of the arm's seed (checked = meta.json argv), tag / init (/ steps / warm-up) swapped
  local s init C t; if [[ $1 == b2s0 ]]; then s=0; init=RB0-F-s0; else s=${1#*s}; init=RB0-P${1%s*}-s0; fi
  C=$(cat ${CF[$s]}) || die "no ${CF[$s]}"
  [[ "$(cut -d' ' -f2- <<< "$C")" == "$(python3 -c "import glob, json; print(' '.join(json.load(open(sorted(glob.glob('$DATA_DIR/runs/route_mem/train-${B2[$s]}/*/meta.json'))[-1]))['argv']))")" ]] \
    || die "${CF[$s]} != ${B2[$s]} meta.json argv"
  [[ $C == *" --tag ${B2[$s]} "* && $C == *" --init RB0-F-s0 "* && $C == *" --seed $s "* && $C == *" --steps 9500 "* && $C == *" --warmup 500 "* ]] || die "unexpected B2 command"
  t=${2:-$(tag $1)}; C=${C/ --tag ${B2[$s]} / --tag $t }; C=${C/ --init RB0-F-s0 / --init $init }
  [[ -n ${3:-} ]] && { C=${C/ --steps 9500 / --steps $3 }; C=${C/ --warmup 500 / --warmup $4 }; }
  echo "$C"; }
case ${1:-} in
check)
  { for a in $ARMS; do s=${a#*s}; cmd $a > $O/cmd_$a.txt || exit 1
      echo "== ${B2[$s]} -> $(tag $a)"; diff <(tr ' ' '\n' < ${CF[$s]}) <(tr ' ' '\n' < $O/cmd_$a.txt)
      [[ "$(diff <(tr ' ' '\n' < ${CF[$s]}) <(tr ' ' '\n' < $O/cmd_$a.txt) | grep '^>' | cut -c3- | paste -sd' ')" == "$(tag $a) RB0-P${a%s*}-s0" ]] \
        || die "$a: not tag + init only"; done; } > $RES/cmd_diff.txt || exit 1
  cat $RES/cmd_diff.txt; echo "diff ok" ;;
launch)
  [[ ! -e $O/DONE ]] || die "already ran"; [[ -n ${RETRY:-} ]] && rm -rf $RUNS/NPBGG-old $RUNS/NPBGG-new $RUNS/NPBSM   # relaunch after a failed gate
  for a in $ARMS NPBGG-old NPBGG-new NPBSM; do [[ $a == NPB* ]] && x=$a || x=$(tag $a)
    [[ ! -e $RUNS/$x ]] || die "$x exists"; done
  for t in $REFS RB0-P70-s0 RB0-P30-s0; do [[ -s $RUNS/$t/ckpt-final.pt ]] || die "no $t"; done
  bash $ME check > /dev/null || exit 1
  WT=$DATA_DIR/tmp/df_old_$REV; [[ -d $WT ]] || git worktree add --detach $WT $REV > /dev/null || die "worktree $REV"
  # addendum 1: on1_JTX is a zero-copy pool since 10-21 (front.parts.json, no front.npy), which 5858f899 cannot open; the old tree opens that
  # manifest through HEAD's jevdrive.tokens.open_front (bitwise equal to the copied pool, cl_infra/results/zero-copy); nothing else changes
  cp jevdrive/tokens.py $WT/jevdrive/tokens.py && python3 - $WT/experiments/op_parity/scripts/pp_train.py <<'PY' || die "old tree patch"
import sys
p = sys.argv[1]; s = open(p).read()
a = '    return np.load(f, mmap_mode="r") if f.exists() else IndexedMM(f.parent)\n'
b = ('    return np.load(f, mmap_mode="r") if f.exists() else __import__("jevdrive.tokens", fromlist=["open_front"]).open_front(f) '
     'if (f.parent / "front.parts.json").exists() else IndexedMM(f.parent)  # nupb addendum 1\n')
assert s.count(a) == 1 or s.count(b) == 1, "open_front line"
open(p, "w").write(s.replace(a, b))
PY
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt; FE=("${TRN[@]}" --env JEV_RM_EAGER=1 --timeout-h 1)
  a=$(sub 20 nupb-gg-old "${FE[@]}" -- bash -c "cd $WT && $(cmd b2s0 NPBGG-old 60 50)") || exit 1      # the old code runs from its worktree
  b=$(sub 20 nupb-gg-new "${FE[@]}" -- bash -c "$(cmd b2s0 NPBGG-new 60 50)") || exit 1
  c=$(sub 20 nupb-gg-sm "${FE[@]}" -- bash -c "$(cmd 70s0 NPBSM 60 50)") || exit 1
  g=$(sub 20 nupb-gate --after $a,$b,$c --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash -c "bash $ME ident $REV $h && bash $ME smoke") || exit 1
  echo "$(date '+%F %T') gates (code $h, old $REV): runs $a $b $c gate $g" | tee -a $O/stages.txt
  mapfile -t FO < <($V -m jevdrive.cl fanout --name nupb-train --arms $(tr ' ' , <<< "$ARMS") --owner round9-nupb --priority 20 --tries 1 --no-check \
      --after $g --log-dir "$O/pool/nupb-train-{arm}" "${TRN[@]}" --timeout-h 4 -- bash $ME train {arm} < /dev/null | grep -E '^[0-9]')
  [[ ${#FO[@]} == 4 ]] || die "fanout: ${FO[*]}"; printf '%s\n' "${FO[@]}" >> $O/jobs.txt
  R=(); for l in "${FO[@]}"; do t=${l%% *}; a=${l##*-}; T=$(tag $a)
    for c in $(seq 0 17); do R+=($(TRIES=2 sub 19 nupb-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    for i in 0 1 2 3 4 5; do R+=($(TRIES=2 sub 19 nupb-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    echo "$(date '+%F %T') arm $a $T: train $t reads ${R[-24]} .. ${R[-1]}" | tee -a $O/stages.txt; done
  s=$(sub 19 nupb-sum --after $(csv ${R[@]}) --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') summary $s" | tee -a $O/stages.txt ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a NPBGG-old --b NPBGG-new --out round9_nupb_identity --extra-ok \
      --note "B2-s0's command (init RB0-F-s0) 60 steps eager, warm-up 50: code $2 at $DATA_DIR/tmp/df_old_$2 vs $3"; r=$?
  mv experiments/defect_fix/results/round9_nupb_identity.json $RES/identity.json || r=4; cp $RES/identity.json $O/; exit $r ;;
smoke)
  $TR experiments/defect_fix/scripts/stand_b.py smoke --b NPBGG-new --b1 NPBSM --finite-only --out $O/smoke_finite.json > /dev/null; r=$?
  $TR $PY smoke --ok $r --ctl NPBGG-new --arm NPBSM --init RB0-P70-s0 --finite $O/smoke_finite.json --out $RES/smoke.json || exit 2
  rm -rf $RUNS/NPBGG-old $RUNS/NPBGG-new $RUNS/NPBSM ;;
train)
  python3 -c "import json, sys; r = '$RES'; sys.exit(0 if json.load(open(f'{r}/identity.json'))['identical'] and json.load(open(f'{r}/smoke.json'))['pass_'] else 1)" \
    || die "identity / smoke not passed: no training"
  C=$(cmd $2) || exit 1; echo "$(date '+%F %T') train $(tag $2) (code $(git rev-parse --short HEAD)): $C" | tee -a $O/stages.txt; exec $C ;;
sum)
  T=(); for a in $ARMS; do T+=($(tag $a)); done
  $V -m jevdrive.facts table --bench arena --tags ${T[@]} --refs $REFS --out $O/arena.md --title "nuPlan arena held, native: NPB (PAI-mixed base + B2 recipe) vs B2 / JTN-L18-s1" \
    || die "arena table"
  $V -m jevdrive.facts table --bench held460 --tags ${T[@]} --refs $REFS --out $O/held460.md --title "held 460 @18: NPB vs B2 / JTN-L18-s1" || die "held table"
  $V -m jevdrive.facts table --bench arena --tags RB0-P70-s0 --refs RB0-F-s0 --out $O/arena_base.md --title "base stage (d449): RB0-P70-s0 vs RB0-F-s0" || die "base"
  X=(); for t in ${T[@]} $REFS; do X+=($t=$t@off); done
  $AP -m jevdrive.facts ext --read "${X[@]}" --out $O/ext || die "ext"
  $V -c "import pandas as pd; w = pd.read_parquet('$O/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$O/arena_windows.csv.gz', index=False)" \
    || die "windows"
  cp $O/arena.md $O/arena.json $O/held460.md $O/held460.json $O/arena_base.md $O/arena_windows.csv.gz $RES/ || die copy
  date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[-1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
