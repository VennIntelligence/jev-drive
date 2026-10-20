#!/usr/bin/env bash
# NU-FLAT (plans/2026-10-20-nu-flat-prereg.md): three continuations of the nuPlan candidate B2 STANDFMXH-L18-s1 with its exact training command
# (stand_b/cmd_FHS.txt; only tag, --init, --steps, --warmup change), one seed. CTRL = the recipe; F = + SIMP-G onset flattening
# (--flat-brk flat_onset.txt --flat-p 0.5, ROUND2's setting); FR = F + STAND-R x 1 (--on-stand-roll 3.0, decision 399's R1). Steps from the
# registered rule (nu_flat.py smoke -> results/nu_flat/steps.json). Reads per arm: arena held 18 chunks native, held 460 @18, wait set v2; plus
# B2-s1's wait set (reference). Box, repo root; every job through the GPU pool below ROUND2 (PRIO 4, reads PRIO + 1; ROUND2 is 12 / 13).
#   nu_flat.sh check            DRY commands of the three arms against the stored B2-s1 command -> results/nu_flat/cmd_diff.txt
#   nu_flat.sh pre <old rev>    identity (CTRL's command 60 steps eager: <old rev> from its own worktree vs this code without new flags, and
#                               this code with --flat-brk <list> --flat-p 0) + smoke (FR's command 60 steps eager -> smoke.json, steps.json)
#   nu_flat.sh launch           3 trainings (each re-checks identity / smoke / steps) -> per arm reads -> nsum -> $S/DONE
#   nu_flat.sh wait             blocks until $S/DONE or a failed job of this lane (one run_in_background waiter); every round it runs
#                               nu_flat.py guard (cards 2, 3, 4 only while ROUND2's trainings / PAI-FIX chunks are queued)
#   job bodies: ident <old> <new> | smoke | train <arm> | nsum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; S=$O/nu_flat; J=$O/jt; ME=experiments/defect_fix/scripts/nu_flat.sh; PY=experiments/defect_fix/scripts/nu_flat.py
SP=experiments/defect_fix/scripts/stand.py; FM=experiments/defect_fix/scripts/free_m.py; JT=experiments/defect_fix/scripts/jt.sh
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
RES=experiments/defect_fix/results/nu_flat; RUNS=$DATA_DIR/runs/op_parity/runs; AH=$DATA_DIR/runs/alpasim_proxy/arena/hold
B2=STANDFMXH-L18-s1; B2CMD=$O/stand_b/cmd_FHS.txt; FLATG=$O/simp/flat_onset.txt; FP=0.5; VR=3.0; SUB="0 3 6 9 12 15"
declare -A TAG=([C]=B2C-L18-s1 [F]=B2F-L18-s1 [FR]=B2FR-L18-s1) LAB=([C]=B2C [F]=B2F [FR]=B2FR)
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
TRN=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $S/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $S/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-4} \
     --no-check --log-dir $S/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $S/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() {  # cmd <arm> <tag> <steps> <warm>: B2-s1's stored command, tag / init / steps / warm-up swapped, + the arm's switches
  local A=$1 C; C=$(cat $B2CMD) || die "no $B2CMD"
  [[ $C == *" --tag $B2 "* && $C == *" --init RB0-F-s0 "* && $C == *" --steps 9500 "* && $C == *" --warmup 500 "* && $C == *" --seed 1 "* \
     && $C == *" --on-stand 0.0068 "* && $C == *" --on-free 0.001 "* && $C == *" --allow-init "* && $C != *"--flat"* && $C != *"--on-stand-roll"* ]] \
     || die "unexpected B2-s1 command"
  C=${C/ --tag $B2 / --tag $2 }; C=${C/ --init RB0-F-s0 / --init $B2 }; C=${C/ --steps 9500 / --steps $3 }; C=${C/ --warmup 500 / --warmup $4 }
  case $A in C) ;; F) C="$C --flat-brk $FLATG --flat-p $FP" ;; FR) C="$C --flat-brk $FLATG --flat-p $FP --on-stand-roll $VR" ;; *) die "arm $A" ;; esac
  echo "$C"; }
nw() { python3 -c "import json; d = json.load(open('$RES/steps.json')); print(d['steps'], d['warmup'])"; }
case ${1:-} in
check)
  read N W < <(nw 2> /dev/null || echo "3000 150"); cp $B2CMD $S/cmd_b2.txt
  for A in C F FR; do cmd $A ${TAG[$A]} $N $W > $S/cmd_$A.txt || exit 1; done
  { echo "== $B2 (stored training command) -> each arm (N $N, W $W)"
    for A in C F FR; do echo "== -> ${TAG[$A]}"; diff <(tr ' ' '\n' < $S/cmd_b2.txt) <(tr ' ' '\n' < $S/cmd_$A.txt); done; } > $RES/cmd_diff.txt
  cat $RES/cmd_diff.txt
  declare -A NT=([C]=4 [F]=8 [FR]=10)   # tag, init, steps, warm-up (+ --flat-brk FILE --flat-p P) (+ --on-stand-roll V)
  for A in C F FR; do n=$(diff <(tr ' ' '\n' < $S/cmd_b2.txt) <(tr ' ' '\n' < $S/cmd_$A.txt) | grep -c '^>'); [[ $n == "${NT[$A]}" ]] || die "arm $A: $n new tokens, want ${NT[$A]}"; done
  echo "diff ok" ;;
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in NFGG-old NFGG-new NFGGP0-new NFSM-L18-s1; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  [[ -s $FLATG && -s $DATA_DIR/runs/op_parity/cache/on1_JTX/stand_gap.npz ]] || die "no flat list / stand_gap labels"
  h=$(git rev-parse --short HEAD); echo "$h pre (old $rev)" >> $S/code.txt; FE=("${TRN[@]}" --env JEV_RM_EAGER=1 --timeout-h 1)
  a=$(sub nf-gg-old "${FE[@]}" -- bash -c "cd $WT && $(cmd C NFGG-old 60 150)") || exit 1          # the old code runs from its worktree
  b=$(sub nf-gg-new "${FE[@]}" -- bash -c "$(cmd C NFGG-new 60 150)") || exit 1
  c=$(sub nf-gg-p0 "${FE[@]}" -- bash -c "$(cmd C NFGGP0-new 60 150) --flat-brk $FLATG --flat-p 0") || exit 1
  e=$(sub nf-ident --after $a,$b,$c --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident $rev $h) || exit 1
  s=$(sub nf-smoke "${FE[@]}" -- bash $ME smoke) || exit 1
  echo "$(date '+%F %T') pre (code $h, old $rev): identity $a $b $c bitcmp $e smoke $s" | tee -a $S/stages.txt; echo "$e $s" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a NFGG-old --b NFGG-new --out nu_flat/identity_off \
      --note "B2-s1 continuation command (CTRL) 60 steps eager: code $2 at $DATA_DIR/tmp/df_old_$2 vs $3, no new flag"
  $TR experiments/defect_fix/scripts/df.py bitcmp --a NFGG-old --b NFGGP0-new --out nu_flat/identity_p0 \
      --note "the same: code $2 without --flat-brk vs $3 with --flat-brk <SIMP-G onset list> --flat-p 0"
  $TR $PY ident; r=$?; rm -rf $RUNS/NFGG-old $RUNS/NFGG-new $RUNS/NFGGP0-new; exit $r ;;
smoke)   # FR's command (all new switches), 60 steps eager -> smoke.json and the registered step rule -> steps.json
  X=NFSM-L18-s1; C="$(cmd FR $X 60 150)" || exit 1; echo "$(date '+%F %T') smoke: $C" | tee -a $S/stages.txt
  $C > $S/smoke_train.log 2>&1 || die "smoke run (log $S/smoke_train.log)"
  $TR $PY smoke --tag $X --cmd-file $B2CMD --flat-p $FP; r=$?; rm -rf $RUNS/$X; exit $r ;;
launch)
  for A in C F FR; do [[ ! -e $RUNS/${TAG[$A]}/ckpt-final.pt ]] || die "${TAG[$A]} exists"; done
  python3 -c "import json, sys; r = '$RES'; sys.exit(0 if all(json.load(open(f'{r}/{n}.json'))[k] for n, k in (('identity', 'identical'), ('smoke', 'pass_'))) else 1)" \
    || die "identity / smoke not passed"
  bash $ME check > /dev/null || die "command diff"
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $S/code.txt; PR=$((${PRIO:-4} + 1)); ids=()
  for A in C F FR; do T=${TAG[$A]}
    t=$(sub nf-train-$T "${TRN[@]}" --timeout-h 2 -- bash $ME train $A) || exit 1
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub nf-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
    for c in $(seq 0 $((nch - 1))); do ids+=($(PRIO=$PR TRIES=2 sub nf-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    echo "$(date '+%F %T') arm $A $T: train $t" | tee -a $S/stages.txt; done
  if [[ ! -f $DATA_DIR/runs/alpasim_proxy/wait_set/reads/$B2/table/$B2.csv ]]; then
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $B2 | tail -1); [[ -n $r ]] || die "readc $B2"; echo "$r ws-collect-$B2" >> $S/jobs.txt; ids+=($r); fi
  p=$(sub nf-nsum --after $(csv ${ids[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME nsum) || exit 1
  echo "$(date '+%F %T') launch (code $h): reads ${ids[*]} nsum $p" | tee -a $S/stages.txt; echo "nsum $p" ;;
train)
  A=$2; python3 -c "import json, sys; r = '$RES'; sys.exit(0 if all(json.load(open(f'{r}/{n}.json'))[k] for n, k in (('identity', 'identical'), ('smoke', 'pass_'))) else 1)" \
    || die "pre-checks failed: no training"
  read N W < <(nw) || die "no steps.json"; CMD="$(cmd $A ${TAG[$A]} $N $W)" || exit 1
  echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $S/stages.txt; exec $CMD ;;
nsum)    # extract (18 chunks, + 716-window subset via a chunk symlink farm), arena vs CTRL (+ B2-s1), drift, stand.py tables, held, windows, costs
  T4=(${TAG[C]} ${TAG[F]} ${TAG[FR]} $B2); rm -rf $S/sub; X=(); XS=()
  for T in ${T4[@]}; do X+=("$T=$J/arena/$T@off"); XS+=("$T=$S/sub/$T@off"); mkdir -p $S/sub/$T@off
    for c in $SUB; do [[ -d $J/arena/$T@off/c$c/run ]] || die "no $T chunk $c"; ln -s $J/arena/$T@off/c$c $S/sub/$T@off/c$c; done; done
  $AP $FM ext --read "${X[@]}" --out $S/ext_full || die "extract full"
  $AP $FM ext --read "${XS[@]}" --out $S/ext_sub || die "extract subset"
  runs=(); for t in ${TAG[C]}:${LAB[C]} ${TAG[F]}:${LAB[F]} ${TAG[FR]}:${LAB[FR]} $B2:B2S1; do
    d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
    [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
  $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $S/follow --cover none --workers 16 > $S/follow.log 2>&1 || die "rf_follow"
  for A in C F FR; do T=${TAG[$A]}
    if [[ $A == C ]]; then B=$B2; BL=B2S1; REF=(); else B=${TAG[C]}; BL=${LAB[C]}; REF=(--ref $B2); fi
    $V $FM arena --src $S/ext_full --a $T --b $B "${REF[@]}" --out $RES/arena_full_$A.json || die "arena full $A"
    $V $FM arena --src $S/ext_sub --a $T --b $B --out $S/arena_sub_$A.json || die "arena subset $A"
    $AP $SP drift --follow $S/follow --a ${LAB[$A]}@18 --b $BL@18 --out $S/drift_$A.json || die "drift $A"
    cp $S/arena_sub_$A.json $S/drift_$A.json $RES/
    $TR $SP pilot --tag $T --ctl $B --arena $S/arena_sub_$A.json --drift $S/drift_$A.json --ext $S/ext_sub --drift-guard --out $RES/s17table_$A.md > $S/s17_$A.txt \
      || die "table $A"; done
  $TR $PY held --tags ${T4[@]} || die "held"
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext_full/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows.csv.gz', index=False)" || die "windows"
  $TR $PY costs --tags ${TAG[C]} ${TAG[F]} ${TAG[FR]} || die "costs"
  date '+%F %T' > $S/DONE ;;
wait)
  until [[ -f $S/DONE ]] || ! $V - $S/jobs.txt <<'PY'
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
  do $V $PY guard || echo "guard failed" >&2; sleep 300; done
  [[ -f $S/DONE ]] && echo "DONE $(cat $S/DONE)"; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
