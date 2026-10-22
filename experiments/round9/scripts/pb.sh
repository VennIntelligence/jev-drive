#!/usr/bin/env bash
# ROUND9 PAI-BASE (plans/2026-10-22-pai-base.md). Box, repo root; every job through the pool, the whole chain in one submit; ids in $O/jobs.txt,
# stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/pai_base.
#   pb.sh all        wfile + data check -> smoke (RB0-PSM, 300 steps, share 50 %) -> the three bases RB0-P30 / P50 / P70-s0 (cl fanout, eager as
#                    RB0-F) -> each: PAI 281 OFF (8 chunk jobs + tax), draw replay, nuPlan arena held native (18 chunks, guard); RB0-F-s0's 281
#                    OFF and arena read at once; identity = RB0-F-s0's argv rerun (RB0-FR-s0) -> bitcmp; base gate (r281 + pb.py basegate ->
#                    CARRY or exit 3) -> PBNT-L18-s0 / s1 (NTV0-L18-s0's argv, --init CARRY) -> PBDAG-L18-s0 / s1 (R6DAG-L18-s0's argv,
#                    --init PBNT-L18-s<s>) -> PAI 281 OFF + ON each -> ruler (collect + 20 boots) -> final report -> $O/DONE
#   pb.sh wait [M]   one run_in_background waiter: the marker (default DONE), or a failed / cancelled job of this lane
#   pb.sh printargv base <K> <tag> | l18 <seed> <tag> | dag <seed> <init> <tag>
#   job bodies: wfile | train base <K> <tag> [steps] | train l18 <seed> | train dag <seed> | smoke | draws <share> | ident | arena-sum |
#               gate | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/pai_base; ME=experiments/round9/scripts/pb.sh; PY=experiments/round9/scripts/pb.py
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; RUNS=$DATA_DIR/runs/op_parity/runs; F_=$DATA_DIR/runs/pai_fix/facts
W=$O/w/RB0P.npz; RES=experiments/round9/results/pai_base; JT=experiments/defect_fix/scripts/jt.sh
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
declare -A K=([30]=35 [50]=58 [70]=80)                # PAI rows of the 115 imitation (normal) rows per batch: round(share x 115)
FT=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train)
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
mkdir -p $O/pool $O/results $O/w $O/pai281 $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() {  # sub <prio> <name> <pool args...>
  local p=$1 name=$2; shift 2; local id
  id=$($V -m jevdrive.cl submit --owner round9-pb --name "$name" --tries 1 --priority $p --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') \
    && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
argv() {  # argv base <K> <tag> | l18 <seed> <tag> | dag <seed> <init> <tag>: one argument a line, from the reference run's meta.json
  python3 - "$@" <<'PY'
import glob, json, os, sys
D = os.environ["DATA_DIR"]
v = sys.argv[1]
ref = dict(base="RB0-F-s0", l18="NTV0-L18-s0", dag="R6DAG-L18-s0")[v]
a = json.load(open(sorted(glob.glob(f"{D}/runs/route_mem/train-{ref}/*/meta.json"))[-1]))["argv"]
def put(k, x):
    assert a.count(k) == 1, (k, a)
    a[a.index(k) + 1] = x
if v == "base":
    k, tag = int(sys.argv[2]), sys.argv[3]
    put("--tag", tag)
    if k:                                     # mode imit: no row carries the agent hinge, = what rm_train makes of RB0-F's --agent-on-only
        a.remove("--agent-on-only"); put("--agent-lam", "0")
        a += ["--on", "on1_JTPH", "--on-k", str(k), "--on-mode", "imit", "--on-normal", "--on-wfile", f"{D}/runs/round9/pai_base/w/RB0P.npz",
              "--probe-n", "0"]               # monitoring only: R6FIX's grp labels would switch on JT's wait / launch probes (they need nr1_lat)
elif v == "l18":
    put("--tag", sys.argv[3]); put("--seed", sys.argv[2])
    put("--init", open(f"{D}/runs/round9/pai_base/CARRY").read().split()[0] if sys.argv[2:3] else "")
else:
    put("--tag", sys.argv[4]); put("--seed", sys.argv[2]); put("--init", sys.argv[3])
print("\n".join(a))
PY
}
read281() {  # read281 <tag> <ag|agH> <after> <prio> -> "<chunk ids...> <tax id>" (NTV0's chunk table, ACC + GAP, g10, FAST, CONC 4)
  local R=$O/pai281/$1-$2 ids t sv="$SERVE" ex=(); [[ ! -e $R ]] || die "$R exists"
  [[ $2 == agH ]] && { sv="$sv HARMONIZER=1"; ex=(VRAM=30 RAM=40); }
  env "${ex[@]}" ${3:+AFTER=$3} CHUNKS_FROM=$CH SERVE="$sv" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$4 bash experiments/alpasim/scripts/pai_box.sh submit $R $1 \
    > $R.submit.txt || die "pai submit $1 $2"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$1 $2: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt | sed "s/\$/-$2/" >> $O/jobs.txt
  t=$(PRIO=$4 AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $1-281$2 $R | tail -1) && [[ -n $t ]] || die "tax $1 $2"
  echo "$t pai_fix-tax-$1-281$2" >> $O/jobs.txt; echo $ids $t; }
arena() {  # arena <tag> <after> -> 18 chunk ids (jt.sh aread, native: off)
  local ids=() c; for c in $(seq 0 17); do ids+=($(sub 18 pb-ar-$1-c$c ${2:+--after $2} "${RD[@]}" --tries 2 -- bash $JT aread $1 off $c)) || exit 1; done; echo ${ids[@]}; }
case ${1:-} in
all)
  [[ ! -e $O/GATES && ! -e $O/DONE && ! -e $O/CARRY ]] || die "already ran"
  last() { awk -v n=$1 '$2 == n {x = $1} END {print x}' $O/jobs.txt; }
  if [[ -n ${REUSE:-} ]]; then                         # resubmit after a failed smoke: keep the identity rerun and RB0-F-s0's reads (independent)
    ic=$(last pb-ident); rft=$(last pai_fix-tax-RB0-F-s0-281ag); RAR=(); for c in $(seq 0 17); do RAR+=($(last pb-ar-RB0-F-s0-c$c)); done
    [[ -n $ic && -n $rft && ${#RAR[@]} == 18 ]] || die "REUSE: ids"; rm -rf $RUNS/RB0-PSM-s0 $DATA_DIR/runs/route_mem/train-RB0-PSM-s0 $DATA_DIR/tmp/pcs_pb
    echo "$(date '+%F %T') REUSE: identity $ic, RB0-F-s0 281 tax $rft, arena ${RAR[0]} .. ${RAR[-1]}" | tee -a $O/stages.txt; fi
  for X in RB0-P30-s0 RB0-P50-s0 RB0-P70-s0 RB0-PSM-s0 PBNT-L18-s0 PBNT-L18-s1 PBDAG-L18-s0 PBDAG-L18-s1 ${REUSE:-RB0-FR-s0}; do
    [[ $X == 1 || ! -e $RUNS/$X ]] || die "$X exists"; done
  for f in NTV0-L18-s0-281ag NTV0-L18-s0-281agH R6DAG-L18-s0-281ag R6DAG-L18-s1-281ag R6DAG-L18-s2-281ag R6DAG-L18-s0-281agH R6DAG-L18-s1-281agH \
           R6DAG-L18-s2-281agH; do [[ -s $F_/$f.csv ]] || die "no stored $f"; done
  P0=$DATA_DIR/tmp/pcs_r8k2; PT=$DATA_DIR/tmp/pcs_pb; [[ ! -e $PT ]] || die "$PT exists"; for f in $P0/rows.csv.gz $P0/out/fits/paimix.npz; do [[ -s $f ]] || die "no $f"; done
  h=$(git rev-parse --short HEAD); echo "$h all" >> $O/code.txt
  # --- gates: draw file + data check, smoke, identity
  wf=$(sub 22 pb-wfile --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- bash $ME wfile) || exit 1
  sm=$(sub 22 pb-smoke --after $wf "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash -c "bash $ME train base 58 RB0-PSM-s0 300 && bash $ME smoke") || exit 1
  [[ -n ${REUSE:-} ]] || { ir=$(sub 21 pb-ident-run "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 2 -- bash $ME train base 0 RB0-FR-s0) || exit 1
    ic=$(sub 21 pb-ident --after $ir --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1; }
  echo "$(date '+%F %T') gates (code $h): wfile $wf smoke $sm identity ${ir:-reused} $ic" | tee -a $O/stages.txt
  # --- the three bases in parallel (one fanout), each with its own reads; RB0-F-s0's reads now
  mapfile -t FO < <($V -m jevdrive.cl fanout --name pb-base --arms 30,50,70 --owner round9-pb --priority 21 --tries 1 --no-check --after $sm \
      --log-dir "$O/pool/pb-base-{arm}" "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 2 -- bash $ME train base {arm} < /dev/null | grep -E '^[0-9]')
  [[ ${#FO[@]} == 3 ]] || die "fanout: ${FO[*]}"; printf '%s\n' "${FO[@]}" >> $O/jobs.txt
  declare -A TB; for l in "${FO[@]}"; do TB[${l##*-}]=${l%% *}; done
  if [[ -n ${REUSE:-} ]]; then y=($rft); AR=(${RAR[@]}); else y=($(read281 RB0-F-s0 ag "" 21)) || exit 1; AR=($(arena RB0-F-s0 "")) || exit 1; fi; G=(${y[-1]} $ic)
  echo "$(date '+%F %T') RB0-F-s0: 281 OFF ${y[*]} arena ${AR[0]} .. ${AR[-1]}" | tee -a $O/stages.txt
  for s in 30 50 70; do t=${TB[$s]}; [[ -n $t ]] || die "no base id $s"
    y=($(read281 RB0-P$s-s0 ag $t 21)) || exit 1
    d=$(sub 21 pb-draws-$s --after $t --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- bash $ME draws $s) || exit 1
    a=($(arena RB0-P$s-s0 $t)) || exit 1; AR+=(${a[@]}); G+=(${y[-1]} $d)
    echo "$(date '+%F %T') base $s % (K ${K[$s]}): train $t 281 OFF ${y[*]} draws $d arena ${a[0]} .. ${a[-1]}" | tee -a $O/stages.txt; done
  as=$(sub 18 pb-arena-sum --after $(csv ${AR[@]}) --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $ME arena-sum) || exit 1
  g=$(sub 21 pb-gate --after $(csv ${G[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME gate) || exit 1
  echo "$(date '+%F %T') base gate $g (after ${G[*]}), arena summary $as" | tee -a $O/stages.txt
  # --- the carried base through the recipe (speculative: fails with the gate)
  x=()
  for s in 0 1; do l=$(sub 21 pb-l18-s$s --after $g "${FT[@]}" --timeout-h 3 -- bash $ME train l18 $s) || exit 1
    t=$(sub 21 pb-dag-s$s --after $l "${FT[@]}" --timeout-h 4 -- bash $ME train dag $s) || exit 1
    y=($(read281 PBDAG-L18-s$s ag $t 21)) || exit 1; z=($(read281 PBDAG-L18-s$s agH $t 21)) || exit 1; x+=(${y[-1]} ${z[-1]})
    echo "$(date '+%F %T') stage 2 s$s: L18 $l DAG $t OFF ${y[*]} ON ${z[*]}" | tee -a $O/stages.txt; done
  mkdir -p $PT/out/fits && cp $P0/out/fits/*.npz $PT/out/fits/
  X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; EV=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$DATA_DIR/tmp/pcs_r5/site)
  c=$(sub 21 pb-pcs-collect --after $(csv ${x[0]} ${x[2]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" $V $X collect --out $PT/rows.csv.gz \
      --only PBDAG-s0+AG,PBDAG-s1+AG --base $P0/rows.csv.gz) || exit 1
  b=(); for k in $(seq 0 5 95); do b+=($(sub 21 pb-pcs-boot-$k --after $c --vram 0.1 --cpu 4 --ram 16 --timeout-h 3 -- "${EV[@]}" $V $X fit --rows $PT/rows.csv.gz \
      --out $PT/out --variants - --boot 5 --boot0 $k)) || exit 1; done
  r=$(sub 21 pb-report --after $(csv ${x[@]} ${b[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- "${EV[@]}" bash $ME report) || exit 1
  echo "$(date '+%F %T') ruler collect $c boot ${b[0]} .. ${b[-1]} report $r" | tee -a $O/stages.txt ;;
wfile) $V $PY wfile --out $W --json $O/results/wfile.json || die "wfile / data check" ;;
train)  # train base <K|share> <tag> [steps] | train l18 <seed> | train dag <seed>
  case $2 in
  base) k=$3; [[ -n ${K[$k]:-} ]] && { tag=RB0-P$k-s0; k=${K[$k]}; } || tag=$4
        mapfile -t A < <(argv base $k $tag); [[ ${#A[@]} -gt 5 ]] || die "argv base"
        [[ -n ${5:-} ]] && A+=(--steps $5 --eval-every 100) ;;
  l18) tag=PBNT-L18-s$3; [[ -s $O/CARRY ]] || die "no CARRY"; mapfile -t A < <(argv l18 $3 $tag); [[ ${#A[@]} -gt 5 ]] || die "argv l18" ;;
  dag) tag=PBDAG-L18-s$3; mapfile -t A < <(argv dag $3 PBNT-L18-s$3 $tag); [[ ${#A[@]} -gt 5 ]] || die "argv dag" ;;
  esac
  echo "$(date '+%F %T') train $tag (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
smoke) $V $PY smoke --tag RB0-PSM-s0 --k 58 --out $O/results/smoke.json; r=$?; (( r == 0 )) && rm -rf $RUNS/RB0-PSM-s0; exit $r ;;
draws) $V $PY draws --tag RB0-P$2-s0 --k ${K[$2]} --steps 10000 --w $W --out $O/results/draws_$2.json ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a RB0-F-s0 --b RB0-FR-s0 --out round9_pb_identity \
      --extra-ok --note "RB0-F-s0's argv (10-13, code 2962dc63 dirty) rerun on HEAD, eager, 10 000 steps: the PAI-share-0 base"; r=$?
  mv experiments/defect_fix/results/round9_pb_identity.json $O/results/identity.json || r=4; exit $r ;;
arena-sum)
  $V -m jevdrive.facts table --bench arena --tags RB0-P30-s0@off RB0-P50-s0@off RB0-P70-s0@off --refs RB0-F-s0@off --out $O/results/arena_base.md \
      --title "nuPlan arena held, native (guard, informational): RB0-P vs RB0-F" || die "arena table" ;;
gate)
  rd=(RB0F-ag=$F_/RB0-F-s0-281ag.csv); pr=(); for s in 30 50 70; do rd+=(RB0P$s-ag=$F_/RB0-P$s-s0-281ag.csv); pr+=(RB0P$s-ag:RB0F-ag); done
  $TR experiments/round2/scripts/r2.py r281 --read "${rd[@]}" --pairs "${pr[@]}" --out $O/results/pai281_base.md \
      --title "PAI 281, ACC + GAP, Harmonizer OFF: bases RB0-P30 / P50 / P70-s0 (PAI rows from the start) vs RB0-F-s0" || die r281
  $V $PY basegate --r281 $O/results/pai281_base.json --draws $O/results/draws_{30,50,70}.json --ident $O/results/identity.json --out $O/results/gate
  r=$?; date '+%F %T' > $O/GATES; [[ $r == 0 ]] || { echo "base gate: no share carried (rc $r)" > $O/VERDICT; exit 3; } ;;
report)
  PT=$DATA_DIR/tmp/pcs_pb; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py
  sj="R3MV-s0+AG=R3MV-L18-s0-281ag.csv"; for s in 0 1 2; do sj="$sj,DAG-s$s+AG=R6DAG-L18-s$s-281ag.csv"; done
  sj="$sj,PBDAG-s0+AG=PBDAG-L18-s0-281ag.csv,PBDAG-s1+AG=PBDAG-L18-s1-281ag.csv"
  $V $X r5 --rows $PT/rows.csv.gz --out $PT/out --facts $F_ --subjects "$sj" --pairs PBDAG-s0+AG:DAG-s0+AG,PBDAG-s1+AG:DAG-s1+AG > $PT/out/r5.log 2>&1 \
    || die "ruler report (log $PT/out/r5.log)"
  cp $PT/out/r5_report.md $O/results/pcs_pb.md && cp $PT/out/pai_pcs_local_r5.json $O/results/pcs_pb.json || die "copy ruler report"
  rd=(); for m in ag agH; do rd+=(NTV0-$m=$F_/NTV0-L18-s0-281$m.csv); for s in 0 1 2; do rd+=(R6DAG-s$s-$m=$F_/R6DAG-L18-s$s-281$m.csv); done
    for s in 0 1; do rd+=(PBDAG-s$s-$m=$F_/PBDAG-L18-s$s-281$m.csv); done; done
  pr=(); for s in 0 1; do for m in ag agH; do pr+=(PBDAG-s$s-$m:R6DAG-s$s-$m PBDAG-s$s-$m:NTV0-$m); done; done; pr+=(PBDAG-s1-ag:PBDAG-s0-ag)
  $TR experiments/round2/scripts/r2.py r281 --read "${rd[@]}" --pairs "${pr[@]}" --out $O/results/pai281_final.md \
      --title "PAI 281, ACC + GAP: PBDAG s0 / s1 (PAI-from-start base -> L18 -> DAgger) vs R6DAG s0 / s1 / s2 and NTV0; Harmonizer OFF and ON" || die r281
  $V $PY final --read "${rd[@]}" --md $O/results/pai281_final.md --pcs $O/results/pcs_pb.json --out $O/results/final || die final
  date '+%F %T' > $O/DONE ;;
printargv) argv "${@:2}" ;;
wait)
  m=${2:-DONE}
  until [[ -f $O/VERDICT || -f $O/$m ]] || ! $V - $O/jobs.txt <<'PY'
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
  cat $O/VERDICT 2>/dev/null; for f in GATES CARRY DONE; do [[ -f $O/$f ]] && echo "$f $(cat $O/$f)"; done; tail -3 $O/stages.txt; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
