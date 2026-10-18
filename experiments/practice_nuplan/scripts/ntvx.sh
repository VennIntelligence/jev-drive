#!/usr/bin/env bash
# NTVX-L18-s0, the joint arm (plans/2026-10-19-pn-prereg.md supplement 1): NTV0-L18-s0's command with only the pool changed, on1_JT -> on1_JTX,
# and its S-only draw rebuilt by JT00S's rule (JTX00S). Box, repo root. Calls jt.sh unchanged (train DRY print, aread) and pai_box.sh with
# ntv0_reads.sh's serve config; every job through the pool, --after chained; ids in $W/jobs.txt, stages in $W/stages.txt.
#   ntvx.sh check            DRY diff of the training command against NTV0-L18-s0's (only --tag / --on / --on-wfile may differ) -> $W/cmd_diff.txt
#   ntvx.sh submit [after]   weights JTX00S -> train (prio 12) -> held 460 @18 (6), arena held stop3 + off (18 + 18), wait set v2 (readc),
#                            PAI stage (prio 13: submits PAI 281 ACC+GAP and the summary after every read)
#   ntvx.sh weights | train | pai | sum   job bodies; sum writes results/NTVX-L18-s0/ and $W/DONE (ERROR on a failure)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTVX-L18-s0; T0=NTV0-L18-s0; O=$DATA_DIR/runs/defect_fix; JD=$O/jt; W=$DATA_DIR/runs/practice_nuplan/ntvx
E=experiments/practice_nuplan; ME=$E/scripts/ntvx.sh; JT=experiments/defect_fix/scripts/jt.sh; R=$E/results/$T
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; WSH=experiments/alpasim_proxy/scripts/wait_set.sh
LAT=$O/lat/L18NR1v2; AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $W/stages.txt >&2; echo "$*" > $W/ERROR; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner practice-nuplan --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
cmd() { DRY=1 ONK=32 ARCROWS=brake bash $JT train $1 $2 $LAT $3 9500 0 | tail -1; }   # cmd <tag> <pool> <draw>: jt.sh's own command
case ${1:-} in
check)
  diff <(cmd $T0 on1_JT JT00S | tr ' ' '\n') <(cmd $T on1_JTX JTX00S | tr ' ' '\n') > $W/cmd_diff.txt; cat $W/cmd_diff.txt
  [[ $(grep -c '^<' $W/cmd_diff.txt) -eq 3 && $(grep -c '^>' $W/cmd_diff.txt) -eq 3 ]] || die "command diff beyond three tokens"
  [[ $(grep '^>' $W/cmd_diff.txt | tr '\n' ' ') == "> $T > on1_JTX > runs/defect_fix/w/JTX00S.npz " ]] || die "unexpected diff tokens"
  echo "diff ok: --tag, --on, --on-wfile only" ;;
submit)
  A=${2:-}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
  [[ ! -e $DATA_DIR/runs/alpasim_proxy/wait_set/reads/$T/table ]] || die "wait set read of $T exists"
  rm -f $W/ERROR $W/DONE; bash $ME check || exit 1
  w=$(sub ntvx-w-JTX00S ${A:+--after $A} --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME weights) || exit 1
  t=$(sub ntvx-train --after $w --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 3 -- bash $ME train) || exit 1
  ids=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for i in 0 1 2 3 4 5; do
    ids+=($(PRIO=13 TRIES=2 sub ntvx-held-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  for V in stop3 off; do for c in $(seq 0 $((nch - 1))); do
    ids+=($(PRIO=13 TRIES=2 sub ntvx-ar-$V-c$c --after $t --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $JT aread $T $V $c)) || exit 1; done; done
  ws=$(PRIO=13 bash $WSH readc $T $t | tail -1); [[ -n $ws ]] || die "readc $T"; echo "$ws ws-collect-$T" >> $W/jobs.txt; ids+=($ws)
  echo "${ids[*]}" > $W/reads.txt          # pai_box.sh submit needs the checkpoint, so PAI and the summary come from a stage job after training
  pj=$(PRIO=13 sub ntvx-pai --after $t --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME pai) || exit 1
  echo "$(date '+%F %T') submit $T: weights $w train $t reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}, wait-set collect $ws) PAI stage $pj" | tee -a $W/stages.txt ;;
weights)
  $TR experiments/defect_fix/scripts/jt.py weights --pool on1_JTX --k-s 32 --k-tv 0 --hold-pct 15 --tv D10 --no-wait --out JTX00S || die "weights rc $?"
  python3 - experiments/defect_fix/results/jt/w_JT00S.json experiments/defect_fix/results/jt/w_JTX00S.json <<'PY' || die "held-out logs differ from JT00S"
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:])
assert a["held_logs"] == b["held_logs"] and a["s_logs"] == b["s_logs"] == 28 and b["k_tv"] == 0 and b["on_k"] == 32, (a["held_logs"], b["held_logs"])
n0, n1 = a["rows_by_group"]["S"], b["rows_by_group"]["S"]
print(f"held logs identical; trainable S rows {n0} -> {n1}; draws per S row over 9 500 x 32: {304000 / n0:.2f} -> {304000 / n1:.2f}")
PY
  ;;
train)
  C=$(cmd $T on1_JTX JTX00S); [[ $C == *"--tag $T "* && $C == *"--on on1_JTX "* && $C == *"w/JTX00S.npz "* && $C == *"--on-k 32 "* ]] || die "bad command: $C"
  echo "$(date '+%F %T') train $T: $C" | tee -a $W/stages.txt
  exec $C ;;
pai)
  mkdir -p $O/pai_accgap; CHUNKS_FROM=$O/pai_accgap/JTN-L18-s0/chunks SERVE="$PAISERVE" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=13 \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_accgap/$T $T > $W/pai.submit.txt || die "pai submit"
  pp=$(awk '$2 ~ /^pai-/ {print $1}' $W/pai.submit.txt | paste -sd,); [[ -n $pp ]] || die "no PAI job"
  awk '$2 ~ /^pai-/' $W/pai.submit.txt >> $W/jobs.txt
  s=$(PRIO=13 sub ntvx-sum --after $(csv $(cat $W/reads.txt)),$pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') pai $T: PAI $pp summary $s" | tee -a $W/stages.txt ;;
sum)
  mkdir -p $R; cp $W/cmd_diff.txt experiments/defect_fix/results/jt/w_JTX00S.json $R/ || die "copy"
  $AP experiments/defect_fix/scripts/jt.py arena --tags $T@stop3 $T@off $T0@stop3 $T0@off --out $W/report > /dev/null && cp $W/report/arena.md $R/arena.md || die "arena"
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $O/pai_accgap/$T --jl "$T+ACC+GAP" --j2 $O/pai_accgap/$T0 --j2l "$T0+ACC+GAP" \
    --ref $DATA_DIR/runs/route_follow/acc_gap/read/s2 --rl RSG+ACC+GAP --out $R/pai281.md || die "pai281"
  $TR $E/scripts/ntvx.py --tag $T --ref $T0 --res $R || die "ntvx.py"
  date '+%F %T' > $W/DONE; ls -l $R ;;
*) sed -n 2,8p $0; exit 2 ;;
esac
