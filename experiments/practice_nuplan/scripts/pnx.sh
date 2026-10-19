#!/usr/bin/env bash
# PNX (plans/2026-10-19-pn-prereg.md supplement 3): two full-length arms on on1_JTX without the new standing S rows. Box, repo root.
#   NTVXS-L18-s0 = NTV0-L18-s0's command, pool on1_JT -> on1_JTX, draw JT00S -> JTX00SNS (exists: lat_diag.py wns)
#   JTXS-L18-s1  = JTN-L18-s1's command (jt.sh's DRY print, --seed 1), pool on1_JT -> on1_JTX, draw JT00 -> JTX00NS (built here, same rule)
# Calls jt.sh, rm_chain.sh, pai_box.sh unchanged; every job through the pool (priority 11), --after chained; ids in $W/jobs.txt.
#   pnx.sh check     DRY diffs against the parents (only --tag / --on / --on-wfile may differ) -> $W/cmd_diff_<arm>.txt
#   pnx.sh submit    wns -> two 50-step smokes -> smoke gate -> two trainings -> per arm held 460 @18 (6) + native arena held (18) +
#                    PAI stage (submits PAI 281 ACC + GAP, 8 chunks) -> sumsub (after both PAI stages: submits the summary after every read)
#   job bodies: train <arm> [smoke] | pai <arm> | sumsub | sum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; J=$O/jt; W=$DATA_DIR/runs/practice_nuplan/pnx; E=experiments/practice_nuplan; ME=$E/scripts/pnx.sh; PY=$E/scripts/pnx.py
JT=experiments/defect_fix/scripts/jt.sh; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
LAT=$O/lat/L18NR1v2; AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
ARMS="NTVXS-L18-s0 JTXS-L18-s1"; declare -A PAR=([NTVXS-L18-s0]=NTV0-L18-s0 [JTXS-L18-s1]=JTN-L18-s1) SM=([NTVXS-L18-s0]=NTVXSSM-L18-s0 [JTXS-L18-s1]=JTXSSM-L18-s1)
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $W/stages.txt >&2; echo "$*" > $W/ERROR; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner practice-nuplan --name "$name" --tries ${TRIES:-1} --priority 11 \
     --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
# cmd <tag>: the training command of a tag (parents and arms), jt.sh's own DRY print
cmd() { case $1 in
  NTV0-L18-s0)     DRY=1 ONK=32 ARCROWS=brake bash $JT train $1 on1_JT  $LAT JT00S    9500 0 | tail -1 ;;
  NTVXS*-L18-s0)   DRY=1 ONK=32 ARCROWS=brake bash $JT train $1 on1_JTX $LAT JTX00SNS 9500 0 | tail -1 ;;
  JTN-L18-s1)      DRY=1 ARCROWS=brake bash $JT train $1 on1_JT  $LAT JT00    9500 0 | tail -1 | sed 's/ --seed 0$/ --seed 1/' ;;
  JTXS*-L18-s1)    DRY=1 ARCROWS=brake bash $JT train $1 on1_JTX $LAT JTX00NS 9500 0 | tail -1 | sed 's/ --seed 0$/ --seed 1/' ;;
  *) die "no command for $1" ;; esac; }
case ${1:-} in
check)
  for A in $ARMS; do P=${PAR[$A]}; f=$W/cmd_diff_$A.txt
    diff <(cmd $P | tr ' ' '\n') <(cmd $A | tr ' ' '\n') > $f; echo "== $P -> $A"; cat $f
    [[ $(grep -c '^<' $f) -eq 3 && $(grep -c '^>' $f) -eq 3 ]] || die "$A: command diff beyond three tokens"
    grep '^>' $f | grep -vqE "^> ($A|on1_JTX|runs/defect_fix/w/JTX00S?NS\.npz)$" && die "$A: unexpected diff tokens"; done
  echo "diff ok: --tag, --on, --on-wfile only" ;;
submit)
  for A in $ARMS ${SM[@]}; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$A/ckpt-final.pt ]] || die "$A exists"; done
  [[ -e $O/w/JTX00SNS.npz && ! -e $O/w/JTX00NS.npz ]] || die "draw files: JTX00SNS must exist, JTX00NS must not"
  rm -f $W/ERROR $W/DONE; bash $ME check || exit 1
  w=$(sub pnx-wns --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash -c "$TR $PY wns && bash $ME check") || exit 1
  s=(); for A in $ARMS; do s+=($(sub pnx-smoke-$A --after $w --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 1 -- bash $ME train $A smoke)) || exit 1; done
  g=$(sub pnx-smoke-gate --after $(csv ${s[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $TR $PY smoke) || exit 1
  ids=(); pj=()
  for A in $ARMS; do
    t=$(sub pnx-train-$A --after $g --train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 4 -- bash $ME train $A) || exit 1
    for i in 0 1 2 3 4 5; do ids+=($(TRIES=2 sub pnx-held-$A-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $A $i 6)) || exit 1; done
    nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
    for c in $(seq 0 $((nch - 1))); do ids+=($(TRIES=2 sub pnx-ar-$A-c$c --after $t "${RD[@]}" -- bash $JT aread $A off $c)) || exit 1; done
    pj+=($(sub pnx-pai-$A --after $t --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME pai $A)) || exit 1
  done
  echo "${ids[*]}" > $W/reads.txt
  ss=$(sub pnx-sumsub --after $(csv ${pj[@]}) --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME sumsub) || exit 1
  echo "$(date '+%F %T') submit: wns $w smokes ${s[*]} gate $g reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}) PAI stages ${pj[*]} sumsub $ss" | tee -a $W/stages.txt ;;
train)   # train <arm> [smoke]: the arm's command (smoke: the smoke tag, --steps 50, everything else identical)
  A=$2; bash $ME check > /dev/null || exit 1; C=$(cmd $A)
  [[ $C == *"--tag $A "* && $C == *"--on on1_JTX "* && $C == *"--steps 9500 "* ]] || die "bad command: $C"
  [[ ${3:-} == smoke ]] && C=$(cmd ${SM[$A]} | sed 's/--steps 9500 /--steps 50 /')
  echo "$(date '+%F %T') train ${3:-full} $A: $C" | tee -a $W/stages.txt
  exec $C ;;
pai)
  A=$2; CHUNKS_FROM=$O/pai_accgap/JTN-L18-s0/chunks SERVE="$PAISERVE" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 PRIO=11 \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/pai_accgap/$A $A > $W/pai_$A.submit.txt || die "pai submit $A"
  awk '$2 ~ /^pai-/' $W/pai_$A.submit.txt >> $W/jobs.txt
  awk '$2 ~ /^pai-/ {print $1}' $W/pai_$A.submit.txt | paste -sd, > $W/pai_$A.ids; [[ -s $W/pai_$A.ids ]] || die "no PAI job for $A" ;;
sumsub)
  pp=$(cat $W/pai_*.ids | paste -sd,); s=$(sub pnx-sum --after $(csv $(cat $W/reads.txt)),$pp --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') sumsub: PAI $pp summary $s" | tee -a $W/stages.txt ;;
sum)
  $AP experiments/defect_fix/scripts/jt.py arena --tags $(for t in $ARMS NTV0-L18-s0 JTN-L18-s1; do echo $t@off; done) --out $W/arena > /dev/null || die "arena"
  for A in $ARMS; do for R in NTV0-L18-s0 ${PAR[$A]}; do
    $V experiments/defect_fix/scripts/pai281.py --j $O/pai_accgap/$A --jl "$A+ACC+GAP" --j2 $O/pai_accgap/$R --j2l "$R+ACC+GAP" \
      --ref $DATA_DIR/runs/route_follow/acc_gap/read/s2 --rl RSG+ACC+GAP --out $W/pai_${A}_vs_$R.md || die "pai281 $A $R"; done; done
  runs=(); for t in NTVXS-L18-s0:NTVXS NTV0-L18-s0:NTV0 JTXS-L18-s1:JTXS JTN-L18-s1:JTNs1; do
    d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
    [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
  $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $W/follow --cover none --workers 16 > $W/follow.log 2>&1 || die "rf_follow"
  $TR $PY sum || die "pnx.py sum"
  date '+%F %T' > $W/DONE ;;
*) sed -n 2,9p $0; exit 2 ;;
esac
