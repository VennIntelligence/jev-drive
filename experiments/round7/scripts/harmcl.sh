#!/usr/bin/env bash
# HARM-CL (plans/2026-10-21-harmonizer-closedloop.md): PAI 281 under the submission serve with HARMONIZER=1 for NTV0-L18-s0 and R6DAG-L18-s1,
# paired per scene with their stored OFF reads (same chunk table). Box, repo root; every job through the pool at PRIO 21 (behind round 6's 22).
#   harmcl.sh submit    both reads (pai_box.sh, VRAM 30 as harm.sh ON) -> pai_fix.sh tax each -> report (r2.py r281 + harmcl.py) -> $O/DONE
#   harmcl.sh check     the first started chunk's renderer log: harmonizer enabled and the checkpoint loaded (exit 1 if not)
#   harmcl.sh wait      one run_in_background waiter: $O/DONE or a failed job of this lane
#   harmcl.sh report    (job body)
#   paused 10-21 18:20 (plan amendment 1): R6DAG-s1 c003-c007 cancelled by the main session; partial path:
#   harmcl.sh ptax      CPU-only tax of R6DAG-s1's finished chunks (c000-c002) -> facts R6DAG-L18-s1-281agH-p3
#   harmcl.sh waitp     one waiter: both the NTV0 tax and the partial tax finished (done / failed / cancelled)
#   harmcl.sh resume    (when the main session says so) R6DAG-s1's unfinished chunks (SUBMITTED of a cancelled job -> .cancelled), the full tax,
#                       then report -> $O/DONE; wait with `harmcl.sh wait`
#   harmcl.sh partial   (direct, seconds) NTV0 ON - OFF on all its scenes, R6DAG and the DiD on c000-c002 -> results/harmonizer_closedloop_p3.*
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round7/harmcl; ME=experiments/round7/scripts/harmcl.sh; V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
F_=$DATA_DIR/runs/pai_fix/facts; RES=experiments/round7/results; PRIO=${PRIO:-21}; TAGS="NTV0-L18-s0 R6DAG-L18-s1"
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 HARMONIZER=1"; WIZ=controller.gains.rel_front_steering_angle_weight=10
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
mkdir -p $O/pool
case ${1:-} in
submit)
  for f in $F_/NTV0-L18-s0-281ag.csv $F_/R6DAG-L18-s1-281ag.csv; do [[ -s $f ]] || die "no stored $f"; done
  h=$(git rev-parse --short HEAD); x=()
  for T in $TAGS; do R=$O/$T-agH; [[ ! -e $R ]] || die "$R exists"
    CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO VRAM=30 RAM=40 \
      bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.submit.txt || die "pai submit $T"
    ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "$T: not 8 chunks"
    awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
    t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281agH $R | tail -1) && [[ -n $t ]] || die "tax $T"
    echo "$t pai_fix-tax-$T-281agH" >> $O/jobs.txt; x+=($t)
    echo "$(date '+%F %T') submit $T (code $h): chunks $(echo $ids) tax $t" | tee -a $O/stages.txt; done
  r=$($V -m jevdrive.cl submit --owner round7-harmcl --name harmcl-report --tries 1 --priority $PRIO --no-check --after $(IFS=,; echo "${x[*]}") \
      --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 --log-dir $O/pool/report -- bash $ME report < /dev/null | awk '{print $NF}') && [[ -n $r ]] || die "submit report"
  echo "$r harmcl-report" >> $O/jobs.txt; echo "$(date '+%F %T') report $r" | tee -a $O/stages.txt ;;
check)
  g=$(ls -t $O/*-agH/runs/*/sim/native-logs/renderer-0.log 2>/dev/null | tail -1); [[ -n $g ]] || die "no renderer log yet"
  echo "$g"; grep -q '"enable_harmonizer": true' $g && grep -q 'DifixModelFactory: checkpoint ready at .*harmonizer_nontemporal.pt' $g \
    && { grep -E '"enable_harmonizer"|checkpoint ready' $g | cut -c1-200; echo "harmonizer loaded" | tee -a $O/stages.txt; } \
    || { grep -i harmon $g | cut -c1-200; die "harmonizer NOT confirmed in $g"; } ;;
report)
  off=(NTV0-off=$F_/NTV0-L18-s0-281ag.csv R6DAG-off=$F_/R6DAG-L18-s1-281ag.csv); on=(NTV0-on=$F_/NTV0-L18-s0-281agH.csv R6DAG-on=$F_/R6DAG-L18-s1-281agH.csv)
  $TR experiments/round2/scripts/r2.py r281 --read "${off[@]}" "${on[@]}" --pairs NTV0-on:NTV0-off R6DAG-on:R6DAG-off R6DAG-on:NTV0-on R6DAG-off:NTV0-off \
      --out $RES/pai281_harm.md --title "PAI 281, ACC + GAP, Harmonizer ON vs stored OFF: NTV0-L18-s0 and R6DAG-L18-s1" || die r281
  $TR experiments/round7/scripts/harmcl.py --off NTV0=${off[0]#*=} R6DAG=${off[1]#*=} --on NTV0=${on[0]#*=} R6DAG=${on[1]#*=} \
      --out $RES/harmonizer_closedloop.json > $RES/harmonizer_closedloop.txt || die harmcl
  date '+%F %T' > $O/DONE ;;
ptax)
  R=$O/R6DAG-L18-s1-agH; ids=$(awk '$2 ~ /^pai-r6dag-l18-s1-c00[012]$/ {print $1}' $O/jobs.txt | paste -sd,)
  t=$(PRIO=$PRIO AFTER=$ids bash experiments/pai_fix/scripts/pai_fix.sh tax R6DAG-L18-s1-281agH-p3 $R | tail -1) && [[ -n $t ]] || die "ptax"
  echo "$t pai_fix-tax-R6DAG-L18-s1-281agH-p3" >> $O/jobs.txt; echo "$(date '+%F %T') partial tax $t after $ids" | tee -a $O/stages.txt ;;
resume)
  T=R6DAG-L18-s1; R=$O/$T-agH; for r in $R/runs/*; do [[ -f $r/DONE || ! -f $r/SUBMITTED ]] || mv $r/SUBMITTED $r/SUBMITTED.cancelled; done
  sed -i '/ harmcl-report$/d; / pai_fix-tax-R6DAG-L18-s1-281agH$/d; / pai-r6dag-l18-s1-c00[3-7]$/d' $O/jobs.txt   # the cancelled ids: not failures
  CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="$SERVE" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO VRAM=30 RAM=40 \
    bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R.resume.txt || die "pai resume $T"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.resume.txt); [[ -n $ids ]] || die "$T: nothing resubmitted"; awk '$2 ~ /^pai-/ {print $1, $2}' $R.resume.txt >> $O/jobs.txt
  t=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $T-281agH $R | tail -1) && [[ -n $t ]] || die "tax $T"
  echo "$t pai_fix-tax-$T-281agH" >> $O/jobs.txt
  r=$($V -m jevdrive.cl submit --owner round7-harmcl --name harmcl-report --tries 1 --priority $PRIO --no-check --after $t \
      --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 --log-dir $O/pool/report -- bash $ME report < /dev/null | awk '{print $NF}') && [[ -n $r ]] || die "submit report"
  echo "$r harmcl-report" >> $O/jobs.txt; echo "$(date '+%F %T') resume $T: chunks $(echo $ids) tax $t report $r" | tee -a $O/stages.txt ;;
waitp)
  until $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
jobs = P.snapshot()[0]["jobs"]
ids = [l.split()[0] for l in open(sys.argv[1]) if l.split() and l.split()[1] in ("pai_fix-tax-NTV0-L18-s0-281agH", "pai_fix-tax-R6DAG-L18-s1-281agH-p3")]
st = [jobs.get(i, {}).get("state") for i in ids]
print(dict(zip(ids, st)), file=sys.stderr)
sys.exit(0 if len(ids) == 2 and all(s in ("done", "failed", "cancelled") for s in st) else 1)
PY
  do sleep 300; done; tail -2 $O/stages.txt ;;
partial)
  F0=$F_/NTV0-L18-s0-281ag.csv; F1=$F_/R6DAG-L18-s1-281ag.csv; N1=$F_/NTV0-L18-s0-281agH.csv; R1=$F_/R6DAG-L18-s1-281agH-p3.csv
  $TR experiments/round2/scripts/r2.py r281 --read NTV0-off=$F0 NTV0-on=$N1 --pairs NTV0-on:NTV0-off --out $RES/pai281_harm_ntv0.md \
      --title "PAI 281, ACC + GAP, Harmonizer ON vs stored OFF: NTV0-L18-s0, all finished scenes (partial read, plan amendment 1)" || die r281-ntv0
  $TR experiments/round2/scripts/r2.py r281 --read NTV0-off=$F0 R6DAG-off=$F1 NTV0-on=$N1 R6DAG-on=$R1 \
      --pairs NTV0-on:NTV0-off R6DAG-on:R6DAG-off R6DAG-on:NTV0-on R6DAG-off:NTV0-off --out $RES/pai281_harm_p3.md \
      --title "PAI 281, ACC + GAP, Harmonizer ON vs stored OFF on chunks c000-c002 (partial read, plan amendment 1)" || die r281-p3
  $TR experiments/round7/scripts/harmcl.py --off NTV0=$F0 --on NTV0=$N1 --out $RES/harmonizer_closedloop_ntv0.json > $RES/harmonizer_closedloop_ntv0.txt || die ntv0
  $TR experiments/round7/scripts/harmcl.py --off NTV0=$F0 R6DAG=$F1 --on NTV0=$N1 R6DAG=$R1 --out $RES/harmonizer_closedloop_p3.json \
      > $RES/harmonizer_closedloop_p3.txt || die p3
  cat $RES/harmonizer_closedloop_ntv0.txt $RES/harmonizer_closedloop_p3.txt ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
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
  [[ -f $O/DONE ]] && echo "DONE $(cat $O/DONE)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,7p "$0"; exit 2 ;;
esac
