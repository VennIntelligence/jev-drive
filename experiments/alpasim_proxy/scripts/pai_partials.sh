#!/usr/bin/env bash
# PAI-PARTIALS serve sweep (prereg experiments/alpasim_proxy/plans/2026-10-20-pai-partials-sweep-prereg.md). Box, repo root.
#   pai_partials.sh smoke          arm J on 2 scenes (pool), -> $O/smoke
#   pai_partials.sh submit [arms]  every arm's 8 PAI chunks at once (one pool job per chunk, A0's chunk lists) -> sum job after all of them
#   pai_partials.sh sum            job body: pai281 paired read-out per arm vs A0, pai_pcs_local collect -> $O/rows.csv.gz
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTV0-L18-s0; O=$DATA_DIR/runs/alpasim_proxy/pai_partials; A0=$DATA_DIR/runs/defect_fix/pai_accgap/$T; ME=experiments/alpasim_proxy/scripts/pai_partials.sh
BASE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; HL="JEV_VHIST=holdlead"; PRIO=${PRIO:-13}
declare -A ARM=(
  [R0]="$BASE"
  [V1]="$BASE $HL JEV_ACC_VSET=1.1"
  [V1b]="$BASE $HL JEV_ACC_VSET=1.2"
  [V2]="$BASE $HL JEV_ACC_REL=2.0,2.0"
  [V3]="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.5,2.0 $HL JEV_ACC_ALAT=3.0"
  [J]="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.5,2.0 $HL JEV_ACC_VSET=1.1 JEV_ACC_REL=2.0,2.0 JEV_ACC_ALAT=3.0"
)
ORDER="R0 V1 V1b V2 V3 J"
WIZ=controller.gains.rel_front_steering_angle_weight=10
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
case ${1:-} in
smoke)
  mkdir -p $O/smoke; printf '%s\n' clipgt-132ae6cb-1e58-42aa-8bf3-4381eb958e43 clipgt-339deecd-ef83-4e22-8bc9-2b6cf2a5f9b4 > $O/smoke/scenes.txt
  SERVE="${ARM[J]}" WIZ=$WIZ CONC=2 PRIO=15 bash experiments/alpasim/scripts/pai_box.sh submit $O/smoke/J $T $O/smoke/scenes.txt ;;
submit)
  mkdir -p $O/pool; ids=()
  for a in ${2:-$ORDER}; do
    CHUNKS_FROM=$A0/chunks SERVE="${ARM[$a]}" WIZ=$WIZ CONC=4 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $O/$a $T > $O/pool/submit_$a.txt || die "submit $a"
    ids+=($(awk '$2 ~ /^pai-/ {print $1}' $O/pool/submit_$a.txt))
  done
  pp=$(IFS=,; echo "${ids[*]}"); [[ -n $pp ]] || die "no PAI job"
  s=$(.venv/bin/python -m jevdrive.cl submit --owner pai-partials --name pai-partials-sum --tries 1 --priority $PRIO --no-check --log-dir $O/pool/sum \
      --after $pp --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum < /dev/null | awk '{print $NF}') && [[ -n $s ]] || die "sum submit"
  echo "$(date '+%F %T') chunks $pp sum $s" | tee -a $O/stages.txt ;;
sum)
  rm -f $O/SUM_DONE
  for a in $ORDER; do
    .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $O/$a --jl "$a" --ref $A0 --rl "NTV0+ACC+GAP" --out $O/pai281_$a.md || die "pai281 $a"
  done
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $DATA_DIR/runs/defect_fix/pai_accgap/$T-hl --jl HL --ref $A0 --rl "NTV0+ACC+GAP" --out $O/pai281_HL.md || die "pai281 HL"
  .venv/bin/python experiments/alpasim_proxy/scripts/pai_pcs_local.py collect --out $O/rows.csv.gz || die collect
  date '+%F %T' > $O/SUM_DONE ;;
*) sed -n 2,5p $0; exit 2 ;;
esac
