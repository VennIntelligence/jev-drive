#!/usr/bin/env bash
# LINEAGE-PAI281: PAI 281 reads of the model lineage P0 (shipped Cinque, no adapter) -> RB0-F-s0 -> NTV0-L18-s0 under the serve steps
# NAT (no serve switch: serve_fix off, the plan through the driver default as every read), VL (JEV_VCONT=1.0 JEV_LEAD=1), ACC (+ JEV_ACC=1),
# AG (+ ACC + JEV_GAP=0.7,2.7), LG (VL + GAP, no ACC), AG10 (+ ACC + JEV_GAP=1.0,4.0). Box, repo root. Reads go through pai_box.sh unchanged
# with the stored reads' chunk lists (NTV0-L18-s0 + ACC + GAP's), g10, CONC 4 and ALPASIM_FAST=1 (as every stored ACC + GAP read);
# rf_tax extracts (corridor sub-split) are CPU pool jobs, one per read; one summary job --after all.
#   lineage.sh first     the staged first arm (P0-VL, 8 chunks)
#   lineage.sh submit    every other arm, the extracts (new reads --after their chunks, stored reads at once), the summary --after all
#   lineage.sh ex <label> <root> | sum      job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
D=$DATA_DIR/runs; O=$D/defect_fix/lineage; ME=experiments/defect_fix/scripts/lineage.sh; PY=experiments/defect_fix/scripts/lineage_pai281.py
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RF=experiments/route_follow/scripts; PRIO=${PRIO:-13}
CF=$D/defect_fix/pai_accgap/NTV0-L18-s0/chunks; WIZ=controller.gains.rel_front_steering_angle_weight=10; VL="JEV_VCONT=1.0 JEV_LEAD=1"
FIRST=P0-VL; REST="P0-NAT P0-ACC P0-AG RB0-VL RB0-ACC RB0-AG NTV0-NAT NTV0-ACC NTV0-LG NTV0-AG10"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
serve() { case ${1#*-} in NAT) echo "" ;; VL) echo "$VL" ;; ACC) echo "$VL JEV_ACC=1" ;; AG) echo "$VL JEV_ACC=1 JEV_GAP=0.7,2.7" ;;
          LG) echo "$VL JEV_GAP=0.7,2.7" ;; AG10) echo "$VL JEV_ACC=1 JEV_GAP=1.0,4.0" ;; *) die "serve $1" ;; esac; }
tag() { case ${1%%-*} in P0) echo P0 ;; RB0) echo RB0-F-s0 ;; NTV0) echo NTV0-L18-s0 ;; *) die "tag $1" ;; esac; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
read_arm() {  # arm -> pai_box.sh submit (resumable); prints the arm's chunk job ids
  mkdir -p $O/read; CHUNKS_FROM=$CF SERVE="$(serve $1)" WIZ=$WIZ CONC=4 ALPASIM_FAST=1 PRIO=$PRIO \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/read/$1 $(tag $1) > $O/read/$1.submit.txt || die "pai submit $1"
  awk '$2 ~ /^pai-/ {print $1}' $O/read/$1.submit.txt | paste -sd,; }
case ${1:-} in
first)
  mkdir -p $O/pool; git rev-parse --short HEAD > $O/code.txt
  ids=$(read_arm $FIRST); [[ -n $ids ]] || die "no $FIRST job"; echo "$(date '+%F %T') first $FIRST: $ids" | tee -a $O/stages.txt ;;
submit)
  all=()
  for a in $REST; do ids=$(read_arm $a); [[ -n $ids ]] || die "no $a job"; all+=($ids)
    all+=($(sub lin-ex-$a --after $ids --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME ex $a $O/read/$a)); done
  f=$(awk '{print $1}' $O/read/$FIRST/jobs.txt | paste -sd,); [[ -n $f ]] || die "no $FIRST jobs.txt"
  all+=($(sub lin-ex-$FIRST --after $f --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME ex $FIRST $O/read/$FIRST))
  while read -r k r; do all+=($(sub lin-ex-$k --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME ex $k $r)); done < <(.venv/bin/python $PY stored)
  s=$(IFS=,; sub lin-sum --after "${all[*]}" --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') submit: ${#all[@]} jobs, summary $s" | tee -a $O/stages.txt ;;
ex)
  mkdir -p $O/tax; $AP $RF/rf_tax.py extract --runs $(ls -d $3/runs/*/sim) --out $O/tax/$2.tmp --workers 16 > $O/tax/$2.log 2>&1 && mv $O/tax/$2.tmp $O/tax/$2.pkl \
    || die "extract $2" ;;
sum)
  .venv/bin/python $PY table --tax $O/tax --out $O/lineage_pai281.md --json $O/lineage_pai281.json || die "table"; date '+%F %T' > $O/DONE ;;
*) sed -n 2,10p $0; exit 2 ;;
esac
