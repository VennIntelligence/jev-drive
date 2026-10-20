#!/usr/bin/env bash
# PAI-FIX: a small targeted PAI closed-loop exam (stop overshoot + route turns, small launch / lead guard). Box, repo root; every GPU job
# through the pool (pai_box.sh, unchanged); CPU jobs through the pool too.
#   pai_fix.sh tax <name> <read root>...     rf_tax extract of stored / new reads -> $O/tax/<name>.pkl, then pai_fix.py facts -> $O/facts/<name>.csv
#   pai_fix.sh wl <read root>                 map wait lines of that read's scenes -> $O/tax/wl_<basename>.pkl (descriptive only)
#   pai_fix.sh read <tag> <serve> [subsets]   one exam read of a checkpoint: serve = ag (ACC + GAP, the PAI submission serve) | vl (VC + LEAD,
#                                             native); subsets = chunk-name prefixes (default "v t": v = val160 target + guard, t = the 281-derived
#                                             column); chunks = experiments/pai_fix/scenes/chunks (fixed lists, the simulator repeats a scene only
#                                             within an identical list); after the chunks: tax + facts (CPU) -> $O/facts/<tag>-<serve>.csv
#   pai_fix.sh smoke <tag> <serve>            the 1-scene smoke chunk (s000) of the same read, then nothing else
#   pai_fix.sh job-tax | job-wl               job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/pai_fix; ME=experiments/pai_fix/scripts/pai_fix.sh; PRIO=${PRIO:-12}; SC=experiments/pai_fix/scenes
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RF=experiments/route_follow/scripts
WIZ="controller.gains.rel_front_steering_angle_weight=10"
declare -A SRV=([ag]="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7" [vl]="JEV_VCONT=1.0 JEV_LEAD=1")
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner pai-fix --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
mkdir -p $O/pool $O/tax $O/facts
case ${1:-} in
tax)
  n=$2; shift 2; sub fx-tax-$n ${AFTER:+--after $AFTER} --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME job-tax $n "$@" ;;
job-tax)
  n=$2; shift 2; runs=(); for r in "$@"; do runs+=($(ls -d $r/runs/*/sim)); done
  $AP $RF/rf_tax.py extract --runs "${runs[@]}" --out $O/tax/$n.pkl --workers 16 > $O/tax/$n.log 2>&1 || die "extract $n"
  .venv/bin/python experiments/pai_fix/scripts/pai_fix.py facts --pkl $O/tax/$n.pkl --runs "$@" --out $O/facts/$n.csv >> $O/tax/$n.log 2>&1 || die "facts $n" ;;
wl)
  sub fx-wl --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME job-wl $2 ;;
job-wl)
  $AP $RF/rf_tax.py waitlines --runs $(ls -d $2/runs/*/sim) --out $O/tax/wl_$(basename $2).pkl --workers 16 > $O/tax/wl_$(basename $2).log 2>&1 || die "waitlines" ;;
read|smoke)
  T=$2 s=$3; [[ -n ${SRV[$s]:-} ]] || die "serve $s: ag | vl"; R=$O/$([[ $1 == smoke ]] && echo smoke || echo reads)/$T-$s; mkdir -p $R/chunks
  if [[ $1 == smoke ]]; then cp -n $SC/chunks/s000.txt $R/chunks/
  else for p in ${4:-v t}; do cp -n $SC/chunks/$p[0-9][0-9][0-9].txt $R/chunks/; done; fi
  git rev-parse --short HEAD > $R/code.txt
  SERVE="${SRV[$s]}" WIZ="$WIZ" CONC=4 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $T > $R/submit.txt || die "pai_box submit"
  cat $R/submit.txt; ids=$(awk '{print $1}' $R/jobs.txt | paste -sd,)
  [[ $1 == smoke ]] || AFTER=$ids bash $ME tax $T-$s $R ;;
*) sed -n 2,12p $0; exit 2 ;;
esac
