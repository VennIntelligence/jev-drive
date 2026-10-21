#!/usr/bin/env bash
# PAI-PCS-LOCAL-R5: score the stored PAI 281 ACC + GAP reads of R2C / R2J / R3MV (and NTV0 as the reference) on the d392 ruler. CPU-only pool jobs.
#   pai_pcs_local_r5.sh submit    box, repo root: 3 fits (paimix, seeds 1 / 2) + 20 bootstrap shards x 5 draws (PRIO 15, 4 cores) -> r5 report --after all
#   pai_pcs_local_r5.sh report    job body: pai_pcs_local.py r5 -> $T/out/r5_report.md
# drive-irt (82ddd5a) is not in the box env: its pure-python tree (+ jaxtyping, hydra, omegaconf, antlr4) sits in $T/site, put on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
T=$DATA_DIR/tmp/pcs_r5; PY=.venv/bin/python; X=experiments/alpasim_proxy/scripts/pai_pcs_local.py; PRIO=${PRIO:-15}
ENVV="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$T/site"
SUBJ="R2C-s0+AG=R2C-L18-s0-281ag.csv,R2J-s0+AG=R2J-L18-s0-281ag.csv,R3MV-s0+AG=R3MV-L18-s0-281ag.csv"
sub() { local name=$1; shift; $PY -m jevdrive.cl submit --owner pcs-r5 --name "$name" --tries 1 --priority $PRIO --cpu 4 --ram 16 --vram 0.1 --cwd "$PWD" \
  --no-check --log-dir $T/pool/$name "$@" < /dev/null | awk '{print $NF}'; }
case ${1:-} in
submit)
  ids=()
  for v in paimix paimix-s1 paimix-s2; do
    ids+=($(sub pcs-r5-fit-$v -- env $ENVV $PY $X fit --rows $T/rows.csv.gz --out $T/out --variants $v)); done
  for b in $(seq 0 5 95); do
    ids+=($(sub pcs-r5-boot-$b -- env $ENVV $PY $X fit --rows $T/rows.csv.gz --out $T/out --variants - --boot 5 --boot0 $b)); done
  printf '%s\n' "${ids[@]}" > $T/jobs.txt
  r=$(sub pcs-r5-report --after $(paste -sd, $T/jobs.txt) -- env $ENVV bash $0 report); echo "report job $r" | tee -a $T/jobs.txt ;;
report)
  $PY $X r5 --rows $T/rows.csv.gz --out $T/out --facts $DATA_DIR/runs/pai_fix/facts --subjects "$SUBJ" > $T/out/r5.log 2>&1 && date '+%F %T' > $T/DONE ;;
esac
