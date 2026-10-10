#!/usr/bin/env bash
# BODY1 note (j): descriptive PAI read. One pool job per (tag, serving, chunk) over PAI2's six 10-scene chunk files (same lists as the base runs),
# at most $CAP in flight, a submission waits while the memory margin of `cl top` is under $MARGIN GiB. Each stack is pruned when it ends.
#   scripts/tmux_run.sh body1-pai bash experiments/body1/scripts/pai_chain.sh "P2H10S-F-s0 P2H10S-F-s1 ..." "i ii"
# State: $DATA_DIR/runs/body1/pai/{STATUS, DONE | ERROR, log.txt, runs/<tag>_<serving>_<chunk>/}. A stack without results is submitted once more.
set -uo pipefail
cd "$(dirname "$0")/../../.."
TAGS=${1:?tags}; SERV=${2:-"i ii"}; CAP=${CAP:-6}; MARGIN=${MARGIN:-100}
O=$DATA_DIR/runs/body1/pai; S=experiments/alpasim/scripts; CH=experiments/alpasim/results/pai/chunks; mkdir -p "$O/runs"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
VPY=$PWD/.venv/bin/python
flags() { case $1 in i) echo "" ;; ii) echo "JEV_VCONT=1.0 JEV_LEAD=1" ;; esac; }
ok() { [[ -f $1/aggregate/results-summary.json ]]; }
ended() { ok "$1" || [[ -f $1/pool/ERROR || -f $1/pool/DONE ]]; }
margin() { $VPY -m jevdrive.cl top 2>/dev/null | sed -n 's/.*margin \([0-9]*\).*/\1/p' | head -1; }
JOBS=(); for t in $TAGS; do for v in $SERV; do for c in a10 b1 b2a b2b e1 e2; do JOBS+=("${t}_${v}_$c"); done; done; done
inflight() { local n=0 r; for r in "${JOBS[@]}"; do [[ -d $O/runs/$r/pool ]] && ! ended "$O/runs/$r" && n=$((n + 1)); done; echo $n; }
sub() {  # job name
  IFS=_ read -r t v c <<< "$1"; local R=$O/runs/$1
  rm -rf "$R"; mkdir -p "$R/pool"
  # shellcheck disable=SC2046
  $VPY -m jevdrive.cl submit --no-check --owner body1-pai --name "pai-$t-$v" --vram 24 --cpu 6 --ram 40 --timeout-h 2 --tries 1 --log-dir "$R/pool" -- \
    env CONC=4 SH30_TAG=$t $(flags "$v") bash -c "bash $S/pai_native.sh $R $CH/$c.txt; rc=\$?; python3 $S/fix1_pai_box_report.py prune $R; exit \$rc" | tail -1 >> "$O/jobs.txt"
}
for pass in 1 2; do
  echo "$(date '+%F %T') pass $pass" > "$O/STATUS"
  for r in "${JOBS[@]}"; do
    ok "$O/runs/$r" && continue
    while :; do m=$(margin); (( $(inflight) < CAP )) && (( ${m:-0} >= MARGIN )) && break; sleep 30; done
    echo "$(date '+%F %T') submit $r (margin ${m:-?})"; sub "$r"; sleep 20
  done
  while (( $(inflight) > 0 )); do echo "$(date '+%F %T') pass $pass: $(inflight) in flight, $(ls -d $O/runs/*/aggregate 2>/dev/null | wc -l)/${#JOBS[@]} done" > "$O/STATUS"; sleep 60; done
  bad=0; for r in "${JOBS[@]}"; do ok "$O/runs/$r" || bad=$((bad + 1)); done; (( bad == 0 )) && break
done
(( bad == 0 )) && { date > "$O/DONE"; echo "done" > "$O/STATUS"; } || { echo "$bad stacks missing" > "$O/ERROR"; }
