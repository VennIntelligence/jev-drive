#!/usr/bin/env bash
# Lane FIX1 on the Tokyo box (plans/2026-10-09-fix1-prereg.md, "PAI read"): arms +b and +a+b of lib/serve_fix.py on the 40 PAI scenes, with
# the chunk lists of COL1's baseline runs (a10, b1, b2a, b2b; 4 concurrent, Harmonizer off), then the table against the baseline and
# COL1's arm v1 (= +a). Runs from a code copy, as COL1's and pai1's runs do: /data/runs/alpasim/fix1/code/{lib,scripts,COMMIT}.
#   tmux new-window -d -t jev -n fix1-pai 'bash /data/runs/alpasim/fix1/code/scripts/fix1_pai_chain.sh'
# It queues behind COL1: a card is used only after COL1's chains have ended (chain/ and chain2/ DONE | ERROR) and the card has held under
# 2 GiB for a minute; nothing of COL1 or pai1 is written. State: $F/chain/{STATUS, DONE | ERROR, log.txt}; runs in $F/runs/fx_<arm>_<list>.
set -uo pipefail
# Amendment 1: FIX1_ARMS="c1 c2 c1b c2b" FIX1_K=_c FIX1_CODE=$F/code_c FIX1_AFTER=$F/chain adds the (c) arms after the first chain; the table then has every arm.
F=/data/runs/alpasim/fix1; C=/data/runs/alpasim/col1; P1=/data/runs/alpasim/pai1; K=$F/chain${FIX1_K:-}; CODE=${FIX1_CODE:-$F/code}; mkdir -p "$K" "$F/lists" "$F/runs"; rm -f "$K/DONE" "$K/ERROR"
exec > >(tee -a "$K/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$K/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$K/ERROR"; exit 1; }
for l in a10 b1 b2a b2b; do cp -n "$C/$l.txt" "$F/lists/$l.txt" || die "list $l"; done
st "code $(cat $CODE/COMMIT); waiting for COL1's chains${FIX1_AFTER:+ and $FIX1_AFTER}"
[[ -n ${FIX1_AFTER:-} ]] && until [[ -f $FIX1_AFTER/DONE || -f $FIX1_AFTER/ERROR ]]; do sleep 30; done
until [[ -f $C/chain/DONE || -f $C/chain/ERROR ]] && [[ -f $C/chain2/DONE || -f $C/chain2/ERROR ]]; do sleep 30; done
card_free() {  # card index: wait until it has held under 2 GiB for a minute
  local g=$1 ok=0
  until (( ok >= 6 )); do
    (( $(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) < 2048 )) && ok=$((ok + 1)) || ok=0; sleep 10
  done; }
flags() { case $1 in ab) echo "-e JEV_VCONT=1.0 -e JEV_LEAD=1" ;; b) echo "-e JEV_LEAD=1" ;; a) echo "-e JEV_VCONT=1.0" ;; c1) echo "-e JEV_BASE=1" ;; c2) echo "-e JEV_BASE=2" ;;
                    c1b) echo "-e JEV_BASE=1 -e JEV_LEAD=1" ;; c2b) echo "-e JEV_BASE=2 -e JEV_LEAD=1" ;; esac; }
ARMS=${FIX1_ARMS:-ab b}
JOBS=(); for l in b2a b2b a10 b1; do for arm in $ARMS; do JOBS+=("$arm $l"); done; done
worker() {  # card, base port: takes every second job
  local g=$1 port=$2 i
  for (( i = g; i < ${#JOBS[@]}; i += 2 )); do
    read -r arm l <<< "${JOBS[$i]}"; O=$F/runs/fx_${arm}_$l
    [[ -f $O/DONE ]] && continue
    card_free "$g"; echo "$(date '+%F %T') card $g: $arm on $l"
    ( cd "$F" && GPU=$g CONC=4 PAI_BASEPORT=$port DRV_ENV="$(flags "$arm")" bash $CODE/scripts/pai_run.sh "$O" "$F/lists/$l.txt" ) > "$K/fx_${arm}_$l.out" 2>&1 \
      || { sleep 20; card_free "$g"; echo "$(date '+%F %T') card $g: $arm on $l again"
           ( cd "$F" && GPU=$g CONC=4 PAI_BASEPORT=$port DRV_ENV="$(flags "$arm")" bash $CODE/scripts/pai_run.sh "$O" "$F/lists/$l.txt" ) > "$K/fx_${arm}_$l.out2" 2>&1 || return 1; }
  done; }
st "${#JOBS[@]} runs ($ARMS on b2a, b2b, a10, b1), one stack per card"
worker 0 6700 & p0=$!; worker 1 6800 & p1=$!
while kill -0 $p0 2>/dev/null || kill -0 $p1 2>/dev/null; do
  echo "$(date '+%F %T') $(ls $F/runs/fx_*/DONE 2>/dev/null | wc -l) runs done of all chains; $(find $F/runs/fx_* -name _complete 2>/dev/null | wc -l) rollouts" > "$K/STATUS"; sleep 60
done
wait $p0; r0=$?; wait $p1; r1=$?
(( r0 == 0 && r1 == 0 )) || die "runs failed (rc $r0 $r1), see $K/*.out"
d() { local p=$1; shift; local o=(); for l in "$@"; do o+=("$p$l"); done; (IFS=,; echo "${o[*]}"); }
a_runs=$C/runs/v1,$C/runs/v1_b1,$C/runs/v1_b2a,$C/runs/v1_b2b
for r in ${a_runs//,/ }; do [[ -f $r/sim/aggregate/results-summary.json ]] || die "COL1's v1 run $r is missing"; done
X=(); for arm in b ab c1 c2 c1b c2b; do [[ -f $F/runs/fx_${arm}_b1/DONE && -f $F/runs/fx_${arm}_a10/DONE && -f $F/runs/fx_${arm}_b2a/DONE && -f $F/runs/fx_${arm}_b2b/DONE ]] && X+=(--arm "$arm=$(d $F/runs/fx_${arm}_ a10 b1 b2a b2b)"); done
python3 "$CODE/scripts/fix1_pai_report.py" --arm "base=$P1/runs/a10_c4,$C/runs/b1,$C/runs/b2a,$C/runs/b2b" --arm "a=$a_runs" "${X[@]}" --ref "$P1/ref" \
  --subset "untouched (b2a + b2b)=$F/lists/b2a.txt,$F/lists/b2b.txt" --subset "seen (a10 + b1)=$F/lists/a10.txt,$F/lists/b1.txt" --out "$F/fix1_pai.md" > "$K/report.log" 2>&1 || die "report failed, see $K/report.log"
date > "$K/DONE"; st "done: $F/fix1_pai.md"
