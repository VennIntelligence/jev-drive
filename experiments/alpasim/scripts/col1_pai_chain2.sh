#!/usr/bin/env bash
# COL1 PAI chain, part 2 (prereg amendment 2): the diagnostic arm v1 on the remaining scene lists with the chunking of their
# baselines, then extraction (no replay: the arm is read from the simulator logs only). Card 1: b2b then b1, started at once; card 0:
# b2a once col1_pai_chain.sh has released it. STATUS / DONE / ERROR in $C/chain2.
set -uo pipefail
C=/data/runs/alpasim/col1; K=$C/chain2; mkdir -p $K; rm -f $K/DONE $K/ERROR
st() { echo "$(date '+%F %T') $*" | tee $K/STATUS; }
die() { st "ERROR $*"; echo "$*" > $K/ERROR; exit 1; }
run() { ( cd $C && GPU=$2 CONC=4 PAI_BASEPORT=$3 DRV_PY=col1_pai_driver.py bash code/scripts/pai_run.sh $C/runs/v1_$1 $C/$1.txt ) > $K/v1_$1.out 2>&1; }
ex() { mkdir -p $C/x/$1; docker run --rm -v /data:/data -e PYTHONPATH=/data/third_party/alpasim/src/utils -e HF_HUB_OFFLINE=1 alpasim-base:0.89.0 bash -c \
       "umask 0000; cd /repo && uv run python $C/code/scripts/col1_pai_extract.py --run $C/runs/$1 --out $C/x/$1" > $C/x/$1/extract.log 2>&1; }
st "v1 on b2b, then b1 (card 1); b2a after chain 1 (card 0)"
( run b2b 1 6500 && run b1 1 6600 ) & p1=$!
( until [[ -f $C/chain/DONE || -f $C/chain/ERROR ]]; do sleep 20; done; run b2a 0 6400 ) & p0=$!
wait $p1; r1=$?; st "card 1 done rc $r1; waiting for b2a"; wait $p0; r0=$?
(( r0 == 0 && r1 == 0 )) || die "v1 runs rc $r0 $r1"
st "extract"; for n in v1_b2a v1_b2b v1_b1; do ex $n || die "extract $n"; done
date > $K/DONE; st "done"
