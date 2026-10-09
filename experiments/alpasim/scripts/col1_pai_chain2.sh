#!/usr/bin/env bash
# COL1 PAI chain, part 2 (prereg amendment 2): after col1_pai_chain.sh is done, the diagnostic arm v1 on the remaining scene lists with
# the chunking of their baselines, then extract + replay. STATUS / DONE / ERROR in $C/chain2.
set -uo pipefail
C=/data/runs/alpasim/col1; K=$C/chain2; mkdir -p $K; rm -f $K/DONE $K/ERROR
st() { echo "$(date '+%F %T') $*" | tee $K/STATUS; }
die() { st "ERROR $*"; echo "$*" > $K/ERROR; exit 1; }
source <(sed -n '/^IMG=/p;/^xr() {/,/^}/p' $C/code/scripts/col1_pai_chain.sh)
st "waiting for chain 1"; until [[ -f $C/chain/DONE || -f $C/chain/ERROR ]]; do sleep 20; done
[[ -f $C/chain/DONE ]] || die "chain 1 failed"
run() { ( cd $C && GPU=$2 CONC=4 PAI_BASEPORT=$3 DRV_PY=col1_pai_driver.py bash code/scripts/pai_run.sh $C/runs/v1_$1 $C/$1.txt ) > $K/v1_$1.out 2>&1; }
st "v1 on b2a, b2b"; run b2a 0 6400 & pa=$!; run b2b 1 6500 & pb=$!; wait $pa; ra=$?; wait $pb; rb=$?
(( ra == 0 && rb == 0 )) || die "v1 b2 runs rc $ra $rb"
st "v1 on b1; extract + replay v1_b2a"; run b1 1 6600 & pa=$!; xr v1_b2a $C/runs/v1_b2a 0 & pb=$!; wait $pa; ra=$?; wait $pb; rb=$?
(( ra == 0 && rb == 0 )) || die "v1 b1 run / v1_b2a replay rc $ra $rb"
st "extract + replay v1_b2b, v1_b1"; xr v1_b2b $C/runs/v1_b2b 0 & pa=$!; xr v1_b1 $C/runs/v1_b1 1 & pb=$!; wait $pa; ra=$?; wait $pb; rb=$?
(( ra == 0 && rb == 0 )) || die "v1 replays rc $ra $rb"
date > $K/DONE; st "done"
