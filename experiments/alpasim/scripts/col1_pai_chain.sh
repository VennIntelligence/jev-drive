#!/usr/bin/env bash
# COL1 PAI chain on the Tokyo box: extract + replay a finished pai_run.sh run; then, once the scene download is done, run the
# remaining scenes of pai_scenes_40.tsv as two stacks (one per card), extract and replay them. STATUS / DONE / ERROR in $C/chain.
set -uo pipefail
C=/data/runs/alpasim/col1; K=$C/chain; mkdir -p $K; rm -f $K/DONE $K/ERROR
IMG=jev-alpasim:p2h10-f-s0-6f3d05d1
st() { echo "$(date '+%F %T') $*" | tee $K/STATUS; }
die() { st "ERROR $*"; echo "$*" > $K/ERROR; exit 1; }
xr() {  # name run_dir gpu
  mkdir -p $C/x/$1/replay; chmod 777 $C/x/$1/replay
  docker run --rm -v /data:/data -e PYTHONPATH=/data/third_party/alpasim/src/utils -e HF_HUB_OFFLINE=1 alpasim-base:0.89.0 bash -c \
    "umask 0000; cd /repo && uv run python $C/code/scripts/col1_pai_extract.py --run $2 --out $C/x/$1" > $C/x/$1/extract.log 2>&1 || return 1
  docker run --rm --gpus device=$3 -v $C/code/lib/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro \
    -v $C/code/lib/pai_driver.py:/app/jev-drive/experiments/alpasim/lib/pai_driver.py:ro -v $C:/col1 -e SH30_TAG=P2H10-F-s0 $IMG \
    python /col1/code/scripts/col1_pai_replay.py --msgs /col1/x/$1/msgs --out /col1/x/$1/replay > $C/x/$1/replay.log 2>&1
}
[[ -f $C/x/b1/replay.log ]] || { st "b1: extract + replay"; xr b1 $C/runs/b1 0 || die "b1 extract / replay"; }
st "v1: diagnostic arm (amendment 1) on the a10 scenes"
awk -F'\t' 'NR > 1 && $8 == 1 {print $1}' $C/pai_scenes_40.tsv > $C/a10.txt
( cd $C && GPU=1 CONC=4 PAI_BASEPORT=6600 DRV_PY=col1_pai_driver.py bash code/scripts/pai_run.sh $C/runs/v1 $C/a10.txt ) > $K/v1.out 2>&1 || die "v1 run"
st "waiting for the scene download"; until [[ -f $C/FETCH_DONE ]]; do sleep 20; done
awk -F'\t' 'NR > 1 && $8 == 2 {print $1, $2}' $C/pai_scenes_40.tsv | while read -r s u; do
  [[ -s /data/datasets/nurec/all-usdzs/$u.usdz ]] && ! grep -q "$s" $C/b1.txt && echo "$s"; done > $C/b2.txt
n=$(wc -l < $C/b2.txt); h=$(( (n + 1) / 2 )); head -n $h $C/b2.txt > $C/b2a.txt; tail -n +$((h + 1)) $C/b2.txt > $C/b2b.txt
st "b2: $n scenes as two stacks"
( cd $C && GPU=0 CONC=4 PAI_BASEPORT=6400 bash code/scripts/pai_run.sh $C/runs/b2a $C/b2a.txt ) > $K/b2a.out 2>&1 & pa=$!
( cd $C && GPU=1 CONC=4 PAI_BASEPORT=6500 bash code/scripts/pai_run.sh $C/runs/b2b $C/b2b.txt ) > $K/b2b.out 2>&1 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
(( ra == 0 && rb == 0 )) || die "b2 runs rc $ra $rb"
st "b2: extract + replay"
xr b2a $C/runs/b2a 0 & pa=$!; xr b2b $C/runs/b2b 1 & pb=$!; wait $pa; ra=$?; wait $pb; rb=$?
(( ra == 0 && rb == 0 )) || die "b2 extract / replay rc $ra $rb"
st "v1: extract + replay"; xr v1 $C/runs/v1 0 || die "v1 extract / replay"
date > $K/DONE; st "done"
