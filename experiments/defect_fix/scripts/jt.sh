#!/usr/bin/env bash
# JT, NR1 prereg supplement 9 (plans/2026-10-17-nr1-prereg.md): one joint L18 stage from RB0-F-s0 (navtrain + yaw-rate imitation, the big
# on-policy pool with S hinges + lateral 0.5v2 + the arc term, trainval standing / launch imitation rows in the same batches). Box, repo root;
# every job through the GPU pool, --after chained; ids in $O/jt/jobs.txt. Env PRIO (12; the arena / held reads take PRIO + 1).
#   jt.sh provis                 code smoke on a provisional mixed pool on1_JTP (on1_L18RB0 + on1_L18RTB1 S rows + on1_NR1TV, whose S part is
#                                exactly on1_L18NR1's rows: the amendment-1 raster L18NR1v2 serves as its prefix raster): 300 steps, every JT flag
#   jt.sh pool <roll ids>        after the NR1HV-L18-s0 / -s1 train rollouts: geo -> tok -> route features (+ strip) of on1_L18HV; then (with the
#                                route features of on1_L18YS / Y / P) the merged pool on1_JT -> draw weights JT10 / JT00 -> its lateral raster
#   jt.sh smoke <after> <arc>    1 000-step smoke JTSM-L18-s0 (JT10, arc weight <arc>)
#   jt.sh arm <tag> <wfile> <steps> <arc> [after]   training -> arena held split with leash3 and without the hold (18 chunks each) + held 460
#                                @18 (6) -> summary (results/jt/<tag>/)
#   jt.sh train <tag> <pool> <lat dir> <wfile> <steps> <arc>    job body (rm_train)
#   jt.sh aread <tag> <off|leash3> <chunk>                    job body: one arena held chunk (ARENA-HOLD's lists and shard caches, read-only)
#   jt.sh tok <name> | merge | weights | latbuild | sum <tag>  job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; J=$O/jt; ME=experiments/defect_fix/scripts/jt.sh; PY=experiments/defect_fix/scripts/jt.py
TR=$DATA_DIR/envs/op-train/bin/python; NS=$DATA_DIR/envs/navsim2/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RT=experiments/route_mem/scripts/rm_train.py; OL=experiments/onpol_long/scripts/ol_chain.sh; LT=experiments/defect_fix/scripts/df_lat.py
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
RSG="--warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train \
--lr-scale 0.1 --on-k 13 --on-mode self"                                                     # RSG-L18-s0's command minus --init / --on (= df.sh RSG)
NOISE="--ego-scale-speed 0.5 --ego-scale-max 0.2"
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_L18P:p:self on1_L18HV:hv:self on1_NR1TV:tv:both"
ROWS=L18RB0:rb0,L18RTB1:rtb1,L18YS:ys,L18Y:y,L18P:p,L18HV:hv
KS=32; KTV=38                                                                                 # S rows / trainval rows per batch (supplement 9)
mkdir -p $J/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $J/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $J/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
lat_flags() {  # lat_flags <w> <dir>: the amendment-1 pick (lat_pick_v2.env) at weight w on raster <dir>, kept on the host
  source $O/lat_pick_v2.env; local f=""
  [[ $ONC == 1 ]] && f="$f --lat-onc $1"; [[ $SOFT == 1 ]] && f="$f --lat-soft $(python3 -c "print($1 / 5)")"; [[ $SIDE == 1 ]] && f="$f --lat-side $1"
  echo "$f --lat-file $2 --lat-depth $DEPTH --lat-soft-depth $SDEPTH --lat-buf $BUF --lat-host"; }
FT=(--train --vram 40 --cpu 6 --ram 100 --env CL_HIST_KEY=rm-train)
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
case ${1:-} in
provis)
  m=$(sub jt-merge-JTP --vram 0.1 --cpu 4 --ram 64 --timeout-h 1 -- $TR experiments/route_follow/scripts/rf_pait.py merge \
      --pools on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_NR1TV:tv:both --out JTP --nan-keys fut_reg) || exit 1
  w=$(sub jt-w-JTP --after $m --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $TR $PY weights --pool on1_JTP --out JTP10) || exit 1
  t=$(sub jt-train-JTSM0 --after $w "${FT[@]}" --timeout-h 1 -- bash $ME train JTSM0-L18-s0 on1_JTP $O/lat/L18NR1v2 JTP10 300 0.0025) || exit 1
  echo "$(date '+%F %T') provis: merge $m weights $w smoke $t" | tee -a $J/stages.txt ;;
pool)
  R=${2:?roll ids}
  g=$(sub jt-geo-HV --after $R --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $OL geo NR1HV-L18-s0,NR1HV-L18-s1 L18HV) || exit 1
  k=$(sub jt-tok-HV --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME tok L18HV) || exit 1
  r=$(sub jt-route-L18HV --after $k --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- $AP experiments/route_mem/scripts/rm_build.py pool --name L18HV --workers 32) || exit 1
  sub jt-strip-HV --after $r --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $OL strip NR1HV-L18-s0,NR1HV-L18-s1 L18HV > /dev/null || exit 1
  rr=$(awk '$2 ~ /^jt-route-L18(YS|Y|P)$/ {print $1}' $J/jobs.txt | paste -sd,)
  m=$(sub jt-merge-JT --after $r${rr:+,$rr} --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- bash $ME merge) || exit 1
  w=$(sub jt-w-JT --after $m --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME weights) || exit 1
  l=$(sub jt-lat-JT --after $m --vram 0.1 --cpu 64 --ram 160 --timeout-h 2 -- bash $ME latbuild) || exit 1
  echo "$(date '+%F %T') pool: geo $g tok $k route $r merge $m weights $w lat $l" | tee -a $J/stages.txt ;;
smoke)
  A=${2:?after}; W=${3:?arc}
  t=$(sub jt-train-JTSM --after $A "${FT[@]}" --timeout-h 1 -- bash $ME train JTSM-L18-s0 on1_JT $O/lat/JT JT10 1000 $W) || exit 1
  echo "$(date '+%F %T') smoke: $t (arc $W, after $A)" | tee -a $J/stages.txt ;;
arm)
  T=${2:?tag}; WF=${3:?wfile}; N=${4:?steps}; W=${5:?arc}; A=${6:-}
  [[ ! -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "$T exists"
  t=$(sub jt-train-$T ${A:+--after $A} "${FT[@]}" --timeout-h 3 -- bash $ME train $T on1_JT $O/lat/JT $WF $N $W) || exit 1
  ids=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for V in leash3 off; do for c in $(seq 0 $((nch - 1))); do
    ids+=($(PRIO=$((${PRIO:-12} + 1)) TRIES=2 sub jt-ar-$T-$V-c$c --after $t "${RD[@]}" -- bash $ME aread $T $V $c)) || exit 1; done; done
  for i in 0 1 2 3 4 5; do
    ids+=($(PRIO=$((${PRIO:-12} + 1)) TRIES=2 sub jt-held-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  s=$(sub jt-sum-$T --after $(csv ${ids[@]}) --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME sum $T) || exit 1
  echo "$(date '+%F %T') arm $T ($WF, $N steps, arc $W): train $t reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}) summary $s" | tee -a $J/stages.txt ;;
train)
  T=$2; P=$3; LD=$4; WF=$5; N=$6; W=$7
  exec $TR $RT train --tag $T --init RB0-F-s0 --allow-init $RSG --on $P --on-k $((KS + KTV)) --on-mode mixed --on-lon-all --on-normal \
    --on-wfile runs/defect_fix/w/$WF.npz --on-hold-pct 15 $NOISE $(lat_flags 0.5 $LD) --on-arc $W --diag --warmup 500 --lr-floor 0.1 \
    --steps $N --seed 0 ;;
aread)   # aread <tag> <off|leash3> <chunk>: = arena_hold.sh one_run with SH30_TAG=<tag> (sh30hold driver; off = JEV_HOLD unset, = sh30)
  T=$2; V=$3; c=$4; D=$J/arena/$T@$V/c$c/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  k=$(python3 -c "import json; print(next(x['shard'] for x in json.load(open('$AH/meta.json'))['chunks'] if x['chunk'] == $c))") || die "no chunk $c"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D; rc=0
  if [[ $V == off ]]; then env -u JEV_HOLD SH30_TAG=$T bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $AH/lists/c$c.txt "${BASE[@]}" "$KEY.cache_location=$AC/s$k" || rc=$?
  else env JEV_HOLD=$V SH30_TAG=$T bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $AH/lists/c$c.txt "${BASE[@]}" "$KEY.cache_location=$AC/s$k" || rc=$?; fi
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  ;;
tok)     # = ol_chain.sh tok without its stop on the pool gates (they are reported, not a stop here: the rows are S rows of a mixed pool)
  N=$2
  $TR experiments/onpol_nuplan/scripts/on_rows.py tok --name $N --control smoke10e || die "on_rows tok rc $?"
  $TR experiments/onpol_long/scripts/ol_labels.py gate --name $N --out $DATA_DIR/runs/onpol_long/rows/$N || echo "pool gates of $N not passed (reported only)" ;;
merge)
  $TR experiments/route_follow/scripts/rf_pait.py merge --pools $PARTS --out JT --nan-keys fut_reg ;;
weights)
  $TR $PY weights --pool on1_JT --k-s $KS --k-tv $KTV --hold-pct 15 --tv D10 --out JT10 && \
  $TR $PY weights --pool on1_JT --k-s $KS --k-tv $KTV --hold-pct 15 --tv D10 --no-wait --out JT00 ;;
latbuild)
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 \
    $AP $LT build --pool on1_JT --rows $ROWS --out $O/lat/JT --workers 64 --dilate 6 --points ra --prefix ;;
sum)
  T=$2; R=experiments/defect_fix/results/jt/$T; mkdir -p $R
  $AP $PY arena --tags $T@leash3 $T@off --out $J/report/$T && cp $J/report/$T/arena.md $R/arena.md
  $TR $PY held --tags $T > $R/held.txt 2>&1; cat $R/held.txt ;;
*) sed -n 2,16p $0; exit 2 ;;
esac
