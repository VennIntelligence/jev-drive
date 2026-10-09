#!/usr/bin/env bash
# Lane OT3 (plans/2026-10-09-ot3-lambda10-prereg.md, decisions 212 / 213): the competition driver on the lambda-10 base. One self-advancing
# chain per stage in tmux jev (scripts/tmux_run.sh ot3-<stage> experiments/alpasim/scripts/ot3_chain.sh <stage> [args]); every GPU job goes
# through the pool, benchmark reads through jevdrive.bench. Rerunning a stage resumes (finished pool jobs are not resubmitted).
#   train-a  20-step smokes of the three trainers (peak VRAM / RAM read before the grid) -> lambda 10 / 0.3 m, 2 seeds each:
#            AP2H10-AB (AP2 input standard), OT10a05-F (NAVSIM standard + the +-0.5 m off-track rows), OT10a15-F (+ the +-1.5 m rows)
#   rows     yaw-rate rows (ot3_rows.py): 64-row zero gate + shard 2 -> probe gate on the existing checkpoints -> the other 11 shards
#   train-b  <tag> <trainer args...>: 20-step smoke then seeds 0 / 1 of one yaw-rate-row recipe (ot3_rows.py train)
#   probe    <name> <tags...>: continuation-slope probe on the held-out yaw-rate rows
#   side     <name> <ref group> <tag...>: navtest / navhard through jevdrive.bench + the AlpaSim-standard offline read (seed-0 tags)
#   guard    <tag...>: HUGSIM 64 (spec_plan_smooth) for the single best configuration
# The closed loop runs beside it: OT_LANE=ot3 scripts/ot2_loop.py <stage> ... (waits for each checkpoint file).
# State: $DATA_DIR/runs/alpasim/ot3/chain-<stage>/{STATUS, DONE, ERROR, log.txt, jobs.txt}; pool logs under .../ot3/pool/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=$1; shift
O=$DATA_DIR/runs/alpasim/ot3
D=$O/chain-$STAGE${CHAIN_SUFFIX:-}; mkdir -p "$D" "$O/results"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
S=experiments/alpasim/scripts
SP=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$O/pool
L2=$DATA_DIR/runs/alpasim/ot2/pool
A=$DATA_DIR/runs/alpasim/ap2
CK=$DATA_DIR/runs/op_parity/runs
K=12
ALL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
REST=$(for i in 0 1 3 4 5 6 7 8 9 10 11; do echo -n "navtrain_full.s${i}of$K "; done)
SPLIT=navsim/op-parity-full
H10="--hinge-lam 10 --hinge-margin 0.3"
status() { echo "$(date '+%F %T') ot3 $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        $CL queue 2>/dev/null | awk -v p="$ld/log.txt" '($2 == "queued" || $2 == "running" || $2 == "inbox") && index($0, p) {f = 1} END {exit !f}' && return
        rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner alpasim-ot3 --priority 14 --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n $ld" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
peaks() { for ld in "$@"; do id=$(awk -v p="$ld" '$3 == p {i = $1} END {print i}' "$D/jobs.txt"); echo "== $ld ($id)"; $CL show "$id" 2>/dev/null | grep -iE "peak|vram|rss|ram|cpu" | head -8
          grep -h "step 20:" "$ld/log.txt" | sed 's/.*INFO *//' | cut -c1-200; done; }
# Declarations = measured peaks plus margin. Pool history 2026-10-09: ap2 / apo training 35.7-37.1 GB VRAM, 96 GB RSS of which the mapped
# host token stores are reclaimable page cache (the pool admits on anon + shmem: 40 GB); ot_rows training 26.3 GB VRAM, 41.6 GB RSS;
# off-track prep 29-42 GB VRAM / 29 GB RAM / 19 cores. The smokes below re-read them for the lambda-10 runs.
TR="--train --vram 48 --cpu 6 --ram 45"
TF="--data $ALL --split $SPLIT --batch 128 --warmup 300 --eval-every 1000"

if [[ $STAGE == train-a ]]; then
  status "waiting for lane OT2's 12 +-1.5 m prep jobs (DONE files)"
  waitdirs $(for d in $ALL; do echo $L2/prep-a15-$d; done)
  status "20-step smokes: ap2_train lambda 10, ot_rows + ot1, ot_rows + ot2"
  sub ot3-smoke $L/smoke-ap2h10 $TR -- $PY $S/ap2_train.py $TF --cold backwarp $H10 --steps 20 --eval-every 20 --seed 0 --tag smoke-ot3-ap2h10
  sub ot3-smoke $L/smoke-ot1    $TR -- $PY $SP/ot_rows.py train $TF $H10 --ot-mass 0.1 --ot ot1 --steps 20 --eval-every 20 --seed 0 --tag smoke-ot3-ot1
  sub ot3-smoke $L/smoke-ot2    $TR -- $PY $SP/ot_rows.py train $TF $H10 --ot-mass 0.1 --ot ot2 --steps 20 --eval-every 20 --seed 0 --tag smoke-ot3-ot2
  waitdirs $L/smoke-ap2h10 $L/smoke-ot1 $L/smoke-ot2
  peaks $L/smoke-ap2h10 $L/smoke-ot1 $L/smoke-ot2
  grep -h "off-track rows\|train .* dev" $L/smoke-*/log.txt | sed 's/.*INFO *//' | cut -c1-300
  status "trainings: AP2H10-AB, OT10a05-F, OT10a15-F, seeds 0 / 1"
  for s in 0 1; do
    sub ot3-t $L/t-AP2H10-AB-s$s $TR -- $PY $S/ap2_train.py $TF --cold backwarp $H10 --steps 10000 --seed $s --tag AP2H10-AB-s$s
    sub ot3-t $L/t-OT10a05-F-s$s $TR -- $PY $SP/ot_rows.py train $TF $H10 --ot-mass 0.1 --ot ot1 --steps 10000 --seed $s --tag OT10a05-F-s$s
    sub ot3-t $L/t-OT10a15-F-s$s $TR -- $PY $SP/ot_rows.py train $TF $H10 --ot-mass 0.1 --ot ot2 --steps 10000 --seed $s --tag OT10a15-F-s$s
  done
  TAGS=$(for r in AP2H10-AB OT10a05-F OT10a15-F; do echo -n "$r-s0 $r-s1 "; done)
  waitdirs $(for t in $TAGS; do echo $L/t-$t; done)
  for t in $TAGS; do [[ -f $CK/$t/ckpt-final.pt ]] || die "no checkpoint $t"; done
  grep -h "dev @ 10000" $L/t-*/log.txt | sed 's/.*INFO//' | cut -c1-400
elif [[ $STAGE == rows ]]; then
  prep() { local d=$1; shift; sub ot3-prep $L/prep-yr1-$d${TAGX:-} --vram 44 --cpu 20 --ram 40 -- $PY $S/ot3_rows.py prep --data $d --workers 18 "$@"; }
  status "yaw-rate rows: zero gate (64 rows) and shard 2"
  TAGX=-zero prep navtrain_full.s2of12 --limit 64 --zero
  prep navtrain_full.s2of12
  waitdirs $L/prep-yr1-navtrain_full.s2of12-zero $L/prep-yr1-navtrain_full.s2of12
  status "probe gate on shard 2 (existing checkpoints)"
  sub ot3-probe $L/probe-gate --vram 32 --cpu 4 --ram 40 -- $PY $S/ot3_rows.py probe --name gate --data navtrain_full.s2of12 --split $SPLIT \
      --tags P2H10-F-s0 P2H10-F-s1 P2-F-s0 SH30-F-s0 OT30-F-s0 --gate
  waitdirs $L/probe-gate
  cat $O/results/yr_probe_gate.md
  status "yaw-rate rows: the other 11 shards"
  for d in $REST; do prep $d; done
  waitdirs $(for d in $REST; do echo $L/prep-yr1-$d; done)
elif [[ $STAGE == train-b ]]; then
  TAG=$1; shift
  status "$TAG: 20-step smoke"
  sub ot3-smoke $L/smoke-$TAG $TR -- $PY $S/ot3_rows.py train $TF "$@" --steps 20 --eval-every 20 --seed 0 --tag smoke-ot3-$TAG
  waitdirs $L/smoke-$TAG; peaks $L/smoke-$TAG
  status "$TAG: seeds 0 / 1"
  for s in 0 1; do sub ot3-t $L/t-$TAG-s$s $TR -- $PY $S/ot3_rows.py train $TF "$@" --steps 10000 --seed $s --tag $TAG-s$s; done
  waitdirs $L/t-$TAG-s0 $L/t-$TAG-s1
  for s in 0 1; do [[ -f $CK/$TAG-s$s/ckpt-final.pt ]] || die "no checkpoint $TAG-s$s"; done
  grep -h "dev @ 10000" $L/t-$TAG-s*/log.txt | sed 's/.*INFO//' | cut -c1-400
elif [[ $STAGE == probe ]]; then
  NAME=$1; shift
  status "continuation probe $NAME: $*"
  sub ot3-probe $L/probe-$NAME --vram 32 --cpu 4 --ram 70 -- $PY $S/ot3_rows.py probe --name $NAME --split $SPLIT --tags "$@"
  waitdirs $L/probe-$NAME
  cat $O/results/yr_probe_$NAME.md
elif [[ $STAGE == side ]]; then
  NAME=$1; shift
  TAGS="$*"
  G=$(for t in $TAGS; do echo -n "$t@gimm "; done)
  status "navtest / navhard through jevdrive.bench: $TAGS"
  "${B[@]}" run --model $TAGS --bench navtest --priority 6 || die "bench navtest"
  "${B[@]}" run --model $G --bench navhard --priority 6 || die "bench navhard"
  S0=$(for t in $TAGS; do [[ $t == *-s0 ]] && echo -n "${t%-s0}=$t "; done)
  sub ot3-offline $L/offline-$NAME --vram 32 --cpu 8 --ram 60 -- $PY $S/ap2_offline.py plans --name ot3-$NAME --scenes $DATA_DIR/runs/alpasim/scenes_navtest_full_part001_48.txt \
      --models P2H10=P2H10-F-s0 $S0
  waitdirs $L/offline-$NAME
  keys=$($VPY -c "import numpy as np, re, sys; print(' '.join(k for k in np.load(sys.argv[1]).files if re.search('_m[14]\$', k)))" $A/offline/ot3-$NAME/poses.npz)
  [[ -f $A/offline/ot3-$NAME/scores.csv ]] || "${B[@]}" score-poses --poses $A/offline/ot3-$NAME/poses.npz --keys $keys --tokens $A/offline/ot3-$NAME/subset.txt \
      --out $A/offline/ot3-$NAME/scores.csv --traffic non_reactive --priority 6 --owner alpasim-ot3 --wait || die "score-poses"
  $PY $S/ap2_offline.py report --name ot3-$NAME --ref P2H10 --scores $A/offline/ot3-$NAME/scores.csv --out $O/results/ap2_offline_$NAME || die "offline report"
  "${B[@]}" status --model $TAGS --bench navtest --wait || die "navtest"
  "${B[@]}" status --model $G --bench navhard --wait || die "navhard"
  for r in $(for t in $TAGS; do echo "${t%-s[01]}"; done | sort -u); do
    "${B[@]}" report --bench navtest --arms $r=$r-s0+$r-s1 --vs P2H10=P2H10-F-s0+P2H10-F-s1 SH30=SH30-F-s0+SH30-F-s1 --out $O/results/bench-$r || die "report navtest $r"
    "${B[@]}" report --bench navhard --arms $r=$r-s0@gimm+$r-s1@gimm --vs P2H10=P2H10-F-s0@gimm+P2H10-F-s1@gimm SH30=SH30-F-s0@gimm+SH30-F-s1@gimm \
        --out $O/results/bench-$r || die "report navhard $r"
  done
elif [[ $STAGE == guard ]]; then
  HP=spec_plan_smooth; LIST=experiments/hugsim/scripts/derot_all64.txt
  status "HUGSIM 64 ($HP): $*"
  "${B[@]}" run --model "$@" --bench hugsim --preset $HP --scenarios "$LIST" --priority 12 || die "bench hugsim"
  "${B[@]}" status --model "$@" --bench hugsim --preset $HP --wait || die "hugsim"
else
  die "unknown stage $STAGE"
fi
status "done"; date > "$D/DONE"
