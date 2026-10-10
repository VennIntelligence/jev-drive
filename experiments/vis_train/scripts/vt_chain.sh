#!/usr/bin/env bash
# vis_train (VT, plans/2026-10-10-vis-train-prereg.md): one self-advancing chain per arm and seed, started in tmux jev
#   scripts/tmux_run.sh vt-<arm>-s<seed> experiments/vis_train/scripts/vt_chain.sh <arm> <seed> <steps>
# Everything it runs is a pool job, submitted at once: the training job (preflight smoke, resumable, --tries), one CPU job per snapshot that
# waits for the snapshot's checkpoint (--when-exists) and starts its navtest read (`jevdrive.bench run`, which submits its own stages and
# returns), and after training the final reads: navtest, navhard (protocol W: prereg amendment 2), for A / B the test-time masked and
# shuffled memory, for the plain-P2 arms F / F0 HUGSIM 64 (spec_plan_smooth, the SH30 reference protocol) once the last arm (VT_LAST) is done.
# State: $DATA_DIR/runs/vis_train/chain/<arm>-s<seed>/{STATUS, DONE, ERROR, log.txt, jobs.txt, pool/}. Rerunning the same command resumes.
# Speed flags of the training step: VT_SPEED (default from the throughput section of plans/state.md); sizes: VT_VRAM / VT_CPU / VT_RAM.
set -uo pipefail
cd "$(dirname "$0")/../../.."
ARM=$1 SEED=$2 STEPS=$3
TAG=VT-$ARM-s$SEED
O=$DATA_DIR/runs/vis_train R=$DATA_DIR/runs/op_parity/runs
D=$O/chain/$ARM-s$SEED L=$O/chain/$ARM-s$SEED/pool
mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
B="$VPY -m jevdrive.bench"
S=experiments/vis_train/scripts
LIST=experiments/hugsim/scripts/derot_all64.txt
status() { echo "$(date '+%F %T') vis_train $TAG: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {print i}' "$D/jobs.txt" 2>/dev/null; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running" || $2 == "inbox") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner vis_train --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && return 1; done; return 0; }

case $ARM in
  A|B) EVERY=5000 MEM=1 VRAM=${VT_VRAM:-44} CPU=${VT_CPU:-8} RAM=${VT_RAM:-64} SPEED=${VT_SPEED-} ;;
  C)   EVERY=5000 MEM=0 VRAM=${VT_VRAM:-60} CPU=${VT_CPU:-12} RAM=${VT_RAM:-96} SPEED=${VT_SPEED-} ;;
  F)   EVERY=5000 MEM=0 VRAM=${VT_VRAM:-12} CPU=${VT_CPU:-4} RAM=${VT_RAM:-48} SPEED= ;;
  A0)  EVERY=10000 MEM=0 VRAM=${VT_VRAM:-12} CPU=${VT_CPU:-4} RAM=${VT_RAM:-48} SPEED= ;;
  F0)  EVERY=10000 MEM=0 VRAM=${VT_VRAM:-12} CPU=${VT_CPU:-4} RAM=${VT_RAM:-48} SPEED= ;;
  *) echo "arm $ARM?"; exit 2 ;;
esac
(( STEPS % 5000 == 0 && STEPS >= 5000 )) || die "steps must be a multiple of 5000"

# ---------------------------------------------------------------- tag-collision guard (first start only: no job of this chain exists yet)
if [[ ! -s $D/jobs.txt ]]; then
  hit=$(ls -d "$R/$TAG" "$R/$TAG"-k[0-9]* "$DATA_DIR"/runs/bench/nav*/"$TAG"@* "$DATA_DIR"/runs/bench/nav*/"$TAG"-k[0-9]* "$DATA_DIR"/runs/bench/hugsim/"$TAG"_* \
        "$O/train-$TAG" "$DATA_DIR"/runs/bench/ol/*/plans/"$TAG"[@-]* 2>/dev/null)
  [[ -z $hit ]] || die "tag collision, nothing submitted: $hit"
fi

# ---------------------------------------------------------------- training
smoke="$PY $S/vt.py train --arm $ARM --seed $SEED --steps 6 --eval-every 3 --snap-every 3 --tag smoke-vt-$ARM-s$SEED --scratch --data navtrain_full.s0of12 $SPEED"
status "submit: train $STEPS steps, snapshot reads every $EVERY"
sub "vt-t-$ARM-s$SEED" "$L/train" --train --vram "$VRAM" --cpu "$CPU" --ram "$RAM" --priority 10 --tries 4 --preflight "$smoke" -- \
    $PY $S/vt.py train --arm "$ARM" --seed "$SEED" --steps "$STEPS" --resume $SPEED
TID=$(jid "vt-t-$ARM-s$SEED")

# ---------------------------------------------------------------- reads: one starter job per snapshot, the final ones after training
BJ=()
for k in $(seq "$EVERY" "$EVERY" $((STEPS - 1))); do
  kk=$(printf '%02d' $((k / 1000)))
  sub "vt-b-$ARM-s$SEED-k$kk" "$L/b-k$kk" --vram 0.5 --cpu 1 --ram 4 --when-exists "$R/$TAG-k$kk/ckpt-final.pt" -- \
      $B run --model "$TAG-k$kk" --bench navtest
  BJ+=("$TAG-k$kk")
done
fin="$B run --model $TAG --bench navtest navhard"
(( MEM )) && fin="$fin && $B run --model $TAG:noside $TAG:mshuf --bench navtest"
sub "vt-b-$ARM-s$SEED-final" "$L/b-final" --vram 0.5 --cpu 1 --ram 4 ${TID:+--after "$TID"} --when-exists "$R/$TAG/ckpt-final.pt" -- bash -c "$fin"
if [[ $ARM == F || $ARM == F0 ]]; then           # closed loop for the plain P2 checkpoints, once the last training job has released its cores
  sub "vt-b-$ARM-s$SEED-hugsim" "$L/b-hugsim" --vram 0.5 --cpu 1 --ram 4 ${TID:+--after "$TID"} --when-exists "$R/${VT_LAST:-VT-C-s0}/ckpt-final.pt" -- \
      $B run --model "$TAG" --bench hugsim --preset spec_plan_smooth --scenarios "$LIST"
fi

# ---------------------------------------------------------------- wait
status "queued: train job ${TID:-?}; waiting"
if ! waitdirs "$L/train"; then
  for n in $(awk '$2 ~ /^vt-b-/ {print $1}' "$D/jobs.txt"); do $CL cancel "$n" > /dev/null 2>&1; done   # their checkpoints will not appear
  die "training failed or stopped: $L/train/ERROR (snapshots that exist were read; see $R/$TAG/STOP if a stop rule fired)"
fi
status "trained; final reads"
waitdirs "$L/b-final" || die "final read starter failed: $L/b-final/ERROR"
for k in "${BJ[@]}"; do until [[ -f $L/b-${k##*-}/DONE || -f $L/b-${k##*-}/ERROR ]]; do sleep 30; done; done
$B status --model "${BJ[@]}" "$TAG" --bench navtest --wait || die "navtest reads"
$B status --model "$TAG" --bench navhard --wait || die "navhard read"
if (( MEM )); then $B status --model "$TAG:noside" "$TAG:mshuf" --bench navtest --wait || die "masked / shuffled reads"; fi
status "done (HUGSIM of F / F0 runs on its own: $L/b-hugsim)"; date > "$D/DONE"
