#!/usr/bin/env bash
# FLOW1 (experiments/flowhead/plans/2026-10-10-flow-head-prereg.md), part 1: tag guard, identity of the trainer's default path, smokes of
# both heads through the bench `poses` stage, the four full runs (two at a time), training sanity, and the navtest / navhard (G) bench runs.
# One self-advancing chain: it only submits pool jobs and waits on their DONE / ERROR.
#   scripts/tmux_run.sh flow1 bash experiments/flowhead/scripts/flow_train_chain.sh
# State: $DATA_DIR/runs/flowhead/flow1/chain/{STATUS, DONE | ERROR, log.txt, jobs.txt}; job logs under chain/pool/<job>/. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/flowhead/flow1
D=$O/chain; L=$D/pool; mkdir -p "$L" "$O/smoke"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
F=experiments/flowhead/scripts
BN=("$VPY" -m jevdrive.bench)
TAGS="FMH-F-s0 RGH-F-s0 FMH-F-s1 RGH-F-s1"
SMOKES="smoke-flow1-fm smoke-flow1-rg smoke-flow1-new smoke-flow1-old smoke-flow1-new2"
HF="--hinge-lam 30 --hinge-margin 0.5"
ALL=$(for k in $(seq 0 11); do echo -n "navtrain_full.s${k}of12 "; done)
status() { echo "$(date '+%F %T') flowhead flow1: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner flowhead --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }

# ---------------------------------------------------------------- 0. every tag of this lane must be new (checked once, before the first submit)
if [[ ! -f $D/TAGS_NEW ]]; then
  for t in $TAGS $SMOKES; do
    hits=$(ls -d "$DATA_DIR/runs/op_parity/runs/$t" "$DATA_DIR/runs/op_parity/train-$t" "$DATA_DIR"/runs/bench/*/"$t"* 2>/dev/null)
    [[ -z $hits ]] || die "tag $t is not new: $hits"
  done
  date '+%F %T' > "$D/TAGS_NEW"
fi

# ---------------------------------------------------------------- 1. identity of the default path, smokes
status "stage 1: identity of the trainer's default path (60 steps, this commit against the parent of the --thead change) and head smokes"
T60="--arm P2 --seed 0 --frames warp --host --data navtrain_full.s0of12 --split navsim/op-parity-full --steps 60 --batch 64 --warmup 10 --eval-every 30 $HF"
BASE=$(git log -1 --format=%H --grep='FLOW1: trajectory head option' -- $S/pp_train.py)
[[ -n $BASE ]] || die "commit of the --thead change not found"
W=$O/base_tree
[[ -d $W ]] || git worktree add --detach "$W" "$BASE~1" || die "worktree"
sub flow1-ident-new "$L/ident-new" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- $PY $S/pp_train.py $T60 --tag smoke-flow1-new
sub flow1-ident-old "$L/ident-old" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 --cwd "$W" -- $PY $S/pp_train.py $T60 --tag smoke-flow1-old
for k in fm rg; do
  sub flow1-smoke-$k "$L/smoke-$k" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- $PY $S/pp_train.py $T60 --thead $k --tag smoke-flow1-$k
done
waitdirs "$L/ident-new" "$L/ident-old" "$L/smoke-fm" "$L/smoke-rg"
git worktree remove --force "$W" || true
if ! $PY $F/flow1.py ident --a smoke-flow1-new --b smoke-flow1-old --out "$O/ident.json"; then
  sub flow1-ident-new2 "$L/ident-new2" --train --vram 20 --cpu 4 --ram 30 --timeout-h 1 -- $PY $S/pp_train.py $T60 --tag smoke-flow1-new2
  waitdirs "$L/ident-new2"
  $PY $F/flow1.py ident --a smoke-flow1-new --b smoke-flow1-new2 --out "$O/ident_repeat.json" \
    && die "the default path of pp_train.py is not bit-identical to its parent commit (the repeat is identical to itself)"
  status "identity: the box is not run-to-run deterministic (ident_repeat.json); the default path is unchanged by construction only"
fi
for k in fm rg; do
  sub flow1-smoke-poses-$k "$L/smoke-poses-$k" --vram 24 --cpu 8 --ram 24 --timeout-h 1 -- $PY -m jevdrive.bench.stage thead-poses smoke-flow1-$k navtest "$O/smoke/$k.npz"
done
waitdirs "$L/smoke-poses-fm" "$L/smoke-poses-rg"
$PY $F/flow1.py smoke "$O/smoke/fm.npz" "$O/smoke/rg.npz" || die "smoke poses"
$PY $F/flow1.py trained smoke-flow1-fm smoke-flow1-rg --steps 60 | tee "$O/smoke/trained.txt"   # informative: 60 steps need not reach the ADE line

# ---------------------------------------------------------------- 2. four full runs, two at a time (about one card of VRAM)
kind() { case $1 in FMH*) echo fm;; RGH*) echo rg;; RHH*) echo rh;; esac; }
for s in 0 1; do
  status "stage 2: full runs, seed $s"
  for a in FMH RGH; do
    sub flow1-t-$a-s$s "$L/t-$a-s$s" --train --vram 30 --cpu 6 --ram 40 --timeout-h 3 -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
        --data $ALL --split navsim/op-parity-full --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $HF --thead "$(kind $a)" --tag $a-F-s$s
  done
  waitdirs "$L/t-FMH-s$s" "$L/t-RGH-s$s"
done
$PY $F/flow1.py trained $TAGS | tee "$O/trained.txt" || die "training sanity"

# ---------------------------------------------------------------- 3. bench: navtest (W) and navhard (G)
status "stage 3: bench navtest / navhard"
"${BN[@]}" run --model $TAGS --bench navtest || die "bench navtest"
"${BN[@]}" run --model $(for t in $TAGS; do echo -n "$t@gimm "; done) --bench navhard || die "bench navhard"
"${BN[@]}" status --model $TAGS --bench navtest --wait || die "navtest"
"${BN[@]}" status --model $(for t in $TAGS; do echo -n "$t@gimm "; done) --bench navhard --wait || die "navhard"
status "done"; date '+%F %T' > "$D/DONE"
