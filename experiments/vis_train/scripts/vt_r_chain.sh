#!/usr/bin/env bash
# vis_train arm R and the registered continuation (plans/2026-10-10-vis-train-prereg.md amendment 7), one self-advancing chain in tmux jev:
#   scripts/tmux_run.sh vt-R experiments/vis_train/scripts/vt_r_chain.sh all      # everything, gated in the pool; rerunning resumes
#   scripts/tmux_run.sh vt-R-smoke experiments/vis_train/scripts/vt_r_chain.sh smoke   # the staged launch: bank vtr_0, 300 steps of R-0 and of a continuation
# `all` submits at once:
#   vt-rgate            scripts/vt_r.py gate: waits for the navtest reads of the first-wave chains, writes $D/gate.json and the sentinel files
#                       GATE_PASS | GATE_FAIL and FALLBACK_A | FALLBACK_B, cancels the jobs of the side that does not run
#   vtr-bank-*          token banks runs/op_parity/mem/vtr_{0,A-s0,A-s1,B-s0,B-s1}/ (navtrain 12 shards, navtest, navhard)   [GATE_PASS]
#   vtr-t-<arm>-s<seed> pp_train.py, the SH30 recipe from the shipped weights with --mem vtr_*: VTR-{0,A,B}-s{0,1}            [GATE_PASS, after its bank]
#   vtr-b-<arm>-s<seed> starts the reads: navtest, navhard, navtest :noside / :mshuf (`jevdrive.bench run`)                    [after its training]
#   vt-t-<X>2-s<seed>   the continuation of arm X = A | B: vt.py train --init VT-X-s<seed> --tag VT-X2-s<seed>, 30 000 steps,  [FALLBACK_X]
#   vt-b-<X>2-s<seed>-* a navtest read per snapshot, the final reads as in vt_chain.sh
# State: $DATA_DIR/runs/vis_train/chain/R/{STATUS, DONE, ERROR, log.txt, jobs.txt, gate.json, GATE_*, FALLBACK_*, pool/}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
MODE=${1:-all}
O=$DATA_DIR/runs/vis_train R=$DATA_DIR/runs/op_parity/runs M=$DATA_DIR/runs/op_parity/mem
D=$O/chain/R; [[ $MODE == smoke ]] && D=$O/chain/R-smoke
L=$D/pool
mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
B="$VPY -m jevdrive.bench"
S=experiments/vis_train/scripts
OWNER=vis_train-R
FULL=$(for i in $(seq 0 11); do printf 'navtrain_full.s%dof12 ' "$i"; done)
status() { echo "$(date '+%F %T') vis_train R ($MODE): $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {print i}' "$D/jobs.txt" 2>/dev/null; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running" || $2 == "inbox") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner "$OWNER" --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && return 1; done; return 0; }
after() { local i; i=$(jid "$1"); [[ -n $i && ! -f $2/DONE ]] && echo "--after $i"; }   # the dependency, unless that job already finished in an earlier start

# the SH30 recipe (the configuration stored in SH30-F-s<seed>/ckpt-final.pt) + the memory bank; pp_train.py has no tag guard, so a finished tag is never rerun
train() {  # tag, seed, kind, steps, extra flags
  echo "[[ ! -e $R/$1/ckpt-final.pt ]] || { echo '$1: already finished'; exit 0; }; exec $PY experiments/op_parity/scripts/pp_train.py --arm P2 --mem $3" \
       "--seed $2 --steps $4 --batch 128 --data $FULL --split navsim/op-parity-full --frames warp --host --hinge-lam 30 --hinge-margin 0.5 --compile --tag $1 $5"
}
BANK="--vram 16 --cpu 4 --ram 24"          # one branch encoder at inference (the Stage-0 token job's booking)
TRAIN="--vram ${VTR_VRAM:-16} --cpu 4 --ram 48"   # staged launch: 12.6 GB peak, 7 it/s compiled on a shared card; a light cached-token job, outside the per-card training cap

if [[ $MODE == smoke ]]; then
  # ---------------------------------------------------------------- staged launch: one R arm for 300 steps, and 100 steps of a continuation
  T=smoke-vtr-0-s0 C=smoke-vt-A2-s0 CI=${VTR_SMOKE_INIT:-VT-A-s0-k15}
  [[ -f $L/train/DONE ]] || rm -rf "$R/$T"       # pp_train.py writes into an existing tag dir
  sub vtr-bank-0 "$O/chain/R/pool/bank-0" --vram 0.5 --cpu 2 --ram 8 --priority 6 -- $PY $S/vt_r.py bank --kind vtr_0 --src cinque
  sub vtr-smoke-t "$L/train" $TRAIN --priority 1 $(after vtr-bank-0 "$O/chain/R/pool/bank-0") -- bash -c "$(train $T 0 vtr_0 300 '--warmup 100 --eval-every 100')"
  sub vtr-smoke-check "$L/check" --vram 14 --cpu 2 --ram 48 --priority 1 --after "$(jid vtr-smoke-t)" -- $PY $S/vt_r.py check --tag $T
  pl=""; for o in "" :noside :mshuf; do pl="$pl $PY -m jevdrive.bench.stage parity-plans $T$o navtest $D/plans$o.npz &&"; done
  sub vtr-smoke-plans "$L/plans" --vram 24 --cpu 4 --ram 24 --priority 1 --after "$(jid vtr-smoke-t)" -- bash -c "$pl true"
  sub vtr-smoke-cont "$L/cont" --train --vram 32 --cpu 8 --ram 64 --priority 1 -- \
      $PY $S/vt.py train --arm A --seed 0 --steps 100 --eval-every 50 --snap-every 50 --tag $C --init "$CI" --scratch --enc-compile --compile
  status "queued: $(tr '\n' ' ' < "$D/jobs.txt")"
  waitdirs "$O/chain/R/pool/bank-0" "$L/train" "$L/check" "$L/plans" "$L/cont" || die "a smoke job failed: see $L/*/ERROR"
  status "done: $O/R/check-$T.json, $D/plans*.npz, $R/$C/evals.json"; date > "$D/DONE"; exit 0
fi
[[ $MODE == all ]] || { echo "mode $MODE?"; exit 2; }

# ---------------------------------------------------------------- tag-collision guard (first start only: no job of this chain exists yet)
RT=(); CT=(); for s in 0 1; do RT+=(VTR-0-s$s VTR-A-s$s VTR-B-s$s); CT+=(VT-A2-s$s VT-B2-s$s); done
if [[ ! -s $D/jobs.txt ]]; then
  hit=$(for t in "${RT[@]}" "${CT[@]}"; do
          ls -d "$R/$t" "$R/$t"-k[0-9]* "$DATA_DIR"/runs/bench/nav*/"$t"@* "$DATA_DIR"/runs/bench/nav*/"$t"-k[0-9]* "$DATA_DIR"/runs/bench/hugsim/"$t"_* \
                "$O/train-$t" "$DATA_DIR/runs/op_parity/train-$t" "$DATA_DIR"/runs/bench/ol/*/plans/"$t"[@-]* 2>/dev/null; done
        ls -d "$M"/vtr_[AB]-s[01] 2>/dev/null)
  [[ -z $hit ]] || die "tag collision, nothing submitted: $hit"
  [[ ! -e $M/vtr_0 ]] || grep -q '"src": "cinque"' "$M/vtr_0/bank.json" || die "$M/vtr_0 exists and is not the Cinque t0 bank"
fi

# ---------------------------------------------------------------- arm R (released by GATE_PASS)
status "submit: arm R (banks, 6 trainings, reads), the continuation of A and of B, the gate"
G="--when-exists $D/GATE_PASS"
sub vtr-bank-0 "$L/bank-0" --vram 0.5 --cpu 2 --ram 8 --priority 9 $G -- $PY $S/vt_r.py bank --kind vtr_0 --src cinque
for s in 0 1; do for x in A B; do
  sub "vtr-bank-$x-s$s" "$L/bank-$x-s$s" $BANK --priority 9 $G -- $PY $S/vt_r.py bank --kind "vtr_$x-s$s" --src "$x" --seed "$s"
done; done
for s in 0 1; do for x in 0 A B; do
  t=VTR-$x-s$s; k=vtr_$x-s$s; b=bank-$x-s$s; [[ $x == 0 ]] && k=vtr_0 b=bank-0
  sub "vtr-t-$x-s$s" "$L/t-$x-s$s" $TRAIN --priority 8 --tries 2 $G $(after "vtr-$b" "$L/$b") -- bash -c "$(train "$t" "$s" "$k" 10000 '--warmup 300 --eval-every 1000')"
  sub "vtr-b-$x-s$s" "$L/b-$x-s$s" --vram 0.5 --cpu 1 --ram 4 --priority 8 $(after "vtr-t-$x-s$s" "$L/t-$x-s$s") --when-exists "$R/$t/ckpt-final.pt" -- \
      bash -c "$B run --model $t --bench navtest navhard && $B run --model $t:noside $t:mshuf --bench navtest"
done; done

# ---------------------------------------------------------------- the continuation of A and of B (one of them is released by FALLBACK_<X>, the other cancelled by the gate)
CS=${VTR_CONT_STEPS:-30000}
for x in A B; do for s in 0 1; do
  t=VT-${x}2-s$s; l=$L/c-$x-s$s
  sub "vt-t-${x}2-s$s" "$l/train" --train --vram "${VT_VRAM:-32}" --cpu 8 --ram 64 --priority 4 --tries 4 --when-exists "$D/FALLBACK_$x" -- \
      $PY $S/vt.py train --arm "$x" --seed "$s" --steps "$CS" --tag "$t" --init "VT-$x-s$s" --resume --enc-compile --compile
  for k in $(seq 5000 5000 $((CS - 1))); do kk=$(printf '%02d' $((k / 1000)))
    sub "vt-b-${x}2-s$s-k$kk" "$l/b-k$kk" --vram 0.5 --cpu 1 --ram 4 --when-exists "$R/$t-k$kk/ckpt-final.pt" -- $B run --model "$t-k$kk" --bench navtest
  done
  sub "vt-b-${x}2-s$s-final" "$l/b-final" --vram 0.5 --cpu 1 --ram 4 $(after "vt-t-${x}2-s$s" "$l/train") --when-exists "$R/$t/ckpt-final.pt" -- \
      bash -c "$B run --model $t --bench navtest navhard && $B run --model $t:noside $t:mshuf --bench navtest"
done; done

# ---------------------------------------------------------------- the gate (last: jobs.txt is complete when it reads it)
sub vt-rgate "$L/gate" --vram 0.5 --cpu 1 --ram 6 --priority 10 --when-exists "$R/VT-A-s0/ckpt-final.pt" -- $VPY $S/vt_r.py gate --jobs "$D/jobs.txt"

# ---------------------------------------------------------------- wait
status "queued; waiting for the gate (job $(jid vt-rgate): after VT-A-s0's final checkpoint and the navtest reads of the first-wave chains)"
until [[ -f $D/GATE_PASS || -f $D/GATE_FAIL || -f $L/gate/DONE || -f $L/gate/ERROR ]]; do sleep 30; done
if [[ -f $D/GATE_PASS ]]; then
  status "gate passed: arm R training ($(cat "$D/GATE_PASS"))"
  for s in 0 1; do for x in 0 A B; do waitdirs "$L/t-$x-s$s" "$L/b-$x-s$s" || die "R-$x-s$s: training or its read starter failed ($L/t-$x-s$s, $L/b-$x-s$s)"; done; done
  status "arm R trained; reads"
  $B status --model "${RT[@]}" --bench navtest --wait || die "R navtest reads"
  $B status --model "${RT[@]}" --bench navhard --wait || die "R navhard reads"
  MM=(); for t in "${RT[@]}"; do MM+=("$t:noside" "$t:mshuf"); done
  $B status --model "${MM[@]}" --bench navtest --wait || die "R masked / shuffled reads"
  RMSG="R read"
elif [[ -f $D/GATE_FAIL ]]; then
  RMSG="R not run: the gate failed ($(cat "$D/GATE_FAIL"))"
else
  RMSG="R not released: the gate's reads did not arrive ($L/gate/log.txt)"
fi
status "$RMSG; waiting for the continuation choice"
until [[ -f $D/FALLBACK_A || -f $D/FALLBACK_B || -f $L/gate/DONE || -f $L/gate/ERROR ]]; do sleep 30; done
x=""; [[ -f $D/FALLBACK_A ]] && x=A; [[ -f $D/FALLBACK_B ]] && x=B
[[ -n $x ]] || die "$RMSG; no continuation was chosen ($L/gate/log.txt)"
status "$RMSG; continuation of arm $x training (VT-${x}2-s{0,1}, $CS steps)"
CM=(); for s in 0 1; do
  waitdirs "$L/c-$x-s$s/train" "$L/c-$x-s$s/b-final" || die "$RMSG; continuation VT-${x}2-s$s failed ($L/c-$x-s$s)"
  for k in $(seq 5000 5000 $((CS - 1))); do CM+=("VT-${x}2-s$s-k$(printf '%02d' $((k / 1000)))"); done; CM+=("VT-${x}2-s$s")
done
$B status --model "${CM[@]}" --bench navtest --wait || die "continuation navtest reads"
$B status --model "VT-${x}2-s0" "VT-${x}2-s1" --bench navhard --wait || die "continuation navhard reads"
$B status --model "VT-${x}2-s0:noside" "VT-${x}2-s0:mshuf" "VT-${x}2-s1:noside" "VT-${x}2-s1:mshuf" --bench navtest --wait || die "continuation masked / shuffled reads"
status "done: $RMSG; continuation of arm $x read"; date > "$D/DONE"
