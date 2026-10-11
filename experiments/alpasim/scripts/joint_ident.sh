#!/usr/bin/env bash
# Lane JOINT: identity of experiments/alpasim/scripts/joint_train.py against the trainer each sub-recipe comes from. 60 steps, seed 0, one shard
# (navtrain_full.s2of12), the original trainer with its own flags against joint_train.py with the matching flags (and --label-bank, whose values
# are bit for bit the dense rasters'). Pass = every trained net and adapter weight equal bit for bit (joint_train.py ident).
#   scripts/tmux_run.sh joint-ident bash experiments/alpasim/scripts/joint_ident.sh
# State: $DATA_DIR/runs/alpasim/joint1/ident/{STATUS, DONE | ERROR, log.txt, pool/<job>/}; results experiments/alpasim/results/joint1/ident_<name>.json
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/alpasim/joint1/ident; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$PWD/.venv/bin/python -m jevdrive.cl"
R=experiments/alpasim/results/joint1; mkdir -p "$R"
status() { echo "$(date '+%F %T') joint ident: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; [[ -f $L/$n/DONE ]] && return; rm -f "$L/$n/ERROR"
        $CL submit --owner alpasim-joint1 --name "jid-$n" --train --priority 15 --vram 20 --cpu 4 --ram 40 --timeout-h 1 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 20; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
status "code $(git rev-parse --short HEAD)"
SH="--data navtrain_full.s2of12 --split navsim/op-parity-full --batch 128 --warmup 300 --eval-every 30 --steps 60 --seed 0"
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"
JT="$PY experiments/alpasim/scripts/joint_train.py train $SH --label-bank"
sub pp-o  $PY experiments/op_parity/scripts/pp_train.py --arm P2 --frames warp --host $SH --hinge-lam 10 --tag JID-PP-o
sub pp-j  $JT --tag JID-PP-j
sub mem-o $PY experiments/op_parity/scripts/pp_train.py --arm P2 --frames warp --host $SH --hinge-lam 10 --mem vtr_W0 --tag JID-MEM-o
sub mem-j $JT --mem vtr_W0 --tag JID-MEM-j
sub s-o   $PY experiments/body1/scripts/bd4_train.py train $SH $S --tag JID-S-o
sub s-j   $JT $S --tag JID-S-j
sub ap-o  $PY experiments/alpasim/scripts/ap2_train.py $SH --cold backwarp --hinge-lam 10 --hinge-margin 0.3 --tag JID-AP-o
sub ap-j  $JT --std alpasim --tag JID-AP-j
sub apy-o $PY experiments/alpasim/scripts/ot3_rows.py train $SH --std alpasim --ot-mass 0.1 --hinge-lam 10 --hinge-margin 0.3 --tag JID-APY-o
sub apy-j $JT --std alpasim --yr-mass 0.1 --tag JID-APY-j
sub yr-o  $PY experiments/alpasim/scripts/ot3_rows.py train $SH --std navsim --ot-mass 0.1 --hinge-lam 10 --hinge-margin 0.3 --tag JID-YR-o
sub yr-j  $JT --yr-mass 0.1 --tag JID-YR-j
status "12 jobs submitted; waiting"
waitdirs pp-o pp-j mem-o mem-j s-o s-j ap-o ap-j apy-o apy-j yr-o yr-j
fail=()
for n in PP MEM S AP APY YR; do
  $PY experiments/alpasim/scripts/joint_train.py ident --a JID-$n-o --b JID-$n-j --out $R/ident_${n,,}.json > /dev/null || fail+=($n)
done
grep -h "dev @ 60" $L/*/log.txt | sed 's/.*INFO *//' | cut -c1-260
[[ ${#fail[@]} -eq 0 ]] || die "not bit-identical: ${fail[*]} (see $R/ident_*.json)"
status "all six sub-recipes bit-identical"; date '+%F %T' > "$D/DONE"
