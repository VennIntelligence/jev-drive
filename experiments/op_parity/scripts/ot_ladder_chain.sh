#!/usr/bin/env bash
# Lane OT2 piece A (decision 209): the off-track recovery probe on the adaptation ladder, one pool job.
#   scripts/tmux_run.sh ot2-a experiments/op_parity/scripts/ot_ladder_chain.sh <name> <ot prefix> <tags...> [-- <A:B pairs...>]
# State: $DATA_DIR/runs/op_parity/ot_rows/chain-ladder-<name>/{STATUS, DONE, ERROR, log.txt}; output .../ot_rows/ladder_<name>.{md,json}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
NAME=$1 OT=$2; shift 2
TAGS=() PAIRS=(); while (( $# )); do [[ $1 == -- ]] && { shift; PAIRS=("$@"); break; }; TAGS+=("$1"); shift; done
O=$DATA_DIR/runs/op_parity/ot_rows
D=$O/chain-ladder-$NAME; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
status() { echo "$(date '+%F %T') ot2 ladder $NAME: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
LD=$D/pool-$(date +%H%M%S)
# VRAM: the 12-shard token stores stay on the host (page cache); the models and 256-row batches measured under 10 GB in ot_rows.py probe
.venv/bin/python -m jevdrive.cl submit --owner alpasim-ot2 --name ot2-ladder --priority 14 --vram 14 --cpu 4 --ram 70 --log-dir "$LD" -- \
  "$DATA_DIR/envs/op-train/bin/python" experiments/op_parity/scripts/ot_ladder.py --name "$NAME" --ot "$OT" --tags "${TAGS[@]}" --pairs "${PAIRS[@]}" || die "submit"
status "submitted, waiting ($LD)"
until [[ -f $LD/DONE || -f $LD/ERROR ]]; do sleep 20; done
[[ -f $LD/ERROR ]] && die "job failed: $LD/log.txt"
cat "$O/ladder_$NAME.md"
status "done"; date > "$D/DONE"
