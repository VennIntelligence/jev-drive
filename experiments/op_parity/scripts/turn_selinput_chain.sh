#!/usr/bin/env bash
# op_parity turn selector input (plans/2026-10-08-turn-selector-input-prereg.md). One self-advancing chain in tmux jev:
#   scripts/tmux_run.sh turn-selinput experiments/op_parity/scripts/turn_selinput_chain.sh
# Stages (STAGES="feats smoke select nn report" by default); the neural arms run as a GPU pool job (jevdrive.cl submit) while `select` runs on the CPU.
# State: $DATA_DIR/runs/op_parity/turn_selinput/chain/{STATUS, DONE, ERROR, log.txt}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/turn_selinput
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CLPY=$DATA_DIR/envs/jevdrive/bin/python
S=experiments/op_parity/scripts/turn_selinput.py
status() { echo "$(date '+%F %T') op_parity turn-selinput: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
for st in ${STAGES:-feats smoke select nn report}; do
  status "$st"
  case $st in
    feats)  $PY $S feats || die feats ;;
    smoke)  $PY $S select --smoke || die smoke-select ;;
    select) rm -f "$O/nn.pkl"
            $CLPY -m jevdrive.cl submit --name tsi-nn --owner op_parity --vram 10 --cpu 8 --log-dir "$O/pool/nn" -- $PY $S nn || die submit-nn
            $PY $S select || die select
            status "select done; waiting for the nn pool job"
            while [[ ! -f $O/nn.pkl ]]; do
              last=$(ls -dt "$O"/nn/*/ 2>/dev/null | head -1)
              [[ -n $last && -f $last/ERROR ]] && die "nn job: $(head -c 300 "$last/ERROR")"
              sleep 120
            done ;;
    nn)     rm -f "$O/nn.pkl"
            $CLPY -m jevdrive.cl submit --name tsi-nn --owner op_parity --vram 10 --cpu 8 --log-dir "$O/pool/nn" -- $PY $S nn || die submit-nn ;;
    report) $PY $S report || die report ;;
  esac
done
status "done"; date > "$D/DONE"
