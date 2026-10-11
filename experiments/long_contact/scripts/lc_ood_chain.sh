#!/usr/bin/env bash
# LONG-PROBE out-of-distribution rows (prereg section 10, descriptive), after the main chain (lc_chain.sh) is DONE: decision 220's nuPlan
# decisions (lc_ood.py) and LOWDIAG's stored PAI rollouts (lc_pai.py), one GPU job at a time, then the CPU reports.
#   scripts/tmux_run.sh long-probe-ood bash experiments/long_contact/scripts/lc_ood_chain.sh
# State: $DATA_DIR/runs/long_contact/ood_chain/{STATUS, DONE | ERROR, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/long_contact
D=$O/ood_chain; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=$DATA_DIR/jev-drive/experiments/long_contact/scripts
status() { echo "$(date '+%F %T') long-probe ood: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
LAST=""
sub() { local n=$1 ld=$2; shift 2; if [[ -f $ld/DONE ]]; then return; fi
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        if [[ -n $live ]]; then LAST=$live; return; fi
        rm -f "$ld/ERROR"
        local aft=(); [[ -n $LAST ]] && aft=(--after "$LAST")
        local id; id=$($CL submit --owner long-probe --name "$n" --log-dir "$ld" "${aft[@]}" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; LAST=$id; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }

status "waiting for the main chain"
until [[ -f $O/chain/DONE || -f $O/chain/ERROR ]]; do sleep 60; done
[[ -f $O/chain/ERROR ]] && die "main chain failed"
status "GPU: decision 220 nuPlan decisions, PAI replay with token dump, PAI scores"
sub lc-ood-g2 "$L/g2" --vram 12 --cpu 8 --ram 32 --timeout-h 1 -- $PY $S/lc_ood.py g2
sub lc-pai-replay "$L/pai-replay" --vram 16 --cpu 8 --ram 40 --timeout-h 3 --env ALPASIM_SRC=$DATA_DIR/third_party/alpasim --env SH30_TAG=P2H10-F-s0 -- $PY $S/lc_pai.py replay --tag P2H10-F-s0
sub lc-pai-score "$L/pai-score" --vram 8 --cpu 4 --ram 16 --timeout-h 1 -- $PY $S/lc_pai.py score
waitdirs "$L/g2" "$L/pai-replay" "$L/pai-score"
status "CPU reports"
$PY $S/lc_ood.py g2-report || die "g2-report"
$PY $S/lc_pai.py report || die "pai report"
status "done"
echo ok > "$D/DONE"
