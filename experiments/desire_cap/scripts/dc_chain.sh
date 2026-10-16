#!/usr/bin/env bash
# DESIRE-CAP chain (plans/2026-10-16-desire-cap-prereg.md). Box, repo root. One submit: rows (CPU) -> plans of every checkpoint x desire arm
# (GPU) -> report (CPU), chained with --after. AFTER = the CAP-LEDGER job that builds the shared row cache's route memory (led-rmnav); empty when
# idx_capledger_test and its route features already exist. Priority 0: the pool's training / PAI reads keep precedence.
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/desire_cap/pool; mkdir -p $O
TR=$DATA_DIR/envs/op-train/bin/python; S=experiments/desire_cap/scripts/dc.py
sub() { local name=$1 after=$2; shift 2; .venv/bin/python -m jevdrive.cl submit --owner desire-cap --name "$name" --tries 1 --priority ${PRIO:-0} --no-check \
        --log-dir $O/$name ${after:+--after $after} "$@" | awk '{print $NF}'; }
i1=$(sub dc-rows "${AFTER:-}" --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- $TR $S rows) || exit 1
i2=$(sub dc-plans "$i1" --vram 16 --cpu 4 --ram 48 --timeout-h 3 -- $TR $S plans) || exit 1
i3=$(sub dc-report "$i2" --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- $TR $S report --out $DATA_DIR/runs/desire_cap/report) || exit 1
echo "$i1 $i2 $i3" | tee -a $O/jobs.txt
