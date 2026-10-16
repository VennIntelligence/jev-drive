#!/usr/bin/env bash
# CAP-LEDGER chain (plans/2026-10-16-cap-ledger-prereg.md). Box, repo root, after `led.py select`. One submit: define + index (CPU) ->
# pp_prep W (GPU) -> route memory (CPU) -> plans of every checkpoint under H and F (GPU) -> ledger (CPU), chained with --after.
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/cap_ledger/pool; mkdir -p $O
NS=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
S=experiments/cap_ledger/scripts/led.py; D=idx_capledger_test; R=$DATA_DIR/runs/cap_ledger
TAGS="P0 P2-F-s0 P2-F-s1 P2H10-F-s0 P2H10-F-s1 YR10m10-F-s0 YR10m10-F-s1 APY10m10-AB-s0 APY10m10-AB-s1 YS-L18-s0 YS-L18-s1 YS2G-L18-s0 YS2G-L18-s1 RB0-F-s0 RB0-F-s1 RSG-L18-s0 RSG-L18-s1 RBSG2-L18-s0 RBSG2-L18-s1"
sub() { local name=$1 after=$2; shift 2; .venv/bin/python -m jevdrive.cl submit --owner cap-ledger --name "$name" --tries 1 --priority ${PRIO:-12} --no-check \
        --log-dir $O/$name ${after:+--after $after} "$@" | awk '{print $NF}'; }
i1=$(sub led-index "" --vram 0.5 --cpu 4 --ram 24 --timeout-h 1 -- bash -c "$NS $S define && $NS $S index") || exit 1
i2=$(sub led-prep "$i1" --vram 10 --cpu 12 --ram 40 --timeout-h 2 -- $TR experiments/op_parity/scripts/pp_prep.py --data $D --frames warp --no-side --synth cpu --workers 12) || exit 1
i3=$(sub led-rmnav "$i2" --vram 0.5 --cpu 12 --ram 40 --timeout-h 1 -- $AP experiments/route_mem/scripts/rm_build.py nav --data $D --workers 12) || exit 1
i4=$(sub led-plans "$i3" --vram 10 --cpu 4 --ram 40 --timeout-h 2 -- $TR $S plans --tags $TAGS) || exit 1
i5=$(sub led-ledger "$i4" --vram 0.5 --cpu 2 --ram 16 --timeout-h 1 -- $TR $S ledger --rows $R/select/rows.npz --plans $R/plans \
       --wc-rows $DATA_DIR/runs/wait_cap/select/rows.npz --wc-plans $DATA_DIR/runs/wait_cap/plans --out $R/report) || exit 1
echo "$i1 $i2 $i3 $i4 $i5" | tee -a $O/jobs.txt
