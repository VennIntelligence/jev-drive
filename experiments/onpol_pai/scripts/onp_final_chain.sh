#!/usr/bin/env bash
# ONPOL-PAI, the rest of the night on the Tokyo box as one self-advancing chain (tmux jev:onp-final):
#   1. waits for the fine-tune control's training (stage `full` with ARMS=FT, started by hand) and for the cut of pass 2 (CUT_DONE)
#   2. full trainings of ON-SELF and ON-IMIT, one per card (onp_night.sh full: cache + G-id of pass log first, held read last)
#   3. G-load of the three checkpoints through the driver image
#   4. the 281-scene read through pai_queue.sh in the order SELF, IMIT, FT (served configuration, CONC 4, CHUNK 36)
# Writes $L/final/{STATUS, DONE | ERROR, log.txt}. ON-BOTH is not here: it stopped at the pilot (prereg section 7).
set -uo pipefail
L=${L:-/data/runs/onpol_pai}; F=$L/final; T=$L/code/train; Q=$L/code/pass1; mkdir -p "$F"; rm -f "$F"/{DONE,ERROR}
exec > >(tee -a "$F/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$F/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$F/ERROR"; exit 1; }
st "waiting for FT training and the pass-2 cut"
until [[ -f $L/train/full/DONE || -f $L/train/full/ERROR ]]; do sleep 20; done
[[ -f $L/train/full/DONE ]] || die "FT training: $(cat "$L/train/full/STATUS")"
until [[ -f $L/pass2/CUT_DONE || -f $L/pass2/CUT_ERROR ]]; do sleep 20; done
[[ -f $L/pass2/CUT_DONE ]] || die "pass-2 cut: $(cat "$L/pass2/CUT_STATUS")"
cp "$L/train/held/full.md" "$F/held_ft.md" 2>/dev/null
st "full trainings: SELF, IMIT"
(cd "$T" && ARMS="SELF IMIT" bash experiments/onpol_pai/scripts/onp_night.sh full "0 1") || die "full trainings: $(cat "$L/train/full/STATUS")"
tags="P2H10S-SELF-ON-F-s0 P2H10S-IMIT-ON-F-s0 P2H10S-FT-F-s0"
for t in $tags; do
  [[ -f /data/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no checkpoint $t"
  (cd "$T" && bash experiments/onpol_pai/scripts/onp_night.sh gload "$t" 1) || die "G-load $t"
done
(cd "$T" && NAME=final bash experiments/onpol_pai/scripts/onp_night.sh held 1 P2H10S-F-s0 $tags) || st "held read of the final tags failed (not fatal)"
st "281 read: SELF, IMIT, FT"
S=/data/runs/alpasim/pai_full
(cd "$Q" && CARDS='0 1' CHUNK=36 CONC=4 SERVE='-e JEV_VCONT=1.0 -e JEV_LEAD=1' bash experiments/alpasim/scripts/pai_queue.sh "$L/read281" \
  "self|P2H10S-SELF-ON-F-s0|$S/s_*.tsv" "imit|P2H10S-IMIT-ON-F-s0|$S/s_*.tsv" "ft|P2H10S-FT-F-s0|$S/s_*.tsv") || die "queue: $(cat "$L/read281/STATUS")"
date > "$F/DONE"; st "done: 281 read of SELF, IMIT, FT in $L/read281"
