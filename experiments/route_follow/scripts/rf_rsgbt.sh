#!/usr/bin/env bash
# RSGboth-TRACK (serving-side reads, no new training): held 460 / 18-decision reads of RSGboth-L18-s0 (and RSG-L18-s0 as control) with
# the spin-fix controller-reference switch JEV_TRACK (experiments/alpasim/lib/track_ref.py), against the stored plain reads. The reads were
# submitted with rm_chain.sh submit-read (READ_ENV + READ_SFX), labels:
#   RSGboth-L18-s0-rtgate   JEV_TRACK=gate            RSGboth-L18-s0-rtgatep  JEV_TRACK=gate JEV_TRACK_CFG=preview=0.5
#   RSGboth-L18-s0-rttimep  JEV_TRACK=time JEV_TRACK_CFG=preview=0.5            RSG-L18-s0-rtgatep  (control, gate + preview 0.5)
# report: held.py, ol_confirm held18, rf_follow, rf_inside, zeros_mech collect, rf_rsgbv.py diag (zeros.csv: type x road class). Box, repo root.
#   rf_rsgbt.sh report
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/rsgboth_track; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python
NEW="RSGboth-L18-s0-rtgate RSGboth-L18-s0-rtgatep RSGboth-L18-s0-rttimep RSG-L18-s0-rtgatep"; REF="RSGboth-L18-s0 RSG-L18-s0"; ALL="$NEW $REF"
mkdir -p $O
R=""; for t in $ALL; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
H=(); F=(); PH=(); PF=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
for t in $NEW; do for r in $REF; do PH+=("$t:$r"); PF+=("$t@18:$r@18"); done; done
PH+=("RSGboth-L18-s0-rtgatep:RSGboth-L18-s0-rtgate" "RSGboth-L18-s0-rttimep:RSGboth-L18-s0-rtgatep")
PF+=("RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18" "RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18")
$P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --pairs "${PH[@]}" --out $O/held18; rc1=$?
$P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}"; rc2=$?
$P experiments/route_follow/scripts/rf_inside.py --follow $O/follow --pairs "${PF[@]}" --out $O/inside.json > /dev/null; rc3=$?
C=""; for t in $ALL; do for i in 0 1 2 3 4 5; do C="$C $t~c$i=$M/$t/c$i/run"; done; done
$P experiments/zeros_mech/scripts/zm_read.py collect --workers 16 --runs $C > $O/zm_collect.log 2>&1; rc5=$?
Z=$(grep -o "runs/zeros_mech/collect/[0-9-]*" $O/zm_collect.log | tail -1); Z=${Z:+$DATA_DIR/$Z}; [[ -n $Z ]] || Z=$(ls -td $DATA_DIR/runs/zeros_mech/collect/*/ | head -1)
DP=(); for t in $NEW RSGboth-L18-s0; do DP+=("$t:RSG-L18-s0"); done
$P experiments/route_follow/scripts/rf_rsgbv.py diag --collect $Z --follow $O/follow --out $O/diag --pairs "${DP[@]}" > /dev/null; rc6=$?
echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 zm $rc5 diag $rc6 ($Z)"; exit $((rc0 + rc1 + rc2 + rc3 + rc5 + rc6))
