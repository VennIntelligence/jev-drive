#!/usr/bin/env bash
# Pre-submission re-audit of the NTV0 ACC-GAP image (2026-10-21): test.sh on card 0, then smokes of c000 (card 1) and c001 (card 0)
# against the box native read defect_fix/pai_accgap/NTV0-L18-s0 (code 2e2d0e46), gains 10, the image as built (no driver env).
cd ~/mycode/jev-drive
IMG=jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7
O=/data/runs/alpasim/pai_image/audit_ntv0_v2; P=/data/runs/alpasim/pai_image
W=controller.gains.rel_front_steering_angle_weight=10
( env DATA_DIR=/data GPU=1 PAI_BASEPORT=6700 WIZ=$W bash experiments/alpasim/docker/pai_smoke.sh $IMG $O/smoke_c000 $P/lists/c000.txt $P/native/ntv0accgap_c000/aggregate/results-summary.json > $O/smoke_c000.out 2>&1 ) &
bash experiments/alpasim/docker/test.sh $IMG $O/test "" 0 > $O/test.out 2>&1; echo "test rc=$?" >> $O/rc.txt
( env DATA_DIR=/data GPU=0 PAI_BASEPORT=6400 WIZ=$W bash experiments/alpasim/docker/pai_smoke.sh $IMG $O/smoke_c001 $P/lists/c001.txt $P/native/ntv0accgap_c001/aggregate/results-summary.json > $O/smoke_c001.out 2>&1 ) &
wait
for c in c000 c001; do python3 experiments/alpasim/scripts/pai_image_cmp.py --a $O/smoke_$c --b $P/native/ntv0accgap_$c --out $O/smoke_$c/cmp_box.md > $O/cmp_$c.out 2>&1; done
touch $O/ALLDONE
