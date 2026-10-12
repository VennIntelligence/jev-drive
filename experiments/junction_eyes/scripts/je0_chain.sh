#!/usr/bin/env bash
# junction_eyes step 0 GPU jobs (the WA extractor holds about 37 GB of host RAM per job: --ram 45) (plans/2026-10-12-prereg-A.md). Every job is a pool job. Box, repo root:
#   bash experiments/junction_eyes/scripts/je0_chain.sh wa       -> WA-Cf features of the J_nav tokens of shards s0, s1, s5-s11 (3 sub-shards each)
#                                                                   + G-ref (rows 0, 3, ..., 597 of lb_navtest re-extracted)
#   bash experiments/junction_eyes/scripts/je0_chain.sh probes   -> JEP-<arm>-s<seed>: R-A / R-B x 2 seeds, blind x 2 rulers, old ruler x 2,
#                                                                   then WA-Cf / Cinque-slot-7 x 2 seeds (after the WA jobs: AFTER="id,id,...")
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR; S=$D/runs/junction_eyes/step0; REPO=$(pwd)
OPT=$D/envs/op-train/bin/python
CL() { .venv/bin/python -m jevdrive.cl "$@"; }
WAPY="cd $D/third_party/wajepa && PYTHONPATH=$D/third_party/wajepa:$D/third_party/navsim $D/envs/wajepa/bin/python $REPO/experiments/op_probe/scripts/opb_wajepa.py"
case ${1:?stage} in
wa)
  for f in $S/wa/rows/navtrain_full.s*of12.npy; do
    d=$(basename $f .npy)
    for i in 0 1 2; do
      [[ -e $D/runs/op_probe/feats/WA/$d.s${i}of3.npz ]] && continue      # resume: only the missing sub-shards
      CL submit --name JEW-${d#navtrain_full.}-$i --owner junction-eyes --vram 24 --cpu 16 --ram 45 --tries 2 --timeout-h 2 --no-check \
        --log-dir $S/wa/pool/$d.$i -- bash -c "$WAPY --data $d --rows-file $f --shard $i 3 --batch 16 --workers 16"
    done
  done
  $OPT -c "import numpy as np; np.save('$S/wa/rows/gref.npy', np.arange(0, 600, 3))"
  [[ -e $D/runs/op_probe/feats/WA/lb_navtest-gref.npz ]] || CL submit --name JEW-gref --owner junction-eyes --vram 24 --cpu 16 --ram 45 --tries 2 --timeout-h 1 --no-check --log-dir $S/wa/pool/gref -- \
    bash -c "$WAPY --data lb_navtest --rows-file $S/wa/rows/gref.npy --tag=-gref --batch 16 --workers 16" ;;
probes)
  P() { local n=$1; shift; CL submit --name JEP-$n --owner junction-eyes --vram 60 --cpu 8 --ram 12 --tries 2 --timeout-h 2 --no-check \
          --log-dir $S/pool/JEP-$n "$@"; }
  for s in 0 1; do
    P A-s$s -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler A --arm Z --seed $s
    P B-s$s -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler B --arm Z --seed $s
    P old-s$s -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler old --arm Z --seed $s
  done
  P Ablind-s0 -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler A --arm blind --seed 0
  P Bblind-s0 -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler B --arm blind --seed 0
  for s in 0 1; do
    P As7-s$s -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler A --arm s7 --seed $s
    P Awa-s$s ${AFTER:+--after $AFTER} -- $OPT experiments/junction_eyes/scripts/je0.py probe --ruler A --arm wa --seed $s
  done ;;
*) echo "unknown stage $1" >&2; exit 1 ;;
esac
