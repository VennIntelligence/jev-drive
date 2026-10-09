#!/usr/bin/env bash
# Lane OT2 piece B (plans/2026-10-09-ot2-dose-prereg.md, decision 210): AP2 input standard + off-track rows, amplitude x share, 2 seeds each.
# One self-advancing chain per stage in tmux jev (scripts/tmux_run.sh ot2-b experiments/alpasim/scripts/ot2_b_chain.sh <stage>); every GPU / CPU
# job goes through the pool, benchmark reads through jevdrive.bench. Rerunning a stage resumes (finished pool jobs are not resubmitted).
#   train   trainer check (ap2_ot.py --ot-mass 0 = ap2_train.py over 20 steps) + one 20-step smoke with off-track rows (VRAM) -> AP2-AB-s1 and the
#           four +-0.5 m trainings (cached ot1 rows) -> +-1.5 m rows: shard 2, sign gate (ot_ladder.py), the other 11 shards -> the four +-1.5 m
#           trainings + the ladder probe on the +-1.5 m rows -> training sanity
#   side    AlpaSim-standard offline read (seed 0 of every recipe, m = 1 / 4) + navtest / navhard through jevdrive.bench for every checkpoint
#   guard   <tag...>: HUGSIM 64 (spec_plan_smooth) for the named checkpoints (the best recipe and the AP2 baseline)
# The closed loop runs beside it: scripts/ot2_loop.py b ... (waits for each checkpoint file).
# State: $DATA_DIR/runs/alpasim/ot2/chain-<stage>/{STATUS, DONE, ERROR, log.txt, jobs.txt}; pool logs under .../ot2/pool/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=$1; shift
O=$DATA_DIR/runs/alpasim/ot2
D=$O/chain-$STAGE; mkdir -p "$D" "$O/results"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
S=experiments/alpasim/scripts
SP=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$O/pool
A=$DATA_DIR/runs/alpasim/ap2
K=12
ALL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
REST=$(for i in 0 1 3 4 5 6 7 8 9 10 11; do echo -n "navtrain_full.s${i}of$K "; done)
SPLIT=navsim/op-parity-full
ARMS="a05m10 a05m25 a15m10 a15m25"
mass() { [[ $1 == *m10 ]] && echo 0.1 || echo 0.25; }
status() { echo "$(date '+%F %T') ot2-b $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        $CL queue 2>/dev/null | awk -v p="$ld/log.txt" '($2 == "queued" || $2 == "running" || $2 == "inbox") && index($0, p) {f = 1} END {exit !f}' && return
        rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner alpasim-ot2 --priority 12 --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
# Declarations are measured peaks plus margin (pool history 2026-10-09): ot-prep 29-42 GB VRAM / 29 GB RAM / 19 cores; ap2-t-full 35.7 GB / 96 GB RAM
# RSS, of which the memory-mapped host token stores (front / backwarp / off-track tokens, shared page cache, reclaimable) are the bulk; the pool
# admits on non-reclaimable memory (anon + shmem + kernel), so --ram declares that part: ot-t-full's whole RSS was 41.6 GB with 26 GB of
# mapped front tokens. The off-track rows add their hinge rasters on the card.
TR="--train --vram 48 --cpu 6 --ram 40"
TF="--data $ALL --split $SPLIT --batch 128 --warmup 300 --eval-every 1000 --cold backwarp"

if [[ $STAGE == train ]]; then
  status "trainer check: 20 steps of ap2_train.py vs ap2_ot.py --ot-mass 0; 20 steps with off-track rows"
  sub apo-smoke $L/smoke-ref  $TR -- $PY $S/ap2_train.py $TF --steps 20 --eval-every 20 --seed 0 --tag smoke-ap2
  sub apo-smoke $L/smoke-ot0  $TR -- $PY $S/ap2_ot.py train $TF --steps 20 --eval-every 20 --seed 0 --tag smoke-apo0 --ot-mass 0
  sub apo-smoke $L/smoke-ot25 $TR -- $PY $S/ap2_ot.py train $TF --steps 20 --eval-every 20 --seed 0 --tag smoke-apo25 --amp a05 --ot-mass 0.25
  waitdirs $L/smoke-ref $L/smoke-ot0 $L/smoke-ot25
  grep -h "step 20:\|dev @ 20\|off-track rows" $L/smoke-ref/log.txt $L/smoke-ot0/log.txt $L/smoke-ot25/log.txt | sed 's/.*INFO *//' | cut -c1-300
  $VPY - $L/smoke-ref/log.txt $L/smoke-ot0/log.txt <<'PYEOF' || die "trainer check: ap2_ot.py --ot-mass 0 does not reproduce ap2_train.py over 20 steps"
import re, sys
def read(f):                                           # the losses of the step-20 line and the dev line
    t = open(f).read()
    return [float(x) for k in ("step 20: ", "dev @ 20: ") for x in re.findall(r" (-?\d+\.\d+)", " " + re.search(k + "([^;\n]*)", t).group(1))]
a, b = read(sys.argv[1]), read(sys.argv[2])
print("ap2_train:", a, "\nap2_ot m=0:", b)
sys.exit(0 if len(a) == len(b) and all(abs(x - y) <= 1e-3 + 2e-3 * abs(x) for x, y in zip(a, b)) else 1)
PYEOF
  status "trainings: AP2-AB-s1 and the +-0.5 m arms"
  sub apo-t $L/t-AP2-AB-s1 $TR -- $PY $S/ap2_train.py $TF --steps 10000 --seed 1 --tag AP2-AB-s1
  for arm in a05m10 a05m25; do for s in 0 1; do
    sub apo-t $L/t-$arm-s$s $TR -- $PY $S/ap2_ot.py train $TF --steps 10000 --seed $s --tag APO-$arm-s$s --amp a05 --ot-mass $(mass $arm); done; done
  status "+-1.5 m rows: shard 2, sign gate"
  prep() { sub ot-prep $L/prep-a15-$1 --vram 44 --cpu 20 --ram 40 -- $PY $S/ap2_ot.py prep --amp a15 --data $1 --workers 18; }
  prep navtrain_full.s2of12; waitdirs $L/prep-a15-navtrain_full.s2of12
  sub ot2-ladder $L/ladder-gate --vram 32 --cpu 4 --ram 30 -- $PY $SP/ot_ladder.py --name ot2gate --ot ot2 --data navtrain_full.s2of12 --split $SPLIT \
      --tags P0 SH30-F-s0 OT30-F-s0
  waitdirs $L/ladder-gate
  cat $DATA_DIR/runs/op_parity/ot_rows/ladder_ot2gate.md
  $VPY - <<'PYEOF' || die "sign gate: OT30-F-s0 does not respond to the +-1.5 m offsets with the right sign (lateral or yaw response at 4 s < 0.1)"
import json, os, sys
r = json.load(open(os.environ["DATA_DIR"] + "/runs/op_parity/ot_rows/ladder_ot2gate.json"))["models"]["OT30-F-s0"]
sys.exit(0 if min(r["dy_4s"][0], r["yaw_4s"][0]) >= 0.1 else 1)
PYEOF
  status "+-1.5 m rows: the other 11 shards"
  for d in $REST; do prep $d; done
  waitdirs $(for d in $REST; do echo $L/prep-a15-$d; done)
  status "trainings: the +-1.5 m arms; ladder probe on the +-1.5 m rows"
  for arm in a15m10 a15m25; do for s in 0 1; do
    sub apo-t $L/t-$arm-s$s $TR -- $PY $S/ap2_ot.py train $TF --steps 10000 --seed $s --tag APO-$arm-s$s --amp a15 --ot-mass $(mass $arm); done; done
  sub ot2-ladder $L/ladder-ot2 --vram 32 --cpu 4 --ram 70 -- $PY $SP/ot_ladder.py --name ladder_ot2 --ot ot2 --split $SPLIT \
      --tags P0 P2-F-s0 P2-F-s1 RMH10-F-s0 RMH10-F-s1 SH30-F-s0 SH30-F-s1 AP2-AB-s0 OT30-F-s0 OT30-F-s1 \
      --pairs P2-F-s0:P0 SH30-F-s0:P0 SH30-F-s1:P0 AP2-AB-s0:P0 OT30-F-s0:P0 OT30-F-s0:SH30-F-s0 OT30-F-s1:SH30-F-s1
  waitdirs $L/ladder-ot2 $L/t-AP2-AB-s1 $(for arm in $ARMS; do for s in 0 1; do echo $L/t-$arm-s$s; done; done)
  for t in AP2-AB-s1 $(for arm in $ARMS; do echo APO-$arm-s0 APO-$arm-s1; done); do [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no checkpoint $t"; done
  grep -h "dev @ 10000" $L/t-*/log.txt | sed 's/.*INFO//' | cut -c1-400
elif [[ $STAGE == side ]]; then
  TAGS="AP2-AB-s0 AP2-AB-s1 $(for arm in $ARMS; do echo -n "APO-$arm-s0 APO-$arm-s1 "; done)"
  status "navtest / navhard through jevdrive.bench (low priority), AlpaSim-standard offline read"
  "${B[@]}" run --model $TAGS --bench navtest --priority 2 || die "bench navtest"
  "${B[@]}" run --model $(for t in $TAGS; do echo -n "$t@gimm "; done) --bench navhard --priority 2 || die "bench navhard"
  sub ot-ap2-offline $L/offline-b --vram 32 --cpu 8 --ram 60 -- $PY $S/ap2_offline.py plans --name ot2-b --scenes $DATA_DIR/runs/alpasim/scenes_navtest_full_part001_48.txt \
      --models AP2=AP2-AB-s0 $(for arm in $ARMS; do echo -n "$arm=APO-$arm-s0 "; done)
  waitdirs $L/offline-b
  keys=$($VPY -c "import numpy as np, re, sys; print(' '.join(k for k in np.load(sys.argv[1]).files if re.search('_m[14]\$', k)))" $A/offline/ot2-b/poses.npz)
  [[ -f $A/offline/ot2-b/scores.csv ]] || "${B[@]}" score-poses --poses $A/offline/ot2-b/poses.npz --keys $keys --tokens $A/offline/ot2-b/subset.txt \
      --out $A/offline/ot2-b/scores.csv --traffic non_reactive --priority 2 --owner alpasim-ot2 --wait || die "score-poses"
  $PY $S/ap2_offline.py report --name ot2-b --ref AP2 --scores $A/offline/ot2-b/scores.csv --out $O/results/ap2_offline_ot2-b || die "offline report"
  "${B[@]}" status --model $TAGS --bench navtest --wait || die "navtest"
  "${B[@]}" status --model $(for t in $TAGS; do echo -n "$t@gimm "; done) --bench navhard --wait || die "navhard"
  for arm in $ARMS; do
    "${B[@]}" report --bench navtest --arms $arm=APO-$arm-s0+APO-$arm-s1 --vs AP2=AP2-AB-s0+AP2-AB-s1 SH30=SH30-F-s0+SH30-F-s1 OT30=OT30-F-s0+OT30-F-s1 --out $O/results/bench-$arm || die "report navtest $arm"
    "${B[@]}" report --bench navhard --arms $arm=APO-$arm-s0@gimm+APO-$arm-s1@gimm --vs AP2=AP2-AB-s0@gimm+AP2-AB-s1@gimm SH30=SH30-F-s0@gimm+SH30-F-s1@gimm OT30=OT30-F-s0@gimm+OT30-F-s1@gimm \
        --out $O/results/bench-$arm || die "report navhard $arm"
  done
elif [[ $STAGE == guard ]]; then
  HP=spec_plan_smooth; LIST=experiments/hugsim/scripts/derot_all64.txt
  status "HUGSIM 64 ($HP): $*"
  "${B[@]}" run --model "$@" --bench hugsim --preset $HP --scenarios "$LIST" --priority 12 || die "bench hugsim"
  "${B[@]}" status --model "$@" --bench hugsim --preset $HP --wait || die "hugsim"
else
  die "unknown stage $STAGE"
fi
status "done"; date > "$D/DONE"
