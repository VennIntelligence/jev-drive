#!/usr/bin/env bash
# op_parity navhard readout (results/navhard.md), one-shot chain in tmux jev (scripts/tmux_run.sh). GPU work on two cards only
# (GPUS_ARMS for our arms, GPUS_WJ for WA-JEPA; the third card belongs to the unfreeze lane), CPU scoring overlaps inference:
#   arms:    prep lb_navhard (tab + side, W front) -> plans P0 + 6 full-run checkpoints -> nav-export -> nav_harness per model (CPU)
#   WA-JEPA: request check on 64 navtest tokens vs its stored export -> navhard requests, 6 runner shards on one card -> nav_harness
# State: $DATA_DIR/runs/op_parity/navhard/{STATUS, DONE, ERROR, log.txt, jobs.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/navhard; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
NAV2=$DATA_DIR/envs/navsim2/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
GA=${GPUS_ARMS:-0} GW=${GPUS_WJ:-1}
MODELS="P0 P1-F-s0 P1-F-s1 P2-F-s0 P2-F-s1 P3-F-s0 P3-F-s1"
status() { echo "$(date '+%F %T') op_parity navhard: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
aft() { [[ $1 == done ]] && echo "" || echo "--after $1"; }

# ---------------------------------------------------------------- our arms (card GA)
status "prep + plans (cards $GA), WA-JEPA (card $GW)"
j_prep=$(sub ppH-prep $L/prep --gpus $GA --vram 20 --cpu 22 --ram 40 \
  --preflight "$PY $S/pp_prep.py --data lb_navhard --limit 16 --workers 8 --no-side" -- \
  bash -c "$PY $S/pp_prep.py --data lb_navhard --workers 20 && $PY $S/pp_prep.py --data lb_navhard --frames warp --workers 20")
j_plans=$(sub ppH-plans $L/plans --gpus $GA --vram 30 --cpu 8 --ram 24 $(aft $j_prep) -- \
  $PY $S/pp_eval.py --data lb_navhard --frames warp plans --models $MODELS --tag navhard)
stems=$(for m in $MODELS; do echo -n "warp@cinque_PP$m "; done)
j_exp=$(sub ppH-export $L/export --gpus $GA --vram 1 --cpu 4 --ram 16 --env OPI_ROOT=op_lb $(aft $j_plans) -- \
  $JEV experiments/op_openloop/lib/op_interp.py nav-export --data lb_navhard --adapters base --plans $stems)
for m in $MODELS; do
  sub ppH-h-$m $L/h-$m --gpus $GA,$GW --vram 1 --cpu 10 --ram 24 $(aft $j_exp) -- \
    $NAV2 experiments/op_guard/scripts/nav_harness.py --poses $DATA_DIR/runs/op_lb/lb_navhard/preds/warp-cinque_PP${m}__base.npz \
    --out $D/harness/$m --procs 10 >/dev/null
done

# ---------------------------------------------------------------- WA-JEPA (card GW): its runner, fp32 (its NAVSIM path)
# check first: 1 200 navtest tokens through our request path, scored by the devkit, vs its stored navtest per-token scores
W=$D/wajepa; mkdir -p "$W"
CHK=wj_navtestlogs
j_wc=$(sub ppH-wj-check $L/wj-check3 --gpus $GW --vram 40 --cpu 20 --ram 48 -- bash -c "
  bash $S/pp_navhard_wajepa.sh navtest 1200 $W/navtestlogs &&
  OPENBLAS_CORETYPE=Haswell NAVSIM_THREADS=20 TOKENS_FILE=$W/navtestlogs_req.tokens experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v2 navtest $CHK $W/navtestlogs_preds.npz > $W/score_check.log 2>&1 &&
  $PY $S/pp_navhard.py wcheck --name $CHK")
j_wj=$(sub ppH-wj $L/wj --gpus $GW --vram 40 --cpu 24 --ram 48 -- bash $S/pp_navhard_wajepa.sh navhard_two_stage 0 $W/navhard)
sub ppH-h-wajepa $L/h-wajepa --gpus $GA,$GW --vram 1 --cpu 10 --ram 24 $(aft $j_wj) -- \
  $NAV2 experiments/op_guard/scripts/nav_harness.py --poses $W/navhard_preds.npz --out $D/harness/wajepa --procs 10 >/dev/null

# ---------------------------------------------------------------- report
waitdirs $L/plans $L/wj $L/wj-check3; status "GPU inference done (cards $GA $GW free); scoring"
date > "$D/GPU_DONE"
waitdirs $(for m in $MODELS wajepa; do echo $L/h-$m; done)
$PY $S/pp_navhard.py report || die "report"
status "done"
date > "$D/DONE"
