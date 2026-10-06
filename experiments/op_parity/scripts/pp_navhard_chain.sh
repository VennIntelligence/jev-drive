#!/usr/bin/env bash
# op_parity navhard readout (results/navhard.md), one-shot chain in tmux jev (scripts/tmux_run.sh). bench places our arms; the retained WA-JEPA path uses its existing pool restrictions:
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
B=("$PWD/.venv/bin/python" -m jevdrive.bench)
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

# ---------------------------------------------------------------- our arms (shared bench stages)
status "bench arms; retained WA-JEPA jobs on allowed cards $GW"
"${B[@]}" run --model $MODELS --bench navhard || die "bench navhard arms"

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
waitdirs $L/wj $L/wj-check3
# Bench runs place their own stages; GPU_DONE remains a conservative completion gate for the unfreeze training lane.
"${B[@]}" status --model $MODELS --bench navhard --wait || die "bench navhard"
status "GPU inference and bench scoring done (cards free)"
date > "$D/GPU_DONE"
waitdirs $L/h-wajepa
$PY $S/pp_navhard.py report || die "report"
status "done"
date > "$D/DONE"
