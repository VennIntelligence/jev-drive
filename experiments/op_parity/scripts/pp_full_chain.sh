#!/usr/bin/env bash
# op_parity full run under protocol W (results/stageB.md; approved 2026-10-06): one-shot chain, run in tmux jev via scripts/tmux_run.sh.
# Every GPU job goes through the pool (it places them); this driver only submits, waits and checks. Staged launch (docs/long-runs.md):
#   1 cache shard -> sanity -> the other shards -> sanity + split registration -> 1 training (P2-s0) -> sanity -> the other 5 trainings
#   -> navtest plans (W frames) + CPU scoring, HUGSIM 64 (exam / fixed and spec) for P0 and the 6 checkpoints, in parallel.
# State: $DATA_DIR/runs/op_parity/full/{STATUS, DONE, ERROR, log.txt, jobs.txt}. Rerunning resumes: finished pool jobs are not resubmitted.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/full; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$DATA_DIR/runs/op_parity/pool/full
K=${K:-12}
SPLIT=navsim/op-parity-full
STEPS=${STEPS:-10000} BATCH=${BATCH:-128}
status() { echo "$(date '+%F %T') op_parity full: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
# sub NAME LOGDIR args... -> job id; skipped (prints "done") when LOGDIR/DONE exists
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

# ---------------------------------------------------------------- 1. cache (render keys + warp + side + W teacher), shard 0 first
smoke_c="$PY $S/pp_prep.py --data navtrain_full.s0of$K --frames warp --limit 16 --workers 8"
cache() { sub ppF-c$1 $L/c$1 --vram 30 --cpu 22 --ram 40 --preflight "$smoke_c" -- $PY $S/pp_prep.py --data navtrain_full.s$1of$K --frames warp --workers 20 >/dev/null; }
status "cache shard 0 / $K"
cache 0; waitdirs $L/c0
$PY $S/pp_full_check.py cache --k $K --shards 0 || die "cache sanity, shard 0"
status "cache shards 1..$((K - 1))"
for i in $(seq 1 $((K - 1))); do cache $i; done
waitdirs $(for i in $(seq 1 $((K - 1))); do echo $L/c$i; done)
$PY $S/pp_full_check.py cache --k $K --shards all --register $SPLIT || die "cache sanity / split"
DATA=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)

# ---------------------------------------------------------------- 2. training: P2-s0 first
smoke_t="$PY $S/pp_train.py --arm P3 --frames warp --host --data navtrain_full.s0of$K --split $SPLIT --steps 3 --batch 16 --eval-every 3 --tag smoke-full"
train() { sub ppF-t-$1-s$2 $L/t-$1-s$2 --train --vram 30 --cpu 6 --ram 24 --preflight "$smoke_t" -- $PY $S/pp_train.py --arm $1 --seed $2 \
            --frames warp --host --data $DATA --split $SPLIT --steps $STEPS --batch $BATCH --warmup 300 --eval-every 1000 --tag $1-F-s$2 >/dev/null; }
status "training P2-F-s0 (stage 1)"
train P2 0; waitdirs $L/t-P2-s0
$PY $S/pp_full_check.py train --tag P2-F-s0 || die "training sanity, P2-F-s0"
status "training P1 / P2 / P3 x s0 / s1"
for a in P1 P2 P3; do for s in 0 1; do train $a $s; done; done
TAGS="P1-F-s0 P1-F-s1 P2-F-s0 P2-F-s1 P3-F-s0 P3-F-s1"

# ---------------------------------------------------------------- 3. readouts as checkpoints arrive
waitdirs $(for a in P1 P2 P3; do for s in 0 1; do echo $L/t-$a-s$s; done; done)
status "readouts: navtest plans + scoring, HUGSIM 64 x {exam, spec}"
sub ppF-plans $L/plans --vram 30 --cpu 8 --ram 24 -- $PY $S/pp_eval.py --data lb_navtest --frames warp plans --models P0 $TAGS --tag full >/dev/null
waitdirs $L/plans
sub ppF-score $L/score --vram 1 --cpu 24 --ram 48 --env NAVSIM_THREADS=22 -- $PY $S/pp_eval.py --data lb_navtest --frames warp score --models P0 $TAGS >/dev/null
LIST=experiments/hugsim/scripts/derot_all64.txt
for t in P0 $TAGS; do
  sub ppF-h-$t $L/h-exam-$t --vram 40 --cpu 14 --ram 45 -- bash $S/pp_hugsim.sh arm $t $LIST 6 >/dev/null
  sub ppF-hs-$t $L/h-spec-$t --vram 40 --cpu 14 --ram 45 --env PRESET=spec -- bash $S/pp_hugsim.sh arm $t $LIST 6 >/dev/null
done
waitdirs $L/score $(for t in P0 $TAGS; do echo $L/h-exam-$t $L/h-spec-$t; done)
$PY $S/pp_eval.py --data lb_navtest --frames warp report --models P0 $TAGS --ref P1-F-s0 --tag full || die "navtest report"
status "done"
date > "$D/DONE"
