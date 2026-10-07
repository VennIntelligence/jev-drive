#!/usr/bin/env bash
# op_parity s2-thinhead lane (plans/2026-10-08-s2-thinhead-prereg.md): one self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job
# goes through the pool. Stages (each skipped when its marker exists; rerunning resumes):
#   q0       hindsight ceiling (CPU)
#   plans    WP2 plans on r2-train rows, token path (GPU, op-train env)         | in parallel
#   extract  long-span multi-frame Qwen3 features of the 479 rater frames (GPU)  |
#   prep     the input / target table (CPU)
#   heads    frozen-feature arms, out-of-fold (ridge heads on the CPU in this window, MLP heads as a small GPU job)
#   ft       Q3 joint fine-tune pilot: cache -> smoke -> pretrain (a) -> k-fold (b) -> report;  latency: the head-path bench
#   STAGES=... selects; KEY=<name> keeps one state per chain instance: $DATA_DIR/runs/op_parity/s2_thinhead/{STATUS,DONE,ERROR}.<KEY>, chain.<KEY>.log, jobs.txt.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/s2_thinhead; mkdir -p "$D"; K=${KEY:-main}                # KEY: one state per chain instance (main, ft, ...)
DONE=$D/DONE.$K; ERR=$D/ERROR.$K; rm -f "$DONE" "$ERR"
exec > >(tee -a "$D/chain.$K.log") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
STAGES=${STAGES:-q0 plans extract prep heads}
status() { echo "$(date '+%F %T') op_parity s2_thinhead ($K): $*" | tee "$D/STATUS.$K"; }
die() { status "ERROR $*"; echo "$*" > "$ERR"; exit 1; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return 0
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '($3 == n || $4 == n) && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return 0; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }

if has q0 && [[ ! -f $D/q0.done ]]; then status "q0"; PYTHONPATH=. $J $S/s2_thinhead.py q0 || die q0; touch "$D/q0.done"; fi
W=()
if has plans; then sub s2th-plans $L/plans --vram 8 --cpu 6 --ram 40 -- $PY $S/s2_thinhead.py plans; W+=($L/plans); fi
if has extract; then sub s2th-extract $L/extract --vram 14 --cpu 8 --ram 24 -- env PYTHONPATH=. $J $S/s2_thinhead.py extract; W+=($L/extract); fi
[[ ${#W[@]} -gt 0 ]] && { status "waiting for ${W[*]##*/}"; waitdirs "${W[@]}"; }
if has prep && [[ ! -f $D/prep.done ]]; then status "prep"; PYTHONPATH=. $J $S/s2_thinhead.py prep || die prep; touch "$D/prep.done"; fi
if has heads; then
    status "heads"
    sub s2th-mlp $L/mlp --vram 6 --cpu 4 --ram 12 -- env PYTHONPATH=. $J $S/s2_thinhead_heads.py fit --part mlp
    [[ -f $D/ridge.done ]] || { PYTHONPATH=. $J $S/s2_thinhead_heads.py fit --part ridge || die ridge; touch "$D/ridge.done"; }
    waitdirs $L/mlp
    PYTHONPATH=. $J $S/s2_thinhead_heads.py report || die report
fi
if has q2 && [[ ! -f $D/q2.done ]]; then status "q2"; PYTHONPATH=. $J $S/s2_thinhead_heads.py q2 || die q2; touch "$D/q2.done"; fi
if has figs; then PYTHONPATH=. $J $S/s2_thinhead_heads.py figs || die figs; fi
if has ft; then                                                              # Q3: the joint fine-tune pilot (chained in the pool)
    F="env PYTHONPATH=. $J $S/s2_thinhead_ft.py"
    status "ft: cache + smoke"
    sub s2th-ft-cache $L/ft-cache --vram 16 --cpu 8 --ram 20 -- $F cache
    waitdirs $L/ft-cache
    sub s2th-ft-smoke $L/ft-smoke --train --vram 30 --cpu 10 --ram 24 -- bash -c "$F pretrain --n-pre 32 --n-dev 16 --tag _smoke && $F kfold --tag _smoke"
    waitdirs $L/ft-smoke
    status "ft: pretrain + kfold"
    sub s2th-ft-pre $L/ft-pre --train --vram 30 --cpu 10 --ram 24 -- $F pretrain
    waitdirs $L/ft-pre
    sub s2th-ft-kfold $L/ft-kfold --train --vram 30 --cpu 6 --ram 24 -- $F kfold
    waitdirs $L/ft-kfold
    PYTHONPATH=. $J $S/s2_thinhead_ft.py report || die "ft report"
fi
if has latency; then
    sub s2th-latency $L/latency --vram 14 --cpu 4 --ram 12 -- env PYTHONPATH=. $J $S/s2_thinhead.py latency
    waitdirs $L/latency
fi
status "done ($STAGES)"; touch "$DONE"
