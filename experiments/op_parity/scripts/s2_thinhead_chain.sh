#!/usr/bin/env bash
# op_parity s2-thinhead lane (plans/2026-10-08-s2-thinhead-prereg.md): one self-advancing chain in tmux jev (scripts/tmux_run.sh); every GPU job
# goes through the pool. Stages (each skipped when its marker exists; rerunning resumes):
#   q0       hindsight ceiling (CPU)
#   plans    WP2 plans on r2-train rows, token path (GPU, op-train env)         | in parallel
#   extract  long-span multi-frame Qwen3 features of the 479 rater frames (GPU)  |
#   prep     the input / target table (CPU)
#   heads    frozen-feature arms, out-of-fold (ridge heads on the CPU in this window, MLP heads as a small GPU job)
#   STAGES=... selects (default: all that exist). State: $DATA_DIR/runs/op_parity/s2_thinhead/{STATUS, DONE, ERROR, chain.log, jobs.txt}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/s2_thinhead; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/chain.log") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
L=$D/pool
STAGES=${STAGES:-q0 plans extract prep heads}
status() { echo "$(date '+%F %T') op_parity s2_thinhead: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
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
status "done ($STAGES)"; touch "$D/DONE"
