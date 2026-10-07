#!/usr/bin/env bash
# op_parity representation fix, stage 0 (plans/2026-10-07-representation-design.md 6.1): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh rep-s0 experiments/op_parity/scripts/rep_chain.sh). Every GPU / CPU job goes through the pool.
#   vj21 extraction (equivalence check first) + X4 extraction -> decoders -> opb_score on T20 / rest of eval tokens / rest of S5 -> report
# State: $DATA_DIR/runs/op_parity/rep/{STATUS, DONE, ERROR, log.txt, jobs.txt}. Rerunning resumes (finished jobs are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/op_parity/rep; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
JEV=$DATA_DIR/envs/jevdrive/bin/python
NAV2=$DATA_DIR/envs/navsim2/bin/python
CL="$JEV -m jevdrive.cl"
S=experiments/op_parity/scripts
O=$DATA_DIR/runs/op_probe/rep
L=$D/pool
status() { echo "$(date '+%F %T') op_parity rep s0: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && { echo done; return; }
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && { echo "$live"; return; }; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; echo "$id"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }

status "extraction: VJ21 (wajepa env) + X4 (W render)"
sub rep-vj21 $L/vj21 --vram 20 --cpu 16 --ram 48 -- bash -c "cd \$DATA_DIR/third_party/wajepa && PYTHONPATH=\$DATA_DIR/third_party/wajepa:\$DATA_DIR/third_party/navsim \$DATA_DIR/envs/wajepa/bin/python $PWD/$S/rep.py vj21 --workers 14" >/dev/null
sub rep-x4 $L/x4 --vram 8 --cpu 24 --ram 48 -- $PY $S/rep.py x4 --workers 22 >/dev/null
waitdirs $L/vj21 $L/x4

status "decoders (8 arms) + junction probe"
sub rep-decode $L/decode --vram 40 --cpu 8 --ram 96 -- $PY $S/rep.py decode >/dev/null
waitdirs $L/decode

status "scoring (opb_score.py, devkit pdm_score)"
P=$O/decoder_poses.npz
sub rep-score-t20 $L/score-t20 --vram 0.5 --cpu 40 --ram 64 -- $NAV2 experiments/op_probe/scripts/opb_score.py --poses $P \
    --keys E V WA V+WA VJ21 V+VJ21 X4 --tokens $O/tokens_t20.txt --out $O/score_t20.csv --procs 40 >/dev/null
sub rep-score-ev $L/score-ev --vram 0.5 --cpu 16 --ram 32 -- $NAV2 experiments/op_probe/scripts/opb_score.py --poses $P \
    --keys V WA V+VJ21 --tokens $O/tokens_rest_eval.txt --out $O/score_rest_eval.csv --procs 16 >/dev/null
sub rep-score-s5 $L/score-s5 --vram 0.5 --cpu 24 --ram 48 -- $NAV2 experiments/op_probe/scripts/opb_score.py --poses $P \
    --keys V V+VJ21 --tokens $O/tokens_rest_s5.txt --out $O/score_rest_s5.csv --procs 24 >/dev/null
waitdirs $L/score-t20 $L/score-ev $L/score-s5

status "report"
$PY $S/rep.py report > $D/report.txt || die "report"
status "done"
date > "$D/DONE"
