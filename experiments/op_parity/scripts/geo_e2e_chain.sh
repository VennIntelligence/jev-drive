#!/usr/bin/env bash
# op_parity geo-e2e (plans/2026-10-09-geo-e2e-prereg.md, decision 200): one self-advancing chain in tmux jev
# (scripts/tmux_run.sh geo-e2e experiments/op_parity/scripts/geo_e2e_chain.sh). Every GPU / CPU job through the pool, every navtest read through
# jevdrive.bench. Arms = the SH30 pilot recipe (P2, hinge lambda 30 / margin 0.5 m, navsim/op-parity-s234, 3 000 steps x 64) + pp_train --mem-e2e:
#   JB (GEB) b: tokenizer over true drivable SDF + true agent occupancy, trained jointly with the adapter (primary)
#   JX (GEX) x: the same on rasters shuffled across logs (control)
#   JW (GEW) b, --mem-init: JB started from the pre-trained geo-oracle tokenizer B
#   JP (GEP) p: the same on the logged-path field (label leak; positive control of the set-up)
# Baseline GH0-F-s{0,1} and the frozen-tokenizer GOB / GOX-F-s{0,1} are decision 197's checkpoints (reused, not retrained).
# All of it is PRIVILEGED input: oracle probes only, never a reportable driver.
#   warm-start tokenizer + 8 trainings (2 seeds, all at once) -> navtest (15 reads) -> four_dirs replay -> report
# State: $DATA_DIR/runs/op_parity/geo_e2e/chain/{STATUS, DONE, ERROR, log.txt, jobs.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/geo_e2e
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
B=("$VPY" -m jevdrive.bench)
L=$D/pool
DATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
SPLIT=navsim/op-parity-s234
HF="--hinge-lam 30 --hinge-margin 0.5"
INIT=$O/tok_b_init.pt
VRAM=${GE_VRAM:-26}          # GB. Measured: 30-step, 64-row smoke incl. dev eval + navtest export peaks at 19.9 (b) / 18.2 (p);
                             # the frozen-bank arms grew from 11.9 (30 steps) to 17.4 over 3 000 steps, so the full run is booked at 26
declare -A TAG=([JB]=GEB [JX]=GEX [JW]=GEW [JP]=GEP) KIND=([JB]=b [JX]=x [JW]=b [JP]=p)
ARMS=(JB JX JW JP)
status() { echo "$(date '+%F %T') op_parity geo-e2e: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
train() {  # arm seed
  local t=${TAG[$1]}-F-s$2 k=${KIND[$1]} gate=() init=""
  [[ $1 == JW ]] && { gate=(--when-exists "$INIT"); init="--mem-init $INIT"; }
  local smoke="$PY $S/pp_train.py --arm P2 --mem-e2e $k --frames warp --host --data navtrain_full.s2of12 --split $SPLIT --steps 3 --batch 16 --eval-every 3 $HF --tag smoke-$t"
  sub ge-t-$t $L/t-$t --train --vram $VRAM --cpu 6 --ram 40 "${gate[@]}" --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --mem-e2e $k $init --seed $2 \
      --frames warp --host --data $DATA --split $SPLIT --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $HF --tag $t
}

status "warm-start tokenizer + 8 trainings (JB JX JW JP x seeds 0 1)"
for t in GH0-F-s0 GH0-F-s1 GOB-F-s0 GOB-F-s1 GOX-F-s0 GOX-F-s1; do
  [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "missing reused checkpoint $t"
done
for d in $DATA lb_navtest; do [[ -f $DATA_DIR/runs/op_parity/mem/geo_x/$d.perm.npy ]] || die "missing geo_x permutation of $d"; done
sub ge-tok-init $L/tok-init --vram 16 --cpu 6 --ram 48 --preflight "$PY $S/geo_oracle.py tok --kind b --smoke" -- $PY $S/geo_oracle.py tok --kind b --weights "$INIT"
for s in 0 1; do for arm in "${ARMS[@]}"; do train $arm $s; done; done
specs=(GH0-F-s1 GOB-F-s1 GOX-F-s1)
for s in 0 1; do
  for arm in "${ARMS[@]}"; do
    t=${TAG[$arm]}-F-s$s
    waitdirs $L/t-$t
    $PY $S/pp_full_check.py train --tag $t || die "training sanity $t"
    [[ -f $DATA_DIR/runs/op_parity/mem/ge_$t/lb_navtest.npy ]] || die "no navtest bank for $t"
    specs+=($t)
  done
  specs+=(GEB-F-s$s:noside GEW-F-s$s:noside)
done
rm -rf "$DATA_DIR"/runs/op_parity/mem/ge_smoke-GE?-F-s?                      # the preflights' scratch banks
status "navtest: ${specs[*]}"
"${B[@]}" run --model "${specs[@]}" --bench navtest || die "bench navtest"
"${B[@]}" status --model "${specs[@]}" --bench navtest --wait || die "navtest"
status "four_dirs replay"
sub ge-replay $L/replay --vram 0.5 --cpu 48 --ram 64 -- $NAV $S/turn_oracle.py replay --name geo_e2e --models "${specs[@]}"
waitdirs $L/replay
status "report"
sub ge-report $L/report --vram 0.5 --cpu 8 --ram 32 -- $PY $S/geo_e2e.py report --seeds 0 1 --replays geo_s0 geo_e2e
waitdirs $L/report
status "done"; date > "$D/DONE"
