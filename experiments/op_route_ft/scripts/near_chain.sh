#!/usr/bin/env bash
# rc-*-near chain (plans/2026-10-05-route-ft-prereg.md, section "2026-10-06 near"). Every phase runs on a lease the caller holds. STATUS / DONE / ERROR
# in $R/chain/near-<phase>/. Data root $N = $DATA_DIR/runs/op_route_ft/carla_near/{plan.path, render/, packed/}.
#   near_chain.sh render "<gpu> <idx> <cpus>" ...   one CARLA server + renderer per spec (from `python -m jevdrive.cl lease`), carla_pairs_render_ol.py
#                                                  unchanged; LIMIT=<n> env renders only the first n poses of shard 0 (stage 5); IDS=a,b only these ids
#   near_chain.sh pack                              CPU: carla_pairs_pack + near columns (near_pack.py)
#   near_chain.sh bank <gpu>                        stage-3 trunks + original outputs per near pose (rft.py bank --which near)
#   near_chain.sh pilot <gpu> <cpus>                rc-bear-near and rc-bear-fix 400 steps in parallel (tags pilot-bear-near / pilot-bear-fix, both with the
#                                                  decision-130 action target); real-row gain (pre_diag) and near readout
#   near_chain.sh full <gpu> <cpus> [ctl]           rc-bear-near + rc-bear-fix (+ rc-ctl-near with `ctl`) 4000 steps, readouts, ONNX + adapter, equivalence
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=$DATA_DIR/runs/op_route_ft; N=$R/carla_near
PY=$DATA_DIR/envs/op-train/bin/python; PYC=$DATA_DIR/envs/carla/bin/python; PYJ=$DATA_DIR/envs/jevdrive/bin/python
S=experiments/op_route_ft/scripts; RC=experiments/op_route_cmd/scripts
phase=$1; shift
D=$R/chain/near-$phase; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
say() { echo "$(date '+%F %T') near-$phase: $*" | tee -a "$D/log.txt" > /dev/null; echo "$(date '+%F %T') near-$phase: $*" > "$D/STATUS"; }
die() { say "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
case $phase in
  render)
    P=$(cat "$N/plan.path"); K=$#; i=0; pids=()
    extra=(); [ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT"); [ -n "${IDS:-}" ] && extra+=(--ids "$IDS")
    say "rendering $K shards ${extra[*]}"
    for spec in "$@"; do
      read -r gpu idx cpus <<< "$spec"
      $PYC $RC/carla_pairs_render_ol.py --plan "$P" --out "$N/render" --gpu "$gpu" --idx "$idx" --cpus "$cpus" --shard "$i/$K" --hero "${extra[@]}" \
        > "$D/render_$i.log" 2>&1 &
      pids+=($!); i=$((i+1)); sleep 8
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ $rc = 0 ] || die "a render shard failed (see $D/render_*.log, $N/render/ERROR_*)" ;;
  pack)
    say "pack"; $PYJ $S/near_pack.py --plan "$(cat $N/plan.path)" --render "$N/render" --root "$N/packed" >> "$D/log.txt" 2>&1 || die "pack"
    $PYC $RC/carla_pairs_stats.py "$N/render" > "$N/render_stats.txt" 2>&1 ;;
  bank)
    gpu=$1; say "bank"
    CUDA_VISIBLE_DEVICES=$gpu $PY $S/rft.py bank --which near >> "$D/log.txt" 2>&1 || die "bank" ;;
  pilot)
    gpu=$1; cpus=$2
    run() { CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpus" "$PY" "$@"; }
    say "pilot rc-bear-near + rc-bear-fix 400 steps"
    run $S/rft.py train --arm rc-bear-near --steps 400 --tag pilot-bear-near --fresh > "$D/train-near.log" 2>&1 & p1=$!
    run $S/rft.py train --arm rc-bear-fix --steps 400 --tag pilot-bear-fix --fresh > "$D/train-fix.log" 2>&1 & p2=$!
    wait $p1 || die "train near"; wait $p2 || die "train fix"
    M="O pilot-bear pilot-bear-fix pilot-bear-near"
    say "real-row gain"; run $S/pre_diag.py --models $M --out near_pilot_diag.json >> "$D/log.txt" 2>&1 || die "pre_diag"
    say "near readout"; run $S/near_eval.py eval --models $M >> "$D/log.txt" 2>&1 || die "near_eval"
    say "ol readouts"; run $S/rft.py evalol --models pilot-bear-fix pilot-bear-near --carla ol >> "$D/log.txt" 2>&1 || die "evalol" ;;
  full)
    gpu=$1; cpus=$2; ctl=${3:-}
    run() { CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpus" "$PY" "$@"; }
    arms="rc-bear-near rc-bear-fix"; [ "$ctl" = ctl ] && arms="rc-bear-near rc-bear-fix rc-ctl-near"
    say "train $arms"; pids=()
    for a in $arms; do run $S/rft.py train --arm $a --fresh > "$D/train-$a.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || die "train (see $D/train-*.log)"; done
    models=$(for a in $arms; do echo -n "$a-s0 "; done)
    say "readouts"
    run $S/rft.py evalol --models $models --carla ol >> "$D/log.txt" 2>&1 || die "evalol"
    run $S/pre_diag.py --models O rc-ctl-s0 rc-bear-s0 rc-bear-pre-s0 $models --out near_diag.json >> "$D/log.txt" 2>&1 || die "pre_diag"
    run $S/near_eval.py eval --models O rc-ctl-s0 rc-bear-s0 rc-bear-pre-s0 $models >> "$D/log.txt" 2>&1 || die "near_eval"
    for a in $arms; do
      ad=$R/runs/$a-s0/adapter.npz; [ "$a" = rc-ctl-near ] && ad=none
      say "onnx $a"
      $PY $S/route_onnx.py build --ckpt $R/runs/$a-s0/ckpt-final.pt --adapter $ad --out $R/onnx/$a-s0.onnx >> "$D/log.txt" 2>&1 || die "onnx $a"
    done
    say "equivalence rc-bear-near"
    run $S/route_onnx.py ref --ckpt $R/runs/rc-bear-near-s0/ckpt-final.pt --adapter $R/runs/rc-bear-near-s0/adapter.npz \
      --out $R/onnx/rc-bear-near-s0.ref.npz >> "$D/log.txt" 2>&1 || die "ref"
    CUDA_VISIBLE_DEVICES=$gpu $DATA_DIR/envs/openpilot/bin/python $S/route_onnx.py check --onnx $R/onnx/rc-bear-near-s0.onnx \
      --ref $R/onnx/rc-bear-near-s0.ref.npz >> "$D/log.txt" 2>&1 || die "check" ;;
  *) die "unknown phase $phase" ;;
esac
say "done"; date '+%F %T' > "$D/DONE"
