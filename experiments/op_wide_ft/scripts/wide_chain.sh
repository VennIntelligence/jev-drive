#!/usr/bin/env bash
# op_wide_ft chain (plans/2026-10-05-wide-ft-prereg.md). Every GPU phase runs as a GPU-pool job (submit lines below); STATUS / DONE / ERROR in
# $W/chain/<phase>[-<tag>]/. Data root W = $DATA_DIR/runs/op_wide_ft (OP_RFT_ROOT for rft.py: bank/, runs/, onnx/, evalol_*/ land here).
#
#   render <first> <n> <total>   inside a pool job with --carla <n>: CARLA shards first .. first+n-1 of <total>, carla_pairs_render_ol.py with
#                                --wide-focals 160 --no-pano, server index CL_IDX + k, cores = CL_CPUS split n ways. LIMIT=<k> / IDS=a,b as the renderer.
#       python -m jevdrive.cl submit --name wf-render0 --vram 45 --carla 6 --cpu 18 --log-dir $W/chain/pool/render0 -- \
#           bash experiments/op_wide_ft/scripts/wide_chain.sh render 0 6 18
#   pack                         CPU (tmux): p58 (frames as rendered) and p116 (wide slot = frames_wf160), both in the ol packed root's pose order
#   bank                         pool GPU: stage-3 trunks + original outputs per pose for w58 and w116 (rft.py bank)
#   train <arm> [steps] [tag]    pool GPU (--train): rft.py train, run dir $W/runs/<tag>
#   evalol <models...>           pool GPU: rft.py evalol on the w58 and the w116 CARLA sets (2 x 2: model x input wide FOV)
#   onnx <tag> [none]            pool GPU: serving ONNX + adapter ($W/onnx/<tag>.onnx) and the port-vs-ONNX equivalence check
set -uo pipefail
cd "$(dirname "$0")/../../.."
export OP_RFT_ROOT=$DATA_DIR/runs/op_wide_ft
W=$OP_RFT_ROOT; OL=$DATA_DIR/runs/op_route_cmd/carla_pairs_s10000ol
PY=$DATA_DIR/envs/op-train/bin/python; PYC=$DATA_DIR/envs/carla/bin/python; PYJ=$DATA_DIR/envs/jevdrive/bin/python; PYO=$DATA_DIR/envs/openpilot/bin/python
RS=experiments/op_route_ft/scripts; RC=experiments/op_route_cmd/scripts
phase=$1; shift
D=$W/chain/$phase${TAG:+-$TAG}; [ "$phase" = render ] && D=$W/chain/render-$1; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
say() { echo "$(date '+%F %T') $phase: $*" | tee -a "$D/log.txt" > /dev/null; echo "$(date '+%F %T') $phase: $*" > "$D/STATUS"; }
die() { say "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
case $phase in
  render)
    first=$1; n=$2; total=$3; P=$(cat "$OL/plan.path"); out=$W/carla/render; mkdir -p "$out"
    IFS=, read -ra cl <<< "$("$PYJ" -c "import sys; from jevdrive.cl.pool import parse_cpus; print(','.join(map(str, parse_cpus(sys.argv[1]))))" "$CL_CPUS")"
    per=$(( ${#cl[@]} / n )); [ "$per" -ge 1 ] || die "too few cores (${#cl[@]}) for $n shards"
    extra=(); [ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT"); [ -n "${IDS:-}" ] && extra+=(--ids "$IDS")
    say "shards $first..$((first + n - 1)) of $total on card $CL_GPU, idx $CL_IDX.., $per cores each ${extra[*]}"
    pids=()
    for k in $(seq 0 $((n - 1))); do
      cpus=$(IFS=,; echo "${cl[*]:$((k * per)):$per}")
      $PYC $RC/carla_pairs_render_ol.py --plan "$P" --out "$out" --gpu "$CL_GPU" --idx $((CL_IDX + k)) --cpus "$cpus" --shard "$((first + k))/$total" \
        --hero --wide-focals 160 --no-pano "${extra[@]}" > "$D/render_$((first + k)).log" 2>&1 &
      pids+=($!); sleep 8
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ $rc = 0 ] || die "a render shard failed (see $D/render_*.log, $out/ERROR_*)" ;;
  pack)
    P=$(cat "$OL/plan.path"); R=$W/carla/render
    say "pack p58"; $PYJ $RC/carla_pairs_pack.py --plan "$P" --render "$R" --root "$W/carla/p58" --ids-from "$OL/packed" >> "$D/log.txt" 2>&1 || die "pack p58"
    say "pack p116"; $PYJ $RC/carla_pairs_pack.py --plan "$P" --render "$R" --root "$W/carla/p116" --ids-from "$OL/packed" --wide-key frames_wf160 \
      >> "$D/log.txt" 2>&1 || die "pack p116"
    $PYJ - "$W/carla" "$OL/packed" >> "$D/log.txt" 2>&1 <<'EOF' || die "pack check"
import sys, numpy as np
from pathlib import Path
w, ol = Path(sys.argv[1]), Path(sys.argv[2])
t = {k: np.load(p / "samples/route_carla/tab.npz", allow_pickle=True) for k, p in (("ol", ol), ("p58", w / "p58"), ("p116", w / "p116"))}
r = {k: np.load(p / "route.npz", allow_pickle=True) for k, p in (("ol", ol), ("p58", w / "p58"), ("p116", w / "p116"))}
for k in ("p58", "p116"):
    assert (t[k]["id"] == t["ol"]["id"]).all() and (t[k]["split"] == t["ol"]["split"]).all(), k
    assert (r[k]["id"] == r["ol"]["id"]).all() and np.allclose(r[k]["poly"], r["ol"]["poly"]), k
a, b = (np.load(w / k / "samples/route_carla/imgs.npy", mmap_mode="r") for k in ("p58", "p116"))
i = np.random.default_rng(0).choice(len(a), 64, replace=False)
assert (a[i, :, 0] == b[i, :, 0]).all(), "road slots differ"
print("pack check ok: poses", len(a), "road identical, wide differs on", float((a[i, :, 1] != b[i, :, 1]).mean()), "of the bytes")
EOF
    ;;
  bank)
    for w in w58 w116; do say "bank $w"; $PY $RS/rft.py bank --which $w >> "$D/log.txt" 2>&1 || die "bank $w"; done ;;
  train)
    arm=$1; steps=${2:-0}; tag=${3:-$arm-s0}
    say "train $arm steps $steps tag $tag"; $PY $RS/rft.py train --arm "$arm" --steps "$steps" --tag "$tag" --fresh > "$D/train-$tag.log" 2>&1 || die "train $tag" ;;
  evalol)
    for c in w58 w116; do say "evalol $c: $*"; $PY $RS/rft.py evalol --models "$@" --carla $c >> "$D/log.txt" 2>&1 || die "evalol $c"; done ;;
  onnx)
    t=$1; ad=$W/runs/$t/adapter.npz; [ "${2:-}" = none ] && ad=none
    say "onnx $t"; $PY $RS/route_onnx.py build --ckpt $W/runs/$t/ckpt-final.pt --adapter $ad --out $W/onnx/$t.onnx >> "$D/log.txt" 2>&1 || die "onnx $t"
    say "equivalence $t"
    $PY $RS/route_onnx.py ref --ckpt $W/runs/$t/ckpt-final.pt --adapter $ad --out $W/onnx/$t.ref.npz >> "$D/log.txt" 2>&1 || die "ref"
    $PYO $RS/route_onnx.py check --onnx $W/onnx/$t.onnx --ref $W/onnx/$t.ref.npz >> "$D/log.txt" 2>&1 || die "check" ;;
  *) die "unknown phase $phase" ;;
esac
say "done"; date '+%F %T' > "$D/DONE"
