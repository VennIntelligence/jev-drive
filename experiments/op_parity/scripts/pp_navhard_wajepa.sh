#!/usr/bin/env bash
# WA-JEPA's released checkpoint on requests built from our NAVSIM index (pp_navhard.py req), its own runner (wajepa_run.py, fp32 = its
# NAVSIM path), NS shard processes on the one card the pool gave this job -> <out>_preds.npz (tokens, poses: the harness pose format).
#   pp_navhard_wajepa.sh <split> <n (0 = all)> <out prefix>
set -euo pipefail
cd "$(dirname "$0")/../../.."
split=$1 n=$2 out=$3 NS=${NS:-6}
PY=$DATA_DIR/envs/op-train/bin/python
RUN=$PWD/experiments/top10/lib/top10_t2/wajepa_run.py
$PY experiments/op_parity/scripts/pp_navhard.py req --split "$split" --n "$n" --out "${out}_req.npz"
wj() { (cd "$DATA_DIR/third_party/wajepa" && PYTHONPATH=$DATA_DIR/third_party/wajepa:$DATA_DIR/third_party/navsim "$DATA_DIR/envs/wajepa/bin/python" "$@"); }
pids=()
for i in $(seq 0 $((NS - 1))); do
  wj "$RUN" "${out}_req.npz" --out "${out}.s$i.npz" --shard "$i" "$NS" --no-amp --workers 3 > "${out}.s$i.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "a WA-JEPA shard failed (${out}.s*.log)"; exit 1; }
wj "$RUN" --merge $(for i in $(seq 0 $((NS - 1))); do echo "${out}.s$i.npz"; done) --out "${out}.npz"
$PY experiments/op_parity/scripts/pp_navhard.py preds --wajepa "${out}.npz" --out "${out}_preds.npz"
