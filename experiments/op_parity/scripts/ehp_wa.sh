#!/usr/bin/env bash
# WA-JEPA plans for the ego-history probe: requests for the selected navtest tokens x variants (ehp.py wareq), its own runner (fp32, batch 1).
# A shard job runs shards [S0, S0 + NS) of NT total on the card the pool gave it:  ehp_wa.sh shards <S0> <NS> <NT>   (several jobs = several cards)
# then one merge job:  ehp_wa.sh merge <NT>   ->  $DATA_DIR/runs/op_parity/ehp/wa.npz (keys "token|variant", traj (n, 8, 3)).  Needs `ehp.py select` first.
set -euo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/ehp
PY=$DATA_DIR/envs/op-train/bin/python
RUN=$PWD/experiments/top10/lib/top10_t2/wajepa_run.py
wj() { (cd "$DATA_DIR/third_party/wajepa" && PYTHONPATH=$DATA_DIR/third_party/wajepa:$DATA_DIR/third_party/navsim "$DATA_DIR/envs/wajepa/bin/python" "$@"); }
case $1 in
shards)
  S0=$2 NS=$3 NT=$4
  [[ -f $O/sets.csv ]] || { echo "run ehp.py select first"; exit 1; }
  [[ -f $O/wa_full_req.npz ]] || $PY experiments/op_parity/scripts/pp_navhard.py req --split navtest --n 0 --out "$O/wa_full_req.npz"
  [[ -f $O/wa_req.npz ]] || $PY experiments/op_parity/scripts/ehp.py wareq --full "$O/wa_full_req.npz" --out "$O/wa_req.npz"
  pids=()
  for i in $(seq "$S0" $((S0 + NS - 1))); do
    wj "$RUN" "$O/wa_req.npz" --out "$O/wa.s$i.npz" --shard "$i" "$NT" --no-amp --workers 2 > "$O/wa.s$i.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || { echo "a WA-JEPA shard failed ($O/wa.s*.log)"; exit 1; } ;;
merge)
  NT=$2
  wj "$RUN" --merge $(for i in $(seq 0 $((NT - 1))); do echo "$O/wa.s$i.npz"; done) --out "$O/wa.npz" ;;
esac
echo done
