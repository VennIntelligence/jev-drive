#!/usr/bin/env bash
# Run one pending full W seed through the per-seed guard; launch with tmux_run.sh.
set -euo pipefail
[[ ( $# == 3 || $# == 5 ) && $1 =~ ^[012]$ && $2 =~ ^[0-6]$ && $3 =~ ^[0-9,-]+$ ]] || { echo 'usage: nq4_w_seed.sh <seed 0-2> <gpu 0-6> <cpus> [--timeout-s seconds]' >&2; exit 2; }
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
seed=$1 gpu=$2 cpus=$3
limit=3000
if [[ $# == 5 ]]; then
    [[ $4 == --timeout-s && $5 =~ ^[1-9][0-9]*$ ]] || exit 2
    limit=$5
fi
out=$DATA_DIR/runs/nq4/cx/w-seed$seed
mkdir -p "$out"
exec > >(tee -a "$out/log.txt") 2>&1
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '%s\n' "$$" > "$out/wrapper.pid"
event() { printf '{"t":%s,"kind":"%s","seed":%s,"gpu":%s}\n' "$(date +%s)" "$1" "$seed" "$gpu" >> "$out/events.jsonl"; }
event start
finish() {
    local rc=$?
    if (( rc != 0 )); then
        printf 'W seed %s failed with exit code %s; see log.txt\n' "$seed" "$rc" > "$out/ERROR"
        event failed
    fi
}
trap finish EXIT
echo "[$(date '+%F %T %Z')] W seed $seed GPU $gpu cores $cpus wrapper PID $$ timeout ${limit}s"
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
(( free >= 20000 )) || { echo "GPU $gpu has only $free MB free; need 20000 MB"; exit 1; }
timeout --kill-after=60 "$limit" taskset -c "$cpus" .venv/bin/python -m jevdrive.nq4_w run --seed "$seed" &
pid=$!
printf '%s\n' "$pid" > "$out/pid"
echo "[$(date '+%F %T %Z')] W seed $seed timeout PID $pid"
wait "$pid"
.venv/bin/python - "$seed" <<'PY'
import json
import sys
from jevdrive.common import data_dir
from jevdrive.nq4_w import CFG, wdir
from jevdrive.nq4_w_guard import valid_outputs, signature_for
seed = int(sys.argv[1])
root = data_dir() / "runs/nq4/w"
record = json.loads((root / "seed-locks" / f"seed{seed}" / "complete.json").read_text())
assert record["seed"] == seed and record["signature"] == signature_for(CFG, CFG["steps"], [wdir("meta.parquet"), wdir("z.npy")])
assert valid_outputs(record["result"]["dir"], CFG["steps"])
chain = root / "chain"
chain.mkdir(parents=True, exist_ok=True)
target = chain / f"seed{seed}.done"
temporary = target.with_suffix(".done.cx.tmp")
temporary.write_text(f"validated guarded completion: {record['result']['dir']}\n")
temporary.replace(target)
PY
event end
rm -f "$out/ERROR"
trap - EXIT
