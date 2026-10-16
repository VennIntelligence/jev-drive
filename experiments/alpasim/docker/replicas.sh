#!/usr/bin/env bash
# Several replicas of a submission image on ONE card at once, as the organisers' ec2 topology places them (8gpu_32rollouts: 16 replicas on
# GPUs 4-7 = 4 per card, 2 concurrent rollouts each): every replica with test.sh's limits and --network none, its own
# ALPASIM_CONTESTANT_REPLICA_INDEX, then probe.py inside every replica at the same time. Measures per-replica cold start, the card's peak
# memory and each replica's share of it (host pids mapped to containers by cgroup), client-side `drive` latency per replica under the
# shared card, and that all replicas report the same version (the runtime asserts it).
#   experiments/alpasim/docker/replicas.sh <image:version> <out dir> [replicas, default 4] [gpu index, default 1]
# Env: PROBE (probe.py arguments; default "--track pai --sessions 2 --rollouts 2" for a PAI image, "--sessions 2 --rollouts 8" otherwise).
# Writes <out>/{STATUS, DONE | ERROR, log.txt, result.json, probe_<i>.json, driver_<i>.log}. The card must be idle.
set -uo pipefail
IMG=$1 OUT=$2 K=${3:-4} GPU=${4:-1}
P0=jev-drv-rep-$$
mkdir -p "$OUT"; rm -f "$OUT"/{DONE,ERROR}; exec > >(tee -a "$OUT/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
cleanup() { for i in $(seq 0 $((K - 1))); do docker logs "$P0-$i" > "$OUT/driver_$i.log" 2>&1; docker rm -f "$P0-$i" >/dev/null 2>&1; done; }
die() { st "ERROR $*"; echo "$*" > "$OUT/ERROR"; touch "$OUT/.stop"; cleanup; exit 1; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"; }
fam=$(docker image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^JEV_DRIVER=//p')
if [[ $fam == pai ]]; then P=${PROBE:-"--track pai --sessions 2 --rollouts 2"}; else P=${PROBE:-"--sessions 2 --rollouts 8"}; fi
base=$(used); st "$K replicas of $IMG on GPU $GPU (card holds $base MiB), probe $P in each"
t0=$(date +%s.%N)
for i in $(seq 0 $((K - 1))); do
  docker run -d --name "$P0-$i" --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8 \
    --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network none --gpus "device=$GPU" \
    -e ALPASIM_CONTESTANT_REPLICA_INDEX=$i -e ALPASIM_CONTESTANT_REPLICAS=$K "$IMG" >/dev/null || die "docker run $i"
done
cold=(); got=0                                         # indexed by replica; got = how many are listening
while (( got < K )); do
  for i in $(seq 0 $((K - 1))); do
    [[ -n ${cold[$i]:-} ]] && continue
    [[ $(docker inspect -f '{{.State.Running}}' "$P0-$i") == true ]] || die "replica $i exited before listening"
    docker logs "$P0-$i" 2>&1 | grep -q "listening on" && { cold[$i]=$(echo "$(date +%s.%N) - $t0" | bc); got=$((got + 1)); }
  done
  (( $(date +%s) - ${t0%.*} > 600 )) && die "not all listening after 600 s"; sleep 0.3
done
idle=$(used); st "all $K listening (cold start ${cold[*]} s), card $idle MiB"
( peak=0; declare -A per
  while [[ ! -f $OUT/.stop ]]; do
    u=$(used); (( u > peak )) && peak=$u
    while IFS=', ' read -r pid mem; do
      for i in $(seq 0 $((K - 1))); do
        id=$(docker inspect -f '{{.Id}}' "$P0-$i" 2>/dev/null | cut -c1-12)
        [[ -n $id ]] && grep -q "$id" "/proc/$pid/cgroup" 2>/dev/null && (( mem > ${per[$i]:-0} )) && per[$i]=$mem
      done
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU")
    echo "$peak $(for i in $(seq 0 $((K - 1))); do printf '%s,' "${per[$i]:-0}"; done)" > "$OUT/.peak"; sleep 0.5
  done ) & sampler=$!
pids=(); for i in $(seq 0 $((K - 1))); do docker exec "$P0-$i" python /app/probe.py $P > "$OUT/probe_$i.json" & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
touch "$OUT/.stop"; wait "$sampler"; read -r peak per < "$OUT/.peak"; rm -f "$OUT"/.stop "$OUT"/.peak
cleanup
(( fail == 0 )) || die "a probe failed, see probe_*.json / driver_*.log"
python3 - "$OUT" "$IMG" "$K" "$base" "$idle" "$peak" "$per" "$GPU" "$(for i in $(seq 0 $((K - 1))); do printf '%s,' "${cold[$i]}"; done)" <<'PY'
import json, subprocess, sys
out, img, k, base, idle, peak, per, gpu, cold = sys.argv[1:]
k = int(k); pr = [json.load(open(f"{out}/probe_{i}.json")) for i in range(k)]
logs = [open(f"{out}/driver_{i}.log").read() for i in range(k)]
res = {"image": img, "replicas": k, "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", gpu], text=True).strip(),
       "cold_start_s": [round(float(c), 1) for c in cold.strip(",").split(",")], "card_mib": {"before": int(base), "idle": int(idle), "peak": int(peak)},
       "replica_peak_mib": [int(x) for x in per.strip(",").split(",")], "versions": sorted({p["version"] for p in pr}),
       "drive_ms": [p["drive_ms"] for p in pr], "drives": sum(p["drives"] for p in pr), "tracebacks": sum(l.count("Traceback") for l in logs)}
res["pass"] = bool(len(res["versions"]) == 1 and res["tracebacks"] == 0 and max(res["replica_peak_mib"]) < 16 * 1024)
json.dump(res, open(f"{out}/result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PY
[[ $(python3 -c "import json; print(json.load(open('$OUT/result.json'))['pass'])") == True ]] || die "a check failed, see result.json"
date > "$OUT/DONE"; st "done"
