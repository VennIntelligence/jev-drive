#!/usr/bin/env bash
# Constraint tests of a built submission image, one driver per run, under the organisers' published limits (e2e_challenge/README.md at
# 0bb4c4b) and the hardening flags of their run_local_container.sh, plus no network at all:
#   read-only root, /tmp 2 GiB and /run 64 MiB tmpfs, --network none, all capabilities dropped, uid 10001, 8 cpus, 32 GiB RAM, one GPU.
# Measures: cold start (docker run -> "listening"), peak GPU memory of the card while 2 concurrent synthetic rollouts run (probe.py inside
# the container), client- and server-side `drive` latency, /tmp and /run high-water marks, files changed outside them (docker diff).
#   experiments/alpasim/docker/test.sh <image:version> <out dir> [JEV_TAG] [gpu index]
# Writes <out>/{STATUS, DONE | ERROR, log.txt, result.json, probe.json, drive.jsonl, driver.log, cmd.txt}. The card must be idle.
set -uo pipefail
IMG=$1 OUT=$2 TAG=${3:-} GPU=${4:-0}
N=jev-drv-test-$$
mkdir -p "$OUT"; rm -f "$OUT"/{DONE,ERROR}; exec > >(tee -a "$OUT/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$OUT/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$OUT/ERROR"; docker inspect "$N" >/dev/null 2>&1 && { docker logs "$N" > "$OUT/driver.log" 2>&1; docker rm -f "$N" >/dev/null; }; exit 1; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"; }
run=(docker run -d --name "$N" --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8
     --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network none --gpus "device=$GPU"
     -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=1)
[[ -n $TAG ]] && run+=(-e "JEV_TAG=$TAG")
run+=("$IMG")
printf '%q ' "${run[@]}" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"

base=$(used); st "start $IMG ${TAG:-<image default>} on GPU $GPU (card holds $base MiB)"
t0=$(date +%s.%N)
"${run[@]}" >/dev/null || die "docker run"
until docker logs "$N" 2>&1 | grep -q "listening on"; do
  [[ $(docker inspect -f '{{.State.Running}}' "$N") == true ]] || die "driver exited before listening"
  (( $(date +%s) - ${t0%.*} > 600 )) && die "not listening after 600 s"
  sleep 0.2
done
cold=$(echo "$(date +%s.%N) - $t0" | bc); idle=$(used); st "listening after $cold s, card $idle MiB"

( peak=0; tmp=0; runm=0
  while docker inspect "$N" >/dev/null 2>&1 && [[ ! -f $OUT/.stop ]]; do
    u=$(used); (( u > peak )) && peak=$u
    read -r a b < <(docker exec "$N" sh -c 'df -B1 --output=used /tmp /run | tail -2 | tr "\n" " "' 2>/dev/null); (( ${a:-0} > tmp )) && tmp=$a; (( ${b:-0} > runm )) && runm=$b
    echo "$peak $tmp $runm" > "$OUT/.peak"; sleep 0.5
  done ) & sampler=$!
st "probe: 2 concurrent rollouts x 8, as fast as the driver answers"
P="--sessions 2 --rollouts 8"
docker exec "$N" python /app/probe.py $P > "$OUT/probe.json" || die "probe failed"
cat "$OUT/probe.json"
docker exec "$N" cat /tmp/alpasim-driver/drive.jsonl > "$OUT/drive.jsonl"
docker exec "$N" sh -c 'du -ab /tmp /run 2>/dev/null | sort -n | tail -15' > "$OUT/tmp_files.txt"
touch "$OUT/.stop"; wait "$sampler"; read -r peak tmp runm < "$OUT/.peak"; rm -f "$OUT"/.stop "$OUT"/.peak
docker diff "$N" > "$OUT/docker_diff.txt"
docker logs "$N" > "$OUT/driver.log" 2>&1
t1=$(date +%s.%N); docker stop -t 20 "$N" >/dev/null; stop=$(echo "$(date +%s.%N) - $t1" | bc); rc=$(docker inspect -f '{{.State.ExitCode}}' "$N"); docker rm "$N" >/dev/null
python3 - "$OUT" "$IMG" "${TAG:-default}" "$cold" "$base" "$idle" "$peak" "$tmp" "$runm" "$stop" "$rc" "$GPU" <<'PY'
import json, re, subprocess, sys
out, img, tag, cold, base, idle, peak, tmp, runm, stop, rc, gpu = sys.argv[1:]
rows = [json.loads(l) for l in open(f"{out}/drive.jsonl")]
dr = [r["ms"] for r in rows if r.get("kind") == "drive"]
pct = lambda v, q: round(sorted(v)[min(len(v) - 1, int(q * len(v)))], 1)
srv = {k: {"p50": pct([d[k] for d in dr], .5), "p90": pct([d[k] for d in dr], .9), "p99": pct([d[k] for d in dr], .99), "max": round(max(d[k] for d in dr), 1)}
       for k in ("total", "wait", "frames", "encode", "policy")}
log = open(f"{out}/driver.log").read()
# docker diff under --read-only lists what the runtime injects (NVIDIA libraries and tools, docker-init, the refreshed ld.so cache) and
# their parent directories; anything else would be a write of ours outside /tmp and /run.
paths = [l.split(" ", 1) for l in open(f"{out}/docker_diff.txt").read().splitlines()]
changed = [p for k, p in paths if not re.search(r"nvidia|libcuda|docker-init|ld\.so\.cache|ldconfig", p) and not any(q.startswith(p + "/") for _, q in paths)]
res = {"image": img, "tag": tag, "image_bytes": int(subprocess.check_output(["docker", "image", "inspect", img, "--format", "{{.Size}}"])),
       "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", gpu], text=True).strip(),
       "cold_start_s": round(float(cold), 1), "gpu_mib": {"before": int(base), "idle": int(idle), "peak": int(peak), "driver_peak": int(peak) - int(base)},
       "tmp_peak_bytes": int(tmp), "run_peak_bytes": int(runm), "docker_diff_lines": len(changed),
       "changed_outside_tmp": changed, "read_only_errors": log.count("Read-only file system"), "tracebacks": log.count("Traceback"), "stop_s": round(float(stop), 1), "exit_code": int(rc),
       "probe": json.load(open(f"{out}/probe.json")), "server_ms": srv, "drives": len(dr),
       "closed": sum(r.get("kind") == "close" for r in rows), "inference_errors": sum(r.get("inference_error", 0) for r in rows if r.get("kind") == "close")}
res["pass"] = bool(res["read_only_errors"] == 0 and res["tracebacks"] == 0 and res["docker_diff_lines"] == 0 and res["inference_errors"] == 0
                   and res["gpu_mib"]["driver_peak"] < 16 * 1024 and res["tmp_peak_bytes"] < 2 * 2**30 and res["run_peak_bytes"] < 64 * 2**20
                   and res["image_bytes"] < 40 * 2**30 and res["drives"] == res["probe"]["drives"])
json.dump(res, open(f"{out}/result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PY
[[ $(python3 -c "import json,sys; print(json.load(open('$OUT/result.json'))['pass'])") == True ]] || die "a constraint failed, see result.json"
date > "$OUT/DONE"; st "done"
