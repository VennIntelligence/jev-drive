#!/usr/bin/env bash
# Live-container audit of the YR10m10-F-s0 image under the test.sh flags (read-only, --network none, caps dropped, uid 10001).
set -uo pipefail
I=$1 G=${2:-1} O=/data/runs/alpasim/tokyo_image/yr10m10_chain/live_audit; mkdir -p $O; N=jev-yr-audit
docker rm -f $N >/dev/null 2>&1
t0=$(date +%s.%N)
docker run -d --name $N --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network none --gpus device=$G \
  -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=1 $I >/dev/null
until docker logs $N 2>&1 | grep -q "listening on"; do sleep 0.2; done
echo "cold_start_s $(echo "$(date +%s.%N) - $t0" | bc)"
docker exec -i $N python - <<PY
import os, socket, json, glob
pid = next(p.split("/")[2] for p in glob.glob("/proc/[0-9]*/cmdline") if b"serve.py" in open(p,"rb").read() or b"sh30_driver" in open(p,"rb").read())
env = dict(kv.split("=",1) for kv in open(f"/proc/{pid}/environ").read().split("\0") if "=" in kv)
st = dict(l.split(":",1) for l in open(f"/proc/{pid}/status"))
print("server pid", pid, "uid", st["Uid"].split(), "CapEff", st["CapEff"].strip(), "NoNewPrivs", st.get("NoNewPrivs","?").strip())
print("server env:", json.dumps({k: env[k] for k in sorted(env) if k.startswith(("JEV_","SH30_","AP2_","ALPASIM_","PAI_"))}))
for p in ("/app/x", "/x", "/app/data/runs/op_parity/runs/YR10m10-F-s0/x", "/run/x", "/tmp/x"):
    try: open(p,"w").write("1"); print("write", p, "OK"); os.remove(p)
    except OSError as e: print("write", p, "refused:", e.strerror)
print("interfaces", sorted(os.listdir("/sys/class/net")))
for h in (("1.1.1.1",443),("pypi.org",443)):
    try: socket.create_connection(h, timeout=5); print("connect", h, "OK (FAIL)")
    except OSError as e: print("connect", h, "refused:", e)
import grpc
from alpasim_grpc.v0 import egodriver_pb2_grpc, common_pb2
ch = grpc.insecure_channel("127.0.0.1:6789"); grpc.channel_ready_future(ch).result(timeout=30)
v = egodriver_pb2_grpc.EgodriverServiceStub(ch).get_version(common_pb2.Empty(), timeout=10)
print("get_version:", str(v).replace("\n"," "))
PY
docker exec $N python /app/probe.py --sessions 2 --rollouts 2
docker exec $N sh -c "tail -3 /tmp/alpasim-driver/drive.jsonl | cut -c1-300; df -B1 --output=used /tmp /run"
docker logs $N 2>&1 | grep -E "serving:|ready in|YR10m10|Traceback|Error" | head
echo "docker diff (outside runtime injections):"; docker diff $N | grep -vE "nvidia|libcuda|docker-init|ld\.so\.cache|ldconfig" 
docker stop -t 20 $N >/dev/null; echo "exit $(docker inspect -f {{.State.ExitCode}} $N)"; docker rm $N >/dev/null
