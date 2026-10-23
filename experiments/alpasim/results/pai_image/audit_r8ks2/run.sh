#!/usr/bin/env bash
# PAI probe image R8K-L18-s2 (2026-10-23): build from b251b657 (the code of the box's PAI 281 reads round8/s2/pai281/R8K-L18-s2-ag{,H}) with
# closure.txt of 95aaf308 (+ jevdrive/tokens.py), then a CPU-only start, test.sh + replicas.sh on card 0, and smokes against the box reads:
# card 0 Harmonizer off c000 / c001 / c007, card 1 Harmonizer on c007 (pai_smoke.sh HARMONIZER=1). Gains 10, the image as built (no driver env).
set -uo pipefail
A=/data/runs/alpasim/pai_image/audit_r8ks2; WB=$A/wt_build; WT=$A/wt_tools; P=/data/runs/alpasim/pai_image; N=$P/native
W=controller.gains.rel_front_steering_angle_weight=10; D=$WT/experiments/alpasim/docker
st() { echo "$(date '+%F %T') $*" | tee -a $A/stages.txt; }
fail() { st "ERROR $*"; echo "$*" > $A/ERROR; exit 1; }
rm -f $A/DONE $A/ERROR
st "build in $WB ($(git -C $WB rev-parse --short HEAD) + closure.txt of 95aaf308)"
(cd $WB && env DATA_DIR=/data TRACK=pai TAG=R8K-L18-s2 EXTRA=R8K-L18-s2 ACC=1 GAP=0.7,2.7 OUT=$A/build bash experiments/alpasim/docker/build.sh) > $A/build.log 2>&1 || fail build
IMG=$(cat $A/build/IMAGE); echo "$IMG" > $A/IMAGE; st "built $IMG $(cat $A/build/image.txt)"

st "CPU-only start (no --gpus, test.sh flags, --network none)"
C=jev-cpu-start-$$; t0=$(date +%s)
docker run -d --name $C --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 --network none \
  -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=1 "$IMG" > /dev/null
r=timeout; while (( $(date +%s) - t0 < 300 )); do
  docker logs $C 2>&1 | grep -q "listening on" && { r=listening; break; }
  [[ $(docker inspect -f '{{.State.Running}}' $C) == true ]] || { r=exited; break; }; sleep 1; done
ts=$(( $(date +%s) - t0 )); docker stop -t 30 $C > /dev/null; ec=$(docker inspect -f '{{.State.ExitCode}}' $C); docker logs $C > $A/cpu_start.log 2>&1; docker rm $C > /dev/null
echo "{\"result\": \"$r\", \"seconds\": $ts, \"exit_code_after_sigterm\": $ec}" > $A/cpu_start.json; st "CPU-only start: $(cat $A/cpu_start.json)"

while pgrep -f "rsync -a autodl:" > /dev/null; do sleep 30; done   # the native reads are still being copied
for h in off on; do for c in c000 c001 c007; do [[ -s $N/r8ks2_${h}_$c/aggregate/results-summary.json && -s $N/r8ks2_${h}_$c/driver-logs/drive.jsonl ]] || fail "native $h $c missing"; done; done

smoke() {  # smoke <gpu> <port> <harm 0|1> <chunk>
  local o=$A/smoke_$([[ $3 == 1 ]] && echo on || echo off)_$4 n=r8ks2_$([[ $3 == 1 ]] && echo on || echo off)_$4
  st "smoke $o (GPU $1, HARMONIZER=$3)"
  env DATA_DIR=/data GPU=$1 PAI_BASEPORT=$2 HARMONIZER=$3 WIZ=$W bash $D/pai_smoke.sh "$IMG" $o $P/lists/$4.txt $N/$n/aggregate/results-summary.json > $o.out 2>&1 \
    || { st "smoke $o FAILED"; return 1; }
  python3 $WT/experiments/alpasim/scripts/pai_image_cmp.py --a image=$o --b native=$N/$n --out $o/cmp_box.md > $o.cmp.out 2>&1; st "smoke $o done: $(sed -n 3p $o/compare.md)"; }
( smoke 1 6700 1 c007; echo $? > $A/card1.rc ) &
( st "test.sh card 0"; bash $D/test.sh "$IMG" $A/test "" 0 > $A/test.out 2>&1; echo "test rc=$?" | tee -a $A/stages.txt
  st "replicas.sh card 0"; bash $D/replicas.sh "$IMG" $A/replicas 4 0 > $A/replicas.out 2>&1; echo "replicas rc=$?" | tee -a $A/stages.txt
  r=0; for c in c000 c001 c007; do smoke 0 6400 0 $c || r=1; done; echo $r > $A/card0.rc ) &
wait
st "all done: card0 rc $(cat $A/card0.rc) card1 rc $(cat $A/card1.rc)"; date > $A/DONE
