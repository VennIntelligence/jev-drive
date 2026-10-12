#!/usr/bin/env bash
# ROAD-PROBE chain (plans/2026-10-12-road-probe-prereg.md). Every GPU stage is a pool job. Box, repo root:
#   bash experiments/road_probe/scripts/rp_chain.sh submit-roll            -> 3 held rollout jobs of YS-L18-s1 with the frame dump (cl fanout)
# Job bodies (run by the pool): roll <chunk>.
# Driver path = onpol_long read1 (alpasim_long's 18-decision read: run.sh sh30, c0b_chain.OVERRIDES, rebuilt cache after160) + SH30_DUMP.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/road_probe; SPL=experiments/onpol_nuplan/results/split; W=$DATA_DIR/runs/road_probe; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata; NCH=3; TAG=YS-L18-s1
CL() { .venv/bin/python -m jevdrive.cl "$@"; }
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
lists() {  # the held split in NCH interleaved chunks
  local d=$W/lists; mkdir -p "$d"; [[ $(wc -l < $SPL/held.txt) -eq 460 ]] || die "held.txt"
  for i in $(seq 0 $((NCH - 1))); do awk -v i=$i -v n=$NCH '(NR - 1) % n == i' $SPL/held.txt > "$d/c$i.txt"; done; }
STAGE=${1:?stage}; shift
case $STAGE in
roll)    # one held chunk with the frame dump; asl kept only for zero / failed rollouts (as read1)
  i=$1; D=$W/roll/c$i/run; L=$W/lists/c$i.txt
  [[ -f $CK/$TAG/ckpt-final.pt ]] || die "no checkpoint $TAG"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$TAG SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "$(date '+%F %T') roll c$i done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
submit-roll)
  lists; mkdir -p $W/roll
  CL fanout --arms "$(seq -s, 0 $((NCH - 1)))" --name rp-roll --owner road-probe --vram 32 --cpu 8 --ram 60 --tries 1 --priority ${PRIO:-13} \
     --timeout-h 3 --no-check --log-dir "$W/roll/c{arm}/pool" -- bash $E/scripts/rp_chain.sh roll "{arm}" ;;
*) die "unknown stage $STAGE" ;;
esac
