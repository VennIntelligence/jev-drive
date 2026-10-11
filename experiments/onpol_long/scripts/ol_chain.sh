#!/usr/bin/env bash
# ONPOL-LONG round 1 chain (plans/2026-10-12-ol1-prereg.md): on-policy rows at 18 decisions, hinge training, held 460 read at 18 decisions.
# Every stage is a GPU-pool job; the submit stages queue them with --after. Box, repo root:
#   bash experiments/onpol_long/scripts/ol_chain.sh submit-roll <BASE tag> <after ids,comma separated>   -> 6 rollout jobs (cl fanout)
#   bash experiments/onpol_long/scripts/ol_chain.sh submit-rest <BASE tag> <recipe Y|S> <rollout ids>    -> rows, training, read jobs
# Job bodies (run by the pool, not by hand): roll1 <BASE> <chunk>; geo <BASE>; tok <BASE>; strip <BASE>; read1 <tag>; report.
# Driver path = alpasim_long's 18-decision read: run.sh sh30, c0b_chain.OVERRIDES + the rebuilt cache after160 (num_timesteps_after 160).
# State: $DATA_DIR/runs/onpol_long/{roll/<BASE>/c<i>, rows/<name>, read/<tag>}; the pool's log dirs beside them.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/onpol_long; ON=experiments/onpol_nuplan; SPL=$ON/results/split; W=$DATA_DIR/runs/onpol_long; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata; NCH=6
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
CL() { .venv/bin/python -m jevdrive.cl "$@"; }
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
lists() {  # the train split in NCH interleaved chunks; never a held scene
  local d=$W/lists; mkdir -p "$d"
  [[ -z $(comm -12 <(sort $SPL/train.txt) <(sort $SPL/held.txt)) ]] || die "train and held overlap"
  for i in $(seq 0 $((NCH - 1))); do awk -v i=$i -v n=$NCH '(NR - 1) % n == i' $SPL/train.txt > "$d/c$i.txt.tmp"
    if [[ -f $d/c$i.txt ]]; then cmp -s "$d/c$i.txt" "$d/c$i.txt.tmp" || die "$d/c$i.txt changed"; rm "$d/c$i.txt.tmp"; else mv "$d/c$i.txt.tmp" "$d/c$i.txt"; fi
  done; }
STAGE=${1:?stage}; shift
case $STAGE in
roll1)   # one chunk: rollouts with the frame dump, then the simulator state (onpol_nuplan's on_loop.py step), then ot2_loop's pruning
  T=$1; i=$2; D=$W/roll/$T/c$i/run; L=$W/lists/c$i.txt
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ -f $C/nuplan_test/scenes_list.dill ]] || die "no cache $C"
  [[ -z $(comm -12 <(sort "$L") <(sort $SPL/held.txt)) ]] || die "$L has a held scene"
  [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  $P experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/onpol_sim.pkl" --jobs 8 || die "c1_extract rc $?"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason"):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "$(date '+%F %T') roll1 $T c$i done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
submit-roll)
  T=$1; AFTER=${2:-}; lists
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"
  mkdir -p "$W/roll/$T"
  CL fanout --arms "$(seq -s, 0 $((NCH - 1)))" --name "ol-roll-${T%%-F-s0}" --owner onpol-long --vram 32 --cpu 8 --ram 60 --tries 1 --priority ${PRIO:-13} \
     --timeout-h 3 --no-check ${AFTER:+--after $AFTER} --log-dir "$W/roll/$T/c{arm}/pool" -- bash $E/scripts/ol_chain.sh roll1 "$T" "{arm}" ;;
*) die "unknown stage $STAGE" ;;
esac
