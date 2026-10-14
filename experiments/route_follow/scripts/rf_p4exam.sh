#!/usr/bin/env bash
# P4 exam ceiling read (prereg experiments/route_follow/plans/2026-10-14-p4-exam-ceiling-prereg.md): the P4 branch family, unchanged
# (rf_p4.sh job: phase A identity, reach, phase B cells, read), on the hard_exam turn scenes, Tokyo only. READ-ONLY: no design value is
# taken from it and no row is ever built from it (the root holds NO_ROWS, rf_selfpool.py refuses it); every rollout.asl is deleted after
# its scene's read (drive.jsonl, controller CSV, metrics, read.json kept).
#   rf_p4exam.sh stage <root> <name> <N | rest>   the next N unclaimed scenes of <root>/plan.json (largest expected cell count first) on
#                                                  CARDS ("0 1") x SLOTS (2) workers; one rf_p4.sh job per scene with P4_PAR launches (1);
#                                                  per scene after its job: S1 of phase A against r281/rsg (s1.txt), rollout.asl removed;
#                                                  at the end the merged read (below). State lines go to /data/runs/route_night/CHAIN_STATUS.
#   rf_p4exam.sh read <root>                       merge -> read.json, table --ceiling -> table.md / summary.json, s1 count -> s1.txt
# Turn-gallery fill (P4 prereg Amendment 5, read-only viewing material): camera re-runs of a finished Tokyo P4 root (P4_SRC, e.g. p4_exam)
#   P4_SRC=<run root> rf_p4exam.sh vinit <vroot> <s8,s8,..>   NO_ROWS, plan.json, select.json (rf_p4.py vselect --family: identity +
#                                                  g1 / g1.25 / g1.5 raw at D40 and D0)
#   P4_SRC=<run root> rf_p4exam.sh vstage <vroot> <name> <N | rest>   the next N unclaimed scenes of select.json on CARDS x SLOTS workers,
#                                                  one rf_p4.sh video per scene (P4_PAR 1); per scene the identity's S1 against r281/rsg
#                                                  (s1.txt); at the end vread
#   P4_SRC=<run root> rf_p4exam.sh vread <vroot>   rf_p4.sh vcollect (bit-identity of every re-run vs P4_SRC, manifest.json) + S1 count
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cmd=${1:?stage|read|vinit|vstage|vread}; O=$(mkdir -p "${2:?root}" && cd "$2" && pwd); shift 2
REF=/data/runs/route_night/r281/rsg; CS=/data/runs/route_night/CHAIN_STATUS
if [[ $cmd == vinit ]]; then
  SRC=${P4_SRC:?P4_SRC = the P4 run root}
  echo "P4 turn-gallery fill (read-only viewing material, P4 prereg Amendment 5): never a training row; rf_selfpool.py refuses this root" > "$O/NO_ROWS"
  cp "$SRC/plan.json" "$O/plan.json"
  python3 "$here/rf_p4.py" vselect --family --scenes "${1:?s8 list}" --read "$SRC/read.json" --plan "$SRC/plan.json" --out "$O/select.json"; exit $?
fi
[[ -f $O/NO_ROWS && -f $O/plan.json ]] || { echo "$O: no plan.json / NO_ROWS (rf_p4.sh examplan / rf_p4exam.sh vinit first)" >&2; exit 1; }
st() { echo "$(date '+%F %T') p4exam $*" | tee -a "$O/exam_log.txt" >> "$CS"; }

worker() {  # <card> <slot> <queue>: claim scenes of the queue one by one
  local c=$1 w=$2 q=$3 sc sd t0
  while read -r sc; do
    mkdir "$O/claims/$sc" 2>/dev/null || continue
    sd=$O/$sc; t0=$(date +%s)
    CARD=$c SLOT0=$w P4_PAR=${P4_PAR:-1} bash "$here/rf_p4.sh" job "$O" "$sc"
    if [[ -f $sd/DONE ]]; then
      python3 "$here/rf_p4.py" s1 --run "$sd/${sc:7:8}-a" --ref "$REF" > "$sd/s1.txt" 2>&1
      find "$sd" -name rollout.asl -delete
      st "card $c slot $w ${sc:7:8} done in $(( $(date +%s) - t0 )) s, $(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "$sd/read.json") rollouts; $(tail -1 "$sd/s1.txt")"
    else
      st "card $c slot $w ${sc:7:8} ERROR: $(cat "$sd/ERROR" 2>/dev/null)"
    fi
  done < "$q"
}

vworker() {  # <card> <slot> <queue>: camera re-runs of the queued scenes (rf_p4.sh video), S1 of the identity against r281/rsg
  local c=$1 w=$2 q=$3 sc sd t0
  while read -r sc; do
    mkdir "$O/claims/$sc" 2>/dev/null || continue
    sd=$O/$sc; t0=$(date +%s)
    CARD=$c SLOT0=$w P4_PAR=1 bash "$here/rf_p4.sh" video "$O" "${sc:7:8}"
    if [[ -f $sd/DONE ]]; then
      for r in "$sd/${sc:7:8}"-v*/; do python3 "$here/rf_p4.py" s1 --run "${r%/}" --ref "$REF" 2>&1 | grep 'cell id:'; done > "$sd/s1.txt"
      find "$sd" -name rollout.asl -delete
      st "vfill card $c slot $w ${sc:7:8} done in $(( $(date +%s) - t0 )) s, $(ls "$sd"/*-v*/sim/rollouts/*/*/*camera_front_wide*.mp4 2>/dev/null | wc -l) mp4; $(tail -1 "$sd/s1.txt")"
    else
      st "vfill card $c slot $w ${sc:7:8} ERROR: $(cat "$sd/ERROR" 2>/dev/null)"
    fi
  done < "$q"
}

vreadall() {
  bash "$here/rf_p4.sh" vcollect "$O" > /dev/null
  { cat "$O"/clipgt-*/s1.txt 2>/dev/null; echo "S1: $(cat "$O"/clipgt-*/s1.txt 2>/dev/null | grep -c IDENTICAL) / $(cat "$O"/clipgt-*/s1.txt 2>/dev/null | grep -c .) identity re-runs bit-identical to r281/rsg"; } > "$O/s1.txt"
}

readall() {
  python3 "$here/rf_p4.py" merge --root "$O" --out "$O/read.json" &&
    python3 "$here/rf_p4.py" table --read "$O/read.json" --plan "$O/plan.json" --md "$O/table.md" --json "$O/summary.json" --ceiling > /dev/null
  { grep -h 'cell id:' "$O"/clipgt-*/s1.txt; echo "S1: $(grep -h 'cell id:' "$O"/clipgt-*/s1.txt | grep -c IDENTICAL) / $(grep -h 'cell id:' "$O"/clipgt-*/s1.txt | grep -c .) phase A sessions bit-identical to r281/rsg"; } > "$O/s1.txt"
}

case $cmd in
stage)
  name=${1:?stage name}; N=${2:?N or rest}; mkdir -p "$O/claims"
  python3 - "$O" "$N" > "$O/queue_$name.txt" <<'PY'
import json, os, sys
O, N = sys.argv[1:]
P = json.load(open(f"{O}/plan.json"))["scenes"]
# expected rollouts = 1 + 8 per distinct handover decision (log timing; the job's own reach decides)
n = {s: 1 + 8 * len({max(int(k), 17) for k in p["k_log"].values()}) for s, p in P.items()}
todo = sorted((s for s in P if not os.path.exists(f"{O}/claims/{s}")), key=lambda s: (-n[s], s))
print("\n".join(todo if N == "rest" else todo[: int(N)]))
PY
  nq=$(grep -c . "$O/queue_$name.txt"); (( nq > 0 )) || { echo "nothing left"; exit 0; }
  st "stage $name: $nq scenes on cards ${CARDS:-0 1} x ${SLOTS:-2} slots, P4_PAR ${P4_PAR:-1}, code $(git -C "$here" rev-parse --short HEAD), root $O"
  pids=()
  for c in ${CARDS:-0 1}; do for ((w = 0; w < ${SLOTS:-2}; w++)); do
    worker "$c" "$w" "$O/queue_$name.txt" & pids+=($!); sleep 20
  done; done
  wait "${pids[@]}"
  readall
  st "stage $name finished: $(grep -c . "$O/queue_$name.txt") scenes queued, $(ls "$O"/clipgt-*/DONE 2>/dev/null | wc -l) scenes DONE in the root, $(ls "$O"/clipgt-*/ERROR 2>/dev/null | wc -l) ERROR; $(tail -1 "$O/s1.txt")"
  date > "$O/STAGE_${name}_DONE" ;;
read)
  readall; cat "$O/table.md"; tail -1 "$O/s1.txt" ;;
vstage)
  [[ -n ${P4_SRC:-} && -f $O/select.json ]] || { echo "P4_SRC and $O/select.json (vinit) needed" >&2; exit 1; }
  name=${1:?stage name}; N=${2:?N or rest}; mkdir -p "$O/claims"
  python3 - "$O" "$N" > "$O/queue_$name.txt" <<'PY'
import json, os, sys
O, N = sys.argv[1:]
sel, P = json.load(open(f"{O}/select.json")), json.load(open(f"{O}/plan.json"))["scenes"]
full = {s[7:15]: s for s in P}
todo = sorted((full[s] for s in sel if not os.path.exists(f"{O}/claims/{full[s]}")), key=lambda s: (-len(sel[s[7:15]]), s))
print("\n".join(todo if N == "rest" else todo[: int(N)]))
PY
  nq=$(grep -c . "$O/queue_$name.txt"); (( nq > 0 )) || { echo "nothing left"; exit 0; }
  st "vfill stage $name: $nq scenes on cards ${CARDS:-0 1} x ${SLOTS:-2} slots, code $(git -C "$here" rev-parse --short HEAD), src $P4_SRC, root $O"
  pids=()
  for c in ${CARDS:-0 1}; do for ((w = 0; w < ${SLOTS:-2}; w++)); do
    vworker "$c" "$w" "$O/queue_$name.txt" & pids+=($!); sleep 20
  done; done
  wait "${pids[@]}"
  vreadall
  st "vfill stage $name finished: $(ls "$O"/clipgt-*/DONE 2>/dev/null | wc -l) scenes DONE, $(ls "$O"/clipgt-*/ERROR 2>/dev/null | wc -l) ERROR; $(tail -1 "$O/vcollect.txt"); $(tail -1 "$O/s1.txt")"
  date > "$O/STAGE_${name}_DONE" ;;
vread)
  vreadall; cat "$O/vcollect.txt"; tail -1 "$O/s1.txt" ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
