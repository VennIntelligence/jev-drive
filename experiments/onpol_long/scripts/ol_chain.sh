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
geo)     # rows of one base's rollouts (onpol_nuplan on_rows.py geo, unedited) + labels K and C (ol_labels.py build); CPU
  T=$1; N=$2; R=$W/roll/$T
  python3 - "$R" "$T" "$NCH" <<'PY' || die "rollout chunks missing"
import json, os, sys
R, T, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = {f"{T}@c{i}": [f"{R}/c{i}/run"] for i in range(n) if os.path.exists(f"{R}/c{i}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", n, "chunks"); sys.exit(len(m) != n)
PY
  $NAV $ON/scripts/on_rows.py geo --name "$N" --roll "$R" --workers ${OL_WORKERS:-48} || die "on_rows geo rc $?"
  $NAV $E/scripts/ol_labels.py build --name "$N" --workers ${OL_WORKERS:-48} || die "ol_labels build rc $?" ;;
tok)     # tokens + teacher + gate R1 (on_rows.py tok, unedited), then the pool gates (ol_labels.py gate: R1 / R2 as amended, K0); GPU
  N=$1; CTL=$2
  $TR $ON/scripts/on_rows.py tok --name "$N" --control "$CTL" || die "on_rows tok rc $?"
  $TR $E/scripts/ol_labels.py gate --name "$N" --out "$W/rows/$N" || die "pool gates failed ($W/rows/$N/gates.md): the family stops here" ;;
strip)   # after the tokens: drop the slot frames (`cur`, 98 % of a dump) from this run's own dumps; JPEG, poses, ego, mu stay
  T=$1; N=$2
  [[ -f $DATA_DIR/runs/op_parity/cache/on1_$N@warp/front.npy && -f $W/rows/$N/gates.json ]] || die "no tokens / gates for $N"
  du -sm $W/roll/$T/c*/run/driver-logs/dump | awk '{s += $1} END {print "dumps before strip:", s, "MB"}'
  $TR - "$W/roll/$T" <<'PY' || die "strip rc $?"
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
def one(f):
    z = dict(np.load(f))
    if "cur" not in z:
        return 0
    del z["cur"]
    np.savez_compressed(f + ".tmp.npz", **z)
    os.replace(f + ".tmp.npz", f)
    return 1
fs = sorted(glob.glob(f"{sys.argv[1]}/c*/run/driver-logs/dump/*.npz"))
with ProcessPoolExecutor(16) as ex:
    print("stripped", sum(ex.map(one, fs, chunksize=64)), "of", len(fs))
PY
  du -sm $W/roll/$T/c*/run/driver-logs/dump | awk '{s += $1} END {print "dumps after strip:", s, "MB"}' ;;
read1)   # one checkpoint on the held 460 scenes at 18 decisions (alpasim_long's driver path), then ot2_loop's pruning
  TAG=$1; D=$W/read/$TAG/run
  [[ -f $CK/$TAG/ckpt-final.pt ]] || die "no checkpoint $TAG"; [[ $(wc -l < $SPL/held.txt) -eq 460 ]] || die "held.txt"
  [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$TAG bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list $SPL/held.txt "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  ;;
report)  # the registered read of one family against its control and base (ol_report.py)
  F=$1; B=$( [[ $F == Y ]] && echo YR10m10-F || echo P2H10S-F ); H=$DATA_DIR/runs/alpasim_long/held/h18
  R=(); for t in ${F}SKC-L18-s0 ${F}SKC-L18-s1 ${F}NONE-L18-s0 ${F}NONE-L18-s1; do R+=("$t@18=$W/read/$t/run"); done
  for t in $B-s0 $B-s1; do [[ -f $H/$t/aggregate/results-summary.json ]] && R+=("$t@18=$H/$t"); done
  $P $E/scripts/ol_report.py --family $F --runs "${R[@]}" --out $W/report/$F || die "ol_report rc $?" ;;
submit-rest)   # everything after the rollouts of one base, chained with --after
  T=$1; F=$2; RIDS=$3; AL=${4:-}; N=L18$F; B=${T%-s0}; CTL=$( [[ $F == Y ]] && echo smoke10eY || echo smoke10e )
  [[ -f $DATA_DIR/runs/onpol_nuplan/rows/$CTL/r1.json ]] || die "no interpreted control $CTL"
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner onpol-long --name "$name" --tries 1 --priority ${PRIO:-13} --no-check ${after:+--after $after} \
            --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name" >> $W/jobs.txt; }
  g=$(sub ol-geo-$F "$RIDS" --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $E/scripts/ol_chain.sh geo $T $N) || exit 1
  k=$(sub ol-tok-$F "$g" --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $E/scripts/ol_chain.sh tok $N $CTL) || exit 1
  sub ol-strip-$F "$k" --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip $T $N > /dev/null || exit 1
  if [[ $F == Y ]]; then REC="--yr-mass 0.1 --label-bank"; SK="--agent-lam 10 --agent-on-only --shape --ho-w 3"
  else REC="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank"; SK=""; fi
  ONR="--on on1_$N --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --on-lon ol_kc"
  OT="env PYTHONPATH=$PWD/$E/lib $TR $ON/scripts/on_train.py train --exp onpol_long --init $T --steps ${OL_STEPS:-10000}"
  rids=()
  for s in 0 1; do
    a=$(sub ol-train-${F}SKC-s$s "$k" --train --vram 40 --cpu 6 --ram 60 --timeout-h 4 -- $OT $REC $SK $ONR --seed $s --tag ${F}SKC-L18-s$s) || exit 1
    c=$(sub ol-train-${F}NONE-s$s "$AL" --train --vram 40 --cpu 6 --ram 60 --timeout-h 4 -- $OT $REC --seed $s --tag ${F}NONE-L18-s$s) || exit 1
    for x in "$a ${F}SKC-L18-s$s" "$c ${F}NONE-L18-s$s"; do set -- $x
      rids+=($(sub ol-read-$2 "$1" --vram 32 --cpu 8 --ram 90 --timeout-h 3 -- bash $E/scripts/ol_chain.sh read1 $2)) || exit 1; done
  done
  sub ol-report-$F "$(IFS=,; echo "${rids[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $E/scripts/ol_chain.sh report $F > /dev/null || exit 1
  echo "submitted for $T: $(tail -n 12 $W/jobs.txt | tr '\n' ';')" ;;
*) die "unknown stage $STAGE" ;;
esac
