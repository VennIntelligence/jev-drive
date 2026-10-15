#!/usr/bin/env bash
# OTHERS-TURN (plans/2026-10-15-others-turn-prereg.md): third-party drivers on the 24 held turns at 18 decisions, read with rf_follow.py.
#   bash experiments/others_turn/scripts/ot_chain.sh check     G1: asl records vs WA-JEPA's own drive records (stored zero-score rollouts), CPU
#   bash experiments/others_turn/scripts/ot_chain.sh rerun     pool fanout: ltf / dd / garage as shipped on results/turn24.txt (al_families.sh's command)
#   bash experiments/others_turn/scripts/ot_chain.sh read      views of the reruns, rf_follow (24 turns) + ol_confirm (held18 zeros / mean) vs RSG-L18-s0
# Out: $DATA_DIR/runs/others_turn/{check,rerun/<arm>,view/<arm>,follow,held18,g3.json}
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/others_turn; O=$DATA_DIR/runs/others_turn; W=$DATA_DIR/runs/alpasim_long; F=$W/fam; C=$DATA_DIR/datasets/alpasim_nuplan_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; KEY=runtime.scene_provider.trajdata; T24=$E/results/turn24.txt
mkdir -p $O
RSG=$(ls -d $DATA_DIR/runs/route_mem/read/RSG-L18-s0/c*/run | paste -sd,)
YR=$(python3 -c "import csv; print(','.join(sorted({r['run'] for r in csv.DictReader(open('$W/held/read_apy/per_scene.csv')) if r['label'] == 'YR10m10-F-s0@18'})))")
WJ=$F/wajepa0,$F/wajepa1,$F/wajepa2,$F/wajepa3
case ${1:-} in
check)
  for c in 0 1 2 3; do $P $E/scripts/ot_asl_records.py check --run $F/wajepa$c --out $O/check/wajepa$c.json || exit 1; done ;;
rerun)
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  X="$KEY.cache_location=$C/after160 $KEY.dataset.extra_params.num_timesteps_after=160"
  .venv/bin/python -m jevdrive.cl fanout --arms ltf,dd,garage --name ot-turn24 --owner others-turn --cpu 8 --ram 60 --tries 1 --priority 12 \
    --timeout-h 1 --no-check --vram 34 --log-dir "$O/rerun/{arm}/pool" \
    -- bash -c "mkdir -p $O/rerun/{arm} && exec bash experiments/alpasim/scripts/run.sh $O/rerun/{arm} {arm} --scene-list $T24 ${BASE[*]} $X" ;;
read)
  for a in ltf dd garage; do
    [[ -e $O/rerun/$a/pool/DONE ]] || { echo "rerun $a not done"; exit 1; }
    d=$(ls -d $O/rerun/$a/2*/ 2>/dev/null | tail -1); d=${d%/}; [[ -n $d ]] || d=$O/rerun/$a
    rm -rf $O/view/$a; $P $E/scripts/ot_asl_records.py view --run $d --out $O/view/$a || exit 1
  done
  $P experiments/route_follow/scripts/rf_follow.py --cover none --workers 16 --out $O/follow \
    --runs "RSG-L18-s0@18=$RSG" "YR10m10-F-s0@18=$YR" "WAJEPA@18=$WJ" "LTF@18=$O/view/ltf" "DD@18=$O/view/dd" "G0014@18=$O/view/garage" \
    --pairs WAJEPA@18:RSG-L18-s0@18 LTF@18:RSG-L18-s0@18 DD@18:RSG-L18-s0@18 G0014@18:RSG-L18-s0@18 > $O/follow.log 2>&1 || { tail $O/follow.log; exit 1; }
  $P experiments/onpol_long/scripts/ol_confirm.py --base RSG-L18-s0 --out $O/held18 \
    --runs "RSG-L18-s0@held18=$RSG" "YR10m10-F-s0@held18=$YR" "WAJEPA@held18=$WJ" "LTF@held18=$F/ltf" "DD@held18=$F/dd" "G0014@held18=$F/garage" \
    > $O/held18.log 2>&1 || { tail $O/held18.log; exit 1; }
  python3 - $O $F <<'PY'
import json, sys
from pathlib import Path
O, F = Path(sys.argv[1]), Path(sys.argv[2])
def scores(d):
    return {r["clipgt_id"]: r["score"] for r in json.loads((Path(d) / "aggregate/results-summary.json").read_text())["rollouts"]}
g3 = {}
for a, b in (("ltf", "ltf"), ("dd", "dd"), ("garage", "garage")):
    new, old = scores(O / "view" / a), scores(F / b)
    common = sorted(set(new) & set(old))
    g3[a] = dict(n=len(common), zero_agree=sum((new[s] == 0) == (old[s] == 0) for s in common),
                 mean_new=sum(new[s] for s in common) / max(len(common), 1), mean_old=sum(old[s] for s in common) / max(len(common), 1))
(O / "g3.json").write_text(json.dumps(g3, indent=1) + "\n"); print(json.dumps(g3))
PY
  ;;
*) echo "ot_chain.sh check | rerun | read" >&2; exit 2 ;;
esac
