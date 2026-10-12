#!/usr/bin/env bash
# ALPASIM-LONG amendment 3: reference driver families on the held 460 at 18 decisions (cache after160), against their stored 10-decision runs.
#   scripts/tmux_run.sh al-fam bash experiments/alpasim_long/scripts/al_families.sh
# Jobs: `cl fanout` ltf / dd / garage (one job each, 460 scenes) and WA-JEPA in 4 chunks (results/held_chunk{0..3}.txt). State and read:
# $DATA_DIR/runs/alpasim_long/fam/{STATUS,DONE,ERROR,log.txt, <arm>/ run dirs, read/{per_scene.csv,report.md,report.json}}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/alpasim_long; W=$DATA_DIR/runs/alpasim_long; O=$W/fam; C=$DATA_DIR/datasets/alpasim_nuplan_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; KEY=runtime.scene_provider.trajdata; HELD=experiments/onpol_nuplan/results/split/held.txt
mkdir -p "$O"; exec 9> "$O/lock"; flock -n 9 || { echo "another al_families.sh holds $O/lock" >&2; exit 1; }
rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') al-fam: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD)"
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
X="$KEY.cache_location=$C/after160 $KEY.dataset.extra_params.num_timesteps_after=160"
common=(--owner alpasim-long --cpu 8 --ram 60 --tries 1 --priority 12 --timeout-h 3 --no-check)
if [[ ! -e $O/submitted ]]; then
  .venv/bin/python -m jevdrive.cl fanout --arms ltf,dd,garage --name al-f18 "${common[@]}" --vram 34 --log-dir "$O/{arm}/pool" \
    -- bash -c "mkdir -p $O/{arm} && exec bash experiments/alpasim/scripts/run.sh $O/{arm} {arm} --scene-list $HELD ${BASE[*]} $X" || die "fanout families"
  .venv/bin/python -m jevdrive.cl fanout --arms 0,1,2,3 --name al-f18-wajepa "${common[@]}" --vram 36 --log-dir "$O/wajepa{arm}/pool" \
    -- bash -c "mkdir -p $O/wajepa{arm} && WAJ_AMP=0 exec bash experiments/alpasim/scripts/run.sh $O/wajepa{arm} wajepa --scene-list $E/results/held_chunk{arm}.txt ${BASE[*]} $X" || die "fanout wajepa"
  touch "$O/submitted"; status "submitted 7 jobs"
fi
for d in ltf dd garage wajepa0 wajepa1 wajepa2 wajepa3; do
  until [[ -e $O/$d/pool/DONE || -e $O/$d/pool/ERROR ]]; do sleep 30; done; status "$d: $(ls $O/$d/pool | grep -E '^(DONE|ERROR)$')"
done
A=$DATA_DIR/runs/alpasim_proxy/anchors
old() { python3 -c "import csv,sys; print(','.join(sorted({r['run'] for r in csv.DictReader(open(sys.argv[1]))})))" "$1"; }
R=("LTF@10=$(old $E/../alpasim_proxy/results/anchors/ltf.csv)" "DD@10=$(old $E/../alpasim_proxy/results/anchors/dd.csv)"
   "G0014@10=$(old $E/../alpasim_proxy/results/anchors/g0014.csv)"
   "WAJEPA@10=$(python3 -c "import json; print(','.join(json.load(open('$DATA_DIR/runs/alpasim/c0c/manifest.json'))['WA-JEPA (reference)']))")"
   "LTF@18=$O/ltf" "DD@18=$O/dd" "G0014@18=$O/garage" "WAJEPA@18=$O/wajepa0,$O/wajepa1,$O/wajepa2,$O/wajepa3")
for lab in YR10m10-F-s0 APY10m10-AB-s0; do R+=($(python3 - "$W/held/read_apy/per_scene.csv" "$lab" <<'PY'
import csv, sys
runs = {}
for r in csv.DictReader(open(sys.argv[1])):
    if r["label"].startswith(sys.argv[2] + "@"): runs.setdefault(r["label"], set()).add(r["run"])
print(" ".join(f"{k}={','.join(sorted(v))}" for k, v in sorted(runs.items())))
PY
)); done
$P $E/scripts/long_read.py collect --runs "${R[@]}" --scenes "$HELD" --out "$O/read" --jobs 32 || die "collect rc $?"
$P $E/scripts/long_read.py report --in "$O/read" --out "$O/read/report.md" --json "$O/read/report.json" || die "report rc $?"
du -sm "$O" > "$O/du_mb_before_prune"
python3 - "$O/read/per_scene.csv" "$O" <<'PY'
import csv, glob, os, sys
for r in csv.DictReader(open(sys.argv[1])):
    if r["run"].startswith(sys.argv[2]) and float(r["score"] or 0) > 0:
        for f in glob.glob(f"{r['run']}/rollouts/{r['scene']}/*/rollout.asl"): os.remove(f)
PY
status "done"; date '+%F %T' > "$O/DONE"
