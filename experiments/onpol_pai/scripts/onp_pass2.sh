#!/usr/bin/env bash
# ONPOL-PAI pass 2 on the Tokyo box: the scenes are rolled once more with the simulated ego on the scene's LOGGED path, the model fed and
# recorded as in pass 1 (lib/onp_log_driver.py). One unmodified experiments/alpasim/scripts/pai_run.sh stack per card and chunk, so a run
# directory has the layout of a pass-1 run (driver/drive.jsonl + dump/, sim/rollouts/*/rollout.asl, sim/aggregate/results-summary.json).
#   onp_pass2.sh <out root> <checkpoint tag> <scene list>...          lists: plain scene ids or a pai_scenes.py tsv (stage <= 1 rows)
#   tmux new-window -d -t jev -n onp-pass2 "CARDS=0 bash /data/runs/onpol_pai/dev/scripts/onp_pass2.sh /data/runs/onpol_pai/pass2 P2H10S-F-s0 /data/runs/onpol_pai/split/train160.tsv"
# pai_run.sh mounts the driver variant from its own lib directory and experiments/alpasim/ is not this lane's to edit, so the stack is
# started from a lane-scratch clone ($CODE, cloned from $REPO on first use and then left alone) into whose experiments/alpasim/lib/ this
# script copies lib/onp_log_driver.py on every start. The clone's commit and the driver's sha256 are written to <out>/code.json.
# Env: CARDS ("0"), CONC (4), CHUNK (scenes per stack start, 40), SERVE (driver switches, default those of pass 1:
#   "-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e PAI_DUMP=100000"), CODE (/data/runs/onpol_pai/code/pass2), REPO (~/mycode/jev-drive), WAIT_S (how long
#   a worker waits for its card to show <= 1 GB before a chunk, 1800), HORIZON (seconds of log returned per call, 4.0),
#   RUNPFX (default empty; prefixes the per-chunk run dirs and so the compose / container names, as pai_eval.sh),
#   WIZ (extra wizard overrides, space separated, e.g. "controller.gains.rel_front_steering_angle_weight=20"),
#   MODE: driver (default) = the log is returned by the driver and tracked by the stack's MPC (runtime override
#           simulation_config.send_recording_ground_truth=true, nothing else changed)
#         force = pai_driver.py unchanged, the runtime itself holds the ego on the log for the whole rollout
#           (simulation_config.force_gt_duration_us=$FORCE_US, default 30 s; the controller is called but its answer is not used)
# State: <out>/{STATUS, DONE | ERROR, log.txt, eval.json, code.json}, chunk lists in <out>/chunks/, one run per chunk in
# <out>/runs/<lowercase tag>-log_<chunk> (finished chunks are skipped: rerunning resumes; an unfinished chunk is rerun whole).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
O=$(mkdir -p "$1" && cd "$1" && pwd) T=$2; shift 2
D=${DATA_DIR:-/data}; C=$D/runs/op_parity/runs/$T; L=${RUNPFX:-}${T,,}
CARDS=${CARDS:-0}; CONC=${CONC:-4}; CHUNK=${CHUNK:-40}; SERVE=${SERVE--e JEV_VCONT=1.0 -e JEV_LEAD=1 -e PAI_DUMP=100000}
CODE=${CODE:-$D/runs/onpol_pai/code/pass2}; REPO=${REPO:-$HOME/mycode/jev-drive}; MODE=${MODE:-driver}; WAIT_S=${WAIT_S:-1800}
mkdir -p "$O"/{chunks,runs}; rm -rf "$O"/{DONE,ERROR,claims}; mkdir "$O/claims"; exec > >(tee -a "$O/log.txt") 2>&1
st() { echo "$(date '+%F %T') $T: $*" | tee "$O/STATUS"; }
die() { st "ERROR $*"; echo "$*" >> "$O/ERROR"; exit 1; }
[[ -f $C/ckpt-final.pt ]] || die "no checkpoint $C"
[[ -d $CODE/.git ]] || git clone -q "$REPO" "$CODE" || die "clone of $REPO into $CODE"
run=$CODE/experiments/alpasim/scripts/pai_run.sh; drv=onp_log_driver.py
case $MODE in
  driver) cp "$here/../lib/$drv" "$CODE/experiments/alpasim/lib/$drv" || die "copy of $drv"
          over=(runtime.simulation_config.send_recording_ground_truth=true); serve="$SERVE -e ONP_HORIZON=${HORIZON:-4.0}" ;;
  force)  drv=; over=("runtime.simulation_config.force_gt_duration_us=${FORCE_US:-30000000}"); serve=$SERVE ;;
  *)      die "MODE $MODE" ;;
esac
echo "{\"mode\": \"$MODE\", \"code\": \"$CODE\", \"commit\": \"$(git -C "$CODE" rev-parse HEAD)\", \"driver\": \"${drv:-pai_driver.py}\", \"driver_sha256\": \"$(sha256sum "$CODE/experiments/alpasim/lib/${drv:-pai_driver.py}" | cut -c1-64)\", \"overrides\": \"${over[*]} ${WIZ:-}\", \"serve\": \"$serve\"}" > "$O/code.json"
if ! ls "$O"/chunks/*.txt >/dev/null 2>&1; then           # written once: a resumed run keeps its chunks
  for f in "$@"; do if head -1 "$f" | grep -q '^scene_id'; then awk -F'\t' 'NR > 1 && $8 <= 1 {print $1}' "$f"; else grep . "$f"; fi; done \
    | awk '!seen[$0]++' | split -l "$CHUNK" -d -a 3 --additional-suffix=.txt - "$O/chunks/c"
fi
n=$(cat "$O"/chunks/*.txt | wc -l); t0=$(date +%s)
st "$MODE: $n scenes in $(ls "$O"/chunks | wc -l) chunks, cards $CARDS, $CONC concurrent per card"
used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
worker() {  # card: takes chunks until none is left; never starts a stack on a card somebody else is using
  local g=$1 f c r k w
  for f in "$O"/chunks/*.txt; do
    c=$(basename "$f" .txt); r=$O/runs/${L}-log_$c
    mkdir "$O/claims/$c" 2>/dev/null || continue
    [[ -f $r/DONE ]] && continue
    for k in 1 2; do
      w=$(date +%s); until (( $(used "$g") <= 1024 )); do
        (( $(date +%s) - w > WAIT_S )) && { echo "card $g still has $(used "$g") MiB in use after $WAIT_S s: $c not started" >> "$O/ERROR"; return 1; }; sleep 30; done
      echo "$(date '+%F %T') card $g: $c ($(wc -l < "$f") scenes), attempt $k"
      GPU=$g CONC=$CONC PAI_BASEPORT=$((6400 + 100 * g)) TAG=$T DRV_PY=$drv DRV_ENV="$serve -v $C:/app/data/runs/op_parity/runs/$T:ro" \
        bash "$run" "$r" "$f" "${over[@]}" ${WIZ:-} > "$r.out" 2>&1 && break
      (( k == 2 )) && echo "chunk $c failed twice on card $g: $(cat "$r/ERROR" 2>/dev/null)" >> "$O/ERROR"
    done
  done
}
pids=(); for g in $CARDS; do worker "$g" & pids+=($!); sleep 20; done      # staggered, as pai_eval.sh: two stacks starting at once race for the docker network
wait "${pids[@]}"
left=$(for f in "$O"/chunks/*.txt; do [[ -f $O/runs/${L}-log_$(basename "$f" .txt)/DONE ]] || echo "$f"; done | wc -l)
(( left > 0 )) && ! [[ -f $O/ERROR ]] && echo "$left chunks unfinished" >> "$O/ERROR"
w=$(( $(date +%s) - t0 ))
echo "{\"tag\": \"$T\", \"mode\": \"$MODE\", \"scenes\": $n, \"cards\": \"$CARDS\", \"conc\": $CONC, \"chunk\": $CHUNK, \"wall_s\": $w, \"scenes_per_h\": $(( n * 3600 / (w + 1) )), \"chunks_left\": $left}" > "$O/eval.json"
[[ -f $O/ERROR ]] && { st "ERROR $(tail -1 "$O/ERROR")"; exit 1; }
date > "$O/DONE"; st "done: $n scenes in $w s"
