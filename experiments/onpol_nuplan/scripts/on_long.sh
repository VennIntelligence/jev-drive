#!/usr/bin/env bash
# Lane ONPOL-NUPLAN round 2, item A (plans/2026-10-12-onpol2-prereg.md): the long-horizon smoke. Ten TRAIN scenes
# (results/long/smoke10_train.txt = the first 10 of on_read_loop.py's default_rng(0) permutation of train.txt, sorted), one checkpoint,
# the unchanged driver path of the lane's reads (run.sh sh30, c0b_chain.OVERRIDES), one stack at a time through the GPU pool:
#   ship     the shipped config, no window override
#   a80      + runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=80     (the override path at the shipped value)
#   b160     + ...num_timesteps_after=160                                                      (the registered wider window)
#   c160new  + ...num_timesteps_after=160 runtime.scene_provider.trajdata.cache_location=<new dir under this stage>
#            (a rebuild into a NEW cache; the shipped cache $DATA_DIR/datasets/alpasim_nuplan/nuplan_test is never written)
# then scripts/on_long_smoke.py reads gate A1 and the per-scene table into <stage dir>/smoke.json (copy it to results/long/ on the Mac).
#   scripts/tmux_run.sh on-long-smoke bash experiments/onpol_nuplan/scripts/on_long.sh smoke [<tag>]        (default tag P2H10S-F-s0)
# State: $DATA_DIR/runs/onpol_nuplan/long/smoke/{STATUS, DONE, ERROR, log.txt, runs/<arm>/}. Resumable: an arm whose pool job ended (DONE or
# ERROR in runs/<arm>/pool) is not run again. An arm that fails is recorded and the chain goes on: a failing wider window is a result.
# Declared per job (10 scenes; docs/alpasim.md: 11 GB + 0.1 GB per scene, VRAM peak 23-32 GB): --vram 32 --cpu 8 --ram 16 --timeout-h 0.5.
# Stage `held` does not exist: the smoke found no scene beyond 10 decisions (results/long/smoke.md), so item A stopped there and no held
# scene is rolled at a longer horizon by this script. The script refuses every list that is not inside results/split/train.txt.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?stage: smoke}; TAG=${2:-P2H10S-F-s0}
E=experiments/onpol_nuplan; LIST=$E/results/long/smoke10_train.txt; W=$DATA_DIR/runs/onpol_nuplan/long; O=$W/$STAGE
KEY=runtime.scene_provider.trajdata
[[ $STAGE == smoke ]] || { echo "on_long.sh: stage '$STAGE' refused: only the train-scene smoke exists (item A stopped at the smoke, results/long/smoke.md)" >&2; exit 2; }
mkdir -p "$O/runs"; exec 9> "$O/lock"; flock -n 9 || { echo "another on_long.sh $STAGE holds $O/lock" >&2; exit 1; }
rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') on-long $STAGE: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
[[ $(wc -l < "$LIST") -eq 10 ]] || die "$LIST is not 10 scenes"
[[ -z $(comm -23 <(sort "$LIST") <(sort $E/results/split/train.txt)) ]] || die "$LIST has a scene outside the train split"
[[ -z $(comm -12 <(sort "$LIST") <(sort $E/results/split/held.txt)) ]] || die "$LIST has a held scene"
[[ -f $DATA_DIR/runs/op_parity/runs/$TAG/ckpt-final.pt ]] || die "no checkpoint $TAG"
status "code $(git rev-parse --short HEAD); tag $TAG; alpasim $(git -C "$DATA_DIR/third_party/alpasim" rev-parse --short HEAD)"
source <(python3 - <<'EOF'
import sys; sys.path.insert(0, "experiments/alpasim/scripts")
import c0b_chain as B
print("BASE=(" + " ".join(B.OVERRIDES) + ")")
EOF
)
for arm in ship a80 b160 c160new; do
  D=$O/runs/$arm
  if [[ -e $D/pool/DONE || -e $D/pool/ERROR ]]; then status "$arm: already ran ($(ls "$D/pool" | grep -E '^(DONE|ERROR)$' | tr '\n' ' ')), skipped"; continue; fi
  [[ ! -e $D ]] || die "$arm: $D exists without a pool result (a job may be running: check \`cl top\`, then remove the dir)"
  case $arm in
    ship)    X=() ;;
    a80)     X=("$KEY.dataset.extra_params.num_timesteps_after=80") ;;
    b160)    X=("$KEY.dataset.extra_params.num_timesteps_after=160") ;;
    c160new) X=("$KEY.dataset.extra_params.num_timesteps_after=160" "$KEY.cache_location=$O/cache_after160") ;;
  esac
  mkdir -p "$D"
  id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-onpol --name alpasim-onpol-long-sh30 --vram 32 --cpu 8 --ram 16 --timeout-h 0.5 --tries 1 \
        --priority "${OT_PRIO:-12}" --log-dir "$D/pool" -- env SH30_TAG="$TAG" bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$LIST" "${BASE[@]}" "${X[@]}" | awk '{print $NF}') \
    || die "$arm: submit failed"
  status "$arm: submitted $id (${X[*]:-no window override})"
  until [[ -e $D/pool/DONE || -e $D/pool/ERROR ]]; do sleep 10; done
  du -sm "$D" | cut -f1 > "$D/du_mb_before_prune"
  if [[ -e $D/pool/DONE ]]; then
    "$DATA_DIR/third_party/alpasim/.venv/bin/python" experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/read_sim.pkl" --jobs 8 || die "$arm: extraction failed"
    find "$D/rollouts" -name rollout.asl -delete
    status "$arm: done"
  else
    status "$arm: pool ERROR ($(head -c 200 "$D/pool/ERROR" | tr '\n' ' ')), recorded"
  fi
done
"$DATA_DIR/third_party/alpasim/.venv/bin/python" $E/scripts/on_long_smoke.py --stage "$O" --list "$LIST" --out "$O/smoke.json" || die "on_long_smoke.py rc $?"
status "all done"; date '+%F %T' > "$O/DONE"
