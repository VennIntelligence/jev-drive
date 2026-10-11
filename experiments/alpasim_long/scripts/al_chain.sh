#!/usr/bin/env bash
# ALPASIM-LONG chain (plans/2026-10-12-long-prereg.md). Box, from the repo root, in tmux:
#   scripts/tmux_run.sh al-<stage> bash experiments/alpasim_long/scripts/al_chain.sh <stage>
# stages (each stops at its gate; state in $DATA_DIR/runs/alpasim_long/<stage>/{STATUS,DONE,ERROR,log.txt}):
#   cache    build caches after80 and after160 from the raw logs (CPU; build_cache.py), then gate G0 (after80 against the shipped cache)
#   smoke    smoke10 (10 train scenes), YR10m10-F-s0, three pool jobs at once: ship (shipped cache), a80 (cache after80), b160 (cache
#            after160); then onpol_nuplan's on_long_smoke.py (its shipped_window part = G1; arm b160's rollouts / errors / decisions = G2)
#            and long_read.py collect
#   held     the read: `cl fanout` of the six checkpoints on held.txt at 18 decisions (cache after160) plus YR10m10-F s0 s1 at 10 decisions
#            (shipped cache); collect + report as the fan-outs' --collect step. Refuses to start unless smoke/DONE exists.
# Driver path of every run = the lane's reads: run.sh sh30, c0b_chain.OVERRIDES; added only cache_location and num_timesteps_after.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?stage: cache | smoke | held}
E=experiments/alpasim_long; W=$DATA_DIR/runs/alpasim_long; O=$W/$STAGE; C=$DATA_DIR/datasets/alpasim_nuplan_long
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; KEY=runtime.scene_provider.trajdata; SPL=experiments/onpol_nuplan/results/split
mkdir -p "$O"; exec 9> "$O/lock"; flock -n 9 || { echo "another al_chain.sh $STAGE holds $O/lock" >&2; exit 1; }
rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') al-$STAGE: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD); alpasim $(git -C "$DATA_DIR/third_party/alpasim" rev-parse --short HEAD); free $(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -d ' ')"
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
win() { echo "$KEY.cache_location=$C/after$1" "$KEY.dataset.extra_params.num_timesteps_after=$1"; }
CL() { .venv/bin/python -m jevdrive.cl "$@"; }

case $STAGE in
cache)
  n=$(ls "$DATA_DIR/datasets/nuplan_raw/test" | grep -c '\.db$'); [[ $n -eq 136 ]] || die "raw logs: $n of 136 (fetch_logs.sh not finished)"
  for N in 80 160; do
    if [[ -f $C/after$N/nuplan_test/scenes_list.dill ]]; then status "after$N exists, skipped"; continue; fi
    status "building after$N"
    PYTHONPATH=$DATA_DIR/envs/nuplan_devkit_overlay $P $E/scripts/build_cache.py build --after $N --out "$C/after$N" --workers ${AL_WORKERS:-32} \
      > "$O/build_after$N.log" 2>&1 || die "build after$N rc $? ($O/build_after$N.log)"
    grep -c "not found, skipping" "$O/build_after$N.log" | xargs -I{} status "after$N built; missing-db warnings: {}"
  done
  $P $E/scripts/build_cache.py compare --a "$C/after80" --b "$DATA_DIR/datasets/alpasim_nuplan" --out "$O/g0_compare.json" || die "compare rc $?"
  $P - "$O/g0_compare.json" <<'PY' || die "gate G0 failed (g0_compare.json)"
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
ok = s["scenes"] == 1491 and s["errors"] == 0 and s["exact"] == 1491
print("G0", "PASS" if ok else "FAIL", {k: s[k] for k in ("scenes", "errors", "exact", "prefix_equal", "len_pairs")})
sys.exit(0 if ok else 1)
PY
  $P $E/scripts/build_cache.py compare --a "$C/after160" --b "$C/after80" --out "$O/a160_vs_a80.json" > /dev/null || die "compare 160 rc $?" ;;
smoke)
  [[ -f $W/cache/DONE ]] || die "cache stage not done"
  LIST=experiments/onpol_nuplan/results/long/smoke10_train.txt; TAG=YR10m10-F-s0
  [[ -z $(comm -12 <(sort "$LIST") <(sort $SPL/held.txt)) ]] || die "$LIST has a held scene"
  for arm in ship a80 b160; do
    D=$O/runs/$arm
    if [[ -e $D/pool/DONE || -e $D/pool/ERROR ]]; then status "$arm already ran"; continue; fi
    [[ ! -e $D ]] || die "$arm: $D exists without a pool result"
    case $arm in ship) X=() ;; a80) X=($(win 80)) ;; b160) X=($(win 160)) ;; esac
    mkdir -p "$D"
    id=$(CL submit --owner alpasim-long --name alpasim-long-smoke --vram 32 --cpu 8 --ram 24 --timeout-h 1 --tries 1 --priority 12 --log-dir "$D/pool" \
         -- env SH30_TAG=$TAG bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$LIST" "${BASE[@]}" "${X[@]}" | awk '{print $NF}') || die "$arm submit"
    status "$arm submitted $id (${X[*]:-shipped})"
  done
  for arm in ship a80 b160; do until [[ -e $O/runs/$arm/pool/DONE || -e $O/runs/$arm/pool/ERROR ]]; do sleep 15; done; du -sm "$O/runs/$arm" | cut -f1 > "$O/runs/$arm/du_mb_before_prune"; done
  $P experiments/onpol_nuplan/scripts/on_long_smoke.py --stage "$O" --list "$LIST" --out "$O/smoke.json" > "$O/smoke_table.md" || die "on_long_smoke rc $?"
  R=(); for arm in ship a80 b160; do [[ -e $O/runs/$arm/pool/DONE ]] && R+=("$TAG@$([[ $arm == b160 ]] && echo 18 || echo 10)$([[ $arm == a80 ]] && echo r)=$(ls -d $O/runs/$arm)"); done
  $P $E/scripts/long_read.py collect --runs "${R[@]}" --out "$O/read" --jobs 8 || die "collect rc $?" ;;
held)
  [[ -f $W/smoke/DONE ]] || die "smoke stage not done"
  [[ $(wc -l < $SPL/held.txt) -eq 460 ]] || die "held.txt is not 460 scenes"
  T18=${AL_TAGS18:-YR10m10-F-s0,YR10m10-F-s1,P2H10S-F-s0,P2H10S-F-s1,SSELFnoL-ON-s0,SSELFnoL-ON-s1}; T10=${AL_TAGS10:-YR10m10-F-s0,YR10m10-F-s1}
  for t in ${T18//,/ } ${T10//,/ }; do [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no checkpoint $t"; done
  common=(--owner alpasim-long --vram 32 --cpu 8 --tries 1 --priority 12 --timeout-h 3 --no-check)  # --no-check: cl fanout static check calls a missing pool.fanout_fill
  CL fanout --arms "$T18" --name al-h18 "${common[@]}" --ram ${AL_RAM18:-90} --log-dir "$O/h18/{arm}/pool" \
    -- bash -c "mkdir -p $O/h18/{arm} && SH30_TAG={arm} exec bash experiments/alpasim/scripts/run.sh $O/h18/{arm} sh30 --scene-list $SPL/held.txt ${BASE[*]} $(win 160)" \
    || die "fanout h18"
  CL fanout --arms "$T10" --name al-h10 "${common[@]}" --ram 69 --log-dir "$O/h10/{arm}/pool" \
    -- bash -c "mkdir -p $O/h10/{arm} && SH30_TAG={arm} exec bash experiments/alpasim/scripts/run.sh $O/h10/{arm} sh30 --scene-list $SPL/held.txt ${BASE[*]}" \
    || die "fanout h10"
  status "submitted: 18 decisions $T18; 10 decisions $T10"
  for d in $(for t in ${T18//,/ }; do echo h18/$t; done) $(for t in ${T10//,/ }; do echo h10/$t; done); do
    until [[ -e $O/$d/pool/DONE || -e $O/$d/pool/ERROR ]]; do sleep 30; done; status "$d: $(ls $O/$d/pool | grep -E '^(DONE|ERROR)$')"
  done
  OLD=$DATA_DIR/runs/onpol_nuplan/read/held/manifest.json
  R=(); for t in ${T18//,/ }; do [[ -e $O/h18/$t/pool/DONE ]] && R+=("$t@18=$(ls -d $O/h18/$t)"); done
  for t in ${T10//,/ }; do [[ -e $O/h10/$t/pool/DONE ]] && R+=("$t@10=$(ls -d $O/h10/$t)"); done
  for t in P2H10S-F-s0 P2H10S-F-s1 SSELFnoL-ON-s0 SSELFnoL-ON-s1; do R+=("$t@10=$(python3 -c "import json,sys; print(','.join(json.load(open('$OLD'))[sys.argv[1]]))" $t)"); done
  $P $E/scripts/long_read.py collect --runs "${R[@]}" --out "$O/read" --jobs 32 || die "collect rc $?"
  $P $E/scripts/long_read.py report --in "$O/read" --out "$O/read/report.md" --json "$O/read/report.json" || die "report rc $?"
  du -sm $O/h18 $O/h10 | tee "$O/du_mb_before_prune"
  python3 - "$O/read/per_scene.csv" "$O" <<'PY'   # keep rollout.asl of zero-score rollouts of this stage's runs only (ot2_loop's rule)
import csv, glob, os, sys
n = 0
for r in csv.DictReader(open(sys.argv[1])):
    if r["run"].startswith(sys.argv[2]) and float(r["score"] or 0) > 0:
        for f in glob.glob(f"{r['run']}/rollouts/{r['scene']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  ;;
*) die "unknown stage $STAGE" ;;
esac
status "done"; date '+%F %T' > "$O/DONE"
