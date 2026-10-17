#!/usr/bin/env bash
# ARENA (scripts/arena.py, prereg plans/2026-10-17-arena-prereg.md): uniform 5.5 s windows over every public reconstruction, read by every
# model in the pool. Box, repo root. The body is one function called on the last line, so editing this file never changes a running job.
#   arena.sh select                 pool CPU job: arena.py select (all candidate windows on the 1.0 s grid) -> $W/select/<ts>/cands.csv.gz
#   arena.sh pick <step>            by hand: the set on the <step> s grid -> $W/set/ (set.csv.gz, meta.json, lists/shard<k>.json, lists/c<n>.txt);
#                                   copies set.csv.gz / meta.json to results/arena/
#   arena.sh smoke [n]              feasibility: n windows (default 12, spread over strata) -> mini cache -> replay, starter, RSG-L18-s0 and
#                                   APY10m10-AB-s0 (ap2) reads -> $W/smoke/report.txt (pool jobs; prints the ids)
#   arena.sh chain                  the whole read: one build job per shard -> one read job per (tag, chunk), tags in $TAGS order (the pool
#                                   starts equal-priority jobs in submit order) -> two summary jobs (core tags, all tags). Env PRIO (9).
#   arena.sh build <k>              job body: the trajdata cache of shard k ($C/s<k>), the controls against the shipped cache, arena.py check
#   arena.sh read <tag> <chunk>     job body: one chunk (<= 150 windows) with <tag>'s driver -> $W/reads/<tag>/c<chunk>/run (idempotent)
#   arena.sh sum <name> <tag ..>    job body: waits (cap SUM_CAP s, default 54000) until every read job of the tags ended, then collect + report
#                                   over every finished read -> $W/report/<name>/{arena.md, rows.csv.gz, rows_valid.csv.gz, figs/}
# Drivers as wait_set.sh: checkpoints / P0 / YR10m10-F-s0 sh30 (the public 10-decision read path + the cache), APY10m10-AB-s0 ap2,
# starter = AlpaSim's go-straight kit, replay = the starter with force-GT over the whole run (human replay).
set -uo pipefail
cd "$(dirname "$0")/../../.."
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
E=experiments/alpasim_proxy; W=$DATA_DIR/runs/alpasim_proxy/arena; C=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; ST=$W/set
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; PY="$P $E/scripts/arena.py"; WSPY="$P $E/scripts/wait_set.py"; CK=$DATA_DIR/runs/op_parity/runs
KEY=runtime.scene_provider.trajdata
REPLAY=(runtime.simulation_config.force_gt_duration_us=20000000 ++eval.aggregation_modifiers.remove_preengagement_timesteps=false)
TAGS=${TAGS:-"replay RSG-L18-s0 NR1F-L18-s0 NR1FB10-L18-s0 RTB1ARO-L18-s0 YR10m10-F-s0 APY10m10-AB-s0 P0 starter"}
CORE="replay RSG-L18-s0 NR1F-L18-s0 NR1FB10-L18-s0 RTB1ARO-L18-s0"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
mkdir -p $W/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-proxy --name "$name" --priority ${PRIO:-9} \
    --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
RD=(--tries 2 --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)     # the public reads' declaration (df.sh, wait_set.sh)

one_run() {   # one_run <tag> <run dir> <scene list> <cache>: run.sh with the tag's driver; rc 137 when the platform killed a runtime worker
  local T=$1 R=$2 L=$3 CA=("$KEY.cache_location=$4") rc=0
  mkdir -p "$R"
  case $T in
    starter) bash experiments/alpasim/scripts/run.sh "$R" starter --scene-list $L "${BASE[@]}" "${CA[@]}" || rc=$? ;;
    replay)  bash experiments/alpasim/scripts/run.sh "$R" starter --scene-list $L "${BASE[@]}" "${CA[@]}" "${REPLAY[@]}" || rc=$? ;;
    APY10m10-AB-s0) env AP2_TAG=$T bash experiments/alpasim/scripts/run.sh "$R" ap2 --scene-list $L "${BASE[@]}" "${CA[@]}" || rc=$? ;;
    *)       env SH30_TAG=$T bash experiments/alpasim/scripts/run.sh "$R" sh30 --scene-list $L "${BASE[@]}" "${CA[@]}" || rc=$? ;;
  esac
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$R/native_summary.json" 2>/dev/null && { echo "$(date '+%F %T') runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($R)"
  fi
  python3 - "$R" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason") and (r.get("score") or 0) > 0:
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
}

build_cache() {   # build_cache <shard json> <cache dir> <controls txt>
  local J=$1 D=$2 CT=$3
  [[ -f $D/check.json ]] && python3 -c "import json, sys; s = json.load(open('$D/check.json'))['summary']; sys.exit(0 if s['ok'] == s['windows'] else 1)" && { echo "cache $D checked"; return 0; }
  rm -rf $D
  PYTHONPATH=$DATA_DIR/envs/nuplan_devkit_overlay $WSPY build --set $J --out $D --workers ${BUILD_WORKERS:-16} || die "build rc $?"
  if [[ -s $CT ]]; then
    $P experiments/alpasim_long/scripts/build_cache.py compare --a $D --b $DATA_DIR/datasets/alpasim_nuplan --scenes $CT --out $D/controls_vs_shipped.json \
      | python3 -c 'import json, sys; s = json.load(sys.stdin); print(s); sys.exit(0 if s["errors"] == 0 and s["exact"] == s["scenes"] else 1)' || die "controls differ from the shipped cache"
  fi
  $PY check --set $J --cache $D || die "cache check failed"
}

main() {
STAGE=${1:?stage}; shift
case $STAGE in
select)
  sub ar-select --tries 1 --vram 0.1 --cpu 48 --ram 64 --timeout-h 2 -- bash -c "$PY select --workers 48" ;;
pick)
  S=${1:?step}; [[ ! -e $ST/meta.json ]] || die "$ST exists (move it away to re-pick)"
  f=$(ls -d $W/select/2*/cands.csv.gz | tail -1); [[ -f $f ]] || die "no cands"
  $PY pick --cands $f --step $S --out $ST || die "pick rc $?"
  mkdir -p $E/results/arena && cp $ST/set.csv.gz $ST/meta.json $E/results/arena/ && echo "set from $f -> $ST, copied to $E/results/arena" ;;
smoke)
  n=${1:-12}; D=$W/smoke; [[ -f $ST/set.csv.gz ]] || die "pick first"; mkdir -p $D
  $P - $ST/set.csv.gz $D $n <<'PY' || die "smoke list"
import json, sys
import pandas as pd
S = pd.read_csv(sys.argv[1]); n = int(sys.argv[3])
g = S.groupby("stratum", group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, n // 6)), random_state=0)).head(n)
json.dump(dict(windows=g[["id", "log", "token", "asset", "start_idx", "src", "split"]].to_dict("records"), controls=[]), open(f"{sys.argv[2]}/smoke.json", "w"))
open(f"{sys.argv[2]}/list.txt", "w").write("".join(i + "\n" for i in g.id))
print(g.stratum.value_counts().to_dict())
PY
  : > $D/controls.txt
  b=$(sub ar-smoke-build --tries 1 --vram 0.1 --cpu 8 --ram 24 --timeout-h 1 --env BUILD_WORKERS=8 -- bash $E/scripts/arena.sh smokebuild) || exit 1
  ids=()
  for T in replay starter RSG-L18-s0 APY10m10-AB-s0; do
    ids+=($(sub ar-smoke-$T --after $b "${RD[@]}" -- bash $E/scripts/arena.sh smokeread $T)) || exit 1
  done
  q=$(sub ar-smoke-report --after $(IFS=,; echo "${ids[*]}") --tries 1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash -c \
      "$PY collect --reads $D/reads --out $D/rows.csv.gz && $P -c \"import pandas as pd; r = pd.read_csv('$D/rows.csv.gz'); print(r.groupby('tag').score.describe()); print(r[['tag', 'id', 'score', 'type', 'dist', 'gt']].to_string())\" > $D/report.txt") || exit 1
  echo "smoke: build $b reads ${ids[*]} report $q" | tee -a $W/stages.txt ;;
smokebuild)
  build_cache $W/smoke/smoke.json $C/smoke $W/smoke/controls.txt ;;
smokeread)
  T=$1; D=$W/smoke/reads/$T/c0/run; rm -rf $D; one_run $T $D $W/smoke/list.txt $C/smoke ;;
build)
  k=$1; build_cache $ST/lists/shard$k.json $C/s$k $ST/lists/shard${k}_controls.txt ;;
read)
  T=$1; c=$2; D=$W/reads/$T/c$c/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; return 0; }
  k=$(python3 -c "import json; print(next(s['shard'] for s in json.load(open('$ST/meta.json'))['shards'] if $c in s['chunks']))") || die "no shard for chunk $c"
  [[ -f $C/s$k/check.json ]] || die "shard $k not built"
  rm -rf $D; one_run $T $D $ST/lists/c$c.txt $C/s$k ;;
chain)
  [[ -f $ST/meta.json ]] || die "pick first"
  for T in $TAGS; do [[ $T == P0 || $T == starter || $T == replay || -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; done
  declare -A B; ids=()
  while read -r k; do
    B[$k]=$(sub ar-build-s$k --tries 2 --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash $E/scripts/arena.sh build $k) || exit 1
  done < <(python3 -c "import json; [print(s['shard']) for s in json.load(open('$ST/meta.json'))['shards']]")
  echo "builds: ${B[*]}" | tee -a $W/stages.txt
  for T in $TAGS; do
    tid=()
    while read -r c k; do
      tid+=($(sub ar-read-$T-c$c --after ${B[$k]} "${RD[@]}" -- bash $E/scripts/arena.sh read $T $c)) || exit 1
    done < <(python3 -c "import json; [print(c, s['shard']) for s in json.load(open('$ST/meta.json'))['shards'] for c in s['chunks']]")
    echo "reads $T: ${tid[0]} .. ${tid[-1]} (${#tid[@]})" | tee -a $W/stages.txt; ids+=("${tid[@]}")
  done
  s1=$(sub ar-sum-core --tries 1 --vram 0.1 --cpu 4 --ram 16 --timeout-h 16 -- bash $E/scripts/arena.sh sum core $CORE) || exit 1
  s2=$(sub ar-sum-all --tries 1 --vram 0.1 --cpu 4 --ram 16 --timeout-h 16 -- bash $E/scripts/arena.sh sum all $TAGS) || exit 1
  echo "chain: ${#ids[@]} reads; summaries core $s1, all $s2" | tee -a $W/stages.txt ;;
sum)
  N=$1; shift; t0=$(date +%s); O=$W/report/$N; mkdir -p $O
  n=$(( $(python3 -c "import json; print(json.load(open('$ST/meta.json'))['chunks'])") * $# ))   # read jobs expected
  while :; do
    fin=0; for T in "$@"; do for f in $W/pool/ar-read-$T-c*/STATUS; do [[ -f $f ]] || continue
      case "$(tail -1 $f)" in *": done"*|*": failed"*|*": cancel"*|*": skip"*|*": error"*) fin=$((fin+1));; esac; done; done
    open=$((n - fin)); (( open <= 0 )) && break; (( $(date +%s) - t0 > ${SUM_CAP:-54000} )) && { echo "cap reached: $open read jobs open"; break; }; sleep 300
  done
  date; for T in "$@"; do for f in $W/pool/ar-read-$T-c*/STATUS; do tail -1 $f; done; done | grep -v ": done" | tee $O/not_done.txt
  $PY collect --reads $W/reads --out $O/rows.csv.gz || die "collect"
  $PY report --set $ST/set.csv.gz --rows $O/rows.csv.gz --out $O --step "$(python3 -c "import json; print(json.load(open('$ST/meta.json'))['step_s'])")" > $O/stdout.txt || die "report"
  date '+%F %T' > $O/DONE ;;
*) die "unknown stage $STAGE" ;;
esac
}
main "$@"; exit $?
