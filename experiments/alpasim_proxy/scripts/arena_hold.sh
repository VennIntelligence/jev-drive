#!/usr/bin/env bash
# ARENA-HOLD (scripts/arena_hold.py, prereg plans/2026-10-17-arena-prereg.md supplement 1): RSG-L18-s0 with the serve-side static hold
# (JEV_HOLD, experiments/alpasim/lib/static_hold.py, served by the sh30hold driver = lib/sh30_hold.py around sh30_driver.py) on the
# arena's held-split windows, reusing the arena's set and shard caches read-only. Box, repo root. Writes only under $W/hold.
#   arena_hold.sh lists                      the held windows -> $H/{meta.json, lists/c<n>.txt, lists/smoke.txt} (by hand, seconds)
#   arena_hold.sh chain                      smoke (JEV_HOLD off + full on 12 held windows) -> gate -> one read job per (variant, chunk)
#                                            after the gate -> one summary job. Env PRIO (10). Prints and appends the ids to $H/stages.txt
#   arena_hold.sh read <variant> <chunk>     job body: one chunk with JEV_HOLD=<variant> (off = unset) -> $H/reads/<variant>/c<chunk>/run
#   arena_hold.sh smokeread <variant> [list] job body: the smoke list (lists/<list>.txt, default smoke) -> $H/smoke/<variant>/run
#   arena_hold.sh gate | sum                 job bodies: arena_hold.py gate; wait for every read job, collect, arena_hold.py report
#                                            (sum: env VARIANTS, GATE = the gate's job name, OUT = the report dir, VS = a read to pair against too)
# Supplement 2 (stop3 = leash3 + the stop leash, prereg supplement 2):
#   arena_hold.sh diag                       by hand, seconds: arena_hold.py diag -> $H/diag.md (costs of leash3, the stop3 estimate) and
#                                            the smoke list lists/smoke2.txt (+ .json with its shard)
#   arena_hold.sh chain2                     smoke (stop3 on smoke2) -> gate2 -> one stop3 read per chunk -> summary into $H/report2
#                                            (paired against leash3 and RSG). Env PRIO
#   arena_hold.sh gate2                      job body: arena_hold.py gate2 on the stop3 smoke
# The body is one function called on the last line.
set -uo pipefail
cd "$(dirname "$0")/../../.."
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
E=experiments/alpasim_proxy; W=$DATA_DIR/runs/alpasim_proxy/arena; H=$W/hold; C=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; ST=$W/set
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; PY="$P $E/scripts/arena_hold.py"
KEY=runtime.scene_provider.trajdata
VARIANTS=${VARIANTS:-"full lead leash3"}
TAG=RSG-L18-s0
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
mkdir -p $H/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-proxy --name "$name" --priority ${PRIO:-10} \
    --no-check --log-dir $H/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id $name" >> $H/jobs.txt; echo "$id"; }
RD=(--tries 2 --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)     # arena.sh RD (our drivers)

one_run() {   # one_run <variant> <run dir> <scene list> <cache>
  local V=$1 R=$2 L=$3 rc=0
  mkdir -p "$R"
  if [[ $V == off ]]; then
    env -u JEV_HOLD SH30_TAG=$TAG bash experiments/alpasim/scripts/run.sh "$R" sh30hold --scene-list $L "${BASE[@]}" "$KEY.cache_location=$4" || rc=$?
  else
    env JEV_HOLD=$V SH30_TAG=$TAG bash experiments/alpasim/scripts/run.sh "$R" sh30hold --scene-list $L "${BASE[@]}" "$KEY.cache_location=$4" || rc=$?
  fi
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

main() {
STAGE=${1:?stage}; shift
case $STAGE in
lists)
  [[ ! -e $H/meta.json ]] || die "$H/meta.json exists"
  $PY lists --set $ST/set.csv.gz --out $H ;;
chain)
  [[ -f $H/meta.json ]] || die "lists first"
  s=()
  for V in off full; do s+=($(sub ah-smoke-$V "${RD[@]}" -- bash $E/scripts/arena_hold.sh smokeread $V)) || exit 1; done
  g=$(sub ah-gate --after $(IFS=,; echo "${s[*]}") --tries 1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $E/scripts/arena_hold.sh gate) || exit 1
  ids=()
  for V in $VARIANTS; do
    tid=()
    while read -r c; do tid+=($(sub ah-read-$V-c$c --after $g "${RD[@]}" -- bash $E/scripts/arena_hold.sh read $V $c)) || exit 1
    done < <(python3 -c "import json; [print(c['chunk']) for c in json.load(open('$H/meta.json'))['chunks']]")
    echo "reads $V: ${tid[0]} .. ${tid[-1]} (${#tid[@]})" | tee -a $H/stages.txt; ids+=("${tid[@]}")
  done
  q=$(sub ah-sum --tries 1 --vram 0.1 --cpu 4 --ram 16 --timeout-h 16 -- bash $E/scripts/arena_hold.sh sum) || exit 1
  echo "$(date '+%F %T') chain: smoke ${s[*]} gate $g, ${#ids[@]} reads, summary $q" | tee -a $H/stages.txt ;;
smokeread)
  V=$1; SL=${2:-smoke}; D=$H/smoke/$V/run; rm -rf $D
  if [[ -f $H/lists/$SL.json ]]; then k=$(python3 -c "import json; print(json.load(open('$H/lists/$SL.json'))['shard'])")
  else k=$(python3 -c "import json; print(json.load(open('$H/meta.json'))['smoke']['shard'])"); fi
  one_run $V $D $H/lists/$SL.txt $C/s$k ;;
gate)
  $PY gate --off $H/smoke/off/run --full $H/smoke/full/run --ref-reads $W/reads/$TAG --out $H/smoke/gate.txt ;;
diag)
  $PY diag --set $ST/set.csv.gz --corridor $W/corridor --hold $H --out $H/diag.md --smoke $H/lists/smoke2.txt ;;
chain2)
  [[ -f $H/lists/smoke2.txt ]] || die "diag first"
  s=$(sub ah2-smoke-stop3 "${RD[@]}" -- bash $E/scripts/arena_hold.sh smokeread stop3 smoke2) || exit 1
  g=$(sub ah2-gate --after $s --tries 1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $E/scripts/arena_hold.sh gate2) || exit 1
  tid=()
  while read -r c; do tid+=($(sub ah-read-stop3-c$c --after $g "${RD[@]}" -- bash $E/scripts/arena_hold.sh read stop3 $c)) || exit 1
  done < <(python3 -c "import json; [print(c['chunk']) for c in json.load(open('$H/meta.json'))['chunks']]")
  q=$(sub ah2-sum --after $g --tries 1 --vram 0.1 --cpu 4 --ram 16 --timeout-h 16 -- env VARIANTS=stop3 GATE=ah2-gate OUT=$H/report2 VS=leash3 \
      bash $E/scripts/arena_hold.sh sum) || exit 1
  echo "$(date '+%F %T') chain2 (supplement 2): smoke $s gate $g, reads stop3 ${tid[0]} .. ${tid[-1]} (${#tid[@]}), summary $q" | tee -a $H/stages.txt ;;
gate2)
  $PY gate2 --run $H/smoke/stop3/run --out $H/smoke/gate2.txt ;;
read)
  V=$1; c=$2; D=$H/reads/$V/c$c/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; return 0; }
  k=$(python3 -c "import json; print(next(x['shard'] for x in json.load(open('$H/meta.json'))['chunks'] if x['chunk'] == $c))") || die "no chunk $c"
  [[ -f $C/s$k/check.json ]] || die "shard $k not built"
  rm -rf $D; one_run $V $D $H/lists/c$c.txt $C/s$k ;;
sum)
  t0=$(date +%s); O=${OUT:-$H/report}; mkdir -p $O
  n=$(( $(python3 -c "import json; print(len(json.load(open('$H/meta.json'))['chunks']))") * $(echo $VARIANTS | wc -w) ))
  while :; do
    fin=0; for V in $VARIANTS; do for f in $H/pool/ah-read-$V-c*/STATUS; do [[ -f $f ]] || continue
      case "$(tail -1 $f)" in *": done"*|*": failed"*|*": cancel"*|*": skip"*|*": error"*) fin=$((fin+1));; esac; done; done
    g=$(tail -1 $H/pool/${GATE:-ah-gate}/STATUS 2>/dev/null)
    [[ $g == *": failed"* || $g == *": error"* || $g == *": cancel"* ]] && { echo "gate failed: $g"; break; }
    (( n - fin <= 0 )) && break; (( $(date +%s) - t0 > ${SUM_CAP:-54000} )) && { echo "cap reached: $((n - fin)) read jobs open"; break; }; sleep 300
  done
  for V in $VARIANTS; do for f in $H/pool/ah-read-$V-c*/STATUS; do tail -1 $f; done; done | grep -v ": done" | tee $O/not_done.txt
  $P $E/scripts/arena.py collect --reads $H/reads --out $O/rows.csv.gz || die "collect"
  $P $E/scripts/arena.py collect --reads $W/reads --out $O/ref_rows.csv.gz || die "collect ref"
  $PY report --set $ST/set.csv.gz --hold $O --ref $O/ref_rows.csv.gz --out $O ${VS:+--variants $VS $VARIANTS --vs $VS} > $O/stdout.txt || die "report"
  date '+%F %T' > $O/DONE ;;
*) die "unknown stage $STAGE" ;;
esac
}
main "$@"; exit $?
