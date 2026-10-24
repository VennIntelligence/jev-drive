#!/usr/bin/env bash
# ARENA-V2 (plans/2026-10-25-arena-v2-check.md): the decision-355 private-board anchors + JTN-L18-s1 native / stop3 re-read natively on the
# arena held split (arena/hold lists and shard caches, read-only), keeping every ASL the fixed corridor scorer can change (round9 cb2v2.py
# prune), then rescored with the old and the fixed (1b024f0) scorer (cb2v2.py score). Box, repo root; every job through the pool at
# PRIO (default 0: lowest, fills idle cards only). Writes only under $X.
#   arena_v2.sh wave <name> "<chunks>" ["<tags>"]   one read job per (tag, chunk) + one v2 job per tag after its reads (all chunks read so far)
#   arena_v2.sh read <tag> <chunk>                  job body (idempotent)
#   arena_v2.sh v2 <tag>                            job body: cb2v2 score old (must reproduce the read) and new -> $X/v2/<tag>/{old,new}
#   arena_v2.sh wait <name>                         one run_in_background waiter: returns when every job of the wave ended
# The body is one function called on the last line.
set -uo pipefail
cd "$(dirname "$0")/../../.."
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
W=$DATA_DIR/runs/alpasim_proxy/arena; AH=$W/hold; X=$W/v2read; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V2=experiments/round9/scripts/cb2v2.py; NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src
KEY=runtime.scene_provider.trajdata
TAGS="G0014 YR10m10-F-s0 starter DD-sample LTF-sample APY10m10-AB-s0 JTN-L18-s1-native JTN-L18-s1-stop3"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
mkdir -p $X/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner alpasim-proxy-v2 --name "$name" --priority ${PRIO:-0} \
    --no-check --log-dir $X/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id $name" >> $X/jobs.txt; echo "$id"; }
RD=(--tries 2 --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)     # arena.sh RD
R3=(--tries 2 --vram 24 --cpu 8 --ram 40 --timeout-h 2)                                      # arena.sh R3 (third-party drivers)
decl() { case $1 in LTF-sample|DD-sample|G0014) echo "${R3[@]}" ;; *) echo "${RD[@]}" ;; esac; }

main() {
case ${1:?stage} in
read)
  T=$2; c=$3; D=$X/reads/$T/c$c/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; return 0; }
  k=$(python3 -c "import json; print(next(x['shard'] for x in json.load(open('$AH/meta.json'))['chunks'] if x['chunk'] == $c))") || die "no chunk $c"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  L=$AH/lists/c$c.txt; CA="$KEY.cache_location=$AC/s$k"; RS=experiments/alpasim/scripts/run.sh; rc=0
  rm -rf $D; mkdir -p $D
  case $T in
    starter)    bash $RS $D starter --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    LTF-sample) bash $RS $D ltf --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    DD-sample)  bash $RS $D dd --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    G0014)      bash $RS $D garage --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    APY10m10-AB-s0) env AP2_TAG=$T bash $RS $D ap2 --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    YR10m10-F-s0)   env SH30_TAG=$T bash $RS $D sh30 --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    JTN-L18-s1-native) env -u JEV_HOLD SH30_TAG=JTN-L18-s1 bash $RS $D sh30hold --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    JTN-L18-s1-stop3)  env JEV_HOLD=stop3 SH30_TAG=JTN-L18-s1 bash $RS $D sh30hold --scene-list $L "${BASE[@]}" "$CA" || rc=$? ;;
    *) die "unknown tag $T" ;;
  esac
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  $AP $V2 prune --run $D || die "prune" ;;
v2)
  T=$2; O=$X/v2/$T; mkdir -p $O
  $AP $V2 score --variant old --runs $X/reads/$T/c*/run --out $O/old --workers ${WORKERS:-16}; echo "old rc $?" > $O/old_rc.txt   # rc 2 = a stored score not reproduced (reported)
  PYTHONPATH=$NEW $AP $V2 score --variant new --runs $X/reads/$T/c*/run --out $O/new --workers ${WORKERS:-16} || die "v2 new $T" ;;
wave)
  N=$2; CH=$3; TG=${4:-$TAGS}; [[ ! -e $X/wave_$N ]] || die "wave $N already submitted"
  for T in $TG; do [[ $T == starter || $T == *-sample || $T == G0014 || $T == JTN-L18-s1-* || -f $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt ]] || die "no checkpoint $T"; done
  echo "$(git rev-parse --short HEAD) chunks $CH tags $TG" > $X/wave_$N; ids=()
  for T in $TG; do
    r=(); for c in $CH; do r+=($(sub av2-$N-$T-c$c $(decl $T) -- bash experiments/alpasim_proxy/scripts/arena_v2.sh read $T $c)) || exit 1; done
    ids+=("${r[@]}" $(sub av2-$N-v2-$T --after $(IFS=,; echo "${r[*]}") --tries 1 --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- \
      bash experiments/alpasim_proxy/scripts/arena_v2.sh v2 $T)) || exit 1
  done
  echo "${ids[*]}" >> $X/wave_$N; echo "$(date '+%F %T') wave $N: ${#ids[@]} jobs" | tee -a $X/stages.txt ;;
wait)
  N=$2; ids=$(tail -1 $X/wave_$N)
  while :; do open=0
    for f in $X/pool/av2-$N-*/STATUS; do case "$(tail -1 $f 2>/dev/null)" in *": done"*|*": failed"*|*": cancel"*|*": error"*|*": skip"*) ;; *) open=$((open+1));; esac; done
    n=$(ls -d $X/pool/av2-$N-* 2>/dev/null | wc -l); (( open == 0 && n == $(wc -w <<< "$ids") )) && break; sleep 600; done
  for f in $X/pool/av2-$N-*/STATUS; do tail -1 $f; done | grep -v ": done" ; echo "wave $N ended $(date '+%F %T')" ;;
*) die "unknown stage $1" ;;
esac
}
main "$@"; exit $?
