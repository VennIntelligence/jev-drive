#!/usr/bin/env bash
# ROUND5 PAI road raster (plans/2026-10-21-pai-road-raster.md): box, repo root; every job through the pool at PRIO 5.
#   road_pai.sh chain <old rev>   submit: build (CPU) -> index on1_PAIH / PAIH05 / PAIH15 / JTPH (CPU); g1ref, g2 (CPU, independent);
#                                 G3 = R2J-L18-s0's recorded command 60 steps eager (JEV_RM_EAGER=1) three times: <old rev> in its own worktree,
#                                 this code, this code + --on-road-pai road_on1_JTPH.npz --on-road-pai-w 0 -> g3cmp (df.py bitcmp x 2) -> report -> DONE
#   road_pai.sh wait              one blocking waiter (run_in_background): returns at $O/DONE or when a job of this lane failed
#   job bodies: build | index <pool> | g1ref | g2 | g3run <tag> [old rev | w0] | g3cmp <old rev> | report
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round5/road_pai; ME=experiments/round5/scripts/road_pai.sh; PY=experiments/round5/scripts/road_pai.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
POOLS=(on1_PAIH on1_PAIH05 on1_PAIH15 on1_JTPH); REF=R2J-L18-s0; G3OLD=RP3-old; G3NEW=RP3-new; G3W0=RP3W0-new
mkdir -p $O/pool
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round5-road --name "$name" --tries 1 --priority ${PRIO:-5} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
case ${1:-} in
chain)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/rp_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || die "worktree $rev"
  for X in $G3OLD $G3NEW $G3W0; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  rm -f $O/DONE; git rev-parse --short HEAD > $O/code.txt
  NC=$($V -c "from jevdrive.common import n_cpus; print(n_cpus())")             # the box's cores; each job takes a share, its body uses n_cpus()
  CB=$(( NC / 3 )); CI=$(( NC / 10 > 4 ? NC / 10 : 4 ))
  FE=(--train --vram 40 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1)   # r2.sh pre's identity runs
  b=$(sub rp-build --vram 0.1 --cpu $CB --ram $(( CB * 3 )) --timeout-h 3 -- bash $ME build) || exit 1
  ix=(); for P in "${POOLS[@]}"; do ix+=($(sub rp-index-$P --after $b --vram 0.1 --cpu $CI --ram 40 --timeout-h 2 -- bash $ME index $P)) || exit 1; done
  r=$(sub rp-g1ref --vram 0.1 --cpu $CI --ram 32 --timeout-h 1 -- bash $ME g1ref) || exit 1
  g=$(sub rp-g2 --vram 0.1 --cpu $CI --ram 40 --timeout-h 2 -- bash $ME g2) || exit 1
  t1=$(sub rp-g3-old "${FE[@]}" -- bash $ME g3run $G3OLD $rev) || exit 1
  t2=$(sub rp-g3-new "${FE[@]}" -- bash $ME g3run $G3NEW) || exit 1
  t3=$(sub rp-g3-w0 --after ${ix[3]} "${FE[@]}" -- bash $ME g3run $G3W0 w0) || exit 1
  c=$(sub rp-g3cmp --after $t1,$t2,$t3 --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME g3cmp $rev) || exit 1
  x=$(sub rp-report --after $(csv ${ix[@]} $r $g $c) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME report) || exit 1
  echo "$(date '+%F %T') chain (code $(cat $O/code.txt), old $rev, cores $NC): build $b index ${ix[*]} g1ref $r g2 $g g3 $t1 $t2 $t3 cmp $c report $x" | tee -a $O/stages.txt ;;
wait)
  until [[ -f $O/DONE ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $O/DONE ]] && cat $O/DONE ;;
build) $AP $PY build ;;
index) $TR $PY index --pool $2 ;;
g1ref) $TR $PY g1ref ;;
g2)    $AP $PY g2 ;;
g3run)  # g3run <tag> [<old rev> | w0]: R2J-L18-s0's recorded argv (meta.json) with --tag <tag> --steps 60; old rev: from its worktree
  T=$2; M=$(ls -d $DATA_DIR/runs/route_mem/train-$REF/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  mapfile -t A < <(python3 -c "
import json, sys
a = json.load(open('$M'))['argv']
a[a.index('--tag') + 1] = '$T'
a[len(a) - 1 - a[::-1].index('--steps') + 1] = '60'
print('\n'.join(a))")
  X=(); [[ ${3:-} == w0 ]] && X=(--on-road-pai $O/road_on1_JTPH.npz --on-road-pai-w 0)
  if [[ -n ${3:-} && ${3:-} != w0 ]]; then cd $DATA_DIR/tmp/rp_old_$3 || die "no worktree $3"; fi
  echo "$(date '+%F %T') g3run $T in $PWD: ${A[*]} ${X[*]}" >> $O/stages.txt
  exec $TR "${A[@]}" "${X[@]}" ;;
g3cmp)  # g3cmp <old rev>: old vs new (the change is inert) and new vs new + W 0 (G3) -> $O/g3_{off,w0}.json
  DF=experiments/defect_fix/scripts/df.py
  $TR $DF bitcmp --a $G3OLD --b $G3NEW --out road_pai_g3_off --note "R2J-L18-s0 command 60 steps eager: code $2 vs $(cat $O/code.txt), no new flag"
  $TR $DF bitcmp --a $G3NEW --b $G3W0 --out road_pai_g3_w0 --note "the same code: without vs with --on-road-pai road_on1_JTPH.npz --on-road-pai-w 0"
  cp experiments/defect_fix/results/road_pai_g3_off.json $O/g3_off.json; cp experiments/defect_fix/results/road_pai_g3_w0.json $O/g3_w0.json
  exit 0 ;;   # the verdict is in the json files (report); the chain goes on to the report either way
report)
  $TR $PY report --g3 off=$O/g3_off.json w0=$O/g3_w0.json || exit 1
  du -sb $O | awk '{printf "{\"disk_gb\": %.2f}\n", $1 / 1e9}' > $O/DONE ;;
*) sed -n 2,8p $0; exit 2 ;;
esac
