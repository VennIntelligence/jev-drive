#!/usr/bin/env bash
# PRACTICE-NUPLAN lane (plans/2026-10-19-pn-prereg.md). Box, repo root; every GPU / CPU stage is a pool job (priority PRIO, 11), --after chained;
# ids in $W/jobs.txt, stage lines in $W/stages.txt.
#   pn.sh select                 job body: pn.py select (windows, splits, lists)
#   pn.sh smoke                  staged launch 1: the 10-window smoke list through roll -> geo -> tok -> route -> anchor -> strip (pool PNSM, never merged)
#   pn.sh launch                 staged launch 2 + the control: first roll chunk of every shard (7) -> stage2 gate (pn.py stage2; on pass it runs
#                                `pn.sh rest`); control JTNC1K-L18-s0 (JTN recipe, 1 000 steps, on1_JT / JT00) -> held 460 @18 (6)
#   pn.sh rest                   the remaining roll chunks; per shard geo -> tok -> route / anchor / strip; merge on1_JTX (+ .a, weights JTX00);
#                                pilot JTX1K-L18-s0 -> held 460 @18 (6); gate (pn.py gate) after both arms' reads
#   pn.sh roll <chunk|smoke> | geo <shard|SM> | tok <pool> | route <pool> | anchor <pool> | strip <pool> | merge | train <tag>    job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
E=experiments/practice_nuplan; ME=$E/scripts/pn.sh; PY=$E/scripts/pn.py; W=$DATA_DIR/runs/practice_nuplan; S=$W/select
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
ON=experiments/onpol_nuplan/scripts/on_rows.py; RB=experiments/route_mem/scripts/rm_build.py; JT=experiments/defect_fix/scripts/jt.sh
AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata; CK=$DATA_DIR/runs/op_parity/runs
T=JTN-L18-s0                                                                         # the rolling policy (prereg 2)
PILOT=JTX1K-L18-s0; CTRL=JTNC1K-L18-s0; REF=JTNS1K-L18-s0                           # pilot / fresh control / stored control (NR1 supplement 11)
JTPARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"   # on1_JT's parts in order (jt.sh PARTS)
JTA="on1_L18RB0:rb0,on1_L18RTB1:rtb1,on1_L18YS:ys,on1_L18Y:y,on1_NR1TV:tv"
LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner practice-nuplan --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-11} \
     --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
nchunks() { python3 -c "import json; print(len(json.load(open('$S/meta.json'))['chunks']))"; }
shard_of() { python3 -c "import json; print(next(c['shard'] for c in json.load(open('$S/meta.json'))['chunks'] if c['chunk'] == $1))"; }
chunks_of() { python3 -c "import json; print(' '.join(str(c['chunk']) for c in json.load(open('$S/meta.json'))['chunks'] if c['shard'] == $1))"; }
shards() { python3 -c "import json; print(' '.join(map(str, sorted({c['shard'] for c in json.load(open('$S/meta.json'))['chunks']}))))"; }
RO=(--vram 32 --cpu 8 --ram 60 --timeout-h 3 --env CL_HIST_KEY=pn-roll)
RD=(--vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 1)
chain() {  # chain <pool> <after ids>: geo -> tok (R1 / R2 checked) -> route -> anchor; strip after tok. Prints "tok route anchor strip" ids
  local N=$1 A=$2 g k r n s
  g=$(sub pn-geo-$N --after $A --vram 0.1 --cpu 24 --ram 64 --timeout-h 2 -- bash $ME geo $N) || exit 1
  k=$(sub pn-tok-$N --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME tok $N) || exit 1
  r=$(sub pn-route-$N --after $k --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME route $N) || exit 1
  n=$(sub pn-anchor-$N --after $r --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME anchor $N) || exit 1
  s=$(sub pn-strip-$N --after $k --vram 0.1 --cpu 16 --ram 16 --timeout-h 2 -- bash $ME strip $N) || exit 1
  echo "$k $r $n $s"; }
reads() {  # reads <tag> <after>: held 460 @18 in 6 chunks (rm_chain.sh read, as the JT family)
  local i; for i in 0 1 2 3 4 5; do
    TRIES=2 sub pn-held-$1-$i --after $2 "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $1 $i 6 || exit 1; done; }
traincmd() {  # traincmd <tag> <pool> <wfile> -> $W/cmd/<tag>.txt: jt.sh's own JTN command (DRY), 1 000 steps, warm-up 50 (= JTNS1K-L18-s0's)
  mkdir -p $W/cmd; DRY=1 ARCROWS=brake WARM=50 bash $JT train $1 $2 $LAT $3 1000 0 | tail -1 > $W/cmd/$1.txt
  grep -q "rm_train.py train --tag $1 " $W/cmd/$1.txt || die "no command for $1"; }
case ${1:-} in
select) $AP $PY select ;;
smoke)
  [[ -f $S/meta.json ]] || die "run select first"
  r=$(sub pn-roll-smoke "${RO[@]}" -- bash $ME roll smoke) || exit 1
  ids=($(chain PNSM $r)) || exit 1
  echo "$(date '+%F %T') smoke: roll $r tok/route/anchor/strip ${ids[*]}" | tee -a $W/stages.txt ;;
launch)
  [[ -f $S/meta.json && -f $W/smoke_ok ]] || die "select + an inspected smoke first ($W/smoke_ok)"
  for x in $PILOT $CTRL; do [[ ! -e $CK/$x/ckpt-final.pt ]] || die "$x exists"; done
  traincmd $CTRL on1_JT JT00
  c=$(sub pn-train-$CTRL "${FT[@]}" -- bash $ME train $CTRL) || exit 1
  cr=($(reads $CTRL $c)) || exit 1
  rr=(); for k in $(shards); do j=$(chunks_of $k | awk '{print $1}')
    rr+=($(sub pn-roll-c$j "${RO[@]}" -- bash $ME roll $j)) || exit 1; done
  runs=$(for k in $(shards); do j=$(chunks_of $k | awk '{print $1}'); echo -n "$W/roll/$T/c$j/run "; done)
  n=$(python3 -c "import json; print(json.load(open('$S/meta.json'))['kept'])")
  g=$(sub pn-stage2 --after $(csv ${rr[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 --env CTRL_READS=$(csv ${cr[@]}) -- \
      bash -c "$AP $PY stage2 --runs $runs --total-windows $n > $W/stage2.json && bash $ME rest") || exit 1
  echo "$(date '+%F %T') launch: control $c reads ${cr[*]}; stage-2 rolls ${rr[*]}; stage2 gate + rest $g" | tee -a $W/stages.txt ;;
rest)     # run by the stage2 job (CTRL_READS: the control's read ids)
  first=" $(for k in $(shards); do chunks_of $k | awk '{print $1}'; done | tr '\n' ' ') "
  done_ids=(); for k in $(shards); do                                                     # stage-2 chunks are finished: no --after on them
    rr=(); for j in $(chunks_of $k); do [[ $first == *" $j "* ]] && continue; rr+=($(sub pn-roll-c$j "${RO[@]}" -- bash $ME roll $j)) || exit 1; done
    A=$( ((${#rr[@]})) && csv ${rr[@]} || echo "${CL_POOL_JOB:-}")
    done_ids+=($(chain PN$k $A)) || exit 1
  done
  m=$(sub pn-merge --after $(csv ${done_ids[@]}) --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- bash $ME merge) || exit 1
  t=$(sub pn-train-$PILOT --after $m "${FT[@]}" -- bash $ME train $PILOT) || exit 1
  pr=($(reads $PILOT $t)) || exit 1
  q=$(sub pn-gate --after $(csv ${pr[@]}),${CTRL_READS:?} --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- \
      $TR $PY gate --pilot $PILOT --control $CTRL --ref $REF) || exit 1
  echo "$(date '+%F %T') rest: shard chains ${done_ids[*]}; merge $m; pilot $t reads ${pr[*]}; gate $q" | tee -a $W/stages.txt ;;
post)     # post <control read ids>: amendment 3's resubmission of rest's tail (shard chains after their roll ids in jobs.txt, merge, pilot, reads, gate)
  done_ids=(); for k in $(shards); do
    rr=(); for j in $(chunks_of $k); do rr+=($(awk -v n=pn-roll-c$j '$2 == n {i = $1} END {print i}' $W/jobs.txt)); done
    done_ids+=($(chain PN$k $(csv ${rr[@]}))) || exit 1
  done
  m=$(sub pn-merge --after $(csv ${done_ids[@]}) --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- bash $ME merge) || exit 1
  t=$(sub pn-train-$PILOT --after $m "${FT[@]}" -- bash $ME train $PILOT) || exit 1
  pr=($(reads $PILOT $t)) || exit 1
  q=$(sub pn-gate --after $(csv ${pr[@]}),${2:?control reads} --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- \
      $TR $PY gate --pilot $PILOT --control $CTRL --ref $REF) || exit 1
  echo "$(date '+%F %T') post: shard chains ${done_ids[*]}; merge $m; pilot $t reads ${pr[*]}; gate $q" | tee -a $W/stages.txt ;;
roll)     # roll <chunk|smoke>: the shard's arena cache, sh30 (no hold, = the default nuPlan serve of the L18 pools), frame dumps, then ol_chain roll1's steps
  j=$2; L=$S/lists/c$j.txt; k=0; [[ $j == smoke ]] && L=$S/lists/smoke.txt || k=$(shard_of $j); D=$W/roll/$T/c$j/run
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  [[ -z $(comm -12 <(sort $L) <(sort experiments/onpol_nuplan/results/split/held.txt)) ]] || die "$L has a held scene"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf "$D"; mkdir -p "$D"
  env -u JEV_HOLD SH30_TAG=$T SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "$KEY.cache_location=$AC/s$k" \
    || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  $AP experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/onpol_sim.pkl" --jobs 8 || die "c1_extract rc $?"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason"):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "$(date '+%F %T') roll $T c$j (shard $k) done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
geo)      # geo <pool>: rows of the pool's chunks (on_rows.py geo with ON_SPLIT / ON_TOK_TIMES; no other change)
  N=$2; R=$W/stage/$N; mkdir -p $R
  if [[ $N == PNSM ]]; then cs=smoke; else cs=$(chunks_of ${N#PN}); fi
  python3 - "$R" "$W/roll/$T" "$T" $cs <<'PY' || die "roll chunks missing ($N)"
import json, os, sys
R, base, T, cs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
m = {f"{T}@c{c}": [f"{base}/c{c}/run"] for c in cs if os.path.exists(f"{base}/c{c}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", len(cs), "chunks"); sys.exit(len(m) != len(cs))
PY
  ON_SPLIT=alpasim_nuplan/practice-starts ON_TOK_TIMES=$S/tok_times.json $NAV $ON geo --name $N --roll $R --workers ${PN_WORKERS:-24} || die "on_rows geo rc $?" ;;
tok)      # tok <pool>: tokens + teacher + gate R1 (on_rows.py tok, control smoke10e as ol_chain), then R1 / R2 (prereg amendment 1: ego <= 0.3 m, frame <= 0.1 m, actors >= 0.995 within 0.5 m, median <= 0.05 m)
  N=$2; ON_SPLIT=alpasim_nuplan/practice-starts $TR $ON tok --name $N --control smoke10e || die "on_rows tok rc $?"
  python3 - $DATA_DIR/runs/onpol_nuplan/rows/$N <<'PY' || die "pool gates failed ($N): not merged"
import json, sys
d = sys.argv[1]; r1, r2 = json.load(open(f"{d}/r1.json")), json.load(open(f"{d}/r2.json"))
ok2 = r2["ego_max_m"] <= 0.3 and r2["frame_err_max_m"] <= 0.1 and r2["act_share_within"] >= 0.995 and r2["act_median_m"] <= 0.05   # prereg amendment 1
print("R1 (registered)", r1["pass"], "R2 (amended)", ok2, "ego_max", r2["ego_max_m"], "act_share", r2["act_share_within"], "frame excluded", len(r2["frame_excluded"]))
q = next(iter(r1["tags"].values()))
ok1 = r1["a"] and r1["b"] and (r1["c"] or q["control_mu_share"] <= q["mu_share"] - 0.2)          # prereg amendment 3: (c) by separation on standing-heavy pools
print("R1 (a, b, c / amended c)", r1["a"], r1["b"], r1["c"], ok1, "mu_share", q["mu_share"], "control_mu_share", q["control_mu_share"], "control_pose_share", q["control_pose_share"])
sys.exit(0 if ok1 and ok2 else 2)
PY
  ;;
route|anchor)   # route / anchor features of the pool rows (rm_build pool / anchor --pool), the shard's own scene loader (EV_CFG = a run's config)
  N=$2; c=$(python3 -c "import json; print(json.load(open('$W/stage/$N/manifest.json')).popitem()[1][0])")
  [[ -f $c/generated-user-config-0.yaml ]] || die "no config in $c"
  if [[ $1 == route ]]; then EV_CFG=$c/generated-user-config-0.yaml $AP $RB pool --name $N --workers 8 || die "rm_build pool rc $?"
  else EV_CFG=$c/generated-user-config-0.yaml $AP $RB anchor --pool $N --workers 8 || die "rm_build anchor rc $?"; fi ;;
strip)    # ol_chain strip: drop the slot frames (`cur`) of the pool's dumps after the tokens
  N=$2; [[ -f $DATA_DIR/runs/op_parity/cache/on1_$N@warp/front.npy ]] || die "no tokens for $N"
  for d in $(python3 -c "import json; print(' '.join(v[0] for v in json.load(open('$W/stage/$N/manifest.json')).values()))"); do
  $TR - "$d" <<'PY' || die "strip rc $?"
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
fs = sorted(glob.glob(f"{sys.argv[1]}/driver-logs/dump/*.npz"))
with ProcessPoolExecutor(16) as ex:
    print(sys.argv[1], "stripped", sum(ex.map(one, fs, chunksize=64)), "of", len(fs))
PY
  done ;;
merge)    # on1_JTX = on1_JT's parts (same order: L18NR1v2 stays the prefix raster) + the shard pools; .a features; draw weights JTX00 (= JT00's rule)
  P=(); A=(); for k in $(shards); do P+=(on1_PN$k:pn$k:self); A+=(on1_PN$k:pn$k); done
  $TR experiments/route_follow/scripts/rf_pait.py merge --pools $JTPARTS ${P[@]} --out JTX --nan-keys fut_reg || die "merge rc $?"
  $AP $RB anchor --merge on1_JTX=$JTA,$(csv ${A[@]}) || die "anchor merge rc $?"
  $TR experiments/defect_fix/scripts/jt.py weights --pool on1_JTX --k-s 32 --k-tv 38 --hold-pct 15 --tv D10 --no-wait --out JTX00 || die "weights rc $?"
  traincmd $PILOT on1_JTX JTX00
  diff <(tr ' ' '\n' < $W/cmd/$CTRL.txt) <(tr ' ' '\n' < $W/cmd/$PILOT.txt) > $W/cmd/diff.txt; cat $W/cmd/diff.txt
  [[ $(grep -c '^[<>]' $W/cmd/diff.txt) -eq 6 ]] || die "pilot and control commands differ beyond tag / pool / wfile" ;;
train)    # train <tag>: the command written at submit time (traincmd)
  [[ -f $W/cmd/$2.txt ]] || die "no command for $2"; exec $(cat $W/cmd/$2.txt) ;;
*) sed -n 2,13p $0; exit 2 ;;
esac
