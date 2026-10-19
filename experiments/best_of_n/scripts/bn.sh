#!/usr/bin/env bash
# BEST-OF-N lane (plans/2026-10-19-bn-prereg.md). Box, repo root; every stage is a pool job (priority PRIO, 12), --after chained;
# ids in $W/jobs.txt, stage lines in $W/stages.txt.
#   bn.sh select                 job body: bn.py select (windows, split, lists)
#   bn.sh smoke                  staged launch 1: the 10-window smoke list x 4 models -> pick (no gap) -> geo -> tok -> route / anchor / strip ->
#                                target (pool BNSM<k>, never merged) -> $W/smoke_done
#   bn.sh launch                 staged launch 2 + the control: control BNC2K-L18-s1 (JTN-L18-s1 + 2 000 steps of JTN's recipe, on1_JT / JT00)
#                                -> its reads; the first two roll chunks x 4 models -> stage2 checklist (bn.py stage2; on pass it runs `bn.sh rest`)
#   bn.sh rest                   the remaining roll chunks -> pick (gap 0.2), which runs `bn.sh post`
#   bn.sh post                   per shard geo -> tok -> route / anchor / strip; merge (target, on1_BNX, .a, weights BNX, command diff) ->
#                                BNX2K-L18-s1 -> its reads -> gate (bn.py gate) -> $W/DONE
#   bn.sh roll <tag> <chunk|smoke> | geo|tok|route|anchor|strip <pool> | merge | train <tag> | gate     job bodies
#   bn.sh wait <sentinel>        blocks until $W/<sentinel>, $W/ERROR or a failed job of jobs.txt (one background blocker; no agent polling)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
E=experiments/best_of_n; ME=$E/scripts/bn.sh; PY=$E/scripts/bn.py; W=$DATA_DIR/runs/best_of_n; S=$W/select
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
ON=experiments/onpol_nuplan/scripts/on_rows.py; RB=experiments/route_mem/scripts/rm_build.py; JT=experiments/defect_fix/scripts/jt.sh
AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata; CK=$DATA_DIR/runs/op_parity/runs
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; TOKT=$DATA_DIR/runs/practice_nuplan/select/tok_times.json; SPL=alpasim_nuplan/best-of-n
TAGS="JTN-L18-s1 NTV0-L18-s0 RSG-L18-s0 NR1FB10-L18-s0"; BASE=JTN-L18-s1; ARM=BNX2K-L18-s1; CTRL=BNC2K-L18-s1; STEPS=2000
JTPARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"   # on1_JT's parts in order (jt.sh PARTS)
JTA="on1_L18RB0:rb0,on1_L18RTB1:rtb1,on1_L18YS:ys,on1_L18Y:y,on1_NR1TV:tv"
LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2
mkdir -p $W/pool
die() { echo "$(date '+%F %T') ERROR $*" >&2; echo "$(date '+%F %T') $*" >> $W/ERROR; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner best-of-n --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $W/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $W/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
short() { case $1 in JTN-L18-s1) echo jtn1;; NTV0-L18-s0) echo ntv0;; RSG-L18-s0) echo rsg;; NR1FB10-L18-s0) echo fb10;; esac; }
nchunks() { python3 -c "import json; print(len(json.load(open('$S/meta.json'))['chunks']))"; }
shard_of() { python3 -c "import json; print(next(c['shard'] for c in json.load(open('$S/meta.json'))['chunks'] if c['chunk'] == $1))"; }
RO=(--vram 32 --cpu 8 --ram 60 --timeout-h 3 --env CL_HIST_KEY=pn-roll)
RD=(--vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train --timeout-h 1)
rolls() {  # rolls <chunk ..>: every model on every chunk -> ids
  local c t; for c in "$@"; do for t in $TAGS; do sub bn-roll-$(short $t)-$c "${RO[@]}" -- bash $ME roll $t $c || exit 1; done; done; }
chain() {  # chain <pool> <after>: geo -> tok -> route -> anchor; strip after tok. Prints "tok anchor strip" ids
  local N=$1 A=$2 g k r n s
  g=$(sub bn-geo-$N --after $A --vram 0.1 --cpu 24 --ram 64 --timeout-h 2 -- bash $ME geo $N) || exit 1
  k=$(sub bn-tok-$N --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME tok $N) || exit 1
  r=$(sub bn-route-$N --after $k --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME route $N) || exit 1
  n=$(sub bn-anchor-$N --after $r --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME anchor $N) || exit 1
  s=$(sub bn-strip-$N --after $k --vram 0.1 --cpu 16 --ram 16 --timeout-h 2 -- bash $ME strip $N) || exit 1
  echo "$k $n $s"; }
reads() {  # reads <tag> <after>: arena held native (jt.sh aread off, 18 chunks) + held 460 @18 (6) -> ids
  local c i nch; nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for c in $(seq 0 $((nch - 1))); do TRIES=2 sub bn-ar-$1-c$c --after $2 "${RD[@]}" -- bash $JT aread $1 off $c || exit 1; done
  for i in 0 1 2 3 4 5; do TRIES=2 sub bn-held-$1-$i --after $2 "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $1 $i 6 || exit 1; done; }
traincmd() {  # traincmd <tag> <pool> <wfile> -> $W/cmd/<tag>.txt: jt.sh's JTN command (DRY), STEPS steps, warm-up 50, from JTN-L18-s1, seed 1
  mkdir -p $W/cmd; DRY=1 ARCROWS=brake WARM=50 bash $JT train $1 $2 $LAT $3 $STEPS 0 | tail -1 | sed -e "s/ --init RB0-F-s0 / --init $BASE /" -e 's/ --seed 0$/ --seed 1/' > $W/cmd/$1.txt
  grep -q "rm_train.py train --tag $1 --init $BASE --allow-init .* --steps $STEPS --seed 1$" $W/cmd/$1.txt || die "bad command for $1: $(cat $W/cmd/$1.txt)"; }
case ${1:-} in
select) $AP $PY select ;;
smoke)
  [[ -f $S/meta.json ]] || die "run select first"
  k=$(python3 -c "import json; print(json.load(open('$S/meta.json'))['smoke_shard'])")
  rr=($(rolls smoke)) || exit 1
  p=$(sub bn-pick-smoke --after $(csv ${rr[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $AP $PY pick --label BNSM --chunks smoke --gap 0) || exit 1
  ids=($(chain BNSM$k $p)) || exit 1
  t=$(sub bn-target-smoke --after ${ids[1]} --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash -c "$TR $PY target --pools on1_BNSM$k && date > $W/smoke_done") || exit 1
  echo "$(date '+%F %T') smoke: rolls ${rr[*]} pick $p tok/anchor/strip ${ids[*]} target $t" | tee -a $W/stages.txt ;;
launch)
  [[ -f $S/meta.json && -f $W/smoke_ok ]] || die "select + an inspected smoke first ($W/smoke_ok)"
  for x in $ARM $CTRL; do [[ ! -e $CK/$x/ckpt-final.pt ]] || die "$x exists"; done
  traincmd $CTRL on1_JT JT00
  c=$(sub bn-train-$CTRL "${FT[@]}" -- bash $ME train $CTRL) || exit 1
  cr=($(reads $CTRL $c)) || exit 1
  rr=($(rolls c0 c1)) || exit 1
  n=$(python3 -c "import json; print(json.load(open('$S/meta.json'))['kept'] * 4)")
  runs=$(for x in c0 c1; do for t in $TAGS; do echo -n "$W/roll/$t/$x/run "; done; done)
  g=$(sub bn-stage2 --after $(csv ${rr[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 --env CTRL_READS=$(csv ${cr[@]}) -- \
      bash -c "$AP $PY stage2 --runs $runs --total-rollouts $n > $W/stage2.json && bash $ME rest") || exit 1
  echo "$(date '+%F %T') launch: control $c reads ${cr[*]}; stage-2 rolls ${rr[*]}; stage2 + rest $g" | tee -a $W/stages.txt ;;
rest)     # run by the stage2 job (CTRL_READS: the control's read ids)
  ch=(); for j in $(seq 2 $(($(nchunks) - 1))); do ch+=(c$j); done
  rr=($(rolls ${ch[@]})) || exit 1
  all=(c0 c1 ${ch[@]})
  p=$(sub bn-pick --after $(csv ${rr[@]}) --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 --env CTRL_READS=${CTRL_READS:?} -- \
      bash -c "$AP $PY pick --label BN --chunks ${all[*]} && bash $ME post") || exit 1
  echo "$(date '+%F %T') rest: rolls ${rr[*]}; pick + post $p" | tee -a $W/stages.txt ;;
post)     # run by the pick job: the shards with winners (pick/BN_shards.txt)
  ids=(); for k in $(cat $W/pick/BN_shards.txt); do ids+=($(chain BN$k ${CL_POOL_JOB:-})) || exit 1; done
  m=$(sub bn-merge --after $(csv ${ids[@]}) --vram 0.1 --cpu 4 --ram 64 --timeout-h 2 -- bash $ME merge) || exit 1
  t=$(sub bn-train-$ARM --after $m "${FT[@]}" -- bash $ME train $ARM) || exit 1
  ar=($(reads $ARM $t)) || exit 1
  q=$(sub bn-gate --after $(csv ${ar[@]}),${CTRL_READS:?} --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME gate) || exit 1
  echo "$(date '+%F %T') post: shard chains ${ids[*]}; merge $m; arm $t reads ${ar[*]}; gate $q" | tee -a $W/stages.txt ;;
roll)     # roll <tag> <chunk|smoke>: the shard's arena cache, sh30 (no hold, = the default nuPlan serve of the L18 pools), frame dumps (= pn.sh roll)
  T=$2; j=$3; L=$S/lists/$j.txt; k=$( [[ $j == smoke ]] && python3 -c "import json; print(json.load(open('$S/meta.json'))['smoke_shard'])" || shard_of ${j#c})
  D=$W/roll/$T/$j/run
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  [[ -z $(comm -12 <(sort $L) <(sort experiments/onpol_nuplan/results/split/held.txt)) ]] || die "$L has a held scene"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf "$D"; mkdir -p "$D"
  env -u JEV_HOLD SH30_TAG=$T SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "$KEY.cache_location=$AC/s$k" \
    || die "run.sh rc $? ($T $j)"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary ($T $j)"
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
  echo "$(date '+%F %T') roll $T $j (shard $k) done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
geo)      # geo <pool>: rows of the winners' rollouts (on_rows.py geo unchanged, fed the view dirs of bn.py pick)
  N=$2; [[ -f $W/stage/$N/manifest.json ]] || die "no manifest for $N"
  ON_SPLIT=$SPL ON_TOK_TIMES=$TOKT $NAV $ON geo --name $N --roll $W/stage/$N --workers ${BN_WORKERS:-24} || die "on_rows geo rc $? ($N)" ;;
tok)      # tok <pool>: tokens + teacher + R1 (on_rows.py tok, control smoke10e); the pool gates are applied as draw weights in merge (prereg 5)
  N=$2; ON_SPLIT=$SPL $TR $ON tok --name $N --control smoke10e || die "on_rows tok rc $? ($N)"
  python3 - $DATA_DIR/runs/onpol_nuplan/rows/$N <<'PY'
import json, sys
d = sys.argv[1]; r1, r2 = json.load(open(f"{d}/r1.json")), json.load(open(f"{d}/r2.json"))
ok2 = r2["ego_max_m"] <= 0.3 and r2["frame_err_max_m"] <= 0.1 and r2["act_share_within"] >= 0.995 and r2["act_median_m"] <= 0.05
print("R2 (PN amendment 1)", ok2, "ego_max", r2["ego_max_m"], "frame_err_max", r2["frame_err_max_m"], "act_share", r2["act_share_within"], "act_median", r2["act_median_m"])
print("R1 b", r1["b"], {t: (q["n"], round(q["mu_share"], 4), q["control"], q["control_mu_share"]) for t, q in r1["tags"].items()})
PY
  ;;
route|anchor)   # route / anchor features (rm_build pool / anchor --pool), the shard's own scene loader (EV_CFG = a run's config)
  N=$2; c=$(python3 -c "import json; print(list(json.load(open('$W/stage/$N/manifest.json')).values())[0][0])")
  [[ -f $c/generated-user-config-0.yaml ]] || die "no config in $c"
  if [[ $1 == route ]]; then EV_CFG=$c/generated-user-config-0.yaml $AP $RB pool --name $N --workers 8 || die "rm_build pool rc $? ($N)"
  else EV_CFG=$c/generated-user-config-0.yaml $AP $RB anchor --pool $N --workers 8 || die "rm_build anchor rc $? ($N)"; fi ;;
strip)    # drop the slot frames (`cur`) of the winners' dumps after the tokens (= pn.sh strip)
  N=$2; [[ -f $DATA_DIR/runs/op_parity/cache/on1_$N@warp/front.npy ]] || die "no tokens for $N"
  $TR - $DATA_DIR/runs/onpol_nuplan/rows/$N/meta.npz <<'PY' || die "strip rc $?"
import os, sys
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
fs = sorted({os.path.realpath(x) for x in np.load(sys.argv[1])["dump"].astype(str)})
with ProcessPoolExecutor(16) as ex:
    print("stripped", sum(ex.map(one, fs, chunksize=64)), "of", len(fs))
PY
  ;;
merge)    # target (fut := served) -> on1_BNX = on1_JT's parts + BN<k> (both) -> .a features -> draw weights BNX -> the arm's command, diffed with the control's
  P=(); A=(); for k in $(cat $W/pick/BN_shards.txt); do P+=(on1_BN$k:bn$k:both); A+=(on1_BN$k:bn$k); done
  $TR $PY target --pools ${P[@]%%:*} || die "target rc $?"
  $TR experiments/route_follow/scripts/rf_pait.py merge --pools $JTPARTS ${P[@]} --out BNX --nan-keys fut_reg || die "merge rc $?"
  $AP $RB anchor --merge on1_BNX=$JTA,$(csv ${A[@]}) || die "anchor merge rc $?"
  $TR $PY weights --pool on1_BNX --base JT00 --parts ${P[@]} --out BNX || die "weights rc $?"
  traincmd $ARM on1_BNX BNX
  diff <(tr ' ' '\n' < $W/cmd/$CTRL.txt) <(tr ' ' '\n' < $W/cmd/$ARM.txt) > $W/cmd/diff.txt; cat $W/cmd/diff.txt
  [[ $(grep -c '^[<>]' $W/cmd/diff.txt) -eq 6 ]] || die "arm and control commands differ beyond tag / pool / wfile" ;;
train)    # train <tag>: the command written at submit time (traincmd)
  [[ -f $W/cmd/$2.txt ]] || die "no command for $2"; echo "$(date '+%F %T') train $2: $(cat $W/cmd/$2.txt)" >> $W/stages.txt
  exec env ARCROWS=brake $(cat $W/cmd/$2.txt) ;;
gate)
  $TR $PY gate --arm $ARM --control $CTRL --base $BASE || die "gate rc $?"
  date '+%F %T' > $W/DONE ;;
wait)     # wait <sentinel>
  while :; do
    [[ -f $W/$2 ]] && { echo "$2"; cat $W/$2; exit 0; }
    [[ -f $W/ERROR ]] && { echo ERROR; cat $W/ERROR; exit 1; }
    f=$(.venv/bin/python - $W/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
ids = dict(l.split()[:2] for l in open(sys.argv[1]) if l.strip())
st = P.job_states(list(ids))
print(" ".join(f"{n}({i})" for i, n in ids.items() if st[i] in ("failed", "cancelled")))
PY
)
    [[ -n $f ]] && { echo "FAILED: $f"; exit 1; }
    sleep 120
  done ;;
*) sed -n 2,14p $0; exit 2 ;;
esac
