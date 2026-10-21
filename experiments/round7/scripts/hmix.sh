#!/usr/bin/env bash
# HARM-MIX (plans/2026-10-22-harmonizer-mix.md): Harmonizer-ON twins of the PAI human-state rows (on1_PAIH / PAIH05 / PAIH15) on the scenes of
# hmix.py select, rendered with pai_rows' H recipe (pr.sh: onp_log_driver, A0 serve, g10 + recording ground truth, NTV0-L18-s0 in the loop,
# FAST, CONC 4) + HARMONIZER=1 -> onp_rows -> onp_cache build -> rf_pai_pool (on1_PAIHON, names on:paih:..) -> pr.py stretch 0.5 / 1.5
# (on1_PAIHON05 / 15); the offcheck scenes once more OFF (on1_PAIHOFFC) -> hmix.py gates (A-D) -> hmix.py twin (on1_PAIHTW) -> DONE.
# Trainer switch rm_train --twin (lib/twin_tokens.py): identity at R6DAG's argv. Box, repo root; every job through the pool at PRIO 24.
#   hmix.sh chain        select (inline) -> ON c000 roll -> check (gate B on its log) -> c001 + the OFF check chunk -> rest (job: every other chunk in a
#                        window of W, then the tail)
#   hmix.sh ident        identity: R6DAG-L18-s0 argv 60 steps eager at $PRE (worktree) vs HEAD without --twin -> results/harm_mix/identity.json
#   hmix.sh tident       after DONE: HEAD without --twin vs HEAD --twin PAIHTW:0 (identity), and a --twin PAIHTW:0.5 smoke -> results/harm_mix/
#   hmix.sh wait [M]     blocks until $O/<M> (glob; default DONE; IDENT_<out>.<rc>, SMOKE), $O/STOP or a failed job of this lane (one run_in_background waiter)
#   job bodies: check <c> | rows <V> <c> | clean <V> <c> | rest "<chunks>" <after> | cache <V> | pool | poolc | stretch <f> | gates | twin |
#               identrun a|b|c|s | identcmp <x> <y> <out> | smokechk
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round7/hmix; ME=experiments/round7/scripts/hmix.sh; PY=experiments/round7/scripts/hmix.py; PR=experiments/pai_rows/scripts/pr.py
TR=$DATA_DIR/envs/op-train/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python; V=.venv/bin/python; RF=experiments/route_follow/scripts
T=NTV0-L18-s0; PRIO=${PRIO:-24}; W=${W:-6}; RES=$O/res; RUNS=$DATA_DIR/runs/op_parity/runs; DS=experiments/round6/scripts/dag.sh
PRE=${HMIX_PRE:-287194b8}                                          # the last commit before rm_train --twin (identity run a)
# RES: result files stay outside the repo tree on the box (copied into results/harm_mix/ on the Mac), so box pulls are never blocked
LOGSERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 PAI_DUMP=100000 DRV_PY=../../onpol_pai/lib/onp_log_driver.py"
WIZ="controller.gains.rel_front_steering_angle_weight=10 runtime.simulation_config.send_recording_ground_truth=true"
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
stop() { echo "$(date '+%F %T') $*" | tee $O/STOP >> $O/stages.txt; exit 2; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round7-hmix --name "$name" --tries 1 --priority $PRIO --no-check \
     --log-dir $O/pool/$name.$(date +%m%d%H%M%S) "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
roll() {  # roll <V ON|OFF> <chunk> [after] -> the chunk's job id; each chunk its own read root (pr.sh's layout)
  local v=$1 c=$2 r=$O/roll/$1/$2 id e; mkdir -p $r/chunks; [[ $c == x000 ]] && cp $O/offcheck.txt $r/chunks/$c.txt || cp $O/chunks/$c.txt $r/chunks/
  e=(SERVE="$LOGSERVE" WIZ="$WIZ" KEEP_ASL=1 CONC=4 PRIO=$PRIO AFTER=${3:-})
  [[ $v == ON ]] && e=(SERVE="$LOGSERVE HARMONIZER=1" WIZ="$WIZ" KEEP_ASL=1 CONC=4 PRIO=$PRIO AFTER=${3:-} VRAM=30 RAM=40)
  env "${e[@]}" bash experiments/alpasim/scripts/pai_box.sh submit $r $T > $r/submit.txt || die "roll submit $v $c"
  id=$(awk '$2 ~ /^pai-/ {print $1}' $r/submit.txt); [[ -n $id && $(wc -w <<< "$id") == 1 ]] || die "roll $v $c: no single job"
  echo "$id pai-$v-$c" >> $O/jobs.txt; echo "$id"; }
unit() {  # unit <V> <chunk> <after roll> <after rows>: roll -> rows -> clean; prints "roll rows clean"
  local r w c; r=$(roll $1 $2 $3) || exit 1
  w=$(sub hmix-rows-$1-$2 --after $(csv $r ${4:-}) --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows $1 $2) || exit 1
  c=$(sub hmix-clean-$1-$2 --after $w --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean $1 $2) || exit 1; echo "$r $w $c"; }
run() { local X; X=$(bash $DS printargv dag "$@") || die "R6DAG argv"; mapfile -t A <<< "$X"; }
case ${1:-} in
chain)
  [[ ! -d $O/roll ]] || die "$O/roll exists: chain runs once"
  git rev-parse --short HEAD >> $O/code.txt; $TR $PY select --out $O > $O/select.log || die select
  ch=($(cd $O/chunks && ls c*.txt | sed 's/.txt$//')); [[ ${#ch[@]} -ge 3 ]] || die "chunks"
  r0=$(roll ON c000) || exit 1
  w0=$(sub hmix-rows-ON-c000 --after $r0 --vram 0.5 --cpu 8 --ram 40 --timeout-h 3 -- bash $ME rows ON c000) || exit 1
  k0=$(sub hmix-check-c000 --after $w0 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME check c000) || exit 1
  c0=$(sub hmix-clean-ON-c000 --after $k0 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME clean ON c000) || exit 1
  u1=($(unit ON c001 $k0 "")) || exit 1
  ux=($(unit OFF x000 $k0 "")) || exit 1
  rs=$(sub hmix-rest --after ${u1[1]} --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME rest "${ch[*]:2}" $(csv $w0 $c0 ${u1[1]} ${u1[2]} ${ux[1]} ${ux[2]})) || exit 1
  echo "$(date '+%F %T') chain (code $(git rev-parse --short HEAD)): c000 roll $r0 rows $w0 check $k0 clean $c0; c001 ${u1[*]}; OFF x000 ${ux[*]}; rest $rs" | tee -a $O/stages.txt ;;
rest)  # rest "<chunks>" <after ids>: the remaining ON chunks (W at once hold rollout.asl / dumps), then the tail
  [[ ! -f $O/STOP ]] || exit 2; ch=($2); rw=(${3//,/ }); cl=()
  for i in "${!ch[@]}"; do a=""; (( i >= W )) && a=${cl[$((i - W))]}
    u=($(unit ON ${ch[$i]} "$a" "")) || exit 1; rw+=(${u[1]} ${u[2]}); cl+=(${u[2]}); done
  a=$(csv ${rw[@]})
  ca=$(sub hmix-cache-ON --after $a --vram 20 --cpu 8 --ram 64 --timeout-h 4 -- bash $ME cache ON) || exit 1
  cb=$(sub hmix-cache-OFF --after $a --vram 20 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME cache OFF) || exit 1
  po=$(sub hmix-pool --after $ca --vram 0.1 --cpu 8 --ram 96 --timeout-h 3 -- bash $ME pool) || exit 1
  pc=$(sub hmix-poolc --after $cb --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME poolc) || exit 1
  s5=$(sub hmix-stretch05 --after $po --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 0.5) || exit 1
  s15=$(sub hmix-stretch15 --after $po --vram 12 --cpu 8 --ram 96 --timeout-h 4 -- bash $ME stretch 1.5) || exit 1
  g=$(sub hmix-gates --after $(csv $s5 $s15 $pc) --vram 0.1 --cpu 8 --ram 96 --timeout-h 2 -- bash $ME gates) || exit 1
  t=$(sub hmix-twin --after $g --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME twin) || exit 1
  echo "$(date '+%F %T') rest: ${#ch[@]} chunks (window $W), cleans ${cl[*]}; cache $ca $cb pool $po $pc stretch $s5 $s15 gates $g twin $t" | tee -a $O/stages.txt ;;
check)  # gate B early: the chunk's renderer log has the Harmonizer enabled and its checkpoint loaded, and the chunk gave its row file
  g=$(ls $O/roll/ON/$2/runs/*/sim/native-logs/renderer-*.log 2>/dev/null | head -1); [[ -n $g ]] || stop "check $2: no renderer log"
  grep -q '"enable_harmonizer": true' $g && grep -q 'checkpoint ready at .*harmonizer_nontemporal.pt' $g || stop "check $2: harmonizer NOT confirmed in $g"
  n=$(ls $O/rows/ON/*.npz 2>/dev/null | wc -l); (( n >= 1 )) || stop "check $2: no row file"
  echo "$(date '+%F %T') check $2: harmonizer loaded ($g), $n row file(s)" | tee -a $O/stages.txt ;;
rows)  # rows <V> <chunk>: the chunk's rollouts -> $O/rows/<V>_<chunk>/<scene>.npz, linked into $O/rows/<V>/ (pr.sh rows)
  v=$2 c=$3 P=$2_$3; G=$DATA_DIR/tmp/hmix_stage/$P; R=$O/rows; mkdir -p $G $R/$v
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD $AP experiments/onpol_pai/scripts/onp_rows.py rows --pass $P \
      --runs "$O/roll/$v/$c/runs/*" --out $R --stage $G --tag $T --jobs 8 || exit 1
  env PAI_DUMP=100000 JEV_REPO=$PWD PYTHONPATH=$PWD/experiments/alpasim/lib:$PWD $TR experiments/onpol_pai/scripts/onp_rows.py cold --pass $P --out $R --stage $G --jobs 8 || exit 1
  rm -rf $G; for f in $R/$P/*.npz; do [[ -e $f ]] && ln -sfn $f $R/$v/$(basename $f); done
  n=$(ls $R/$P/*.npz 2>/dev/null | wc -l); m=$(wc -w < $O/roll/$v/$c/chunks/$c.txt); echo "$(date '+%F %T') rows $v $c: $n / $m scene files" | tee -a $O/stages.txt
  (( n >= m * 9 / 10 )) || die "rows $v $c: $n of $m scenes" ;;
clean)  # clean <V> <chunk>: rollout.asl and frame dumps of the chunk (rows built); drive.jsonl (route messages) and renderer logs stay
  r=$O/roll/$2/$3; n=$(find $r/runs -name rollout.asl | wc -l); find $r/runs -name rollout.asl -delete; rm -rf $r/runs/*/sim/driver-logs/dump
  echo "$(date '+%F %T') clean $2 $3: removed $n rollout.asl and the frame dumps; $(du -sh $r | cut -f1) left; free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
cache) $TR experiments/onpol_pai/scripts/onp_cache.py build --pass $2 --rows $O/rows/$2 --out $O/cache ;;
pool)  # on1_PAIHON: pai_rows H's pool recipe on the ON rows (names on:paih:<scene>:<k>, as on1_PAIH)
  $TR $RF/rf_pai_pool.py --name PAIHON --rows $O/rows/ON --cache $O/cache/ON --runs "$O/roll/ON/*/runs/*" --split $O/scenes.txt --fut log_fut --prefix paih --src human ;;
poolc) $TR $RF/rf_pai_pool.py --name PAIHOFFC --rows $O/rows/OFF --cache $O/cache/OFF --runs "$O/roll/OFF/*/runs/*" --split $O/offcheck.txt --fut log_fut --prefix paih --src human ;;
stretch) $TR $PR stretch --pool PAIHON --rows $O/rows/ON --f $2 --name PAIHON$(tr -d . <<< $2) ;;
gates) $TR $PY gates --out $O > $O/gates.log 2>&1; r=$?; cat $O/gates.log; cp $O/gates.json $RES/gates.json; (( r == 0 )) || die "gates crashed"
  python3 -c "import json, sys; g = json.load(open('$O/gates.json'))['pass']; sys.exit(0 if all(g.values()) else 1)" || stop "gates not all passed: $(python3 -c "import json; print(json.load(open('$O/gates.json'))['pass'])")" ;;
twin) $TR $PY twin --out $O --name PAIHTW | tee $RES/twin_build.json && date '+%F %T' > $O/DONE ;;
ident)
  ia=$(sub hmix-ident-a "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun a) || exit 1
  ib=$(sub hmix-ident-b "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun b) || exit 1
  ic=$(sub hmix-ident-cmp --after $ia,$ib --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME identcmp HMIDA-L18-s0 HMIDB-L18-s0 identity) || exit 1
  echo "$(date '+%F %T') ident (code $(git rev-parse --short HEAD), pre $PRE): $ia $ib cmp $ic" | tee -a $O/stages.txt ;;
tident)
  [[ -f $O/DONE ]] || die "no twin store yet"
  ib=$(sub hmix-tident-b "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun b2) || exit 1
  ic=$(sub hmix-tident-c "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun c) || exit 1
  is=$(sub hmix-tsmoke "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun s) || exit 1
  i1=$(sub hmix-tsmoke1 "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun s1) || exit 1
  ix=$(sub hmix-tident-cmp --after $ib,$ic --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME identcmp HMIDB2-L18-s0 HMIDC-L18-s0 identity_p0) || exit 1
  sk=$(sub hmix-tsmoke-chk --after $is,$i1 --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME smokechk) || exit 1
  echo "$(date '+%F %T') tident (code $(git rev-parse --short HEAD)): $ib $ic $is $i1 cmp $ix smoke $sk" | tee -a $O/stages.txt ;;
identrun)  # identrun a | b | b2 | c | s | s1: 60 steps eager, R6DAG-L18-s0's argv; a from a worktree at $PRE; c / s / s1 = --twin PAIHTW:0 / 0.5 / 1
  declare -A IX=([a]=HMIDA [b]=HMIDB [b2]=HMIDB2 [c]=HMIDC [s]=HMSM [s1]=HMSM1); X=${IX[$2]}-L18-s0; run 60 $X 0
  [[ $2 == c ]] && A+=(--twin PAIHTW:0); [[ $2 == s ]] && A+=(--twin PAIHTW:0.5); [[ $2 == s1 ]] && A+=(--twin PAIHTW:1.0)
  if [[ $2 == a ]]; then WT=$DATA_DIR/tmp/hmix_wt_$PRE; [[ -d $WT ]] || git worktree add --detach $WT $PRE || die "worktree $PRE"
    [[ $(git -C $WT rev-parse --short HEAD) == ${PRE:0:8} ]] || die "worktree is not at $PRE"; cd $WT; fi
  [[ ! -e $RUNS/$X ]] || die "$X exists"
  echo "$(date '+%F %T') identity $2 ($(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
identcmp)  # identcmp <tag a> <tag b> <out>
  $TR experiments/defect_fix/scripts/df.py bitcmp --a $2 --b $3 --out round7_hmix_$4 --note "R6DAG-L18-s0 argv 60 steps eager: $2 vs $3 (rm_train --twin)"; r=$?
  mv experiments/defect_fix/results/round7_hmix_$4.json $RES/$4.json || r=4
  date '+%F %T' > $O/IDENT_$4.$r; (( r == 0 )) && rm -rf $RUNS/$2 $RUNS/$3; exit $r ;;
smokechk)  # the p = 0.5 / 1 smokes: finished 60 steps, swap share within 0.05 of p, twin rows drawn within 25 % of the expected 60 x rows a batch
  $TR - $DATA_DIR/runs/route_mem $RES/smoke.json <<'PY' || die "smoke check"
import glob, json, sys
out = {}
for tag, p in (("HMSM", 0.5), ("HMSM1", 1.0)):
    d = sorted(glob.glob(f"{sys.argv[1]}/train-{tag}-L18-s0/*/"))[-1]
    s = json.load(open(d + "DONE")) if glob.glob(d + "DONE") else {}        # jevdrive.run: DONE = wall time + the run's summary
    tw = s.get("twin", {})
    exp = 60 * tw.get("twin_rows_per_batch", 0)
    ok = bool(tw) and tw["drawn_with_twin"] > 0 and abs(tw["swap_share"] - p) < 0.05 and abs(tw["drawn_with_twin"] / exp - 1) < 0.25 and s.get("steps") == 60
    out[tag] = dict(run=d, p=p, twin=tw, expected_drawn=exp, steps=s.get("steps"), loss_last=s.get("loss"), pass_=ok)
out["pass_"] = all(v["pass_"] for v in out.values())
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
sys.exit(0 if out["pass_"] else 1)
PY
  rm -rf $RUNS/HMSM-L18-s0 $RUNS/HMSM1-L18-s0; date '+%F %T' > $O/SMOKE ;;
wait)
  M=${2:-DONE}
  until compgen -G "$O/$M" > /dev/null || [[ -f $O/STOP ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
print("failed:", *bad, sep="\n  ", file=sys.stderr) if bad else None
sys.exit(1 if bad else 0)
PY
  do sleep 900; done
  for f in $O/$M; do [[ -f $f ]] && echo "$(basename $f) $(cat $f)"; done; [[ -f $O/STOP ]] && echo "STOP $(cat $O/STOP)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
