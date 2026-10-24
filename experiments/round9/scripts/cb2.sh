#!/usr/bin/env bash
# ROUND9 CB2 (clean B2; plans/2026-10-25-cb2.md = the registration): a clean nuPlan model from the shipped weights that ties B2 (STANDFMXH-L18-s1).
# Box, repo root; every job through the GPU pool; ids in $O/jobs.txt, stages in $O/stages.txt, lane dir $O = $DATA_DIR/runs/round9/cb2.
#   cb2.sh step0 | creep   offline step 0 (before registration): b3 teacher on existing nuPlan rows -> cb2.py creep -> $O/STEP0 / STEP0B
#   cb2.sh s1           stage 1: B2-s0 / s1 native re-reads (ASL kept) + RB0-F-s0 practice-starts roll c0 -> c0chk -> s1rest (job) submits the rest:
#                       rolls c1-45, shard chains, merge on1_JTPR, CB2S, gates, P1N training, reads, rescoring, sum1 (E1 gate; on a pass it runs s2)
#   cb2.sh s2           (run by sum1) stage 2: P1N rolls onpol-train + practice-starts -> DAgger pools (b3 + R3) -> on1_JTPRD / CB2D -> gates -> D1N
#                       -> reads, rescoring, PAI 281 side read -> sum2 (adoption line; on a pass it runs s3)
#   cb2.sh s3           (run by sum2) D1N seed 1 -> reads, rescoring -> sum3 -> $O/DONE
#   cb2.sh wait         one run_in_background waiter: $O/DONE, $O/STOP or a failed / cancelled job of this lane
#   job bodies: teach | proll | pgeo | ptok | proute | panchor | pstrip | merge1 | ident1 | smoke <1|2> | train <arm> <tag> [seed] | post <tag> |
#               aread <tag> <c> | hread <tag> <i> | v2 <bench> <key> | oroll <i> | olab <N> | dgate | append | pair <a|b> | paircmp | sum <1|2|3>
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round9/cb2; ME=experiments/round9/scripts/cb2.sh; PY=experiments/round9/scripts/cb2.py; RES=experiments/round9/results/cb2
V=.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; AS=$DATA_DIR/third_party/alpasim; AP=$AS/.venv/bin/python
NT=experiments/nuplan_dag/scripts/nd_teach.py; EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD); V2=experiments/round9/scripts/cb2v2.py
NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src; SNU=experiments/round9/scripts/snu.py; PC=experiments/round9/scripts/pcl.py
PL=experiments/round9/scripts/p1l.py; ON=experiments/onpol_nuplan/scripts/on_rows.py; RB=experiments/route_mem/scripts/rm_build.py
OL=experiments/onpol_long/scripts/ol_chain.sh; RF=experiments/route_follow/scripts
S=$DATA_DIR/runs/practice_nuplan/select; AC=$DATA_DIR/datasets/alpasim_nuplan_arena/v1; KEY=runtime.scene_provider.trajdata
J=$DATA_DIR/runs/defect_fix/jt; AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; HR=$DATA_DIR/runs/route_mem/read; W=$DATA_DIR/runs/onpol_long
LC=$DATA_DIR/datasets/alpasim_nuplan_long/after160; CK=$DATA_DIR/runs/op_parity/runs; CR=$DATA_DIR/runs/op_parity/cache; F_=$DATA_DIR/runs/pai_fix/facts
X=${CB2V:-}; export CB2V=$X   # plan amendment 2: B = P1N' (B2's S-row rule)
RB0=RB0-F-s0; P1N=CB2P1N${X}-F-s0; D1N=CB2D1N${X}-F-s0; D1N1=CB2D1N${X}-F-s1; B2S1=STANDFMXH-L18-s1; B2S0=STANDFMXH-L18-s0; T_ON=10000
JTPARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"   # on1_JT's parts in order (pn.sh)
JTA="on1_L18RB0:rb0,on1_L18RTB1:rtb1,on1_L18YS:ys,on1_L18Y:y,on1_NR1TV:tv"
PRIO=${PRIO:-14}
RO=(--vram 32 --cpu 8 --ram 60 --timeout-h 3 --env CL_HIST_KEY=pn-roll)
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)          # nuPlan reads: 4 a card
FE=(--train --vram 40 --cpu 6 --ram 40 --env CL_HIST_KEY=rm-train --env JEV_RM_EAGER=1) # eager, as P1L / D1
SERVE_PAI="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ_PAI=controller.gains.rel_front_steering_angle_weight=10   # = dg4.sh (D3's 281 read)
CH=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks
mkdir -p $O/pool $O/results $O/res $O/teach $O/stage $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round9-cb2 --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check ${X:+--env CB2V=$X} \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
nch() { python3 -c "import json; print(len(json.load(open('$S/meta.json'))['chunks']))"; }
shard_of() { python3 -c "import json; print(next(c['shard'] for c in json.load(open('$S/meta.json'))['chunks'] if c['chunk'] == $1))"; }
chunks_of() { python3 -c "import json; print(' '.join(str(c['chunk']) for c in json.load(open('$S/meta.json'))['chunks'] if c['shard'] == $1))"; }
shards() { python3 -c "import json; print(' '.join(map(str, sorted({c['shard'] for c in json.load(open('$S/meta.json'))['chunks']}))))"; }
BASE_SRC='import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")'
reads() {  # reads <tag> <after|-> -> arena 18 + held 6 job ids, then v2 jobs (arena, held) -> prints "<v2 arena id> <v2 held id>"
  local t=$1 af=${2:-} R=() c i a h
  for c in $(seq 0 17); do R+=($(TRIES=2 sub cb2-ar-$t-c$c ${af:+--after $af} "${RD[@]}" -- bash $ME aread $t $c)) || exit 1; done
  a=$(sub cb2-v2-arena-$t --after $(csv ${R[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME v2 arena $t@cb2) || exit 1
  R=(); for i in 0 1 2 3 4 5; do R+=($(TRIES=2 sub cb2-held-$t-$i ${af:+--after $af} "${RD[@]}" -- bash $ME hread $t $i)) || exit 1; done
  h=$(sub cb2-v2-held-$t --after $(csv ${R[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME v2 held460 $t-cb2) || exit 1
  echo $a $h; }
pchains() {  # pchains <policy> <rows prefix> <part prefix> <after ids>: practice-starts shard chains geo -> tok -> route / anchor, strip; prints the last ids
  local T=$1 NP=$2 PP=$3 A=$4 k g t r n st out=()
  for k in $(shards); do
    rr=(); for j in $(chunks_of $k); do rr+=($(awk -v n=cb2-proll-$T-c$j '$2 == n {i = $1} END {print i}' $O/jobs.txt)); done
    g=$(sub cb2-pgeo-$NP$k --after $(csv ${rr[@]}) --vram 0.1 --cpu 24 --ram 64 --timeout-h 2 -- bash $ME pgeo $T $NP$k $k) || exit 1
    t=$(sub cb2-ptok-$NP$k --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME ptok $NP$k) || exit 1
    r=$(sub cb2-proute-$NP$k --after $t --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME proute $NP$k) || exit 1
    n=$(sub cb2-panchor-$NP$k --after $r --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME panchor $NP$k) || exit 1
    st=$(sub cb2-pstrip-$NP$k --after $t --vram 0.1 --cpu 16 --ram 16 --timeout-h 2 -- bash $ME pstrip $NP$k) || exit 1
    out+=($n $st); done
  echo ${out[@]}; }
case ${1:-} in
step0)
  [[ ! -e $O/STEP0 ]] || die "step 0 already ran"; mkdir -p $O/step0/teach
  echo "$(git rev-parse --short HEAD) step0" >> $O/code.txt; P0="L18RB0 L18D3 PN0 PN1 PN2 PN3 PN4 PN5"; ids=()
  for n in $P0; do ids+=($(sub cb2-s0-teach-$n --vram 0.1 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME teach $n $O/step0/teach)) || exit 1; done
  c=$(sub cb2-s0-creep --after $(csv ${ids[@]}) --vram 0.1 --cpu 2 --ram 48 --timeout-h 1 -- \
      bash -c "$V $PY creep --pools $P0 --teach-dir $O/step0/teach --out $O/step0/creep && cp $O/step0/creep.json $O/step0/creep.md $RES/ && date > $O/STEP0") || exit 1
  echo "$(date '+%F %T') step0 (code $(git rev-parse --short HEAD)): teach ${ids[*]} creep $c" | tee -a $O/stages.txt ;;
creep)  # step 0 amendment (before registration): creep re-run with rule R3 added (teacher files unchanged)
  c=$(sub cb2-s0-creep2 --vram 0.1 --cpu 2 --ram 48 --timeout-h 1 -- \
      bash -c "$V $PY creep --pools L18RB0 L18D3 PN0 PN1 PN2 PN3 PN4 PN5 --teach-dir $O/step0/teach --out $O/step0/creep2 && cp $O/step0/creep2.json $O/step0/creep2.md $RES/ && date > $O/STEP0B") || exit 1
  echo "$(date '+%F %T') creep2 (code $(git rev-parse --short HEAD)): $c" | tee -a $O/stages.txt ;;
teach)  # teach <rows> <dir>: nd_teach.py teach (b3, unedited) on runs/onpol_nuplan/rows/<rows> -> <dir>/<rows>.npz
  [[ ! -e $3/$2.npz ]] || die "$3/$2.npz exists"; "${EA[@]}" $AP $NT teach --name $2 --out $3/$2.npz --workers 16 ;;
# ------------------------------------------------------------------------------------------------------------------------------ stage 1
s1)
  [[ ! -e $O/S1 ]] || die "stage 1 already submitted"
  for x in $P1N $D1N $D1N1 CB2ID1-F-s0 CB2SM1-F-s0 CB2SM2-F-s0 CB2PA-F-s0 CB2PB-F-s0; do
    [[ ! -e $CK/$x && -z $(ls -d $DATA_DIR/runs/route_mem/train-$x 2>/dev/null) ]] || die "$x exists"; done
  for p in $CR/on1_JTPR $CR/on1_JTPRD $O/proll $J/arena/$B2S1@cb2 $J/arena/$B2S0@cb2 $HR/$B2S1-cb2 $HR/$B2S0-cb2; do [[ ! -e $p ]] || die "$p exists"; done
  for k in $(shards); do for p in $CR/on1_CBR$k $CR/on1_CBP$k $DATA_DIR/runs/onpol_nuplan/rows/CBR$k; do [[ ! -e $p ]] || die "$p exists"; done; done
  [[ -s $DATA_DIR/runs/round9/pool_pilot/w/JT00S-rb0.npz && -s $CK/$RB0/ckpt-final.pt && -f $O/STEP0B ]] || die "inputs missing"
  [[ $(nch) == 46 ]] || die "practice chunks $(nch)"
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 300 )) || die "free $free GB < 300 GB"
  h=$(git rev-parse --short HEAD); echo "$h s1" >> $O/code.txt; date > $O/S1
  b=(); for t in $B2S1 $B2S0; do b+=($(reads $t)) || exit 1; done; echo ${b[@]} > $O/b2v2.ids
  r0=$(sub cb2-proll-$RB0-c0 "${RO[@]}" -- bash $ME proll $RB0 0) || exit 1
  k=$(sub cb2-c0chk1 --after $r0 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY c0chk --new $O/proll/$RB0/c0/run \
      --ref $DATA_DIR/runs/practice_nuplan/roll/JTN-L18-s0/c0/run --list $S/lists/c0.txt --out $O/results/c0chk1.json) || exit 1
  r=$(sub cb2-s1rest --after $k --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME s1rest) || exit 1
  echo "$(date '+%F %T') s1 (code $h, free $free GB): B2 re-reads + v2 ${b[*]}; roll c0 $r0 c0chk $k s1rest $r" | tee -a $O/stages.txt ;;
s1rest)
  h=$(git rev-parse --short HEAD); echo "$h s1rest" >> $O/code.txt
  for j in $(seq 1 45); do sub cb2-proll-$RB0-c$j "${RO[@]}" -- bash $ME proll $RB0 $j > /dev/null || exit 1; done
  ch=($(pchains $RB0 CBR pr "")) || exit 1
  m=$(sub cb2-merge1 --after $(csv ${ch[@]}) --vram 0.1 --cpu 4 --ram 96 --timeout-h 2 -- bash $ME merge1) || exit 1
  d=$(sub cb2-diff --after $m --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- $V $PY diff) || exit 1
  i=$(sub cb2-ident1 --after $m "${FE[@]}" --timeout-h 1 -- bash $ME ident1) || exit 1
  s=$(sub cb2-smoke1 --after $m "${FE[@]}" --timeout-h 1 -- bash $ME smoke 1) || exit 1
  t=$(sub cb2-train-$P1N --after $d,$i,$s "${FE[@]}" --timeout-h 5 -- bash $ME train P1N $P1N) || exit 1
  ra=$(sub cb2-ratio-$P1N --after $d,$i,$s --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $P1N --on-start $T_ON --train-id $t \
       --out $O/results/ratio_$P1N.json) || exit 1
  p=$(sub cb2-post-$P1N --after $t,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $P1N) || exit 1
  v=($(reads $P1N $p)) || exit 1
  su=$(sub cb2-sum1 --after $(csv ${v[@]} $(cat $O/b2v2.ids)) --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME sum 1) || exit 1
  echo "$(date '+%F %T') s1rest (code $h): chains ${ch[*]} merge $m diff $d ident $i smoke $s train $t ratio $ra post $p v2 ${v[*]} sum1 $su" | tee -a $O/stages.txt ;;
s1b)    # plan amendment 2: P1N' = P1N with CB2SB (B2's S-row rule on the same pool); wfile + STAND share -> diff, smoke -> train -> reads -> sum 1
  [[ $X == B && ! -e $O/S1$X ]] || die "s1b needs CB2V=B, once"
  for x in $P1N CB2SM1$X-F-s0; do [[ ! -e $CK/$x && -z $(ls -d $DATA_DIR/runs/route_mem/train-$x 2>/dev/null) ]] || die "$x exists"; done
  h=$(git rev-parse --short HEAD); echo "$h s1b" >> $O/code.txt; date > $O/S1$X
  w=$(sub cb2-wfile$X --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- $V $PY wfile) || exit 1
  d=$(sub cb2-diff$X --after $w --vram 0.1 --cpu 1 --ram 4 --timeout-h 1 -- $V $PY diff) || exit 1
  s=$(sub cb2-smoke1$X --after $w "${FE[@]}" --timeout-h 1 -- bash $ME smoke 1) || exit 1
  t=$(sub cb2-train-$P1N --after $d,$s "${FE[@]}" --timeout-h 5 -- bash $ME train P1N $P1N) || exit 1
  ra=$(sub cb2-ratio-$P1N --after $d,$s --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $P1N --on-start $T_ON --train-id $t \
       --out $O/results/ratio_$P1N.json) || exit 1
  p=$(sub cb2-post-$P1N --after $t,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $P1N) || exit 1
  v=($(reads $P1N $p)) || exit 1
  su=$(sub cb2-sum1$X --after $(csv ${v[@]}) --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME sum 1) || exit 1
  echo "$(date '+%F %T') s1b (code $h): wfile $w diff $d smoke $s train $t ratio $ra post $p v2 ${v[*]} sum1 $su" | tee -a $O/stages.txt ;;
proll)  # proll <policy> <chunk>: pn.sh roll's body (shard arena cache, sh30 no hold, frame dumps, c1_extract, prune), lane dir
  T=$2; j=$3; L=$S/lists/c$j.txt; k=$(shard_of $j); D=$O/proll/$T/c$j/run
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  [[ -z $(comm -12 <(sort $L) <(sort experiments/onpol_nuplan/results/split/held.txt)) ]] || die "$L has a held scene"
  source <(python3 -c "$BASE_SRC"); rm -rf "$D"; mkdir -p "$D"
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
  echo "roll $T c$j (shard $k): $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
pgeo)   # pgeo <policy> <rows> <shard>: on_rows.py geo over the shard's chunks (ON_SPLIT practice-starts, the select's tok_times; = pn.sh geo)
  T=$2; N=$3; R=$O/stage/$N; mkdir -p $R
  python3 - "$R" "$O/proll/$T" "$T" $(chunks_of $4) <<'PY' || die "roll chunks missing ($N)"
import json, os, sys
R, base, T, cs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
m = {f"{T}@c{c}": [f"{base}/c{c}/run"] for c in cs if os.path.exists(f"{base}/c{c}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", len(cs), "chunks"); sys.exit(len(m) != len(cs))
PY
  ON_SPLIT=alpasim_nuplan/practice-starts ON_TOK_TIMES=$S/tok_times.json $NAV $ON geo --name $N --roll $R --workers 24 || die "on_rows geo rc $?" ;;
ptok)   # ptok <rows>: on_rows.py tok (control smoke10e) + pn.sh's amended R1 / R2; a shard that fails them is marked GATE_FAIL and not merged
  N=$2; ON_SPLIT=alpasim_nuplan/practice-starts $TR $ON tok --name $N --control smoke10e || die "on_rows tok rc $?"
  python3 - $DATA_DIR/runs/onpol_nuplan/rows/$N > $O/stage/$N/gate.txt <<'PY'
import json, sys
d = sys.argv[1]; r1, r2 = json.load(open(f"{d}/r1.json")), json.load(open(f"{d}/r2.json"))
ok2 = r2["ego_max_m"] <= 0.3 and r2["frame_err_max_m"] <= 0.1 and r2["act_share_within"] >= 0.995 and r2["act_median_m"] <= 0.05
q = next(iter(r1["tags"].values()))
ok1 = r1["a"] and r1["b"] and (r1["c"] or q["control_mu_share"] <= q["mu_share"] - 0.2)
print("R1", ok1, r1["a"], r1["b"], r1["c"], "mu_share", q["mu_share"], "control", q["control_mu_share"], "R2", ok2, "ego_max", r2["ego_max_m"], "act_share", r2["act_share_within"])
sys.exit(0 if ok1 and ok2 else 2)
PY
  rc=$?; cat $O/stage/$N/gate.txt; (( rc == 0 )) || { touch $O/stage/$N/GATE_FAIL; echo "$(date '+%F %T') $N fails R1 / R2: not merged" | tee -a $O/stages.txt; }; true ;;
proute|panchor)  # rm_build pool / anchor on the shard's rows with its own scene loader (EV_CFG); skipped for a GATE_FAIL shard
  N=$2; [[ -e $O/stage/$N/GATE_FAIL ]] && { echo "$N GATE_FAIL: skipped"; exit 0; }
  [[ $1 == proute && -f $CR/on1_$N/route.npz ]] && { echo "$N route done"; exit 0; }
  [[ $1 == panchor && -f $DATA_DIR/runs/route_mem/feat/on1_$N.a.npz ]] && { echo "$N anchor done"; exit 0; }
  c=$(python3 -c "import json; print(json.load(open('$O/stage/$N/manifest.json')).popitem()[1][0])")
  if [[ $1 == proute ]]; then EV_CFG=$c/generated-user-config-0.yaml $AP $RB pool --name $N --workers 8 || die "rm_build pool rc $?"
  else EV_CFG=$c/generated-user-config-0.yaml $AP $RB anchor --pool $N --workers 8 || die "rm_build anchor rc $?"; fi ;;
pstrip)  # drop the slot frames (`cur`) of the shard's dumps after the tokens (= pn.sh strip)
  N=$2; [[ -f $CR/on1_$N@warp/front.npy ]] || die "no tokens for $N"
  for d in $(python3 -c "import json; print(' '.join(v[0] for v in json.load(open('$O/stage/$N/manifest.json')).values()))"); do
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
merge1)  # on1_JTPR = on1_JT's parts (same order: L18NR1v2 stays the prefix raster) + the passing CBR shards (pn.sh merge); .a features; CB2S / CB2S0
  P=(); A=(); for k in $(shards); do [[ -e $O/stage/CBR$k/GATE_FAIL ]] && continue; P+=(on1_CBR$k:pr$k:self); A+=(on1_CBR$k:pr$k); done
  (( ${#P[@]} >= 4 )) || die "only ${#P[@]} practice shards pass R1 / R2 (< 4: registered stop)"
  echo "$(date '+%F %T') merge1: ${P[*]}" | tee -a $O/stages.txt
  $TR $RF/rf_pait.py merge --pools $JTPARTS ${P[@]} --out JTPR --nan-keys fut_reg || die "merge rc $?"
  $AP $RB anchor --merge on1_JTPR=$JTA,$(csv ${A[@]}) || die "anchor merge rc $?"
  $V $PY wfile || die wfile ;;
ident1)  # G1: P1L's argv on on1_JTPR / CB2S0 (STAND / FREE off) to step 125, every scalar <= 100 vs the stored PCL1L-F-s0 (pool_held / pool_fit aside)
  tag=CB2ID1-F-s0; mapfile -t A < <($V $PY argv P1LZ $tag); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') ident1 (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $V $SNU ident --tag $tag --ref PCL1L-F-s0 --upto 100 --stop 125 --out $O/results/g1_ident.json -- $TR "${A[@]}"; r=$?; rm -rf $CK/$tag; exit $r ;;
smoke)   # smoke 1 (P1N on CB2S, k 32) | 2 (D1N on CB2D, k 40): 60 steps, T = 30
  if [[ $2 == 1 ]]; then tag=CB2SM1$X-F-s0 arm=P1N wf=CB2S$X k=32; else tag=CB2SM2$X-F-s0 arm=D1N wf=CB2D$X k=40; fi
  mapfile -t A < <($V $PY argv $arm $tag --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') smoke $2 (code $(git rev-parse --short HEAD)): ${A[*]}" | tee -a $O/stages.txt
  $TR "${A[@]}" > $O/smoke$2.log 2>&1 || die "smoke $2 trainer (log $O/smoke$2.log)"
  $V $PY smoke --tag $tag --wfile runs/round9/cb2/w/$wf.npz --k $k --out $O/results/smoke$2$X.json; r1=$?
  $V $PL lrcheck --tag $tag --on-start 30 --out $O/results/smoke$2${X}_lr.json; r2=$?
  rm -rf $CK/$tag; exit $(( r1 || r2 )) ;;
train)   # train <arm> <tag> [seed]
  mapfile -t A < <($V $PY argv $2 $3 ${4:+--seed $4}); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') train $3 (code $(git rev-parse --short HEAD), eager ${JEV_RM_EAGER:-0}): ${A[*]}" | tee -a $O/stages.txt
  exec $TR "${A[@]}" ;;
post)    # post <tag>: DONE, final dev ADE <= 0.566, the whole run's lr vs the formula
  $V $PC post --tag $2 --out $O/results/post_$2.json || die "post $2"
  $V $PL lrcheck --tag $2 --on-start $T_ON --full --out $O/results/lr_full_$2.json || die "lr_full $2" ;;
aread)   # aread <tag> <chunk>: jt.sh aread's body (arena held split, sh30hold with JEV_HOLD unset = native) into jt/arena/<tag>@cb2; cb2v2 prune
  T=$2; c=$3; D=$J/arena/$T@cb2/c$c/run
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  k=$(python3 -c "import json; print(next(x['shard'] for x in json.load(open('$AH/meta.json'))['chunks'] if x['chunk'] == $c))") || die "no chunk $c"
  source <(python3 -c "$BASE_SRC"); rm -rf $D; mkdir -p $D; rc=0
  env -u JEV_HOLD SH30_TAG=$T bash experiments/alpasim/scripts/run.sh $D sh30hold --scene-list $AH/lists/c$c.txt "${BASE[@]}" "$KEY.cache_location=$AC/s$k" || rc=$?
  if [[ $rc -ne 0 ]]; then
    grep -q "exit code -9" "$D/native_summary.json" 2>/dev/null && { echo "runtime worker killed (-9): exit 137 for the pool's kill retry" >&2; exit 137; }
    die "run.sh rc $rc ($D)"; fi
  $AP $V2 prune --run $D || die "prune"
  $AP -m jevdrive.facts chunk --bench arena --run $D || true ;;
hread)   # hread <tag> <i>: rm_chain.sh read's body (held 460 @18: sh30, after160 cache) into route_mem/read/<tag>-cb2; cb2v2 prune
  T=$2; i=$3; D=$HR/$T-cb2/c$i/run; L=$W/lists/held_n6/c$i.txt
  [[ -f $D/aggregate/results-summary.json ]] && { echo "$D already read"; exit 0; }
  source <(python3 -c "$BASE_SRC"); rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=$T SH30_DUMP=0 bash experiments/alpasim/scripts/run.sh $D sh30 --scene-list $L "${BASE[@]}" "$KEY.cache_location=$LC" \
    "$KEY.dataset.extra_params.num_timesteps_after=160" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  $AP $V2 prune --run $D || die "prune"
  $AP -m jevdrive.facts chunk --bench held460 --run $D || true ;;
v2)      # v2 <arena|held460> <key>: rescore the read; old variant must reproduce every stored score, then the new (1b024f0) variant
  b=$2; k=$3; [[ $b == arena ]] && R=$J/arena/$k || R=$HR/$k
  $AP $V2 score --variant old --runs $R/c*/run --out $O/v2/$b/$k/old --workers 32 || die "v2 old $k (reproduction)"
  PYTHONPATH=$NEW $AP $V2 score --variant new --runs $R/c*/run --out $O/v2/$b/$k/new --workers 32 || die "v2 new $k" ;;
sum)     # sum <1|2|3>: tables (old scorer = facts cache, new = cb2v2 table), extracts, the stage's registered line
  $V $PY sum --stage $2 || die "sum $2"
  if [[ $2 == 1 ]]; then
    if [[ -f $O/results/E1_PASS$X ]]; then bash $ME s2 || die "s2 submit"; else echo "$(date '+%F %T') E1 fails ($P1N): stop" | tee -a $O/stages.txt; date > $O/STOP$X; fi
  elif [[ $2 == 2 ]]; then
    if [[ -f $O/results/ADOPT_PASS$X ]]; then bash $ME s3 || die "s3 submit"; else date > $O/DONE$X; fi
  else date > $O/DONE$X; fi ;;
# ------------------------------------------------------------------------------------------------------------------------------ stage 2
s2)
  [[ ! -e $O/S2$X ]] || die "stage 2 already submitted"; [[ -s $CK/$P1N/ckpt-final.pt ]] || die "no $P1N"
  for p in $W/roll/$P1N $CR/on1_L18CBP$X $DATA_DIR/runs/onpol_nuplan/rows/L18CBP$X $W/rows/L18CBP$X; do [[ ! -e $p ]] || die "$p exists"; done
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 200 )) || die "free $free GB < 200 GB"
  h=$(git rev-parse --short HEAD); echo "$h s2" >> $O/code.txt; date > $O/S2$X
  o=(); for i in 0 1 2 3 4 5; do o+=($(sub cb2-oroll$X-c$i --vram 32 --cpu 8 --ram 60 --timeout-h 3 -- bash $OL roll1 $P1N $i)) || exit 1; done
  k=$(sub cb2-c0chk2$X --after ${o[0]} --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY c0chk --new $W/roll/$P1N/c0/run --ref $W/roll/$RB0/c0/run \
      --list $W/lists/c0.txt --out $O/results/c0chk2$X.json) || exit 1
  g=$(sub cb2-ogeo$X --after $(csv ${o[@]}),$k --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $OL geo $P1N L18CBP$X) || exit 1
  t=$(sub cb2-otok$X --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash -c "bash $OL tok L18CBP$X smoke10eY; echo tok-and-gate rc \$?; \
      test -f $CR/on1_L18CBP$X@warp/front.npy -a -f $W/rows/L18CBP$X/gates.json") || exit 1
  st=$(sub cb2-ostrip$X --after $t --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $OL strip $P1N L18CBP$X) || exit 1
  po=$(sub cb2-opool$X --after $t --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- $AP $RB pool --name L18CBP$X --workers 32) || exit 1
  an=$(sub cb2-oanchor$X --after $po --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- \
       env EV_CFG=$W/roll/$P1N/c0/run/generated-user-config-0.yaml $AP $RB anchor --pool L18CBP$X --workers 8) || exit 1
  pg=$(sub cb2-opgate$X --after $po --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- $V $PY pgate --name L18CBP$X --out $O/results/pgate_L18CBP$X.json) || exit 1
  lo=$(sub cb2-olab-L18CBP$X --after $an,$pg --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME olab L18CBP$X) || exit 1
  for j in $(seq 0 45); do sub cb2-proll-$P1N-c$j "${RO[@]}" -- bash $ME proll $P1N $j > /dev/null || exit 1; done
  ch=($(pchains $P1N CBP$X cp "")) || exit 1
  L=(); for k_ in $(shards); do L+=($(sub cb2-olab-CBP$X$k_ --after $(csv ${ch[@]}) --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME olab CBP$X$k_)) || exit 1; done
  bash $ME s2tail $lo,$(csv ${L[@]}) || exit 1
  echo "$(date '+%F %T') s2 (code $h, free $free GB): onpol rolls ${o[*]} c0chk $k geo $g tok $t strip $st pool $po anchor $an pgate $pg olab $lo;" \
       "practice chains ${ch[*]} olab ${L[*]}; dgate $dg append $ap dweights $dw pair $pa $pb $pc smoke $sm train $tr ratio $ra post $pt v2 ${v[*]}" \
       "pai $ids tax $tx v2 $pv sum2 $su" | tee -a $O/stages.txt ;;
retok)   # plan amendment 4: retok <rows> <practice|onpol>: R1 re-run (tokens / teacher from the cache) with the CB2P1N-F-s0 control -> cb2.py rgate
  N=$2; k=$3; [[ $k == practice ]] && sp=alpasim_nuplan/practice-starts || sp=alpasim_nuplan/onpol-train
  [[ -f $CR/on1_$N@warp/front.npy ]] || die "no tokens for $N"
  ON_SPLIT=$sp ON_R1_CONTROL=CB2P1N-F-s0 ON_R1_ULP=${CB2_ULP:-} $TR $ON tok --name $N --control $([[ $k == practice ]] && echo smoke10e || echo smoke10eY) || die "on_rows tok rc $?"
  [[ $k == onpol ]] && { $TR experiments/onpol_long/scripts/ol_labels.py gate --name $N --out $W/rows/$N; echo "ol_labels gate rc $? (informational: cb2.py rgate is the line)"; }
  $V $PY rgate --name $N --kind $k --out $O/results/rgate_$N${CB2_ULP:+_a6}.json; rc=$?
  if [[ $k == practice ]]; then
    if (( rc == 0 )); then rm -f $O/stage/$N/GATE_FAIL; else touch $O/stage/$N/GATE_FAIL; echo "$(date '+%F %T') $N fails R1 / R2 (amendment 4): not merged" | tee -a $O/stages.txt; fi
    exit 0; fi
  exit $rc ;;
s2r)     # plan amendment 4: the stage-2 tail again from the R1 gates (cancels this variant's never-run tail first; rolls / rows / tokens reused)
  [[ $X == B && -e $O/S2$X && ! -e $O/S2R$X ]] || die "s2r needs CB2V=B after s2, once"
  h=$(git rev-parse --short HEAD); echo "$h s2r" >> $O/code.txt; date > $O/S2R$X
  $V - $O/jobs.txt <<'PY' > $O/s2r_cancel.txt || die "cancel list"
import sys
from jevdrive.cl import pool as P
jobs = P.snapshot()[0]["jobs"]
keep = ("cb2-otokB", "cb2-ostripB", "cb2-opoolB", "cb2-oanchorB")
for l in open(sys.argv[1]):
    s = l.split()
    if len(s) > 1 and jobs.get(s[0], {}).get("state") == "queued" and s[0] > "1025-0459" and s[1] not in keep:
        print(s[0], s[1])
PY
  while read -r i n; do $V -m jevdrive.cl cancel $i > /dev/null || die "cancel $i"; done < $O/s2r_cancel.txt
  echo "$(date '+%F %T') s2r: cancelled $(wc -l < $O/s2r_cancel.txt) never-run tail jobs ($O/s2r_cancel.txt)" | tee -a $O/stages.txt
  wc -l < $O/jobs.txt > $O/relat$X.start
  rm -f $O/STOP$X; rm -rf $O/pai281/$D1N-ag $O/pai281/$D1N-ag.submit.txt
  ot=$(awk '$2 == "cb2-otokB" {i = $1} END {print i}' $O/jobs.txt); oa=$(awk '$2 == "cb2-oanchorB" {i = $1} END {print i}' $O/jobs.txt)
  rt=$(sub cb2-retok-L18CBP$X --after $ot,$oa --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME retok L18CBP$X onpol) || exit 1
  lo=$(sub cb2-olab-L18CBP$X --after $rt --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME olab L18CBP$X) || exit 1
  L=($lo); for k_ in $(shards); do N=CBP$X$k_
    r=$(sub cb2-retok-$N --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME retok $N practice) || exit 1
    a=$(sub cb2-proute-$N --after $r --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME proute $N) || exit 1
    b=$(sub cb2-panchor-$N --after $a --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME panchor $N) || exit 1
    L+=($(sub cb2-olab-$N --after $b --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME olab $N)) || exit 1; done
  bash $ME s2tail $(csv ${L[@]}) || exit 1 ;;
s2r6)    # plan amendment 6: R1 re-run with the per-point fp16-step check for all 8 pools -> labels where missing -> s2tail -> s2pai
  [[ $X == B && -e $O/S2R$X && ! -e $O/S2R6$X ]] || die "s2r6 needs CB2V=B after s2r, once"
  h=$(git rev-parse --short HEAD); echo "$h s2r6" >> $O/code.txt; date > $O/S2R6$X
  wc -l < $O/jobs.txt > $O/relat$X.start; rm -f $O/STOP$X; rm -rf $O/pai281/$D1N-ag $O/pai281/$D1N-ag.submit.txt
  export CB2_ULP=1
  rt=$(sub cb2-retok6-L18CBP$X --vram 16 --cpu 16 --ram 64 --timeout-h 2 --env CB2_ULP=1 -- bash $ME retok L18CBP$X onpol) || exit 1
  lo=$(sub cb2-olab-L18CBP$X --after $rt --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME olab L18CBP$X) || exit 1
  L=($lo); for k_ in $(shards); do N=CBP$X$k_
    r=$(sub cb2-retok6-$N --vram 16 --cpu 16 --ram 64 --timeout-h 2 --env CB2_ULP=1 -- bash $ME retok $N practice) || exit 1
    a=$(sub cb2-proute-$N --after $r --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME proute $N) || exit 1
    b=$(sub cb2-panchor-$N --after $a --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME panchor $N) || exit 1
    L+=($(sub cb2-olab-$N --after $b --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME olab $N)) || exit 1; done
  bash $ME s2tail $(csv ${L[@]}) || exit 1 ;;
s2tail)  # s2tail <after ids>: dgate -> append -> dweights -> pair / smoke -> D1N -> reads, PAI 281, sum2 (part of s2; s2r reuses it)
  dg=$(sub cb2-dgate$X --after $1 --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME dgate) || exit 1
  ap=$(sub cb2-append$X --after $dg --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME append) || exit 1
  dw=$(sub cb2-dweights$X --after $ap --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME dweights) || exit 1
  pa=$(sub cb2-pair-a$X --after $dw "${FE[@]}" --timeout-h 1 -- bash $ME pair a) || exit 1
  pb=$(sub cb2-pair-b$X --after $dw "${FE[@]}" --timeout-h 1 -- bash $ME pair b) || exit 1
  pc=$(sub cb2-paircmp$X --after $pa,$pb --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME paircmp) || exit 1
  sm=$(sub cb2-smoke2$X --after $dw "${FE[@]}" --timeout-h 1 -- bash $ME smoke 2) || exit 1
  tr=$(sub cb2-train-$D1N --after $pc,$sm "${FE[@]}" --timeout-h 5 -- bash $ME train D1N $D1N) || exit 1
  ra=$(sub cb2-ratio-$D1N --after $pc,$sm --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $D1N --on-start $T_ON --train-id $tr \
       --out $O/results/ratio_$D1N.json) || exit 1
  pt=$(sub cb2-post-$D1N --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $D1N) || exit 1
  v=($(reads $D1N $pt)) || exit 1
  bash $ME s2pai $pt $(csv ${v[@]}) || exit 1
  echo "$(date '+%F %T') s2tail: dgate $dg append $ap dweights $dw pair $pa $pb $pc smoke $sm train $tr ratio $ra post $pt v2 ${v[*]} pai $ids tax $tx v2 $pv sum2 $su" | tee -a $O/stages.txt ;;
s2pai)   # s2pai <post id> <v2 ids>: D1N's PAI 281 side read + tax + scorer_v2, then sum2 (part of s2tail; amendment 4: re-run after the cancelled
         # tail's log dirs were cleared)
  R=$O/pai281/$D1N-ag; mkdir -p $O/pai281; [[ ! -e $R ]] || die "$R exists"
  AFTER=$2 CHUNKS_FROM=$CH SERVE="$SERVE_PAI" WIZ=$WIZ_PAI CONC=4 ALPASIM_FAST=1 PRIO=$PRIO bash experiments/alpasim/scripts/pai_box.sh submit $R $D1N \
    > $R.submit.txt || die "pai submit"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "not 8 PAI chunks"; awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  tx=$(PRIO=$PRIO AFTER=$(paste -sd, <<< "$ids") bash experiments/pai_fix/scripts/pai_fix.sh tax $D1N-281ag $R | tail -1) && [[ -n $tx ]] || die "tax"
  echo "$tx pai_fix-tax-$D1N-281ag" >> $O/jobs.txt
  pv=$(sub cb2-pai-v2$X --after $tx --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 -- bash -c "rm -rf $O/sv2$X; PYTHONPATH=$NEW $AP experiments/alpasim/scripts/scorer_v2.py \
       score --variant new --root $O/sv2$X --reads $D1N-ag --workers 16") || exit 1
  su=$(sub cb2-sum2$X --after $3,$pv --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME sum 2) || exit 1
  echo "$(date '+%F %T') s2pai: pai $ids tax $tx v2 $pv sum2 $su" | tee -a $O/stages.txt ;;
olab)    # olab <rows>: b3 teacher -> nd_teach label (T2 / T4, dagger.npz) -> cb2.py dlabel (R3, tab.fut, G0 on these rows); GATE_FAIL shards skipped
  N=$2; [[ -e $O/stage/$N/GATE_FAIL ]] && { echo "$N GATE_FAIL: skipped"; exit 0; }
  [[ -f $CR/on1_$N/LABELLED && -f $O/res/dlabel_$N.json ]] && { echo "$N already labelled (amendment 4 run): kept"; exit 0; }
  "${EA[@]}" $AP $NT teach --name $N --out $O/teach/$N.npz --workers 16 || die "teach $N"
  $TR $NT label --name $N --teach $O/teach/$N.npz --res $O/res || die "label $N"
  $V $PY dlabel --name $N --teach $O/teach/$N.npz --res $O/res || die "dlabel $N (G0)" ;;
dgate)   # T2 / T4 pooled over the DAgger pools (nd_teach gates); every pool's G0 passed in dlabel
  P=(on1_L18CBP$X); for k in $(shards); do [[ -e $O/stage/CBP$X$k/GATE_FAIL ]] || P+=(on1_CBP$X$k); done; echo "${P[*]}" > $O/dpools$X.txt
  $TR $NT gates --res $O/res --pools ${P[@]} || die "T2 / T4 pooled" ;;
append)  # on1_JTPRD = on1_JTPR + the DAgger pools (mode both, prefix cd)
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 120 )) || die "free $free GB < 120 GB"
  $TR $NT append --base on1_JTPR --add "$(cat $O/dpools$X.txt)" --out on1_JTPRD$X --prefix cd --res $O/res || die append ;;
dweights) $V $PY dweights --pools $(sed 's/on1_//g' $O/dpools$X.txt) || die dweights ;;
pair)    # weight-0 pair: P1N's argv, 60 steps, T = 30; a on on1_JTPR / CB2S, b on on1_JTPRD / CB2D0
  [[ $2 == a ]] && { Y=CB2PA$X-F-s0; arm=P1N; } || { Y=CB2PB$X-F-s0; arm=P1NZ; }
  mapfile -t A < <($V $PY argv $arm $Y --steps 60 --on-start 30 --save-at none); [[ ${#A[@]} -gt 5 ]] || die "argv"
  echo "$(date '+%F %T') pair $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/pair_$2$X.log 2>&1 || die "pair $2 (log $O/pair_$2$X.log)" ;;
paircmp) $V $SNU pair --a CB2PA$X-F-s0 --b CB2PB$X-F-s0 --upto 60 --out $O/results/pair_w0$X.json; r=$?; rm -rf $CK/CB2PA$X-F-s0 $CK/CB2PB$X-F-s0; exit $r ;;
# ------------------------------------------------------------------------------------------------------------------------------ stage 3
s3)
  [[ ! -e $O/S3$X ]] || die "stage 3 already submitted"; date > $O/S3$X; h=$(git rev-parse --short HEAD); echo "$h s3" >> $O/code.txt
  tr=$(sub cb2-train-$D1N1 "${FE[@]}" --timeout-h 5 -- bash $ME train D1N $D1N1 1) || exit 1
  ra=$(sub cb2-ratio-$D1N1 --vram 0.1 --cpu 1 --ram 4 --timeout-h 8 -- $V $PC ratio --tag $D1N1 --on-start $T_ON --train-id $tr --out $O/results/ratio_$D1N1.json) || exit 1
  pt=$(sub cb2-post-$D1N1 --after $tr,$ra --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME post $D1N1) || exit 1
  v=($(reads $D1N1 $pt)) || exit 1
  su=$(sub cb2-sum3$X --after $(csv ${v[@]}) --vram 0.1 --cpu 16 --ram 64 --timeout-h 3 -- bash $ME sum 3) || exit 1
  echo "$(date '+%F %T') s3 (code $h): train $tr ratio $ra post $pt v2 ${v[*]} sum3 $su" | tee -a $O/stages.txt ;;
wait)
  until [[ -f $O/DONE$X || -f $O/STOP$X ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
import os
s0 = sys.argv[1].replace("jobs.txt", f"relat{os.environ.get('CB2V', '')}.start")
k0 = int(open(s0).read()) if os.path.exists(s0) else 0                # amendment 4: the cancelled tail before s2r is history
for l in open(sys.argv[1]).readlines()[k0:]:
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  for m in DONE$X STOP$X; do [[ -f $O/$m ]] && echo "$m $(cat $O/$m)"; done; tail -4 $O/stages.txt; true ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
