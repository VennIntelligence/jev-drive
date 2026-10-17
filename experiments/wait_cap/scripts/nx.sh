#!/usr/bin/env bash
# Decision 335's follow-up pilots (prereg draft plans/2026-10-17-retain-anticopy-draft.md). Box, repo root; every job through the GPU pool,
# --after chained; ids in $X/jobs.txt. Nothing here launches a pilot unless called with rta-pilot / rtb-pilot / ac-pilot.
#   nx.sh codecheck <rev> [P]  60-step identity: RB0-F recipe and RSG-L18 recipe, code of <rev> vs this code with every new flag off -> nx.py bitcmp
#                           (results/next/rt_identity.json, ac_identity.json; the RSG one also = RSG-L18-s0's on_hash_60); P: tag prefix (NXGG)
#   nx.sh check             the read code on existing checkpoints: controls' dev plans, history-scale read of WTCP / WTJ2P, both gates --dry
#   nx.sh approach          the navtrain approach rows (CPU) -> runs/wait_cap/next/approach_nav.npz (needed by every (iii) training)
#   nx.sh check-rt          (after approach) arrival replay of P0 / WTCP / RB0-F-s0 / RSG-L18-s0 / WTJ2P, hist RB0-F-s0 / RSG-L18-s0, dc plans of WTCP,
#                           rtgate --variant a --dry with RSG-L18-s0 as a stand-in arm
#   nx.sh smoke             40-step smoke runs RTSMOKE-F-s0 (--kd-lam 3 --d2-lam 3) and ACSMOKE-L18-s0 (--ego-scale-speed 0.5 --ego-scale-max 0.4)
#   nx.sh smoke-rt          40-step smoke runs of both (iii) variants (RTSMOKEA-L18-s0, RTSMOKEB-F-s0; lambda 3, approach rows on)
#   nx.sh rta-pilot         (iii-a) RTA03P-L18-s0 / RTA1P-L18-s0: RSG-L18 recipe 1 000 steps + --kd-lam L --kd-nav .. --kd-nav-k 8 --d2-lam 1, L 0.3 / 1
#                           (after the ac identity: same recipe) -> waitcap / ledger F / desire / dev / hist / arrival reads -> rtgate --variant a (C WTCP-L18-s0)
#   nx.sh rtb-pilot         (iii-b) RTB1-F-s0 / RTB3-F-s0: RB0-F recipe 10 000 steps + the same terms, L 1 / 3 (after the rt identity) -> the same reads
#                           -> rtgate --variant b (C RB0-F-s0)
#   nx.sh ac-pilot          (iv) ACS2P-L18-s0 / ACS4P-L18-s0: RSG-L18 recipe 1 000 steps + --ego-scale-speed 0.5 --ego-scale-max 0.2 | 0.4 (after ac identity)
#                           -> dev plans, history-scale read, waitcap plans -> nx.py acgate --submit (control WTCP-L18-s0; PASS -> ac-held)
#   nx.sh ac-held T,T       held 460 @18 (6 chunks per tag) -> rf_follow (tags + WTCP-L18-s0) -> nx.py acheld
#   nx.sh rtb-cont          prereg 补记 1 (the user's exploratory continuation of RTB1-F-s0 after its pilot STOP), one chain, two variants:
#                           (A) RTB1O-L18-s0 = the RSG-L18 recipe (WTC-L18-s0's command, 2 500 steps) with --init RTB1-F-s0 on the existing pool on1_L18RB0;
#                           (B) RTB1-F-s0 rolls onpol-train at 18 decisions (ol_chain roll1 x 6) -> rows L18RTB1 (ol_chain geo / tok, control smoke10eY /
#                           strip) -> rm_build pool -> poolgate -> RTB1N-L18-s0 = the same recipe on on1_L18RTB1; after each training `cont-reads`
#   nx.sh rtb-decomp        补记 1: wt_devdecomp of RTB1-F-s0 / RTB3-F-s0 vs RB0-F-s0 on the gate-4 dev rows -> results/next/rtb_devdecomp.{json,md}
#   nx.sh cont-reads <a|b> <tag>   job body (after the training): held 460 @18 (6 chunks), static public scene @10 (SH30_DUMP), PAI stop17 (PAI
#                           config), arrival replay -> rf_follow (tag + RSG-L18-s0 + WTC-L18-s0) -> nx.py cont -> results/next/rtb_cont_<a|b>, cont/<a|b>/FINISHED
#   nx.sh train <arm> <seed> <steps> <tag>   job body; arm: rb0 | rsg (the recipes, flags off) | rta0.3 | rta1 | rtb1 | rtb3 | rtsmoke | rtsmokea |
#                           rtsmokeb | acs2 | acs4 | acsmoke | rtbo | rtbn
# Env: PRIO (pool priority, default 9)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
X=$DATA_DIR/runs/wait_cap/next; CK=$DATA_DIR/runs/op_parity/runs; ME=experiments/wait_cap/scripts/nx.sh; PY=experiments/wait_cap/scripts/nx.py
WT=experiments/wait_cap/scripts/wt.py; WC=experiments/wait_cap/scripts/wc.py; DC=experiments/desire_cap/scripts/dc.py
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
RT=experiments/route_mem/scripts/rm_train.py
RB0="--yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"            # RB0-F-s0's command (route-night.md $REC)
RSG="--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 \
--on-split alpasim_nuplan/onpol-train --lr-scale 0.1 --on on1_L18RB0 --on-k 13 --on-mode self"                # wt.sh `train off` = RSG-L18 / WTCP
TEA=$DATA_DIR/runs/desire_cap/train/teacher; RSG_HASH60=ea8cacc3e451c577
RTA=(RTA03P-L18-s0 RTA1P-L18-s0); RTB=(RTB1-F-s0 RTB3-F-s0); ACJ=(ACS2P-L18-s0 ACS4P-L18-s0); RTC=RB0-F-s0; ACC=WTCP-L18-s0
NAV=runs/wait_cap/next/approach_nav.npz; KD="--kd-nav $NAV --kd-nav-k 8 --d2-teacher $TEA"
OLW=$DATA_DIR/runs/onpol_long; OL=experiments/onpol_long/scripts/ol_chain.sh; B0=RTB1-F-s0; NP=L18RTB1                   # 补记 1
STATIC=2021.09.16.15.12.03_veh-42_01037_01434-b5b8d20688385790; PAICH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks      # = wt.sh
STOP17="054b5901 095cf563 127c4f85 2d10a08d 35e082a4 3fe2bac8 44df49fe 4cb7f7ac 4ecc114d 559d4ad6 68c3be44 7d994ed3 9f8acd00 ba40be62 bc309fd6 cbd87234 fc9fde87"
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"; PAIWIZ=controller.gains.rel_front_steering_angle_weight=10
mkdir -p $X/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner wait-next --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $X/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $X/jobs.txt; echo "$id"; }
FW=(--train --vram 24 --cpu 6 --ram 70); FK=(--train --vram 28 --cpu 6 --ram 70)          # FK: the second forwards of --kd-lam / --d2-lam
csv() { local IFS=,; echo "$*"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $X/jobs.txt; }
fresh() { for T in "$@"; do [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }; done; }
case ${1:-} in
codecheck)
  rev=${2:?rev}; P=${3:-NXGG}; W=$DATA_DIR/tmp/nx_old_$rev; [[ -d $W ]] || git worktree add --detach $W $rev > /dev/null || exit 1
  fresh $P-RB0-old $P-RB0-new $P-RSG-old $P-RSG-new
  a=$(sub nx-gg-rb0-old "${FW[@]}" --timeout-h 1 -- $TR $W/$RT train --tag $P-RB0-old $RB0 --steps 60 --seed 0) || exit 1
  b=$(sub nx-gg-rb0-new "${FW[@]}" --timeout-h 1 -- bash $ME train rb0 0 60 $P-RB0-new) || exit 1
  c=$(sub nx-gg-rsg-old "${FW[@]}" --timeout-h 1 -- $TR $W/$RT train --tag $P-RSG-old $RSG --steps 60 --seed 0) || exit 1
  d=$(sub nx-gg-rsg-new "${FW[@]}" --timeout-h 1 -- bash $ME train rsg 0 60 $P-RSG-new) || exit 1
  e=$(sub nx-bitcmp-rt --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a $P-RB0-old --b $P-RB0-new --out rt_identity \
      --note "RB0-F recipe 60 steps from the shipped weights: code $rev vs this code ($(git rev-parse --short HEAD)) with --kd-lam / --d2-lam / --ego-scale-speed off") || exit 1
  f=$(sub nx-bitcmp-ac --after $c,$d --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a $P-RSG-old --b $P-RSG-new --out ac_identity \
      --hash $RSG_HASH60 --note "RSG-L18 recipe 60 steps (WTCP's command): code $rev vs this code ($(git rev-parse --short HEAD)) with every new flag off; on_hash_60 = RSG-L18-s0's") || exit 1
  echo "codecheck $rev ($P): rb0 $a $b bitcmp $e; rsg $c $d bitcmp $f" | tee -a $X/stages.txt ;;
check)     # the read code on existing checkpoints (no training): dev plans of the controls, the history-scale read of WTCP and WTJ2P (whose
           # 4 s shortfall is known, decision 335), then both gates --dry (print only) with stand-in arms
  d=$(sub nx-dev-check --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $WT devplans --tags $RTC RSG-L18-s0) || exit 1
  h=$(sub nx-hist-check --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR $PY hist --tags $ACC WTJ2P-L18-s0) || exit 1
  g=$(sub nx-gate-check --after $d,$h --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash -c "$TR $PY rtgate --dry --j RSG-L18-s0 --c $RTC --lam-of 1=RSG-L18-s0 && \
      $TR $PY acgate --dry --j WTJ2P-L18-s0 --c $ACC --dose-of 0.2=WTJ2P-L18-s0") || exit 1
  echo "check: devplans $d hist $h gates $g" | tee -a $X/stages.txt ;;
approach)
  a=$(sub nx-approach --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR $PY approach) || exit 1
  echo "approach: $a" | tee -a $X/stages.txt ;;
check-rt)
  ap=$(jid nx-approach) || { echo "no approach job" >&2; exit 1; }
  r=$(sub nx-arr-check --vram 12 --cpu 4 --ram 40 --timeout-h 1 -- $TR $PY arrival --tags P0 $ACC $RTC RSG-L18-s0 WTJ2P-L18-s0) || exit 1
  h=$(sub nx-hist-rtcheck --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR $PY hist --tags $RTC RSG-L18-s0) || exit 1
  d=$(sub nx-dc-wtcp --vram 16 --cpu 4 --ram 40 --timeout-h 1 -- $TR $DC plans --tags $ACC --srcs nav pai) || exit 1
  g=$(sub nx-rtgate-check --after $r,$h,$d,$ap --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY rtgate --variant a --dry --j RSG-L18-s0 --c $ACC \
      --lam-of 1=RSG-L18-s0) || exit 1
  echo "check-rt: arrival $r hist $h dc $d gate $g" | tee -a $X/stages.txt ;;
smoke-rt)
  ap=$(jid nx-approach) || { echo "no approach job" >&2; exit 1; }; fresh RTSMOKEA-L18-s0 RTSMOKEB-F-s0
  a=$(sub nx-smoke-rta "${FK[@]}" --after $ap --timeout-h 1 -- bash $ME train rtsmokea 0 40 RTSMOKEA-L18-s0) || exit 1
  b=$(sub nx-smoke-rtb "${FK[@]}" --after $ap --timeout-h 1 -- bash $ME train rtsmokeb 0 40 RTSMOKEB-F-s0) || exit 1
  echo "smoke-rt: a $a b $b" | tee -a $X/stages.txt ;;
rta-pilot|rtb-pilot)
  ap=$(jid nx-approach) || { echo "no approach job" >&2; exit 1; }
  if [[ $1 == rta-pilot ]]; then V=a; JJ=(${RTA[@]}); C=$ACC; ARM=rta; N=1000; b=$(jid nx-bitcmp-ac); TH=1; LL=(0.3 1)   # smoke: lambda 3 moved dev ADE +0.17 in 40 steps
  else V=b; JJ=(${RTB[@]}); C=$RTC; ARM=rtb; N=10000; b=$(jid nx-bitcmp-rt); TH=2; LL=(1 3); fi
  [[ -n $b ]] || { echo "no identity job (nx.sh codecheck)" >&2; exit 1; }; fresh ${JJ[@]}
  t1=$(sub nx-train-${JJ[0]} "${FK[@]}" --after $b,$ap --timeout-h $TH -- bash $ME train ${ARM}${LL[0]} 0 $N ${JJ[0]}) || exit 1
  t3=$(sub nx-train-${JJ[1]} "${FK[@]}" --after $b,$ap --timeout-h $TH -- bash $ME train ${ARM}${LL[1]} 0 $N ${JJ[1]}) || exit 1
  T=$t1,$t3
  r1=$(sub nx-wc-rt$V --after $T --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags ${JJ[@]}) || exit 1
  r2=$(sub nx-led-rt$V --after $T --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR experiments/cap_ledger/scripts/led.py plans --tags ${JJ[@]} --conds F) || exit 1
  r3=$(sub nx-dc-rt$V --after $T --vram 16 --cpu 4 --ram 40 --timeout-h 1 -- $TR $DC plans --tags $C ${JJ[@]} --srcs nav pai) || exit 1
  r4=$(sub nx-dev-rt$V --after $T --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $WT devplans --tags $C ${JJ[@]}) || exit 1
  r5=$(sub nx-hist-rt$V --after $T --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR $PY hist --tags $C ${JJ[@]}) || exit 1
  r6=$(sub nx-arr-rt$V --after $T --vram 12 --cpu 4 --ram 40 --timeout-h 1 -- $TR $PY arrival --tags P0 $C ${JJ[@]}) || exit 1
  g=$(sub nx-rtgate-$V --after $r1,$r2,$r3,$r4,$r5,$r6 --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY rtgate --variant $V --j ${JJ[@]} --c $C \
      --lam-of ${LL[0]}=${JJ[0]},${LL[1]}=${JJ[1]}) || exit 1
  echo "$1: trainings $t1 $t3 reads $r1 $r2 $r3 $r4 $r5 $r6 gate $g" | tee -a $X/stages.txt ;;
ac-pilot)
  b=$(jid nx-bitcmp-ac) || { echo "no ac identity job (nx.sh codecheck)" >&2; exit 1; }; fresh ${ACJ[@]}
  t2=$(sub nx-train-${ACJ[0]} "${FW[@]}" --after $b --timeout-h 1 -- bash $ME train acs2 0 1000 ${ACJ[0]}) || exit 1
  t4=$(sub nx-train-${ACJ[1]} "${FW[@]}" --after $b --timeout-h 1 -- bash $ME train acs4 0 1000 ${ACJ[1]}) || exit 1
  r1=$(sub nx-dev-ac --after $t2,$t4 --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $WT devplans --tags $ACC ${ACJ[@]}) || exit 1
  r2=$(sub nx-hist-ac --after $t2,$t4 --vram 24 --cpu 6 --ram 80 --timeout-h 1 -- $TR $PY hist --tags $ACC ${ACJ[@]}) || exit 1
  r3=$(sub nx-wc-ac --after $t2,$t4 --vram 10 --cpu 4 --ram 40 --timeout-h 1 -- $TR $WC plans --tags ${ACJ[@]}) || exit 1
  g=$(sub nx-acgate --after $r1,$r2,$r3 --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 --env PRIO=${PRIO:-9} -- $TR $PY acgate --j ${ACJ[@]} --c $ACC \
      --dose-of 0.2=${ACJ[0]},0.4=${ACJ[1]} --submit) || exit 1
  echo "ac-pilot: trainings $t2 $t4 reads $r1 $r2 $r3 gate $g" | tee -a $X/stages.txt ;;
ac-held)
  JJ=${2:?T,T}; hh=()
  for T in ${JJ//,/ }; do for i in 0 1 2 3 4 5; do hh+=($(sub nx-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done; done
  R=""; for T in ${JJ//,/ } $ACC; do R="$R $T@18=\$(ls -d $DATA_DIR/runs/route_mem/read/$T/c*/run | paste -sd,)"; done
  f=$(sub nx-acfollow --after "$(csv ${hh[@]})" --vram 0.1 --cpu 16 --ram 100 --timeout-h 2 -- bash -c "mkdir -p $X/acfollow && $AP experiments/route_follow/scripts/rf_follow.py --runs $R --out $X/acfollow/follow --workers 16") || exit 1
  g=$(sub nx-acheld --after $f --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY acheld --j $JJ --c $ACC) || exit 1
  echo "ac-held $JJ: held $(csv ${hh[@]}) follow $f summary $g" | tee -a $X/stages.txt ;;
rtb-cont)
  Y=$X/cont; [[ ! -e $Y ]] || { echo "$Y exists" >&2; exit 1; }; fresh RTB1O-L18-s0 RTB1N-L18-s0
  [[ -f $CK/$B0/ckpt-final.pt && ! -e $OLW/roll/$B0 && ! -e $OLW/rows/$NP && ! -e $DATA_DIR/runs/op_parity/cache/on1_$NP ]] || { echo "no $B0 or $NP exists" >&2; exit 1; }
  mkdir -p $Y/a $Y/b $OLW/roll/$B0; echo $STATIC > $Y/static.txt
  cat $PAICH/*.txt | grep -E "^clipgt-($(echo $STOP17 | tr ' ' '|'))" | sort -u > $Y/stop17.txt
  (( $(wc -l < $Y/stop17.txt) == 17 )) || { echo "stop17 list: $(wc -l < $Y/stop17.txt) scenes" >&2; exit 1; }
  ta=$(sub nx-train-RTB1O-L18-s0 "${FW[@]}" --timeout-h 2 -- bash $ME train rtbo 0 2500 RTB1O-L18-s0) || exit 1
  pa=$(sub nx-cont-a --after $ta --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont-reads a RTB1O-L18-s0) || exit 1
  rr=$(.venv/bin/python -m jevdrive.cl fanout --arms 0,1,2,3,4,5 --name nx-roll-$B0 --owner wait-next --vram 32 --cpu 8 --ram 60 --tries 1 --priority ${PRIO:-9} \
       --timeout-h 3 --no-check --log-dir "$OLW/roll/$B0/c{arm}/pool" -- bash $OL roll1 $B0 "{arm}" < /dev/null | awk '{print $1}' | grep -E '^[0-9]{4}-[0-9]{6}-' | paste -sd,)
  [[ $(echo $rr | tr , '\n' | wc -l) == 6 ]] || { echo "roll fanout: $rr" >&2; exit 1; }; echo "$rr nx-roll-$B0" >> $X/jobs.txt
  g=$(sub nx-geo-$NP --after $rr --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $OL geo $B0 $NP) || exit 1
  k=$(sub nx-tok-$NP --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $OL tok $NP smoke10eY) || exit 1
  s=$(sub nx-strip-$NP --after $k --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $OL strip $B0 $NP) || exit 1
  p=$(sub nx-pool-$NP --after $k --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- $AP experiments/route_mem/scripts/rm_build.py pool --name $NP --workers 32) || exit 1
  q=$(sub nx-poolgate-$NP --after $p --vram 0.5 --cpu 1 --ram 16 --timeout-h 1 -- $AP $DATA_DIR/runs/route_follow/poolgate.py $NP) || exit 1
  tb=$(sub nx-train-RTB1N-L18-s0 "${FW[@]}" --after $q --timeout-h 2 -- bash $ME train rtbn 0 2500 RTB1N-L18-s0) || exit 1
  pb=$(sub nx-cont-b --after $tb --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont-reads b RTB1N-L18-s0) || exit 1
  echo "rtb-cont: A train $ta reads-submitter $pa; B roll $rr geo $g tok $k strip $s pool $p poolgate $q train $tb reads-submitter $pb" | tee -a $X/stages.txt ;;
rtb-cont-a)   # 补记 1 (A) again after its first training stopped on rm_train's init guard (--allow-init added; the recipe is unchanged)
  fresh RTB1O-L18-s0
  ta=$(sub nx-train-RTB1O-L18-s0 "${FW[@]}" --timeout-h 2 -- bash $ME train rtbo 0 2500 RTB1O-L18-s0) || exit 1
  pa=$(sub nx-cont-a --after $ta --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont-reads a RTB1O-L18-s0) || exit 1
  echo "rtb-cont-a: train $ta reads-submitter $pa" | tee -a $X/stages.txt ;;
rtb-seeds)   # 补记 1 addition: (1) RTB1O-L18-s1 (A's command, continuation seed 1, from RTB1-F-s0); (2) RTB1-F-s1 (the base retain recipe,
             # lambda 1, seed 1) -> RTB1OB1-L18-s1 (A's command from it, seed 1); each -> cont-reads (c, d)
  fresh RTB1O-L18-s1 RTB1-F-s1 RTB1OB1-L18-s1
  t1=$(sub nx-train-RTB1O-L18-s1 "${FW[@]}" --timeout-h 2 -- bash $ME train rtbo 1 2500 RTB1O-L18-s1) || exit 1
  p1=$(sub nx-cont-c --after $t1 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont-reads c RTB1O-L18-s1) || exit 1
  b1=$(sub nx-train-RTB1-F-s1 "${FK[@]}" --timeout-h 2 -- bash $ME train rtb1 1 10000 RTB1-F-s1) || exit 1
  t2=$(sub nx-train-RTB1OB1-L18-s1 "${FW[@]}" --after $b1 --timeout-h 2 -- bash $ME train rtbo1 1 2500 RTB1OB1-L18-s1) || exit 1
  p2=$(sub nx-cont-d --after $t2 --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 --env PRIO=${PRIO:-9} -- bash $ME cont-reads d RTB1OB1-L18-s1) || exit 1
  echo "rtb-seeds: (1) train $t1 reads-submitter $p1; (2) base $b1 train $t2 reads-submitter $p2" | tee -a $X/stages.txt ;;
rtb-decomp)   # 补记 1: the dev-ADE decomposition of RTB1 / RTB3 against RB0-F-s0 (CPU, stored plans)
  d=$(sub nx-rtb-decomp --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR $PY devdecomp --j RTB1-F-s0 RTB3-F-s0 --c $RTC --out rtb_devdecomp) || exit 1
  echo "rtb-decomp: $d" | tee -a $X/stages.txt ;;
cont-reads)   # cont-reads <a|b> <tag> (job body, after the training): the reads, rf_follow, the summary
  V=$2; T=$3; Y=$X/cont; [[ -f $CK/$T/ckpt-final.pt ]] || { echo "no $T" >&2; exit 1; }; hh=(); mkdir -p $Y/$V    # a rerun reuses submitted reads
  for i in 0 1 2 3 4 5; do hh+=($(jid nx-held-$T-$i || sub nx-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  st=$(jid nx-static-$T || sub nx-static-$T --vram 20 --cpu 6 --ram 30 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_DUMP=1 \
      -- bash $OL readlist $T $Y/static.txt 10 $Y/$V/static/$T/run) || exit 1
  CHUNK=17 SERVE="$PAISERVE" WIZ=$PAIWIZ PRIO=${PRIO:-9} bash experiments/alpasim/scripts/pai_box.sh submit $Y/$V/paisub/$T $T $Y/stop17.txt > $Y/$V/paisub.submit.txt || exit 1
  pp=$(awk '{print $1}' $Y/$V/paisub.submit.txt | grep -E '^[0-9]{4}-' | paste -sd,); [[ -n $pp ]] || { echo "no PAI stop17 job" >&2; exit 1; }
  ar=$(jid nx-arr-$T || sub nx-arr-$T --vram 12 --cpu 4 --ram 40 --timeout-h 1 -- $TR $PY arrival --tags $T) || exit 1
  f=$(sub nx-follow-$T --after "$(csv ${hh[@]})" --vram 0.1 --cpu 16 --ram 100 --timeout-h 2 -- bash experiments/wait_cap/scripts/wt.sh follow $T RSG-L18-s0,WTC-L18-s0,RSG-L18-s1,WTC-L18-s1 $Y/$V) || exit 1
  c=$(sub nx-cont-sum-$V --after "$f,$st,$pp,$ar" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY cont --j $T --dir $Y/$V --out rtb_cont_$V) || exit 1
  echo "cont-reads $V $T: held $(csv ${hh[@]}) static $st stop17 $pp arrival $ar follow $f summary $c" | tee -a $X/stages.txt ;;
train)   # train <arm> <seed> <steps> <tag>
  case $2 in
    rb0) F="$RB0" ;;
    rsg) F="$RSG" ;;
    rta0.3|rta1) L=${2#rta}; F="$RSG --kd-lam $L --d2-lam 1 $KD" ;;     # D2 fixed at 1 (Amendment C 4: the preferred D2 weight)
    rtb1|rtb3) L=${2#rtb}; F="$RB0 --kd-lam $L --d2-lam $L $KD" ;;
    rtsmokea) F="$RSG --kd-lam 3 --d2-lam 3 $KD" ;;
    rtsmokeb) F="$RB0 --kd-lam 3 --d2-lam 3 $KD" ;;
    rtsmoke) F="$RB0 --kd-lam 3 --d2-lam 3 --kd-mode full --kd-groups W:1,G:1,SY:1 --d2-teacher $TEA" ;;   # the first (full-plan) design's smoke
    acs2) F="$RSG --ego-scale-speed 0.5 --ego-scale-max 0.2" ;;
    acs4|acsmoke) F="$RSG --ego-scale-speed 0.5 --ego-scale-max 0.4" ;;
    rtbo) F="${RSG/--init RB0-F-s0/--init $B0 --allow-init}" ;;                                       # 补记 1 (A): the RSG-L18 recipe from RTB1, old pool
    rtbo1) F="${RSG/--init RB0-F-s0/--init RTB1-F-s1 --allow-init}" ;;                  # 补记 1 addition (2): base seed 1
    rtbn) F="${RSG/--init RB0-F-s0/--init $B0 --allow-init}"; F="${F/--on on1_L18RB0/--on on1_$NP}" ;;   # 补记 1 (B): RTB1's own L18 pool
    *) echo "train: arm $2?" >&2; exit 2 ;;
  esac
  exec $TR $RT train --tag $5 $F --steps $4 --seed $3 ;;
*) sed -n 2,32p $0; exit 2 ;;
esac
