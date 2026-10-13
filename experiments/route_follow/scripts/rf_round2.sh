#!/usr/bin/env bash
# ROUND 2 of the on-policy rows (plans/2026-10-13-round2-prereg.md; fix 2 of research/route-follow/side-bias.md): the pool follows the policy.
#   1. RSGbothSB-L18-s1 rolls the onpol-train split at 18 decisions exactly as RB0-F-s0 did for on1_L18RB0 (ol_chain.sh roll1, unedited:
#      run.sh sh30, c0b_chain.OVERRIDES, after160 cache, the default nuPlan serving), 6 chunks on cards 1-5;
#   2. rows L18SB1 (ol_chain.sh geo / tok / strip = on_rows.py geo + tok with gate R1 and the interpreted control smoke10eY, ol_labels.py gate
#      R1 / R2), route labels + memory features (rm_build.py pool), pool gate (route ok, route0_share_005, cov20: runs/route_follow/poolgate.py),
#      side tags on1_L18SB1s (rf_sidetag.py pool); union on1_L18U2s = on1_L18RB0s (r1) + on1_L18SB1s (r2) (runs/route_follow/r2merge.py, the
#      merge of round-2 pools on1_L18R2 / on1_L18R2s);
#   3. R2SB-L18-s{0,1} (pool on1_L18SB1s) and R2SBU-L18-s{0,1} (pool on1_L18U2s): RSGboth's gentle recipe from RB0-F-s0, --on-mode both, draw
#      weights by the registered rule (`train` below), 60-step preflight smoke; each followed by the held 460 / 18 read (rm_chain.sh
#      submit-read) and the public 1491 / 10 guard (10 ol_chain readlist chunks); one report job.
# Box, repo root.   rf_round2.sh submit | train <pool> <tag> <seed> [rm_train args] | report
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; O=$RF/round2; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
CR=$DATA_DIR/runs/op_parity/cache; P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
T0=RSGbothSB-L18-s1; N=L18SB1
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both"
NEW="R2SB-L18-s0 R2SB-L18-s1 R2SBU-L18-s0 R2SBU-L18-s1"
REF="RSGbothSB-L18-s0 RSGbothSB-L18-s1 RSGboth-L18-s0 RSGboth-L18-s1 RSG-L18-s0 RSG-L18-s1"
CL() { .venv/bin/python -m jevdrive.cl "$@"; }
case $1 in
submit)
  mkdir -p $O/pool; [[ -f $DATA_DIR/runs/op_parity/runs/$T0/ckpt-final.pt ]] || { echo "no $T0"; exit 1; }
  sub() { local name=$1; shift; local id; id=$(CL submit --owner route-follow --name "$name" --tries 1 --priority 14 --no-check \
       --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  mkdir -p $W/roll/$T0
  rt=$(CL fanout --arms 0,1,2,3,4,5 --name r2-roll-$T0 --owner route-follow --vram 32 --cpu 8 --ram 60 --tries 1 --priority 14 --timeout-h 3 --no-check \
       --gpus 1,2,3,4,5 --pin-strict --log-dir "$W/roll/$T0/c{arm}/pool" -- bash $E/scripts/ol_chain.sh roll1 $T0 "{arm}" | awk '{print $1}' | grep -E '^[0-9]{4}-[0-9]{6}-' | paste -sd,)
  [[ $(echo $rt | tr , '\n' | wc -l) == 6 ]] || { echo "roll fanout: $rt"; exit 1; }; echo "$rt r2-roll" >> $O/jobs.txt
  TOK='test -f $DATA_DIR/runs/op_parity/cache/on1_$0@warp/front.npy -a -f $DATA_DIR/runs/onpol_long/rows/$0/gates.json'
  g=$(sub r2-geo --after $rt --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $E/scripts/ol_chain.sh geo $T0 $N) || exit 1
  k=$(sub r2-tok --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash -c "bash $E/scripts/ol_chain.sh tok $N smoke10eY; echo tok-and-gate rc \$?; $TOK" $N) || exit 1
  sub r2-strip --after $k --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip $T0 $N > /dev/null || exit 1
  p=$(sub r2-pool --after $k --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- $P experiments/route_mem/scripts/rm_build.py pool --name $N --workers 32) || exit 1
  q=$(sub r2-gate --after $p --vram 0.5 --cpu 1 --ram 16 --timeout-h 1 -- bash $RF/r2gate.sh $N L18RB0) || exit 1
  t=$(sub r2-sidetag --after $q --vram 0.5 --cpu 2 --ram 32 --timeout-h 1 -- .venv/bin/python experiments/route_follow/scripts/rf_sidetag.py pool \
      --src on1_$N --out on1_${N}s --dry-out $O/sidetag_r1.json) || exit 1
  m=$(sub r2-merge --after $t --vram 0.5 --cpu 4 --ram 80 --timeout-h 1 -- bash -c "$TR $RF/r2merge.py L18U2s L18RB0s ${N}s && $P $RF/poolgate.py L18U2s") || exit 1
  dep=()
  for T in $NEW; do s=${T##*-s}; case $T in R2SBU*) pl=on1_L18U2s; a=$m ;; *) pl=on1_${N}s; a=$t ;; esac
    x=$(sub r2-train-$T --after $a --train --gpus 1,2,3,4,5 --pin-strict --vram 28 --cpu 6 --ram 70 --timeout-h 3 \
        --preflight "bash experiments/route_follow/scripts/rf_round2.sh train $pl ${T%-L18-s*}-smoke-s$s $s --steps 60 --eval-every 60" \
        -- bash experiments/route_follow/scripts/rf_round2.sh train $pl $T $s) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $x || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub r2-g10-$T-$i --after $x --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub r2-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash experiments/route_follow/scripts/rf_round2.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
train)  # the registered weight rule on the pool's own counts: right-turn out : in > 1.5 -> R_in x 2, R_out x 2 n_in / n_out (1 : 1), else uniform
  pl=$2; T=$3; s=$4; shift 4; mkdir -p $O
  WT=$(python3 - "$CR/$pl/tab.npz" "$O/weights_${T}.json" <<'PY'
import json, sys
import numpy as np
r = np.load(sys.argv[1])["rside"].astype(str); o, i = int((r == "R_out").sum()), int((r == "R_in").sum())
w = f"rside=R_in:2,rside=R_out:{2 * i / o:.6f}" if o > 1.5 * i else ""
json.dump(dict(pool=sys.argv[1], R_out=o, R_in=i, ratio=o / max(i, 1), rule="out:in > 1.5 -> 1:1", weight=w), open(sys.argv[2], "w"), indent=1)
print(w)
PY
) || exit 1
  echo "pool $pl weights: ${WT:-uniform}"
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC --on $pl ${WT:+--on-weight $WT} --seed $s "$@" ;;
report)
  ALL="$NEW $REF"
  R=""; for t in $ALL; do R="$R $t=$(ls -d $M/$t/c*/run | paste -sd,)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); F=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  PH=(); PF=(); for s in 0 1; do for a in R2SB R2SBU; do for r in RSG-L18 RSGboth-L18 RSGbothSB-L18; do PH+=("$a-L18-s$s:$r-s$s"); PF+=("$a-L18-s$s@18:$r-s$s@18"); done; done
    PH+=("R2SBU-L18-s$s:R2SB-L18-s$s"); PF+=("R2SBU-L18-s$s@18:R2SB-L18-s$s@18"); done
  MN="R2SB-L18=R2SB-L18-s0,R2SB-L18-s1 R2SBU-L18=R2SBU-L18-s0,R2SBU-L18-s1 RSGbothSB-L18=RSGbothSB-L18-s0,RSGbothSB-L18-s1 RSGboth-L18=RSGboth-L18-s0,RSGboth-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1"
  MP="R2SB-L18:RSG-L18 R2SB-L18:RSGboth-L18 R2SB-L18:RSGbothSB-L18 R2SBU-L18:RSG-L18 R2SBU-L18:RSGbothSB-L18 R2SBU-L18:R2SB-L18"
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --mean $MN --pairs $MP "${PH[@]}" --out $O/held18; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}"; rc2=$?
  $P experiments/route_follow/scripts/rf_inside.py --follow $O/follow --pairs "${PF[@]}" --out $O/inside.json > /dev/null; rc3=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in $ALL; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean $MN --pairs $MP "${PH[@]}" --out $O/guard; rc4=$?
  rc5=0; for T in $NEW; do
    $P experiments/route_follow/scripts/rf_tax.py extract --runs $(ls -d $M/$T/c*/run) --out $O/held_$T.pkl --workers 8 > $O/extract_$T.log 2>&1 || rc5=1; done
  S=(); for T in $NEW; do S+=("$T=$O/held_$T.pkl"); done
  .venv/bin/python experiments/route_follow/scripts/rf_sidebias.py held --set "${S[@]}" --out $O/side --workers 8; rc6=$?
  echo "held $rc0 held18 $rc1 follow $rc2 inside $rc3 guard $rc4 extract $rc5 side $rc6"; exit $((rc0 + rc1 + rc2 + rc3 + rc4 + rc5 + rc6)) ;;
*) echo "rf_round2.sh submit | train <pool> <tag> <seed> [args] | report" >&2; exit 2 ;;
esac
