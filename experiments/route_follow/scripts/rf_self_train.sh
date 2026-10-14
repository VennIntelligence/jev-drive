#!/usr/bin/env bash
# Turning school phase 2, training arm RSGSELF-L18-s{0,1} = RSG-L18's recipe (init RB0-F-s0, gentle schedule, RSG-L18's own on-policy pool
# on1_L18RB0 as `self` rows) plus the self-branch demonstration rows (rf_selfpool.py, pool on1_SELF) as `both` rows in --on-mode mixed, SELF rows
# 30 % of the pool draws, per-scene draw cap inside the SELF rows (rm_train --on-cap-scene). Control: RSG-L18-s{0,1} (stored reads).
# Reads as rf_tschool.sh: held 460 @18 (rm_chain submit-read) and the public 1491 @10 guard (10 ol_chain readlist chunks); report = the ladder
# table (held / public mean and zeros, paired CIs vs RSG-L18) + the anti-lateral-ratchet guard (rf_selfread.py lat). Box, repo root, DATA_DIR set.
#   rf_self_train.sh merge           on1_L18RB0:r:self + $SELF:s:both -> on1_$OUT; W for SHARE of the draws; draw shares before / after the cap
#   rf_self_train.sh submit          mixcheck -> both seeds' training -> reads -> report (job ids in $O/jobs.txt)
#   rf_self_train.sh report          (job body) ladder table -> $O/summary.md
#   rf_self_train.sh dry             the same on the dev pool on1_SELFdev: merge, mixcheck, RSGSELF-DRY-s0 for 60 steps, held 5 scenes, the report
#                                    path on those 5 scenes, then the checkpoint and the merged dev pool are deleted (dry-post). DRY ONLY: the dev
#                                    pool holds hard_exam scenes; nothing from it is ever read on the PAI exam.
#   rf_self_train.sh clean-dry       delete the dry checkpoint and the merged dev pool by hand (if the chain stopped before dry-post)
# Env: SELF (pool, default on1_SELF), OUT (merged pool suffix, default RSGSELF), CAP (max scene share inside the SELF rows, default 0.10),
#      SHARE (SELF share of the draws, default 0.3), O (out dir, default $DATA_DIR/runs/route_follow/selftrain).
set -uo pipefail; cd "$(dirname "$0")/../../.."
RF=$DATA_DIR/runs/route_follow; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
CR=$DATA_DIR/runs/op_parity/cache; CK=$DATA_DIR/runs/op_parity/runs; FE=$DATA_DIR/runs/route_mem/feat
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; S=experiments/route_follow/scripts
SELF=${SELF:-on1_SELF}; OUT=${OUT:-RSGSELF}; CAP=${CAP:-0.10}; SHARE=${SHARE:-0.3}; O=${O:-$RF/selftrain}
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4"
NEW="RSGSELF-L18-s0 RSGSELF-L18-s1"; DRY=RSGSELF-DRY-s0
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
arm() { local w; w=$(cat $O/W_$OUT.txt 2>/dev/null) || die "no $O/W_$OUT.txt: run merge first"
  echo "--on on1_$OUT --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode mixed --on-weight pool=s:$w --on-cap-scene $CAP --on-cap-in pool=s"; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner selftrain --name "$name" --tries 1 --priority 14 --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
held_dirs() { ls -d $M/$1/c*/run | paste -sd,; }
mkdir -p $O/pool
case ${1:-} in
merge)
  for p in on1_L18RB0 $SELF; do [[ -f $CR/$p/tab.npz && -f $FE/$p.npz && -d $CR/$p@warp ]] || die "no pool $p"; done
  [[ $SELF == *dev || ! -e $CR/$SELF/EXAM_SCENES_INSIDE ]] || die "$SELF carries EXAM_SCENES_INSIDE: never trained outside the dry run"
  $TR $S/rf_pait.py merge --out $OUT --pools on1_L18RB0:r:self $SELF:s:both > $O/merge_$OUT.log 2>&1 || { tail $O/merge_$OUT.log; die "merge"; }
  # pool=s weight so the SELF rows are SHARE of the drawn pool rows (no other --on-weight term touches them; the cap keeps their total mass)
  $TR -c "
import numpy as np; t = np.load('$CR/on1_$OUT/tab.npz'); s = t['pool'] == 's'; m = t['mode'][s]
assert (t['mode'][~s] == 'self').all() and (m == 'both').all(), 'modes'
print(round($SHARE / (1 - $SHARE) * (~s).sum() / s.sum(), 4))" > $O/W_$OUT.txt || die "W"
  echo "W = $(cat $O/W_$OUT.txt) (SELF share $SHARE of the pool draws, cap $CAP per scene inside the SELF rows)"
  $TR experiments/route_mem/scripts/rm_train.py draws --on on1_$OUT --on-weight pool=s:$(cat $O/W_$OUT.txt) --on-cap-scene $CAP --on-cap-in pool=s \
      | tee $O/draws_$OUT.json || die "draws (cap infeasible? scenes x CAP must be >= 1)" ;;
submit)
  [[ -f $CR/on1_$OUT/tab.npz && -f $FE/on1_$OUT.npz ]] || die "no on1_$OUT: run merge first"
  [[ -f $CK/RSG-L18-s0/ckpt-final.pt ]] && ls -d $G/read10/RSG-L18-s1/b9 > /dev/null && ls -d $M/RSG-L18-s1/c5/run > /dev/null || die "stored RSG-L18 reads missing"
  ARM=$(arm) || exit 1; echo "ARM: $ARM" >> $O/jobs.txt
  mc=$(sub sf-mixcheck --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- $TR experiments/route_mem/scripts/rm_train.py mixcheck --tag mixcheck-$OUT $REC $ARM --seed 0) || exit 1
  dep=()
  for T in $NEW; do s=${T##*-s}
    t=$(sub sf-tr-$T --after $mc --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $T $REC $ARM --seed $s) || exit 1
    PRIO=14 bash experiments/route_mem/scripts/rm_chain.sh submit-read $T $t || exit 1
    dep+=($(grep " rm-cover-$T$" $DATA_DIR/runs/route_mem/jobs.txt | tail -1 | cut -d' ' -f1))
    for i in $(seq 0 9); do
      dep+=($(sub sf-g10-$T-$i --after $t --vram 27 --cpu 8 --ram 30 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=ol-fast-read -- bash $E/scripts/ol_chain.sh readlist $T \
            $W/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1
    done
  done
  sub sf-report --after "$(IFS=,; echo "${dep[*]}")" --vram 0.5 --cpu 48 --ram 64 --timeout-h 2 -- bash $S/rf_self_train.sh report > /dev/null || exit 1
  cat $O/jobs.txt ;;
report)
  HELD="$NEW RSG-L18-s0 RSG-L18-s1 RB0-F-s0"
  R=""; for t in $HELD; do R="$R $t=$(held_dirs $t)"; done
  python3 $RF/held.py $R > $O/held_raw.txt; rc0=$?
  H=(); L=(); for t in $HELD; do H+=("$t@held18=$(held_dirs $t)"); L+=("$t=$(held_dirs $t)"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RB0-F-s0 --mean RSGSELF-L18=RSGSELF-L18-s0,RSGSELF-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 \
     --pairs RSGSELF-L18:RSG-L18 RSGSELF-L18-s0:RSG-L18-s0 RSGSELF-L18-s1:RSG-L18-s1 --out $O/held18; rc1=$?
  A=$DATA_DIR/runs/alpasim; Q=()
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 $NEW; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P $E/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 RSGSELF-L18=RSGSELF-L18-s0,RSGSELF-L18-s1 \
     --pairs RSGSELF-L18:RSG-L18 RSGSELF-L18-s0:RSG-L18-s0 RSGSELF-L18-s1:RSG-L18-s1 --out $O/guard; rc2=$?
  $P $S/rf_selfread.py lat --runs "${L[@]:0:4}" --out $O/lat --mean RSGSELF-L18=RSGSELF-L18-s0,RSGSELF-L18-s1 RSG-L18=RSG-L18-s0,RSG-L18-s1 \
     --pairs RSGSELF-L18:RSG-L18 RSGSELF-L18-s0:RSG-L18-s0 RSGSELF-L18-s1:RSG-L18-s1 --workers 48; rc3=$?
  $P $S/rf_selfread.py table --held $O/held18 --guard $O/guard --lat $O/lat --arm RSGSELF-L18 --ref RSG-L18 --out $O/summary.md; rc4=$?
  echo "held $rc0 held18 $rc1 guard $rc2 lat $rc3 table $rc4"; exit $((rc0 + rc1 + rc2 + rc3 + rc4)) ;;
dry)
  export SELF=on1_SELFdev OUT=RSGSELFdev CAP=${CAP_DRY:-0.25} O=$RF/selftrain/dry; mkdir -p $O/pool
  [[ ! -e $CK/$DRY ]] || die "$CK/$DRY exists (clean-dry first)"
  bash $S/rf_self_train.sh merge || exit 1
  ARM=$(arm) || exit 1; echo "ARM: $ARM" >> $O/jobs.txt
  awk 'NR % 15 == 1' $W/lists/held_n6/c0.txt | head -5 > $O/held5.txt
  mc=$(sub sfd-mixcheck --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- $TR experiments/route_mem/scripts/rm_train.py mixcheck --tag mixcheck-$OUT $REC $ARM --seed 0) || exit 1
  t=$(sub sfd-tr-$DRY --after $mc --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- $TR experiments/route_mem/scripts/rm_train.py train --tag $DRY \
      ${REC/--steps 2500/--steps 60} $ARM --seed 0) || exit 1
  r=$(sub sfd-read-$DRY --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 1 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 -- \
      bash experiments/route_mem/scripts/rm_chain.sh readlist $DRY $O/held5.txt $O/read/$DRY) || exit 1
  sub sfd-post --after $r --vram 0.5 --cpu 8 --ram 32 --timeout-h 1 -- bash $S/rf_self_train.sh dry-post > /dev/null || exit 1
  cat $O/jobs.txt ;;
dry-post)  # report path on the 5 scenes (DRY vs RSG-L18-s0's stored rollouts of the same scenes), then the deletion
  O=$RF/selftrain/dry; D=$O/read/$DRY
  python3 $RF/held.py $DRY=$D RSG-L18-s0=$(held_dirs RSG-L18-s0) > $O/held_raw.txt; rc0=$?
  $P $E/scripts/ol_confirm.py --runs "$DRY@held18=$D" "RSG-L18-s0@held18=$(held_dirs RSG-L18-s0)" "RB0-F-s0@held18=$(held_dirs RB0-F-s0)" --base RB0-F-s0 \
     --pairs $DRY:RSG-L18-s0 --out $O/held18; rc1=$?
  $P $S/rf_selfread.py lat --runs "$DRY=$D" "RSG-L18-s0=$(held_dirs RSG-L18-s0)" --scenes $O/held5.txt --out $O/lat --pairs $DRY:RSG-L18-s0 --workers 5; rc2=$?
  $P $S/rf_selfread.py table --held $O/held18 --lat $O/lat --arm RSGSELF-DRY --ref RSG-L18 --seeds 0 --out $O/summary.md; rc3=$?
  bash $S/rf_self_train.sh clean-dry; rc4=$?
  echo "held $rc0 held18 $rc1 lat $rc2 table $rc3 clean $rc4"; exit $((rc0 + rc1 + rc2 + rc3 + rc4)) ;;
clean-dry)
  rm -rf $CK/$DRY $CR/on1_RSGSELFdev $CR/on1_RSGSELFdev@warp $FE/on1_RSGSELFdev.npz $DATA_DIR/runs/route_mem/rely/$DRY.json
  for f in $CK/$DRY $CR/on1_RSGSELFdev $FE/on1_RSGSELFdev.npz; do [[ ! -e $f ]] || die "still there: $f"; done
  echo "deleted $CK/$DRY (checkpoint), on1_RSGSELFdev (merged dev pool, cache + @warp + feat)" ;;
*) echo "rf_self_train.sh merge | submit | report | dry | dry-post | clean-dry" >&2; exit 2 ;;
esac
