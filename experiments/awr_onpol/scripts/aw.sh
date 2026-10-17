#!/usr/bin/env bash
# AWR-ONPOL (plans/2026-10-17-awr-onpol-prereg.md): consequence-weighted cloning of on-policy plans on on1_RSGACC. Box, repo root; every job
# through the pool; ids in $X/jobs.txt, stage lines in $X/stages.txt.
#   aw.sh score                 the outcome pack (aw_score.py check + build, H 3 s) -> $X/pack/on1_RSGACC.npz, results/score.md
#   aw.sh codecheck <rev> [P]   60-step identity: the RSG-L18 recipe, code of <rev> vs this code with every --awr flag off -> aw.py bitcmp
#                               (results/identity.json; on_hash_60 = RSG-L18-s0's ea8cacc3e451c577); P: tag prefix (AWGG)
#   aw.sh smoke                 40 steps of arm F1 (after the score job) -> AWRSMOKE-L18-s0
#   aw.sh pilot                 (after the identity and the score jobs) control AWR0-L18-s0 = RSG-ACC p0k13's command, arms AWRF<K1> / AWRF<K2>
#                               (+ --awr-pack, keep K1 / K2), 1 000 steps seed 0 -> held01 + dev-rec forwards -> pref -> pilot gate; the gate job
#                               submits `held <arm>` for every passing arm and then for the control (reported only)
#   aw.sh held <tag>            job body: held 460 @18 (6 chunks) -> rf_follow (tag + RSG-L18-s0) -> held gate vs the stored RSG-L18-s0
#   aw.sh train <arm> <seed> <steps> <tag>   job body; arm: rsg (RSG-L18 recipe, flags off) | ctl (p0k13) | f<keep> (ctl + --awr-pack, filt)
# Env: PRIO (default 9), K1 / K2 (keep thresholds, default 0.1 / 0.5), AWK (rows per step, 16)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
X=$DATA_DIR/runs/awr_onpol; CK=$DATA_DIR/runs/op_parity/runs; ME=experiments/awr_onpol/scripts/aw.sh; PY=experiments/awr_onpol/scripts/aw.py
SC=experiments/awr_onpol/scripts/aw_score.py; RT=experiments/route_mem/scripts/rm_train.py; S=experiments/route_follow/scripts
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RF=$DATA_DIR/runs/route_follow
REC="--init RB0-F-s0 --warmup 1 --lr-scale 0.1 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --eval-every 500"
RSG="$REC --on on1_L18RB0 --on-k 13"                                              # RSG-L18-s0's command (nx.sh $RSG)
PACK=runs/awr_onpol/pack/on1_RSGACC.npz; RSG_HASH60=ea8cacc3e451c577; C0=AWR0-L18-s0
K1=${K1:-0.1}; K2=${K2:-0.5}; AWK=${AWK:-16}; tagk() { echo "AWRF$(echo $1 | tr -d .)-L18-s0"; }
mkdir -p $X/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner awr-onpol --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $X/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $X/jobs.txt; echo "$id"; }
FW=(--train --vram 28 --cpu 6 --ram 120); FR=(--train --vram 24 --cpu 6 --ram 70)          # FR: the RSG recipe on on1_L18RB0 (identity)
csv() { local IFS=,; echo "$*"; }
jid() { awk -v n="$1" '$2 == n {i = $1} END {if (i) print i; else exit 1}' $X/jobs.txt 2>/dev/null; }
fresh() { for T in "$@"; do [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }; done; }
case ${1:-} in
score)
  s=$(sub aw-score --vram 0.1 --cpu 32 --ram 32 --timeout-h 2 -- bash -c "$AP $SC check && $AP $SC build --h 3 --workers 32") || exit 1
  echo "score: $s" | tee -a $X/stages.txt ;;
codecheck)
  rev=${2:?rev}; P=${3:-AWGG}; W=$DATA_DIR/tmp/aw_old_$rev; [[ -d $W ]] || git worktree add --detach $W $rev > /dev/null || exit 1
  fresh $P-RSG-old $P-RSG-new
  c=$(sub aw-gg-rsg-old "${FR[@]}" --timeout-h 1 -- $TR $W/$RT train --tag $P-RSG-old $RSG --steps 60 --seed 0) || exit 1
  d=$(sub aw-gg-rsg-new "${FR[@]}" --timeout-h 1 -- bash $ME train rsg 0 60 $P-RSG-new) || exit 1
  f=$(sub aw-bitcmp --after $c,$d --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY bitcmp --a $P-RSG-old --b $P-RSG-new --out identity --hash $RSG_HASH60 \
      --note "RSG-L18 recipe 60 steps: code $rev vs this code ($(git rev-parse --short HEAD)) with every --awr flag off; on_hash_60 = RSG-L18-s0's") || exit 1
  echo "codecheck $rev ($P): $c $d bitcmp $f" | tee -a $X/stages.txt ;;
smoke)
  s=$(jid aw-score) || { echo "no score job (aw.sh score)" >&2; exit 1; }; fresh AWRSMOKE-L18-s0
  t=$(sub aw-smoke "${FW[@]}" --after $s --timeout-h 1 -- bash $ME train f$K1 0 40 AWRSMOKE-L18-s0) || exit 1
  echo "smoke: $t (after score $s)" | tee -a $X/stages.txt ;;
pilot)
  s=$(jid aw-score) && b=$(jid aw-bitcmp) || { echo "no score / identity job (aw.sh score, codecheck)" >&2; exit 1; }
  A1=$(tagk $K1); A2=$(tagk $K2); fresh $C0 $A1 $A2
  t0=$(sub aw-train-$C0 "${FW[@]}" --after $b --timeout-h 2 -- bash $ME train ctl 0 1000 $C0) || exit 1
  t1=$(sub aw-train-$A1 "${FW[@]}" --after $b,$s --timeout-h 2 -- bash $ME train f$K1 0 1000 $A1) || exit 1
  t2=$(sub aw-train-$A2 "${FW[@]}" --after $b,$s --timeout-h 2 -- bash $ME train f$K2 0 1000 $A2) || exit 1
  T=$t0,$t1,$t2; TG=$C0,$A1,$A2
  f1=$(sub aw-fwd-held01 --after $T --vram 16 --cpu 8 --ram 120 --timeout-h 2 -- $TR $S/rf_recov.py fwd --pool L18RSGallheld --rows held01 --tags $TG) || exit 1
  f2=$(sub aw-fwd-dev --after $T --vram 16 --cpu 8 --ram 160 --timeout-h 2 -- $TR $S/rf_recov.py fwd --pool RSGACC --rows dev-rec --tags $TG --out $RF/rsgacc/fwd) || exit 1
  p=$(sub aw-pref --after $f2 --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- $TR $PY pref --tags $TG) || exit 1
  g=$(sub aw-gate-pilot --after $f1,$p --vram 0.1 --cpu 8 --ram 64 --timeout-h 1 -- bash $ME gate-pilot $A1,$A2) || exit 1
  echo "pilot (K1 $K1, K2 $K2, k $AWK): trainings $T fwd $f1 $f2 pref $p gate $g" | tee -a $X/stages.txt ;;
gate-pilot)   # job body: the pilot gate; on PASS `held <arm>` for each passing arm
  $TR $PY gate --stage pilot --j $2 --c $C0; rc=$?
  (( rc == 0 )) || { echo "pilot gate: no arm passes (rc $rc), the chain stops here"; touch $X/STOP_pilot; exit 0; }
  for t in $(cat $X/pass.txt) $C0; do bash $ME held $t || exit 1; done ;;      # the control is read too (reported, not judged)
held)
  T=${2:?tag}; Y=$X/held/$T; mkdir -p $Y; hh=()
  for i in 0 1 2 3 4 5; do hh+=($(jid aw-held-$T-$i || sub aw-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
      -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
  R=""; for t in $T RSG-L18-s0; do R="$R $t@18=\$(ls -d $DATA_DIR/runs/route_mem/read/$t/c*/run | paste -sd,)"; done
  f=$(sub aw-follow-$T --after "$(csv ${hh[@]})" --vram 0.1 --cpu 16 --ram 100 --timeout-h 2 -- bash -c "$AP $S/rf_follow.py --runs $R --out $Y/follow --workers 16") || exit 1
  g=$(sub aw-gate-held-$T --after $f --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- $TR $PY gate --stage held --j $T --c RSG-L18-s0 --dir $Y) || exit 1
  echo "held $T: reads $(csv ${hh[@]}) follow $f gate $g" | tee -a $X/stages.txt ;;
train)   # train <arm> <seed> <steps> <tag>
  case $2 in
    rsg) F="$RSG" ;;
    ctl|f*) WT=$($TR $S/rf_rsgacc.py weights --cell p0k13) || exit 1; F="$REC --on on1_RSGACC --on-k 13 --on-weight $WT"
            [[ $2 == ctl ]] || F="$F --awr-pack $PACK --awr-k $AWK --awr-mode filt --awr-keep ${2#f}" ;;
    *) echo "train: arm $2?" >&2; exit 2 ;;
  esac
  exec $TR $RT train --tag $5 $F --steps $4 --seed $3 ;;
*) sed -n 2,14p $0; exit 2 ;;
esac
