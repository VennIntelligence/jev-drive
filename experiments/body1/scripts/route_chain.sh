#!/usr/bin/env bash
# BODY1 Amendment 7 chain on the lane's held card (route_run.sh per step; two GPU steps at a time at most). A failed step stops the stage.
#   scripts/tmux_run.sh body1-route-<stage> bash experiments/body1/scripts/route_chain.sh <stage> [args]
#   a                identity runs (new code, switches off) + bit-identity against the old code path and pp_train.py, the two band pilots,
#                    validation-part reads, selection (route_gate.py select). No hold-log read.
#   b <pilot tag>    pilot gate reads of the selected pilot: hold logs and navtest, once (route_gate.py pilot)
#   c <band>         full runs of four seeds (two at a time), then the G3 readers (route_gate.py g3)
# State: $DATA_DIR/runs/body1/route/chain-<stage>/{STATUS, DONE | ERROR}; each step under $DATA_DIR/runs/body1/route/<step>/.
set -uo pipefail
cd "$(dirname "$0")/../../.."
stage=${1:?stage}; shift
R=$DATA_DIR/runs/body1/route; C=$R/chain-$stage; mkdir -p "$C"; rm -f "$C/DONE" "$C/ERROR"
PY=$DATA_DIR/envs/op-train/bin/python; S=experiments/body1/scripts; RUN="bash $S/route_run.sh"
CA=200-207; CB=192-199; CC=184-191; CD=176-183
st() { echo "$(date '+%F %T') $*" | tee "$C/STATUS"; }
die() { echo "$*" | tee "$C/ERROR"; exit 1; }
both() { local a=$1 b=$2; wait "$a"; local ra=$?; wait "$b"; local rb=$?; (( ra == 0 && rb == 0 )); }
HO="--ho ot1:4,yr1:4,bd4:5 --agent-lam 10 --ho-w 3 --ho-excl navsim/body1-val-logs --shape"
ALL=$(for k in $(seq 0 11); do echo -n "navtrain_full.s${k}of12 "; done)
run_dir() { ls -d "$DATA_DIR/runs/op_parity/train-$1"/* | tail -1; }

case $stage in
a)
  st "identity runs"
  $RUN ident-S-new $CA $PY $S/bd4_train.py train --seed 0 --data navtrain_full.s2of12 --steps 60 --eval-every 30 $HO --tag B43A7-IDENT-S1 & p1=$!
  sleep 20
  $RUN ident-P-new $CB $PY $S/bd4_train.py train --seed 0 --data navtrain_full.s2of12 --steps 60 --eval-every 30 --tag B43A7-IDENT-B & p2=$!
  both $p1 $p2 || die "identity runs failed"
  $PY $S/bd4_train.py ident --a "$(run_dir B43A7-IDENT-S0)" --b "$(run_dir B43A7-IDENT-S1)" --out experiments/body1/results/route/ident_a7_shape.json || die "not bit-identical to the P2H10S code path"
  $PY $S/bd4_train.py ident --a "$(run_dir B43A6-IDENT-A)" --b "$(run_dir B43A7-IDENT-B)" --out experiments/body1/results/route/ident_a7_pp.json || die "not bit-identical to pp_train.py"
  st "pilots"
  P="--seed 0 --data navtrain_full.s2of12 navtrain_full.s3of12 --steps 3000 --eval-every 1000 $HO"
  $RUN pilot-b15 $CA $PY $S/bd4_train.py train $P --route-band 1.5 --tag P2H10R-Pb15-s0 & p1=$!
  sleep 20
  $RUN pilot-b25 $CB $PY $S/bd4_train.py train $P --route-band 2.5 --tag P2H10R-Pb25-s0 & p2=$!
  both $p1 $p2 || die "a pilot failed"
  st "validation reads"
  $RUN val-b15 $CA $PY $S/bd4_g3.py --set val --shards 2 3 --ref P2H10-P-s0 --name a7_val_b15 --new P2H10R-Pb15-s0 & p1=$!
  $RUN val-b25 $CB $PY $S/bd4_g3.py --set val --shards 2 3 --ref P2H10-P-s0 --name a7_val_b25 --new P2H10R-Pb25-s0 & p2=$!
  both $p1 $p2 || die "a validation read failed"
  $RUN val-plans $CA $PY $S/prog_ol.py states --set val --shards 2 3 --new P2H10R-Pb15-s0 P2H10R-Pb25-s0 P2H10S-P-s0 --ref P2H10-P-s0 --out experiments/body1/results/route --name a7_val || die "prog_ol val failed"
  taskset -c $CC $PY $S/route_ol.py --name a7_val --set val --shards 2 3 --new P2H10R-Pb15-s0 P2H10R-Pb25-s0 P2H10S-P-s0 --ref P2H10-P-s0 > "$C/route_ol_val.log" 2>&1 || die "route_ol val failed"
  taskset -c $CC $PY $S/route_gate.py select > "$C/select.log" 2>&1 || die "select failed"
  ;;
b)
  tag=${1:?pilot tag}
  st "pilot gate reads of $tag"
  $RUN hold-R $CA $PY $S/bd4_g3.py --set hold --shards 2 3 --ref P2H10-P-s0 P2H10-F-s0 --name a7_hold_R --new "$tag" & p1=$!
  $RUN slope $CB $PY experiments/alpasim/scripts/ot3_rows.py probe --name b43a7 --tags "$tag" P2H10-P-s0 P2H10S-P-s0 --data navtrain_full.s2of12 navtrain_full.s3of12 & p2=$!
  both $p1 $p2 || die "hold read or slope probe failed"
  $RUN arc-hold $CA $PY $S/prog_ol.py states --set hold --shards 2 3 --new "$tag" --ref P2H10-P-s0 --out experiments/body1/results/route --name a7_pilot_hold & p1=$!
  $RUN arc-navtest $CB $PY $S/prog_ol.py states --set navtest --new "$tag" --ref P2H10-P-s0 --out experiments/body1/results/route --name a7_pilot_navtest & p2=$!
  both $p1 $p2 || die "prog_ol failed"
  {
    taskset -c $CC $PY $S/route_ol.py --name a7_pilot_hold --set hold --shards 2 3 --new "$tag" --ref P2H10-P-s0 &&
    taskset -c $CC $PY $S/route_ol.py --name a7_pilot_navtest --set navtest --new "$tag" --ref P2H10-P-s0 &&
    taskset -c $CC $PY $S/route_ol.py --name a6_pilot_hold --set hold --shards 2 3 --new P2H10S-P-s0 P2H10B-Pw3-s0 P2H10B-Pw3-noA-s0 P2H10B-P-Aon-s0 --ref P2H10-P-s0 &&
    taskset -c $CC $PY $S/route_ol.py --name a6_pilot_navtest --set navtest --new P2H10S-P-s0 P2H10B-Pw3-s0 P2H10B-Pw3-noA-s0 P2H10B-P-Aon-s0 --ref P2H10-P-s0 &&
    taskset -c $CC $PY $S/route_gate.py pilot --arm "$tag"
  } > "$C/gate.log" 2>&1 || die "gate readers failed"
  ;;
c)
  band=${1:?band}
  for pair in "0 1" "2 3"; do
    set -- $pair
    st "full runs seeds $1 $2"
    $RUN full-s$1 $CA $PY $S/bd4_train.py train --seed $1 --data $ALL --steps 10000 $HO --route-band "$band" --tag P2H10R-F-s$1 & p1=$!
    sleep 30
    $RUN full-s$2 $CB $PY $S/bd4_train.py train --seed $2 --data $ALL --steps 10000 $HO --route-band "$band" --tag P2H10R-F-s$2 & p2=$!
    both $p1 $p2 || die "a full run failed (seeds $1 $2)"
  done
  NEW="P2H10R-F-s0 P2H10R-F-s1 P2H10R-F-s2 P2H10R-F-s3"; REF="P2H10-F-s0 P2H10-F-s1 P2H10-F-s2 P2H10-F-s3"
  for pair in "0 1" "2 3"; do
    set -- $pair
    st "G3 (a), (b) seeds $1 $2"
    ( $RUN g3a-s$1 $CA $PY $S/bd4_g3.py --name a7_full_hold_s$1 --set hold --new P2H10R-F-s$1 --ref P2H10-F-s$1 && $RUN g3b-s$1 $CA $PY $S/bd4_g3.py --name a7_full_navtest_s$1 --set navtest --new P2H10R-F-s$1 --ref P2H10-F-s$1 ) & p1=$!
    ( $RUN g3a-s$2 $CB $PY $S/bd4_g3.py --name a7_full_hold_s$2 --set hold --new P2H10R-F-s$2 --ref P2H10-F-s$2 && $RUN g3b-s$2 $CB $PY $S/bd4_g3.py --name a7_full_navtest_s$2 --set navtest --new P2H10R-F-s$2 --ref P2H10-F-s$2 ) & p2=$!
    both $p1 $p2 || die "G3 (a) / (b) reader failed (seeds $1 $2)"
  done
  st "G3 (c), (e), corridor"
  $RUN g3c $CA $PY experiments/alpasim/scripts/ot3_rows.py probe --name b43a7full --tags $NEW $REF & p1=$!
  $RUN g3e-navtest $CB $PY $S/prog_ol.py states --set navtest --new $NEW --ref $REF --out experiments/body1/results/route --name a7_full_navtest & p2=$!
  both $p1 $p2 || die "slope probe or prog_ol navtest failed"
  $RUN g3e-hold $CA $PY $S/prog_ol.py states --set hold --new $NEW --ref $REF --out experiments/body1/results/route --name a7_full_hold || die "prog_ol hold failed"
  {
    taskset -c $CC $PY $S/route_ol.py --name a7_full_hold --set hold --new $NEW --ref $REF &&
    taskset -c $CC $PY $S/route_ol.py --name a7_full_navtest --set navtest --new $NEW --ref $REF &&
    taskset -c $CC $PY $S/route_gate.py g3
  } > "$C/gate.log" 2>&1 || die "gate readers failed"
  ;;
*) die "unknown stage $stage" ;;
esac
date '+%F %T' > "$C/DONE"; st "stage $stage done"
