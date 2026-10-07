#!/usr/bin/env bash
# op_parity mixed-domain lane (plans/2026-10-08-mixed-domain-prereg.md): "standstill / launch decided by the scene, not by a per-domain constant of
# the adapter bias", per board. Self-advancing chains in tmux jev (scripts/tmux_run.sh mixed-<stage> experiments/op_parity/scripts/mixed_domain_chain.sh
# <stage>); every GPU job goes through the pool, every navtest / navhard / HUGSIM read through jevdrive.bench. Rerunning a stage resumes it.
#   ng      P2HG = the P2H10 recipe with --stop-gate 0.5 trained in: pilot P2HG-P-s0 (H0 recipe) -> navtest (:sg) -> gate G-NG vs RH0-F-s0
#           -> P2HG-F-s0 / s1 -> navtest (:sg), HUGSIM 64 spec_plan_smooth (parity stop_gate 0.5), WOD val (gated bias), navhard (G, :sg)
#   mx      MX = one adapter on navtrain + WOD r2-train: teacher8 -> pilot MX-P-s0 / MX9-P-s0 (+ the cross reads of the single-domain pilots)
#           -> gate G-MX -> MX-F-s0 / s1 -> WOD val (main + zero / biasmean / biasresid), navtest (+ :dn), HUGSIM 64, navhard (G)
#   report  bias decomposition + all tables -> experiments/op_parity/results/mixed_domain/ (after ng and mx; arms that did not run are left out)
# State: $DATA_DIR/runs/op_parity/mixed/chain-<stage>/{STATUS, DONE, ERROR, GATE_STOP, log.txt, jobs.txt}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?stage: ng | mx | report}
O=$DATA_DIR/runs/op_parity/mixed
D=$O/chain-$STAGE; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR" "$D/GATE_STOP"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
OP=$DATA_DIR/envs/openpilot/bin/python
J=$DATA_DIR/envs/jevdrive/bin/python
VPY=$PWD/.venv/bin/python
CL="$J -m jevdrive.cl"
B=("$VPY" -m jevdrive.bench)
S=experiments/op_parity/scripts
L=$D/pool
ONNX=$DATA_DIR/runs/op_parity/hugsim/onnx
BIAS=$DATA_DIR/runs/op_parity/wod
LIST=experiments/hugsim/scripts/derot_all64.txt
HF="--hinge-lam 10 --hinge-margin 0.3"
K=12 FULL=$(for i in $(seq 0 $((K - 1))); do echo -n "navtrain_full.s${i}of$K "; done)
SG='{"parity": {"stop_gate": 0.5}}'
status() { echo "$(date '+%F %T') op_parity mixed-domain $STAGE: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '$4 == n && ($2 == "queued" || $2 == "running") {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
# WOD val through the decision 155 harness: serve <tag> [gbias] [extra pred tags + bias files ...]
serve() { local tag=$1 kind=${2:-plain}; shift; shift || true
          [[ -f $ONNX/pp-$tag.onnx ]] || $PY $S/pp_hugsim.py onnx --tag $tag --out $ONNX/pp-$tag.onnx >/dev/null || die "onnx $tag"
          [[ -f $BIAS/bias-$tag.npz ]] || { if [[ $kind == gbias ]]; then $PY $S/wod_launch.py gbias --tags $tag; else $PY $S/pp_wod.py bias --tags $tag; fi; } || die "bias $tag"
          local tags="$tag" bs="$BIAS/bias-$tag.npz"
          for v in "$@"; do tags="$tags mx-${tag}_$v"; bs="$bs $O/bias-${tag}_$v.npz"; done
          sub mxd-e-$tag $L/e-$tag --vram 8 --cpu 14 --ram 40 -- $OP scripts/wod_zeroshot_openpilot.py --set rater extra --workers 12 \
              --onnx $ONNX/pp-$tag.onnx --tag $tags --bias $bs; }

if [[ $STAGE == ng ]]; then
  G="--stop-gate 0.5"
  smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data navtrain_full.s2of12 --split navsim/op-parity-s234 --steps 3 --batch 16 --eval-every 3 $HF $G --tag smoke-ng"
  status "pilot: train P2HG-P-s0"
  sub mxd-t-P2HG-P-s0 $L/t-P2HG-P-s0 --train --vram 24 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed 0 --frames warp --host \
      --data navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12 --split navsim/op-parity-s234 --steps 3000 --batch 64 --warmup 100 --eval-every 1000 $HF $G --tag P2HG-P-s0
  waitdirs $L/t-P2HG-P-s0
  status "pilot: navtest (:sg)"
  "${B[@]}" run --model P2HG-P-s0:sg --bench navtest --wait || die "bench navtest P2HG-P-s0"
  $VPY $S/mixed_domain.py gate-ng --name pilot_ng --arm P2HG-P-s0:sg --ref RH0-F-s0; rc=$?
  if (( rc == 3 )); then status "STOP: pilot gate G-NG failed (results/mixed_domain/pilot_ng_gate.json)"; touch "$D/GATE_STOP"; date > "$D/DONE"; exit 0; fi
  (( rc == 0 )) || die "gate-ng"
  status "pilot gate G-NG passed; full P2HG x 2 seeds"
  for s in 0 1; do
    sub mxd-t-P2HG-F-s$s $L/t-P2HG-F-s$s --train --vram 40 --cpu 6 --ram 40 -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
        --data $FULL --split navsim/op-parity-full --steps 10000 --batch 128 --warmup 300 --eval-every 1000 $HF $G --tag P2HG-F-s$s
  done
  waitdirs $L/t-P2HG-F-s0 $L/t-P2HG-F-s1
  for s in 0 1; do $PY $S/pp_full_check.py train --tag P2HG-F-s$s || die "training sanity P2HG-F-s$s"; done
  status "full: navtest (:sg), HUGSIM 64 (stop_gate), WOD val (gated bias), navhard (G, :sg)"
  "${B[@]}" run --model P2HG-F-s0:sg P2HG-F-s1:sg --bench navtest || die "bench navtest"
  "${B[@]}" run --model P2HG-F-s0 P2HG-F-s1 --bench hugsim --preset spec_plan_smooth --opts "$SG" --scenarios "$LIST" || die "bench hugsim"
  for s in 0 1; do serve P2HG-F-s$s gbias; done
  "${B[@]}" run --model P2HG-F-s0@gimm:sg P2HG-F-s1@gimm:sg --bench navhard || status "navhard submit failed (optional read)"
  "${B[@]}" status --model P2HG-F-s0:sg P2HG-F-s1:sg --bench navtest --wait || die "navtest"
  waitdirs $L/e-P2HG-F-s0 $L/e-P2HG-F-s1
  "${B[@]}" status --model P2HG-F-s0 P2HG-F-s1 --bench hugsim --preset spec_plan_smooth --opts "$SG" --wait || die "hugsim"
  "${B[@]}" status --model P2HG-F-s0@gimm:sg P2HG-F-s1@gimm:sg --bench navhard --wait || status "navhard failed (optional read)"
  status "done"; date > "$D/DONE"; exit 0
fi

if [[ $STAGE == mx ]]; then
  M="--wod-mass 0.5"
  PIL="lb_navtrain lb_h1train wod_pilot"
  status "teacher8 (shipped on 8 slots of the WOD rows)"
  sub mxd-teacher8 $L/teacher8 --vram 24 --cpu 8 --ram 48 -- $PY $S/mixed_domain.py teacher8
  # cross reads of the single-domain pilots (inference only)
  "${B[@]}" run --model WP2-pilot-s0 --bench navtest || die "bench navtest WP2-pilot-s0"
  serve P2-W-s0
  waitdirs $L/teacher8
  status "pilot: train MX-P-s0 / MX9-P-s0"
  smoke="$PY $S/pp_train.py --arm P2 --frames warp --host --data $PIL --split navsim/op-parity-pilot --steps 3 --batch 16 --eval-every 3 $M --wod-slots 8 --tag smoke-mx"
  pil() { local tag=$1; shift
          sub mxd-t-$tag $L/t-$tag --train --vram 30 --cpu 6 --ram 40 --preflight "$smoke" -- $PY $S/pp_train.py --arm P2 --seed 0 --frames warp --host \
              --data $PIL --split navsim/op-parity-pilot --steps 600 --batch 128 --warmup 100 --eval-every 200 $M "$@" --tag $tag; }
  pil MX-P-s0 --wod-slots 8
  pil MX9-P-s0
  waitdirs $L/t-MX-P-s0 $L/t-MX9-P-s0
  status "pilot: WOD val and navtest"
  serve MX-P-s0; serve MX9-P-s0
  "${B[@]}" run --model MX-P-s0 MX9-P-s0 --bench navtest || die "bench navtest pilot"
  waitdirs $L/e-MX-P-s0 $L/e-MX9-P-s0 $L/e-P2-W-s0
  "${B[@]}" status --model MX-P-s0 MX9-P-s0 WP2-pilot-s0 --bench navtest --wait || die "navtest pilot"
  $VPY $S/mixed_domain.py gate --name pilot_mx9 --arm MX9-P-s0 || true
  $VPY $S/mixed_domain.py gate --name pilot_mx --arm MX-P-s0; rc=$?
  $VPY $S/mixed_domain.py report --name pilot --boards wod navtest --arms MX=MX-P-s0 MX9=MX9-P-s0 WODP=WP2-pilot-s0 NAVP=P2-W-s0 shipped \
      --refs WODP NAVP shipped MX || die "pilot report"
  if (( rc == 3 )); then status "STOP: pilot gate G-MX failed (results/mixed_domain/pilot_mx_gate.json)"; touch "$D/GATE_STOP"; date > "$D/DONE"; exit 0; fi
  (( rc == 0 )) || die "gate"
  # prereg addendum 1: the full arm keeps 8 + zero slots on WOD rows unless that costs the pilot more than 0.10 RFS on WOD val against 9 real slots
  SL=$($VPY -c "
import json; r = lambda n: json.load(open('experiments/op_parity/results/mixed_domain/%s_gate.json' % n))['RFS']['MX']
print('--wod-slots 8' if r('pilot_mx') >= r('pilot_mx9') - 0.10 else '')") || die "slot rule"
  echo "${SL:-9 real WOD slots}" > "$D/SLOTS"
  status "pilot gate G-MX passed; full MX x 2 seeds (WOD slots: ${SL:-9 real})"
  for s in 0 1; do
    sub mxd-t-MX-F-s$s $L/t-MX-F-s$s --train --vram 52 --cpu 8 --ram 64 -- $PY $S/pp_train.py --arm P2 --seed $s --frames warp --host \
        --data $FULL wod_r2 --split navsim/op-parity-full --steps 10000 --batch 256 --warmup 300 --eval-every 1000 $HF $M $SL --tag MX-F-s$s
  done
  waitdirs $L/t-MX-F-s0 $L/t-MX-F-s1
  for s in 0 1; do $PY $S/pp_full_check.py train --tag MX-F-s$s || die "training sanity MX-F-s$s"; done
  status "full: WOD val (+ bias variants), navtest (+ :dn), HUGSIM 64, navhard (G)"
  $PY $S/mixed_domain.py bias --tags MX-F-s0 MX-F-s1 --harness || die "bias variants"
  for s in 0 1; do serve MX-F-s$s plain zero biasmean biasresid; done
  "${B[@]}" run --model MX-F-s0 MX-F-s1 MX-F-s0:dn MX-F-s1:dn --bench navtest || die "bench navtest"
  "${B[@]}" run --model MX-F-s0 MX-F-s1 --bench hugsim --preset spec_plan_smooth --scenarios "$LIST" || die "bench hugsim"
  "${B[@]}" run --model MX-F-s0@gimm MX-F-s1@gimm --bench navhard || status "navhard submit failed (optional read)"
  waitdirs $L/e-MX-F-s0 $L/e-MX-F-s1
  "${B[@]}" status --model MX-F-s0 MX-F-s1 MX-F-s0:dn MX-F-s1:dn --bench navtest --wait || die "navtest"
  "${B[@]}" status --model MX-F-s0 MX-F-s1 --bench hugsim --preset spec_plan_smooth --wait || die "hugsim"
  "${B[@]}" status --model MX-F-s0@gimm MX-F-s1@gimm --bench navhard --wait || status "navhard failed (optional read)"
  status "done"; date > "$D/DONE"; exit 0
fi

if [[ $STAGE == report ]]; then
  have() { [[ -f $DATA_DIR/runs/op_parity/runs/$1/ckpt-final.pt ]]; }
  tags="P2H10-F-s0 P2H10-F-s1 WP2-full-s0 WP2-full-s1"; arms="P2H10=P2H10-F-s0+P2H10-F-s1 WP2=WP2-full-s0+WP2-full-s1"; vars=""; navx=""
  have WLG-full-s1 && [[ -d $DATA_DIR/runs/wod_zeroshot/preds/op_cinque_WLG-full-s1 || -n $(ls -d $DATA_DIR/runs/*/preds/op_cinque_WLG-full-s1 2>/dev/null) ]] && {
    tags="$tags WLG-full-s0 WLG-full-s1"; arms="$arms WLG=WLG-full-s0+WLG-full-s1"; }
  have P2HG-F-s1 && { tags="$tags P2HG-F-s0 P2HG-F-s1"; arms="P2HG=P2HG-F-s0:sg+P2HG-F-s1:sg $arms"; }
  have MX-F-s1 && { tags="$tags MX-F-s0 MX-F-s1"; arms="MX=MX-F-s0+MX-F-s1 $arms"; navx="MXdn=MX-F-s0:dn+MX-F-s1:dn"
                    for v in zero biasmean biasresid; do vars="$vars MX_$v=mx-MX-F-s0_$v+mx-MX-F-s1_$v"; done; }
  status "bias decomposition: $tags"
  $PY $S/mixed_domain.py bias --tags $tags || die "bias"
  status "tables"
  $VPY $S/mixed_domain.py report --name full --arms $arms shipped --refs P2H10 WP2 shipped ${vars:+--wod-vars $vars} ${navx:+--nav-extra $navx} || die "report"
  status "done"; date > "$D/DONE"; exit 0
fi
die "unknown stage $STAGE"
