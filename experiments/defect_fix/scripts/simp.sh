#!/usr/bin/env bash
# SIMP (block B, plans/2026-10-20-lon-anchor-prereg.md supplement 2): the NTV0 recipe from RB0-F-s0 with the S rows' road hinge + lat_onc merged
# into `region`, agent hinge + lat_side into `clear` (lib/nr1_simp.py; rm_train --simp-region / --simp-clear), lat_soft dropped, + LON-ANCHOR at
# the dose the main session picks from block A; three PAI shares of the 32 S rows (0 / 0.5 / 1.0) + the same-step control (the NTV0 recipe,
# share 0). 1 000 steps, seed 0, on1_JTCP. Box, repo root; every job through the GPU pool, ids in $B/jobs.txt. Env PRIO (11).
#   simp.sh pre <old rev>      draws SIMPW00 / SIMPW05 / SIMPW10 -> identity (the control's command 60 steps eager, <old rev> vs this code, no
#                              SIMP flag) + the offline gate (GPU job: simp.py gate on the share-0.5 arm's command)
#   simp.sh dry <dose>         the four commands and their diffs against the control
#   simp.sh flat               (supplement 3) the brake-needed navtrain tokens (lead_break.py nav: win BRK) -> $B/flat_brk.txt (CPU job)
#   simp.sh arms <dose>        control + 4 arms (A / B / C shares, F = B + flat history on the brake-needed rows; refuse unless identity and
#                              gate passed) -> inertia.py cf / report -> read-out; lead_break.py cf on N / B / F -> simp.py leadread
#   job bodies: weights | gate | flatlist | train <arm> <dose> | cf | report | lbcf | lbread
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
B=$DATA_DIR/runs/defect_fix/simp; ME=experiments/defect_fix/scripts/simp.sh; PY=experiments/defect_fix/scripts/simp.py; LA=experiments/defect_fix/scripts/lon_anchor.py
TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh; IN=experiments/defect_fix/scripts/inertia.py
LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2; RES=experiments/defect_fix/results/lon_anchor; POOL=on1_JTCP; RUNS=$DATA_DIR/runs/op_parity/runs
declare -A TAG=([N]=SIMPN1K-L18-s0 [A]=SIMPA1K-L18-s0 [B]=SIMPB1K-L18-s0 [C]=SIMPC1K-L18-s0 [F]=SIMPF1K-L18-s0) \
           WF=([N]=SIMPW00 [A]=SIMPW00 [B]=SIMPW05 [C]=SIMPW10 [F]=SIMPW05)
FLATF=$B/flat_brk.txt; LB=experiments/defect_fix/scripts/lead_break.py
LATF="--lat-onc 0.5 --lat-soft 0.1 --lat-side 0.5 "
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $B/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $B/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-11} \
     --no-check --log-dir $B/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $B/jobs.txt; echo "$id"; }
base() {  # base <tag> <wfile> <steps> [jt.sh]: NTV0-L18-s0's command (jt.sh train DRY, ONK=32, ARCROWS=brake) on on1_JTCP, warm-up 50, from RB0-F-s0
  local C; C=$(DRY=1 ONK=32 ARCROWS=brake WARM=50 bash ${4:-$JT} train $1 $POOL $LAT $2 $3 0 | tail -1)
  [[ $C == *"--init RB0-F-s0 "* && $C == *"--on-k 32 "* && $C == *"--warmup 50 "* && $C == *"--steps $3 "* && $C == *"w/$2.npz "* && $C == *"$LATF"* \
     && $C == *"--on-arc 0 "* ]] || die "unexpected command: $C"; echo "$C"; }
simp() {  # simp <tag> <wfile> <steps> [dose]: the control's command without lat_onc / lat_soft / lat_side, + region, clear [+ LON-ANCHOR]
  local C; C=$(base $1 $2 $3) || exit 1; echo "${C/$LATF/} --simp-region 1 --simp-clear 1${4:+ --on-anchor $4}"; }
cmd() { [[ $1 == N ]] && { base ${TAG[N]} ${WF[N]} 1000; return; }; echo "$(simp ${TAG[$1]} ${WF[$1]} 1000 ${2:?dose})$([[ $1 == F ]] && echo " --flat-brk $FLATF")"; }
case ${1:-} in
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in SIMPGG-old SIMPGG-new SIMPGATE ${TAG[@]}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  ls experiments/pai_fix/scenes/* > /dev/null 2>&1 || die "experiments/pai_fix/scenes/ missing"
  h=$(git rev-parse --short HEAD); echo $h > $B/code.txt
  w=$(sub simp-weights --vram 0.1 --cpu 2 --ram 24 --timeout-h 1 -- bash $ME weights) || exit 1
  FE=("${FT[@]}" --env JEV_RM_EAGER=1)
  a=$(sub simp-gg-old --after $w "${FE[@]}" --timeout-h 1 -- bash -c "$(base SIMPGG-old SIMPW00 60 $WT/$JT)") || exit 1
  b=$(sub simp-gg-new --after $w "${FE[@]}" --timeout-h 1 -- bash -c "$(base SIMPGG-new SIMPW00 60)") || exit 1
  e=$(sub simp-bitcmp --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a SIMPGG-old \
      --b SIMPGG-new --out lon_anchor/simp_identity --note "the NTV0 recipe from RB0-F-s0 on on1_JTCP / SIMPW00, 60 steps eager: code $rev vs $h without SIMP flags") || exit 1
  g=$(sub simp-gate --after $w "${FT[@]}" --timeout-h 1 -- bash $ME gate) || exit 1
  echo "$(date '+%F %T') pre (code $h, old $rev): weights $w identity $a $b bitcmp $e gate $g" | tee -a $B/stages.txt ;;
dry)
  D=${2:?dose}; for A in N A B C F; do echo "$A: $(cmd $A $D)"; done
  for A in A B C F; do echo "== control -> $A"; diff <(cmd N | tr ' ' '\n') <(cmd $A $D | tr ' ' '\n'); done ;;
ident)  # ident <old rev> <stem>: the control's command 60 steps eager, <old rev> vs this code (no new flag) -> results/lon_anchor/simp_identity_<stem>.json
  rev=${2:?old rev}; X=${3:?stem}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  FE=("${FT[@]}" --env JEV_RM_EAGER=1); h=$(git rev-parse --short HEAD)
  a=$(sub simp-gg-old-$X "${FE[@]}" --timeout-h 1 -- bash -c "$(base SIMPGG$X-old SIMPW00 60 $WT/$JT)") || exit 1
  b=$(sub simp-gg-new-$X "${FE[@]}" --timeout-h 1 -- bash -c "$(base SIMPGG$X-new SIMPW00 60)") || exit 1
  e=$(sub simp-bitcmp-$X --after $a,$b --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- $TR experiments/defect_fix/scripts/df.py bitcmp --a SIMPGG$X-old \
      --b SIMPGG$X-new --out lon_anchor/simp_identity_$X --note "the NTV0 recipe from RB0-F-s0 on on1_JTCP / SIMPW00, 60 steps eager: code $rev vs $h without new flags") || exit 1
  echo "$(date '+%F %T') ident $X (code $h, old $rev): $a $b bitcmp $e" | tee -a $B/stages.txt ;;
smoke)   # smoke [after]: the F arm's command (dose 0.0011 = A's x 1, 60 steps, --diag-every 20) as a code check of region / clear / flatten
  s_=$(sub simp-smoke ${2:+--after $2} "${FT[@]}" --timeout-h 1 -- bash $ME smokebody) || exit 1; echo "$(date '+%F %T') smoke: $s_" | tee -a $B/stages.txt ;;
smokebody)
  X=SIMPSM-L18-s0; C=$(cmd F 0.0011) || exit 1; C=${C/--tag ${TAG[F]} /--tag $X }; C="${C/--steps 1000 /--steps 60 } --diag-every 20"
  echo "$(date '+%F %T') smoke: $C" | tee -a $B/stages.txt
  $C > $B/smoke.log 2>&1 || die "smoke run (log $B/smoke.log)"
  grep -E "SIMP attach|SIMP-F flat|lon_anchor labels" $B/smoke.log | tee $B/smoke_check.txt
  f=$(ls -d $DATA_DIR/runs/route_mem/train-$X/* | tail -1)/events.jsonl
  $TR - $f $RES/simp_smoke.json <<'PY' || die "smoke check"
import json, sys
S = {}
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e.get("kind") == "scalar" and any(k in e["tag"] for k in ("simp_", "flat/", "lon_anchor", "grad/", "dev/ade", "loss/total")):
        S.setdefault(e["tag"], []).append(round(e["value"], 4))
ok = all(any(k in t for t in S) for k in ("on/simp_region_ho", "on/simp_clear_ho", "flat/rows", "on/lon_anchor_ho")) and all(v == v for x in S.values() for v in x)
json.dump(dict(pass_=bool(ok), scalars={k: v[-3:] for k, v in S.items()}), open(sys.argv[2], "w"), indent=1)
print("smoke", "PASS" if ok else "FAIL", {k: v[-1] for k, v in S.items() if "simp" in k or "flat" in k})
sys.exit(0 if ok else 2)
PY
  rm -rf $RUNS/$X ;;
flat) f=$(sub simp-flatlist --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME flatlist) || exit 1; echo "$(date '+%F %T') flat: $f" | tee -a $B/stages.txt ;;
arms)
  D=${2:?dose}; for X in ${TAG[@]}; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  [[ -s $FLATF ]] || die "no $FLATF (simp.sh flat)"
  declare -A T_; for A in N A B C F; do T_[$A]=$(sub simp-train-${TAG[$A]} "${FT[@]}" --timeout-h 2 -- bash $ME train $A $D) || exit 1; done
  f=$(sub simp-cf --after $(IFS=,; echo "${T_[*]}") --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME cf) || exit 1
  r=$(sub simp-report --after $f --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME report) || exit 1
  l=$(sub simp-lbcf --after ${T_[N]},${T_[B]},${T_[F]} --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME lbcf) || exit 1
  k=$(sub simp-lbread --after $l --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME lbread) || exit 1
  echo "$(date '+%F %T') arms (code $(git rev-parse --short HEAD), LON-ANCHOR dose $D): train ${T_[*]} cf $f report $r lead-break cf $l read $k" | tee -a $B/stages.txt ;;
weights) for x in 00:0 05:0.5 10:1.0; do $TR $LA weights --pool $POOL --base JT00S --share ${x#*:} --out SIMPW${x%%:*} --require-fix || exit 1; done ;;
gate)    $TR $PY gate -- $(simp SIMPGATE SIMPW05 1000 | cut -d' ' -f3-) ;;
flatlist)
  $TR - $FLATF $RES/simp_flat.json <<'PY' || die "flat list"
import json, os, sys
import pandas as pd
N = pd.read_parquet(os.environ["DATA_DIR"] + "/runs/defect_fix/lead_break/nav.parquet")
b = N[N.win == "BRK"]
open(sys.argv[1], "w").write("\n".join(b.token.astype(str)) + "\n")
rec = dict(source="runs/defect_fix/lead_break/nav.parquet (lead_break.py nav; navtrain_full, v0 >= 2 m/s)", lead_rows=len(N), brk_rows=len(b),
           brk_logs=int(b.log.nunique()), brk_already_decelerating=int((b.ax <= -0.5).sum()), brk_initiation=int((b.ax > -0.5).sum()))
json.dump(rec, open(sys.argv[2], "w"), indent=1); print(json.dumps(rec))
PY
  ;;
train)
  A=$2; python3 - $RES <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
assert json.load(open(f"{r}/simp_identity.json"))["identical"], "switch-off identity failed"
assert json.load(open(f"{r}/simp_identity_F.json"))["identical"], "switch-off identity (flat history code) failed"
assert json.load(open(f"{r}/simp_smoke.json"))["pass_"], "SIMP-F smoke failed"
assert json.load(open(f"{r}/simp_gate.json"))["gate"]["pass_"], "SIMP gate failed"
assert json.load(open(f"{r}/gate_v2.json"))["G1"]["pass_"], "LON-ANCHOR G1 failed"
PY
  CMD="$(cmd $A $3)" || exit 1; echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $B/stages.txt; exec $CMD ;;
cf)     $TR $IN cf --tags NTV0-L18-s0,${TAG[N]},${TAG[A]},${TAG[B]},${TAG[C]},${TAG[F]},RB0-F-s0,RSG-L18-s0 --gate-tag RSG-L18-s0 --name simp ;;
lbcf)   $TR $LB cf --tags RB0-F-s0,${TAG[N]},${TAG[B]},${TAG[F]} --name simp ;;
lbread) $TR $PY leadread ${TAG[F]} ${TAG[N]} ${TAG[B]} ;;
report)
  $TR $IN report --name simp > $B/inertia_report.log 2>&1 || die "inertia report"
  cp experiments/defect_fix/results/inertia/report_simp.md $RES/simp_inertia_report.md && cp experiments/defect_fix/results/inertia/report_simp.json $RES/simp_inertia_report.json || die "no report"
  $TR $LA inertia --name simp --ctl ${TAG[N]} --arms ${TAG[A]} ${TAG[B]} ${TAG[C]} ${TAG[F]} --out-stem simp_inertia || die "read-out"; date '+%F %T' > $B/DONE ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
