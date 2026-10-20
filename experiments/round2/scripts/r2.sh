#!/usr/bin/env bash
# ROUND2 (plans/2026-10-20-round2-prereg.md): three continuations of NTV0-L18-s0 with the NTV0 recipe, one seed. CTRL = the recipe unchanged
# (on1_JT / JT00S, on-k 32); J = + the SIMP-G onset rows flattened with probability 0.5 + PAI human-state imitation rows (on1_PAIH plain and the
# two time-dilated stretch pools, 32 of the 83 navtrain rows of a batch, plain : f0.5 : f1.5 = 2 : 1 : 1; merged pool on1_JTPH, on-k 64);
# J-half = J on half A's scenes. Steps from the registered rule (r2.py steps; CTRL = J's). Then inertia.py + lead_break.py (offline gates)
# and the PAI-FIX native read (vl, chunks "v t") of the three arms. Box, repo root; every job through the GPU pool; ids in $O/jobs.txt.
#   r2.sh dry <steps>        the three training commands and their diffs against CTRL
#   r2.sh pre <old rev>      identity (CTRL's command 60 steps eager: <old rev> vs this code without new flags, and this code with --flat-brk
#                            <onset list> --flat-p 0) + smoke (probe pools on1_PAIHT*, when the PAI-ROWS reprobe has passed: merge on1_JTPHT ->
#                            fix -> draw R2JT -> J's command 60 steps -> r2.py smoke)
#   r2.sh launch             the whole chain, gated in the queue on PAI-ROWS H/DONE: merge on1_JTPH -> fix -> draws R2JF / R2JA -> steps ->
#                            3 trainings -> per arm PAI-FIX vl -> report; inertia cf -> report -> read-out; lead_break cf -> read-outs -> summary
#   r2.sh stage2             (main session / user, 10-21: J picked) PAI 281 (g10, FAST=1, NTV0 + ACC + GAP's chunk lists, pai_box.sh as the
#                            lineage / accgap reads): R2J + ACC + GAP first at PRIO 20 -> facts -> s2gate (vs stored NTV0 + ACC + GAP; worse =
#                            paired CI upper < 0 or zeros >= +5 -> S2STOP) -> else stage2rest: R2J native, R2C native + ACC + GAP, PAI-FIX
#                            ACC + GAP of R2J -> s2report (r2.py r281 vs R2C and NTV0) -> $O/DONE2
#   r2.sh wait               blocks until $O/DONE, a PAI-ROWS STOP or a failed job of this lane (one run_in_background waiter)
#   job bodies: ident | smokepool | smoketrain | merge | weights | train <arm> | paifix | cf | report | lbcf | lbread | summary
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/round2; ME=experiments/round2/scripts/r2.sh; PY=experiments/round2/scripts/r2.py; RES=experiments/round2/results
TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python; JT=experiments/defect_fix/scripts/jt.sh; RF=experiments/route_follow/scripts
IN=experiments/defect_fix/scripts/inertia.py; LB=experiments/defect_fix/scripts/lead_break.py; LA=experiments/defect_fix/scripts/lon_anchor.py
SP=experiments/defect_fix/scripts/simp.py; DFR=experiments/defect_fix/results; CR=$DATA_DIR/runs/op_parity/cache; RUNS=$DATA_DIR/runs/op_parity/runs
LAT=$DATA_DIR/runs/defect_fix/lat/L18NR1v2; PR=$DATA_DIR/runs/pai_rows; FLATG=$DATA_DIR/runs/defect_fix/simp/flat_onset.txt; INIT=NTV0-L18-s0; FP=0.5
PARTS="on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_L18YS:ys:self on1_L18Y:y:self on1_NR1TV:tv:both"     # = jt.sh: on1_JT's parts in order
declare -A TAG=([C]=R2C-L18-s0 [J]=R2J-L18-s0 [JH]=R2JH-L18-s0)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner round2 --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
base() {  # base <tag> <pool> <wfile> <steps> <on-k> [jt.sh]: NTV0-L18-s0's command (jt.sh train DRY, ARCROWS=brake, warm-up 500) from NTV0-L18-s0
  local C; C=$(DRY=1 ONK=$5 ARCROWS=brake bash ${6:-$JT} train $1 $2 $LAT $3 $4 0 | tail -1)
  [[ $C == *"--init RB0-F-s0 "* && $C == *"--on-k $5 "* && $C == *"--warmup 500 "* && $C == *"--steps $4 "* && $C == *"w/$3.npz "* \
     && $C == *"--on $2 "* && $C == *"--on-arc 0 "* && $C == *"--lr-scale 0.1 "* ]] || die "unexpected command: $C"
  echo "${C/--init RB0-F-s0 /--init $INIT }"; }
cmd() {  # cmd <arm> <steps> [tag pool wfile]: CTRL = the recipe; J / JH = merged pool, on-k 64 (32 S + 32 PAI human rows expected), onset flattening
  local A=$1 N=$2
  case $A in
  C) base ${3:-${TAG[C]}} on1_JT JT00S $N 32 ;;
  J|JH) local W=$([[ $A == J ]] && echo R2JF || echo R2JA); echo "$(base ${3:-${TAG[$A]}} ${4:-on1_JTPH} ${5:-$W} $N 64) --flat-brk $FLATG --flat-p $FP" ;;
  esac; }
steps() { python3 -c "import json; print(json.load(open('$O/steps.json'))['$1']['steps'])"; }
read281() {  # read281 <arm> <vl|ag> <prio>: one PAI 281 read (NTV0 + ACC + GAP's chunk lists, g10, FAST=1, CONC 4) -> "<chunk ids...> <facts job id>"
  local A=$1 s=$2 R=$O/pai281/${TAG[$1]}-$2 SV ids t
  SV=$([[ $s == vl ]] && echo "JEV_VCONT=1.0 JEV_LEAD=1" || echo "JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7")
  CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks SERVE="$SV" WIZ=controller.gains.rel_front_steering_angle_weight=10 CONC=4 \
    ALPASIM_FAST=1 PRIO=$3 bash experiments/alpasim/scripts/pai_box.sh submit $R ${TAG[$A]} > $R.submit.txt || die "pai submit ${TAG[$A]} $s"
  ids=$(awk '$2 ~ /^pai-/ {print $1}' $R.submit.txt); [[ $(wc -w <<< "$ids") == 8 ]] || die "${TAG[$A]} $s: not 8 chunks"
  awk '$2 ~ /^pai-/ {print $1, $2}' $R.submit.txt >> $O/jobs.txt
  t=$(tax281 ${TAG[$A]}-281$s $R "$(paste -sd, <<< "$ids")" $3) || exit 1; echo $ids $t; }
tax281() {  # tax281 <name> <read root> <after ids> <prio>: rf_tax + pai_fix facts -> $DATA_DIR/runs/pai_fix/facts/<name>.csv; prints the job id
  local t; t=$(PRIO=$4 AFTER=$3 bash experiments/pai_fix/scripts/pai_fix.sh tax $1 $2 | tail -1) && [[ -n $t ]] || die "tax $1"
  echo "$t pai_fix-tax-$1" >> $O/jobs.txt; echo $t; }
case ${1:-} in
dry)
  N=${2:-9600}; for A in C J JH; do echo "$A: $(cmd $A $N)"; done
  for A in J JH; do echo "== CTRL -> $A"; diff <(cmd C $N | tr ' ' '\n') <(cmd $A $N | tr ' ' '\n'); done ;;
pre)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in R2GG-old R2GG-new R2GGP0-new R2SM-L18-s0; do [[ ! -e $RUNS/$X/ckpt-final.pt ]] || die "$X exists"; done
  [[ -s $FLATG ]] || die "no $FLATG"; h=$(git rev-parse --short HEAD); echo $h > $O/code.txt
  FE=("${FT[@]}" --env JEV_RM_EAGER=1)
  a=$(sub r2-gg-old "${FE[@]}" --timeout-h 1 -- bash -c "cd $WT && $(cmd C 60 R2GG-old)") || exit 1          # the old code runs from its worktree
  b=$(sub r2-gg-new "${FE[@]}" --timeout-h 1 -- bash -c "$(cmd C 60 R2GG-new)") || exit 1
  c=$(sub r2-gg-p0 "${FE[@]}" --timeout-h 1 -- bash -c "$(cmd C 60 R2GGP0-new) --flat-brk $FLATG --flat-p 0") || exit 1
  e=$(sub r2-ident --after $a,$b,$c --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident $rev $h) || exit 1
  m=$(sub r2-smokepool --when-exists $PR/H/probe/REPROBE_OK --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- bash $ME smokepool) || exit 1
  s=$(sub r2-smoke --after $m "${FT[@]}" --timeout-h 1 -- bash $ME smoketrain) || exit 1
  echo "$(date '+%F %T') pre (code $h, old $rev): identity $a $b $c bitcmp $e; smoke pool $m smoke $s" | tee -a $O/stages.txt ;;
ident)   # ident <old rev> <new rev>: both comparisons + the code each run used (meta.json git sha / cwd) -> results/identity.json
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R2GG-old --b R2GG-new --out round2_identity_off \
      --note "CTRL's command (NTV0 recipe from NTV0-L18-s0, on1_JT / JT00S) 60 steps eager: code $2 at $DATA_DIR/tmp/df_old_$2 vs $3, no new flag"; r1=$?
  $TR experiments/defect_fix/scripts/df.py bitcmp --a R2GG-old --b R2GGP0-new --out round2_identity_p0 \
      --note "the same: code $2 without --flat-brk vs $3 with --flat-brk <SIMP-G onset list> --flat-p 0"; r2=$?
  $TR - $DFR $RES $DATA_DIR $r1 $r2 <<'PY'
import glob, json, sys
d, res, D, r1, r2 = sys.argv[1:6]
m = lambda t: json.load(open(sorted(glob.glob(f"{D}/runs/route_mem/train-{t}/*/meta.json"))[-1]))  # noqa: E731
code = {t: dict(sha=m(t)["git"]["sha"][:8], dirty=m(t)["git"]["dirty"], cwd=m(t)["cwd"]) for t in ("R2GG-old", "R2GG-new", "R2GGP0-new")}
out = dict(off=json.load(open(f"{d}/round2_identity_off.json")), p0=json.load(open(f"{d}/round2_identity_p0.json")), code=code)
out["two_codes"] = code["R2GG-old"]["sha"] != code["R2GG-new"]["sha"] and code["R2GG-old"]["cwd"] != code["R2GG-new"]["cwd"]
out["identical"] = bool(out["off"]["identical"] and out["p0"]["identical"] and out["two_codes"])
json.dump(out, open(f"{res}/identity.json", "w"), indent=1)
print(json.dumps(dict(identical=out["identical"], code=code)))
sys.exit(0 if out["identical"] else 2)
PY
  ;;
smokepool)
  for x in $CR/on1_JTPHT $CR/on1_JTPHT@warp; do rm -rf $x; done
  $TR $RF/rf_pait.py merge --pools on1_L18RB0:rb0:self on1_L18RTB1:rtb1:self on1_PAIHT:ph:both on1_PAIHT05:ph05:both on1_PAIHT15:ph15:both \
      --out JTPHT --nan-keys fut_reg || die "smoke merge"
  $TR $PY fix --pool on1_JTPHT --prefix-only || die "smoke fix"
  $TR $PY weights --pool on1_JTPHT --out R2JT --prefix-only || die "smoke weights" ;;
smoketrain)  # J's command on the smoke pool, 60 steps, --diag-every 20 -> r2.py smoke; then the smoke checkpoint and pool are removed
  X=R2SM-L18-s0; C="$(cmd J 60 $X on1_JTPHT R2JT) --diag-every 20" || exit 1
  echo "$(date '+%F %T') smoke: $C" | tee -a $O/stages.txt
  $C > $O/smoke.log 2>&1 || die "smoke run (log $O/smoke.log)"
  $TR $PY smoke --tag $X --wfile R2JT --flat-p $FP; r=$?
  rm -rf $RUNS/$X $CR/on1_JTPHT $CR/on1_JTPHT@warp $DATA_DIR/runs/route_mem/feat/on1_JTPHT.npz; exit $r ;;
launch)
  for A in C J JH; do [[ ! -e $RUNS/${TAG[$A]}/ckpt-final.pt ]] || die "${TAG[$A]} exists"; done
  [[ ! -e $CR/on1_JTPH ]] || die "on1_JTPH exists: never overwritten"
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$RES/identity.json'))['identical'] and json.load(open('$RES/smoke.json'))['pass_'] else 1)" \
    || die "identity / smoke not passed"
  h=$(git rev-parse --short HEAD); echo "$h launch" >> $O/code.txt
  m=$(sub r2-merge --when-exists $PR/H/DONE --vram 0.1 --cpu 4 --ram 96 --timeout-h 3 -- bash $ME merge) || exit 1
  w=$(sub r2-weights --after $m --vram 0.1 --cpu 4 --ram 48 --timeout-h 1 -- bash $ME weights) || exit 1
  declare -A T_; for A in C J JH; do T_[$A]=$(sub r2-train-${TAG[$A]} --after $w "${FT[@]}" --timeout-h 4 -- bash $ME train $A) || exit 1; done
  x=(); for A in C J JH; do                                       # PAI-FIX native (vl), chunks v + t, behind each training
    o=$(PRIO=13 AFTER=${T_[$A]} bash experiments/pai_fix/scripts/pai_fix.sh read ${TAG[$A]} vl "v t" | tail -1) || die "pai_fix read ${TAG[$A]}"
    x+=(${o%% *}); echo "${o%% *} pai_fix-tax-${TAG[$A]}-vl" >> $O/jobs.txt
    awk '{print $1, $2}' $DATA_DIR/runs/pai_fix/reads/${TAG[$A]}-vl/jobs.txt >> $O/jobs.txt; done
  p=$(sub r2-paifix --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME paifix) || exit 1
  ta=$(csv ${T_[C]} ${T_[J]} ${T_[JH]})
  f=$(sub r2-cf --after $ta --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME cf) || exit 1
  r=$(sub r2-report --after $f --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME report) || exit 1
  l=$(sub r2-lbcf --after $ta --vram 24 --cpu 16 --ram 96 --timeout-h 2 -- bash $ME lbcf) || exit 1
  k=$(sub r2-lbread --after $l --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME lbread) || exit 1
  s=$(sub r2-summary --after $p,$r,$k --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME summary) || exit 1
  echo "$(date '+%F %T') launch (code $h): merge $m (when $PR/H/DONE) weights $w train ${T_[C]} ${T_[J]} ${T_[JH]} PAI-FIX ${x[*]} report $p" \
       "inertia $f $r lead-break $l $k summary $s" | tee -a $O/stages.txt ;;
merge)   # the PAI-ROWS H pools passed (DONE, no STOP, R1 of the full plain pool, the reprobe's f = 1 gate) -> on1_JTPH -> fix
  [[ -f $PR/H/DONE && ! -f $PR/H/STOP && -f $PR/H/probe/REPROBE_OK ]] || die "PAI-ROWS H not passed"
  $TR - $PR <<'PY' || die "PAI-ROWS gate files"
import json, sys
p = sys.argv[1]
r1, g1 = json.load(open(f"{p}/H/r1_PAIH.json")), json.load(open(f"{p}/H/probe/gate1.json"))
ok = lambda d: bool(d.get("pass_", d.get("pass", d.get("ok"))))  # noqa: E731
print("R1", ok(r1), "gate1", ok(g1)); sys.exit(0 if ok(r1) and ok(g1) else 1)
PY
  free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 250 )) || die "free $free GB < 250 GB (on1_JTPH needs about 170)"
  $TR $RF/rf_pait.py merge --pools $PARTS on1_PAIH:ph:both on1_PAIH05:ph05:both on1_PAIH15:ph15:both --out JTPH --nan-keys fut_reg || die "merge"
  $TR $PY fix --pool on1_JTPH || die "fix" ;;
weights)
  $TR $PY weights --pool on1_JTPH --out R2JF || die "weights J"
  $TR $PY weights --pool on1_JTPH --out R2JA --half physical_ai_av/pai-rows-half-a@v1 || die "weights J-half"
  $TR $PY steps --full R2JF --half R2JA --k 32 || die "steps" ;;
train)
  A=$2; python3 - $RES <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
assert json.load(open(f"{r}/identity.json"))["identical"], "switch-off identity failed"
assert json.load(open(f"{r}/smoke.json"))["pass_"], "smoke failed"
PY
  N=$(steps $([[ $A == C ]] && echo CTRL || echo $A)) || die "no steps"; CMD="$(cmd $A $N)" || exit 1
  echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $O/stages.txt; exec $CMD ;;
paifix)
  F_=$DATA_DIR/runs/pai_fix/facts; rd=(NTV0-L18-s0-vl=$F_/NTV0-L18-s0-vl.csv); for A in C J JH; do rd+=(${TAG[$A]}-vl=$F_/${TAG[$A]}-vl.csv); done
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref ${TAG[C]}-vl --out $RES/paifix_vl_vs_ctrl.md --scores $RES/paifix_vl.csv > /dev/null \
    || die "pai_fix report vs CTRL"
  $V experiments/pai_fix/scripts/pai_fix.py report --read "${rd[@]}" --ref NTV0-L18-s0-vl --out $RES/paifix_vl_vs_ntv0.md > /dev/null || die "pai_fix report vs NTV0" ;;
cf)     $TR $IN cf --tags NTV0-L18-s0,${TAG[C]},${TAG[J]},${TAG[JH]},RSG-L18-s0 --gate-tag RSG-L18-s0 --name round2 ;;
report)
  $TR $IN report --name round2 > $O/inertia_report.log 2>&1 || die "inertia report"
  cp $DFR/inertia/report_round2.md $DFR/inertia/report_round2.json $RES/ || die "no inertia report"
  $TR $LA inertia --name round2 --ctl ${TAG[C]} --arms ${TAG[J]} ${TAG[JH]} --out-stem round2_inertia || die "inertia read-out"
  cp $DFR/lon_anchor/round2_inertia.md $RES/inertia.md && cp $DFR/lon_anchor/round2_inertia.json $RES/inertia.json ;;
lbcf)   $TR $LB cf --tags RB0-F-s0,NTV0-L18-s0,${TAG[C]},${TAG[J]},${TAG[JH]} --name round2 ;;
lbread)  # simp.py leadread per arm (its output stem is per name, so each read is copied before the next)
  for A in J JH; do $TR $SP leadread ${TAG[$A]} ${TAG[C]} NTV0-L18-s0 round2 || die "leadread $A"
    cp $DFR/lon_anchor/round2_leadbreak.md $RES/leadbreak_$A.md && cp $DFR/lon_anchor/round2_leadbreak.json $RES/leadbreak_$A.json || die "copy $A"; done ;;
stage2)   # user order (10-21): R2J + ACC + GAP on PAI 281 first (PRIO 20), stored NTV0 + ACC + GAP facts; the gate (in the queue) decides the rest
  mkdir -p $O/pai281; rm -f $O/S2GATE $O/S2STOP; x=($(read281 J ag 20)) || exit 1
  t=$(tax281 NTV0-L18-s0-281ag $DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0 "" 20) || exit 1
  g=$(PRIO=20 sub r2-s2gate --after $(csv ${x[-1]} $t) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME s2gate) || exit 1
  echo "$(date '+%F %T') stage2 (code $(git rev-parse --short HEAD)): R2J ag chunks ${x[*]:0:${#x[@]}-1} facts ${x[-1]}; NTV0 ag facts $t; gate $g" | tee -a $O/stages.txt ;;
s2gate)  # worse = paired mean CI upper < 0 or zeros >= +5 vs stored NTV0 + ACC + GAP -> S2STOP (nothing else submitted); else the rest
  F_=$DATA_DIR/runs/pai_fix/facts
  $TR $PY r281 --read NTV0-ag=$F_/NTV0-L18-s0-281ag.csv R2J-ag=$F_/${TAG[J]}-281ag.csv --pairs R2J-ag:NTV0-ag --out $RES/pai281_R2J_ag.md \
      --title "PAI 281, ACC + GAP: R2J vs stored NTV0-L18-s0" || die "r281 R2J ag"
  w=$(python3 -c "import json; d=json.load(open('$RES/pai281_R2J_ag.json'))['pairs']['R2J-ag:NTV0-ag']; print(int(d['mean']['hi'] < 0 or d['zeros']['diff'] >= 5))")
  if [[ $w == 1 ]]; then echo "$(date '+%F %T') R2J + ACC + GAP worse than NTV0 + ACC + GAP: the rest of stage 2 is not run" | tee $O/S2STOP >> $O/stages.txt
  else bash $ME stage2rest || die "stage2rest"; fi
  date '+%F %T' > $O/S2GATE ;;
stage2rest)  # R2J native, R2C native + ACC + GAP, PAI-FIX ACC + GAP of R2J, the stored native N0's facts -> s2report
  x=(); for q in "J vl" "C vl" "C ag"; do x+=($(read281 $q 13)) || exit 1; done
  x+=($(tax281 NTV0-L18-s0-281vl $DATA_DIR/runs/route_follow/pai_input_loop/N0 "" 13)) || exit 1
  o=$(PRIO=13 bash experiments/pai_fix/scripts/pai_fix.sh read ${TAG[J]} ag "v t" | tail -1) || die "pai_fix read ag"
  x+=(${o%% *}); echo "${o%% *} pai_fix-tax-${TAG[J]}-ag" >> $O/jobs.txt; awk '{print $1, $2}' $DATA_DIR/runs/pai_fix/reads/${TAG[J]}-ag/jobs.txt >> $O/jobs.txt
  s=$(sub r2-s2report --after $(csv ${x[@]}) --vram 0.1 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME s2report) || exit 1
  echo "$(date '+%F %T') stage2rest: ${x[*]} report $s" | tee -a $O/stages.txt ;;
s2report)
  F_=$DATA_DIR/runs/pai_fix/facts
  for s in vl ag; do $TR $PY r281 --read NTV0-$s=$F_/NTV0-L18-s0-281$s.csv R2C-$s=$F_/${TAG[C]}-281$s.csv R2J-$s=$F_/${TAG[J]}-281$s.csv \
      --pairs R2J-$s:R2C-$s R2J-$s:NTV0-$s R2C-$s:NTV0-$s --out $RES/pai281_$s.md --title "PAI 281, $s serve: R2J / R2C vs stored NTV0-L18-s0" || die "r281 $s"; done
  $V experiments/pai_fix/scripts/pai_fix.py report --read NTV0-L18-s0-ag=$F_/NTV0-L18-s0-ag.csv SIMPN1K-L18-s0-ag=$F_/SIMPN1K-L18-s0-ag.csv \
      ${TAG[J]}-ag=$F_/${TAG[J]}-ag.csv --ref NTV0-L18-s0-ag --out $RES/paifix_ag_R2J.md --scores $RES/paifix_ag.csv > /dev/null || die "pai_fix report ag"
  date '+%F %T' > $O/DONE2 ;;
summary)
  $TR $PY costs --tags ${TAG[C]} ${TAG[J]} ${TAG[JH]} || die "costs"; date '+%F %T' > $O/DONE ;;
wait)
  until [[ -f $O/${WAITFOR:-DONE} || -f $PR/H/STOP ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  [[ -f $PR/H/STOP ]] && echo "PAI-ROWS STOP: $(cat $PR/H/STOP)"; [[ -f $O/${WAITFOR:-DONE} ]] && echo "${WAITFOR:-DONE} $(cat $O/${WAITFOR:-DONE})"; true ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
