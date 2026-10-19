#!/usr/bin/env bash
# STAND-B (NR1 prereg supplement 19): arm B of the JTX STAND pilot (STANDFMX1K-L18-s0 = STAND W_S 0.0068 + FREE-M W 0.002 / move / 0.95) with
# (B1) FREE-M's free check made stricter (--on-free-clear 15,2) and (B2) FREE-M at W 0.001, 1 000 steps against JTX1K-L18-s0; and B at full
# length (9 500 steps, warm-up 500: the on1_JTX analogue of JTN-L18-s0). Box, repo root; every job through the GPU pool, ids in $S/jobs.txt.
# Env PRIO (12; reads PRIO + 1).
#   stand_b.sh check               DRY commands of B1, B2 and the full arm against B's / the control's -> $S/cmd_*.txt, results/stand_b/cmd_diff.txt
#   stand_b.sh smoke <old rev>     60 steps eager: B's command on <old rev> and on this code (bit-identity), B1's command (runs, fires no more
#                                  than B) -> results/stand_b/{identity,smoke}.json; prints the check job id
#   stand_b.sh chain [after ids]   the one submission: B1, B2, full trains; their reads; the pilot summary (psum) and the full summary (fsum)
#   stand_b.sh smokeh              (addendum) 60 steps eager of the full-B2 command (FH) -> results/stand_b/smoke_H.json; prints the check job id
#   stand_b.sh checkh              DRY diff of FH against F (only tag and FREE-M dose) -> results/stand_b/cmd_diff_H.txt
#   stand_b.sh chainh [after ids]  the one submission of full-length B2: train -> held 18 chunks + held 460 + PAI 281 -> fsumh
#   job bodies: train <arm> | psum | fsum | fsumh
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=$DATA_DIR/runs/defect_fix; S=$O/stand_b; J=$O/jt; ME=experiments/defect_fix/scripts/stand_b.sh; SB=experiments/defect_fix/scripts/stand_b.py
SP=experiments/defect_fix/scripts/stand.py; FM=experiments/defect_fix/scripts/free_m.py; JT=experiments/defect_fix/scripts/jt.sh
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python
LAT=$O/lat/L18NR1v2; RES=experiments/defect_fix/results/stand_b; POOL=on1_JTX; DRAW=JTX00; CTL=JTX1K-L18-s0; CL=JTX1K; SUB="0 3 6 9 12 15"
WS=$(python3 -c "import json; d = json.load(open('experiments/defect_fix/results/stand/jtx/dose.json')); assert d['in_range']; print(d['W_S'])") || exit 1
CLEAR=15,2
declare -A TAG=([B]=STANDFMX1K-L18-s0 [B1]=STANDFMXG1K-L18-s0 [B2]=STANDFMXH1K-L18-s0 [F]=STANDFMX-L18-s0 [FH]=STANDFMXH-L18-s0) LAB=([B]=STANDFMX [B1]=STANDFMXG [B2]=STANDFMXH)
declare -A WF=([B]=0.002 [B1]=0.002 [B2]=0.001 [F]=0.002 [FH]=0.001)
TF=${TAG[F]}; TH=${TAG[FH]}; NREF=JTN-L18-s1; PREF=NTV0-L18-s0; REF=$DATA_DIR/runs/route_follow/acc_gap/read/s2
PAISERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
AH=$DATA_DIR/runs/alpasim_proxy/arena/hold; RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)
TRN=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
mkdir -p $S/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $S/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner defect-fix --name "$name" --tries ${TRIES:-1} --priority ${PRIO:-12} \
     --no-check --log-dir $S/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $S/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
base() {  # base <tag> <steps> <warm>: JTX1K-L18-s0's recipe (jt.sh train DRY, ARCROWS=brake, on1_JTX, JTX00), checked
  local C; C=$(DRY=1 ARCROWS=brake WARM=$3 bash $JT train $1 $POOL $LAT $DRAW $2 0 | tail -1)
  [[ $C == *"--on-k 70 "* && $C == *"w/$DRAW.npz "* && $C == *"--on $POOL "* && $C == *"--warmup $3 "* && $C == *"--tag $1 "* && $C == *"--steps $2 "* && $C == *"--init RB0-F-s0 "* ]] \
    || die "unexpected command: $C"; echo "$C"; }
term() {  # term <arm>: B's switches (STAND W_S, FREE-M) with the arm's FREE-M weight; B1 adds the stricter free check
  local t="--on-stand $WS --on-stand-margin 1.0 --on-free ${WF[$1]} --on-free-mode move --on-free-frac 0.95"; [[ $1 == B1 ]] && t="$t --on-free-clear $CLEAR"; echo "$t"; }
cmd() {  # cmd <arm> [tag] [steps warm]
  local T=${2:-${TAG[$1]}}; echo "$(base $T ${3:-1000} ${4:-50}) $(term $1)"; }
pai() {  # pai <tag> <out> <after>: PAI 281 with the d366 serve on NTV0 + ACC + GAP's chunk lists -> prints the job ids
  CHUNKS_FROM=$O/pai_accgap/$PREF/chunks SERVE="$PAISERVE" WIZ=$WIZ CONC=4 PRIO=$((${PRIO:-12} + 1)) AFTER=$3 bash experiments/alpasim/scripts/pai_box.sh submit $2 $1 \
    > $S/pai_$1.submit.txt || die "pai submit $1"; awk '$2 ~ /^pai-/' $S/pai_$1.submit.txt >> $S/jobs.txt; awk '$2 ~ /^pai-/ {print $1}' $S/pai_$1.submit.txt | paste -sd,; }
case ${1:-} in
check)
  base $CTL 1000 50 > $S/cmd_ctl.txt || exit 1
  diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $DATA_DIR/runs/practice_nuplan/cmd/$CTL.txt) > /dev/null || die "DRY $CTL != the stored d368 command"
  for A in B B1 B2; do cmd $A > $S/cmd_$A.txt || exit 1; done; cmd F $TF 9500 500 > $S/cmd_F.txt || exit 1
  B0=$(grep -o "experiments/route_mem/scripts/rm_train.py train --tag ${TAG[B]} .*" $O/stand_x/stages.txt | tail -1); [[ -n $B0 ]] || die "B's logged command"
  [[ "$(cut -d' ' -f2- $S/cmd_B.txt)" == "$B0" ]] || die "B's DRY command != its logged training command"
  { echo "== ${TAG[B]} (logged training command) = DRY here: yes"
    for A in B1 B2 F; do echo "== ${TAG[B]} -> ${TAG[$A]}"; diff <(tr ' ' '\n' < $S/cmd_B.txt) <(tr ' ' '\n' < $S/cmd_$A.txt); done
    echo "== $CTL -> ${TAG[B]}"; diff <(tr ' ' '\n' < $S/cmd_ctl.txt) <(tr ' ' '\n' < $S/cmd_B.txt); } > $RES/cmd_diff.txt
  cat $RES/cmd_diff.txt
  declare -A NW=([B1]=3 [B2]=2 [F]=3)   # --tag + (--on-free-clear 15,2) | (W) | (--steps, --warmup)
  for A in B1 B2 F; do n=$(diff <(tr ' ' '\n' < $S/cmd_B.txt) <(tr ' ' '\n' < $S/cmd_$A.txt) | grep -c '^>'); [[ $n == "${NW[$A]}" ]] || die "arm $A: $n new tokens, want ${NW[$A]}"; done
  echo "diff ok" ;;
smoke)
  rev=${2:?old rev}; WT=$DATA_DIR/tmp/df_old_$rev; [[ -d $WT ]] || git worktree add --detach $WT $rev > /dev/null || exit 1
  for X in STBGG-old STBGG-new STBGG-B1; do [[ ! -e $DATA_DIR/runs/op_parity/runs/$X/ckpt-final.pt ]] || die "$X exists"; done
  FE=("${TRN[@]}" --env JEV_RM_EAGER=1 --timeout-h 1); h=$(git rev-parse --short HEAD)
  a=$(sub stb-gg-old "${FE[@]}" -- bash -c "cd $WT && $(cmd B STBGG-old 60 50)") || exit 1
  b=$(sub stb-gg-new "${FE[@]}" -- bash -c "$(cmd B STBGG-new 60 50)") || exit 1
  c=$(sub stb-gg-b1 "${FE[@]}" -- bash -c "$(cmd B1 STBGG-B1 60 50)") || exit 1
  e=$(sub stb-smokecheck --after $a,$b,$c --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash -c "$TR experiments/defect_fix/scripts/df.py bitcmp --a STBGG-old \
      --b STBGG-new --out stand_b/identity --note \"B's command (on1_JTX, JTX00, STAND W_S $WS, FREE-M 0.002 move 0.95) 60 steps eager: code $rev vs $h\" \
      && $TR $SB smoke --b STBGG-new --b1 STBGG-B1 --out $RES/smoke.json") || exit 1
  echo "$(date '+%F %T') smoke: old $a new $b B1 $c check $e" | tee -a $S/stages.txt; echo "$e" ;;
checkh)
  cmd F $TF 9500 500 > $S/cmd_F.txt; cmd FH $TH 9500 500 > $S/cmd_FH.txt; cmd B2 > $S/cmd_B2.txt
  { echo "== $TF -> $TH (full length; only the tag and the FREE-M dose may differ)"; diff <(tr ' ' '\n' < $S/cmd_F.txt) <(tr ' ' '\n' < $S/cmd_FH.txt)
    echo "== ${TAG[B]} -> ${TAG[B2]} (pilot B2; the FREE-M flags must match FH's)"; diff <(tr ' ' '\n' < $S/cmd_B.txt) <(tr ' ' '\n' < $S/cmd_B2.txt); } > $RES/cmd_diff_H.txt
  cat $RES/cmd_diff_H.txt
  n=$(diff <(tr ' ' '\n' < $S/cmd_F.txt) <(tr ' ' '\n' < $S/cmd_FH.txt) | grep -c '^>'); [[ $n == 2 ]] || die "FH: $n new tokens, want 2 (tag, dose)"
  [[ "$(diff <(tr ' ' '\n' < $S/cmd_F.txt) <(tr ' ' '\n' < $S/cmd_FH.txt) | grep '^>' | cut -c3- | paste -sd' ')" == "$TH 0.001" ]] || die "FH diff is not tag + 0.001"
  echo "diff ok" ;;
smokeh)
  [[ ! -e $DATA_DIR/runs/op_parity/runs/STBGG-H/ckpt-final.pt ]] || die "STBGG-H exists"
  FE=("${TRN[@]}" --env JEV_RM_EAGER=1 --timeout-h 1)
  c=$(sub stb-gg-h "${FE[@]}" -- bash -c "$(cmd FH STBGG-H 60 50)") || exit 1
  e=$(sub stb-smokecheck-h --after $c --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash -c "$TR $SB smoke --b STBGG-new --b1 STBGG-H --finite-only --out $RES/smoke_H.json") || exit 1
  echo "$(date '+%F %T') smokeh: run $c check $e" | tee -a $S/stages.txt; echo "$e" ;;
chainh)
  AF=${2:-}; [[ ! -e $DATA_DIR/runs/op_parity/runs/$TH/ckpt-final.pt ]] || die "$TH exists"
  PR=$((${PRIO:-12} + 1))
  t=$(sub stb-train-$TH ${AF:+--after $AF} "${TRN[@]}" --timeout-h 6 -- bash $ME train FH) || exit 1
  fr=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for c in $(seq 0 $((nch - 1))); do fr+=($(PRIO=$PR TRIES=2 sub stb-ar-$TH-c$c --after $t "${RD[@]}" -- bash $JT aread $TH off $c)) || exit 1; done
  for i in 0 1 2 3 4 5; do fr+=($(PRIO=$PR TRIES=2 sub stb-held-$TH-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $TH $i 6)) || exit 1; done
  q=$(pai $TH $S/pai281/$TH $t); [[ -n $q ]] || die "FH PAI"
  f=$(sub stb-fsum-H --after $(csv ${fr[@]}),$q --vram 0.1 --cpu 32 --ram 96 --timeout-h 4 -- bash $ME fsumh) || exit 1
  printf 'fsum_H %s\n' $f >> $S/summary_ids.txt
  echo "$(date '+%F %T') chainh: full train $t reads ${fr[*]} PAI $q fsum_H $f" | tee -a $S/stages.txt
  echo "fsum_H $f" ;;
chain)
  AF=${2:-}; for A in B1 B2 F; do [[ ! -e $DATA_DIR/runs/op_parity/runs/${TAG[$A]}/ckpt-final.pt ]] || die "${TAG[$A]} exists"; done
  PR=$((${PRIO:-12} + 1)); ids=(); ar=()
  for A in B1 B2; do T=${TAG[$A]}
    t=$(sub stb-train-$T ${AF:+--after $AF} "${TRN[@]}" --timeout-h 2 -- bash $ME train $A) || exit 1
    r=$(PRIO=$PR bash experiments/alpasim_proxy/scripts/wait_set.sh readc $T $t | tail -1); [[ -n $r ]] || die "readc $T"; echo "$r ws-collect-$T" >> $S/jobs.txt; ids+=($r)
    for i in 0 1 2 3 4 5; do ids+=($(PRIO=$PR TRIES=2 sub stb-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done
    for c in $SUB; do ar+=($(PRIO=$PR TRIES=2 sub stb-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    echo "$(date '+%F %T') chain arm $A $T: train $t" | tee -a $S/stages.txt; done
  p=$(sub stb-psum --after $(csv ${ids[@]} ${ar[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 3 -- bash $ME psum) || exit 1
  t=$(sub stb-train-$TF ${AF:+--after $AF} "${TRN[@]}" --timeout-h 6 -- bash $ME train F) || exit 1
  fr=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for c in $(seq 0 $((nch - 1))); do fr+=($(PRIO=$PR TRIES=2 sub stb-ar-$TF-c$c --after $t "${RD[@]}" -- bash $JT aread $TF off $c)) || exit 1; done
  for i in 0 1 2 3 4 5; do fr+=($(PRIO=$PR TRIES=2 sub stb-held-$TF-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $TF $i 6)) || exit 1; done
  q=$(pai $TF $S/pai281/$TF $t); [[ -n $q ]] || die "full PAI"
  f=$(sub stb-fsum --after $(csv ${fr[@]}),$q --vram 0.1 --cpu 32 --ram 96 --timeout-h 4 -- bash $ME fsum) || exit 1
  printf 'psum %s\nfsum %s\n' $p $f > $S/summary_ids.txt
  echo "$(date '+%F %T') chain: pilot reads ${ids[*]} ${ar[*]} psum $p; full train $t reads ${fr[*]} PAI $q fsum $f" | tee -a $S/stages.txt
  cat $S/summary_ids.txt ;;
train)   # train <arm>: refuses unless the smoke check and the ratchet coverage of the stricter check passed
  A=$2; python3 - $RES $A <<'PY' || die "pre-checks failed: no training"
import json, sys
r = sys.argv[1]
i, s, c = (json.load(open(f"{r}/{n}.json")) for n in ("identity", "smoke", "ratchet_clear"))
assert i["identical"], "switch-off identity failed"
assert s["pass_"], "B1 smoke failed"
assert c["pass_"] and c["clear"] == [15.0, 2.0], "ratchet coverage of the stricter check"
if sys.argv[2] == "FH":
    assert json.load(open(f"{r}/smoke_H.json"))["pass_"], "FH smoke failed"
PY
  if [[ $A == F ]]; then CMD=$(cmd F $TF 9500 500); elif [[ $A == FH ]]; then CMD=$(cmd FH $TH 9500 500); else CMD=$(cmd $A); fi
  echo "$(date '+%F %T') train ${TAG[$A]}: $CMD" | tee -a $S/stages.txt; exec $CMD ;;
psum)    # B1 / B2 against the control: extract, arena, drift, the supplement 17 rule per arm (+ the supplement 19 PCS line on the Mac), diag
  X="$CTL=$J/arena/$CTL@off ${TAG[B]}=$J/arena/${TAG[B]}@off ${TAG[B1]}=$J/arena/${TAG[B1]}@off ${TAG[B2]}=$J/arena/${TAG[B2]}@off"
  $AP $FM ext --read $X --out $S/ext || die "extract"
  runs=(); for t in $CTL:$CL ${TAG[B1]}:${LAB[B1]} ${TAG[B2]}:${LAB[B2]}; do
    d=$($TR -c "import sys; sys.path[:0] = ['experiments/defect_fix/scripts']; import df; print(','.join(sorted(df.read_dirs('${t%%:*}')['held'])))" | tail -1)
    [[ -n $d ]] || die "no held dirs ${t%%:*}"; runs+=("${t##*:}@18=$d"); done
  $AP experiments/route_follow/scripts/rf_follow.py --runs "${runs[@]}" --out $S/follow --cover none --workers 16 > $S/follow.log 2>&1 || die "rf_follow"
  for A in B1 B2; do T=${TAG[$A]}
    $V $FM arena --src $S/ext --a $T --b $CTL --out $S/arena_$A.json || die "arena $A"
    $AP $SP drift --follow $S/follow --a ${LAB[$A]}@18 --b $CL@18 --out $S/drift_$A.json || die "drift $A"
    cp $S/arena_$A.json $S/drift_$A.json $RES/
    $TR $SP pilot --tag $T --ctl $CTL --arena $S/arena_$A.json --drift $S/drift_$A.json --ext $S/ext --drift-guard --out $RES/pilot_$A.md | tee $S/pilot_$A.txt || die "summary $A"; done
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows.csv.gz', index=False)" || die "windows"
  $AP $SB diag --pair ${TAG[B1]}:$CTL --pair ${TAG[B2]}:$CTL --pair ${TAG[B]}:$CTL --out $S/diag_pilot --workers 32 > $S/diag_pilot.log 2>&1 || die "diag"
  cp $S/diag_pilot/diag.json $RES/diag_pilot.json; cp $S/diag_pilot/win.csv $RES/diag_pilot_windows.csv
  date '+%F %T' > $S/PILOT_DONE ;;
fsum)    # the full arm: arena 18 chunks native against JTN-L18-s1 (+ JTN-L18-s0, RSG), held 460, PAI 281 against NTV0 + ACC + GAP
  $AP $FM ext --read $TF=$J/arena/$TF@off $NREF=$J/arena/$NREF@off JTN-L18-s0=$J/arena/JTN-L18-s0@off \
    RSG-L18-s0=$DATA_DIR/runs/alpasim_proxy/arena/reads/RSG-L18-s0 --out $S/ext_full || die "extract full"
  $V $FM arena --src $S/ext_full --a $TF --b $NREF --ref JTN-L18-s0 RSG-L18-s0 --out $RES/arena_full.json || die "arena full"
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext_full/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows_full.csv.gz', index=False)" || die "windows"
  $V experiments/defect_fix/scripts/pai281.py --j $S/pai281/$TF --jl "$TF+ACC+GAP" --j2 $O/pai_accgap/JTN-L18-s0 --j2l "JTN-L18-s0+ACC+GAP" \
    --ref $O/pai_accgap/$PREF --rl "$PREF+ACC+GAP" --out $RES/pai281_accgap.md || die "pai281"
  $TR $SB full --tag $TF --refs $NREF JTN-L18-s0 RSG-L18-s0 $CTL --arena $RES/arena_full.json --out $RES/full.md || die "full summary"
  date '+%F %T' > $S/FULL_DONE ;;
fsumh)   # full-length B2: the fsum reads with an H suffix on every output
  $AP $FM ext --read $TH=$J/arena/$TH@off $NREF=$J/arena/$NREF@off JTN-L18-s0=$J/arena/JTN-L18-s0@off \
    RSG-L18-s0=$DATA_DIR/runs/alpasim_proxy/arena/reads/RSG-L18-s0 --out $S/ext_full_H || die "extract full H"
  $V $FM arena --src $S/ext_full_H --a $TH --b $NREF --ref JTN-L18-s0 RSG-L18-s0 --out $RES/arena_full_H.json || die "arena full H"
  $V -c "import pandas as pd; w = pd.read_parquet('$S/ext_full_H/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/arena_windows_full_H.csv.gz', index=False)" || die "windows H"
  $V experiments/defect_fix/scripts/pai281.py --j $S/pai281/$TH --jl "$TH+ACC+GAP" --j2 $O/pai_accgap/JTN-L18-s0 --j2l "JTN-L18-s0+ACC+GAP" \
    --ref $O/pai_accgap/$PREF --rl "$PREF+ACC+GAP" --out $RES/pai281_accgap_H.md || die "pai281 H"
  $TR $SB full --tag $TH --refs $NREF JTN-L18-s0 RSG-L18-s0 $CTL --arena $RES/arena_full_H.json --out $RES/full_H.md || die "full summary H"
  date '+%F %T' > $S/FULL_H_DONE ;;
*) sed -n 2,14p $0; exit 2 ;;
esac
