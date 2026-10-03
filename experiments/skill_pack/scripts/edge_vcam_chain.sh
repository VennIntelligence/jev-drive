#!/usr/bin/env bash
# Lane EDGE-FIX: the virtual camera rig scored (plans/2026-10-04-roadedge-diagnosis-plan.md, addenda 3-4). Resumes from disk.
# Arm tags: <cm>[p<pitch>] = virtual height in cm, pitch in deg (m = minus, _ = decimal point), e.g. 140pm0_3; 187 = true height.
# navtrain rollouts vh187 (true height, control) + vh122 / vh130 / vh140 -> official navtrain PDMS -> H*; test-board control
# rollouts meanwhile; TRK path alpha refit on navtrain at H*; test rollouts at H* (native, it_dw3, it_dw3 rot0); selector 0.6;
# TRK pose files; official navtest PDMS + navhard two-stage EPDMS of every arm; paired readouts (edge_vcam_report.py).
# STATUS / DONE / ERROR in $DATA_DIR/runs/skill_pack/edge_fix.
#   GPU=1 scripts/tmux_run.sh edgefix experiments/skill_pack/scripts/edge_vcam_chain.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd); cd "$repo"
R=$DATA_DIR/runs/skill_pack/edge_fix; mkdir -p "$R"; rm -f "$R/ERROR" "$R/DONE"
GPU=${GPU:-1}; PROCS=${PROCS:-5}; VW=${VW:-4}; ST=${ST:-12}; HS=(122 130 140); ALPHAS=(0.25 0.5 0.75 1)
E=$DATA_DIR/envs; P=$DATA_DIR/runs/op_lb; O=Oit_dw3-s0; OX=$DATA_DIR/runs/op_adapt_H/onnx/it_dw3-s0.onnx
JV=$E/jevdrive/bin/python; NV=$E/navsim2/bin/python; OPY=$E/openpilot/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPI_ROOT=op_lb
st() { echo "$(date '+%F %T') $*" | tee -a "$R/STATUS"; }
die() { st "ERROR: $*"; echo "$*" > "$R/ERROR"; exit 1; }
[[ -f $OX ]] || die "missing $OX"
diskok() { local f; f=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc 0-9); (( f >= 150 )) || die "data disk below 150 GB ($f GB)"; }
hgt() { local c=${1%%p*}; [[ $c == 187 ]] && echo 0 || echo "${c:0:1}.${c:1}"; }
pit() { [[ $1 == *p* ]] && echo "${1#*p}" | tr m_ -. || echo 0; }
stem() { echo "vh$1-cinque${2:+_$O}$([[ ${3:-none} == none ]] || echo "_al-$3")"; }     # pred stem: stem <cm> [it] [align]

roll() {   # roll <data> <cm> [it] [align]: op_lb rollout at the virtual height + export to rear-axle poses
  local data=$1 cm=$2 it=${3:-} al=${4:-none} s
  s=$(stem "$cm" "$it" "$al")
  diskok
  if [[ ! -f $P/$data/plans/${s/-/@}.npz ]]; then
    st "roll $data $s"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 $OPY scripts/op_lb.py run --data "$data" --frames "vh$cm" --model cinque --vcam "$(hgt "$cm")" --vpitch="$(pit "$cm")" \
      --vcam-workers "$VW" --procs "$PROCS" --align "$al" ${it:+--onnx "$OX" --tag "$O"} >> "$R/run_$data.log" 2>&1 || die "roll $data $s"
  fi
  [[ -f $P/$data/preds/${s}__base.npz ]] || $JV experiments/op_openloop/lib/op_interp.py nav-export --data "$data" --plans "${s/-/@}" >> "$R/export.log" 2>&1 || die "export $data $s"
}
scored() { ls "$DATA_DIR"/runs/navsim/eval/$1/*/*.csv > /dev/null 2>&1; }
score() {  # score <v1|v2> <split> <data> <stem>; navtrain = the 3 000-token subset cache
  local n=opi_${3}_${4}__base
  scored "${1}_${2}_$n" && return 0
  ( [[ $2 == navtrain ]] && export CACHE_NAME=v1_navtrain_oplb TOKENS_FILE=$P/lb_navtrain/tokens.txt
    NAVSIM_THREADS=$ST experiments/zeroshot_openloop/archive/navsim_zs_score.sh score "$1" "$2" "$n" "$P/$3/preds/${4}__base.npz" > "$R/score_${1}_${2}_$4.log" 2>&1 )
  scored "${1}_${2}_$n" || { st "scoring failed: $1 $2 $4"; return 1; }
  st "scored $1 $2 $4: $(grep -a 'Final' "$R/score_${1}_${2}_$4.log" | tail -1 | awk '{print $NF}')"
}
pwait() { local p; for p in "$@"; do wait "$p" || die "background job $p failed"; done; }

# 1. navtrain height sweep (rollouts on the GPU, scoring in the background as each one lands)
pids=()
for cm in 187 "${HS[@]}"; do roll lb_navtrain "$cm"; score v1 navtrain lb_navtrain "$(stem "$cm")" & pids+=($!); done
# 2. test-board controls on the GPU meanwhile (do not depend on H*)
for data in lb_navtest lb_navhard; do roll $data 187; roll $data 187 it; roll $data 187 it rot0; done
pwait "${pids[@]}"
$NV experiments/skill_pack/scripts/edge_vcam_report.py pick --kind height --base "$(stem 187)" \
  --cands $(for cm in "${HS[@]}"; do echo "$cm=$(stem "$cm")"; done) > "$R/pick_height.log" 2>&1 || die "height pick"
H=$(grep '^PICK' "$R/pick_height.log" | awk '{print $2}'); st "H* = vh$H"; echo "$H" > "$R/H"
# 2b. horizon switch at H* (addendum 4): pitch sweep on navtrain, P* = best; best rig H = H*+P* if its delta > 0
PS=(pm1 pm0_3 p0_3 p1); pids=()
for q in "${PS[@]}"; do roll lb_navtrain "$H$q"; score v1 navtrain lb_navtrain "$(stem "$H$q")" & pids+=($!); done
pwait "${pids[@]}"
$NV experiments/skill_pack/scripts/edge_vcam_report.py pick --kind pitch --base "$(stem "$H")" \
  --cands $(for q in "${PS[@]}"; do echo "$H$q=$(stem "$H$q")"; done) > "$R/pick_pitch.log" 2>&1 || die "pitch pick"
HH=$(grep '^PICK' "$R/pick_pitch.log" | awk '{print $2}'); st "height+horizon = vh$HH"
HO=$H; $NV -c "import json; d = json.load(open('experiments/skill_pack/results/roadedge/vcam/navtrain_pitch.json')); exit(0 if d['$HH']['delta'] > 0 else 1)" && H=$HH
st "best rig = vh$H"; echo "$H" > "$R/BEST"

# 3. TRK path alpha on navtrain at H* (CPU, background) || test rollouts at H* (GPU)
T=$R/trk
(
  [[ -f $T/diag_lb_navtrain_path.pkl ]] || $NV experiments/skill_pack/scripts/trk_precomp.py make --data lb_navtrain --modes path --alphas "${ALPHAS[@]}" \
    --src "vh$H=$P/lb_navtrain/preds/$(stem "$H")__base.npz" --out "$T" --procs 40 >> "$R/trk_make.log" 2>&1 || exit 1
  for al in "${ALPHAS[@]}"; do
    ln -sf "$T/lb_navtrain/vh${H}_path_a$al.npz" "$P/lb_navtrain/preds/$(stem "$H")_trk-path-a${al}__base.npz"
    score v1 navtrain lb_navtrain "$(stem "$H")_trk-path-a$al" &
  done; wait
) & tp=$!
for data in lb_navtest lb_navhard; do roll $data "$HO"; roll $data "$HH"; roll $data "$H" it; roll $data "$H" it rot0; done
pwait $tp
$NV experiments/skill_pack/scripts/edge_vcam_report.py pick --kind trk --base "$(stem "$H")" \
  --cands $(for al in "${ALPHAS[@]}"; do echo "$al=$(stem "$H")_trk-path-a$al"; done) > "$R/pick_trk.log" 2>&1 || die "trk pick"
A=$(grep '^PICK' "$R/pick_trk.log" | awk '{print $2}'); st "TRK alpha* = $A"; echo "$A" > "$R/A"

# 4. selector (ratio 0.6 fixed) and TRK pose files on the test boards
for cm in 187 "$H"; do
  $OPY experiments/skill_pack/scripts/hist_align_select.py --frames "vh$cm" --rule rot0 --tag "$O" --ratio 0.6 --data lb_navhard lb_navtest >> "$R/select.log" 2>&1 || die "select $cm"
done
for data in lb_navtest lb_navhard; do
  [[ -f $T/diag_${data}_path.pkl ]] || $NV experiments/skill_pack/scripts/trk_precomp.py make --data "$data" --modes path --alphas "$A" --out "$T" --procs 48 \
    --src "vh$H=$P/$data/preds/$(stem "$H")__base.npz" "vh187=$P/$data/preds/$(stem 187)__base.npz" >> "$R/trk_make.log" 2>&1 || die "trk make $data"
  for cm in 187 "$H"; do ln -sf "$T/$data/vh${cm}_path_a$A.npz" "$P/$data/preds/$(stem "$cm")_trk-path-a${A}__base.npz"; done
done

# 5. official scoring, 3 jobs at a time
SEL=_al-sel-rot0-r0.6
ARMS=(A0=$(stem 187) A1=$(stem "$HO") A1h=$(stem "$HH") AB=$(stem "$H") A2=$(stem "$H")_trk-path-a$A A0T=$(stem 187)_trk-path-a$A B0=$(stem 187 it) B1=$(stem "$H" it)
      B0s=$(stem 187 it)$SEL B1s=$(stem "$H" it)$SEL G=gimm-cinque GBs=gimm-cinque_$O$SEL)
jobs_=(); for kv in "${ARMS[@]}"; do s=${kv#*=}; jobs_+=("v1 navtest lb_navtest $s" "v2 navhard_two_stage lb_navhard $s"); done
for j in "${jobs_[@]}"; do
  while (( $(jobs -rp | wc -l) >= 3 )); do sleep 20; done
  score $j &
done
wait
for j in "${jobs_[@]}"; do set -- $j; scored "${1}_${2}_opi_${3}_${4}__base" || die "official score missing: $j"; done

# 6. readouts
$NV experiments/skill_pack/scripts/edge_vcam_report.py test --procs 40 --arms "${ARMS[@]}" \
  --contrasts height=A0:A1 horizon=A1:A1h rig=A0:AB trk=AB:A2 trk_on_control=A0:A0T fix_it=B0:B1 fix_it_sel=B0s:B1s \
              sel_at_rig=B1:B1s it_at_rig=AB:B1 all=A0:B1s pipeline=G:A0 vs_shipped_native=G:AB vs_shipped_best=GBs:B1s \
  --did fix_x_it=A0:AB:B0:B1 fix_x_itsel=A0:AB:B0s:B1s trk_x_fix=A0:A0T:AB:A2 > "$R/report_test.log" 2>&1 || die "test readout"
$NV experiments/skill_pack/scripts/edge_vcam_report.py scale --plans gimm@cinque vh187@cinque "vh$HO@cinque" "vh$HH@cinque" "vh187@cinque_$O" "vh$H@cinque_$O" \
  > "$R/report_scale.log" 2>&1 || die "scale readout"
st "done"; touch "$R/DONE"
