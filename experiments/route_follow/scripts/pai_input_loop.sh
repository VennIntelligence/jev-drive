#!/usr/bin/env bash
# PAI-INPUT-LOOP (plans/2026-10-19-pai-input-loop-prereg.md): NTV0-L18-s0 on PAI 281, the frame source (PAI_KEYWARP) with and without ACC + GAP,
# and the speed loop. Box, repo root; every GPU job goes through the pool (priority $PRIO, 12).
#   pai_input_loop.sh submit   N0 (8 chunks, at once), KA c000 -> gate -> KA c001..c007 + KN (8 chunks)
#   pai_input_loop.sh post     after the reads: rf_tax extract of the four arms, N0's driver messages, then cf (GPU) and sum
#   pai_input_loop.sh gate | extract <arm> | msgs | cf | sum      job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTV0-L18-s0; O=$DATA_DIR/runs/route_follow/pai_input_loop; ME=experiments/route_follow/scripts/pai_input_loop.sh; PRIO=${PRIO:-12}
A0=$DATA_DIR/runs/defect_fix/pai_accgap/$T; CH=$A0/chunks; RF=experiments/route_follow/scripts; R=experiments/route_follow/results/pai_input_loop
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; OP=$DATA_DIR/envs/op-train/bin/python
FIX="JEV_VCONT=1.0 JEV_LEAD=1"; ACC="$FIX JEV_ACC=1 JEV_GAP=0.7,2.7"; WIZ=controller.gains.rel_front_steering_angle_weight=10
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner pai-input-loop --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
ids() { awk '$2 ~ /^pai-/ {print $1}' "$1" | paste -sd,; }
read_arm() {  # arm serve keywarp [chunk files...] -> pai_box.sh submit (chunks copied once; a later call adds the rest and resubmits)
  local arm=$1 serve=$2 kw=$3; shift 3; mkdir -p $O/$arm/chunks; cp -n "$@" $O/$arm/chunks/
  CHUNKS_FROM=$O/$arm/chunks SERVE="$serve${kw:+ PAI_KEYWARP=1}" WIZ=$WIZ CONC=4 KEEP_ASL=1 PRIO=$PRIO AFTER=${AFTER:-} \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/$arm $T > $O/$arm/submit.$(date +%H%M%S%N).txt || die "pai submit $arm"; }
case ${1:-} in
submit)
  [[ -e $DATA_DIR/runs/op_parity/runs/$T/ckpt-final.pt && $(ls $CH/*.txt | wc -l) == 8 ]] || die "checkpoint or chunk lists missing"
  mkdir -p $O/pool; git rev-parse --short HEAD > $O/code.txt
  read_arm N0 "$FIX" "" $CH/*.txt
  read_arm KA "$ACC" 1 $CH/c000.txt; c0=$(ids $O/KA/submit.*.txt); [[ -n $c0 ]] || die "no KA c000 job"
  g=$(sub pil-gate --after $c0 --vram 0.1 --cpu 1 --ram 2 --timeout-h 0.2 -- bash $ME gate) || exit 1
  AFTER=$g read_arm KA "$ACC" 1 $CH/*.txt
  AFTER=$g read_arm KN "$FIX" 1 $CH/*.txt
  echo "$(date '+%F %T') submitted: N0 $(cat $O/N0/submit.*.txt | ids /dev/stdin), KA c000 $c0, gate $g, KA rest + KN after the gate" | tee -a $O/stages.txt ;;
gate)  # KA c000: every drive record carries keywarp, no inference error, >= 30 sessions scored
  python3 - $O/KA/runs/ntv0-l18-s0_c000 <<'PY' | tee $O/gate.txt
import json, sys
r = sys.argv[1]; D = [json.loads(l) for l in open(f"{r}/driver/drive.jsonl")]; D = [x for x in D if x["kind"] == "drive"]
S = json.load(open(f"{r}/sim/aggregate/results-summary.json"))["rollouts"]
rpc = sum("RpcError" in str(x.get("failure_reason") or "") for x in S); kw = sum(bool(x.get("keywarp")) for x in D if x.get("infer"))
inf = sum(bool(x.get("infer")) for x in D); sess = len({x["session"] for x in D})
ok = inf > 0 and kw == inf and rpc == 0 and len(S) >= 30
print(f"KA c000: {len(D)} drive records, {inf} inferences, {kw} with keywarp, {sess} sessions, {len(S)} rollouts, {rpc} RPC errors -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
  ;;
post)
  after=$(cat $O/{N0,KA,KN}/submit.*.txt | ids /dev/stdin); e=()
  for arm in A0 N0 KA KN; do e+=($(sub pil-ex-$arm --after $after --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME extract $arm)) || exit 1; done
  m=$(sub pil-msgs --after $after --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME msgs)
  c=$(sub pil-cf --after ${e[1]},$m --vram 12 --cpu 8 --ram 60 --timeout-h 4 -- bash $ME cf)
  s=$(sub pil-sum --after $(IFS=,; echo "${e[*]}"),$c --vram 0.1 --cpu 8 --ram 40 --timeout-h 1 -- bash $ME sum)
  echo "$(date '+%F %T') post: extract ${e[*]} msgs $m cf $c sum $s" | tee -a $O/stages.txt ;;
extract)
  arm=$1; root=$O/$arm; [[ $arm == A0 ]] && root=$A0; mkdir -p $O/tax
  $AP $RF/rf_tax.py extract --runs $(ls -d $root/runs/*/sim) --out $O/tax/$arm.pkl --workers 16 > $O/tax/$arm.log 2>&1 || die "extract $arm" ;;
msgs)  # N0's driver-side messages per scene (front wide camera only), then the rollout.asl files are no longer needed
  mkdir -p $O/msgs; cd experiments/alpasim/scripts
  PYTHONPATH=$DATA_DIR/third_party/alpasim/src/utils $AP - $O <<'PY' || exit 1
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import col1_pai_extract as E
O = sys.argv[1]; J = []
for sim in sorted(glob.glob(f"{O}/N0/runs/*/sim")):
    for r in json.load(open(f"{sim}/aggregate/results-summary.json"))["rollouts"]:
        if os.path.isdir(f"{sim}/rollouts/{r['clipgt_id']}"):
            J.append((sim, r["clipgt_id"], f"{O}/msgs", "camera_front_wide_120fov"))
with ProcessPoolExecutor(16) as ex:
    print(len(list(ex.map(E.one_msgs, J))), "scenes ->", f"{O}/msgs", flush=True)
PY
  ;;
cf)
  $OP $RF/pil_cf.py --msgs $O/msgs --tax $O/tax/N0.pkl --drive "$O/N0/runs/*/driver/drive.jsonl" --tag $T --out $O/cf.pkl || die "cf" ;;
sum)
  mkdir -p $R
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $O/KA --jl "KW+ACC+GAP" --ref $A0 --rl "ACC+GAP" --out $R/pai281_KA_vs_A0.md || die "pai281 KA"
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $O/KN --jl "KW (VCONT+LEAD)" --ref $O/N0 --rl "VCONT+LEAD" --out $R/pai281_KN_vs_N0.md || die "pai281 KN"
  .venv/bin/python experiments/defect_fix/scripts/pai281.py --j $A0 --jl "ACC+GAP" --ref $O/N0 --rl "VCONT+LEAD" --out $R/pai281_A0_vs_N0.md || die "pai281 A0"
  .venv/bin/python $RF/pai_input_loop.py --o $O --a0 $A0 --out $R || die "read-out"
  date '+%F %T' > $R/DONE; ls -l $R ;;
*) sed -n 2,6p $0; exit 2 ;;
esac
