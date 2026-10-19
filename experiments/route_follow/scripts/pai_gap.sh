#!/usr/bin/env bash
# PAI-GAP (plans/2026-10-19-pai-gap-prereg.md): where NTV0-L18-s0 + ACC + GAP loses points on PAI 281. Box, repo root; GPU jobs only via the pool.
#   pai_gap.sh submit     H (log-replay read, onp_log_driver.py, A0's serve and chunk lists): c000 -> gate -> c001..c007; then the CPU jobs
#                         (rf_tax extract of H, rf_tax --asl-actors extract of A0, read-out) chained with --after
#   pai_gap.sh wl         (amendment 1) map wait lines of A0's scenes, a CPU pool job (rf_tax waitlines)
#   pai_gap.sh gate | exh | exasl | waitl | sum      job bodies
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
T=NTV0-L18-s0; O=$DATA_DIR/runs/route_follow/pai_gap; ME=experiments/route_follow/scripts/pai_gap.sh; PRIO=${PRIO:-12}
A0=$DATA_DIR/runs/defect_fix/pai_accgap/$T; RF=experiments/route_follow/scripts; R=experiments/route_follow/results/pai_gap
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 DRV_PY=../../onpol_pai/lib/onp_log_driver.py"
WIZ="controller.gains.rel_front_steering_angle_weight=10 runtime.simulation_config.send_recording_ground_truth=true"
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner pai-gap --name "$name" --tries 1 --priority $PRIO \
     --no-check --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
ids() { awk '$2 ~ /^pai-/ {print $1}' "$@" | paste -sd,; }
read_h() {  # chunk files... -> pai_box.sh submit (copied once; a later call adds the rest)
  mkdir -p $O/H/chunks; cp -n "$@" $O/H/chunks/
  CHUNKS_FROM=$O/H/chunks SERVE="$SERVE" WIZ="$WIZ" CONC=4 PRIO=$PRIO AFTER=${AFTER:-} \
    bash experiments/alpasim/scripts/pai_box.sh submit $O/H $T > $O/H/submit.$(date +%H%M%S%N).txt || die "pai submit H"; }
case ${1:-} in
submit)
  [[ $(ls $A0/chunks/*.txt | wc -l) == 8 ]] || die "A0 chunk lists missing"
  mkdir -p $O/pool; git rev-parse --short HEAD > $O/code.txt
  read_h $A0/chunks/c000.txt; c0=$(ids $O/H/submit.*.txt); [[ -n $c0 ]] || die "no H c000 job"
  g=$(sub pg-gate --after $c0 --vram 0.1 --cpu 1 --ram 2 --timeout-h 0.2 -- bash $ME gate) || exit 1
  AFTER=$g read_h $A0/chunks/*.txt; all=$(ids $O/H/submit.*.txt)
  x=$(sub pg-exasl --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME exasl) || exit 1
  h=$(sub pg-exh --after $all --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME exh) || exit 1
  s=$(sub pg-sum --after $h,$x --vram 0.1 --cpu 16 --ram 60 --timeout-h 1 -- bash $ME sum) || exit 1
  echo "$(date '+%F %T') submitted: H c000 $c0, gate $g, H all $all, exasl $x, exh $h, sum $s" | tee -a $O/stages.txt ;;
gate)  # H c000: log replay served after the force-GT window, no RPC error, >= 30 rollouts
  python3 - $O/H/runs/ntv0-l18-s0_c000 <<'PY' | tee $O/gate.txt
import json, sys
r = sys.argv[1]; D = [json.loads(l) for l in open(f"{r}/driver/drive.jsonl")]; D = [x for x in D if x["kind"] == "drive"]
S = json.load(open(f"{r}/sim/aggregate/results-summary.json"))["rollouts"]
rpc = sum("RpcError" in str(x.get("failure_reason") or "") for x in S); late = [x for x in D if x.get("k", 0) >= 20]
lg = sum(x.get("ret") == "log" and x.get("log_poses") is not None for x in late); inf = sum(bool(x.get("infer")) for x in late)
ok = len(late) > 0 and lg == len(late) and inf > 0 and rpc == 0 and len(S) >= 30
print(f"H c000: {len(D)} drive records, {len(late)} at k >= 20, {lg} returning the log with log_poses, {inf} inferences, {len(S)} rollouts, {rpc} RPC errors -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
  ;;
exh)
  mkdir -p $O/tax; $AP $RF/rf_tax.py extract --runs $(ls -d $O/H/runs/*/sim) --out $O/tax/H.pkl --workers 16 > $O/tax/H.log 2>&1 || die "extract H" ;;
exasl)  # A0 rollouts that kept a rollout.asl: their actor poses, for the logged-traffic gate
  mkdir -p $O/tax; $AP $RF/rf_tax.py extract --runs $(ls -d $A0/runs/*/sim) --out $O/tax/A0asl.pkl --workers 16 --asl-actors > $O/tax/A0asl.log 2>&1 || die "extract A0 asl" ;;
wl)
  sub pg-waitl --vram 0.1 --cpu 16 --ram 40 --timeout-h 1 -- bash $ME waitl ;;
waitl)
  mkdir -p $O/tax; $AP $RF/rf_tax.py waitlines --runs $(ls -d $A0/runs/*/sim) --out $O/tax/wl.pkl --workers 16 > $O/tax/wl.log 2>&1 || die "waitlines" ;;
sum)
  mkdir -p $R; .venv/bin/python $RF/pai_gap.py all --o $O --out $R > $O/sum.log 2>&1 || die "pai_gap.py"; date '+%F %T' > $O/DONE ;;
*) sed -n 2,5p $0; exit 2 ;;
esac
