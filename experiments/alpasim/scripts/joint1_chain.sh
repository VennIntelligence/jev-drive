#!/usr/bin/env bash
# Lane JOINT, phase 1 (plans/2026-10-11-joint1-prereg.md): closed-loop runs on the 1491 public nuPlan scenes through ot2_loop.py (OT_LANE=joint1),
# the box's default driver path (compiled model passes, SH30_COMPILE=1), the c0b / CF1 scene lists (chunk0-2 = the 700, cf1fresh = 391, ot3new = 400).
# One self-advancing chain per stage:   scripts/tmux_run.sh joint1-<stage> bash experiments/alpasim/scripts/joint1_chain.sh <stage>
#   ref    references whose stored reads ran on the uncompiled driver (OT3, 2026-10-09 before 18:22 CST): P2H10-F s0 / s1 on ot3new,
#          APY10m10-AB s0 / s1 on the 700; reproducibility checks of the stored compiled reads: P2H10-F-s0 on chunk0 (against TR1's run) and on
#          cf1fresh (against CF1's), APY10m10-AB-s0 on cf1fresh (against CF1's)
#   refx   only if a check differs: the rest of the stored compiled reference reads, rerun (P2H10-F and P2H10S-F s0 / s1 on the 700, P2H10-F s1 on
#          cf1fresh, APY10m10-AB s1 on cf1fresh and s0 / s1 on ot3new)
#   arms   the trained arms (JAPS, JAPSnoY, JAPSnoA; seeds 0 / 1) on all 1491 scenes; each job waits for its checkpoint (ot2_loop OT2_WAIT_H)
#   smoke  <tag> <driver>: the 48-scene pilot smoke of one pilot checkpoint (list smoke48 = runs/alpasim/scenes_navtest_full_part001_48.txt)
# State: $DATA_DIR/runs/alpasim/joint1/<stage>/{STATUS, DONE, ERROR, log.txt, manifest.json, runs/}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=$1; shift
RA=$DATA_DIR/runs/alpasim; SN=$STAGE${1:+-$1}; O=$RA/joint1/chain-$SN; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') joint1 $STAGE: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD)"
J=()
sh() { J+=("$1:sh30:SH30_TAG=$2:$2${3:+:$3}"); }                     # label tag [list]
a2() { J+=("$1:ap2:AP2_TAG=$2:$2${3:+:$3}"); }
case $STAGE in
  ref)
    for s in 0 1; do sh "P2H10-F-s$s@new" "P2H10-F-s$s" ot3new; a2 "APY10m10-AB-s$s" "APY10m10-AB-s$s"; done
    sh "P2H10-F-s0@chk0" P2H10-F-s0 chunk0; sh "P2H10-F-s0@fresh" P2H10-F-s0 cf1fresh; a2 "APY10m10-AB-s0@fresh" APY10m10-AB-s0 cf1fresh ;;
  refx)
    for s in 0 1; do sh "P2H10-F-s$s" "P2H10-F-s$s"; sh "P2H10S-F-s$s" "P2H10S-F-s$s"; a2 "APY10m10-AB-s$s@new" "APY10m10-AB-s$s" ot3new; done
    sh "P2H10-F-s1@fresh" P2H10-F-s1 cf1fresh; a2 "APY10m10-AB-s1@fresh" APY10m10-AB-s1 cf1fresh ;;
  arms)
    for t in JAPS JAPSnoY; do for s in 0 1; do a2 "$t-s$s" "$t-s$s"; a2 "$t-s$s@fresh" "$t-s$s" cf1fresh; a2 "$t-s$s@new" "$t-s$s" ot3new; done; done
    for s in 0 1; do sh "JAPSnoA-s$s" "JAPSnoA-s$s"; sh "JAPSnoA-s$s@fresh" "JAPSnoA-s$s" cf1fresh; sh "JAPSnoA-s$s@new" "JAPSnoA-s$s" ot3new; done ;;
  smoke)
    T=$1 DRV=$2; [[ -f $RA/c0b/lists/smoke48.txt ]] || cp "$RA/scenes_navtest_full_part001_48.txt" "$RA/c0b/lists/smoke48.txt"
    if [[ $DRV == ap2 ]]; then a2 "$T@smoke48" "$T" smoke48; else sh "$T@smoke48" "$T" smoke48; fi ;;
  *) die "unknown stage $STAGE" ;;
esac
OT_LANE=joint1 OT_PRIO=${OT_PRIO:-13} OT2_MAX_ACTIVE=${OT2_MAX_ACTIVE:-4} OT_OUT=$RA/joint1 python3 experiments/alpasim/scripts/ot2_loop.py "$SN" "${J[@]}" \
  || die "ot2_loop rc $? (see $RA/joint1/$SN/ERROR)"
status "all done"; date '+%F %T' > "$O/DONE"
