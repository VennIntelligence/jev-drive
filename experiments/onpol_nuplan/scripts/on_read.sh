#!/usr/bin/env bash
# Lane ONPOL-NUPLAN (plans/2026-10-11-onpol-prereg.md, "Pilot" and "Read"): closed-loop runs of the lane's checkpoints through on_read_loop.py
# (= ot2_loop.py + the lane's lists, a split guard and the simulator-state extraction <run dir>/read_sim.pkl). Unchanged sh30 driver path
# (run.sh sh30, SH30_TAG=<tag>, compiled passes, no serving switch, no dump), one pool job per checkpoint, each waiting for its
# $DATA_DIR/runs/op_parity/runs/<tag>/ckpt-final.pt (OT2_WAIT_H, default 12 h).
#   scripts/tmux_run.sh on-read-<stage> bash experiments/onpol_nuplan/scripts/on_read.sh <stage> [<tag> ...]
#   smoke <tag> ...   the 48-scene pilot smoke: each checkpoint on results/split/smoke48_train.txt (48 TRAIN scenes), then the checklist
#                     results/pilot/smoke_<tag>.json (48 / 48 rollouts, 0 non-finite plans, 0 driver errors, 10 drive calls per session).
#                     Its score is not a read: nothing here prints or stores it. Exit 2 if a checklist fails.
#                     The json lands in the box's working tree (and in read/smoke/): copy it to the Mac, commit it there and delete the box
#                     copy before the next `git pull` on the box (an untracked file of the same name blocks the pull).
#   held [<tag> ...]  THE READ: each checkpoint on results/split/held.txt (460 scenes), one pool job per checkpoint. Default tags = the arms
#                     SIMIT-ON / SSELF-ON / SBOTH-ON s0 s1, the fourth family (SSELFnoL-ON if its training exists or ON_FOURTH says so, else
#                     SNONE-ON) and the control P2H10S-F s0-s3. Refused: a tag given twice, a tag outside those families (ON_HELD_ANY=1
#                     lifts this), split files that differ from the committed ones; the loop refuses every list but held.txt.
#                     Round 2 (plans/2026-10-12-onpol2-prereg.md, B and C) is read by naming its tags: YSELFnoL-ON / YNONE-ON s0 s1 and their
#                     control YR10m10-F s0 s1; SSELFnoL-ON s2 s3 (the default list is round 1's).
#                     ON_HELD_GATE=all makes every job wait for all checkpoints of the list (no held rollout before the last training ends).
# OT2_MAX_ACTIVE (default 4) stacks at a time; ON_KEEP_ASL=all keeps every rollout.asl (default: ot2_loop's pruning, failed rollouts only).
# State: $DATA_DIR/runs/onpol_nuplan/read/<stage>/{STATUS, DONE, ERROR, log.txt, manifest.json {label: [run dirs]}, runs/}; the chain's own
# STATUS / DONE / ERROR / log.txt in read/chain-<stage>/. A rerun skips the labels already in the manifest (label = <tag> for held,
# <tag>@smoke48 for smoke), so a finished checkpoint is never rolled out on held twice; one chain per stage at a time (lock).
# Estimate for held, 12 checkpoints x 460 scenes: 3.0-3.5 s per scene and stack next to six other stacks (lane nuplan1491 / JOINT-1 / this lane's
# rollouts, 2026-10-11; 4.7 at worst) -> 24-37 min per job, booked 29 of 80 GB; 5-7 stack-hours = 1.8-2.7 card-hours; 1.3-1.9 h wall at 4 stacks
# once the checkpoints exist; 69 GiB RAM declared per job (42-45 measured); rollout logs 12 GB per job before pruning.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:?stage: smoke | held}; shift
E=experiments/onpol_nuplan; SP=$E/results/split; W=$DATA_DIR/runs/onpol_nuplan/read; O=$W/chain-$STAGE; CK=$DATA_DIR/runs/op_parity/runs
mkdir -p "$O"; exec 9> "$O/lock"; flock -n 9 || { echo "another on_read.sh $STAGE holds $O/lock" >&2; exit 1; }
rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') on-read $STAGE: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD); free $(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -d ' ')"
git diff --quiet HEAD -- "$SP" || die "$SP differs from the committed split"
case $STAGE in
  smoke) (( $# )) || die "smoke needs at least one tag"; SPLIT=train LIST=smoke48_train SFX=@smoke48 ;;
  held)
    if (( ! $# )); then
      F=${ON_FOURTH:-$([[ -d $CK/SSELFnoL-ON-s0 || -d $CK/SSELFnoL-ON-s1 ]] && echo SSELFnoL || echo SNONE)}
      set -- {SIMIT,SSELF,SBOTH,$F}-ON-s{0,1} P2H10S-F-s{0,1,2,3}
    fi
    for t; do [[ -n ${ON_HELD_ANY:-} || $t =~ ^(SIMIT|SSELF|SBOTH|SNONE|SSELFnoL)-ON-s[01]$ || $t =~ ^P2H10S-F-s[0-3]$ || $t =~ ^(YSELFnoL|YNONE)-ON-s[01]$ || $t =~ ^YR10m10-F-s[01]$ || $t =~ ^SSELFnoL-ON-s[23]$ ]] || die "tag $t is not an arm or the control of the prereg"; done
    [[ $(wc -l < "$SP/held.txt") -eq 460 ]] || die "held.txt is not the 460 scenes"
    SPLIT=held LIST=held SFX= ;;
  *) die "unknown stage $STAGE" ;;
esac
dup=$(printf '%s\n' "$@" | sort | uniq -d | tr '\n' ' '); [[ -z $dup ]] || die "tags given twice: $dup"
ALL=$(IFS=+; echo "$*")
J=(); for t; do J+=("$t$SFX:sh30:SH30_TAG=$t:$([[ $STAGE == held && ${ON_HELD_GATE:-own} == all ]] && echo "$ALL" || echo "$t"):$LIST"); done
status "$# checkpoints on $LIST: $*"
ON_SPLIT=$SPLIT ON_LISTS=$W/lists OT_LANE=onpol OT_OUT=$W OT_OWNER=alpasim-onpol OT_PRIO=${OT_PRIO:-12} OT2_MAX_ACTIVE=${OT2_MAX_ACTIVE:-4} \
  python3 $E/scripts/on_read_loop.py "$STAGE" "${J[@]}" || die "on_read_loop rc $? (see $W/$STAGE/ERROR)"
if [[ $STAGE == smoke ]]; then
  bad=; for t; do python3 $E/scripts/on_read_loop.py --check "$t" "$W/smoke/manifest.json" "$t@smoke48" "$E/results/pilot/smoke_$t.json" > "$O/check_$t.log" 2>&1 || bad+=" $t"
              cp "$E/results/pilot/smoke_$t.json" "$W/smoke/"; done
  [[ -z $bad ]] || { status "ERROR smoke checklist failed:$bad (results/pilot/smoke_<tag>.json)"; echo "checklist:$bad" > "$O/ERROR"; exit 2; }
fi
status "all done: $*"; date '+%F %T' > "$O/DONE"
