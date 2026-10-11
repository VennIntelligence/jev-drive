#!/usr/bin/env bash
# Lane ONPOL-NUPLAN (plans/2026-10-11-onpol-prereg.md, Data): on-policy rollouts of P2H10S-F-s0 / -s1 on scenes of `alpasim_nuplan/onpol-train`
# with a dump of every decision's model inputs. The unchanged sh30 driver path (run.sh sh30, compiled passes, no serving switch) with its own
# SH30_DUMP on for every session: <run dir>/driver-logs/dump/s<session>_k<decision>.npz = the 8 slot frames as fed to the encoder (cur, uint8
# (8, 2, 6, 128, 256)), valid, the ego vector, mu, the served poses, the camera position, the scene, the CAM_F0 JPEG of t0; drive.jsonl beside
# it has command, history poses, t0, the anchor pose and the decision index. on_loop.py (= ot2_loop.py + the lane's lists, a train-split guard
# and the simulator-state extraction <run dir>/onpol_sim.pkl before the rollout logs are pruned) runs the pool jobs.
#   scripts/tmux_run.sh onpol-roll-<stage> bash experiments/onpol_nuplan/scripts/on_roll.sh <stage>
#   smoke1   1 train scene, seed 0            (stage 1 of the staged launch)
#   smoke10  10 train scenes, seed 0          (stage 10)
#   smoke10e the same 10 scenes with SH30_COMPILE=0 (interpreted model passes): the control of gate R1, where the trainer's eager forward must
#            reproduce the served plan without the compiled passes' rounding. Never a source of rows.
#   full     the 1031 train scenes in 4 interleaved chunks x seeds 0 / 1 = 8 pool jobs, OT2_MAX_ACTIVE (default 3) at a time
# State: $DATA_DIR/runs/onpol_nuplan/roll/<stage>/{STATUS, DONE, ERROR, log.txt, manifest.json {label: [run dirs]}, runs/}; the chain's own
# STATUS / DONE / ERROR / log.txt in roll/chain-<stage>/. Rerunning a stage skips the labels already in its manifest.
# Measured on the smokes (2026-10-11): results/rows/smoke10/ (dump bytes per decision, wall); full estimate in the lane README.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=$1
W=$DATA_DIR/runs/onpol_nuplan/roll; O=$W/chain-$STAGE; mkdir -p "$O"; rm -f "$O/DONE" "$O/ERROR"
exec > >(tee -a "$O/log.txt") 2>&1
status() { echo "$(date '+%F %T') onpol-roll $STAGE: $*" | tee "$O/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$O/ERROR"; exit 1; }
status "code $(git rev-parse --short HEAD); free $(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -d ' ')"
J=()
job() { J+=("P2H10S-F-s$1@$2:sh30:SH30_TAG=P2H10S-F-s$1,SH30_DUMP=1000000${3:+,$3}:P2H10S-F-s$1:$2"); }      # seed list [ENV=VALUE]
case $STAGE in
  smoke1)  job 0 on_smoke1 ;;
  smoke10) job 0 on_smoke10 ;;
  smoke10e) job 0 on_smoke10 SH30_COMPILE=0 ;;
  full)    for s in 0 1; do for c in 0 1 2 3; do job $s on_c$c; done; done ;;
  *) die "unknown stage $STAGE" ;;
esac
[[ $STAGE == full ]] && cap=${OT2_MAX_ACTIVE:-3} || cap=1
ON_LISTS=$W/lists OT_LANE=onpol OT_OUT=$W OT_OWNER=alpasim-onpol OT_PRIO=${OT_PRIO:-12} OT2_MAX_ACTIVE=$cap \
  python3 experiments/onpol_nuplan/scripts/on_loop.py "$STAGE" "${J[@]}" || die "on_loop rc $? (see $W/$STAGE/ERROR)"
n=$(find "$W/$STAGE/runs" -path '*driver-logs/dump/*.npz' | wc -l); b=$(find "$W/$STAGE/runs" -path '*driver-logs/dump/*.npz' -printf '%s\n' | awk '{s += $1} END {print s + 0}')
status "all done: $n dumped decisions, $b bytes ($((b / (n > 0 ? n : 1))) per decision), run dirs $(du -sh "$W/$STAGE/runs" | cut -f1)"
date '+%F %T' > "$O/DONE"
