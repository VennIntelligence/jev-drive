#!/usr/bin/env bash
# ONPOL-PAI pass 1 (on-policy rollouts of P2H10S-F-s0 over the 160 training scenes, Tokyo box): waits for stage B (12 scenes), checks it
# against the pre-registered checklist, then runs stage C (144 scenes) on both cards and checks it. Stage A (4) and B are started by hand
# (plans/2026-10-11-onpol-pai-prereg.md, section 6); all stages run pai_eval.sh from the frozen clone $R/code/pass1.
#   tmux new-window -d -t jev -n onpolC "bash onp_pass1_chain.sh"
# Writes $R/pass1/{CHAIN_STATUS, CHAIN_DONE | CHAIN_ERROR, chain.log}; frees the cards for the lane's agents ($R/GPU0_FREE, $R/GPU1_FREE) at the end.
set -uo pipefail
R=${R:-/data/runs/onpol_pai}; P=$R/pass1; here=$(cd "$(dirname "$0")" && pwd); cd "$R/code/pass1"
st() { echo "$(date '+%F %T') $*" | tee "$P/CHAIN_STATUS"; }
fail() { st "ERROR $*"; touch "$P/CHAIN_ERROR" "$R/GPU1_FREE"; exit 1; }
st "waiting for stage B"
until [[ -f $P/b/DONE || -f $P/b/ERROR ]]; do sleep 20; done
[[ -f $P/b/DONE ]] || fail "stage B failed"
python3 "$here/onp_pass_check.py" "$P/b" 12 | tee -a "$P/chain.log"; (( PIPESTATUS[0] == 0 )) || fail "stage B checklist"
st "stage C: 144 scenes on both cards"
CARDS="0 1" CONC=4 CHUNK=36 SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e PAI_DUMP=100000" \
  bash experiments/alpasim/scripts/pai_eval.sh "$P/c" P2H10S-F-s0 "$R/split/stageC.txt"
[[ -f $P/c/DONE ]] || fail "stage C: $(tail -1 "$P/c/ERROR" 2>/dev/null)"
python3 "$here/onp_pass_check.py" "$P/c" 144 | tee -a "$P/chain.log"; (( PIPESTATUS[0] == 0 )) || fail "stage C checklist"
touch "$R/GPU1_FREE" "$R/GPU0_FREE"; date > "$P/CHAIN_DONE"; st "done: pass 1 complete, cards free"
