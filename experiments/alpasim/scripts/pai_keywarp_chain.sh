#!/usr/bin/env bash
# Lane PAI-KW (plans/2026-10-11-pai-keywarp-prereg.md): the closed-loop runs of the slot switch PAI_KEYWARP on the Tokyo box, one sequential
# chain of pai_eval.sh runs per card, each on the 33 scenes of s_b1a as one simulator launch, served configuration as decision 247.
#   pai_keywarp_chain.sh stage1          one scene: switch on (card 0, PAI_DUMP=1) and off (card 1), base s0 -> $K/stage1/{kw1-on,kw1-off}
#   pai_keywarp_chain.sh full            card 0: base-s0 on, base-s0 off, S-s0 on, S-s0 off (off-path identity against the stored run)
#                                        card 1: base-s1 on, base-s1 off, S-s1 on                      -> $K/<arm>-s<seed>-<on|off>/
# Run from a frozen copy of the repo's experiments/alpasim + jevdrive/openpilot/lead_long.py, in tmux `jev`. A finished run is skipped.
# State: $K/{STATUS, STAGE1_DONE | DONE | ERROR}; every run dir has pai_eval.sh's own STATUS / DONE / ERROR.
set -uo pipefail
S=$(cd "$(dirname "$0")" && pwd); K=${K:-/data/runs/alpasim/keywarp}; L=${LIST:-/data/runs/alpasim/pai_full/s_b1a.tsv}
SV="-e JEV_VCONT=1.0 -e JEV_LEAD=1"; ON="$SV -e PAI_KEYWARP=1"; R=/data/runs/op_parity/runs
st() { echo "$(date '+%F %T') pai-kw $*" | tee "$K/STATUS"; }
mkdir -p "$K"; rm -f "$K/ERROR"
if [[ $1 == stage1 ]]; then
  mkdir -p "$K/stage1"; awk -F'\t' 'NR == 2 {print $1}' "$L" > "$K/stage1/scene.txt"
  one() { GPU=$1 CONC=4 PAI_BASEPORT=$((6400 + 100 * $1)) TAG=P2H10-F-s0 DRV_ENV="$3 -v $R/P2H10-F-s0:/app/data/runs/op_parity/runs/P2H10-F-s0:ro" \
            bash "$S/pai_run.sh" "$K/stage1/$2" "$K/stage1/scene.txt" > "$K/stage1/$2.out" 2>&1 || echo "$2 failed" >> "$K/ERROR"; }
  st "stage 1: one scene, switch on (card 0) and off (card 1)"
  one 0 kw1-on "$ON -e PAI_DUMP=1" & a=$!; sleep 30; one 1 kw1-off "$SV" & b=$!; wait $a $b
  [[ -f $K/ERROR ]] && { st "ERROR $(cat "$K/ERROR")"; exit 1; }
  date > "$K/STAGE1_DONE"; st "stage 1 done"; exit 0
fi
chain() {  # card, then triples: run name, checkpoint tag, driver switches
  local g=$1; shift
  while (( $# )); do
    if [[ ! -f $K/$1/DONE ]]; then
      echo "$(date '+%F %T') card $g: $1"
      CARDS=$g SERVE="$3" bash "$S/pai_eval.sh" "$K/$1" "$2" "$L" > "$K/$1.out" 2>&1 || echo "$1 failed: $(tail -1 "$K/$1/ERROR" 2>/dev/null)" >> "$K/ERROR"
    fi
    shift 3
  done
}
st "full: 7 runs of $(($(wc -l < "$L") - 1)) scenes on two cards"
chain 0 base-s0-on P2H10-F-s0 "$ON" base-s0-off P2H10-F-s0 "$SV" S-s0-on P2H10S-F-s0 "$ON" S-s0-off P2H10S-F-s0 "$SV" & a=$!
sleep 30
chain 1 base-s1-on P2H10-F-s1 "$ON" base-s1-off P2H10-F-s1 "$SV" S-s1-on P2H10S-F-s1 "$ON" & b=$!
wait $a $b
[[ -f $K/ERROR ]] && { st "ERROR $(cat "$K/ERROR")"; exit 1; }
date > "$K/DONE"; st "done"
