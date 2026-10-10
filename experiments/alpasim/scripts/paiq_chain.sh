#!/usr/bin/env bash
# Lane alpasim-pai-quick on the Tokyo box: serving arms +a+b and +c2+b of lib/serve_fix.py on the 40 PAI scenes of paiq_pick.py
# (P2H10-F-s0, CONC 4, unharmonised renderer = pai_run.sh defaults), one arm per card at the same time, then pai_report.py and paiq_diff.py.
#   tmux new-window -d -t jev -n paiq-chain 'bash ~/mycode/jev-drive/experiments/alpasim/scripts/paiq_chain.sh'
# State: $Q/{STATUS, DONE | ERROR, log.txt}; runs in $Q/runs/{smoke_ab, ab, c2b}; table in $Q/paiq_table.md and $Q/paiq_diff.md.
set -uo pipefail
Q=/data/runs/alpasim/paiq; S=$(cd "$(dirname "$0")" && pwd); REF=/data/runs/alpasim/pai1/ref; L=$Q/scenes40.tsv
cd "$Q" && rm -f DONE ERROR && exec > >(tee -a "$Q/log.txt") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$Q/STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$Q/ERROR"; exit 1; }
arm_flags() { case $1 in ab) echo "-e JEV_VCONT=1.0 -e JEV_LEAD=1" ;; c2b) echo "-e JEV_BASE=2 -e JEV_LEAD=1" ;; esac; }
run() {  # arm, card, base port, list, out
  GPU=$2 CONC=4 PAI_BASEPORT=$3 DRV_ENV="$(arm_flags "$1")" bash "$S/pai_run.sh" "$5" "$4"; }
echo "code: $(cd "$S" && git rev-parse --short HEAD) + main's experiments/alpasim/{lib,scripts} (see results file)"
mkdir -p runs; awk -F'\t' 'NR > 1 && NR <= 3 {print $1}' "$L" > smoke2.txt
if [[ ! -f runs/smoke_ab/DONE ]]; then
  st "smoke: ab on 2 scenes, card 0"; run ab 0 6700 smoke2.txt "$Q/runs/smoke_ab" > smoke.out 2>&1 || die "smoke run failed, see smoke.out"
fi
python3 - <<'PY' || die "smoke summary empty"
import json, sys
d = json.load(open("/data/runs/alpasim/paiq/runs/smoke_ab/sim/aggregate/results-summary.json"))
print("smoke rollouts", len(d["rollouts"]), [round(r["score"], 3) for r in d["rollouts"]]); sys.exit(0 if len(d["rollouts"]) == 2 else 1)
PY
st "launch: ab on card 0, c2b on card 1, 40 scenes each"
t0=$(date +%s)
( run ab 0 6700 "$L" "$Q/runs/ab" > ab.out 2>&1; echo "ab wall $(( $(date +%s) - t0 )) s" >> "$Q/walls.txt" ) & p0=$!
( run c2b 1 6800 "$L" "$Q/runs/c2b" > c2b.out 2>&1; echo "c2b wall $(( $(date +%s) - t0 )) s" >> "$Q/walls.txt" ) & p1=$!
while kill -0 $p0 2>/dev/null || kill -0 $p1 2>/dev/null; do
  st "running: $(find runs/ab runs/c2b -name _complete 2>/dev/null | wc -l) rollouts of 80 complete"; sleep 60
done
wait $p0; wait $p1
for a in ab c2b; do [[ -f runs/$a/DONE ]] || die "arm $a did not finish, see $a.out"; done
python3 "$S/pai_report.py" table --runs "+a+b=$Q/runs/ab" "+c2+b=$Q/runs/c2b" --ref "$REF" --out "$Q/paiq_table.md" > report.log 2>&1 || die "table failed"
python3 "$S/paiq_diff.py" --a "+a+b=$Q/runs/ab" --b "+c2+b=$Q/runs/c2b" --ref "$REF" --out "$Q/paiq_diff.md" >> report.log 2>&1 || die "diff failed"
date > DONE; st "done: $Q/paiq_table.md"
