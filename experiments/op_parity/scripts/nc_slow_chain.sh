#!/usr/bin/env bash
# op_parity nc-slow (plans/2026-10-09-nc-slow-prereg.md, lane NC1): is "where to slow down" decidable from inference-time inputs?
# One self-advancing chain in tmux jev (scripts/tmux_run.sh nc-slow experiments/op_parity/scripts/nc_slow_chain.sh):
#   feature extraction (12 navtrain shards + navtest x 2 seeds, GPU pool jobs) || navtrain rest metric cache (nt_cache.py, resumes)
#   -> family (held-out poses x 5 scales) -> score-poses (non-reactive, v2_navtrain) -> G-id -> build -> fits (8 arms) -> navtest cross-fit -> report.
# SMOKE=1: 64 rows of shard 0 and of navtest s0, 240 navtrain tokens scored, G-id on them, then stop.
# State: $DATA_DIR/runs/op_parity/nc_slow/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes (finished pool jobs and scored chunks are kept).
# CPU: at most JOBS x 12 cores of scoring / caching at a time (the AlpaSim lane is CPU-bound today).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/op_parity/nc_slow
D=$O/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
S=experiments/op_parity/scripts
X="$PY $S/nc_slow.py"
JOBS=${JOBS:-4}
L=$O/pool
KEYS="a070 a080 a090 a100 a110"
status() { echo "$(date '+%F %T') op_parity nc-slow: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1; shift; local ld=$L/$n; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="ncs-$n" '($3 == n || $4 == n) && ($2 == "queued" || $2 == "running") {print $1; exit}')   # a queued row has no card column
        [[ -n $live ]] && return; rm -rf "$ld"
        $CL submit --owner op_parity --name "ncs-$n" --log-dir "$ld" "$@" || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n/ERROR"; done; return 0; }
score() {   # score <suffix>
  # shellcheck disable=SC2086
  "$VPY" -m jevdrive.bench score-poses --poses "$O/poses_nt$1.npz" --keys $KEYS --tokens "$O/tokens_nt$1.txt" --out "$O/score_nt$1.csv" \
      --traffic non_reactive --mcache v2_navtrain --jobs "$JOBS" --owner op_parity --wait || die "score-poses$1"
  [[ -f $O/score_nt$1.csv ]] || die "score-poses$1 wrote no CSV"; }

if [[ ${SMOKE:-} == 1 ]]; then
  status "smoke: extraction (64 rows)"
  sub smoke-ext-0 --vram 17 --cpu 3 --ram 6 -- $X extract --shard 0 --limit 64
  sub smoke-ext-nt --vram 17 --cpu 3 --ram 6 -- $X extract --seed 0 --limit 64
  waitdirs smoke-ext-0 smoke-ext-nt
  status "smoke: family + scoring (240 tokens)"
  $X family --limit 120 || die "family smoke"
  score _smoke
  $X gate --smoke || die "G-id smoke"
  status "smoke done"; date > "$D/DONE"; exit 0
fi

status "extraction: 12 shards + navtest x 2 (pool); rest metric cache"
for i in $(seq 0 11); do sub ext-$i --vram 17 --cpu 3 --ram 6 -- $X extract --shard $i; done
for s in 0 1; do sub ext-nt$s --vram 17 --cpu 3 --ram 6 -- $X extract --seed $s; done
[[ -f $O/gate_id.json ]] || "$VPY" $S/nt_cache.py run --stage rest --limit 17 --jobs "$JOBS" || die "nt_cache rest"   # not needed once the family is scored
waitdirs $(for i in $(seq 0 11); do echo ext-$i; done) ext-nt0 ext-nt1

if [[ ! -f $O/gate_id.json ]]; then   # finished once: poses_nt.npz is not rewritten (its hash is the scoring run's identity)
  status "family"; $X family || die "family"
  status "scoring the navtrain family"; score ""
  status "G-id"; $X gate || die "G-id"
fi

status "build"
[[ -f $O/build.json ]] ||
sub build --vram 0.5 --cpu 8 --ram 80 -- $X build
[[ -f $O/build.json ]] || waitdirs build
grep -q '"ok": true' "$O/build.json" || die "extract gates failed (build.json)"

status "fits: 8 arms"
for a in E OP G0 G1; do sub fit-$a --vram 0.5 --cpu 6 --ram 12 -- $X fit --arm $a; done
for a in H V N OPH; do sub fit-$a --vram 8 --cpu 6 --ram 20 -- $X fit --arm $a; done
waitdirs fit-E fit-OP fit-G0 fit-G1 fit-H fit-V fit-N fit-OPH

status "secondary: navtest-internal cross-fit"
for a in E H V N OP OPH G0 G1; do sub xfit-$a --vram 0.5 --cpu 6 --ram 12 -- $X xfit --arm $a; done
waitdirs xfit-E xfit-H xfit-V xfit-N xfit-OP xfit-OPH xfit-G0 xfit-G1

status "report"
sub report --vram 0.5 --cpu 8 --ram 16 -- $X report
waitdirs report
status "done"; date > "$D/DONE"
