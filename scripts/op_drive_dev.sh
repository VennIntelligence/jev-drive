#!/usr/bin/env bash
# op-drive dev run (todos/2026-09-29-op-drive.md, "graded run 2"): tag f on the 10 dev routes, one seed on one card:
# drive + dlon (resume policy $DRIVE_ARGS), dbase linked from the earlier dv run, dbaseslow (linear speed match) and then
# dbaseslow2, dbaseslow3 until the mean-speed ratio v(drive) / v(dbaseslow) is within [0.9, 1.1] (registration P).
# Hand-offs in $OP_ARB_DIR: STATUS, DONE-dev, ERROR. Usage: GPU=1 IDX0=300 SEED=0 WORKERS=6 CPUS=16-47 scripts/op_drive_dev.sh
set -uo pipefail
: "${DATA_DIR:?}" "${GPU:?}" "${IDX0:?}" "${SEED:?}"
cd "$(dirname "$0")/.."
export OP_ARB_ARMS=${OP_ARB_ARMS:-$DATA_DIR/runs/op_drive/arms} OP_ARB_DIR=${OP_ARB_DIR:-$DATA_DIR/runs/op_drive_g$GPU}
export GPU IDX0 WORKERS=${WORKERS:-6} CPUS=${CPUS:-16-47} SEEDS=$SEED LAT_EXEC=curv RESUME_S=5
TAG=${TAG:-f}; A=$OP_ARB_ARMS
mkdir -p "$OP_ARB_DIR"; rm -f "$OP_ARB_DIR/ERROR"
[[ -e $A/$TAG-dbase-s$SEED ]] || ln -s "$A/dv-dbase-s$SEED" "$A/$TAG-dbase-s$SEED"
run() { ARMS="$1" scripts/op_arb.sh set 2 "$TAG" || { echo "set $TAG $1 failed $(date)" > "$OP_ARB_DIR/ERROR"; exit 1; }; }
ratio() {  # mean speed of drive over mean speed of the given slow arm, both over non-warm plan steps of the routes finished in both
    python3 - "$A/$TAG-drive-s$SEED" "$A/$TAG-$1-s$SEED" <<'PYEOF'
import glob, json, os, sys
def speeds(d):
    out = {}
    for f in glob.glob(os.path.join(d, "done", "*.json")):
        rid = os.path.basename(f)[:-5]
        p = os.path.join(d, "attempts", rid, str(json.load(open(f))["attempt"]), "plans.jsonl")
        v = [r["v"] for r in map(json.loads, open(p)) if not r["warm"]]
        out[rid] = sum(v) / max(len(v), 1)
    return out
a, b = speeds(sys.argv[1]), speeds(sys.argv[2])
k = [r for r in a if r in b]
print(round(sum(a[r] for r in k) / max(sum(b[r] for r in k), 1e-6), 3))
PYEOF
}
DRIVE_ARGS='"resume": "nored"' run "drive dlon"
run "dbase dbaseslow"
last=dbaseslow
for n in 2 3 4; do        # 3 iterations at most (registration P); resumable: arms already DONE are skipped
    if [[ -e $A/$TAG-dbaseslow$n-s$SEED/DONE ]]; then last=dbaseslow$n; continue; fi
    r=$(ratio "$last")
    echo "$(date '+%F %T') pacing s$SEED $last: v(drive)/v(slow) = $r" | tee -a "$OP_ARB_DIR/log.txt"
    python3 -c "import sys; sys.exit(0 if 0.9 <= float(sys.argv[1]) <= 1.1 else 1)" "$r" && break
    run "dbaseslow$n"; last=dbaseslow$n
done
echo "$(date '+%F %T') pacing s$SEED final $last: v(drive)/v(slow) = $(ratio "$last")" | tee -a "$OP_ARB_DIR/log.txt"
date > "$OP_ARB_DIR/DONE-dev"
