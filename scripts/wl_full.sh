#!/usr/bin/env bash
# WL stage 4 (todos/2026-09-28-wm-loop.md): the full fork + D2 generation, then the feature steps. One-shot chain:
#   scripts/tmux_run.sh wl-full scripts/wl_full.sh     env: BUDGET_WH=290 DISK_FLOOR_GB=150 STALL_MIN=90 FEAT_DEMAND=2
# 1. runs/sched/demand/wm-loop.json = {card: workers} for the wm-loop row's cards: the G lane drains them at route
#    boundaries and scripts/wl_gen.sh waits for the room (at most 6 CARLA per card).
# 2. scripts/wl_gen.sh STAGE=full SETS="ba p6 d2" under a guard that, every 5 min, adds this lane's live CARLA servers
#    to the worker-hour count (runs/wl/pipe/gen_wh) and drains the runners (no new route, no route killed) with ERROR
#    at the stop line BUDGET_WH or below the disk floor; STALL appears while no run finished for STALL_MIN (and goes
#    away with the next one); an hourly progress line goes to STATUS.md.
#    Hourly, and once more at the end, the prefix check of checklist amendment (a) (python -m jevdrive.wl drops: a fork
#    group whose branches' prefixes disagree with the source run is dropped whole): more than 5 % dropped, overall or
#    per set (per set once 20 groups are checked), drains and stops like the stop line.
# 3. Harness failure (runs without a done record) <= 5 %, the registered checklist item; the whole checklist on the
#    full set (python -m jevdrive.wl sanity --stage full) is written as a description.
# 4. demand shrunk to FEAT_DEMAND workers on the row's first card; scripts/wl_pipeline.sh STEPS="index opspec op" on that
#    card and the row's cores, the drop check again with the openpilot `temporal` cosine, then STEPS="index vjepa z"
#    (index again so newly dropped groups leave the universe). 5. demand file removed.
# STATUS.md, DONE / ERROR / STALL in runs/wl/pipe/. Waits (STATUS line) while `sch_table.py check` fails at the start.
# Resumable: wl_gen skips finished runs, wl_pipeline skips finished feature chunks, gen_wh carries over.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
R=$DATA_DIR/runs/wl P=$DATA_DIR/runs/wl/pipe OUT=$DATA_DIR/runs/wl/gen DEM=$DATA_DIR/runs/sched/demand/wm-loop.json
mkdir -p "$P" "$(dirname "$DEM")"
rm -f "$P/DONE" "$P/ERROR" "$P/STALL" "$OUT/DRAIN"
exec > >(tee -a "$P/log.txt") 2>&1
BUDGET_WH=${BUDGET_WH:-290} DISK_FLOOR_GB=${DISK_FLOOR_GB:-150} STALL_MIN=${STALL_MIN:-90}
status() { echo "$(date '+%F %T') $*" | tee -a "$P/STATUS.md"; }
fail() { status "ERROR: $*"; echo "$*" > "$P/ERROR"; exit 1; }
eval "$(python3 - "$DATA_DIR/runs/sched/table.tsv" <<'PYEOF'
import csv, sys
r = next((r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["lane"] == "wm-loop"), None)
if r is None:
    print('echo "no wm-loop row"; exit 1'); sys.exit()
g = [x for x in r["gpus"].split(",") if x.isdigit()]
span = int(r["idx_span"])
m = dict(x.split(":") for x in r["idx0"].split(",")) if ":" in r["idx0"] else {x: int(r["idx0"]) + k * span for k, x in enumerate(g)}
print('G=(%s) W=%s CPUS=%s IDX="%s"' % (" ".join(g), r["workers"], r["cpus"],
      " ".join(str(i) for x in g for i in range(int(m[x]), int(m[x]) + span))))
PYEOF
)"
demand() { python3 -c 'import json,sys; print(json.dumps({g: int(sys.argv[1]) for g in sys.argv[2:]}))' "$@" > "$DEM.tmp" && mv "$DEM.tmp" "$DEM"; }
total() { .venv/bin/python -c "import pandas as pd; from jevdrive import wl; print(len(pd.read_parquet(wl.rundir('forks.parquet'))) + len(pd.read_parquet(wl.rundir('d2.parquet'))))"; }
ndone() { find "$OUT"/{ba,p6,d2}/done -name '*.json' 2>/dev/null | wc -l; }

guard() {  # every 5 min: worker-hours, disk floor, stall, hourly progress (step 2)
    local wh n last=$(date +%s) nd0=$(ndone) nd tick=0 free
    wh=$(cat "$P/gen_wh" 2>/dev/null || echo 0)
    while sleep 300; do
        n=$(python3 - "${G[*]}" "$IDX" <<'PYEOF'
import os, sys
gpus, idx, n = {f"-graphicsadapter={g}".encode() for g in sys.argv[1].split()}, set(map(int, sys.argv[2].split())), 0
for p in filter(str.isdigit, os.listdir("/proc")):
    try:
        a = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
    except OSError:
        continue
    if b"CarlaUE4-Linux-Shipping" in a[0] and gpus & set(a):
        port = next((int(x.split(b"=")[1]) for x in a if x.startswith(b"-carla-rpc-port=")), None)
        n += port is not None and (port - 2000) // 50 in idx
print(n)
PYEOF
)
        wh=$(python3 -c "print(round($wh + $n * 300 / 3600, 3))"); echo "$wh" > "$P/gen_wh"
        free=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc 0-9)
        nd=$(ndone)
        (( nd > nd0 )) && { last=$(date +%s); nd0=$nd; rm -f "$P/STALL"; }
        if python3 -c "import sys; sys.exit(not $wh >= $BUDGET_WH)"; then
            touch "$OUT/DRAIN"; status "stop line: $wh worker-h >= $BUDGET_WH, draining"; echo "budget $wh worker-h" > "$P/ERROR"; return
        fi
        if (( free < DISK_FLOOR_GB )); then
            touch "$OUT/DRAIN"; status "disk floor: $free GB free < $DISK_FLOOR_GB, draining"; echo "disk $free GB" > "$P/ERROR"; return
        fi
        if (( $(date +%s) - last > STALL_MIN * 60 )) && [[ ! -e $P/STALL ]]; then
            status "STALL: no run finished for $STALL_MIN min ($nd done, $n servers up)"; touch "$P/STALL"
        fi
        if (( ++tick % 12 == 0 )); then
            status "progress: $nd / $TOTAL runs done, $n CARLA up, $wh worker-h, $free GB free"
            if ! drops_gate 20; then
                touch "$OUT/DRAIN"; status "prefix drops above 5 %, draining"; echo "prefix drops above 5 %" > "$P/ERROR"; return
            fi
        fi
    done
}

drops_gate() {  # drops_gate <per-set min groups>: amendment (a) check, one STATUS line; non-zero when it fails
    local r rc
    r=$(taskset -c "$CPUS" .venv/bin/python -m jevdrive.wl drops --gate --gate-min "$1" --out "$OUT" 2>>"$P/log.txt"); rc=$?
    status "prefix drops: $(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d["dropped"], "/", d["checked"], "groups", d["per_set"], "cos runs", d["cos_checked_runs"], "ids", d["dropped_fork_ids"])' "$r" 2>/dev/null || echo "$r")"
    (( rc == 0 ))
}

n=0
until python3 scripts/sch_table.py check > "$P/sch_check.txt" 2>&1; do
    (( n++ % 6 == 0 )) && status "waiting: sch_table.py check fails ($(grep CHECK "$P/sch_check.txt" | tr '\n' ' '))"
    sleep 300
done
TOTAL=$(total)
status "wl-full start: cards ${G[*]} x $W, cores $CPUS, $(ndone) / $TOTAL runs already done, stop line $BUDGET_WH worker-h (spent $(cat "$P/gen_wh" 2>/dev/null || echo 0))"
demand "$W" "${G[@]}"
status "demand $(cat "$DEM")"
guard & GUARD=$!
STAGE=full SETS="ba p6 d2" scripts/wl_gen.sh
rc=$?
kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
[[ -e $P/ERROR ]] && { rm -f "$DEM"; status "demand file removed after the stop"; exit 1; }
(( rc == 0 )) || fail "wl_gen.sh exit $rc"
left=0
for s in ba p6 d2; do
    n=$(.venv/bin/python -m jevdrive.wl ids --stage full --set "$s" --out "$OUT" | tr ',' '\n' | grep -c .)
    status "$s: $n runs without a done record"; left=$(( left + n ))
done
status "generation end: $(ndone) / $TOTAL done, harness failure $left / $TOTAL, $(cat "$P/gen_wh") worker-h"
python3 -c "import sys; sys.exit(not $left > 0.05 * $TOTAL)" && { rm -f "$DEM"; fail "harness failure $left / $TOTAL above 5 %"; }
drops_gate 0 || { rm -f "$DEM"; fail "prefix drops above 5 % (runs/wl/drops.json)"; }
status "checklist on the full set (description)"
taskset -c "$CPUS" .venv/bin/python -m jevdrive.wl sanity --stage full --out "$OUT" > "$P/sanity_full.json" \
    || status "sanity readout failed (description only; generation stands)"

demand "${FEAT_DEMAND:-2}" "${G[0]}"
status "features on GPU ${G[0]}, demand $(cat "$DEM")"
# wl_pipeline logs to the same log.txt itself
GPU=${G[0]} CPUS=$CPUS STEPS="index opspec op" NO_DONE=1 scripts/wl_pipeline.sh > /dev/null
rc=$?
if (( rc == 0 )); then
    drops_gate 0 || { rm -f "$DEM"; fail "prefix drops above 5 % with the temporal cosine (runs/wl/drops.json)"; }
    GPU=${G[0]} CPUS=$CPUS STEPS="index vjepa z" scripts/wl_pipeline.sh > /dev/null
    rc=$?
fi
rm -f "$DEM"
status "demand file removed"
(( rc == 0 )) || { status "ERROR: wl_pipeline.sh exit $rc (see STATUS.md above)"; [[ -e $P/ERROR ]] || echo "wl_pipeline $rc" > "$P/ERROR"; exit 1; }
status "wl-full DONE"
touch "$P/DONE"
