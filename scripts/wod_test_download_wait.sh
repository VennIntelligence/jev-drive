#!/usr/bin/env bash
# Waits for the HUGSIM benchmark download to finish (it shares the box's bandwidth cap with the WOD-E2E
# test-split download, so the two must not run together -- see gpu-plan.md 2026-09-24 ~22:40), then relaunches
# the test-split download (idempotent: it skips the objects already finished) and the WOD-E2E test-submission
# chain behind it. Usage (on the box, in tmux): scripts/tmux_run.sh wodtest-wait scripts/wod_test_download_wait.sh
set -euo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
cd "$(dirname "$0")/.."
log() { echo "$(date '+%H:%M:%S') $*"; }

log "waiting for the HUGSIM benchmark download (scripts/hugsim_fetch.py) to finish"
# A bare process check is not enough: the HUGSIM downloader may be restarted (e.g. while tuning it), leaving a
# gap with no process. Also require the dataset to be (nearly) complete on disk: 60.73 GB in total.
hugsim_busy() {
    pgrep -f hugsim_fetch.py >/dev/null && return 0
    (( $(du -sb "$DATA_DIR/datasets/hugsim" | cut -f1) < 60 * 1000**3 ))
}
while hugsim_busy; do sleep 300; done
log "HUGSIM download is done; resuming the WOD-E2E test-split download"

echo "$(date '+%Y-%m-%d %H:%M') [WOD-E2E-TEST] HUGSIM download finished; resuming test-split download in jev:wodtest-dl" \
  >> "$DATA_DIR/runs/zeroshot-exam/gpu-plan.md"

tmux kill-window -t jev:wodtest-dl 2>/dev/null || true
tmux kill-window -t jev:wodtest-chain 2>/dev/null || true
scripts/tmux_run.sh wodtest-dl scripts/download_waymo_e2e.sh test --route proxy --streams 32
scripts/tmux_run.sh wodtest-chain bash "$DATA_DIR/runs/zeroshot-exam/wod-test/chain.sh"
log "relaunched jev:wodtest-dl and jev:wodtest-chain"
