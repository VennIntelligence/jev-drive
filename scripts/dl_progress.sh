#!/usr/bin/env bash
# Append one progress line per interval to $DATA_DIR/runs/download/navsim/progress.log: trainval camera archives done,
# box rx MB/s, archive-dir MB/s, circuit-breaker limit. Run detached; stop it by pid.
#   nohup scripts/dl_progress.sh [interval_s=600] [total=183] >/dev/null 2>&1 &
set -uo pipefail
: "${DATA_DIR:?}"; A=$DATA_DIR/datasets/navsim/_archives; L=$DATA_DIR/runs/download/navsim/progress.log
iv=${1:-600}; total=${2:-183}; rx() { awk '/eth0/{print $2}' /proc/net/dev; }
sz() { du -sb "$A" | cut -f1; }
a=$(rx); s0=$(sz)
while sleep "$iv"; do
  b=$(rx); s1=$(sz); done_n=$(ls "$A/state" | grep -c 'trainval_camera.*\.done$')
  printf '%s done %s/%s box %.1f MB/s archives %.1f MB/s gate_limit %s\n' "$(date '+%F %T')" "$done_n" "$total" \
    "$(echo "($b-$a)/$iv/1000000" | bc -l)" "$(echo "($s1-$s0)/$iv/1000000" | bc -l)" "$(cat "$A/state/.gate/limit" 2>/dev/null || echo -)" >> "$L"
  a=$b; s0=$s1
done
