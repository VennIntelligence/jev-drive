#!/usr/bin/env bash
# Box-wide memory / process sampler for SIGKILL forensics (todos/2026-09-25-closed-loop-infra-acceptance/sigkill.md).
# Jobs on the box are SIGKILLed from outside with the kernel oom_kill counter at 0. Without root nobody in the
# container can learn a SIGKILL's sender (no dmesg, audit, eBPF or tracefs; the parent's wait status carries only
# the signal number), so the next kill is identified by correlation: what the container's memory looked like in
# the seconds before it, and which process was the largest.
#
#   every 5 s  one line in $DATA_DIR/runs/boxwatch/<YYYYmmdd>.tsv: time, memory.current, anon, file, the
#              memory.events counters (high, max, oom_kill), memory.pressure "full" total (us), pids.current,
#              and the three largest processes by RSS as pid:rss_kB:comm
#   every 60 s a full `ps` (pid ppid pgid user etimes rss nlwp args) in ps/<HHMMSS>.txt, kept 3 h
#
# One instance per box (flock); scripts/slot_run.sh starts it on demand, detached, so any scheduled job arms it.
# Cost: one `ps` every 5 s (~30 ms of one core) and ~4 MB/day.
set -uo pipefail
: "${DATA_DIR:?DATA_DIR is not set}"
W=$DATA_DIR/runs/boxwatch
mkdir -p "$W/ps"
exec 9> "$W/.lock"
flock -n 9 || exit 0
echo "$$ $(date '+%F %T')" > "$W/.pid"
cg=/sys/fs/cgroup
kv() { awk -v k="$2" '$1 == k {print $2; exit}' "$cg/$1" 2>/dev/null; }
n=0
while :; do
    f=$W/$(date +%Y%m%d).tsv
    [[ -s $f ]] || printf 't\tmem_current\tanon\tfile\tev_high\tev_max\tev_oom_kill\tpsi_full_us\tpids\ttop3_pid:rss_kB:comm\n' > "$f"
    top=$(ps -eo pid=,rss=,comm= --sort=-rss 2>/dev/null | head -3 | awk '{printf "%s:%s:%s ", $1, $2, $3}')
    psi=$(awk '/^full/ {sub("total=", "", $5); print $5}' "$cg/memory.pressure" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%T)" "$(cat $cg/memory.current 2>/dev/null)" \
        "$(kv memory.stat anon)" "$(kv memory.stat file)" "$(kv memory.events high)" "$(kv memory.events max)" \
        "$(kv memory.events oom_kill)" "$psi" "$(cat $cg/pids.current 2>/dev/null)" "$top" >> "$f"
    if (( n++ % 12 == 0 )); then
        ps -eo pid,ppid,pgid,user,etimes,rss,nlwp,args --sort=-rss 2>/dev/null | cut -c1-240 > "$W/ps/$(date +%H%M%S).txt"
        find "$W/ps" -name '*.txt' -mmin +180 -delete
    fi
    sleep 5
done
