#!/usr/bin/env bash
# Stop P6 generation cleanly (todos/2026-09-26-night-queue-2.md N1): the p6_gen.sh chains, their b2d_run runners
# (SIGINT: the route in flight is cancelled and never gets done/<id>.json, so a re-run redoes it) and every CARLA server
# of our blocks (ports of server index 800-949) whose runner is gone. Only processes of runs/p6 are touched.
#   scripts/p6_stop.sh [window ...]      default windows: every jev:p6-gen* / p6-smoke*
set -uo pipefail
wins=("$@")
(( ${#wins[@]} )) || mapfile -t wins < <(tmux list-windows -t jev -F '#W' 2>/dev/null | grep -E '^p6-(gen|smoke)')
for w in "${wins[@]}"; do tmux kill-window -t "jev:$w" 2>/dev/null && echo "killed window $w"; done
for p in $(pgrep -f 'p6_gen.sh'); do kill "$p" 2>/dev/null; done
runners() { pgrep -f 'b2d_run.py.*runs/p6/'; }
for p in $(runners); do kill -INT "$p"; done
for _ in $(seq 30); do [[ -z $(runners) ]] && break; sleep 2; done
for p in $(runners); do kill -TERM "$p"; done
sleep 5
for p in $(pgrep -f 'b2d_route.py.*runs/p6/'); do kill -TERM "$p"; done
for p in $(pgrep -f CarlaUE4-Linux-Shipping); do
    port=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -o 'carla-rpc-port=[0-9]*' | cut -d= -f2)
    [[ -z $port ]] && continue
    (( port >= 42000 && port < 49500 )) || continue
    sh=$(ps -o ppid= -p "$p" | tr -d ' '); r=$(ps -o ppid= -p "$sh" | tr -d ' ')
    [[ $r == 1 ]] || continue
    kill -TERM "$p" && echo "stopped CARLA $p (port $port)"
done
sleep 5
for p in $(pgrep -f CarlaUE4-Linux-Shipping); do
    port=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -o 'carla-rpc-port=[0-9]*' | cut -d= -f2)
    [[ -n $port ]] && (( port >= 42000 && port < 49500 )) || continue
    sh=$(ps -o ppid= -p "$p" | tr -d ' '); r=$(ps -o ppid= -p "$sh" | tr -d ' ')
    [[ $r == 1 || -z $r ]] && kill -KILL "$p"
done
echo "runners left: $(runners | wc -l)"
