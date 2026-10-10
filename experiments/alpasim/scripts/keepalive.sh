#!/usr/bin/env bash
# Mac side: every 10 min re-open dead ssh masters on the Tokyo box (forwarded agent, no key copied); exit when the transfer wrote DONE.
W=/data/runs/op_parity/pull_train; T=ujs@100.108.238.8
while true; do
  ssh -o ConnectTimeout=20 $T "test -e $W/DONE" && { echo "$(date '+%F %T') DONE seen, exit"; exit 0; }
  out=$(ssh -A -o ConnectTimeout=20 $T "bash $W/open_masters.sh 32" 2>&1 | grep -v setlocale | grep -v ' up$' | tail -2)
  [[ $out == "32/32 masters live" ]] || echo "$(date '+%F %T') $out"
  sleep 600
done
