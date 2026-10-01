#!/usr/bin/env bash
# Wait script on Mac for remote op_l_b2d batch completion or error alert
ROOT="/root/autodl-tmp/ujs/runs/op_l_b2d"

echo "Waiting for op_l_b2d batch to complete on autodl..."
while true; do
  res=$(ssh autodl "if [ -f $ROOT/ALERT-ERROR ]; then echo 'ALERT'; elif [ -f $ROOT/ALL-COMPLETE ]; then echo 'COMPLETE'; else echo 'RUNNING'; fi" 2>/dev/null)
  if [ "$res" = "ALERT" ]; then
    echo "ALERT: Error detected on autodl!"
    exit 1
  elif [ "$res" = "COMPLETE" ]; then
    echo "COMPLETE: op_l_b2d finished all stages on autodl!"
    exit 0
  fi
  sleep 60
done
