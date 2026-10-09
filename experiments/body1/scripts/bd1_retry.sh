#!/usr/bin/env bash
# Bounded retry for a resumable BODY1 job: rerun the command while it dies by SIGKILL (rc 137, the box's memory.high kills), at most N tries.
# Usage: experiments/body1/scripts/bd1_retry.sh <N> <command> [args ...]   (the command must resume by itself, e.g. bd1_train.py train --auto)
set -u
n=$1; shift
for ((i = 1; i <= n; i++)); do
  "$@"; rc=$?
  [[ $rc -ne 137 ]] && exit $rc
  echo "==> try $i / $n died with rc 137 at $(date +%H:%M:%S); retrying in 20 s" >&2
  sleep 20
done
exit 137
