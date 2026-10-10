#!/usr/bin/env bash
# Open (or repair) N persistent ssh master connections GPU box -> here, for pull_train.py. Run from the Mac so the login comes from the forwarded agent
# (no key is copied to this box):  ssh-add ~/.ssh/id_ed25519; ssh -A ujs@<tokyo> 'bash /data/runs/op_parity/pull_train/open_masters.sh 32'
# A master (ControlPersist=yes) stays up after this session ends; re-run it when one has died (pull_train.py workers just wait and retry).
N=${1:-32}; W=/data/runs/op_parity/pull_train; mkdir -p $W/ctl; source /data/runs/alpasim/tokyo_setup/box.env
for i in $(seq 0 $((N-1))); do
  ssh -S $W/ctl/m$i -O check $BOX_USER@$BOX_HOST 2>/dev/null && continue
  rm -f $W/ctl/m$i
  ssh -M -S $W/ctl/m$i -fN -o ControlPersist=yes -o BatchMode=yes -o ForwardAgent=no -o ServerAliveInterval=20 -o ServerAliveCountMax=9 \
      -o ConnectTimeout=20 -c aes128-gcm@openssh.com -p $BOX_PORT $BOX_USER@$BOX_HOST 2>/dev/null && echo "m$i up" || echo "m$i failed"
  sleep 0.5
done
n=0; for i in $(seq 0 $((N-1))); do ssh -S $W/ctl/m$i -O check $BOX_USER@$BOX_HOST 2>/dev/null && n=$((n+1)); done; echo "$n/$N masters live"
