#!/usr/bin/env bash
# ONPOL-LONG amendment 1: G-load of tags through the PAI driver image (the docker call of onpol_pai onp_night.sh gload), output /data/runs/onpol_long/gload/<tag>.out
set -uo pipefail
cd /data/runs/onpol_pai/code/pass1; IMG=${IMG:-jev-alpasim:p2h10-f-s0-6f3d05d1}; g=${G:-1}; lib=$PWD/experiments/alpasim/lib; O=/data/runs/onpol_long/gload
for t in "$@"; do ck=/data/runs/op_parity/runs/$t
  docker run --rm --init --cap-drop ALL --security-opt no-new-privileges:true --read-only --network none --gpus "device=$g" \
    --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 --tmpfs /run:rw,nosuid,nodev,size=64m,mode=0755 \
    -v "$lib/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro" -v "$lib/pai_driver.py:/app/jev-drive/experiments/alpasim/lib/pai_driver.py:ro" \
    -v "$lib/serve_fix.py:/app/jev-drive/experiments/alpasim/lib/serve_fix.py:ro" -v "$PWD/jevdrive/openpilot/lead_long.py:/app/jev-drive/jevdrive/openpilot/lead_long.py:ro" \
    -v "$ck:/app/data/runs/op_parity/runs/$t:ro" -e "SH30_TAG=$t" -e JEV_VCONT=1.0 -e JEV_LEAD=1 "$IMG" python -c "
import os, sys
import numpy as np
sys.path.insert(0, \"/app/jev-drive/experiments/alpasim/lib\")
import pai_driver as D
from pai_core import C, plan
core = C.Core(os.environ[\"SH30_TAG\"])
core.lead_out, core.base_out = D.FX.NEED_LEAD, D.FX.BASE > 0
z = np.zeros
o = plan(core, z((8,) + C.FRAME, np.uint8), np.ones(8, bool), z((4, 3)), z((4, 2)), z(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
assert np.isfinite(o[\"poses\"]).all() and o[\"poses\"].shape == (8, 3) and \"lead\" in o
print(\"G-load OK:\", core.tag, \"arm\", core.model.arm, \"poses at 4 s\", o[\"poses\"][-1].round(3).tolist())
" > $O/$t.out 2>&1; grep -h "G-load OK" $O/$t.out || { echo "G-load FAIL $t"; tail -3 $O/$t.out; }
done
