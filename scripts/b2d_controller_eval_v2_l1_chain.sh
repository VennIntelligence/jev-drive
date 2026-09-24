#!/usr/bin/env bash
# Task 10 v2 L1 chain after the probe batch: references -> ramp -> profile -> interface modes.
# Stops at the first failing step (set -e); every step is resumable (done.json per route).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
V2=/data/runs/b2d/controller-eval/v2
IN=todos/2026-09-23-tfv6-controller/controller-eval

$PY scripts/b2d_controller_eval_refs.py --probes $V2/probe --cruises $IN/l1-cruises.json --out $V2/refs
test "$($PY -c "import json;print(len(json.load(open('$V2/refs/profile-traces.json'))))")" = 40
for kind in ramp profile; do
  $PY scripts/b2d_controller_eval_l1_v2.py --kind $kind --refs $V2/refs --out $V2/l1-$kind
done
# Interface subset: every third route of the v2 held-out list, p01 only.
$PY - <<EOF
import xml.etree.ElementTree as ET
root = ET.parse('$IN/l1-v2-heldout.xml').getroot()
keep = root.findall('route')[::3]
for r in list(root):
    if r not in keep: root.remove(r)
ET.ElementTree(root).write('$V2/l1-interface-routes.xml', encoding='utf-8', xml_declaration=True)
print(len(keep), 'interface routes')
EOF
for mode in short_2s sparse_5s stop_jitter; do
  $PY scripts/b2d_controller_eval_l1_v2.py --kind profile --refs $V2/refs --interface $mode \
    --routes $V2/l1-interface-routes.xml --seeds p01 --out $V2/l1-interface-$mode
done
echo "L1 v2 chain complete"
