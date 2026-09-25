#!/usr/bin/env bash
# P2 = P + low-speed brake rule. Gated, unattended:
#   1 dev L1 (P, P2, D)                      gate: P2 failures <= D and primary <= 1.05 x D
#   2 dev closed loop, TFv6 A/C/P/P2 on 8 dev scenario routes, seed 0
#                                            gate: P2 vehicle collisions <= C
#   3 held-out: TFv6 P2 16 routes x 2 seeds, TCP P2 on the 8-route seed-0 subset
# Writes PASS/FAIL/DONE/BLOCKED to $V2/SIGNAL.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=/data/envs/tfv6/bin/python
V2=/data/runs/b2d/controller-eval/v2
IN=todos/2026-09-23-tfv6-controller/controller-eval
signal() { echo "$1 $(date -u +%FT%TZ) $2" > $V2/SIGNAL; echo "== $1 $2"; }
trap 'signal BLOCKED "p2-check failed (line $LINENO)"' ERR
P=$(realpath $IN/P-final.json); P2=$(realpath $IN/P2-final.json)
# Start after the running TCP subset (both halves print "end:" when finished).
until grep -q "^end:" $V2/tcp-sub-a.log && grep -q "^end:" $V2/tcp-sub-b.log; do sleep 60; done

signal RUNNING "p2 dev L1"
scripts/b2d_controller_eval_tune_round.sh p2check P,P2,D 2
$PY - <<EOF
import json,sys
t=json.load(open('$IN/tune/p2check-result.json'))['table']
ok=t['P2']['failures']<=t['D']['failures'] and t['P2']['primary']<=1.05*t['D']['primary']
print('gate1', ok, {k:(round(v['primary'],3),round(v['extra_jerk_rms_mps3'],2),v['failures']) for k,v in t.items()})
sys.exit(0 if ok else 3)
EOF

signal RUNNING "p2 dev closed loop"
$PY - <<EOF
import xml.etree.ElementTree as ET
ids=['1773','2050','2084','3072','24240','25358','27506','28198']
src={r.get('id'):r for r in ET.parse('/data/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml').getroot().findall('route')}
root=ET.Element('routes'); [root.append(src[i]) for i in ids]
ET.ElementTree(root).write('$V2/l23-dev.xml')
EOF
B2D_P_CONFIG=$P $PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --routes $V2/l23-dev.xml \
  --route-ids 1773,2050,2084,3072 --arms ACP --seeds 0 --out $V2/dev-l23-P --server-index 110 > $V2/dev-l23-a.log 2>&1 &
a=$!
B2D_P_CONFIG=$P $PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --routes $V2/l23-dev.xml \
  --route-ids 24240,25358,27506,28198 --arms ACP --seeds 0 --out $V2/dev-l23-P --server-index 112 > $V2/dev-l23-b.log 2>&1 &
b=$!
wait $a; wait $b
B2D_P_CONFIG=$P2 $PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --routes $V2/l23-dev.xml \
  --route-ids 1773,2050,2084,3072 --arms P --seeds 0 --out $V2/dev-l23-P2 --server-index 110 > $V2/dev-l23-c.log 2>&1 &
a=$!
B2D_P_CONFIG=$P2 $PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --routes $V2/l23-dev.xml \
  --route-ids 24240,25358,27506,28198 --arms P --seeds 0 --out $V2/dev-l23-P2 --server-index 112 > $V2/dev-l23-d.log 2>&1 &
b=$!
wait $a; wait $b
$PY - <<EOF
import json,glob,sys
from pathlib import Path
def coll(root,arm):
    n=0
    for d in glob.glob(f'{root}/cases/*/route-*/seed-*/{arm}/done.json'):
        done=json.load(open(d)); att=Path(d).parent/f"attempt-{done['attempt']}"
        n+=sum('COLLISION_VEHICLE' in x['event_type'] for x in json.load(open(att/'infractions.json'))['infractions'])
    return n
c={'A':coll('$V2/dev-l23-P','A'),'C':coll('$V2/dev-l23-P','C'),'P':coll('$V2/dev-l23-P','P'),'P2':coll('$V2/dev-l23-P2','P')}
print('gate2 vehicle collisions',c); sys.exit(0 if c['P2']<=c['C'] else 4)
EOF

signal RUNNING "p2 held-out closed loop"
ids=$($PY -c "import xml.etree.ElementTree as E;print(','.join(r.get('id') for r in E.parse('$IN/l23-v2-heldout.xml').getroot().findall('route')))")
half1=$(echo $ids | tr ',' '\n' | awk 'NR%2==1' | paste -sd,); half2=$(echo $ids | tr ',' '\n' | awk 'NR%2==0' | paste -sd,)
export B2D_P_CONFIG=$P2
$PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --route-ids $half1 --arms P --out $V2/l23-tfv6-P2 --server-index 110 > $V2/p2-tfv6-a.log 2>&1 &
a=$!
$PY scripts/b2d_controller_eval_campaign.py --planner tfv6 --route-ids $half2 --arms P --out $V2/l23-tfv6-P2 --server-index 112 > $V2/p2-tfv6-b.log 2>&1 &
b=$!
wait $a; wait $b
$PY scripts/b2d_controller_eval_campaign.py --planner tcp --route-ids 24816,24252,26944,25928 --arms P --seeds 0 --out $V2/l23-tcp-P2 --server-index 110 > $V2/p2-tcp-a.log 2>&1 &
a=$!
$PY scripts/b2d_controller_eval_campaign.py --planner tcp --route-ids 25845,26990,25863,3364 --arms P --seeds 0 --out $V2/l23-tcp-P2 --server-index 112 > $V2/p2-tcp-b.log 2>&1 &
b=$!
wait $a; wait $b
signal DONE "p2 check complete"
