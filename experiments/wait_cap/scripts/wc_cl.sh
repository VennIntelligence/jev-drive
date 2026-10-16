#!/usr/bin/env bash
# WAIT-CAP closed-loop counterpart: cap_audit's collect (unchanged measures) for the lineage labels it did not hold. Box, repo root, CPU.
#   wc_cl.sh pub | held      -> $DATA_DIR/runs/cap_audit/collect-<set>/<ts>/ (dec / roll / log / scene parquet)
set -uo pipefail; cd "$(dirname "$0")/../../.."
AP=$DATA_DIR/third_party/alpasim/.venv/bin/python
case $1 in
pub)
  X=$(python3 - <<'P'
import json, os
R = os.environ["DATA_DIR"] + "/runs/"
o3, cf = (json.load(open(R + f)) for f in ("alpasim/ot3/a/manifest.json", "alpasim/cf1/a/manifest.json"))
d = {k: o3.get(k, []) + o3.get(k + "@new", []) + cf.get(k + "@new", []) + cf.get(k + "@fresh", []) for k in ("P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1")}
d |= {t: [f"route_follow/guard/read10/{t}/b*"] for t in ("RSG-L18-s0", "RSG-L18-s1", "RBSG2-L18-s0", "RBSG2-L18-s1")}
print(json.dumps(d))
P
)
  CA_EXTRA="$X" CA_ONLY=YR10m10-F-s0,P2-F-s0,P2-F-s1,P2H10-F-s0,P2H10-F-s1,RSG-L18-s0,RSG-L18-s1,RBSG2-L18-s0,RBSG2-L18-s1 $AP experiments/cap_audit/scripts/ca.py collect --set pub --workers 16 ;;
held)
  CA_EXTRA='{"RSG-L18-s0": ["onpol_long/roll/RSG-L18-s0-held/c*/run"], "RBSG2-L18-s0": ["onpol_long/roll/RBSG2-L18-s0-held/c*/run"]}' CA_ONLY=YR10m10-F-s0,RSG-L18-s0,RBSG2-L18-s0 \
    $AP experiments/cap_audit/scripts/ca.py collect --set held --workers 16 ;;
*) echo "wc_cl.sh pub | held" >&2; exit 2 ;;
esac
