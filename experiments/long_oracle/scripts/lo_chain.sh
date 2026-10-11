#!/usr/bin/env bash
# long_oracle (plans/2026-10-11-long-oracle-prereg.md): decision 196's taxonomy + scaling oracle (op_parity/scripts/nc_tax_chain.sh, parameterised)
# for each arm, then lo_report.py. CPU only. tmux: scripts/tmux_run.sh long-oracle experiments/long_oracle/scripts/lo_chain.sh
# State: $DATA_DIR/runs/long_oracle/chain/{STATUS, DONE, ERROR, log.txt}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/long_oracle/chain; mkdir -p "$D"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
status() { echo "$(date '+%F %T') long_oracle: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
export NC_TAX_EXP=long_oracle NC_TAX_EXTRA="0.85 0.95 1.05 1.15" JOBS=${JOBS:-8}
# navhard stage 1 (counts only) on G frames, the only navhard harness of these arms; base s0 / s1 have none, so navtest only there
for arm in P2H10S-F P2H10-F; do
  status "nc_tax chain $arm"
  [[ $arm == P2H10S-F ]] && B="navtest navhard" || B=navtest
  NC_TAX_BENCHES="$B" NC_TAX_MODEL="$arm-s{}" NC_TAX_NAVHARD_MODEL="$arm-s{}@gimm" NC_TAX_OUT="runs/long_oracle/$arm" \
    bash experiments/op_parity/scripts/nc_tax_chain.sh || die "nc_tax $arm"
done
status "report"; .venv/bin/python experiments/long_oracle/scripts/lo_report.py || die "lo_report"
status "done"; date > "$D/DONE"
