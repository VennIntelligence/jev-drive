#!/usr/bin/env bash
# op_parity / wa-xboard lane (plans/2026-10-08-wa-xboard-prereg.md): WA-JEPA zero-shot on WOD-E2E val. One self-advancing chain in tmux jev
# (scripts/tmux_run.sh); every GPU job goes through the pool. Rerunning resumes. Stages: g0 -> n1 -> n10 -> all (staged 1 -> 10 -> all).
#   wa_xboard_chain.sh [all|n1|n10|full]
# Stages are run one at a time (look at the montage / outputs between them); `all` chains them only for a rerun.
# State: $DATA_DIR/runs/op_parity/wa_xboard/{STATUS, DONE-<stage>, ERROR, chain.log}.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=${1:-all}
D=$DATA_DIR/runs/op_parity/wa_xboard; mkdir -p "$D"; rm -f "$D/ERROR"
exec > >(tee -a "$D/chain.log") 2>&1
J=$DATA_DIR/envs/jevdrive/bin/python
CL="$J -m jevdrive.cl"
S=experiments/op_parity/scripts
status() { echo "$(date '+%F %T') op_parity wa_xboard: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return 0
        rm -f "$ld/ERROR"; $CL submit --owner op_parity --name "$n" --log-dir "$ld" "$@" >/dev/null || die "submit $n"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 30; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; }
REPO=$PWD
WJ() { echo "cd $DATA_DIR/third_party/wajepa && PYTHONPATH=$DATA_DIR/third_party/wajepa:$DATA_DIR/third_party/navsim $DATA_DIR/envs/wajepa/bin/python $REPO/$S/wa_xboard_run.py"; }

# $1 = suffix ('' = full, _n10 ...), $2 = limit (0 = all)
stage() {
  local suf=$1 lim=$2 L=$D/pool$1
  status "stage ${suf:-full}: render V1 / V2 (limit $lim)"
  for v in V1 V2; do sub wax-render-$v$suf $L/render-$v --cpu 48 --ram 64 -- $J $S/wa_xboard.py render --variant $v --limit $lim; done
  sub wax-req$suf $L/req --cpu 2 --ram 8 -- $J $S/wa_xboard.py req --limit $lim --suffix "$suf"
  waitdirs $L/render-V1 $L/render-V2 $L/req
  status "stage ${suf:-full}: WA-JEPA inference"
  local c1=V1$suf c2=V2$suf
  sub wax-run$suf $L/run --vram 14 --cpu 8 --ram 32 -- bash -c "$(WJ) --work $D --jobs V1$suf:req_V1$suf.npz:$c1 V2$suf:req_V2$suf.npz:$c2 V3$suf:req_V3$suf.npz:$c1 IMG0$suf:req_IMG0$suf.npz:- STATE0$suf:req_STATE0$suf.npz:$c1 V1n2$suf:req_V1$suf.npz:$c1:2"
  waitdirs $L/run
  touch "$D/DONE-${suf:-full}"
}

case $STAGE in
  g0)   sub wax-g0 $D/pool_g0 --cpu 4 --ram 16 -- $J $S/wa_xboard.py g0; waitdirs $D/pool_g0 ;;
  n1)   stage _n1 1 ;;
  n10)  stage _n10 10 ;;
  full) stage "" 0 ;;
  all)  "$0" g0 && "$0" n1 && "$0" n10 && "$0" full ;;
esac
status "$STAGE done"
