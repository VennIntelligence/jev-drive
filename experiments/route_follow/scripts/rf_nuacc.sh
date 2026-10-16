#!/usr/bin/env bash
# nuPlan JEV_ACC (plans/2026-10-16-nuplan-acc-prereg.md): openpilot-style longitudinal (serve_fix (d)) on RSG-L18-s0, held 460 @18, paired
# against a same-code re-read of RSG-L18-s0. Box, repo root; every job through the GPU pool, --after chained, PRIO (13); ids in $O/jobs.txt.
#   rf_nuacc.sh submit    the whole chain: smoke (old code at OLD_REV vs new code, JEV_ACC unset, SH30_COMPILE=0, 2 scenes; new code with
#                         VCONT + LEAD + ACC on 8 non-held scenes) -> gate (rf_nuacc.py ident) -> held 460 @18 x 3 arms (ref = no switch,
#                         acc = VCONT 1.0 + LEAD + ACC, vl = VCONT 1.0 + LEAD; rm_chain.sh read ... <sfx> -> route_mem/read/RSG-L18-s0-<sfx>)
#                         -> report (ol_confirm base ref, rf_follow, rf_nuacc.py report) -> $O/report/nuacc.md
#   rf_nuacc.sh held      (Amendment 1) the 18 held reads + report, no gate
#   rf_nuacc.sh report    (job body)
# Env: PRIO (13), OLD_REV (dd4c8d24 = the commit before the nuPlan wiring).
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
O=$DATA_DIR/runs/route_follow/nuacc; M=$DATA_DIR/runs/route_mem/read; L=$DATA_DIR/runs/onpol_long/lists
S=experiments/route_follow/scripts; ME=$S/rf_nuacc.sh; PY=$S/rf_nuacc.py; RM=experiments/route_mem/scripts/rm_chain.sh
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; V=.venv/bin/python; T=RSG-L18-s0
OLD_REV=${OLD_REV:-dd4c8d24}; OLD=$O/code_old
VL=(--env JEV_VCONT=1.0 --env JEV_LEAD=1); ACC=("${VL[@]}" --env JEV_ACC=1)
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner nuacc --name "$name" --tries 1 --priority ${PRIO:-13} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
submit)
  for x in re1016 acc vl; do [[ ! -e $M/$T-$x ]] || { echo "$M/$T-$x exists: never overwritten" >&2; exit 1; }; done
  [[ ! -e $O/smoke ]] || { echo "$O/smoke exists" >&2; exit 1; }
  mkdir -p $O/pool $O/smoke; [[ -d $OLD ]] || git worktree add --detach $OLD $OLD_REV || exit 1
  head -2 $L/smoke5.txt > $O/smoke/n2.txt
  { cat $L/smoke5.txt; for c in 1 3 5; do sed -n 40p $L/c$c.txt; done; } > $O/smoke/n8.txt
  g=()
  g+=($(sub nuacc-id-old --cwd $OLD --vram 20 --cpu 6 --ram 40 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_COMPILE=0 \
        -- bash $RM readlist $T $O/smoke/n2.txt $O/smoke/n_old)) || exit 1
  g+=($(sub nuacc-id-new --vram 20 --cpu 6 --ram 40 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_COMPILE=0 \
        -- bash $RM readlist $T $O/smoke/n2.txt $O/smoke/n_new)) || exit 1
  g+=($(sub nuacc-smoke-acc --vram 20 --cpu 6 --ram 40 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 "${ACC[@]}" \
        -- bash $RM readlist $T $O/smoke/n8.txt $O/smoke/n_acc)) || exit 1
  gt=$(sub nuacc-gate --after "$(IFS=,; echo "${g[*]}")" --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 -- $P $PY ident \
       --pair "nuplan=$O/smoke/n_old:$O/smoke/n_new" --on "acc=$O/smoke/n_acc" --out $O/smoke/ident.json) || exit 1
  h=()
  for x in re1016 acc vl; do
    sw=(); [[ $x == acc ]] && sw=("${ACC[@]}"); [[ $x == vl ]] && sw=("${VL[@]}")
    for i in 0 1 2 3 4 5; do
      h+=($(sub nuacc-held-$x-$i --after $gt --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 "${sw[@]}" -- bash $RM read $T $i 6 $x)) || exit 1
    done
  done
  r=$(sub nuacc-report --after "$(IFS=,; echo "${h[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  printf '%s\n' "code=$(git rev-parse --short HEAD)" "old=$OLD_REV" "ref=$T-re1016 (no switch)" "acc=$T-acc (JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1)" \
    "vl=$T-vl (JEV_VCONT=1.0 JEV_LEAD=1)" "stored=$T (2026-10-13)" "mode=official (ALPASIM_FAST unset)" > $O/config.txt
  echo "smoke ${g[*]} -> gate $gt; held ${h[*]}; report $r" ;;
held)   # Amendment 1: the held reads + report without the smoke gate (official nuPlan reads are not bit-reproducible run to run)
  for x in re1016 acc vl; do [[ ! -e $M/$T-$x/c0/run/aggregate/results-summary.json ]] || { echo "$M/$T-$x already read" >&2; exit 1; }; done
  h=()
  for x in re1016 acc vl; do
    sw=(); [[ $x == acc ]] && sw=("${ACC[@]}"); [[ $x == vl ]] && sw=("${VL[@]}")
    for i in 0 1 2 3 4 5; do
      h+=($(sub nuacc2-held-$x-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 "${sw[@]}" -- bash $RM read $T $i 6 $x)) || exit 1
    done
  done
  r=$(sub nuacc2-report --after "$(IFS=,; echo "${h[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report) || exit 1
  echo "code=$(git rev-parse --short HEAD) (Amendment 1 resubmit)" >> $O/config.txt
  echo "held ${h[*]}; report $r" ;;
report)
  R=$O/report; mkdir -p $R; H=(); F=()
  for X in $T-re1016 $T-acc $T-vl $T; do d=$(ls -d $M/$X/c*/run | paste -sd,); H+=("$X@held18=$d"); F+=("$X@18=$d"); done
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base $T-re1016 --pairs "$T-acc:$T-vl" --out $R/held18 > $R/held18.log 2>&1 || { echo "ol_confirm failed" >&2; exit 1; }
  $P $S/rf_follow.py --runs "${F[@]}" --out $R/follow --workers 16 > $R/follow.log 2>&1 || echo "rf_follow failed (descriptive only)" >&2
  $P $PY report --dir $R --ref $M/$T-re1016 --acc $M/$T-acc --vl $M/$T-vl --stored $M/$T --workers 16 ;;
*) sed -n 2,9p $0; exit 2 ;;
esac
