#!/usr/bin/env bash
# TTS round (plans/2026-10-16-tts-prereg.md): serve-side test-time candidate selection (lib/tts.py, JEV_TTS) on RSG-L18-s0. Box, repo root;
# every job through the GPU pool, --after chained, priority 9; ids in $O/jobs.txt.
#   rf_tts.sh submit     the whole chain in one go: weight fit (rf_tts.py fit; rc 3 = the registered stop line, then nothing reads) ->
#                        identity gate, 2 scenes per driver (old code = worktree at OLD_REV vs new code with JEV_TTS unset, SH30_COMPILE=0; + a
#                        JEV_TTS=1 smoke each) -> held 460 @18 with JEV_TTS=1 (6 chunks, rm_chain.sh read ... tts -> route_mem/read/RSG-L18-s0-tts)
#                        -> held report (ol_confirm vs the stored RSG-L18-s0 read, rf_follow, rf_tts.py held) -> stage2
#   rf_tts.sh stage2     (job body) when held zeros d <= +3: PAI 281 seed 0 with JEV_TTS=1 (8 chunks, the reference's lists and serve) ->
#                        extracts (rf_recov_post.sh paix) -> PAI report (rf_londiag pai, rf_v2pai turns, rf_tts.py pai); else writes $O/STAGE2
#   rf_tts.sh heldrep | pairep    (job bodies)
# Env: PRIO (9), OLD_REV (b7b4d5c1 = the commit before the switch).
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}
RF=$DATA_DIR/runs/route_follow; O=$RF/tts; RE=$RF/recovery_envelope; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long
S=experiments/route_follow/scripts; ME=$S/rf_tts.sh; PY=$S/rf_tts.py; RM=experiments/route_mem/scripts/rm_chain.sh; PB=experiments/alpasim/scripts/pai_box.sh
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; V=.venv/bin/python
CH=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks; STRICT=experiments/route_follow/results/pai281/strict36.txt; WL=$RE/pai_wl.pkl
WIZ=controller.gains.rel_front_steering_angle_weight=20; T=RSG-L18-s0; A=RSG-L18-s0-tts; W=$O/fit/weights.json
OLD_REV=${OLD_REV:-b7b4d5c1}; OLD=$O/code_old
PAIENV=(--env CL_HIST_KEY=pai-read4 --env CONC=4 --env "WIZ=$WIZ" --env KEEP_VIDEO=0 --env KEEP_ASL=0 --env JEV_VCONT=1.0 --env JEV_LEAD=1)
ON=(--env JEV_TTS=1 --env JEV_TTS_W=$W)
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner tts --name "$name" --tries 1 --priority ${PRIO:-9} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
case ${1:-} in
submit)
  [[ ! -e $M/$A && ! -e $O/ident ]] || { echo "$M/$A or $O/ident exists: never overwritten" >&2; exit 1; }
  [[ -d $OLD ]] || git worktree add --detach $OLD $OLD_REV || exit 1
  f=$(sub tts-fit --vram 16 --cpu 8 --ram 64 --timeout-h 1 --preflight "$TR $PY fit --smoke" -- $TR $PY fit) || exit 1
  mkdir -p $O/ident; L=$O/ident/n2.txt; head -2 $DATA_DIR/runs/onpol_long/lists/held_n6/c0.txt > $L
  g=()
  for x in old new on; do
    cw=(); af=(); sw=(); [[ $x == old ]] && cw=(--cwd $OLD); [[ $x == on ]] && { af=(--after $f); sw=("${ON[@]}"); }
    g+=($(sub tts-id-n-$x "${af[@]}" "${cw[@]}" --vram 20 --cpu 6 --ram 40 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 --env SH30_COMPILE=0 "${sw[@]}" \
          -- bash $RM readlist $T $L $O/ident/n_$x)) || exit 1
    R=$O/ident/p_$x; mkdir -p $R/chunks $R/runs; head -2 $CH/c000.txt > $R/chunks/c000.txt
    g+=($(sub tts-id-p-$x "${af[@]}" "${cw[@]}" --vram 28 --cpu 6 --ram 40 --timeout-h 1 "${PAIENV[@]}" "${sw[@]}" -- bash $PB run $R $T c000)) || exit 1
  done
  p=$O/ident/p_%s/runs/rsg-l18-s0_c000/sim
  gt=$(sub tts-ident --after "$(IFS=,; echo "${g[*]}")" --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 -- $V $PY ident \
       --pair "nuplan=$O/ident/n_old:$O/ident/n_new" "pai=$(printf $p old):$(printf $p new)" --on "nuplan_on=$O/ident/n_on" "pai_on=$(printf $p on)") || exit 1
  h=()
  for i in 0 1 2 3 4 5; do
    h+=($(sub tts-held-$i --after $gt --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 "${ON[@]}" -- bash $RM read $T $i 6 tts)) || exit 1
  done
  r=$(sub tts-heldrep --after "$(IFS=,; echo "${h[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME heldrep) || exit 1
  s=$(sub tts-stage2 --after $r --vram 0.5 --cpu 1 --ram 4 --timeout-h 1 -- bash $ME stage2) || exit 1
  echo "fit $f; ident ${g[*]} -> $gt; held ${h[*]}; report $r; stage2 $s"; echo "$s" > $O/stage2.id ;;
heldrep)
  mkdir -p $O/held; H=(); F=()
  for X in $T $A; do d=$(ls -d $M/$X/c*/run | paste -sd,); H+=("$X@held18=$d"); F+=("$X@18=$d"); done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base $T --out $O/held/held18 > $O/held/held18.log 2>&1 || { echo "ol_confirm failed" >&2; exit 1; }
  $P $S/rf_follow.py --runs "${F[@]}" --out $O/held/follow --workers 16 > $O/held/follow.log 2>&1 || { echo "rf_follow failed" >&2; exit 1; }
  $P $PY held --dir $O/held --runs $M/$A --ref-runs $M/$T ;;
stage2)
  ok=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['stage2'])" $O/held/held.json) || exit 1
  [[ $ok == True ]] || { echo "$(date '+%F %T') held zeros d > +3: no PAI read (prereg 4)" | tee $O/STAGE2; exit 0; }
  R=$O/pai/$A; [[ ! -e $R ]] || { echo "$R exists" >&2; exit 1; }; mkdir -p $R/chunks $R/runs; cp $CH/*.txt $R/chunks/
  printf '%s\n' "SERVE=JEV_VCONT=1.0 JEV_LEAD=1 JEV_TTS=1 JEV_TTS_W=$W" "WIZ=$WIZ" "CONC=4" "code=$(git rev-parse --short HEAD)" "chunks=$CH" > $R/config.txt
  ids=()
  for f in $R/chunks/*.txt; do c=$(basename $f .txt)
    ids+=($(sub tts-pai-$c --vram 28 --cpu 6 --ram 40 --timeout-h 3 "${PAIENV[@]}" "${ON[@]}" -- bash $PB run $R $T $c)) || exit 1; done
  x=$(sub tts-paix --after "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $S/rf_recov_post.sh paix $R) || exit 1
  r=$(sub tts-pairep --after $x --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME pairep) || exit 1
  echo "$(date '+%F %T') PAI ${ids[*]}; extracts $x; report $r" | tee $O/STAGE2 ;;
pairep)
  XA=$O/pai/$A; XR=$RE/pai/$T; Q=$O/paiout; mkdir -p $Q
  printf '{"arm": "%s", "ref": "%s"}\n' "$(cat $XA/track.txt)" "$(cat $XR/track.txt)" > $Q/pai_track.json
  $P $S/rf_londiag.py pai --tax $XA/tax.pkl --dec $XA/dec.pkl --ref-tax $XR/tax.pkl --ref-dec $XR/dec.pkl --wl $WL --workers 16 --out $Q/pai_londiag.json \
    > $Q/pai_londiag.txt 2>&1 || { echo "rf_londiag pai failed" >&2; exit 1; }
  $P $S/rf_v2pai.py turns --pkl "ref=$XR/follow.pkl:x" "arm=$XA/follow.pkl:x" --scenes $STRICT --out $Q/pai_turns.json > $Q/pai_turns.log 2>&1 \
    && $P $S/rf_v2pai.py table --turns $Q/pai_turns.json --a ref --b arm --strict $STRICT --md $Q/pai_turns.md >> $Q/pai_turns.log 2>&1 || { echo "rf_v2pai failed" >&2; exit 1; }
  $P $PY pai --dir $Q --runs $XA/runs --ref-runs $XR/runs --held $O/held ;;
*) sed -n 2,11p $0; exit 2 ;;
esac
