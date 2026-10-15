#!/usr/bin/env bash
# RULE EXPERT (plans/2026-10-15-rule-expert-prereg.md): packs, smoke and the closed-loop reads of the map-based rule expert
# (experiments/alpasim/lib/rule_expert.py) on the box (held 460 / 18 decisions, sh30 driver) and Tokyo (PAI 281, gains 20).
# The model (RSG-L18-s0) still runs in the driver and is logged as poses_model; the served plan is the expert's (drive.jsonl `poses`).
# Variant flags (any command that drives): --expert-lookahead <s> (JEV_EXPERT_LA, default 1.2), --expert-entry-v <m/s> (JEV_EXPERT_EV,
# default 4.6); label XP-L<la>-E<ev> (dots dropped: XP-L12-E46).
# box (repo root):
#   rf_expert.sh box-pack                     pool CPU job: rf_expert.py pack on the held 460 (RSG-L18-s0's read configs) -> $O/pack_held
#   [SFX=s] rf_expert.sh box-smoke [flags]    the 3 held smoke scenes (results/rule_expert/smoke_held.txt), one pool job -> $O/smoke/<label>
#   rf_expert.sh box-read [flags]             6 chunk jobs (rm_chain.sh read RSG-L18-s0 i 6 <label>) -> route_mem/read/RSG-L18-s0-<label>,
#                                             then box-report as a pool job after them
#   rf_expert.sh box-report <label>           held.py, ol_confirm (vs RSG-L18-s0 and -s1), rf_follow, rf_appheld -> $O/report/<label>
# Tokyo (~/mycode/jev-drive):
#   rf_expert.sh tokyo-pack                   alpasim-base:0.89.0 (CPU, niced): rf_expert.py pack on r281/rsg -> $T/pack_pai
#   rf_expert.sh tokyo-smoke [flags]          the 3 PAI smoke turns (results/rule_expert/smoke_pai.txt) on CARDS (default 1) -> $T/smoke/<label>
#   rf_expert.sh tokyo-run [flags]            PAI 281 with the reference chunk lists in tmux jev:xp-<label> on CARDS ("0 1") -> $T/<label>;
#                                             JOIN=1 adds CARDS to a running read
#   rf_expert.sh tokyo-read <label>           rf_pai281 (all281 / exam72 / control75 vs rsg), rf_pai_follow extract, rf_v2pai turns / table,
#                                             rf_appread (22 runway turns) -> $T/<label>/read.md (+ turns.json, runway.md)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); RES=$REPO/experiments/route_follow/results
cmd=${1:?command}; shift
LA=1.2; EV=4.6; ARGS=()
while (( $# )); do case $1 in
  --expert-lookahead) LA=$2; shift 2 ;;
  --expert-entry-v) EV=$2; shift 2 ;;
  *) ARGS+=("$1"); shift ;;
esac; done
LABEL=XP-L${LA//./}-E${EV//./}

case $cmd in
box-*)
  cd "$REPO"; O=$DATA_DIR/runs/route_follow/rule_expert; M=$DATA_DIR/runs/route_mem/read; OL=$DATA_DIR/runs/onpol_long
  P=$DATA_DIR/third_party/alpasim/.venv/bin/python; mkdir -p $O/pool
  sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner rule-expert --name "$name" --tries 1 --priority ${PRIO:-12} \
          --no-check --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
  XENV=(--env JEV_EXPERT=$O/pack_held --env JEV_EXPERT_LA=$LA --env JEV_EXPERT_EV=$EV) ;;
tokyo-*)
  D=${DATA_DIR:-/data}; T=$D/runs/route_night/expert; REF=$D/runs/route_night/r281/rsg; EXAM=$REPO/experiments/alpasim/results/pai_exam
  export WIZ=controller.gains.rel_front_steering_angle_weight=20
  XSERVE="-e JEV_EXPERT=/xp -e JEV_EXPERT_LA=$LA -e JEV_EXPERT_EV=$EV -v $T/pack_pai:/xp:ro"; mkdir -p "$T" ;;
esac

case $cmd in
box-pack)
  sub xp-pack --vram 0.5 --cpu 32 --ram 120 --timeout-h 2 -- $P "$HERE/rf_expert.py" pack --workers 32 --out $O/pack_held --runs $(ls -d $M/RSG-L18-s0/c*/run) ;;
box-smoke)
  L=$RES/rule_expert/smoke_held.txt; D=$O/smoke/$LABEL${SFX:+-$SFX}; [[ ! -e $D ]] || { echo "$D exists" >&2; exit 1; }
  sub xp-smoke-$LABEL${SFX:+-$SFX} --vram 20 --cpu 6 --ram 40 --timeout-h 1 --env CL_HIST_KEY=alpasim-read4 "${XENV[@]}" \
      -- bash experiments/route_mem/scripts/rm_chain.sh readlist RSG-L18-s0 $L $D ;;
box-read)
  ids=()
  for i in 0 1 2 3 4 5; do
    ids+=($(sub xp-read-$LABEL-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 "${XENV[@]}" \
            -- bash experiments/route_mem/scripts/rm_chain.sh read RSG-L18-s0 $i 6 $LABEL)) || exit 1
  done
  sub xp-report-$LABEL --after "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash "$HERE/rf_expert.sh" box-report $LABEL > /dev/null || exit 1
  tail -7 $O/jobs.txt ;;
box-report)
  L=${ARGS[0]:?label}; X=RSG-L18-s0-$L; R=$O/report/$L; mkdir -p $R
  RR=""; H=(); F=(); PH=(); PF=()
  for t in $X RSG-L18-s0 RSG-L18-s1; do d=$(ls -d $M/$t/c*/run | paste -sd,); RR="$RR $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d"); done
  for b in RSG-L18-s0 RSG-L18-s1; do PH+=("$X:$b"); PF+=("$X@18:$b@18"); done
  python3 $DATA_DIR/runs/route_follow/held.py $RR > $R/held_raw.txt; rc0=$?
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --pairs "${PH[@]}" --out $R/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $R/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  $P experiments/route_follow/scripts/rf_appheld.py --follow $R/follow --out $R/entry --pairs "${PF[@]}"; rc3=$?
  echo "held $rc0 held18 $rc1 follow $rc2 entry $rc3"; exit $((rc0 + rc1 + rc2 + rc3)) ;;
tokyo-pack)
  exec nice -n 15 docker run --rm --cpus "${CPUS:-8}" --entrypoint bash \
    -v /data/datasets/nurec:/mnt/nre-data:ro -v /data/third_party/alpasim/src:/repo/src -v /data/third_party/alpasim/plugins:/repo/plugins \
    -v /data/runs:/data/runs:ro -v "$T":/out -v "$HERE/rf_expert.py":/w/rf_expert.py:ro -v "$HERE/rf_corridor.py":/w/rf_corridor.py:ro \
    -v "$HERE/rf_tax.py":/w/rf_tax.py:ro -v "$REPO/experiments/junction_eyes/lib/je_ruler.py":/w/je_ruler.py:ro alpasim-base:0.89.0 \
    -c "umask 0000 && cd /repo && uv run --no-sync python /w/rf_expert.py pack --workers ${WORKERS:-8} --out /out/pack_pai --runs $REF" ;;
tokyo-smoke)
  O=$T/smoke/$LABEL${SFX:+-$SFX}; [[ ! -e $O ]] || { echo "$O exists" >&2; exit 1; }; mkdir -p "$O"; git -C "$REPO" log --oneline -1 > "$O/COMMIT"
  SERVE="$XSERVE" RUNPFX=xps- CHUNK=3 CARDS=${CARDS:-1} bash "$REPO/experiments/alpasim/scripts/pai_eval.sh" "$O" RSG-L18-s0 "$RES/rule_expert/smoke_pai.txt" ;;
tokyo-run)
  O=$T/$LABEL; mkdir -p "$O/chunks"
  ls "$O"/chunks/*.txt > /dev/null 2>&1 || cp "$REF"/chunks/*.txt "$O/chunks/"           # the reference's own chunk lists
  git -C "$REPO" log --oneline -1 >> "$O/COMMIT"
  # the tmux window does not inherit this environment: WIZ (gains 20), SERVE, CARDS, JOIN go on its command line (route v2 Amendment 18)
  tmux new-window -d -t jev -n "xp-$LABEL${JOIN:+-j}" "bash -c 'JOIN=${JOIN:-0} WIZ=$WIZ SERVE=\"$XSERVE\" RUNPFX=xp- CARDS=\"${CARDS:-0 1}\" CHUNK=36 \
bash $REPO/experiments/alpasim/scripts/pai_eval.sh $O RSG-L18-s0 $D/runs/route_night/test281.txt; [[ ${JOIN:-0} == 1 ]] || date > $O/RUN_DONE'"
  echo "launched $LABEL -> $O (tmux jev:xp-$LABEL${JOIN:+-j}); state $O/STATUS, done when $O/RUN_DONE exists" ;;
tokyo-read)
  L=${ARGS[0]:?label}; O=$T/$L; l=${L,,}
  [[ -f $O/DONE ]] || { echo "$O: no DONE ($(cat "$O/STATUS"))" >&2; exit 1; }
  python3 "$HERE/rf_pai281.py" --rows rsg="$REF" "$l=$O" --ref rsg --lists all281=$D/runs/route_night/test281.txt exam72="$EXAM/hard_exam.txt" \
      control75="$EXAM/control.txt" --vs rsg --out "$O/read281.md" || exit 1
  [[ -f $O/follow.pkl ]] || WORKERS=${WORKERS:-8} CPUS=${CPUS:-8} bash "$HERE/rf_pai_follow.sh" "$O/follow.pkl" "$l=$O" || exit 1
  cat "$RES/pai281/strict36.txt" "$EXAM/hard_exam.txt" | awk '!s[$1]++' > "$O/turn_scenes.txt"
  python3 "$HERE/rf_v2pai.py" turns --pkl rsg=$D/runs/route_night/pai_follow/r281.pkl "$l=$O/follow.pkl" --scenes "$O/turn_scenes.txt" --out "$O/turns.json" || exit 1
  python3 "$HERE/rf_v2pai.py" table --turns "$O/turns.json" --a rsg --b "$l" --strict "$RES/pai281/strict36.txt" --exam "$EXAM/hard_exam.txt" --md "$O/turns.md" || exit 1
  python3 "$HERE/rf_appread.py" --a rsg=$D/runs/route_night/pai_follow/r281.pkl --b "$l=$O/follow.pkl" --strict "$RES/pai281/strict36.txt" \
      --runway "$RES/approach/pai_runway22.txt" --out "$O/runway.md" || exit 1
  ${PY:-/data/envs/op-train/bin/python} "$HERE/rf_expert.py" report --kind pai --pack $T/pack_pai --turns "$O/turns.json" --xp "$l" --refs rsg \
      --runs "$l=$O" rsg="$REF" --strict "$RES/pai281/strict36.txt" --control "$EXAM/control.txt" --out "$O/xp" > /dev/null || exit 1
  { echo "# PAI 281: rule expert $L vs RSG-L18-s0 (stored read r281/rsg), gains 20"; echo
    echo "Run \`$O\` ($(head -1 "$O/COMMIT" 2>/dev/null)), reference \`$REF\`; rf_expert.sh tokyo-read."
    echo; echo "## All 281 and the exam / control lists (rf_pai281.py)"; echo; cat "$O/read281.md"
    echo; echo "## Strict turns (36) and exam scenes (72) (rf_v2pai.py turns / table)"; echo; sed 's/^# /### /' "$O/turns.md"
    echo; echo "## Runway turns (22) (rf_appread.py)"; echo; sed 's/^# /### /' "$O/runway.md"
    echo; echo "## Lines and the per-turn failure list (rf_expert.py report)"; echo; sed 's/^# /### /' "$O/xp/lines.md"; echo; sed 's/^# /### /' "$O/xp/turns.md"; } > "$O/read.md"
  echo "-> $O/read.md" ;;
*) sed -n '2,22p' "$0"; exit 2 ;;
esac
