#!/usr/bin/env bash
# RECOVERY-ENVELOPE row pools on the GPU box (prereg experiments/route_follow/plans/2026-10-16-recovery-envelope-prereg.md, sections R and
# "推快的接近"; rf_recov_roll.py). Every simulation is one pool job running the held-read serving (sh30, plain RSG-L18-s0, c0b_chain.OVERRIDES,
# 30 decisions at 2 Hz on a token + 14 s cache) with SH30_DUMP on, as rf_rewind.sh train64 nsim does without it; the rows are on_rows.py geo
# (unedited) -> rf_recov_roll.py select + assemble -> on_rows.py tok -> ol_labels.py build / gate -> rm_build.py pool -> on_lon_pack.py
# (--items B P, as on1_L18RSG) -> strip. Each submit command queues its whole chain at once (--after); AFTER=<ids> gates the first jobs.
# Box, from the repo root. Every submission serves its own worktree <stage>/code at the repo's HEAD.
#
#   rf_recov_roll.sh recov <root> <NAME> <from dir> <id stage dirs, comma> <pack dir> <cache dir> [chunks] [prior read.json]
#        R rows: the D2 / D3 handovers listed in <from dir>/from_D{2,3}.txt (units of <from dir>/tfail.json) re-run with the dump and the
#        expert packs, the source's identity sessions linked as <root>/<NAME>/id for n-read's rewind check, the read compared with the
#        source's own read (rcheck) -> pool on1_<NAME>. RF_K_END is passed to the read when set.
#        d316 (on1_RCVd316):  T=$DATA_DIR/runs/route_follow/rewind_train64
#          rf_recov_roll.sh recov $DATA_DIR/runs/route_follow/recov_roll RCVd316 $T $T/pilot10,$T/rest54 $T/pack \
#            $DATA_DIR/datasets/alpasim_nuplan_long/after280_rw 2 $T/all/read.json
#        all-scene rewind (decision 319, on1_RCVall): its stages smoke / s50 / rest / restb merged by `srcall` into one source dir
#          (all/from_D*.txt, the stages' tfail units, pack symlinks); rows = the source rows.csv label & within_tol rows (RR_ROWS_CSV):
#          A=$DATA_DIR/runs/route_follow/rewind_onpol_all; RR=$DATA_DIR/runs/route_follow/recov_roll
#          rf_recov_roll.sh srcall $RR $A $RR/src_all
#          PRIO=7 RF_K_END=29 RR_ROWS_CSV=$A/rows/rows.csv rf_recov_roll.sh recov $RR RCVall $RR/src_all $A/restb,$A/rest,$A/s50,$A/smoke \
#            $RR/src_all/pack $DATA_DIR/datasets/alpasim_nuplan_long/after280_op 8 $A/all/read.json
#   rf_recov_roll.sh srcall <root> <rewind_onpol_all dir> <out dir>   the merged source dir (read-only on the rewind's files)
#   rf_recov_roll.sh rmerge <root> <NAME> <pool:prefix,pool:prefix,..>   CPU job body: R pools merged (rf_pait.py merge, names on:<prefix>:..)
#        -> on1_<NAME> + <root>/report_<NAME>.json; the data-scaled pool on1_RCVallD = RCVall (pass 0) + the perturbed passes' pools
#        (results/rewind_onpol_all/passes.md; their rows come from rf_rewind.sh train64 ostage RW_ROWS, which runs rowsR / tok / rowsB)
#   rf_recov_roll.sh fast <root> stage <name> <count per class | all> <levels s,s,..> [chunks]
#        fast approaches: (fsel once -> <root>/sel.json) -> stage list (scenes not in an earlier stage) + JEV_PERTURB files (g 1, arc scale s,
#        NPUSH decisions, default 2, at k_p) -> `id` (no dump) and L<100 s> (dump) launches -> fread (<stage>/fread.json) -> <stage>/DONE
#   rf_recov_roll.sh fast <root> rows <NAME> <stage,stage..> <levels>     rows of the pushed sessions of those stages -> pool on1_<NAME>
# Env: RR_CACHE (fast's cache, default after280_op), RR_VRAM (20), PRIO (6), JEV_PY, NPUSH (2), AFTER.
# Job bodies: nsim, rowsR, rowsF, tok, rowsB, fprep, fread.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../../.." && pwd)
cmd=${1:?recov|fast|<job body>}; O=$(mkdir -p "${2:?root}" && cd "$2" && pwd); shift 2
D=$DATA_DIR; PY=${JEV_PY:-$repo/.venv/bin/python}; APY=$D/third_party/alpasim/.venv/bin/python
NAV=$D/envs/navsim2/bin/python; TR=$D/envs/op-train/bin/python; X=$here/rf_recov_roll.py; ON=$repo/experiments/onpol_nuplan
ROWS=$D/runs/onpol_nuplan/rows; T64=$D/runs/route_follow/rewind_train64; AF=280
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
nsub() { local name=$1; shift; local id; id=$(cd "$repo" && .venv/bin/python -m jevdrive.cl submit --owner ${RR_OWNER:-recov-roll} --name "$name" \
         --tries ${TRIES:-2} --priority ${PRIO:-6} --no-check "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"
         echo "$id $name" >> "$O/jobs.txt"; echo "$id"; }
code() {  # <stage dir>: its own worktree at the repo's HEAD
  local W=$1/code
  [[ -d $W ]] || git -C "$repo" worktree add -q --detach "$W" HEAD || exit 1
  git -C "$W" checkout -q --detach "$(git -C "$repo" rev-parse HEAD)" || exit 1
  echo "{\"served\": \"$W\", \"commit\": \"$(git -C "$W" rev-parse HEAD)\", \"dirty\": $(git -C "$W" status --porcelain -uno | grep -c .)}" > "$1/code.json"
}
geo() {  # <manifest dir> <geo rows name> <run dirs..>: on_rows.py geo over runs that carry onpol_sim.pkl, one label per run dir
  local M=$1 G=$2; shift 2; mkdir -p "$M"
  python3 - "$M" "$@" <<'PY' || exit 1
import json, os, sys
M, runs = sys.argv[1], [r for r in sys.argv[2:] if os.path.exists(f"{r}/onpol_sim.pkl")]
json.dump({"RSG-L18-s0@" + "/".join(r.rstrip("/").split("/")[-4:-1]): [r] for r in runs},
          open(f"{M}/manifest.json", "w"), indent=1)
print(len(runs), "runs with the simulator state of", len(sys.argv) - 2)
PY
  $NAV "$ON/scripts/on_rows.py" geo --name "$G" --roll "$M" --workers 32 || die "on_rows geo $G"
}
post_cpu() {  # <NAME> <EV_CFG run dir>: K / C labels, route memory features (the runs' own long-cache config), longitudinal pack
  $NAV "$repo/experiments/onpol_long/scripts/ol_labels.py" build --name "$1" --workers 32 || die "ol_labels build"
  EV_CFG=$2/generated-user-config-0.yaml $APY "$repo/experiments/route_mem/scripts/rm_build.py" pool --name "$1" --workers 32 || die "rm_build pool"
  $TR "$ON/scripts/on_lon_pack.py" pack --name "$1" --items B P || die "on_lon_pack"
}

case $cmd in
nsim)   # <run dir> <list | file@i/n> <cache> id | xp=<from file> | pt=<perturb file> <dump 0|1> [pack]: one 30-decision launch, resumable
  R=$1 L=$2 CA=$3 M=$4 DU=$5 NP=${6:-}
  if [[ $L == *@*/* ]]; then
    f=${L%@*} ij=${L##*@}; L=$(dirname "$R")/list.txt; mkdir -p "$(dirname "$R")"
    awk -v i=${ij%/*} -v n=${ij#*/} '!/^#/ && NF && (c++ % n) == i {print $1}' "$f" > "$L"
  fi
  if [[ ! -s $L ]]; then mkdir -p "$R"; echo "empty list" > "$R/EMPTY"; exit 0; fi
  [[ -f $R/aggregate/results-summary.json && ( $DU == 0 || -f $R/onpol_sim.pkl ) ]] && { echo "$R done"; exit 0; }
  case $M in id) E=(-u JEV_EXPERT -u JEV_EXPERT_FROM -u JEV_PERTURB) ;; xp=*) E=(-u JEV_PERTURB JEV_EXPERT=$NP JEV_EXPERT_FROM=${M#xp=}) ;;
             pt=*) E=(-u JEV_EXPERT -u JEV_EXPERT_FROM JEV_PERTURB=${M#pt=}) ;; *) die "mode $M" ;; esac
  rm -rf "$R"; mkdir -p "$R"
  B=$(cd "$repo" && python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print(" ".join(B.OVERRIDES))')
  (cd "$repo" && env "${E[@]}" SH30_TAG=RSG-L18-s0 SH30_DUMP=$([[ $DU == 1 ]] && echo 1000000 || echo 0) bash experiments/alpasim/scripts/run.sh "$R" sh30 \
     --scene-list "$L" $B runtime.scene_provider.trajdata.cache_location=$CA runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=$AF) || die "run.sh"
  [[ -f $R/aggregate/results-summary.json ]] || die "no results-summary"
  if [[ $DU == 1 ]]; then $APY "$repo/experiments/alpasim/scripts/c1_extract.py" logs --run "$R" --out "$R/onpol_sim.pkl" --jobs 6 || die "c1_extract"; fi
  find "$R/rollouts" -name rollout.asl -delete
  echo "$(date '+%F %T') $R: $(ls "$R/driver-logs/dump" 2>/dev/null | wc -l) dumps, $(du -sh "$R" | cut -f1)" ;;
rowsR)  # <stage dir> <NAME> [prior read]: n-read + rewind check, geo of D2 / D3, unit / row selection, pool, labels
  S=$1 N=$2 P=${3:-}
  $APY "$here/rf_rewind.py" n-read --stage "$S" --ref "$S/tfail.json" --out "$S/read.json" --pkl "$S/pkl" --workers 32 || die "n-read"
  $PY "$X" rcheck --stage "$S" ${P:+--prior "$P"} --prior-runs "$(cat "$S/id_stages.txt")" || die "rcheck"
  for d in 2 3; do geo "$S/geo/D$d" "${N}_D${d}g" "$S"/D$d/c*/run; done
  $APY "$X" rsel --stage "$S" --geo "D2=${N}_D2g,D3=${N}_D3g" --out "$S/sel.json" ${RR_ROWS_CSV:+--rows-csv "$RR_ROWS_CSV" --check "$S/check.json"} || die "rsel"
  $TR "$X" assemble --name "$N" --sel "$S/sel.json" || die "assemble"
  post_cpu "$N" "$(ls -d "$S"/D2/c*/run | head -1)" ;;
rowsF)  # <rows dir> <NAME> <stage,stage..> <levels>: geo per level over the stages' pushed runs, rows k >= k_p, pool, labels
  S=$1 N=$2 STG=$3 LV=$4; mkdir -p "$S"; g=() rd=()
  for s in ${LV//,/ }; do l=L$(python3 -c "print(round(float('$s') * 100))"); rr=()
    for st in ${STG//,/ }; do rr+=("$st"/$l/c*/run); done
    geo "$S/geo/$l" "${N}_${l}g" "${rr[@]}"; g+=("$l=${N}_${l}g")
  done
  for st in ${STG//,/ }; do rd+=("$st/fread.json"); done
  $PY "$X" frows --reads "$(IFS=,; echo "${rd[*]}")" --sel "$O/sel.json" --geo "$(IFS=,; echo "${g[*]}")" --n ${NPUSH:-2} --out "$S/sel.json" || die "frows"
  $TR "$X" assemble --name "$N" --sel "$S/sel.json" || die "assemble"
  post_cpu "$N" "$(ls -d "${STG%%,*}"/L*/c*/run | head -1)" ;;
tok)    # <NAME>: tokens + teacher + gate R1 (on_rows.py tok; the interpreted control smoke10eY as every L18 pool)
  $TR "$ON/scripts/on_rows.py" tok --name "$1" --control smoke10eY || die "on_rows tok" ;;
rowsB)  # <stage dir> <NAME> <run dirs..>: pool gates (ol_labels gate: R1 / R2 / K0), strip the dumps, report
  S=$1 N=$2; shift 2
  $TR "$repo/experiments/onpol_long/scripts/ol_labels.py" gate --name "$N" --out "$ROWS/$N"; echo "ol_labels gate rc $?"
  [[ -f $D/runs/op_parity/cache/on1_$N@warp/front.npy ]] || die "no tokens"
  $PY "$X" strip --runs "$@" || die "strip"
  $PY "$X" report --name "$N" --out "$S" || die "report"
  date > "$S/DONE" ;;
fprep)  # <stage dir> <count> <levels>
  [[ -s $O/sel.json ]] || $TR "$X" fsel --pool L18RSG --turns "$T64/scenes.txt" --out "$O/sel.json" || die "fsel"
  $PY "$X" fprep --sel "$O/sel.json" --stage "$1" --count "$2" --levels "$3" --n ${NPUSH:-2} || die "fprep" ;;
fread)  # <stage dir>
  $APY "$X" fread --stage "$1" --sel "$O/sel.json" --out "$1/fread.json" || die "fread"
  date > "$1/DONE" ;;
rmerge) # <NAME> <pool:prefix,..>
  N=$1; ps=(); for x in ${2//,/ }; do ps+=("on1_$x"); done
  [[ ! -e $D/runs/op_parity/cache/on1_$N/tab.npz ]] || die "on1_$N exists"
  $TR "$here/rf_pait.py" merge --pools "${ps[@]}" --out "$N" || die "merge"
  $TR "$X" report --name "$N" --out "$O" || die "report" ;;
srcall) # <rewind_onpol_all dir> <out dir>
  A=$1 OUT=$2; mkdir -p "$OUT/pack"; cp "$A"/all/from_D2.txt "$A"/all/from_D3.txt "$OUT/"
  python3 - "$A" "$OUT" <<'PY' || die "srcall"
import glob, json, os, sys
A, OUT = sys.argv[1:]
U, gate = {}, None
for s in ("smoke", "s50", "rest", "restb"):
    t = json.load(open(f"{A}/{s}/tfail.json"))
    U.update(t["units"]); gate = gate or t["gate"]
json.dump(dict(gate=gate, units=U), open(f"{OUT}/tfail.json", "w"), indent=1)
n = 0
for s in ("smoke", "s50", "rest"):
    for f in glob.glob(f"{A}/{s}/pack/*.npz"):
        d = f"{OUT}/pack/{os.path.basename(f)}"
        if not os.path.lexists(d):
            os.symlink(f, d); n += 1
need = {l.split()[0] for d in (2, 3) for l in open(f"{OUT}/from_D{d}.txt") if l.strip()}
miss = [x for x in need if not os.path.exists(f"{OUT}/pack/{x}.npz")]
print(len(U), "units,", n, "packs,", len(need), "handover scenes, missing packs", len(miss)); sys.exit(len(miss) > 0)
PY
  ;;
recov)
  N=${1:?NAME} FD=${2:?from dir} IDS=${3:?id stages} NP=${4:?pack} CA=${5:?cache} n=${6:-2} PRIOR=${7:-}; S=$O/$N
  [[ ! -e $S/SUBMITTED ]] || die "$S already submitted"
  [[ -s $FD/tfail.json && -s $FD/from_D2.txt && -s $FD/from_D3.txt && -d $NP && ( -d $CA || -n ${AFTER:-} ) ]] || die "source incomplete"
  mkdir -p "$S"; cp "$FD"/{tfail.json,from_D2.txt,from_D3.txt} "$S/"; code "$S"
  echo "$IDS" > "$S/id_stages.txt"
  i=0; for st in ${IDS//,/ }; do for r in "$st"/id/c*/run; do [[ -d $r ]] || continue; mkdir -p "$S/id/c$i"; ln -sfn "$r" "$S/id/c$i/run"; i=$((i + 1)); done; done
  XS="bash $S/code/experiments/route_follow/scripts/rf_recov_roll.sh"; sims=()
  G=(--vram ${RR_VRAM:-20} --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 --env "JEV_PY=$PY" ${AFTER:+--after $AFTER})
  for d in 2 3; do for c in $(seq 0 $((n - 1))); do
    sims+=($(nsub rcv-$N-D$d-c$c "${G[@]}" --log-dir "$S/pool/D$d-c$c" -- $XS nsim "$O" "$S/D$d/c$c/run" "$S/from_D$d.txt@$c/$n" "$CA" "xp=$S/from_D$d.txt" 1 "$NP")) || exit 1
  done; done
  E=(--env "JEV_PY=$PY"); [[ -n ${RF_K_END:-} ]] && E+=(--env RF_K_END=$RF_K_END); [[ -n ${RR_ROWS_CSV:-} ]] && E+=(--env "RR_ROWS_CSV=$RR_ROWS_CSV")
  a=$(nsub rcv-$N-rowsA --after "$(IFS=,; echo "${sims[*]}")" "${E[@]}" --vram 0.5 --cpu 32 --ram 96 --timeout-h 2 --log-dir "$S/pool/rowsA" -- $XS rowsR "$O" "$S" "$N" "$PRIOR") || exit 1
  t=$(nsub rcv-$N-tok --after "$a" --vram 16 --cpu 16 --ram 64 --timeout-h 2 --log-dir "$S/pool/tok" -- $XS tok "$O" "$N") || exit 1
  nsub rcv-$N-rowsB --after "$t" --env "JEV_PY=$PY" --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 --log-dir "$S/pool/rowsB" -- \
    bash -c "$XS rowsB $O $S $N $S/D2/c*/run $S/D3/c*/run" > /dev/null || exit 1
  date > "$S/SUBMITTED"; tail -$(( ${#sims[@]} + 3 )) "$O/jobs.txt" ;;
fast)
  sub=${1:?stage|rows}; shift
  case $sub in
  stage)
    name=${1:?name} cnt=${2:?count} LV=${3:?levels} n=${4:-1}; S=$O/$name; CA=${RR_CACHE:-$D/datasets/alpasim_nuplan_long/after280_op}
    [[ ! -e $S/SUBMITTED ]] || die "$S already submitted"
    mkdir -p "$S"; code "$S"; XS="bash $S/code/experiments/route_follow/scripts/rf_recov_roll.sh"; sims=()
    p=$(nsub rcf-$name-prep ${AFTER:+--after $AFTER} --env "JEV_PY=$PY" --env NPUSH=${NPUSH:-2} --vram 0.5 --cpu 8 --ram 48 --timeout-h 1 \
        --log-dir "$S/pool/prep" -- $XS fprep "$O" "$S" "$cnt" "$LV") || exit 1
    G=(--vram ${RR_VRAM:-20} --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 --env "JEV_PY=$PY" --after "$p")
    for c in $(seq 0 $((n - 1))); do
      sims+=($(nsub rcf-$name-id-c$c "${G[@]}" --log-dir "$S/pool/id-c$c" -- $XS nsim "$O" "$S/id/c$c/run" "$S/list.txt@$c/$n" "$CA" id 0)) || exit 1
      for s in ${LV//,/ }; do l=L$(python3 -c "print(round(float('$s') * 100))")
        sims+=($(nsub rcf-$name-$l-c$c "${G[@]}" --log-dir "$S/pool/$l-c$c" -- $XS nsim "$O" "$S/$l/c$c/run" "$S/list.txt@$c/$n" "$CA" "pt=$S/pt_$l.txt" 1)) || exit 1
      done
    done
    nsub rcf-$name-read --after "$(IFS=,; echo "${sims[*]}")" --env "JEV_PY=$PY" --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 --log-dir "$S/pool/read" -- \
      $XS fread "$O" "$S" > /dev/null || exit 1
    date > "$S/SUBMITTED"; tail -$(( ${#sims[@]} + 2 )) "$O/jobs.txt" ;;
  rows)
    N=${1:?NAME} STG=${2:?stages} LV=${3:?levels}; S=$O/rows_$N; ss=()
    [[ ! -e $S/SUBMITTED ]] || die "$S already submitted"
    mkdir -p "$S"; code "$S"; XS="bash $S/code/experiments/route_follow/scripts/rf_recov_roll.sh"
    for st in ${STG//,/ }; do ss+=("$O/$st"); done; SD=$(IFS=,; echo "${ss[*]}")
    a=$(nsub rcf-$N-rowsA ${AFTER:+--after $AFTER} --env "JEV_PY=$PY" --env NPUSH=${NPUSH:-2} --vram 0.5 --cpu 32 --ram 96 --timeout-h 2 \
        --log-dir "$S/pool/rowsA" -- $XS rowsF "$O" "$S" "$N" "$SD" "$LV") || exit 1
    t=$(nsub rcf-$N-tok --after "$a" --vram 16 --cpu 16 --ram 64 --timeout-h 2 --log-dir "$S/pool/tok" -- $XS tok "$O" "$N") || exit 1
    runs=""; for st in "${ss[@]}"; do runs="$runs $st/L*/c*/run"; done
    nsub rcf-$N-rowsB --after "$t" --env "JEV_PY=$PY" --vram 0.5 --cpu 16 --ram 48 --timeout-h 1 --log-dir "$S/pool/rowsB" -- \
      bash -c "$XS rowsB $O $S $N $runs" > /dev/null || exit 1
    date > "$S/SUBMITTED"; tail -3 "$O/jobs.txt" ;;
  *) die "fast stage|rows" ;;
  esac ;;
*) sed -n '2,28p' "$0"; exit 2 ;;
esac
