#!/usr/bin/env bash
# NUPLAN-DAG (plans/2026-10-22-nuplan-dag.md): the PAI DAgger map-teacher method moved to nuPlan (AlpaSim nuPlan closed loop).
# Part 1: PAI checkpoints (R8K-L18-s2, R6DAG-L18-s1) on nuPlan held 460 @18 and the arena held split (native serve, B2's adoption reads),
# paired against the stored B2 (STANDFMXH-L18-s1) and NTV0-L18-s0 reads.
# Part 2: student B2 rolled natively (sh30, no hold, frame dumps) on the 1031 onpol-train scenes (18 decisions, long cache) -> rows (on_rows geo /
# tok, as onpol_long / practice_nuplan) -> the map teacher (nd_teach.py) -> pool on1_NDAG (fut := teacher label on drawable rows) -> on1_JTXD =
# on1_JTX + on1_NDAG (mode both) -> draw NDAG (B2's JTX00 x 70 / 78 + 8 DAgger rows a batch) -> T3 identity + smoke -> NDAG-L18-s0 / s1 (B2's argv,
# only pool / draw / on-k / batch / yr-mass / tag / seed changed) -> arena + held 460 reads -> p2sum.
# Box, repo root; every job through the GPU pool at PRIO 22; ids in $O/jobs.txt, stages in $O/stages.txt, a stop reason in $O/VERDICT.
#   nd.sh p1 [tags]       Part 1 reads + p1sum
#   nd.sh probe           c000 (6 scenes): roll -> geo / tok (R1 / R2) -> teach -> label (T2 / T4) -> plots; T1 on all 1031 logs; chk1 -> rest
#   nd.sh dry             the training / identity commands against B2's logged argv
#   nd.sh wait <marker>   blocks until $O/<marker>, $O/VERDICT or a failed / cancelled job of this lane (one run_in_background waiter)
#   job bodies: p1sum | lists | roll <c> | geo <N> <chunks..> | tok <N> | route <N> | anchor <N> | strip <N> | teach <N> | label <N> [write] |
#               plots <N> | t1all | t1probe | chk1 | rest | append | weights | smoke | identrun <a|b> | ident | build | train <seed> | p2sum
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/root/autodl-tmp/ujs}; export DATA_DIR
O=${ND_O:-$DATA_DIR/runs/nuplan_dag}; ME=experiments/nuplan_dag/scripts/nd.sh; PY=experiments/nuplan_dag/scripts/nd.py; V=.venv/bin/python
TR=$DATA_DIR/envs/op-train/bin/python; AP=$DATA_DIR/third_party/alpasim/.venv/bin/python; RES=${ND_RES:-$O/res}   # lane outputs outside the repo tree (copied into experiments/nuplan_dag/results at report time)
JT=experiments/defect_fix/scripts/jt.sh; FM=experiments/defect_fix/scripts/free_m.py; J=$DATA_DIR/runs/defect_fix/jt; AH=$DATA_DIR/runs/alpasim_proxy/arena/hold
B2=STANDFMXH-L18-s1; NTV0=NTV0-L18-s0; P1TAGS="R8K-L18-s2 R6DAG-L18-s1"; PRIO=${PRIO:-22}
RD=(--vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4)          # nuPlan reads: 4 a card (alpasim-read4)
mkdir -p $O/pool $RES
die() { echo "$(date '+%F %T') ERROR $*" | tee -a $O/stages.txt >&2; exit 1; }
sub() { local name=$1; shift; local id; id=$($V -m jevdrive.cl submit --owner nuplan-dag --name "$name" --tries ${TRIES:-1} --priority $PRIO --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || die "submit $name failed"; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
csv() { local IFS=,; echo "$*"; }
# ---- Part 2
NT=experiments/nuplan_dag/scripts/nd_teach.py; ONR=experiments/onpol_nuplan/scripts/on_rows.py; RB=experiments/route_mem/scripts/rm_build.py
AS=$DATA_DIR/third_party/alpasim; NAV=$DATA_DIR/envs/navsim2/bin/python; CK=$DATA_DIR/runs/op_parity/runs; CR=$DATA_DIR/runs/op_parity/cache
LC=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata; SPL=experiments/onpol_nuplan/results/split
EA=(env JEV_REPO=$PWD PYTHONPATH=$AS/src/utils:$PWD)
B2S0=STANDFMXH-L18-s0; NP=NDP; NM=JTXD; NSTEP=9500
PARTS="on1_NDP $(for i in $(seq 1 24); do printf 'on1_ND%03d ' $i; done)"          # amendment 3: the chunk pools in merge order (c000 = the probe pool)
KDAG=$(cat $RES/kdag.txt 2>/dev/null || echo 0)                                   # amendment 1: k_dag from the drawable count (weights stage)
RO=(--vram 32 --cpu 8 --ram 40 --timeout-h 3 --env CL_HIST_KEY=nd-roll)
FT=(--train --vram 40 --cpu 6 --ram 30 --env CL_HIST_KEY=rm-train)
T1CFG="$DATA_DIR/runs/onpol_long/roll/RB0-F-s0/c*/run/generated-user-config-0.yaml"
argv() {  # argv <variant> <steps> <tag> [seed]: one argument a line. b2 = B2's logged argv; d0 = b2 on on1_JTXD / NDAG0 (DAgger rows at weight 0: T3);
          # dag = b2 on on1_JTXD / NDAG, 8 DAgger rows a batch (--on-k 78 --batch 136 --yr-mass 0.095: 45 normal + 13 yaw-rate + 70 pool rows unchanged)
  local M; M=$(ls -d $DATA_DIR/runs/route_mem/train-$B2/*/ | tail -1)meta.json; [[ -f $M ]] || die "no $M"
  ND_KDAG=$KDAG python3 - "$M" "$@" <<'PY'
import json, os, sys
m, var, steps, tag = sys.argv[1:5]
seed = (sys.argv[5:] or [None])[0]
a = json.load(open(m))["argv"]
last = lambda k: len(a) - 1 - a[::-1].index(k)  # noqa: E731
assert a[a.index("--init") + 1] == "RB0-F-s0" and a[a.index("--tag") + 1] == "STANDFMXH-L18-s1" and a[last("--on-wfile") + 1] == "runs/defect_fix/w/JTX00.npz"
assert a[last("--on") + 1] == "on1_JTX" and a[last("--on-k") + 1] == "70" and a[last("--seed") + 1] == "1" and "--batch" not in a and a.count("--yr-mass") == 1
assert a[last("--steps") + 1] == "9500" and "--save-at" not in a
a[a.index("--tag") + 1] = tag
a[last("--steps") + 1] = steps
if var in ("d0", "dag"):
    a[last("--on") + 1] = "on1_JTXD"
    a[last("--on-wfile") + 1] = f"runs/nuplan_dag/w/{'NDAG0' if var == 'd0' else 'NDAG'}.npz"
if var == "dag":                                                   # amendment 1: k = k_dag rows a batch, 45 normal + 13 yaw-rate + 70 pool unchanged
    k = int(os.environ["ND_KDAG"]); assert 1 <= k <= 8, f"k_dag {k}"
    a[last("--on-k") + 1] = str(70 + k)
    a[last("--yr-mass") + 1] = f"{13 / (128 + k):.4f}"
    assert round((128 + k) * float(a[last("--yr-mass") + 1])) == 13
    a += ["--batch", str(128 + k)]
if seed is not None:
    a[last("--seed") + 1] = seed
print("\n".join(a))
PY
}
run() { local X; X=$(argv "$@") || die "argv $*"; mapfile -t A <<< "$X"; }
roll() {  # roll <chunk> [after] -> job id: B2 native (sh30, JEV_HOLD unset) on the chunk, frame dumps, simulator-state extract, prune
  TRIES=2 sub nd-roll-$1 ${2:+--after $2} "${RO[@]}" -- bash $ME rolljob $1; }
case ${1:-} in
p1)
  T=(${2:-$P1TAGS}); ids=(); nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))")
  for t in ${T[@]}; do [[ -f $DATA_DIR/runs/op_parity/runs/$t/ckpt-final.pt ]] || die "no $t"
    for c in $(seq 0 $((nch - 1))); do ids+=($(TRIES=2 sub nd-ar-$t-c$c "${RD[@]}" -- bash $JT aread $t off $c)) || exit 1; done
    for i in 0 1 2 3 4 5; do ids+=($(TRIES=2 sub nd-held-$t-$i "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $t $i 6)) || exit 1; done; done
  s=$(sub nd-p1sum --after $(csv ${ids[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 4 -- bash $ME p1sum "${T[*]}") || exit 1
  echo "$(date '+%F %T') p1 (code $(git rev-parse --short HEAD)): ${T[*]}: reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}) p1sum $s" | tee -a $O/stages.txt ;;
p1quick) s=$(sub nd-p1quick --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash -c "$TR $PY quick --tags $P1TAGS --refs $B2 $NTV0 --out $RES/p1_quick.md --title 'Part 1: PAI checkpoints on nuPlan' && date > $O/P1QUICK") || exit 1
  echo "$(date '+%F %T') p1quick $s" | tee -a $O/stages.txt ;;
p1resum) s=$(sub nd-p1sum --vram 0.1 --cpu 32 --ram 96 --timeout-h 4 -- bash $ME p1sum) || exit 1; echo "$(date '+%F %T') p1sum resubmitted $s" | tee -a $O/stages.txt ;;
p1sum)  # the arena extract (native_gap rule) of the PAI tags + B2 + NTV0 -> paired arena / held 460 tables, zero classes, lateral-zero split
  T=(${2:-$P1TAGS}); X=$O/p1/ext; mkdir -p $O/p1; rd=(); for t in ${T[@]} $B2 $NTV0; do rd+=($t=$J/arena/$t@off); done
  $TR $PY quick --tags ${T[@]} --refs $B2 $NTV0 --out $RES/p1_quick.md --title "Part 1: PAI checkpoints on nuPlan" || die "quick"
  date '+%F %T' > $O/P1QUICK
  $AP $FM ext --read ${rd[@]} --out $X --workers 32 > $O/p1/ext.log 2>&1 || die "extract (log $O/p1/ext.log)"
  for t in ${T[@]} $NTV0; do $V $FM arena --src $X --a $t --b $B2 --ref $NTV0 --out $O/p1/arena_$t.json > /dev/null || die "arena $t"; done
  $V -c "import pandas as pd; w = pd.read_parquet('$X/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/p1_arena_windows.csv.gz', index=False)" || die windows
  $TR $PY held --tags ${T[@]} $NTV0 --refs $B2 $NTV0 --out $O/p1/held.json || die "held"
  $AP $PY latsplit --ext $X --tags ${T[@]} $B2 $NTV0 --out $O/p1/latsplit.csv || die "latsplit"
  $V $PY p1md --root $O/p1 --tags ${T[@]} --b2 $B2 --ntv0 $NTV0 --out $RES/p1.md || die "p1md"
  date '+%F %T' > $O/P1DONE ;;
probe)   # c000 (6 scenes) -> rows NDP -> teacher -> T2 / T4 / plots; T1 on the 1031 train logs; chk1 -> rest (the full chain) on pass
  [[ ! -e $O/roll/c000 ]] || die "$O/roll/c000 exists: never overwritten"; [[ -f $CK/$B2/ckpt-final.pt ]] || die "no $B2"
  "${EA[@]}" $AP $NT lists --out $O/lists --n 24 || die lists
  h=$(git rev-parse --short HEAD); echo "$h probe" >> $O/code.txt
  r=$(roll c000) || exit 1
  g=$(sub nd-geo-$NP --after $r --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME geo $NP c000) || exit 1
  k=$(sub nd-tok-$NP --after $g --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $ME tok $NP) || exit 1
  t=$(sub nd-teach-$NP --after $g --vram 0.1 --cpu 8 --ram 32 --timeout-h 1 -- bash $ME teach $NP) || exit 1
  l=$(sub nd-label-$NP --after $k,$t --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME label $NP) || exit 1
  p=$(sub nd-plots-$NP --after $l --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME plots $NP) || exit 1
  t1=$(sub nd-t1all --vram 0.1 --cpu 32 --ram 96 --timeout-h 2 -- bash $ME t1all) || exit 1
  t1p=$(sub nd-t1probe --after $r --vram 0.1 --cpu 6 --ram 16 --timeout-h 1 -- bash $ME t1probe) || exit 1
  c=$(sub nd-chk1 --after $l,$p,$t1,$t1p --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME chk1) || exit 1
  echo "$(date '+%F %T') probe (code $h): roll $r geo $g tok $k teach $t label $l plots $p t1all $t1 t1probe $t1p chk1 $c" | tee -a $O/stages.txt ;;
lists) "${EA[@]}" $AP $NT lists --out $O/lists --n 24 ;;
rolljob)  # rolljob <chunk>: ol_fast.sh roll's body for the student B2 natively (JEV_HOLD unset = B2's adoption serve)
  c=$2; D=$O/roll/$c/run; L=$O/lists/$c.txt
  [[ -z $(comm -12 <(sort $L) <(sort $SPL/held.txt)) ]] || die "$L has a held scene"; [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
  rm -rf $D; mkdir -p $D
  env -u JEV_HOLD SH30_TAG=$B2 SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh $D sh30 --scene-list $L "${BASE[@]}" "$KEY.cache_location=$LC" \
    "$KEY.dataset.extra_params.num_timesteps_after=160" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  $AP experiments/alpasim/scripts/c1_extract.py logs --run $D --out $D/onpol_sim.pkl --jobs 8 || die "c1_extract rc $?"
  python3 - $D <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason"):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "$(date '+%F %T') roll $c: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1), free $(df -h $DATA_DIR | awk 'NR == 2 {print $4}')" | tee -a $O/stages.txt ;;
geo)      # geo <pool> <chunks..>: on_rows.py geo (unedited, default split onpol-train) over the chunks' runs
  N=$2; shift 2; R=$O/stage/$N; mkdir -p $R
  python3 - $R $O/roll $B2 "$@" <<'PY' || die "roll chunks missing ($N)"
import json, os, sys
R, base, T, cs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
m = {f"{T}@{c}": [f"{base}/{c}/run"] for c in cs if os.path.exists(f"{base}/{c}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", len(cs), "chunks"); sys.exit(len(m) != len(cs))
PY
  $NAV $ONR geo --name $N --roll $R --workers ${ND_GEO_WORKERS:-24} || die "on_rows geo rc $?" ;;
tokgate)  # tokgate <pool>: the R1 / R2 check of tok alone (after an amendment)
  N=$2; TOKGATE_ONLY=1 bash $ME tok $N ;;
probe2)   # amendment 2: label -> plots -> chk1 (-> rest) after the tok gate re-checked (teach / tok outputs kept)
  g=$(sub nd-tokgate-$NP --vram 0.1 --cpu 1 --ram 2 --timeout-h 1 -- bash $ME tokgate $NP) || exit 1
  l=$(sub nd-label-$NP --after $g --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME label $NP) || exit 1
  p=$(sub nd-plots-$NP --after $l --vram 0.1 --cpu 2 --ram 16 --timeout-h 1 -- bash $ME plots $NP) || exit 1
  c=$(sub nd-chk1 --after $l,$p --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME chk1) || exit 1
  echo "$(date '+%F %T') probe2 (code $(git rev-parse --short HEAD)): tokgate $g label $l plots $p chk1 $c" | tee -a $O/stages.txt ;;
tok)      # tok <pool>: tokens + teacher + gate R1 (control smoke10e: the RB0 lineage, as practice_nuplan), R1 / R2 as pn.sh (amended lines)
  N=$2; [[ -n ${TOKGATE_ONLY:-} ]] || $TR $ONR tok --name $N --control smoke10e || die "on_rows tok rc $?"
  python3 - $DATA_DIR/runs/onpol_nuplan/rows/$N <<'PY' || die "pool gates R1 / R2 failed ($N)"
import json, sys
d = sys.argv[1]; r1, r2 = json.load(open(f"{d}/r1.json")), json.load(open(f"{d}/r2.json"))
ok2 = r2["ego_max_m"] <= 0.3 and r2["frame_err_max_m"] <= 0.1 and r2["act_share_within"] >= 0.995 and r2["act_median_m"] <= 0.05
q = next(iter(r1["tags"].values()))
# amendment 2: (a) also met at one fp16 ulp (mu_share_one_ulp >= 0.99, max <= 0.047 m = the largest in the admitted pools PN0 / L18RB0):
# B2's fast plans put 37 % of the probe rows' points beyond 16 m, where one ulp (0.031 m) exceeds the 0.03 m line (L18RB0 in on1_JTX: 0.982)
a1 = r1["a"] or (q["mu_share_one_ulp"] >= 0.99 and q["mu_max"] <= 0.047)
ok1 = a1 and r1["b"] and (r1["c"] or q["control_mu_share"] <= q["mu_share"] - 0.2)
print("R1", r1["a"], a1, r1["b"], r1["c"], ok1, "mu_share", q["mu_share"], "one ulp", q["mu_share_one_ulp"], "max", q["mu_max"], "control", q["control_mu_share"],
      "R2", ok2, "ego_max", r2["ego_max_m"], "act_share", r2["act_share_within"])
sys.exit(0 if ok1 and ok2 else 2)
PY
  ;;
route|anchor)   # rm_build pool / anchor on the pool's rows with a run's config (EV_CFG)
  N=$2; c=$(python3 -c "import json; print(json.load(open('$O/stage/$N/manifest.json')).popitem()[1][0])")
  if [[ $1 == route ]]; then EV_CFG=$c/generated-user-config-0.yaml $AP $RB pool --name $N --workers 8 || die "rm_build pool rc $?"
  else EV_CFG=$c/generated-user-config-0.yaml $AP $RB anchor --pool $N --workers 8 || die "rm_build anchor rc $?"; fi ;;
strip)    # drop the slot frames of the pool's dumps after the tokens (pn.sh strip)
  N=$2; [[ -f $CR/on1_$N@warp/front.npy ]] || die "no tokens for $N"
  for d in $(python3 -c "import json; print(' '.join(v[0] for v in json.load(open('$O/stage/$N/manifest.json')).values()))"); do
  $TR - "$d" <<'PY' || die "strip rc $?"
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
def one(f):
    z = dict(np.load(f))
    if "cur" not in z:
        return 0
    del z["cur"]
    np.savez_compressed(f + ".tmp.npz", **z)
    os.replace(f + ".tmp.npz", f)
    return 1
fs = sorted(glob.glob(f"{sys.argv[1]}/driver-logs/dump/*.npz"))
with ProcessPoolExecutor(16) as ex:
    print(sys.argv[1], "stripped", sum(ex.map(one, fs, chunksize=64)), "of", len(fs))
PY
  done ;;
teach) mkdir -p $O/teach; "${EA[@]}" $AP $NT teach --name $2 --out $O/teach/$2.npz --workers ${ND_TEACH_WORKERS:-8} ;;
label) $TR $NT label --name $2 --teach $O/teach/$2.npz --res $RES ${3:+--write} ;;
plots) $TR $NT plots --name $2 --teach $O/teach/$2.npz --out $RES/figs/probe ;;
t1all) "${EA[@]}" $AP $NT t1 --cfg-glob "$T1CFG" --out-dir $RES/t1 --workers 32 ;;
t1probe) "${EA[@]}" $AP $NT t1 --cfg-glob "$O/roll/c000/run/generated-user-config-0.yaml" --out-dir $RES/t1_probe --workers 6 ;;
chk1)
  python3 - $RES <<'PY' || die "chk1 inputs"
import json, shutil, sys
r = sys.argv[1]
shutil.copy(f"{r}/t1/t1.json", f"{r}/t1_all.json")
PY
  $TR - $RES $O <<'PY' || { echo "STOP probe gates (see $RES/chk1.json)" > $O/VERDICT; exit 1; }
import json, sys
from pathlib import Path
R, O = Path(sys.argv[1]), Path(sys.argv[2])
lab = json.loads((R / "label_NDP.json").read_text())
t1 = json.loads((R / "t1" / "t1.json").read_text())
rs = json.loads((O / "roll/c000/run/aggregate/results-summary.json").read_text())["rollouts"]
scored = sum(1 for r in rs if r.get("score") is not None and not r.get("failure_reason"))
fe = lab["frame_err_k0"]
ck = dict(scored=(scored, ">= 5 / 6", scored >= 5), rows=(lab["rows"], ">= 60", lab["rows"] >= 60),
          teacher_share=(lab["candidates"] / max(lab["fed_ge2"], 1), ">= 0.80 of k >= 2, v >= 2 m/s", lab["candidates"] >= 0.8 * lab["fed_ge2"]),
          frame=(fe, "k = 0 anchor vs log p99 <= 0.10 m", fe is not None and fe[2] <= 0.10),
          T1=(t1["table"].get("all but lane-change"), "teacher median 2 s <= 0.30, 4 s <= 0.50 m (1031 train logs)", t1["passed"]),
          T2=(lab["T2"]["teacher_inside"], ">= 0.90", lab["T2"]["passed"]), T4=(lab["T4"]["dropped_share"], "<= 0.10 dropped", lab["T4"]["passed"]))
ok = all(v[2] for v in ck.values())
(R / "chk1.json").write_text(json.dumps(dict(passed=ok, checks=ck), indent=1, default=str))
print(json.dumps(dict(passed=ok, checks=ck), indent=1, default=str))
sys.exit(0 if ok else 1)
PY
  date '+%F %T' > $O/PROBE_OK ;;      # amendment 3: the full chain is submitted at once (`full`); its merge waits for this job
full)     # amendment 3: full <chk1 job id>: the 24 roll chunks now; per chunk its rows pipeline (geo -> tok -> teach -> label, route -> anchor,
          # strip) --after that chunk only, one pool (and staging / rows dir) per chunk: on1_ND001 .. on1_ND024 (+ the probe pool on1_NDP = c000);
          # the pooled gates, append, weights, smoke, identity, build and the trainings after every chunk AND the probe's chk1
  C1=${2:?chk1 job id}; RR=(${ND_ROLLS//,/ })                                     # ND_ROLLS: the 24 roll ids already queued (10-22: the old rest's)
  [[ ${#RR[@]} == 24 || ! -e $O/roll/c001 ]] || die "full already submitted"
  h=$(git rev-parse --short HEAD); echo "$h full" >> $O/code.txt
  ln=$(sub nd-label-$NP-w --after $C1 --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME label $NP write) || exit 1
  ro=$(sub nd-route-$NP --after $C1 --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME route $NP) || exit 1
  an=$(sub nd-anchor-$NP --after $ro --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME anchor $NP) || exit 1
  st=$(sub nd-strip-$NP --after $C1 --vram 0.1 --cpu 16 --ram 16 --timeout-h 2 -- bash $ME strip $NP) || exit 1
  dn=($ln $an); ss=($st)
  for i in $(seq 1 24); do c=$(printf 'c%03d' $i); N=$(printf 'ND%03d' $i)
    if [[ ${#RR[@]} == 24 ]]; then r=${RR[$((i - 1))]}; else r=$(roll $c) || exit 1; fi
    g=$(sub nd-geo-$N --after $r --vram 0.1 --cpu 16 --ram 48 --timeout-h 2 --env ND_GEO_WORKERS=16 -- bash $ME geo $N $c) || exit 1
    k=$(sub nd-tok-$N --after $g --vram 16 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME tok $N) || exit 1
    t=$(sub nd-teach-$N --after $g --vram 0.1 --cpu 8 --ram 32 --timeout-h 2 -- bash $ME teach $N) || exit 1
    l=$(sub nd-label-$N --after $k,$t --vram 0.1 --cpu 4 --ram 32 --timeout-h 1 -- bash $ME label $N write) || exit 1
    ro=$(sub nd-route-$N --after $k --vram 0.1 --cpu 8 --ram 48 --timeout-h 2 -- bash $ME route $N) || exit 1
    an=$(sub nd-anchor-$N --after $ro --vram 0.1 --cpu 8 --ram 80 --timeout-h 2 -- bash $ME anchor $N) || exit 1
    st=$(sub nd-strip-$N --after $k --vram 0.1 --cpu 8 --ram 16 --timeout-h 2 -- bash $ME strip $N) || exit 1
    dn+=($l $an); ss+=($st)
    echo "$(date '+%F %T') full $c -> $N: roll $r geo $g tok $k teach $t label $l route $ro anchor $an strip $st" >> $O/stages.txt; done
  ga=$(sub nd-gates --after $(csv ${dn[@]}),$C1 --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME gates) || exit 1
  ap=$(sub nd-append --after $ga --vram 0.1 --cpu 4 --ram 160 --timeout-h 3 -- bash $ME append) || exit 1
  wt=$(sub nd-weights --after $ap --vram 0.1 --cpu 2 --ram 32 --timeout-h 1 -- bash $ME weights) || exit 1
  sm=$(sub nd-smoke --after $wt "${FT[@]}" --timeout-h 1 -- bash $ME smoke) || exit 1
  ia=$(sub nd-ident-a --after $wt "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun a) || exit 1
  ib=$(sub nd-ident-b --after $wt "${FT[@]}" --env JEV_RM_EAGER=1 --timeout-h 1 -- bash $ME identrun b) || exit 1
  ic=$(sub nd-ident --after $ia,$ib --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME ident) || exit 1
  bd=$(sub nd-build --after $sm,$ic,$(csv ${ss[@]}) --vram 0.1 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME build) || exit 1
  nch=$(python3 -c "import json; print(len(json.load(open('$AH/meta.json'))['chunks']))"); ids=()
  for s in 0 1; do T=NDAG-L18-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || die "$T exists"
    t=$(sub nd-train-s$s --after $bd "${FT[@]}" --timeout-h 6 -- bash $ME train $s) || exit 1
    for c in $(seq 0 $((nch - 1))); do ids+=($(TRIES=2 sub nd-ar-$T-c$c --after $t "${RD[@]}" -- bash $JT aread $T off $c)) || exit 1; done
    for i in 0 1 2 3 4 5; do ids+=($(TRIES=2 sub nd-held-$T-$i --after $t "${RD[@]}" -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1; done; done
  ps=$(sub nd-p2sum --after $(csv ${ids[@]}) --vram 0.1 --cpu 32 --ram 96 --timeout-h 4 -- bash $ME p2sum) || exit 1
  echo "$(date '+%F %T') full (code $h): 24 chunk pipelines (stages.txt lines above); NDP label $ln anchor $an strip $st; gates $ga append $ap weights $wt" \
       "smoke $sm identity $ia $ib $ic build $bd reads ${ids[0]} .. ${ids[-1]} (${#ids[@]}) p2sum $ps" | tee -a $O/stages.txt ;;
gates)    # amendment 3: T2 / T4 pooled over the chunk pools' label files (each chunk's label is written, never stops alone)
  $TR $NT gates --res $RES --pools $PARTS || { echo "STOP T2 / T4 pooled over the chunk pools ($RES/gates.json)" > $O/VERDICT; exit 1; } ;;
append) free=$(df -BG --output=avail $DATA_DIR | tail -1 | tr -dc 0-9); (( free >= 150 )) || die "free $free GB < 150 GB"
  $TR $NT append --base on1_JTX --add "$PARTS" --out on1_$NM --res $RES ;;
weights)  # amendment 1: k_dag = min(8, floor(4.6 x drawable / 9 500)) rows a batch (<= 4.6 expected draws per DAgger row, the JT ceiling)
  $TR $NT weights --pool on1_$NM --base JTX00 --src $(echo $PARTS | tr ' ' ,) --k-on 70 --k-dag 0 --out NDAG0 --res $RES || die NDAG0
  $TR $NT weights --pool on1_$NM --base JTX00 --src $(echo $PARTS | tr ' ' ,) --k-on 70 --k-dag auto --out NDAG --res $RES || die NDAG
  python3 -c "import json; print(json.load(open('$RES/w_NDAG.json'))['k_dag'])" > $RES/kdag.txt; echo "$(date '+%F %T') k_dag $(cat $RES/kdag.txt)" | tee -a $O/stages.txt ;;
smoke)
  X=NDSM-L18-s0; [[ ! -e $CK/$X ]] || die "$X exists"; run dag 60 $X 0
  echo "$(date '+%F %T') smoke: ${A[*]} --diag-every 20" | tee -a $O/stages.txt
  $TR "${A[@]}" --diag-every 20 > $O/smoke.log 2>&1 || die "smoke run (log $O/smoke.log)"
  $TR $NT smoke --tag $X --wfile NDAG --k-dag $(cat $RES/kdag.txt) --res $RES; r=$?; rm -rf $CK/$X; exit $r ;;
identrun)  # identrun a|b: 60 steps eager: a = B2's argv (on1_JTX / JTX00), b = the same on on1_JTXD / NDAG0
  declare -A IX=([a]=NDIDA [b]=NDIDB) VX=([a]=b2 [b]=d0); X=${IX[$2]}-L18-s1; [[ ! -e $CK/$X ]] || die "$X exists"; run ${VX[$2]} 60 $X
  echo "$(date '+%F %T') identity $2: ${A[*]}" | tee -a $O/stages.txt; $TR "${A[@]}" > $O/ident_$2.log 2>&1 || die "identity run $2 (log $O/ident_$2.log)" ;;
ident)
  $TR experiments/defect_fix/scripts/df.py bitcmp --a NDIDA-L18-s1 --b NDIDB-L18-s1 --out nuplan_dag_identity \
      --note "B2 (STANDFMXH-L18-s1) argv 60 steps eager: on1_JTX / JTX00 vs on1_JTXD / NDAG0 (on1_NDAG rows at weight 0)"; r=$?
  mv experiments/defect_fix/results/nuplan_dag_identity.json $RES/identity.json || r=4
  rm -rf $CK/NDIDA-L18-s1 $CK/NDIDB-L18-s1; exit $r ;;
build)
  python3 -c "import json, sys; r = '$RES'; sys.exit(0 if json.load(open(f'{r}/smoke.json'))['pass_'] and json.load(open(f'{r}/identity.json'))['identical'] and json.load(open(f'{r}/gates.json'))['passed'] else 1)" \
    || { echo "STOP smoke / identity (T3) / T2 / T4 on the full pool" > $O/VERDICT; exit 1; }
  date '+%F %T' > $O/BUILT ;;
train) s=$2; [[ -f $O/BUILT && ! -f $O/VERDICT ]] || die "not built"; run dag $NSTEP NDAG-L18-s$s $s
  echo "$(date '+%F %T') train s$s: ${A[*]}" | tee -a $O/stages.txt; exec $TR "${A[@]}" ;;
dry) echo "NDAG s1: $(argv dag $NSTEP NDAG-L18-s1 1 | xargs)"; echo "== B2 -> NDAG s1"; diff <(argv b2 $NSTEP $B2) <(argv dag $NSTEP NDAG-L18-s1 1)
  echo "== B2 -> NDAG s0"; diff <(argv b2 $NSTEP $B2) <(argv dag $NSTEP NDAG-L18-s0 0); echo "== identity a -> b"; diff <(argv b2 60 NDIDA-L18-s1) <(argv d0 60 NDIDB-L18-s1); true ;;
p2sum)  # NDAG s0 / s1 against the same-seed B2 (s0 / s1) and NTV0: arena (extract), held 460, zero classes, lateral split
  T=(NDAG-L18-s0 NDAG-L18-s1); X=$O/p2/ext; mkdir -p $O/p2; rd=(); for t in ${T[@]} $B2S0 $B2 $NTV0; do rd+=($t=$J/arena/$t@off); done
  $TR $PY quick --tags ${T[@]} --refs $B2S0 $B2 $NTV0 --out $RES/p2_quick.md --title "Part 2: NDAG vs B2" || die "quick"
  date '+%F %T' > $O/P2QUICK
  $AP $FM ext --read ${rd[@]} --out $X --workers 32 > $O/p2/ext.log 2>&1 || die "extract (log $O/p2/ext.log)"
  $V $FM arena --src $X --a NDAG-L18-s0 --b $B2S0 --ref $B2 $NTV0 --out $O/p2/arena_NDAG-L18-s0.json > /dev/null || die "arena s0"
  $V $FM arena --src $X --a NDAG-L18-s1 --b $B2 --ref $B2S0 $NTV0 --out $O/p2/arena_NDAG-L18-s1.json > /dev/null || die "arena s1"
  $V -c "import pandas as pd; w = pd.read_parquet('$X/win.parquet'); w[['tag', 'id', 'score', 'type']].to_csv('$RES/p2_arena_windows.csv.gz', index=False)" || die windows
  $TR $PY held --tags ${T[@]} --refs $B2S0 $B2 $NTV0 --out $O/p2/held.json || die "held"
  $AP $PY latsplit --ext $X --tags ${T[@]} $B2S0 $B2 $NTV0 --out $O/p2/latsplit.csv || die "latsplit"
  cp $O/p2/*.json $O/p2/latsplit.csv $RES/ 2>/dev/null; date '+%F %T' > $O/P2DONE ;;
wait)  # returns when $O/<marker> or VERDICT exists or one of this lane's jobs failed / was cancelled
  M=${2:?marker}
  until [[ -f $O/VERDICT || -f $O/$M ]] || ! $V - $O/jobs.txt <<'PY'
import sys
from jevdrive.cl import pool as P
last = {}
for l in open(sys.argv[1]):
    if l.split():
        last[l.split()[1]] = l.split()[0]
jobs = P.snapshot()[0]["jobs"]
bad = [f"{n} {i} {jobs[i]['state']}: {jobs[i].get('why', '')}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
if bad:
    print("failed:", *bad, sep="\n  ", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  do sleep 300; done
  cat $O/VERDICT 2>/dev/null; [[ -f $O/$M ]] && echo "$M $(cat $O/$M)"; tail -3 $O/stages.txt; true ;;
*) sed -n 2,17p "$0"; exit 2 ;;
esac
