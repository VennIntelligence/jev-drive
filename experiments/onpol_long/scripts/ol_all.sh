#!/usr/bin/env bash
# Next round (2026-10-13: nuPlan is training-only, PAI is the exam): on-policy rows of one checkpoint from ALL 1491 public nuPlan scenes at 18
# decisions. The train 1031 are ol_chain's own rollouts and pool (roll/<T>, rows <A>; reused: their dumps are already stripped of the slot
# frames, so they cannot be re-tokenized); this script rolls the 460 onpol-held scenes with the same driver path (ol_chain roll1's body, held
# lists), builds their rows / route labels, and merges both pools. Rows record alpasim_nuplan/onpol-all (onpol-train + onpol-held): on_rows.py
# runs unedited except its SPLIT constant (its geo asserts every rollout is in SPLIT). Box, repo root:
#   bash experiments/onpol_long/scripts/ol_all.sh submit <T> <train rows> <OUT> [control]   -> 6 held rollouts (cl fanout), held rows <OUT>held
#     (geo -> tok -> strip; tok -> route pool), merged with the train pool into on1_<OUT> (ol_pool_merge.py), gates
# Job bodies (run by the pool): roll1 <T> <chunk>; geo <T> <N>; tok <N> <control>; pool <N>; merge <OUT> <A> <B>; gate <pool> <rows names>.
# Strip is ol_chain's. State: held rollouts $W/roll/<T>-held/c<i>/run, lists $W/lists/held6; rows / pools as ol_chain (rows/<N>, cache on1_<N>,
# route_mem/feat/on1_<N>); the merged pool has no rows dir and no K / C labels (S / route recipes only).
# Trainer flags for the merged pool: --on on1_<OUT> --on-split alpasim_nuplan/onpol-all (the trainers only record the split).
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/onpol_long; ON=experiments/onpol_nuplan; SPL=$ON/results/split; W=$DATA_DIR/runs/onpol_long; CK=$DATA_DIR/runs/op_parity/runs
C=$DATA_DIR/datasets/alpasim_nuplan_long/after160; KEY=runtime.scene_provider.trajdata; NCH=6; ALL=alpasim_nuplan/onpol-all
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; NAV=$DATA_DIR/envs/navsim2/bin/python; TR=$DATA_DIR/envs/op-train/bin/python
CL() { .venv/bin/python -m jevdrive.cl "$@"; }
die() { echo "$(date '+%F %T') ERROR $*" >&2; exit 1; }
source <(python3 -c 'import sys; sys.path.insert(0, "experiments/alpasim/scripts"); import c0b_chain as B; print("BASE=(" + " ".join(B.OVERRIDES) + ")")')
LONG=("$KEY.cache_location=$C" "$KEY.dataset.extra_params.num_timesteps_after=160")
onrows() {  # onrows <python> <on_rows args>: on_rows.py's own CLI with SPLIT = onpol-all
  local py=$1; shift
  $py - "$ALL" "$@" <<'PY'
import sys, textwrap
sys.path.insert(0, "experiments/onpol_nuplan/scripts")
import on_rows as M
assert M.SPLIT == "alpasim_nuplan/onpol-train", M.SPLIT
M.SPLIT, sys.argv = sys.argv[1], [M.__file__] + sys.argv[2:]
exec(textwrap.dedent(open(M.__file__).read().split('if __name__ == "__main__":\n', 1)[1]), M.__dict__)
PY
}
lists() {  # onpol-held in NCH interleaved chunks
  local d=$W/lists/held6; mkdir -p "$d"; [[ $(wc -l < $SPL/held.txt) -eq 460 ]] || die "held.txt"
  for i in $(seq 0 $((NCH - 1))); do awk -v i=$i -v n=$NCH '(NR - 1) % n == i' $SPL/held.txt > "$d/c$i.txt.tmp"
    if [[ -f $d/c$i.txt ]]; then cmp -s "$d/c$i.txt" "$d/c$i.txt.tmp" || die "$d/c$i.txt changed"; rm "$d/c$i.txt.tmp"; else mv "$d/c$i.txt.tmp" "$d/c$i.txt"; fi
  done; }
STAGE=${1:?stage}; shift
case $STAGE in
roll1)   # ol_chain roll1 on a held chunk: rollouts with the frame dump, the simulator state, pruning
  T=$1; i=$2; D=$W/roll/$T-held/c$i/run; L=$W/lists/held6/c$i.txt
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ -f $C/nuplan_test/scenes_list.dill ]] || die "no cache $C"
  [[ -s $L && -z $(comm -23 <(sort "$L") <(sort $SPL/held.txt)) ]] || die "$L is not a held chunk"
  [[ ! -e $D/aggregate/results-summary.json ]] || die "$D already ran"
  rm -rf "$D"; mkdir -p "$D"
  env SH30_TAG=$T SH30_DUMP=1000000 bash experiments/alpasim/scripts/run.sh "$D" sh30 --scene-list "$L" "${BASE[@]}" "${LONG[@]}" || die "run.sh rc $?"
  [[ -f $D/aggregate/results-summary.json ]] || die "no results-summary"
  $P experiments/alpasim/scripts/c1_extract.py logs --run "$D" --out "$D/onpol_sim.pkl" --jobs 8 || die "c1_extract rc $?"
  python3 - "$D" <<'PY'
import glob, json, os, sys
D = sys.argv[1]; n = 0
for r in json.load(open(f"{D}/aggregate/results-summary.json"))["rollouts"]:
    if not r.get("failure_reason"):
        for f in glob.glob(f"{D}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            os.remove(f); n += 1
print("pruned", n, "rollout.asl")
PY
  echo "$(date '+%F %T') roll1 $T held c$i done: $(ls $D/driver-logs/dump | wc -l) dumps, $(du -sh $D | cut -f1)" ;;
geo)     # manifest of the held roll dir (labels <T>@h<i>: on_rows reads the seed from the label's tag), rows, labels K and C
  T=$1; N=$2; R=$W/roll/$T-held
  python3 - "$R" "$T" "$NCH" <<'PY' || die "held rollout chunks missing"
import json, os, sys
R, T, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = {f"{T}@h{i}": [f"{R}/c{i}/run"] for i in range(n) if os.path.exists(f"{R}/c{i}/run/onpol_sim.pkl")}
json.dump(m, open(f"{R}/manifest.json", "w"), indent=1)
print(len(m), "of", n, "chunks"); sys.exit(len(m) != n)
PY
  onrows $NAV geo --name "$N" --roll "$R" --workers ${OL_WORKERS:-48} || die "on_rows geo rc $?"
  $NAV $E/scripts/ol_labels.py build --name "$N" --workers ${OL_WORKERS:-48} || die "ol_labels build rc $?" ;;
tok)     # ol_chain tok with the onpol-all record: tokens + teacher + gate R1, then the pool gates (R1 / R2 as amended, K0); GPU
  N=$1; CTL=$2
  onrows $TR tok --name "$N" --control "$CTL" || die "on_rows tok rc $?"
  $TR $E/scripts/ol_labels.py gate --name "$N" --out "$W/rows/$N" || die "pool gates failed ($W/rows/$N/gates.md)" ;;
pool)    # route memory features + route labels of the rows (route_mem rm_build pool); CPU
  $P experiments/route_mem/scripts/rm_build.py pool --name "$1" --workers ${OL_WORKERS:-32} || die "rm_build pool rc $?" ;;
merge)   # on1_<OUT> = on1_<A> (prefix r1) + on1_<B> (prefix r2) (ol_pool_merge.py: tab, labels, route.npz, front, teacher), feat/on1_<OUT>.npz alike
  OUT=$1; A=$2; B=$3
  $TR $E/scripts/ol_pool_merge.py --out on1_$OUT --pools on1_$A:r1 on1_$B:r2 || die "ol_pool_merge rc $?"
  $TR - "$OUT" "$A" "$B" <<'PY' || die "feature merge"
import os, sys
import numpy as np
D = os.environ["DATA_DIR"]; FE, CR = f"{D}/runs/route_mem/feat", f"{D}/runs/op_parity/cache"; out, *src = sys.argv[1:]
Z = [np.load(f"{FE}/on1_{n}.npz") for n in src]
for z, n in zip(Z, src):
    assert np.array_equal(z["names"], np.load(f"{CR}/on1_{n}/tab.npz")["names"]), n
names = np.concatenate([np.array([f"on:{p}:{t[3:]}" for t in z["names"].tolist()]) for z, p in zip(Z, ("r1", "r2"))])
tab, r = np.load(f"{CR}/on1_{out}/tab.npz")["names"], np.load(f"{CR}/on1_{out}/route.npz")
assert np.array_equal(names, tab) and np.array_equal(r["tokens"], tab), "merged feature names differ from the merged tab / route.npz"
np.savez(f"{FE}/on1_{out}.npz", names=names, **{k: np.concatenate([z[k] for z in Z]) for k in ("f", "k", "ok", "route0_d")})
print(f"feat on1_{out}", len(names), "rows; per pool", [len(z["names"]) for z in Z], "route ok", float(r["ok"].mean()))
PY
  ;;
gate)    # route-night pool gate on a pool: rows >= 10 000, route.npz ok >= 0.95, route memory at the anchor within 5 cm on >= 0.99 of the ok rows,
         # and the rows gates (gates.json) of every source rows name passed
  N=$1; shift
  $P - "$N" "$@" <<'PY' || die "pool gate failed"
import json, os, sys
import numpy as np
n, *rows = sys.argv[1:]; D = os.environ["DATA_DIR"]
f = np.load(f"{D}/runs/route_mem/feat/on1_{n}.npz"); r = np.load(f"{D}/runs/op_parity/cache/on1_{n}/route.npz")
g = {x: bool(json.load(open(f"{D}/runs/onpol_long/rows/{x}/gates.json"))["pass"]) for x in rows}
ok, d = f["ok"], f["route0_d"]
res = dict(pool=n, rows=int(len(ok)), feat_ok=float(ok.mean()), route_ok=float(r["ok"].mean()), cov20=float(r["cov20"].mean()),
           route0_share_005=float((d[ok] <= 0.05).mean()), route0_med=float(np.nanmedian(d)), rows_gates=g)
res["pass"] = bool(res["rows"] >= 10000 and res["route_ok"] >= 0.95 and res["route0_share_005"] >= 0.99 and all(g.values()))
os.makedirs(f"{D}/runs/onpol_long/report", exist_ok=True)
json.dump(res, open(f"{D}/runs/onpol_long/report/poolgate-{n}.json", "w"), indent=1)
print(json.dumps(res)); sys.exit(0 if res["pass"] else 2)
PY
  ;;
submit)  # held rollouts -> geo -> tok -> strip / pool (held rows <OUT>held) -> merge with the train pool <A> into <OUT> -> gate
  T=$1; A=$2; OUT=$3; CTL=${4:-smoke10eY}; H=${OUT}held; lists
  [[ -f $CK/$T/ckpt-final.pt ]] || die "no checkpoint $T"; [[ -f $DATA_DIR/runs/onpol_nuplan/rows/$CTL/r1.json ]] || die "no interpreted control $CTL"
  [[ -f $W/rows/$A/gates.json && -f $DATA_DIR/runs/route_mem/feat/on1_$A.npz && -f $DATA_DIR/runs/op_parity/cache/on1_$A/route.npz ]] || die "train pool $A incomplete"
  mkdir -p "$W/roll/$T-held"
  sub() { local name=$1 after=$2; shift 2; local id; id=$(CL submit --owner ${OL_OWNER:-next-round} --name "$name" --tries 1 --priority ${PRIO:-13} --no-check \
            ${after:+--after $after} --log-dir "$W/pool/$name" "$@" | awk '{print $NF}') && [[ -n $id ]] || die "submit $name"; echo "$id"; echo "$id $name (ol_all)" >> $W/jobs.txt; }
  r=$(CL fanout --arms "$(seq -s, 0 $((NCH - 1)))" --name "ol-rollheld-${T}" --owner ${OL_OWNER:-next-round} --vram 32 --cpu 8 --ram 60 --tries 1 --priority ${PRIO:-13} \
       --timeout-h 3 --no-check --log-dir "$W/roll/$T-held/c{arm}/pool" -- bash $E/scripts/ol_all.sh roll1 "$T" "{arm}" | awk '{print $1}' | paste -sd,)
  [[ $(echo "$r" | tr , '\n' | grep -c .) -eq $NCH ]] || die "fanout ids: $r"
  echo "$r ol-rollheld-$T (ol_all)" >> $W/jobs.txt
  g=$(sub ol-geo-$H "$r" --vram 0.5 --cpu 48 --ram 96 --timeout-h 2 -- bash $E/scripts/ol_all.sh geo $T $H) || exit 1
  k=$(sub ol-tok-$H "$g" --vram 16 --cpu 16 --ram 64 --timeout-h 2 -- bash $E/scripts/ol_all.sh tok $H $CTL) || exit 1
  s=$(sub ol-strip-$H "$k" --vram 0.5 --cpu 16 --ram 16 --timeout-h 2 -- bash $E/scripts/ol_chain.sh strip $T-held $H) || exit 1
  p=$(sub ol-pool-$H "$k" --vram 0.5 --cpu 32 --ram 80 --timeout-h 1 -- bash $E/scripts/ol_all.sh pool $H) || exit 1
  m=$(sub ol-merge-$OUT "$p" --vram 0.5 --cpu 4 --ram 80 --timeout-h 1 -- bash $E/scripts/ol_all.sh merge $OUT $A $H) || exit 1
  q=$(sub ol-gate-$OUT "$m" --vram 0.5 --cpu 1 --ram 16 --timeout-h 1 -- bash -c "bash $E/scripts/ol_all.sh gate $H $H; bash $E/scripts/ol_all.sh gate $OUT $A $H") || exit 1
  echo "submitted: rollouts $r; geo $g; tok $k; strip $s; pool $p; merge $m; gate $q" ;;
*) die "unknown stage $STAGE" ;;
esac
