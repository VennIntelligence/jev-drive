#!/usr/bin/env bash
# LR x STEPS pilot (plans/2026-10-15-lrsteps-pilot-prereg.md): RSG-L18-s0's exact command (route_mem/train-RSG-L18-s0 meta.json) with
# --lr-scale {0.03, 0.1, 0.3} x --steps {1250, 2500, 5000}, seed 0; the cell lr10n2500 IS RSG-L18-s0 (stored checkpoint and held read).
# Each training is followed (--after) by the held 460 / 18-decision read (rm_chain.sh read, 6 chunks); one report job after every read.
# Box, repo root.
#   rf_lrsteps.sh first            the staged first cell (lr10n5000) + its read
#   rf_lrsteps.sh rest             the other 7 cells + reads + the report job (after every read in $O/reads.txt)
#   rf_lrsteps.sh seed1 <cell>     seed 1 of a cell + read + a second report job (PRIO=<p> sets the pool priority, default 14)
#   rf_lrsteps.sh guard <tag..>    public 1491 / 10-decision guard reads + guard-report (Amendment 2)
#   rf_lrsteps.sh train <cell> <seed> | report | guard-report      (job bodies)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/lrsteps; G=$DATA_DIR/runs/route_follow/guard; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_lrsteps.sh
REC="--init RB0-F-s0 --warmup 1 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0"
GRID="lr03n1250 lr03n2500 lr03n5000 lr10n1250 lr10n2500 lr10n5000 lr30n1250 lr30n2500 lr30n5000"; FIRST=lr10n5000
tag() { [[ $1 == lr10n2500 ]] && echo RSG-L18-s$2 || echo RSG$1-L18-s$2; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority ${PRIO:-14} --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
cell() {  # cell seed: training + 6 read chunks after it; read ids -> $O/reads.txt
  local T; T=$(tag $1 $2); [[ $T != RSG-L18-s* ]] || { echo "$T is stored, not retrained" >&2; exit 1; }
  local t; t=$(sub lrs-train-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME train $1 $2) || exit 1
  for i in 0 1 2 3 4 5; do
    sub lrs-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O/reads.txt || exit 1
  done; }
report_job() { sub lrs-report${1:-} --after "$(paste -sd, $O/reads.txt)" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report > /dev/null || exit 1; }
case $1 in
first) mkdir -p $O/pool; cell $FIRST 0; cat $O/jobs.txt ;;
rest)  for c in $GRID; do [[ $c == lr10n2500 || $c == $FIRST ]] || cell $c 0; done; report_job; cat $O/jobs.txt ;;
seed1) cell $2 1; report_job -s1; tail -8 $O/jobs.txt ;;
guard)  # guard <tag..>: public 1491 / 10-decision reads (10 chunks each, rf_allrows.sh's way) + one guard-report job after them
  shift; rr=(); for T in "$@"; do [[ -f $CK/$T/ckpt-final.pt ]] || { echo "no $T"; exit 1; }
    for i in $(seq 0 9); do rr+=($(sub lrs-g10-$T-$i --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash experiments/onpol_long/scripts/ol_chain.sh \
        readlist $T $DATA_DIR/runs/onpol_long/lists/all_n10/c$i.txt 10 $G/read10/$T/b$i)) || exit 1; done; done
  sub lrs-guard-report --after "$(IFS=,; echo "${rr[*]}")" --vram 0.5 --cpu 16 --ram 32 --timeout-h 1 -- bash $ME guard-report > /dev/null || exit 1; tail -n 22 $O/jobs.txt ;;
guard-report)  # ol_confirm all10 vs YR10m10-F-s0 (stored manifests), RSG-L18 s0 / s1 guard reads as reference (rf_round2.sh report's guard block)
  A=$DATA_DIR/runs/alpasim; Q=(); X=RSGlr03n5000-L18
  Q+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1 $X-s0 $X-s1; do Q+=("$T@all10=$(ls -d $G/read10/$T/b*/ | sed 's#/$##' | paste -sd,)"); done
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${Q[@]}" --base YR10m10-F-s0 --mean RSG-L18=RSG-L18-s0,RSG-L18-s1 $X=$X-s0,$X-s1 \
     --pairs $X:RSG-L18 $X-s0:RSG-L18-s0 $X-s1:RSG-L18-s1 --out $O/guard ;;
train)
  c=$2; s=$3; N=${c#*n}; EV=$(( N == 1250 ? 250 : 500 )); LS=$(python3 -c "print(int('${c:2:2}') / 100)")
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $(tag $c $s) $REC --lr-scale $LS --steps $N --eval-every $EV --seed $s ;;
report)
  ALL=""; for c in $GRID; do ALL="$ALL $(tag $c 0)"; done
  for c in $GRID; do [[ -d $M/$(tag $c 1) ]] && ALL="$ALL $(tag $c 1)"; done; ALL="$ALL RSG-L18-s1"
  ALL=$(for t in $ALL; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && echo $t; done | awk '!s[$0]++' | paste -sd' ')
  R=""; H=(); F=(); PF=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); R="$R $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d")
    [[ $t == RSG-L18-s* ]] || PF+=("$t@18:RSG-L18-s${t##*-s}@18"); done
  python3 $DATA_DIR/runs/route_follow/held.py $R > $O/held_raw.txt; rc0=$?
  MN=(); PM=(); for c in $GRID; do T1=$(tag $c 1); [[ $c != lr10n2500 && " $ALL " == *" $T1 "* ]] && { MN+=("RSG$c-L18=$(tag $c 0),$T1"); PM+=("RSG$c-L18:RSG-L18" "$T1:RSG-L18-s1"); }; done
  [[ ${#MN[@]} -gt 0 ]] && MN+=("RSG-L18=RSG-L18-s0,RSG-L18-s1")
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --mean "${MN[@]}" --pairs "${PM[@]}" --out $O/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  $P - "$O" "$CK" "$GRID" <<'PY'; rc3=$?
import json, sys
import pandas as pd
O, CK, grid = sys.argv[1], sys.argv[2], sys.argv[3].split()
tag = lambda c, s=0: f"RSG-L18-s{s}" if c == "lr10n2500" else f"RSG{c}-L18-s{s}"  # noqa: E731
lr = lambda c: int(c[2:4]) / 100  # noqa: E731
pos = lambda c: (("03", "10", "30").index(c[2:4]), ("1250", "2500", "5000").index(c.split("n")[1]))  # noqa: E731
J = json.load(open(f"{O}/held18/confirm.json"))["held18"]
S = pd.read_csv(f"{O}/held18/per_scene.csv"); S = S[S.label.str.endswith("@held18")]; S["tag"] = S.label.str.split("@").str[0]
sc = set.intersection(*[set(X[X.offroad_or_collision_at_fault.notna()].scene) for _, X in S.groupby("tag")]); S = S[S.scene.isin(sc)]
z = S.score == 0
S["zero"], S["z_atf"] = z.astype(int), (z & (S.offroad_or_collision_at_fault > 0)).astype(int)
for c, k in (("z_col", "collision_at_fault"), ("z_off", "offroad"), ("z_cor", "left_corridor_laterally")):
    S[c] = (z & (S[k] > 0)).astype(int)
A = S.groupby("tag").agg(score=("score", "mean"), zero=("zero", "sum"), z_atf=("z_atf", "sum"), z_col=("z_col", "sum"), z_off=("z_off", "sum"),
                         z_cor=("z_cor", "sum"), gt=("dist_to_gt_trajectory", "mean"))
def dev(t):
    try:
        return json.load(open(f"{CK}/{t}/route_mem.json"))["dev_ade"]
    except Exception:
        return float("nan")
f = lambda v, p=1: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
rows, md = [], [f"# lr x steps pilot: held 452 / 18 decisions ({len(sc)} paired scenes), base RSG-L18-s0", "",
                "| cell (lr-scale, steps) | tag | zeros 452 | d zeros vs RSG-L18-s0 [95 % CI] | at-fault | d at-fault | col / off / cor | mean score | d score | GT (m) | dev ADE |",
                "|:--|:--|--:|:--|--:|:--|:--|--:|:--|--:|--:|"]
for c in grid + [g + "@s1" for g in grid] + ["ref@s1"]:
    cc, s = (c.split("@")[0], 1) if "@" in c else (c, 0)
    t = "RSG-L18-s1" if cc == "ref" else tag(cc, s)
    if t not in A.index or (cc == "lr10n2500" and s == 1):
        continue
    a, d = A.loc[t], J.get(t)
    name = "RSG-L18-s1 (seed reference)" if cc == "ref" else f"({lr(cc):g}, {cc.split('n')[1]})" + (" s1" if s else "")
    rows.append(dict(cell=cc, seed=s, tag=t, zeros=int(a.zero), d_zero=d and d["zero"], z_atf=int(a.z_atf), d_atf=d and d["z_atf"], score=a.score,
                     d_score=d and d["score"], gt=a["gt"], dev_ade=dev(t)))
    md.append(f"| {name} | {t} | {a.zero:g} | {f(d['zero']) if d else '-'} | {a.z_atf:g} | {f(d['z_atf']) if d else '-'} | {a.z_col:g} / {a.z_off:g} / {a.z_cor:g} | "
              f"{a.score:.4f} | {f(d['score'], 4) if d else '-'} | {a['gt']:.3f} | {dev(t):.4f} |")
P2 = [(k, v) for k, v in J.items() if " - " in k]
if P2:
    md += ["", "Seed means and same-seed pairs (A - B on the 452, log-clustered 95 % CI):", "", "| A - B | zeros | at-fault | mean score | GT (m) |", "|:--|:--|:--|:--|:--|"]
    md += [f"| {k} | {f(v['zero'])} | {f(v['z_atf'])} | {f(v['score'], 4)} | {f(v['gt'], 3)} |" for k, v in P2]
G = {r["cell"]: r for r in rows if r["seed"] == 0 and r["cell"] in grid}
md += ["", "Held-452 zeros, 3 x 3 (rows lr-scale, columns steps):", "", "| lr-scale \\ steps | 1250 | 2500 | 5000 |", "|:--|--:|--:|--:|"]
for l in ("03", "10", "30"):
    md.append(f"| {int(l) / 100:g} | " + " | ".join(str(G[f'lr{l}n{n}']['zeros']) if f"lr{l}n{n}" in G else "-" for n in (1250, 2500, 5000)) + " |")
if len(G) == 9:
    best = min(G.values(), key=lambda r: (r["zeros"], r["z_atf"], -r["score"], abs(pos(r["cell"])[0] - 1) + abs(pos(r["cell"])[1] - 1)))
    win = best["cell"] != "lr10n2500" and best["d_zero"][2] < 0
    worse = [r["cell"] for r in G.values() if r["d_zero"] and r["d_zero"][1] > 0]
    md += ["", f"Rule (prereg): fewest zeros = {best['cell']} ({best['tag']}, {best['zeros']})"
           + (f", d vs RSG-L18-s0 {f(best['d_zero'])}" if best["d_zero"] else "") + f" -> **{'WINNER: ' + best['tag'] if win else 'FLAT (RSG-L18 stays the base)'}**."
           + (f" Cells with zeros CI entirely above RSG-L18-s0 (worse): {', '.join(worse)}." if worse else " No cell is worse with a CI above 0.")]
    json.dump(dict(rows=rows, best=best["cell"], winner=best["cell"] if win else None, worse=worse), open(f"{O}/lrsteps.json", "w"), indent=1, default=float)
open(f"{O}/lrsteps.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
PY
  echo "held $rc0 held18 $rc1 follow $rc2 table $rc3"; exit $((rc0 + rc1 + rc2 + rc3)) ;;
*) echo "rf_lrsteps.sh first | rest | seed1 <cell> | train <cell> <seed> | report" >&2; exit 2 ;;
esac
