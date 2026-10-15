#!/usr/bin/env bash
# ON-K pilot (plans/2026-10-15-onk-pilot-prereg.md): RSG-L18-s0's exact command (route_mem/train-RSG-L18-s0 meta.json) with the on-policy rows
# per batch --on-k {26, 51} at --ho-w 3, plus --on-k 51 --ho-w 1; the arm k13 IS RSG-L18-s0 (stored checkpoint and held read, not retrained).
# Each training is followed (--after) by the held 460 / 18-decision read (rm_chain.sh read, 6 chunks); a report job after every read.
# Pool priority PRIO (default -1: below every other lane, fills idle cards only). Box, repo root.
#   rf_onk.sh first               the staged first arm (k51) + its read
#   rf_onk.sh rest                k26, k51w1 + reads + the seed-0 report job (after every read in $O/reads.txt)
#   rf_onk.sh seed1 <arm..>       seed 1 of the arms + reads + the 2-seed report job
#   rf_onk.sh train <arm> <seed> | report      (job bodies)
set -uo pipefail; cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/route_follow/onk; M=$DATA_DIR/runs/route_mem/read; CK=$DATA_DIR/runs/op_parity/runs
P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_onk.sh
REC="--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0"
ARMS="k13 k26 k51 k51w1"; FIRST=k51
flags() { case $1 in k26) echo "--on-k 26 --ho-w 3" ;; k51) echo "--on-k 51 --ho-w 3" ;; k51w1) echo "--on-k 51 --ho-w 1" ;; *) echo "no arm $1" >&2; exit 2 ;; esac; }
tag() { [[ $1 == k13 ]] && echo RSG-L18-s$2 || echo RSG$1-L18-s$2; }
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority ${PRIO:--1} --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
arm() {  # arm seed: training + 6 read chunks after it; read ids -> $O/reads.txt
  local T; T=$(tag $1 $2); [[ $T != RSG-L18-s* ]] || { echo "$T is stored, not retrained" >&2; exit 1; }
  [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists: never overwritten" >&2; exit 1; }
  local t; t=$(sub onk-train-$T --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME train $1 $2) || exit 1
  for i in 0 1 2 3 4 5; do
    sub onk-read-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 \
        -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6 >> $O/reads.txt || exit 1
  done; }
report_job() { sub onk-report${1:-} --after "$(paste -sd, $O/reads.txt)" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME report > /dev/null || exit 1; }
case $1 in
first) mkdir -p $O/pool; arm $FIRST 0; cat $O/jobs.txt ;;
rest)  for c in $ARMS; do [[ $c == k13 || $c == $FIRST ]] || arm $c 0; done; report_job; cat $O/jobs.txt ;;
seed1) shift; for c in "$@"; do arm $c 1; done; report_job -s1; tail -n $(( 7 * $# + 1 )) $O/jobs.txt ;;
train) FL=$(flags $2) || exit 2; exec $TR experiments/route_mem/scripts/rm_train.py train --tag $(tag $2 $3) $REC $FL --seed $3 ;;
report)
  ALL=""; for s in 0 1; do for c in $ARMS; do ALL="$ALL $(tag $c $s)"; done; done
  ALL=$(for t in $ALL; do ls $M/$t/c*/run/aggregate/results-summary.json > /dev/null 2>&1 && echo $t; done | awk '!s[$0]++' | paste -sd' ')
  R=""; H=(); F=(); PF=(); for t in $ALL; do d=$(ls -d $M/$t/c*/run | paste -sd,); R="$R $t=$d"; H+=("$t@held18=$d"); F+=("$t@18=$d")
    [[ $t == RSG-L18-s* ]] || PF+=("$t@18:RSG-L18-s${t##*-s}@18"); done
  python3 $DATA_DIR/runs/route_follow/held.py $R > $O/held_raw.txt; rc0=$?
  MN=(); PM=(); for c in $ARMS; do T1=$(tag $c 1); [[ $c != k13 && " $ALL " == *" $T1 "* ]] && { MN+=("RSG$c-L18=$(tag $c 0),$T1"); PM+=("RSG$c-L18:RSG-L18" "$T1:RSG-L18-s1"); }; done
  [[ ${#MN[@]} -gt 0 ]] && MN+=("RSG-L18=RSG-L18-s0,RSG-L18-s1")
  $P experiments/onpol_long/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --mean "${MN[@]}" --pairs "${PM[@]}" --out $O/held18 > /dev/null; rc1=$?
  $P experiments/route_follow/scripts/rf_follow.py --runs "${F[@]}" --out $O/follow --workers 16 --pairs "${PF[@]}" > /dev/null; rc2=$?
  $P - "$O" "$CK" "$ARMS" <<'PY'; rc3=$?
import json, sys
import pandas as pd
O, CK, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
tag = lambda c, s=0: f"RSG-L18-s{s}" if c == "k13" else f"RSG{c}-L18-s{s}"  # noqa: E731
KW = dict(k13=(13, 3), k26=(26, 3), k51=(51, 3), k51w1=(51, 1))        # (on-k, ho-w)
def press(c):  # expected on-policy hinge pressure per batch, ho_w x k / n_imit, n_imit = 0.75 nN + 13 (prereg)
    k, w = KW[c]; return w * k / (0.75 * (128 - 13 - k) + 13)
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
rows, md = [], [f"# ON-K pilot: held 452 / 18 decisions ({len(sc)} paired scenes), base RSG-L18-s0", "",
                "| arm (on-k, ho-w) | pressure x k13 | tag | zeros 452 | d zeros vs RSG-L18-s0 [95 % CI] | at-fault | d at-fault | col / off / cor | mean score | d score | GT (m) | dev ADE |",
                "|:--|--:|:--|--:|:--|--:|:--|:--|--:|:--|--:|--:|"]
for s in (0, 1):
    for c in arms:
        t = tag(c, s)
        if t not in A.index:
            continue
        a, d = A.loc[t], J.get(t)
        rows.append(dict(arm=c, seed=s, tag=t, zeros=int(a.zero), d_zero=d and d["zero"], z_atf=int(a.z_atf), d_atf=d and d["z_atf"], score=a.score,
                         d_score=d and d["score"], gt=a["gt"], dev_ade=dev(t)))
        md.append(f"| ({KW[c][0]}, {KW[c][1]}){' s1' if s else ''} | {press(c) / press('k13'):.2f} | {t} | {a.zero:g} | {f(d['zero']) if d else '-'} | {a.z_atf:g} | "
                  f"{f(d['z_atf']) if d else '-'} | {a.z_col:g} / {a.z_off:g} / {a.z_cor:g} | {a.score:.4f} | {f(d['score'], 4) if d else '-'} | {a['gt']:.3f} | {dev(t):.4f} |")
P2 = [(k, v) for k, v in J.items() if " - " in k]
if P2:
    md += ["", "Seed means and same-seed pairs (A - B on the 452, log-clustered 95 % CI):", "", "| A - B | zeros | at-fault | mean score | GT (m) |", "|:--|:--|:--|:--|:--|"]
    md += [f"| {k} | {f(v['zero'])} | {f(v['z_atf'])} | {f(v['score'], 4)} | {f(v['gt'], 3)} |" for k, v in P2]
md += ["", "Lines (prereg): candidate = zeros CI high < 0 or score CI low > 0 vs RSG-L18-s0; stop (no seed 1) = zeros CI low > +5.", ""]
verdict = {}
for r in rows:
    if r["seed"] or r["arm"] == "k13" or not r["d_zero"]:
        continue
    cand = r["d_zero"][2] < 0 or r["d_score"][1] > 0
    stop = r["d_zero"][1] > 5
    worse = r["d_zero"][1] > 0
    verdict[r["arm"]] = dict(candidate=cand, stop=stop, worse=worse)
    md.append(f"- {r['arm']} ({r['tag']}): zeros {f(r['d_zero'])}, score {f(r['d_score'], 4)} -> **{'CANDIDATE' if cand else 'no gain'}**"
              + ("; zeros CI above 0 (worse)" if worse else "") + ("; **STOP** (no seed 1)" if stop else "; seed 1 per the stop line"))
json.dump(dict(rows=rows, verdict=verdict), open(f"{O}/onk.json", "w"), indent=1, default=float)
open(f"{O}/onk.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
PY
  echo "held $rc0 held18 $rc1 follow $rc2 table $rc3"; exit $((rc0 + rc1 + rc2 + rc3)) ;;
*) echo "rf_onk.sh first | rest | seed1 <arm..> | train <arm> <seed> | report" >&2; exit 2 ;;
esac
