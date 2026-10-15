#!/usr/bin/env bash
# SLOW S1 / S2 round (plans/2026-10-15-slow-s1s2-prereg.md + Amendments 1-3): the 2 x 2 grid on the RSG-L18 recipe (rf_onk.sh REC):
#   S1   = --ego-drop-speed 0.5 --ego-drop-open (speed history dropped on open rows only), pool on1_L18RB0 (self)
#   Sall = --ego-drop-speed 0.5 (all rows, RSGallV's drop on RSG-L18's own pool): S1's control
#   S3   = merged pool on1_RSGPAITV (on1_L18RB0 rows `self` + on1_PAITV human rows `both`, --on-mode mixed), PAITV ~10 % of the drawn pool rows
#   S2   = S3 + S1's drop (open rows only; the PAITV rows carry no open label, so they are never dropped)
# Box, repo root. Every GPU job through the pool; ids in $O/jobs.txt.
#   rf_slow.sh gates          the bit-identity smokes (60 steps; old code = worktree at OLD_REV) + the merged pool + the compare job
#   rf_slow.sh submit [ARMS]  pilots (1000 steps, seeds 0 / 1) -> trainings (2500, seeds 0 / 1, --after the arm's pilots) -> held 460 @18
#                             (6 chunks) + public 1491 @10 guard (10 chunks) per training -> confirm, deficit, report (after every read)
#   rf_slow.sh pilot          pilot gate table (prereg Amendment 3 lines) from the pilot runs -> $O/pilot.{md,json}
#   rf_slow.sh report         lines 1 (held) and 2 (guard) pass / fail with numbers, from confirm / deficit outputs (+ rear-ended counts)
#   rf_slow.sh tokyo          ON TOKYO: pull the seed-0 checkpoints (+ sidecars) as they appear, md5 checked; write PAI_READ_REQUEST.md (+ box copy)
#   rf_slow.sh smoke | train | pool | gatecmp | confirm | deficit     (job bodies)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/data}
RF=$DATA_DIR/runs/route_follow; O=$RF/slow; G=$RF/guard; M=$DATA_DIR/runs/route_mem/read; E=experiments/onpol_long; W=$DATA_DIR/runs/onpol_long
CK=$DATA_DIR/runs/op_parity/runs; P=$DATA_DIR/third_party/alpasim/.venv/bin/python; TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_slow.sh
OLD_REV=${OLD_REV:-6c3a6ae2}; OLD=$O/code_old
BASE="--init RB0-F-s0 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train"
RB="--on on1_L18RB0 --on-mode self"; OPEN="--ego-drop-speed 0.5 --ego-drop-open"; MP=on1_RSGPAITV; PK=runs/onpol_nuplan/rows/L18RB0/lon.npz
ARMS=${ARMS:-"S1 Sall S3 S2"}
wv() { [[ -s $O/paitv_weight.txt ]] || { echo "no $O/paitv_weight.txt (rf_slow.sh pool)" >&2; exit 1; }; cat $O/paitv_weight.txt; }
flags() { case $1 in
  S1) echo "$RB $OPEN" ;; Sall) echo "$RB --ego-drop-speed 0.5" ;;
  S3) echo "--on $MP --on-mode mixed --on-weight pool=v:$(wv)" ;;
  S2) echo "--on $MP --on-mode mixed --on-weight pool=v:$(wv) $OPEN --open-pack $PK" ;;
  *) echo "no arm $1" >&2; exit 2 ;; esac; }
tag() { echo RSG$1${3:-}-L18-s$2; }                                   # tag ARM SEED [P (pilot)]
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner route-follow --name "$name" --tries 1 --priority ${PRIO:-12} --no-check \
     --log-dir $O/pool/$name "$@" | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
dirs() { ls -d $1/$2/b*/ 2>/dev/null | sed 's#/$##' | paste -sd,; }
hdirs() { ls -d $M/$1/c*/run 2>/dev/null | paste -sd,; }
PAIRS=(RSGS1-L18:RSG-L18 RSGSall-L18:RSG-L18 RSGS3-L18:RSG-L18 RSGS2-L18:RSG-L18 RSGS1-L18:RSGSall-L18 RSGS2-L18:RSGS3-L18)
MEAN=(RSG-L18=RSG-L18-s0,RSG-L18-s1); for a in $ARMS; do MEAN+=(RSG$a-L18=RSG$a-L18-s0,RSG$a-L18-s1); done
case $1 in
# ---------------------------------------------------------------- gates (bit identity) and the merged pool
gates)
  mkdir -p $O/pool
  [[ -d $OLD ]] || git worktree add --detach $OLD $OLD_REV || exit 1
  [[ -f $DATA_DIR/runs/op_parity/cache/on1_PAITV/tab.npz && -f $DATA_DIR/runs/route_mem/feat/on1_PAITV.npz ]] || { echo "no on1_PAITV on the box" >&2; exit 1; }
  pl=$(sub slow-pool --vram 0.5 --cpu 4 --ram 80 --timeout-h 2 -- bash $ME pool) || exit 1
  ids=()
  for x in old:off new:off new:off2 old:d05 new:d05 new:s1 new:s2w0; do c=${x%%:*}; k=${x#*:}; A=(); [[ $k == s2w0 ]] && A=(--after $pl)
    ids+=($(sub slow-g-$c-$k "${A[@]}" --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME smoke $c $k)) || exit 1; done
  sub slow-gatecmp --after "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME gatecmp > /dev/null || exit 1
  tail -n 9 $O/jobs.txt ;;
smoke)   # smoke <old|new> <off|off2|d05|s1|s2w0>: 60 steps of the RSG-L18 recipe, tag SLOWG-<code>-<kind>
  C=.; [[ $2 == old ]] && C=$OLD
  case $3 in off|off2) F="$RB" ;; d05) F="$RB --ego-drop-speed 0.5" ;; s1) F="$RB $OPEN" ;; s2w0) F="--on $MP --on-mode mixed --on-weight pool=v:0 $OPEN --open-pack $PK" ;; esac
  exec $TR $C/experiments/route_mem/scripts/rm_train.py train --tag SLOWG-$2-$3 $BASE $F --steps 60 --seed 0 ;;
pool)    # the merged pool (rf_pait.py merge; nuPlan rows first and in their own order, mode self; PAITV rows mode both) + the PAITV weight
  $TR experiments/route_follow/scripts/rf_pait.py merge --out RSGPAITV --pools on1_L18RB0:r:self on1_PAITV:v:both || exit 1
  $TR - "$DATA_DIR" "$O" <<'PY' || exit 1
import json, sys
import numpy as np
D, O = sys.argv[1:3]
t = np.load(f"{D}/runs/op_parity/cache/on1_RSGPAITV/tab.npz")
r = np.load(f"{D}/runs/op_parity/cache/on1_L18RB0/tab.npz")
p = t["pool"].astype(str)
nr, nv = int((p == "r").sum()), int((p == "v").sum())
assert nr == len(r["names"]) and (p[:nr] == "r").all() and np.array_equal(t["ego"][:nr], r["ego"]) and np.array_equal(t["fut"][:nr], r["fut"])
assert all(a == "on:r:" + b[3:] for a, b in zip(t["names"][:nr].tolist(), r["names"].tolist())), "merged names"
w = round(0.1 / 0.9 * nr / nv, 6)                                   # PAITV = 10 % of the drawn pool rows
open(f"{O}/paitv_weight.txt", "w").write(f"{w}\n")
v = t["speed"][p == "v"].astype(float)
rec = dict(rows_r=nr, rows_v=nv, weight_v=w, expected_v_share=w * nv / (w * nv + nr), v_speed=dict(p50=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(v.max())),
           scenes_v=len(set(t["log"][p == "v"].tolist())), mode=dict(zip(*[x.tolist() for x in np.unique(t["mode"], return_counts=True)])))
json.dump(rec, open(f"{O}/pool.json", "w"), indent=1); print(json.dumps(rec))
PY
  ;;
gatecmp)
  $TR - "$DATA_DIR" "$O" <<'PY'
import glob, json, sys
import numpy as np
import torch
D, O = sys.argv[1:3]
RSG_HASH60 = "ea8cacc3e451c577"                                    # RSG-L18-s0's stored on_hash_60 (DONE summary)
def run(tag):
    d = sorted(glob.glob(f"{D}/runs/route_mem/train-{tag}/*/"))[-1]
    sc = {}
    for l in open(d + "events.jsonl"):
        e = json.loads(l)
        if e["kind"] == "scalar":
            sc[(e["tag"], e["step"])] = e["value"]
    return sc, json.load(open(d + "DONE")), torch.load(f"{D}/runs/op_parity/runs/{tag}/ckpt-final.pt", map_location="cpu", weights_only=False)["model"]
def cmp(a, b, note):
    (sa, da, ma), (sb, db, mb) = run(a), run(b)
    keys = sorted(set(sa) & set(sb), key=str)
    keys = [k for k in keys if not k[0].startswith(("throughput/", "loss/drop/"))]
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g])
    same_keys = all(set(ma[g]) == set(mb[g]) for g in ("net", "parity"))
    r = dict(a=a, b=b, note=note, scalars=len(keys), only_a=sorted({k[0] for k in set(sa) - set(sb)}), only_b=sorted({k[0] for k in set(sb) - set(sa)}),
             scalar_max_abs_diff=dl, weight_max_abs_diff=dw, weight_keys_equal=same_keys,
             on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")], dev_ade=[da.get("dev_ade"), db.get("dev_ade")])
    r["identical"] = bool(dl == 0 and dw == 0 and same_keys and r["on_hash_60"][0] == r["on_hash_60"][1])
    return r
R = dict(
    G0_everything_off=cmp("SLOWG-old-off", "SLOWG-new-off", "old code vs new code, every new flag off (RSG-L18 recipe, 60 steps)"),
    G0_repeat=cmp("SLOWG-new-off", "SLOWG-new-off2", "new code twice (run-to-run determinism)"),
    G1_open_flag_off=cmp("SLOWG-old-d05", "SLOWG-new-d05", "--ego-drop-speed 0.5 alone: old code vs new code (--ego-drop-open off)"),
    G2_paitv_weight0=cmp("SLOWG-new-s1", "SLOWG-new-s2w0", "S1 on on1_L18RB0 vs S1 flags on the merged on1_RSGPAITV, --on-mode mixed, PAITV weight 0"))
R["G0_everything_off"]["rsg_l18_s0_on_hash_60"] = RSG_HASH60
R["G0_everything_off"]["identical"] &= R["G0_everything_off"]["on_hash_60"][0] == RSG_HASH60
s1 = json.load(open(sorted(glob.glob(f"{D}/runs/route_mem/train-SLOWG-new-s1/*/DONE"))[-1]))
R["S1_smoke_drop"] = dict(store=s1.get("ego_drop_open"), realised=s1.get("ego_drop_open_realised"))
R["pass"] = all(R[k]["identical"] for k in ("G0_everything_off", "G0_repeat", "G1_open_flag_off", "G2_paitv_weight0"))
json.dump(R, open(f"{O}/gates.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
raise SystemExit(0 if R["pass"] else 2)
PY
  ;;
# ---------------------------------------------------------------- the round
submit)
  shift; [[ $# -gt 0 ]] && ARMS="$*"
  mkdir -p $O/pool; rr=()
  for a in $ARMS; do flags $a > /dev/null || exit 1
    for s in 0 1; do [[ ! -e $CK/$(tag $a $s P)/ckpt-final.pt && ! -e $CK/$(tag $a $s)/ckpt-final.pt ]] || { echo "$a s$s exists: never overwritten" >&2; exit 1; }; done
    p0=$(sub slow-pilot-$(tag $a 0 P) --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME train $a 0 P) || exit 1
    p1=$(sub slow-pilot-$(tag $a 1 P) --train --vram 28 --cpu 6 --ram 70 --timeout-h 1 -- bash $ME train $a 1 P) || exit 1
    for s in 0 1; do T=$(tag $a $s)
      t=$(sub slow-tr-$T --after $p0,$p1 --train --vram 28 --cpu 6 --ram 70 --timeout-h 2 -- bash $ME train $a $s) || exit 1
      for i in 0 1 2 3 4 5; do
        rr+=($(sub slow-held-$T-$i --after $t --vram 20 --cpu 6 --ram 40 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash experiments/route_mem/scripts/rm_chain.sh read $T $i 6)) || exit 1
      done
      for i in $(seq 0 9); do
        rr+=($(sub slow-g10-$T-$i --after $t --vram 20 --cpu 6 --ram 30 --timeout-h 2 --env CL_HIST_KEY=alpasim-read4 -- bash $E/scripts/ol_chain.sh readlist $T $W/lists/all_n10/c$i.txt 10 $O/read10/$T/b$i)) || exit 1
      done
    done
  done
  a=$(IFS=,; echo "${rr[*]}")
  c=$(sub slow-confirm --after "$a" --vram 0.5 --cpu 16 --ram 48 --timeout-h 2 -- bash $ME confirm) || exit 1
  d=$(sub slow-deficit --after "$a" --vram 0.5 --cpu 32 --ram 32 --timeout-h 1 -- bash $ME deficit) || exit 1
  sub slow-report --after "$c,$d" --vram 0.5 --cpu 2 --ram 8 --timeout-h 1 -- bash $ME report > /dev/null || exit 1
  echo "submitted $(( ${#rr[@]} + 3 )) read / summary jobs after the trainings of: $ARMS"; tail -n 4 $O/jobs.txt ;;
train)   # train ARM SEED [P]: P = the 1000-step pilot
  F=$(flags $2) || exit 2; N=2500; [[ ${4:-} == P ]] && N=1000
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $(tag $2 $3 ${4:-}) $BASE $F --steps $N --seed $3 ;;
confirm)
  A=$DATA_DIR/runs/alpasim; H=(); R=()
  R+=("YR10m10-F-s0@all10=$(python3 -c "import json; a = json.load(open('$A/ot3/a/manifest.json')); b = json.load(open('$A/cf1/a/manifest.json')); print(','.join(a['YR10m10-F-s0'] + a['YR10m10-F-s0@new'] + b['YR10m10-F-s0@fresh']))")")
  for T in RSG-L18-s0 RSG-L18-s1; do H+=("$T@held18=$(hdirs $T)"); R+=("$T@all10=$(dirs $G/read10 $T)"); done
  for a in $ARMS; do for s in 0 1; do T=$(tag $a $s); [[ -n $(hdirs $T) ]] && H+=("$T@held18=$(hdirs $T)"); [[ -n $(dirs $O/read10 $T) ]] && R+=("$T@all10=$(dirs $O/read10 $T)"); done; done
  $P $E/scripts/ol_confirm.py --runs "${H[@]}" --base RSG-L18-s0 --mean "${MEAN[@]}" --pairs "${PAIRS[@]}" --out $O/held18; r1=$?
  $P $E/scripts/ol_confirm.py --runs "${R[@]}" --base YR10m10-F-s0 --mean "${MEAN[@]}" --pairs "${PAIRS[@]}" --out $O/confirm; r2=$?
  exit $((r1 + r2)) ;;
deficit)
  R=(); for T in RSG-L18-s0 RSG-L18-s1; do R+=("$T=$(dirs $G/read10 $T)"); done
  for a in $ARMS; do for s in 0 1; do T=$(tag $a $s); [[ -n $(dirs $O/read10 $T) ]] && R+=("$T=$(dirs $O/read10 $T)"); done; done
  $P experiments/route_follow/scripts/rf_deficit.py --runs "${R[@]}" --mean "${MEAN[@]}" --pairs "${PAIRS[@]}" --out $O/deficit --workers 32 ;;
pilot)
  $TR - "$DATA_DIR" "$O" "$ARMS" <<'PY'
import glob, json, sys
import numpy as np
D, O, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
ADE_MAX, RSG_ADE = 0.538, 0.528
def run(t):
    g = sorted(glob.glob(f"{D}/runs/route_mem/train-{t}/*/"))
    if not g or not glob.glob(g[-1] + "DONE"):
        return None
    d = g[-1]
    sc = {}
    for l in open(d + "events.jsonl"):
        e = json.loads(l)
        if e["kind"] == "scalar":
            sc.setdefault(e["tag"], {})[e["step"]] = e["value"]
    return sc, json.load(open(d + "DONE")), json.load(open(d + "rely.json"))
rows, md = [], ["# SLOW pilot gate (1000 steps; prereg 闸 + Amendment 1: judged on seed 0, seed 1 reported)", "",
                "| arm | seed | dev ADE @1000 (<= 0.538) | G-rely read | dropped share / expected (0.8 .. 1.2) | open share (off / yr / pool) | PAITV imit first 100 -> last 100 steps (<= 0.8) | pass |",
                "|:--|--:|--:|:--|:--|:--|:--|:--|"]
for a in arms:
    for s in (0, 1):
        t = f"RSG{a}P-L18-s{s}"
        R = run(t)
        if R is None:
            md.append(f"| {a} | {s} | not done | | | | | |")
            continue
        sc, done, rely = R
        ade = sc["dev/ade"][max(sc["dev/ade"])]
        ok = dict(ade=ade <= ADE_MAX, rely=bool(rely["read"]))
        dd = done.get("ego_drop_open_realised")
        dtxt, otxt = "-", "-"
        if a in ("S1", "S2"):
            r = dd["ratio_to_expected"]
            ok["drop"] = r is not None and 0.8 <= r <= 1.2
            dtxt = f"{dd['dropped_share']:.4f} / {dd['expected']:.4f} = {r:.3f}"
            otxt = " / ".join("-" if dd[f"open_share_{k}"] is None else f"{dd[f'open_share_{k}']:.3f}" for k in ("off", "yr", "pool"))
        itxt = "-"
        if a in ("S2", "S3"):
            im = sc.get("loss/on/imit", {})
            st = sorted(im)
            first, last = np.mean([im[k] for k in st if k <= 100]), np.mean([im[k] for k in st if k > max(st) - 100])
            ok["imit"] = bool(last <= 0.8 * first)
            itxt = f"{first:.4f} -> {last:.4f} (x {last / first:.3f})"
        p = all(ok.values())
        rows.append(dict(arm=a, seed=s, tag=t, dev_ade=ade, rely=rely, drop=dd, ok=ok, passed=p))
        md.append(f"| {a} | {s} | {ade:.4f} | {rely['read']} (d_junction {rely['d_junction']:+.4f}, diff lo {rely['diff_lo']:+.4f}) | {dtxt} | {otxt} | {itxt} | **{'pass' if p else 'FAIL ' + ','.join(k for k, v in ok.items() if not v)}** |")
md += ["", "Gate (prereg): dev ADE (memory on) <= RSG-L18 0.528 + 0.01; G-rely read true; S1 / S2 dropped share within +-20 % of 0.5 x open share;",
       "S2 (and S3) PAITV imitation loss (`loss/on/imit`, the 25-step means of the mixed `both` rows) at the end <= 0.8 x its start. Arm verdict = seed 0."]
open(f"{O}/pilot.md", "w").write("\n".join(md) + "\n")
json.dump(rows, open(f"{O}/pilot.json", "w"), indent=1, default=float)
print("\n".join(md))
PY
  ;;
report)
  $TR - "$DATA_DIR" "$O" "$ARMS" "$M" <<'PY'
import glob, json, sys
from pathlib import Path
import numpy as np
D, O, arms, M = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4]
H = json.load(open(f"{O}/held18/confirm.json")).get("held18", {}) if Path(f"{O}/held18/confirm.json").exists() else {}
C = json.load(open(f"{O}/confirm/confirm.json")).get("all10", {}) if Path(f"{O}/confirm/confirm.json").exists() else {}
F = json.load(open(f"{O}/deficit/deficit.json")).get("pairs", {}) if Path(f"{O}/deficit/deficit.json").exists() else {}
def rear(tag):                                                        # rollouts hit from behind (metrics.collision_rear > 0) per scene, held 18
    out = {}
    for f in glob.glob(f"{M}/{tag}/c*/run/aggregate/results-summary.json"):
        for r in json.load(open(f))["rollouts"]:
            out[r["clipgt_id"]] = float((r.get("metrics") or {}).get("collision_rear") or 0) > 0
    return out
def rear_mean(base):
    R = [rear(f"{base}-s{s}") for s in (0, 1)]
    return R
f = lambda v, p=4: "n/a" if v is None else f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
md = ["# SLOW S1 / S2: lines 1 (held 460 @18) and 2 (public 1491 @10 guard), seed means vs RSG-L18 (prereg 读数与线)", ""]
md += ["| arm | held zeros d (<= +3) | held GT d m (<= +0.03) | held score d (>= 0) | rear-ended arm / RSG (<= 0.7) | line 1 | guard score d (CI lo > -0.01) | guard zeros d (<= +5) | plan 2 s short d (CI lo > 0) | line 2 |",
       "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
res = {}
R0 = rear_mean("RSG-L18")
for a in arms:
    k = f"RSG{a}-L18 - RSG-L18"
    h, c, d = H.get(k), C.get(k), F.get(f"RSG{a}-L18:RSG-L18")
    Ra = rear_mean(f"RSG{a}-L18")
    sc = set.intersection(*[set(x) for x in R0 + Ra]) if all(R0 + Ra) else set()
    rr = (np.mean([sum(x[s] for s in sc) for x in Ra]), np.mean([sum(x[s] for s in sc) for x in R0])) if sc else None
    l1 = None if h is None or rr is None else bool(h["zero"][0] <= 3 and h["gt"][0] <= 0.03 and rr[0] <= 0.7 * rr[1] and h["score"][0] >= 0)
    l2 = None if c is None or d is None else bool(c["score"][1] > -0.01 and c["zero"][0] <= 5 and d["lo"] > 0)
    res[a] = dict(held=h, guard=c, deficit=d, rear=rr, rear_scenes=len(sc), line1=l1, line2=l2)
    dtx = "n/a" if d is None else "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"])
    md.append(f"| {a} | {f(h and h['zero'], 1)} | {f(h and h['gt'], 3)} | {f(h and h['score'])} | {'n/a' if rr is None else f'{rr[0]:.1f} / {rr[1]:.1f} = {rr[0] / max(rr[1], 1e-9):.2f}'} | "
              f"**{ {None: 'pending', True: 'PASS', False: 'FAIL'}[l1] }** | {f(c and c['score'])} | {f(c and c['zero'], 1)} | "
              f"{dtx} | **{ {None: 'pending', True: 'PASS', False: 'FAIL'}[l2] }** |")
md += ["", "Interaction readings (Amendment 1): " + "; ".join(f"{k}: guard zeros {f(C.get(k, {}).get('zero'), 1)}, held zeros {f(H.get(k, {}).get('zero'), 1)}"
                                                          for k in ("RSGS1-L18 - RSGSall-L18", "RSGS2-L18 - RSGS3-L18")), "",
       "Line 1: zeros <= RSG + 3 (point), GT <= RSG + 0.03 m (point; CI shown), rear-ended (collision_rear > 0, seed mean on the common scenes) <= 0.7 x RSG, "
       "mean score >= RSG (point; CI shown). Line 2: public mean-score difference CI low > -0.01, zeros <= RSG + 5 (point), plan 2 s short (rf_deficit, positive = "
       "less short) CI low > 0. Sources: held18/confirm.json, confirm/confirm.json, deficit/deficit.json under " + O]
open(f"{O}/report.md", "w").write("\n".join(md) + "\n")
json.dump(res, open(f"{O}/report.json", "w"), indent=1, default=float)
print("\n".join(md))
PY
  ;;
tokyo)   # ON TOKYO (the box cannot reach it): pull the seed-0 final checkpoints of the four arms as they appear (poll every 5 min, WAIT=0: once),
         # md5 equal on both sides; then the PAI read request, written here and on the box ($BOXD/runs/route_follow/slow/PAI_READ_REQUEST.md)
  BOX=${BOX:-autodl}; BOXD=${BOXD:-/root/autodl-tmp/ujs}; TD=/data/runs/op_parity/runs; O=/data/runs/route_follow/slow; mkdir -p $O; Q=$O/PAI_READ_REQUEST.md
  done_=(); miss=(S1 S2 S3 Sall)
  while :; do left=()
    for a in "${miss[@]}"; do T=$(tag $a 0)
      if ! ssh $BOX "test -f $BOXD/runs/op_parity/runs/$T/ckpt-final.pt -a -f $BOXD/runs/op_parity/runs/$T/route_mem.json"; then left+=($a); continue; fi
      mkdir -p $TD/$T && rsync -a $BOX:$BOXD/runs/op_parity/runs/$T/ $TD/$T/ || { echo "rsync $T failed" >&2; exit 1; }
      for f in ckpt-final.pt adapter.pt route_mem.json; do
        a1=$(ssh $BOX "md5sum $BOXD/runs/op_parity/runs/$T/$f" | cut -d' ' -f1); b1=$(md5sum $TD/$T/$f | cut -d' ' -f1)
        [[ -n $a1 && $a1 == $b1 ]] || { echo "md5 $T/$f differs ($a1 / $b1)" >&2; exit 1; }; echo "$(date '+%F %T') $T/$f md5 $a1 ok"
      done; done_+=($T); done
    miss=("${left[@]}"); (( ${#miss[@]} == 0 || ${WAIT:-1} == 0 )) && break; sleep 300; done
  miss=$(for a in "${miss[@]}"; do tag $a 0; done | paste -sd' ')
  {
  echo "# SLOW S1 / S2: PAI 281 read request (prereg line 3, Amendment 1 order S1, S2, S3, Sall), written $(date '+%F %T') by rf_slow.sh tokyo"
  echo
  echo "Checkpoints copied to Tokyo $TD/<tag>/ (ckpt-final.pt, adapter.pt, route_mem.json md5-checked): ${done_[*]:-none}${miss:+; not yet copied: $miss}."
  echo "Serving = the reference /data/runs/route_night/v2pai/RSG-L18-s0-gate exactly (sv9.sh gateread, decision 309): pai_eval.sh with the default SERVE"
  echo "(\"-e JEV_VCONT=1.0 -e JEV_LEAD=1\", JEV_TRACK unset = gate serve), WIZ gains 20, RUNPFX gate-, CHUNK 36, the reference's own 8 chunk lists, CONC 4;"
  echo "seed 0 only. Tokyo cards are owned by the expert read: run these after it, in this order (one at a time on both cards, or one per card;"
  echo "read as many as fit; the first ones are judged first). Line 3 after each (CPU, Tokyo repo):"
  echo "  python3 experiments/route_follow/scripts/rf_londiag.py pai --run /data/runs/route_night/v2pai/<tag>-gate --ref /data/runs/route_night/v2pai/RSG-L18-s0-gate"
  echo
  echo '```bash'
  echo 'cd ~/mycode/jev-drive && git pull'
  for a in S1 S2 S3 Sall; do T=$(tag $a 0); V=/data/runs/route_night/v2pai/$T-gate
    echo "mkdir -p $V/chunks && cp /data/runs/route_night/v2pai/RSG-L18-s0-gate/chunks/*.txt $V/chunks/ && git log --oneline -1 > $V/COMMIT && \\"
    echo "  RUNPFX=gate- CHUNK=36 CARDS=\"0 1\" WIZ=controller.gains.rel_front_steering_angle_weight=20 bash experiments/alpasim/scripts/pai_eval.sh $V $T /data/runs/route_night/test281.txt"
  done
  echo '```'
  } > $Q; cat $Q; rsync -a $Q $BOX:$BOXD/runs/route_follow/slow/ ;;
*) sed -n 2,16p $0; exit 2 ;;
esac
