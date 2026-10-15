#!/usr/bin/env bash
# RECOVERY-ENVELOPE round (plans/2026-10-16-recovery-envelope-prereg.md): arms on the RSG-L18 recipe (rf_onk.sh REC)
#   L    = --env-lam W (on_envelope: asymmetric speed envelope) on pool on1_RCVL (on1_L18RB0 S rows + RSG-L18-s0's own envelope rows
#          [+ the pushed fast approaches on1_RCVfast]), --on-mode self
#   R0   = pool on1_RCVR0 (S rows self + d316 recovery rows both: the expert's served plan), --on-mode mixed
#   RL0  = R0 + L on one pool, --on-mode mixed --on-lon-all
#   LF   = L + the pushed fast approaches (on1_RCVfast, envelope-labelled self rows), pool on1_RCVLF
#   RLF  = RL + the pushed fast approaches: the combined candidate (Amendment 6), pool on1_RCVRLF
#   RLFD = RLF with the data-scaled recovery pool on1_RCVallD (pass 0 + the perturbed passes, Amendment 7; REC_POOL_D), pool on1_RCVRLFD;
#          the pack sets the per-row R weight from the pool's row count, so k6 = 6 recovery rows per batch as in RLF (same dose, more data)
#   R / RL = the same with the all-scene rewind's recovery rows (decision 319), pool on1_RCVR / on1_RCVRL (REC_POOL=<rec pool>)
# Box, repo root; every GPU job through the pool; ids in $O/jobs.txt.
#   rf_recov.sh gates                       bit identity (60 steps; old code = worktree at OLD_REV): G0 every new flag off old vs new, G0 repeat
#   rf_recov.sh gates2                      G2 --on-mode self + --env-lam vs --on-mode mixed --on-lon-all on the same all-self pool on1_RCVL
#   rf_recov.sh label <pool>                CPU job: rf_recov.py label (envelope labels of every row of on1_<pool>)
#   rf_recov.sh pack <arm>                  CPU job: rf_recov.py pack -> on1_RCV<arm> + $O/flags_RCV<arm>.txt
#   rf_recov.sh base                        fwd jobs of the reference plans (RSG-L18-s0; APP1X-L18-s0 on the held decisions 0 / 1: instrument check)
#   rf_recov.sh pilot <arm> [W]             1 000-step pilot RCV<arm>P[W]-L18-s0 (+ fwd reads + gate job, all --after)
#   rf_recov.sh full <arm> [W]              2 x 2 500-step trainings RCV<arm>-L18-s{0,1} + rf_recov_post.sh submit (held, guard, PAI s0, report)
#   rf_recov.sh train <arm> <seed> <steps> <tag> [W] | smoke <old|new> <kind> | gatecmp      (job bodies)
# Env: PRIO (pool priority, default 2; pilots 7: four-minute jobs ahead of the long rewind lane), OLD_REV (default 6a2fdac6 = the prereg commit, before any trainer change), AFTER (ids gating pilot/full)
set -uo pipefail; cd "$(dirname "$0")/../../.."; DATA_DIR=${DATA_DIR:-/data}
RF=$DATA_DIR/runs/route_follow; O=$RF/recovery_envelope; CK=$DATA_DIR/runs/op_parity/runs
TR=$DATA_DIR/envs/op-train/bin/python; ME=experiments/route_follow/scripts/rf_recov.sh; PY=experiments/route_follow/scripts/rf_recov.py
OLD_REV=${OLD_REV:-6a2fdac6}; OLD=$O/code_old
REC="--init RB0-F-s0 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train"
RB="--on on1_L18RB0 --on-k 13 --on-mode self"
mkdir -p $O/pool
sub() { local name=$1; shift; local id; id=$(.venv/bin/python -m jevdrive.cl submit --owner recovery-envelope --name "$name" --tries 1 --priority ${PRIO:-2} --no-check \
     --log-dir $O/pool/$name "$@" < /dev/null | awk '{print $NF}') && [[ -n $id ]] || { echo "submit $name failed" >&2; exit 1; }; echo "$id $name" >> $O/jobs.txt; echo "$id"; }
pool_of() { case $1 in L) echo RCVL ;; LF) echo RCVLF ;; R0) echo RCVR0 ;; RL0) echo RCVRL0 ;; R) echo RCVR ;; RL) echo RCVRL ;; RLF) echo RCVRLF ;; RLFD) echo RCVRLFD ;; *) echo "no arm $1" >&2; exit 2 ;; esac; }
kind_of() { case $1 in L|LF) echo L ;; R0|R) echo R ;; RL0|RL|RLF|RLFD) echo RL ;; esac; }
rec_of() { case $1 in R0|RL0) echo ${REC_POOL0:-RCVd316} ;; R|RL|RLF) echo ${REC_POOL:-RCVall} ;; RLFD) echo ${REC_POOL_D:-RCVallD} ;; *) echo "" ;; esac; }
flags() {  # arm [W] -> the arm's trainer flags
  local f=$O/flags_$(pool_of $1).txt; [[ -s $f ]] || { echo "no $f (rf_recov.sh pack $1)" >&2; exit 1; }
  local F W=${2:-}; F=$(cat $f)
  if [[ $W == *u ]]; then W=${W%u}; F=${F/--on-weight /--on-weight env=free:0,}; fi    # diagnostic: upper bounds only (free rows never drawn)
  if [[ $W == k* ]]; then                                            # R dose: k<n>[_<L weight>]: n recovery rows per batch instead of the pack's 3
    local KR=${W%%_*}; [[ $W == *_* ]] && W=${W#*_} || W=""
    F=$(python3 -c "import re,sys; f,k=sys.argv[1],int(sys.argv[2]); m=re.search(r'pool=r:([0-9.]+)',f); n=re.search(r'--on-k ([0-9]+)',f)
print(f.replace(m.group(0),'pool=r:%.5f'%(float(m.group(1))*k/3)).replace(n.group(0),'--on-k %d'%(int(n.group(1))+k-3)))" "$F" ${KR#k}); fi
  if [[ $W == *l* ]]; then F="$F --env-lo-scale $(python3 -c "print(round(float('${W#*l}') / float('${W%l*}'), 6))")"; W=${W%l*}; fi   # <up>l<lo>
  [[ $(kind_of $1) != R ]] && F="$F --env-lam ${W:?W}"; echo "$F"; }
FW=(--train --vram 24 --cpu 6 --ram 70)                              # measured peak 19.6 GB (rcv-pilot history)
case ${1:-} in
gates)
  [[ -d $OLD ]] || git worktree add --detach $OLD $OLD_REV || exit 1
  ids=()
  for x in old:off new:off new:off2; do
    ids+=($(sub rcv-g-${x/:/-} "${FW[@]}" --timeout-h 1 -- bash $ME smoke ${x%%:*} ${x#*:})) || exit 1; done
  sub rcv-gatecmp --after "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME gatecmp > /dev/null || exit 1
  tail -n $(( ${#ids[@]} + 1 )) $O/jobs.txt ;;
gates2)  # G2 (after on1_RCVL exists: AFTER=<pack id>) + the compare again
  ids=()
  for k in envself envmixall; do ids+=($(sub rcv-g-new-$k "${FW[@]}" --timeout-h 1 ${AFTER:+--after $AFTER} -- bash $ME smoke new $k)) || exit 1; done
  sub rcv-gatecmp2 --after "$(IFS=,; echo "${ids[*]}")" --vram 0.5 --cpu 4 --ram 16 --timeout-h 1 -- bash $ME gatecmp > /dev/null || exit 1
  tail -n 3 $O/jobs.txt ;;
smoke)   # smoke <old|new> <off|off2|envself|envmixall>: 60 steps of the RSG-L18 recipe, tag RCVG-<code>-<kind>
  C=.; [[ $2 == old ]] && C=$OLD
  P=$(cat $O/flags_RCVL.txt 2>/dev/null | sed 's/--on-mode self//')
  case $3 in off|off2) F="$RB" ;; envself) F="$P --on-mode self --env-lam 1" ;; envmixall) F="$P --on-mode mixed --on-lon-all --env-lam 1" ;; esac
  exec $TR $C/experiments/route_mem/scripts/rm_train.py train --tag RCVG-$2-$3 $REC $F --steps 60 --seed 0 ;;
gatecmp)
  $TR - "$DATA_DIR" "$O" <<'PY'
import glob, json, os, sys
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
def cmp(a, b, note, loss_only=False):
    (sa, da, ma), (sb, db, mb) = run(a), run(b)
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith(("throughput/", "loss/drop/"))]
    if loss_only:                                                  # G2: the modes log different per-family scalars; the loss and dev reads must agree
        keys = [k for k in keys if k[0].startswith(("loss/total", "dev/", "loss/on/env", "on/env"))] or keys
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g])
    same = all(set(ma[g]) == set(mb[g]) for g in ("net", "parity"))
    r = dict(a=a, b=b, note=note, scalars=len(keys), scalar_max_abs_diff=dl, weight_max_abs_diff=dw, weight_keys_equal=same,
             on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")], dev_ade=[da.get("dev_ade"), db.get("dev_ade")])
    r["identical"] = bool(dl == 0 and dw == 0 and same and r["on_hash_60"][0] == r["on_hash_60"][1])
    return r
R = dict(G0_everything_off=cmp("RCVG-old-off", "RCVG-new-off", "old code (prereg commit) vs new code, every new flag off (RSG-L18 recipe, 60 steps)"),
         G0_repeat=cmp("RCVG-new-off", "RCVG-new-off2", "new code twice"))
R["G0_everything_off"]["rsg_l18_s0_on_hash_60"] = RSG_HASH60
R["G0_everything_off"]["identical"] &= R["G0_everything_off"]["on_hash_60"][0] == RSG_HASH60
if os.path.exists(f"{D}/runs/op_parity/runs/RCVG-new-envmixall/ckpt-final.pt"):
    R["G2_lon_all"] = cmp("RCVG-new-envself", "RCVG-new-envmixall", "--env-lam 1 on on1_RCVL: --on-mode self vs --on-mode mixed --on-lon-all (every row self)", True)
R["pass"] = all(v["identical"] for k, v in R.items() if isinstance(v, dict))
json.dump(R, open(f"{O}/gates.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
raise SystemExit(0 if R["pass"] else 2)
PY
  ;;
label)
  sub rcv-label-$2 --vram 0.5 --cpu 32 --ram 64 --timeout-h 1 ${AFTER:+--after $AFTER} -- $TR $PY label --pool $2 --workers 32 ;;
pack)
  A=$2; K=$(kind_of $A); EP=${ENV_POOLS:-L18RSG$([[ $A == LF || $A == RLF || $A == RLFD ]] && echo ,RCVfast)}
  sub rcv-pack-$A --vram 0.5 --cpu 8 --ram 96 --timeout-h 1 ${AFTER:+--after $AFTER} -- $TR $PY pack --arm $K --out $(pool_of $A) --env-pools $EP --rec "$(rec_of $A)" ;;
base)    # reference plans: RSG-L18-s0 (+ APP1X-L18-s0 on the held decisions 0 / 1, the turning instrument's wrong-model check)
  b1=$(sub rcv-fwd-base-held01 --vram 20 --cpu 6 --ram 64 --timeout-h 1 ${AFTER:+--after $AFTER} -- $TR $PY fwd --pool L18RSGallheld --rows held01 --tags RSG-L18-s0,APP1X-L18-s0) || exit 1
  b2=$(sub rcv-fwd-base-devenv --vram 20 --cpu 6 --ram 64 --timeout-h 1 ${AFTER:+--after $AFTER} -- $TR $PY fwd --pool L18RSG --rows dev-env --tags RSG-L18-s0) || exit 1
  echo "$b1 $b2" ;;
pilot)   # pilot <arm> [W]
  A=$2; W=${3:-}; K=$(kind_of $A); T=RCV${A}P${W//./}-L18-s0; RP=$(rec_of $A)
  [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
  t=$(sub rcv-pilot-$T "${FW[@]}" --timeout-h 1 ${AFTER:+--after $AFTER} -- bash $ME train $A 0 1000 $T $W) || exit 1
  f=($(sub rcv-fwd-$T-held01 --after $t --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY fwd --pool L18RSGallheld --rows held01 --tags $T)) || exit 1
  [[ $K != R ]] && { f+=($(sub rcv-fwd-$T-devenv --after $t --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY fwd --pool L18RSG --rows dev-env --tags $T)) || exit 1; }
  [[ $K != L ]] && { f+=($(sub rcv-fwd-$T-devrec --after $t --vram 20 --cpu 6 --ram 64 --timeout-h 1 -- $TR $PY fwd --pool $RP --rows dev-rec --tags RSG-L18-s0,$T)) || exit 1; }
  sub rcv-gate-$T --after "$(IFS=,; echo "${f[*]}")" --vram 0.5 --cpu 4 --ram 32 --timeout-h 1 -- $TR $PY gate --tag $T --arm $K --rec "${RP:-none}" > /dev/null || exit 1
  tail -n $(( ${#f[@]} + 2 )) $O/jobs.txt ;;
full)    # full <arm> [W]: both seeds' trainings, then the post-training chain (rf_recov_post.sh)
  A=$2; W=${3:-}; ids=(); N=${STEPS:-2500}; X=${STEPS:+$((STEPS / 1000))k}; SEEDS=${SEEDS:-0 1}   # STEPS: step-count arm RCV<arm><N/1000>k
  for s in $SEEDS; do T=RCV$A$X-L18-s$s; [[ ! -e $CK/$T/ckpt-final.pt ]] || { echo "$T exists" >&2; exit 1; }
    ids+=($(sub rcv-train-$T "${FW[@]}" --timeout-h $(( N / 2500 * 2 + 1 )) ${AFTER:+--after $AFTER} -- bash $ME train $A $s $N $T $W)) || exit 1; done
  echo "$A$X W=${W:-none} steps $N seeds $SEEDS trainings ${ids[*]}" >> $O/full.txt
  set -- $SEEDS; T1=-; J1=-; (( $# > 1 )) && { T1=RCV$A$X-L18-s$2; J1=${ids[1]}; }
  bash experiments/route_follow/scripts/rf_recov_post.sh submit RCV$A$X RCV$A$X-L18-s$1 $T1 ${ids[0]} $J1 ;;
train)   # train <arm> <seed> <steps> <tag> [W]
  F=$(flags $2 ${6:-}) || exit 1
  exec $TR experiments/route_mem/scripts/rm_train.py train --tag $5 $REC $F --steps $4 --seed $3 ;;
*) sed -n 2,20p $0; exit 2 ;;
esac
