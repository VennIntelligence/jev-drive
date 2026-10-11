#!/usr/bin/env bash
# EYES-vs-SENSE (plans/2026-10-11-eyes-sense-prereg.md): every GPU job through the pool, one self-advancing chain.
#   1. identity runs of (a) / (c) with the stored checkpoints' flags (ESa / ESc, seeds 0 / 1, full length) and the 300-step pilots of (b) / (d)
#   2. pilot checklist of (b) / (d) -> their full runs (ESb / ESd, seeds 0 / 1)
#   3. identity rule: ESa vs P2H10-F, ESc vs P2H10S-F (experiments/alpasim/scripts/joint_train.py ident) -> results/identity.json, the tags of (a) / (c)
#   4. bench: navtest + navhard (G frames) for (b) / (d) and, if the rule fails, for ESa / ESc; navtest :noside / :mshuf / :sideoff for (b) / (d);
#      if the rule fails, the 700-scene nuPlan closed loop of ESa / ESc (c0b chunk0-2, ot2_loop, default driver path)
#   scripts/tmux_run.sh es-chain bash experiments/eyes_sense/scripts/es_chain.sh
# State: $DATA_DIR/runs/eyes_sense/chain/{STATUS, DONE | ERROR, log.txt, pool/<job>/}. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
D=$DATA_DIR/runs/eyes_sense/chain; L=$D/pool; mkdir -p "$L"; rm -f "$D/DONE" "$D/ERROR"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
VPY=$PWD/.venv/bin/python
CL="$VPY -m jevdrive.cl"
BN=("$VPY" -m jevdrive.bench)
R=experiments/eyes_sense/results; mkdir -p "$R"
status() { echo "$(date '+%F %T') eyes_sense: $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 v=$2; shift 2; [[ -f $L/$n/DONE ]] && return
        $CL queue 2>/dev/null | awk -v n="es-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
        $CL submit --owner eyes_sense --name "es-$n" --train --priority 15 --vram "$v" --cpu 6 --ram 60 --timeout-h 3 --log-dir "$L/$n" -- "$@" > /dev/null || die "submit $n"; }
waitdirs() { for n in "$@"; do until [[ -f $L/$n/DONE || -f $L/$n/ERROR ]]; do sleep 30; done; [[ -f $L/$n/ERROR ]] && die "job failed: $L/$n"; done; return 0; }
S="--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape"
JT="$PY experiments/alpasim/scripts/joint_train.py train --exp eyes_sense --label-bank"
status "code $(git rev-parse --short HEAD); identity runs (a) / (c), pilots (b) / (d)"
for s in 0 1; do sub "id-a-s$s" 30 $JT --seed $s --tag "ESa-s$s"; sub "id-c-s$s" 36 $JT $S --seed $s --tag "ESc-s$s"; done
sub p-b 30 $JT --mem vtr_W0 --seed 0 --steps 300 --eval-every 300 --tag ESb-P300
sub p-d 36 $JT $S --mem vtr_W0 --seed 0 --steps 300 --eval-every 300 --tag ESd-P300
waitdirs p-b p-d
for a in b d; do                                                       # pilot checklist: finite losses, the memory read back at the final dev evaluation
  $PY - "$a" "$R/pilot_$a.json" <<'PYEOF' || die "pilot checklist failed for ($a)"
import json, math, re, sys, glob, os
a, out = sys.argv[1], sys.argv[2]
log = open(sorted(glob.glob(os.path.expandvars(f"$DATA_DIR/runs/eyes_sense/joint-train-ES{a}-P300/*/log.txt")))[-1]).read()
st = re.findall(r"step (\d+): (.*?); ([\d.]+) it/s, ([\d.]+) GB", log)[-1]
vals = dict((k.strip(), float(v)) for k, v in (x.rsplit(" ", 1) for x in st[1].split(", ")))
dev = dict((k.strip(), float(v)) for k, v in (x.rsplit(" ", 1) for x in re.findall(r"dev @ 300: (.*)", log)[-1].split(", ")))
comp = re.search(r"per batch of (\d+): (\d+) normal, (\d+) yaw-rate, (\d+) hinge-only", log).groups()
need = ["mem_ade_on", "mem_ade_masked", "mem_ade_shuf"]
fail = [] if all(math.isfinite(v) for v in vals.values()) else ["non-finite loss"]
fail += [f"dev lacks {k}" for k in need if k not in dev]
if not fail and dev["mem_ade_masked"] == dev["mem_ade_on"] and dev["mem_ade_shuf"] == dev["mem_ade_on"]:
    fail.append("masking / shuffling the memory leaves the dev ADE unchanged")
if a == "d" and comp != ("128", "115", "0", "13"):
    fail.append(f"composition {comp}")
res = dict(arm=a, composition=comp, losses=vals, dev=dev, peak_gb=float(st[3]), it_s=float(st[2]), fail=fail, ok=not fail,
           hinge_only_memory="masked by construction (joint_train.py draw(): sm[nN + k_yr:] = False)" if a == "d" else "n/a")
open(out, "w").write(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
sys.exit(1 if fail else 0)
PYEOF
done
# the vtr_W0 bank: rebuilt after the disk cleanup of 2026-10-11 (scripts/es_bank.py docstring) unless all 14 data dirs are in its bank.json
BANK=$DATA_DIR/runs/op_parity/mem/vtr_W0
DATAS="$(for i in $(seq 0 11); do echo -n "navtrain_full.s${i}of12 "; done)lb_navtest lb_navhard"
subx() { local n=$1; shift; [[ -f $L/$n/DONE ]] && return
         $CL queue 2>/dev/null | awk -v n="es-$n" '$4 == n && ($2 == "queued" || $2 == "running") {f = 1} END {exit !f}' && return; rm -f "$L/$n/ERROR"
         $CL submit --owner eyes_sense --name "es-$n" --priority 15 --timeout-h 3 --log-dir "$L/$n" "$@" > /dev/null || die "submit $n"; }
if ! python3 -c "import json, sys; sys.exit(0 if len(json.load(open('$BANK/bank.json'))['rows']) == 14 else 1)" 2>/dev/null; then
  status "rebuilding the vtr_W0 bank: side-view cache (vt_side.py split + build), then es_bank.py"
  subx side-split --vram 0 --cpu 2 --ram 48 -- $PY experiments/vis_train/scripts/vt_side.py split --datas $DATAS
  waitdirs side-split
  for d in $DATAS; do subx "side-build-$d" --vram 0 --cpu 6 --ram 16 -- $PY experiments/vis_train/scripts/vt_side.py build --data "$d"; done
  waitdirs $(for d in $DATAS; do echo "side-build-$d"; done)
  subx bank --vram 12 --cpu 4 --ram 40 -- $PY experiments/eyes_sense/scripts/es_bank.py
  waitdirs bank
fi
status "pilots passed; full runs (b) / (d)"
for s in 0 1; do sub "f-b-s$s" 30 $JT --mem vtr_W0 --seed $s --tag "ESb-s$s"; sub "f-d-s$s" 36 $JT $S --mem vtr_W0 --seed $s --tag "ESd-s$s"; done
waitdirs id-a-s0 id-a-s1 id-c-s0 id-c-s1
status "identity rule"
A=P2H10-F; C=P2H10S-F; ida=1; idc=1
for s in 0 1; do
  $PY experiments/alpasim/scripts/joint_train.py ident --a "ESa-s$s" --b "P2H10-F-s$s" --out "$R/identity_a_s$s.json" > /dev/null || ida=0
  $PY experiments/alpasim/scripts/joint_train.py ident --a "ESc-s$s" --b "P2H10S-F-s$s" --out "$R/identity_c_s$s.json" > /dev/null || idc=0
done
[[ $ida == 1 ]] || A=ESa; [[ $idc == 1 ]] || C=ESc
echo "{\"a\": \"$A\", \"c\": \"$C\", \"a_identical\": $ida, \"c_identical\": $idc}" > "$R/identity.json"
status "identity: (a) = $A, (c) = $C; waiting for the full runs of (b) / (d)"
waitdirs f-b-s0 f-b-s1 f-d-s0 f-d-s1
NEW="ESb-s0 ESb-s1 ESd-s0 ESd-s1"; EXTRA=""
[[ $A == ESa ]] && EXTRA="$EXTRA ESa-s0 ESa-s1"; [[ $C == ESc ]] && EXTRA="$EXTRA ESc-s0 ESc-s1"
OPT=$(for t in $NEW; do echo -n "$t:noside $t:mshuf $t:sideoff "; done)
status "bench: navtest + navhard for $NEW$EXTRA, navtest options"
"${BN[@]}" run --model $NEW $EXTRA $OPT --bench navtest --priority 12 || die "bench navtest"
"${BN[@]}" run --model $(for t in $NEW $EXTRA; do echo -n "$t@gimm "; done) --bench navhard --priority 12 || die "bench navhard"
if [[ -n $EXTRA ]]; then                                               # the identity rule failed: (a) / (c) need their own 700-scene closed loop
  J=(); for t in $EXTRA; do J+=("$t:sh30:SH30_TAG=$t:$t"); done
  OT_LANE=eyes_sense OT_OUT=$DATA_DIR/runs/eyes_sense OT_OWNER=eyes_sense OT_PRIO=13 OT2_MAX_ACTIVE=4 python3 experiments/alpasim/scripts/ot2_loop.py cl "${J[@]}" \
    || die "closed loop of $EXTRA"
fi
"${BN[@]}" status --model $NEW $EXTRA $OPT --bench navtest --wait || die "navtest"
"${BN[@]}" status --model $(for t in $NEW $EXTRA; do echo -n "$t@gimm "; done) --bench navhard --wait || die "navhard"
$CL usage --hours 24 > "$D/usage.txt" 2>&1 || true
status "all done"; date '+%F %T' > "$D/DONE"
