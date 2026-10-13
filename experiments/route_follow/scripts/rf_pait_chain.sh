#!/usr/bin/env bash
# Turning school on1_PAIT, full build on the Tokyo box (rf_pait.py docstring): the PAI scenes without log-following renders are rolled along
# the log (onp_pass2.sh MODE=driver, P2H10S-F-s0, card 0 only), turned into rows (onp_rows.sh) and tokens (onp_cache.py build + gid-ulp),
# given their route messages (rf_pait.py route), packed (PAITn), merged with the 80 scenes of /data/runs/onpol_pai pass log (PAIT80) into
# on1_PAIT, copied to the main box and merged there with on1_CFTv into on1_PAITC; then the four registered arms with their held / guard reads
# are submitted there (rf_tschool.sh submit) and /data/runs/route_night/EXAM_READ_REQUEST is written for the Tokyo exam reads. Touches
# /data/runs/route_night/CARD0_FREE2 when it exits after having taken card 0. Starts only when /data/runs/route_night/CARD0_FREE exists (the spin-fix lane holds card 0).
#   rf_pait_chain.sh <scene list>        e.g. /data/runs/route_night/pait/roll.txt (priority scenes first)
# Env: C (code snapshot, /data/runs/route_night/pait/code), W (work root, /data/runs/route_night/pait), BOX (rsync target host, autodl),
#   BOXD (box DATA_DIR, /root/autodl-tmp/ujs). State: $W/CHAIN_STATUS, CHAIN_DONE | CHAIN_ERROR, chain.log.
set -uo pipefail
L=$(readlink -f "$1"); D=${DATA_DIR:-/data}; C=${C:-$D/runs/route_night/pait/code}; W=${W:-$D/runs/route_night/pait}
BOX=${BOX:-autodl}; BOXD=${BOXD:-/root/autodl-tmp/ujs}; PY=$D/envs/op-train/bin/python; O=$W/out
mkdir -p "$W" "$O"; rm -f "$W"/CHAIN_{DONE,ERROR}; exec > >(tee -a "$W/chain.log") 2>&1
st() { echo "$(date '+%F %T') $*" | tee "$W/CHAIN_STATUS"; }
die() { st "ERROR $*"; echo "$*" > "$W/CHAIN_ERROR"; exit 1; }
st "waiting for $D/runs/route_night/CARD0_FREE"
until [[ -f $D/runs/route_night/CARD0_FREE ]]; do sleep 60; done
trap 'touch $D/runs/route_night/CARD0_FREE2' EXIT                    # card 0 handed on (lane P2 waits on it), on success or error
st "roll: $(wc -l < "$L") scenes along the log on card 0"
# the onpol_pai dev mirror's onp_pass2.sh + onp_log_driver.py and a clone of the pass-2 code: the 80 log scenes' exact serving; the copy
# only adds RUNPFX to the run dir names (pt-...: compose project / container names cannot clash with other reads, pai_eval.sh's RUNPFX)
X=$W/onp; mkdir -p "$X/scripts" "$X/lib"; cp "$D/runs/onpol_pai/dev/lib/onp_log_driver.py" "$X/lib/" || die "copy log driver"
sed 's/L=${T,,}/L=${RUNPFX:-}${T,,}/' "$D/runs/onpol_pai/dev/scripts/onp_pass2.sh" > "$X/scripts/onp_pass2.sh"
grep -qF 'L=${RUNPFX:-}${T,,}' "$X/scripts/onp_pass2.sh" || die "RUNPFX patch of onp_pass2.sh"
RUNPFX=pt- CODE=$W/code_roll REPO=$D/runs/onpol_pai/code/pass2 CARDS=0 MODE=driver CHUNK=40 bash "$X/scripts/onp_pass2.sh" "$W/pass" P2H10S-F-s0 "$L" || die "onp_pass2"
st "rows"
ONP_OUT=$W/rows JEV=$D/runs/onpol_pai/code/pass2 bash "$D/runs/onpol_pai/dev/scripts/onp_rows.sh" pait "$W/pass/runs/*" || die "onp_rows"
st "tokens + teacher (card 0)"
cd "$C" || die "cd $C"
CUDA_VISIBLE_DEVICES=0 DATA_DIR=$D $PY experiments/onpol_pai/scripts/onp_cache.py build --pass pait --rows "$W/rows/pait" --out "$W/cache" || die "onp_cache build"
CUDA_VISIBLE_DEVICES=0 DATA_DIR=$D $PY experiments/onpol_pai/scripts/onp_cache.py gid-ulp --pass pait --out "$W/cache" --n-ulp 4 || st "gid-ulp line not met (recorded, chain continues)"
st "route messages"
docker run --rm --network none --user 0:0 --entrypoint bash -e DATA_DIR=$D -e PYTHONPATH=$D/third_party/alpasim/src/utils:/jev \
  -v $D:$D:ro -v $D/datasets/nurec:/mnt/nre-data:ro -v "$O:$O" -v "$C:/jev:ro" -w /repo alpasim-base:0.89.0 \
  -c "umask 0000; /repo/.venv/bin/python /jev/experiments/route_follow/scripts/rf_pait.py route --rows $W/rows/pait --out $O --workers 8" || die "route"
st "pack + merge"
SC=("$D/runs/route_night/train160.txt" "$O/hard_train.txt")
DATA_DIR=$D $PY experiments/route_follow/scripts/rf_pait.py pack --name PAITn --cache "$W/cache/pait" --route "$O/route" --scenes "${SC[@]}" --hard "$O/hard_train.txt" --out "$O" || die "pack PAITn"
DATA_DIR=$D $PY experiments/route_follow/scripts/rf_pait.py r1 --name PAITn --cache "$W/cache/pait" --out "$O" --device cuda || st "r1 PAITn line not met (recorded)"
DATA_DIR=$D $PY experiments/route_follow/scripts/rf_pait.py merge --out PAIT --pools on1_PAIT80:a on1_PAITn:b || die "merge"
st "copy to $BOX"
for p in on1_PAIT on1_PAIT@warp; do rsync -a --partial "$D/runs/op_parity/cache/$p/" "$BOX:$BOXD/runs/op_parity/cache/$p/" || die "rsync $p"; done
rsync -a --partial "$D/runs/route_mem/feat/on1_PAIT.npz" "$BOX:$BOXD/runs/route_mem/feat/" || die "rsync feat"
ssh "$BOX" "mkdir -p $BOXD/runs/route_follow/pait" && rsync -a "$O"/gates_PAIT*.json "$BOX:$BOXD/runs/route_follow/pait/" || st "gates copy failed (not fatal)"
st "on1_PAITC = on1_PAIT + on1_CFTv on $BOX"
ssh "$BOX" "cd ~/data/jev-drive && git pull -q; export DATA_DIR=$BOXD && $BOXD/envs/op-train/bin/python experiments/route_follow/scripts/rf_pait.py merge --out PAITC --pools on1_PAIT:a on1_CFTv:b" || die "merge PAITC on $BOX"
st "submit the four registered arms + held / guard reads on $BOX (rf_tschool.sh)"
ssh "$BOX" "cd ~/data/jev-drive && git pull -q; export DATA_DIR=$BOXD && bash experiments/route_follow/scripts/rf_tschool.sh submit" > "$W/tschool_jobs.txt" || die "rf_tschool submit"
cat > "$D/runs/route_night/EXAM_READ_REQUEST" <<EOF
# Turning school exam read request (experiments/route_follow/plans/2026-10-13-turning-school-prereg.md), written $(date '+%F %T') by rf_pait_chain.sh
# checkpoints train on the main box (rf_tschool.sh, job ids in $W/tschool_jobs.txt): pull each when its ckpt-final.pt + route_mem.json exist
# scenes: experiments/alpasim/results/pai_exam/hard_exam.txt (72) + control.txt (75); never hard_train
# serving: controller.gains.rel_front_steering_angle_weight=20, JEV_VCONT=1.0 JEV_LEAD=1, the v6 spin-fix controller once it has passed its
#   guard (else the current one); RSG-L18-s0 and RNONEG-L18-s{0,1} under the identical serving (re-read if their stored read differs)
# order: primary arm first
RPAIT-L18-s0
RPAIT-L18-s1
RPAITC-L18-s0
RPAITC-L18-s1
RSG-L18-s0
RNONEG-L18-s0
RNONEG-L18-s1
EOF
date > "$W/CHAIN_DONE"; st "done: on1_PAIT on $BOX, arms submitted, EXAM_READ_REQUEST written"
