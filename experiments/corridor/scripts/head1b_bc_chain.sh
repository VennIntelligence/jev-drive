#!/usr/bin/env bash
# HEAD1b steps B and C (experiments/corridor, amendment "补记 2026-10-10（HEAD1b）" of plans/2026-10-10-head1-prereg.md). EXPLORATORY: a second
# attempt after the missed registered gate G3 (user-authorised); nothing here is a registered read.
# One self-advancing chain in tmux jev (scripts/tmux_run.sh h1b-bc bash experiments/corridor/scripts/head1b_bc_chain.sh): it only submits pool
# jobs and waits on their DONE / ERROR; every navtest read through jevdrive.bench.
#   stage 0  code-path smoke on a SMOKE input ($O/smoke/prof: logged-path labels / the stage-1 head; no read touches it), the identity check of
#            the memory channel, and step C's missing no-memory baseline seed (P2H10S-P-s1; it needs no head)
#   stage 1  gated on $H1/final/DONE (the step-A heads): the head on P2H10S's hinge-only off-track rows, the qp tokenizer, then
#            B: HP / HPX x seeds 0, 1 on decision 204's SH30 pilot recipe (baseline GH0-F-s0/s1 reused)
#            C: the qp memory on the P2H10S pilot recipe x seeds 0, 1 (baseline P2H10S-P-s0 reused + s1)
#   stage 2  navtest (bench), the turn replay, the two reports; this lane's navtest memory banks are dropped after scoring
# State: $DATA_DIR/runs/corridor/head1b/chain/{STATUS, DONE | ERROR, log.txt, jobs.txt}; reports in $DATA_DIR/runs/corridor/head1b/report. Rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=$DATA_DIR/runs/corridor/head1b
D=$O/chain; L=$D/pool; mkdir -p "$L" "$O/report"; rm -f "$D/DONE" "$D/ERROR"; touch "$D/jobs.txt"
exec > >(tee -a "$D/log.txt") 2>&1
PY=$DATA_DIR/envs/op-train/bin/python
NAV=$DATA_DIR/envs/navsim2/bin/python
VPY=$PWD/.venv/bin/python
CL="$DATA_DIR/envs/jevdrive/bin/python -m jevdrive.cl"
SP=experiments/op_parity/scripts; SC=experiments/corridor/scripts; SB=experiments/body1/scripts
B=("$VPY" -m jevdrive.bench)
H1=$DATA_DIR/runs/corridor/head1; FIN=$H1/final/DONE
PR=$DATA_DIR/runs/op_parity/path_req; MEM=$DATA_DIR/runs/op_parity/mem; RUNS=$DATA_DIR/runs/op_parity
SMK=$O/smoke/prof
BDATA="navtrain_full.s2of12 navtrain_full.s3of12 navtrain_full.s4of12"
BF="--arm P2 --frames warp --host --data $BDATA --split navsim/op-parity-s234 --batch 64 --warmup 100 --hinge-lam 30 --hinge-margin 0.5"   # decision 204's SH30 pilot
CF="--data navtrain_full.s2of12 navtrain_full.s3of12 --ho ot1:4,yr1:4,bd4:5 --agent-lam 10 --ho-w 3 --ho-excl navsim/body1-val-logs --shape"   # P2H10S-P-s0's flags
BARMS=(HP-F-s0 HP-F-s1 HPX-F-s0 HPX-F-s1); CARMS=(P2H10S-HP-P-s0 P2H10S-HP-P-s1)
status() { echo "$(date '+%F %T') head1b B/C (exploratory): $*" | tee "$D/STATUS"; }
die() { status "ERROR $*"; echo "$*" > "$D/ERROR"; exit 1; }
sub() { local n=$1 ld=$2; shift 2; [[ -f $ld/DONE ]] && return
        local live; live=$($CL queue 2>/dev/null | awk -v n="$n" '($2 == "queued" || $2 == "running") && ($3 == n || $4 == n) {print $1; exit}')
        [[ -n $live ]] && return; rm -f "$ld/ERROR"
        local id; id=$($CL submit --owner corridor-head1b --name "$n" --log-dir "$ld" "$@") || die "submit $n"; echo "$id $n" >> "$D/jobs.txt"; }
waitdirs() { for ld in "$@"; do until [[ -f $ld/DONE || -f $ld/ERROR ]]; do sleep 20; done; [[ -f $ld/ERROR ]] && die "job failed: $ld/ERROR"; done; return 0; }
summary() { cat "$(ls -d "$RUNS/train-$1"/*/ | tail -1)DONE"; }

for t in GH0-F-s0 GH0-F-s1 P2H10S-P-s0; do [[ -f $RUNS/runs/$t/ckpt-final.pt ]] || die "missing reused checkpoint $t"; done
for d in $BDATA lb_navtest; do [[ -f $MEM/geo_x/$d.perm.npy ]] || die "missing geo_x permutation of $d"; done

# ---------------------------------------------------------------- stage 0
sub h1b-c-base-s1 $L/t-P2H10S-P-s1 --train --vram 20 --cpu 4 --ram 40 --timeout-h 1 -- $PY $SB/bd4_train.py train --seed 1 $CF --steps 3000 --eval-every 1000 --tag P2H10S-P-s1
sub h1b-ident $L/ident --vram 8 --cpu 4 --ram 32 --timeout-h 1 -- $PY $SP/path_req.py ident --tag GH0-F-s0
if [[ ! -f $D/SMOKE_OK ]]; then
  status "stage 0: smoke on the SMOKE input $SMK"
  $PY $SP/path_req.py selftest || die "path_req selftest"
  $PY $SC/head1b_prof.py smoke || die "smoke profiles"
  S1=$(ls $H1/train-L-f0-p1000-s0/*/ckpt.pt | tail -1)
  E="--env H1B_PROF=$SMK"
  sub h1b-smoke-ot $L/smoke-ot --vram 8 --cpu 8 --ram 32 --timeout-h 1 -- $PY $SC/head1b_prof.py ot --smoke --out $SMK --ckpt all=$S1
  sub h1b-smoke-tok $L/smoke-tok $E --vram 4 --cpu 2 --ram 16 --timeout-h 1 -- $PY $SP/path_req.py tok --kind qp --smoke
  for k in qp qpx; do
    sub h1b-smoke-$k $L/smoke-$k $E --train --vram 24 --cpu 4 --ram 40 --timeout-h 1 -- $PY $SP/pp_train.py $BF --mem-e2e $k --mem-init $PR/tok/qs.pt --seed 0 --steps 30 --eval-every 30 --tag H1BSMOKE-$k
  done
  waitdirs $L/smoke-ot
  sub h1b-smoke-c $L/smoke-c $E --train --vram 32 --cpu 4 --ram 40 --timeout-h 1 -- $PY $SB/bd4_train.py train --seed 0 $CF --steps 60 --eval-every 30 --mem-e2e qp --mem-init $PR/tok/qs.pt --tag H1BSMOKE-C
  waitdirs $L/smoke-tok $L/smoke-qp $L/smoke-qpx $L/smoke-c
  for t in H1BSMOKE-qp H1BSMOKE-qpx H1BSMOKE-C; do
    [[ -f $MEM/ge_$t/lb_navtest.npy ]] || die "smoke $t wrote no navtest bank"
    summary $t | grep -q '"e2e_dev_ade_mismatched"' || die "smoke $t has no dev diagnostics"
    rm -rf "$MEM/ge_$t" "$RUNS/runs/$t"
  done
  date > "$D/SMOKE_OK"
fi
waitdirs $L/ident
cp $PR/ident_GH0-F-s0.json $O/report/ident.json

# ---------------------------------------------------------------- stage 1: everything below waits for the heads
status "stage 1: jobs queued behind $FIN (off-track profiles, tokenizer, B: ${BARMS[*]}, C: ${CARMS[*]})"
sub h1b-ot $L/ot --when-exists $FIN --vram 8 --cpu 8 --ram 32 --timeout-h 1 -- $PY $SC/head1b_prof.py ot --match-final --out $O/prof_ot
sub h1b-tok-qp $L/tok-qp --when-exists $FIN --vram 4 --cpu 2 --ram 16 --timeout-h 1 -- $PY $SP/path_req.py tok --kind qp
for s in 0 1; do
  sub h1b-t-HP-F-s$s $L/t-HP-F-s$s --when-exists $PR/tok/qp.pt --train --vram 24 --cpu 4 --ram 40 --timeout-h 1 -- $PY $SP/pp_train.py $BF --mem-e2e qp --mem-init $PR/tok/qp.pt --seed $s --steps 3000 --eval-every 1000 --tag HP-F-s$s
  sub h1b-t-HPX-F-s$s $L/t-HPX-F-s$s --when-exists $PR/tok/qp.pt --train --vram 24 --cpu 4 --ram 40 --timeout-h 1 -- $PY $SP/pp_train.py $BF --mem-e2e qpx --mem-init $PR/tok/qp.pt --seed $s --steps 3000 --eval-every 1000 --tag HPX-F-s$s
done
AFT=(); [[ -f $L/ot/DONE ]] || AFT=(--after "$(awk '$2 == "h1b-ot" {i = $1} END {print i}' "$D/jobs.txt")")
for s in 0 1; do
  sub h1b-t-P2H10S-HP-P-s$s $L/t-P2H10S-HP-P-s$s --when-exists $PR/tok/qp.pt "${AFT[@]}" --train --vram 32 --cpu 4 --ram 40 --timeout-h 1 -- $PY $SB/bd4_train.py train --seed $s $CF --steps 3000 --eval-every 1000 \
      --mem-e2e qp --mem-init $PR/tok/qp.pt --tag P2H10S-HP-P-s$s
done
waitdirs $L/ot $L/tok-qp $L/t-P2H10S-P-s1 $(for t in "${BARMS[@]}" "${CARMS[@]}"; do echo $L/t-$t; done)
cp $O/prof_ot/quality.json $O/report/c_offtrack_profile_quality.json; cp $PR/tok/qp.json $O/report/tok_qp.json
for t in "${BARMS[@]}"; do $PY $SP/pp_full_check.py train --tag $t || die "training sanity $t"; done
for t in "${BARMS[@]}" "${CARMS[@]}"; do
  summary $t | grep -q "head1/final/prof" || die "$t did not read the final profiles"
  [[ -f $MEM/ge_$t/lb_navtest.npy || -f $DATA_DIR/runs/bench/navtest/$t@warp/DONE ]] || die "no navtest bank for $t"
done

# ---------------------------------------------------------------- stage 2
UN=("${BARMS[@]}" "${CARMS[@]}" P2H10S-P-s0 P2H10S-P-s1)
ALL=("${UN[@]}" HP-F-s0:noside HP-F-s1:noside P2H10S-HP-P-s0:noside P2H10S-HP-P-s1:noside)
status "stage 2: navtest: ${ALL[*]}"
"${B[@]}" run --model "${ALL[@]}" --bench navtest || die "bench navtest"
"${B[@]}" status --model "${ALL[@]}" --bench navtest --wait || die "navtest"
sub h1b-replay $L/replay --vram 0.5 --cpu 36 --ram 64 -- $NAV $SP/turn_oracle.py replay --name h1b --models "${UN[@]}" GH0-F-s0 GH0-F-s1
waitdirs $L/replay
rm -rf $L/report-b $L/report-c
sub h1b-report-b $L/report-b --vram 0.5 --cpu 8 --ram 32 -- $PY $SC/head1_pilot_report.py --name b --base H0=GH0-F-s0,GH0-F-s1 --arms HP=HP-F-s0,HP-F-s1 HPX=HPX-F-s0,HPX-F-s1 \
    --shuffled HP=HPX --lines memory --replays h1b --prof $H1/final/prof --r2 --pred stage1-head-decision-243=0.48 --out $O/report
sub h1b-report-c $L/report-c --vram 0.5 --cpu 8 --ram 32 -- $PY $SC/head1_pilot_report.py --name c --base S=P2H10S-P-s0,P2H10S-P-s1 --arms SHP=P2H10S-HP-P-s0,P2H10S-HP-P-s1 \
    --lines memory --replays h1b --widening --prof $H1/final/prof --prof-data navtrain_full.s2of12 navtrain_full.s3of12 --prof-split navsim/op-parity-full-train --out $O/report
waitdirs $L/report-b $L/report-c
for t in "${BARMS[@]}" "${CARMS[@]}"; do rm -rf "$MEM/ge_$t"; done      # this lane's own navtest banks, after scoring
$CL usage --hours 12 > "$D/usage.txt" 2>&1 || true
status "done"; date '+%F %T' > "$D/DONE"
