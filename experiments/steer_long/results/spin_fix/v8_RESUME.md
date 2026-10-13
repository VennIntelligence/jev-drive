# SPIN-FIX v8: resume after the box reboot (2026-10-13 23:40 JST)

## State
- Code: v8 = `JEV_TRACK=gate` at **7d001d79** (v7 gate + PP: the arc from the car onto each plan node reached by 2 s, checked when
  v0 >= 5 m/s; + anchor guard: tilted pose |(qx, qy)| > 0.3 -> the controller's planar yaw). a1fbf3fd also had GATE_LO 1.5 for
  low-speed accelerating calls; dropped in 7d001d79 because on nuPlan held RSG those openings saved 2 scenes (see below).
  Off / time / curv / full unchanged (3000 random plans, identical arrays; gate with pp=0 = v7).
- PAI: `/data/runs/route_night/SPIN_READ_REQUEST2` written (84 scenes: 38 spin set then 46 controls, RSG-L18-s0, gains 20, VCONT 1.0, LEAD 1,
  JEV_TRACK=gate), code = clean Tokyo worktree `/data/runs/steer_long/spinfix/v8/wt` at 7d001d79, suggested output
  `/data/runs/route_night/r281/spin_read8`. The card-1 watcher runs it; nobody else starts Tokyo jobs. Read it with
  `sf_read.py --list spin_fix/spin_set.txt` / `--list spin_fix/ctrl_all.txt --base /data/runs/route_night/r281/rsg --arm v8=<out>`
  (or rsync to the box as for v7: `$DATA_DIR/runs/steer_long/spinfix/v8/pai/`).
- Offline expectation (sf_screen.py on v7's own closed loop): only 60cc35db and d8bfaaf2 change on the spin set, none of the 46 controls,
  1 / 452 RSG and 2 / 452 RSGboth nuPlan held rollouts. So the controls should read as v7 (zeros 7 vs 6: 268949d2 = gate at low speed,
  09528162 = curv re-timing; line 2 stays failed unless the 2 scenes move).
- Box reads done before the reboot (a1fbf3fd, i.e. with the dropped GATE_LO; not the final v8):
  `route_mem/read/RSG-L18-s0-sf8gate` (6/6 chunks): quick paired read on 460 scenes -0.0100 [-0.025, +0.003], zeros 38 vs 34 (v7 gate on
  the same count: -0.0055, 36). v8.0 vs v7 differ on 3 scenes; 2 of them (2021.09.16.17.40.09_..47e0ff3b, 2021.10.06.08.16.17_..69dec595)
  went 1 -> 0 because GATE_LO closed openings at 3.2-4.5 m/s. `RSGboth-L18-s0-sf8gate` 5/6 chunks (c5 cancelled): ignore.
  `smoke_iso` (a1fbf3fd tree with the pre-v8 files) exists; rerun it anyway.
- Cancelled before start for the reboot: smoke v8, RSGboth held c5, plain RSGboth rerun x6, public x10.

## Commands after the reboot (box, from ~/data/jev-drive, `export DATA_DIR=/root/autodl-tmp/ujs`)
```bash
git pull
S=$DATA_DIR/runs/steer_long/spinfix/v8
git -C $S/wt checkout -q --detach 7d001d79
git -C $S/wt_iso checkout -q -- . && git -C $S/wt_iso checkout -q --detach 7d001d79 && \
  git -C $S/wt_iso checkout -q a1fbf3fd^ -- experiments/alpasim/lib/track_ref.py experiments/alpasim/lib/pai_driver.py experiments/alpasim/lib/sh30_driver.py
sub() { .venv/bin/python -m jevdrive.cl submit --owner spin-fix --tries 1 --priority 14 "$@" | awk '{print $NF}'; }
# 1. smoke5 switch off: same tree, v8 files vs pre-v8 files, both on card 1, then compare all decisions' plans
a=$(sub --name sf81-smoke-iso --vram 24 --cpu 6 --ram 30 --timeout-h 1 --gpus 1 --pin-strict --log-dir $S/pool/smoke81-iso -- bash $S/wt_iso/experiments/route_mem/scripts/rm_chain.sh readlist RSG-L18-s0 $DATA_DIR/runs/onpol_long/lists/smoke5.txt $S/smoke81_iso)
sub --name sf81-smoke-v8 --vram 24 --cpu 6 --ram 30 --timeout-h 1 --gpus 1 --pin-strict --after $a --log-dir $S/pool/smoke81-v8 -- bash $S/wt/experiments/route_mem/scripts/rm_chain.sh readlist RSG-L18-s0 $DATA_DIR/runs/onpol_long/lists/smoke5.txt $S/smoke81_v8
# 2. held 452 @18, both policies (line 3), + a plain RSGboth rerun on the current tree as the perturbation-noise control
for T in RSG-L18-s0 RSGboth-L18-s0; do for i in 0 1 2 3 4 5; do sub --name sf81-held-$T-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 --env JEV_TRACK=gate --log-dir $S/pool/held81-$T-$i -- bash $S/wt/experiments/route_mem/scripts/rm_chain.sh read $T $i 6 sf81gate; done; done
for i in 0 1 2 3 4 5; do sub --name sf81-plainre-RSGboth-$i --vram 20 --cpu 6 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env CL_HIST_KEY=alpasim-read4 --log-dir $S/pool/plainre81-$i -- bash $S/wt/experiments/route_mem/scripts/rm_chain.sh read RSGboth-L18-s0 $i 6 sf81plain; done
# 3. public 1479 @10 (line 4)
for i in 0 1 2 3 4 5 6 7 8 9; do sub --name sf81-pub-$i --vram 32 --cpu 8 --ram 40 --timeout-h 2 --gpus 1,2,3,4,5 --pin-strict --env JEV_TRACK=gate --log-dir $S/pool/pub81-$i -- bash $S/wt/experiments/onpol_long/scripts/ol_chain.sh readlist RSG-L18-s0 $DATA_DIR/runs/onpol_long/lists/all_n10/c$i.txt 10 $S/read10_gate81/b$i; done
```
Then: `ol_confirm.py` as in v7 (`--runs <label>@held18=<6 run dirs> ... --base RSG-L18-s0 --pairs ...`; RSG side: RSG-L18-s0, -s1,
-sf7gate, -sf81gate; RSGboth side: RSGboth-L18-s0, -s1, -rtgate, -sf81gate, -sf81plain, paired against RSGboth-L18-s0; all10 as
v7_nuplan_pub.md), spins with `sf_detect.py` on the controller rows, `sf_share.py <gate>:<plain>` for firing share and cost split,
the smoke comparison of the two smoke81 dirs (plans of all decisions). Write `v8_nuplan_held.md`, `v8_nuplan_pub.md`, section 7 of
results/spin_fix.md, commit, push.

## Expectation stated before the reads
RSG held ~ v7 (mean -0.006, zeros +2: passes). RSGboth held ~ v7's -0.024 / +8 (fails), unless the plain rerun shows the same
spread: 80 % of v7's RSGboth cost sits in rollouts where the gate never opened, and the served reference equals the plan's 8 points within
3 cm on 99 % of calls, so the v8 gate change cannot move it; if it is real, the cause is curv's re-timing, which the PAI spins need
(the plain-by-default screen, v8_screen.txt, leaves 3-8 of 26 spins without a trigger in the second before the onset).
