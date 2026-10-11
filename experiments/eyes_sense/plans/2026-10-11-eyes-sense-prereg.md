# EYES-vs-SENSE: same eyes, better car sense (pre-registration, 2026-10-11)

Lane JOINT, phase 2; topic `experiments/eyes_sense/`. Follows decisions 136, 232, 233, 236, 243, 244, 245, 248. Written and pushed before any score of any
arm of this phase is read. No WA-JEPA weights or features; no privileged input at inference (GT boxes, map rasters and logged futures are training
labels only). Nothing is submitted.

## Question

The paper's claim under test, "same eyes, better car sense": perception already suffices, and consequence supervision on the own plan (the body
lesson S) is what moves the boards. Decision 248 closed the camera / encoder line under continuation and wrote its reopening condition: a
joint-training arm where side views lower both > 45 deg cut-inside and off-road without a < 5 deg, NC or EC cost. This phase is that joint
training, crossed with S, so that EYES (more of the same frozen eyes: two side views) and SENSE (the body lesson) are read in one 2 x 2.

## Arms (one trainer, `experiments/alpasim/scripts/joint_train.py`; lambda 10 / 0.3 m, 10 000 steps x 128, from the shipped weights, seeds 0 / 1)

| Arm | Tags | Flags (besides the defaults = the P2H10 recipe) |
|:--|:--|:--|
| (a) base P2H10 | `ESa-s{0,1}`, or the stored `P2H10-F-s{0,1}` (identity rule) | none |
| (b) + EYES | `ESb-s{0,1}` | `--mem vtr_W0` |
| (c) + SENSE | `ESc-s{0,1}`, or the stored `P2H10S-F-s{0,1}` (identity rule) | `--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape` |
| (d) both | `ESd-s{0,1}` | (c)'s flags + `--mem vtr_W0` |

Every run adds `--label-bank` (bit for bit) and runs eager.

- EYES = decision 248's RW0 bank `vtr_W0`: the frozen shipped encoder on the three t0 views CAM_F0 / CAM_L0 / CAM_R0 (N, 3, 32, 512), no trained
  vision weight; fed through the adapter's memory channel as pp_train.py `--mem` feeds it (arm `P2+vtr_W0`, 3-camera adapter), the bank masked on
  25 % of the rows, all three views together, from its own rng stream ([seed, 0, 11]). Unlike decision 248's R arms, the policy recipe is
  lambda 10 (P2H10), not SH30's lambda 30.
- In (d) the 13 hinge-only rows per batch always read the memory masked: their perturbed states have no rendered side views, and the source token's
  views would carry an off-track cue that never occurs at test time.
- **Identity rule for (a) and (c)**: the joint trainer runs with the flags of the stored checkpoints (`ESa` = P2H10-F's, `ESc` = P2H10S-F's, both
  seeds, all 12 shards). If every trained net and adapter weight equals the stored checkpoint bit for bit (`joint_train.py ident`), (a) / (c) are the
  stored checkpoints with every stored read; otherwise (a) / (c) are ESa / ESc and every read below is run for them. The 60-step identity of
  the trainer against pp_train.py (plain and with the memory channel) and bd4_train.py is phase 1's gate (experiments/alpasim/plans/2026-10-11-joint1-prereg.md).
- Not a design choice of this lane, stated so it is not read into the result: (b) and (d) add the channel and the side views together (there is no
  front-only bank control); decision 248's RW0 - R0 (the same bank against a bank of Cinque's own t0 token, SH30 recipe) was a null on every cell.
  Inside the arms the side views are probed by `:sideoff` (both side views masked, CAM_F0 kept), which lies outside the training distribution (the
  bank drop masks the three views together, as in RW0); it is read descriptively.

## Pilot

(b) and (d): seed 0, 300 steps on all shards. Checklist: finite losses; the run's log line shows memory on 75 % of the normal rows and masked on every
hinge-only row ((d)); the final dev evaluation reports the dev ADE with the memory on / masked / from a row of another log (finite; masked and
other-log values differ from on). A failed item stops that arm's fan-out. (a) / (c) need no pilot: their recipes are the stored ones.

## Reads

Open loop through `python -m jevdrive.bench` (navtest 12 146 tokens; navhard two-stage on G frames, `<tag>@gimm`, 225 groups), tables by
`scripts/es_read.py`, which imports experiments/vis_train/scripts/vt_read.py's definitions (instrumented replay of failing tokens, turn buckets from
`jevdrive.bench.tables.navtest_strata`, failure classes, by-log cluster bootstrap):
1. navtest EPDMS and every sub-score by turn bucket (all, < 5, 5-20, 20-45, > 45, > 20 deg); failure rates in percentage points: off-road (DAC < 1),
   cut-inside (off-road whose first footprint departure in the LQR replay is on the inside of the turn), under-turn (off-road, not inside, 4 s
   heading gain < 0.9), NC classes (decision 196); flips per class and bucket, "saved N / hurt M" (tokens, both seeds summed); decision 248's
   no-regression table (every sub-metric x bucket, up / down / did not move).
2. navhard combined / stage 1 / stage 2, per seed and paired with the comparison arm (by the stage-1 log).
3. Test-time options on (b) and (d): `:noside` (bank masked), `:mshuf` (the bank row of a token of another log), `:sideoff` (both side views masked,
   CAM_F0 kept; new in `jevdrive.bench` for parity three-view memory arms, the same mask as vis_train's `vt.py`). **Channel-read check**: arm minus
   `arm:mshuf` navtest EPDMS (seed mean) >= 0.2; below it the arm's side-view comparison is recorded as "not measured".
4. nuPlan closed loop, 700 scenes (c0b chunk0-2, the box's default driver path, `sh30`), for (a) and (c): the 700 subset of phase 1's runs of the
   stored checkpoints when the identity rule holds, else ESa / ESc run on the same lists. Zeros by class (at-fault collision, offroad, corridor),
   mean scene score, > 45 deg bucket, fixed-vs-broken. **(b) and (d) are not run in closed loop**: AlpaSim delivers CAM_L0 / CAM_R0 JPEGs, but the
   driver has no path that renders them as vis_train's side-view pairs (lib/side_store.py: protocol-W pair of the t0 slot along the camera's
   mounting yaw) and encodes them into the bank; that is a new serving path with its own equivalence check against the stored bank, not a cheap
   addition. They are read on the open-loop boards only.

Statistics: seed mean per token (both seeds on both sides), difference = arm minus comparison, by-log cluster bootstrap (B 10 000, seed 0;
`jevdrive.stats`, vt_read.CB); "CI below 0" = upper bound < 0. Closed loop as phase 1 (two-seed per-scene means, by-log CI for the mean, paired
scene bootstrap for zero counts).

## Registered lines (from the lane brief; not changed after any read)

**Side views**, for (b) - (a) and for (d) - (c) separately: side views **REOPEN** only if all of
- the > 45 deg cut-inside rate difference has a CI below 0, and
- the > 45 deg off-road rate difference has a CI below 0, and
- no cell of {EPDMS in the < 5 deg bucket, NC (all tokens), EC (all tokens)} has a CI below 0 ("down"), and
- the arm's channel is read (check above; else "not measured").
Otherwise the line **stays closed** for that comparison. The verdict is stated per comparison; the line reopens if either comparison reopens, and
the report names which.

**Headline rows** (c) - (a) and (d) - (b): read descriptively with the same tables. Stated expectations (decisions 233 / 236 / 244): navhard stage 2
up, > 45 deg cut-inside down, and, for (c) - (a) where the closed loop exists, nuPlan at-fault collision + offroad zeros down and corridor zeros up.

## How this differs from earlier decisions

- **248 (VT)**: every VT arm continued a trained SH30 policy (continuation, optimizer restarted); R trained from scratch but with SH30's lambda-30
  recipe and no body lesson. Here: one joint training from the shipped weights on the lambda-10 recipe, only the untrained frozen 3-view bank (no
  trained vision weight), crossed with S; read in 248's tables.
- **244 (S-DROP)**: ablated S's own ingredients; here S is one factor of a 2 x 2 and its arms are the stored checkpoints if the identity rule holds.
- **233 / 236**: P2H10S against P2H10 descriptively (4 seeds closed loop, navhard / HUGSIM); here the same contrast is the headline row (c) - (a) with two
  seeds, next to (d) - (b).
- **212, 213, 216**: AlpaSim-standard and yaw-row recipes; none enters this phase (NAVSIM standard, no yaw-rate rows).

## Known limits, stated before the data

- Two seeds; navtest has been read for (a) and (c) before (decisions 232, 244) and the vtr_W0 bank's arm on the SH30 recipe (decision 248).
- `:sideoff` is out of the training distribution; (b) / (d) closed loop not run.
- Many cells per comparison: about 5 % of navtest cells cross 0 under no effect (decision 248's note); the registered line uses four named quantities.

## Cost, before running (card-hours)

| Item | Card-hours |
|:--|--:|
| ESa / ESc identity runs (4 x 10 000 steps, eager, about 25 min each) | 1.7 |
| pilots (b), (d) (300 steps) | 0.3 |
| ESb / ESd full runs (4) | 1.8 |
| navtest + navhard for (b), (d) (4 checkpoints) and the 12 option reads; replay and tables on CPU | 2.0 |
| if the identity rule fails: navtest + navhard for ESa / ESc (1.0) and their 700-scene closed loop (4 x 700, 1.5) | (2.5) |
| **phase 2** | **5.8 (8.3)** |

## Execution

`scripts/es_chain.sh` (pool jobs: identity runs, pilots, full runs, bench reads, the read-out), state under `$DATA_DIR/runs/eyes_sense/`. Results
`results/contrast.md` (+ tables), README, INDEX line, decision draft.
