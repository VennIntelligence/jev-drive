# SMIX-VALUE: what S-MIX is worth, measured on CPU before any GPU is spent on it (2026-10-22)

The question: is an isolated S-MIX arm, or S-MIX plus a refreshed pool plus PAI-state S rows with `road_pai` (decision 418), worth GPU? If so,
how large is the expected effect in PAI-281 scene points, and how uncertain? Everything here is CPU-only and uses stored files: no training,
no closed-loop reads, no pool jobs. Background: [research/s-rows-and-paper.md](../../../../research/s-rows-and-paper.md) (the S-row audit, E0 / E1),
decisions 434 (R6DAG), 436 (loss map) and 441 (R7J = R6DAG + longitudinal hygiene + S-MIX + heading command, not adopted).

Code: [../../scripts/smix_value.py](../../scripts/smix_value.py) (`fwd`, `batch`, `learn`, `cover`, `pai`) and the chain
[../../scripts/smix_value.sh](../../scripts/smix_value.sh). Box outputs: `$DATA_DIR/runs/round7/smix_value/`. Here: [batch.json](batch.json) (Q1),
[learn.json](learn.json) (Q2), [cover.json](cover.json) + [cover_events.csv](cover_events.csv) (Q3), [pai.json](pai.json) (Q4).

**Verdict: an isolated S-MIX arm is not worth GPU, and the extended version (+ refreshed pool + PAI-state S rows with road_pai) should not get
GPU before a CPU gate.** Expected size of an isolated S-MIX arm on PAI 281: **0 points, plausible range -3 to +3 scene points (-0.01 to +0.01
mean)**. One seed's read SD is about 5.6 points (0.020), so two seeds detect only effects of about 10 points or more. The arm cannot be read
even if the estimate is right.**

## Definitions (reused from research/s-rows/audit.py and joint.py `smix_mult`)

- **S rows**: the 87 012 rows of the nuPlan S pool parts rb0 / rtb1 / y / ys in on1_JTPHD. 68 939 are live under R6DAG (15 % held out by log).
- **Active**: any of the five S terms (agent, road, lat_side, lat_onc, lat_soft) is > 0 on the checkpoint's own eval-mode plan.
- **Unlearnable**: the ego is already off the raster at t0, or the row is agent / road active while the human-consistent path `fut` violates too.
- **Learnable-active (LA)**: active and not unlearnable, under R6DAG-L18-s1's own forward (E0). These are the S-MIX x4 rows (10 503 rows).
- **Hinge mass**: the sum of the five terms at the trainer's weights (ho_w 3 x lam 10 for agent / road, ho_w 3 x 1 for the lateral three). It is
  rescaled to each arm's ho_w where it enters a batch. "Per imitation row" uses the joint plan's pressure denominator
  (0.75 x navtrain + 13 yaw-rate + 32 PAI human + 8 DAgger).

## Q1. Batch arithmetic: R6DAG vs R7J's S-MIX draw

From the two stored draw files (`round6/dag/w/R6DAG.npz`, `round7/joint/w/R7J.npz`) and R6DAG-s1's per-row terms. Batch 136.

| quantity | R6DAG | R7J (S-MIX) | R7J / R6DAG |
|:--|--:|--:|--:|
| S rows per batch | 32 | 58 | 1.81 |
| navtrain rows per batch | 51 | 25 | |
| ho_w | 3 | 0.86 (after the G3 gradient-ratio rescale) | |
| live S rows | 68 939 | 61 735 (8 912 unlearnable rows set to 0) | |
| active share of S draws | 22.3 % | 30.5 % | |
| LA share of S draws | 11.9 % | 30.5 % (every active draw is LA) | |
| **LA rows per batch** | **3.8** | **17.7** | **4.66** |
| active rows per batch | 7.1 | 17.7 | 2.48 |
| hinge mass on LA rows | 32.9 % | 100 % | |
| hinge mass on unlearnable rows | 67.1 % | 0 % | |
| row pressure (rows x ho_w / imitation rows), in RSG multiples | 2.41x | 1.60x | 0.66 |
| expected S hinge per imitation row (all S rows) | 0.290 | 0.162 | 0.56 |
| **expected LA hinge per imitation row** | **0.096** | **0.162** | **1.70** |
| LA hinge per imitation row, agent / road / onc / side / soft | 0.066 / 0.011 / 0.010 / 0.006 / 0.004 | 0.112 / 0.018 / 0.016 / 0.009 / 0.007 | 1.70 for each |
| share of hinge mass: agent / road / lateral three | 36 / 55 / 8 % | 69 / 11 / 20 % | |
| S draws above 13 m/s / above 8 m/s | 0.99 % / 18.5 % | 0.84 % / 15.8 % | |

Reading. S-MIX did what it was built to do on paper. Learnable-active rows per batch went from 3.8 to 17.7 (x4.7). The S hinge that can
actually be learned rose x1.7 per imitation row, and the unlearnable two thirds of R6DAG's S hinge was removed. Total S hinge per imitation row
fell to 0.56x, which is why the G3 smoke matched the gradient ratio (0.97x) with ho_w 0.86. In gradient terms, R7J pushed the same S force as
R6DAG, all of it on learnable rows. Agent-dominated: 69 % of R7J's S hinge is the agent term, and the road term fell from 55 % to 11 %
(nearly all road mass sat on unlearnable rows). The speed profile did not change: under 1 % of S draws are above 13 m/s in either arm.

## Q2. Did S-MIX teach anything on its target rows?

CPU forwards (`smix_value.py fwd`) of the trainer's own view of the S rows: same argv and same term weights for every checkpoint, and
eval-mode plans. R7J is run with the route-heading command on the S rows (`_cmdh`, as it trained); R7J-s1 is also run without it. The
row sample is stratified: all 10 503 S-MIX target rows (x4), plus a seeded 15 % uniform sample of all 87 012 S rows. That is 22 062 rows per
checkpoint, and the non-target strata use the uniform rows only. The CPU forward of R6DAG-s1 matches the stored GPU forward
(`trainee_R6DAG-L18-s1.parquet`): target hinge mass is identical to 1e-4 relative, and activation is within 0.1 pp. NTV0 (the init of both
arms) comes from the stored GPU forward.

**Selection caveat.** The target rows were picked as *active under R6DAG-s1*, so R6DAG-s1 is 99.9 % active on them by construction, and any
other model looks better there (regression to the mean). The unbiased comparators are NTV0 (the shared init) and R6DAG-s0 (same recipe, not
used in the selection).

Activation (%) / mean hinge (trainer weights) on the 10 503 target rows, by term:

| checkpoint | agent | road | lat_side | lat_onc | lat_soft | any term: act / mass | mass change from NTV0 |
|:--|:--|:--|:--|:--|:--|:--|--:|
| NTV0-L18-s0 (init) | 37.4 / 1.707 | 22.8 / 0.309 | 25.8 / 0.124 | 5.7 / 0.250 | 11.7 / 0.102 | 89.7 / 2.493 | |
| R6DAG-L18-s0 | 38.2 / 1.678 | 22.9 / 0.227 | 27.0 / 0.127 | 5.7 / 0.222 | 11.9 / 0.101 | 92.6 / 2.355 | -5.5 % |
| R6DAG-L18-s1 (selector) | 38.3 / 1.654 | 27.3 / 0.248 | 28.8 / 0.131 | 5.8 / 0.219 | 12.6 / 0.103 | 99.9 / 2.354 | -5.6 % |
| R7J-L18-s0 (cmd heading) | 37.0 / 1.642 | 21.5 / 0.243 | 26.0 / 0.121 | 5.7 / 0.241 | 11.7 / 0.103 | 88.8 / 2.350 | -5.7 % |
| R7J-L18-s1 (cmd heading) | 36.5 / 1.620 | 21.9 / 0.251 | 26.0 / 0.122 | 5.7 / 0.245 | 11.7 / 0.104 | 88.9 / 2.343 | -6.0 % |
| R7J-L18-s1 (old command) | 36.8 / 1.640 | 22.7 / 0.260 | 26.2 / 0.125 | 5.7 / 0.242 | 11.9 / 0.103 | 90.2 / 2.370 | -4.9 % |

Paired contrasts on the target rows (row bootstrap, [learn.json](learn.json)):

| contrast | any-term activation, pp | hinge mass, relative | agent mass | road mass | onc mass | side mass |
|:--|:--|:--|:--|:--|:--|:--|
| R7J-s0 - R6DAG-s0 (unbiased same-seed pair) | -3.8 [-4.3, -3.2] | **-0.2 % [-1.2, +0.6]** | -2.1 % [-3.2, -1.1] | +7.4 % [+3.2, +12.3] | +8.2 % [+5.8, +10.7] | -5.0 % [-7.6, -2.3] |
| R7J-s1 - R7J-s0 (R7J seed spread) | +0.1 [-0.3, +0.6] | -0.3 % [-0.8, +0.2] | -1.3 % | +3.1 % | +2.0 % | +1.0 % |
| R6DAG-s0 - NTV0 (what R6DAG's 9 500 steps did) | +2.9 [+2.3, +3.5] | -5.5 % [-6.7, -4.5] | -1.7 % | -26.7 % | -11.3 % | +2.5 % |
| R7J-s1 with / without the heading command | -1.3 | -1.1 % | | | | |

Off the target rows (the rows S-MIX down-weighted), on the uniform sample:

| stratum (rows in sample) | NTV0 | R6DAG s0 / s1 | R7J s0 / s1 | R7J - R6DAG |
|:--|:--|:--|:--|:--|
| x2: learnable, inactive under R6DAG-s1, zero / partial source (4 108): act % | 3.4 | 1.7 / 0.0* | 3.7 / 3.3 | inactive rows re-activate (+2.1 pp vs s0) |
| x1: learnable, inactive, full-score source (6 068): act % | 2.0 | 1.2 / 0.0* | 2.2 / 2.2 | +1.1 pp vs s0 |
| x0: unlearnable, weight 0 under S-MIX (1 383): mass | 6.39 | 6.21 / 6.23 | 6.50 / 6.52 | +4.5 % [+3.0, +6.9] |
| all live S rows (10 480): act % / mass | 23.4 / 1.037 | 22.9 / 0.995 | 23.7 / 1.024 | **mass +3.0 % [+1.9, +4.5]** (both seeds) |

(* zero by selection under R6DAG-s1.)

Reading.
1. **S-MIX gave its target rows 4.7x the draws (and 1.7x the learnable hinge per imitation row) and moved them no further than R6DAG
   did.** From the shared init, R6DAG-s0 cut target hinge mass by 5.5 % and R7J by 5.7 / 6.0 %. The same-seed difference is -0.2 %
   [-1.2, +0.6]. The R7J seeds agree to 0.3 %, so this is not noise. The term mix only shifted: agent 2 % lower, onc and road 7-8 % higher (R7J's
   road and onc pressure fell with ho_w 3 -> 0.86 and the road rows that were zeroed). Target activation is even 3.8 pp lower than R6DAG-s0's,
   but that is mostly R6DAG-s0 drifting toward the selector's own errors (NTV0 89.7 %, R6DAG-s0 92.6 %, R7J 88.8 %): about 1 pp below the init.
2. **What S-MIX gave up is measurable.** On the rows it zeroed or down-weighted, R7J is worse than R6DAG: the unlearnable rows +4.5 % mass,
   learnable inactive rows re-activate about 2x as often. On the full live pool (R6DAG's draw distribution), R7J's S hinge is +3.0 % higher
   on both seeds.
3. **These rows are mostly not learnable at this pressure, whichever way they are drawn.** The whole L18 S schedule moves target hinge by
   about 5 % (NTV0 -> R6DAG, and RB0 -> NTV0 in the audit). Re-sampling concentrates the draws but does not raise that ceiling. The likely
   reason (audit section 2) is that the remaining active hinge is dominated by agent hinge in lead < 20 m situations, where the shape-only
   gradient cannot remove the overlap without a speed change.
4. The command confound is small on S rows: R7J-s1 with and without the heading command differs by 1 % of target mass.

## Q3. Which PAI loss classes can S rows touch at all?

Lost scenes of R6DAG-s0 and R6DAG-s1 on PAI 281 (ACC + GAP), classified exactly as in loss_map.py. Offroad straight, junction-straight and
turn-outside are merged, as in loss_map row 4. Event speed = the ego's maximum served v0 over the 2.5 s before the pai_fix event, taken from
the driver records of the stored reads. Points are per seed (the mean of s0 and s1). Coverage = the share of S draws whose row speed lies in
the class band [p10 - 2, p90 + 2] m/s. "Geo" also requires the situation: |human 4 s heading change| > 30 deg for turn-inside, < 5 deg for
straight / drift. "LA" also requires learnable-active. The S terms only carry agent (collision), road (offroad) and lateral (onc / side /
soft: drift) signals. Overshoot, rear-hit, slow and route-turn have no S term at all.

| class | lost pts / seed | event speed p10 / p50 / p90 (m/s) | events > 13 m/s | S term that applies | S draws in speed band (R6DAG) | + geometry | + LA, R6DAG -> R7J | S-addressable pts / seed (upper bound) |
|:--|--:|:--|--:|:--|--:|--:|:--|--:|
| offroad: turn, cutting the inside | 8.5 | 4.6 / 6.4 / 10.2 | 0 % | road (+ lateral) | 74 % | 9.9 % | 1.9 % -> 4.8 % | 8.5 |
| offroad: straight / junction straight / turn outside | 7.5 | 3.3 / 9.7 / 22.2 | 33 % | road (+ lateral) | 89 % | 52 % | 4.8 % -> 12.4 % | ~5.0 |
| drift / lane change out of the corridor | 4.5 | 19.2 / 31.9 / 38.6 | 89 % | lateral onc / soft / side | **0 %** | 0 % | 0 -> 0 | ~0.5 |
| at-fault collision (front / side) | 6.0 | 4.7 / 7.2 / 14.1 | 17 % | agent | 75 % | 75 % | 8.3 % -> 21.2 % | ~5.0 |
| stop overshoot | 19.5 | 4.9 / 8.3 / 16.2 | 31 % | none | | | | 0 |
| rear-hit partials | 8.3 | 2.2 / 10.1 / 24.3 | 27 % | none (followers) | | | | 0 |
| slow partials | 0.8 | 11.3 / 15.6 / 34.3 | 58 % | none | | | | 0 |
| route turn missed | 2.0 | 6.4 / 8.3 / 10.1 | 0 % | none | | | | 0 |
| **total** | **57.1** | | | | | | | **~19 (33 %)** |

The S pool's speed distribution (draw-weighted, R6DAG; R7J is within 3 points everywhere): < 3 m/s 28 %, 3-8 53 %, 8-13 17.5 %,
13-20 1.0 %, >= 20 m/s 0 %. The pool's maximum row speed is 15.3 m/s and its p99 is 12.9 m/s. Situations: free road 49 %, lead < 20 m 40 %,
turn > 30 deg 6 %, standing 5 %.

Reading.
1. **Two thirds of what R6DAG loses on PAI 281 is out of reach of any S row by construction.** Overshoot, rear-hit, slow and route-turn
   (30.6 of 57.1 points per seed) have no S term. Drift / lane change (4.5 points) happens at 19-39 m/s, and the pool has no row above 15.3 m/s.
   A third of the straight offroads are above 13 m/s too.
2. **The S-addressable block is about 19 points per seed, at most:** turn-inside offroad, low-speed straight / junction offroad and
   collisions. Of these, the loss map shows that some read of ours already solves turn-inside (7.5 recoverable), straight (about 3.9) and
   collisions (about 3.4). Collisions are mostly seed noise (3 of 9 distinct scenes fail on both seeds).
3. **Even inside that block, the S signal is thin where it matters.** For turn-inside offroad, only 9.9 % of S draws are turns in the right
   speed band. 1.9 % (R6DAG) or 4.8 % (R7J) of draws are learnable-active turn rows, and those are nuPlan urban turns, not PAI geometry. The
   road term carries the offroad signal, and 70 % of road-active rows were unlearnable (audit). S-MIX zeroes those rows, so R7J's road hinge
   per imitation row is 0.018 against an agent hinge of 0.112.

## Q4. R7J vs R6DAG on PAI 281, per class (the stored reads)

pai_fix facts, 278 scenes scored without RPC error in all six reads. Points lost per class; paired scene bootstrap (4 000 draws) for the
same-seed contrasts. A third R6DAG seed (R6DAG-L18-s2, read 0.790) measures seed noise. "Lateral" = turn-inside + straight / junction /
outside offroad + drift.

| class (pts lost) | R6DAG s0 / s1 / s2 | R7J s0 / s1 | R7J-s0 - R6DAG-s0 | R7J-s1 - R6DAG-s1 | seed-mean R7J - R6DAG | R6DAG seed spread (s1 - s0, s2 - s1) |
|:--|:--|:--|:--|:--|--:|:--|
| mean score | 0.775 / 0.815 / 0.790 | 0.786 / 0.782 | | | -0.009 (SE about 0.019) | +0.040, -0.025 |
| offroad turn-inside | 10 / 7 / 8 | 10 / 8 | +0 [-4, +4] | +1 [-3, +6] | +0.7 | -3, +1 |
| offroad straight / junction / outside | 10 / 5 / 10 | 6 / 14 | -4 [-9, +1] | **+9 [+4, +15]** | +1.7 | -5, +5 |
| drift / lane change | 5 / 4 / 4 | 3 / 5 | -2 [-5, 0] | +1 [-2, +4] | -0.3 | -1, 0 |
| **lateral total** | 25 / 16 / 22 | 19 / 27 | **-6 [-12, 0]** | **+11 [+4, +19]** | **+2.0 (SE about 4.2)** | -9, +6 |
| collision (AF-front + side) | 6 / 6 / 6 | 8 / 5 | +2 [-3, +7] | -1 [-6, +4] | +0.5 | 0, 0 |
| overshoot | 20 / 19 / 20 | 22 / 20 | +2 [0, +5] | +1 [0, +3] | +1.3 | -1, +1 |
| rear-hit (partial pts) | 8.7 / 7.7 / 6.7 | 6.4 / 5.9 | -2.3 [-4.8, +0.6] | -1.8 [-3.8, -0.2] | -1.6 | -1.0, -0.9 |
| slow (partial pts) | 0.9 / 0.8 / 0.5 | 1.1 / 0.7 | +0.2 | -0.0 | +0.1 | |
| route turn | 2 / 2 / 3 | 3 / 2 | +1 | 0 | +0.2 | |
| all lost pts | 62.6 / 51.5 / 58.3 | 59.5 / 60.6 | -3.1 [-10.3, +3.8] | +9.2 [+0.7, +18.2] | +2.6 | -11.1, +6.8 |

What can and cannot be attributed.
- The same-seed pairs disagree in sign on the lateral classes (-6 and +11). The R6DAG seeds alone span 16-25 lateral points (s1 - s0 = -9
  [-16, -3]). Seed numbers do not pair training noise: R7J changed the draw files, so its s1 draws different rows from R6DAG-s1. The fair
  contrast is the seed mean, +2.0 lateral points for R7J (SE about 4.2 from R6DAG's three-seed spread), indistinguishable from 0.
- R7J changed three things at once. (1) Longitudinal labels and F2 on the PAI human and DAgger rows: the rear-hit drop (-1.6 points by seed
  mean) is that change's registered target. (2) S-MIX, which also halved navtrain (51 -> 25) and scaled ho_w to 0.86. ho_w multiplies the
  lateral terms and the agent term on the DAgger rows as well. (3) The route-heading command, which changes 27 % of the S rows' commands
  (23 778 / 87 012) and 17 % of served commands. Any of the three can move lateral classes, so no lateral or collision difference can be
  assigned to S-MIX.
- What can be said: the joint arm with S-MIX shows no lateral or collision gain larger than about 8 points per seed (the upper 95 % bound on a
  gain, from +2.0 +- 2 x 4.2). Its collisions are flat (+0.5).

## Q5. Verdict and expected size

**Isolated S-MIX (E1 alone): no.** Three independent measurements agree.
- *Mechanism (Q2):* 4.7x the learnable-active draws produced -0.2 % [-1.2, +0.6] extra hinge reduction on the target rows relative to
  R6DAG-s0. The rest of the pool got 3 % worse. There is no offline sign of extra learning to carry into the closed loop.
- *Reach (Q3):* at most about 19 of the 57 points lost per seed are of a kind any S term addresses (turn-inside offroad, low-speed straight /
  junction offroad, collisions). The S-MIX target rows inside that block are 4.8 % (turns in band) to 21 % (collision band) of R7J's S draws,
  all from nuPlan urban geometry. Drift (4.5 points, 19-39 m/s), stop overshoot (19.5), rear-hit (8.3), slow and route-turn are out of reach
  by construction.
- *Closed loop (Q4):* the only S-MIX read we have (R7J, confounded) is +2.0 lateral points worse by the seed mean (SE about 4.2) and +0.5 on
  collisions. That is consistent with zero and bounds a lateral gain at about 8 points at most.

Expected value, built from the parts: S-addressable block about 19 points per seed, of which about 14 are demonstrably recoverable by some
recipe of ours (loss map). Take the share that a stronger S signal could plausibly move. The whole S schedule's measured effect on these
rows is about 5 % hinge, and S-MIX adds 0 to it. The S dose and source history (decisions 311, 348, 367, 402) has been flat to negative on
PAI since 10-13. A realistic prior for the isolated arm is 0 +- 3 points (0 +- 0.01 mean). The cost is about 0.7 + 1.4 card-hours per seed
for the training and the read, and with seed noise of 0.020 the read cannot confirm the arm.

**S-MIX + refreshed pool + PAI-state S rows with road_pai (E2 + E1): not yet; it needs a CPU gate first.** This is the only version that
changes reach: PAI-state rows carry PAI speeds (the drift and high-speed straight-offroad events are 19-39 m/s) and PAI geometry. Its
ceiling is the S-addressable block plus most of drift, about 23 points per seed. The evidence against it is already on the record:
- PAI S rows were flat twice: d367 -0.010 [-0.053, +0.033] and d402 -0.007 [-0.028, +0.013].
- The road_pai hinge triggers before 60 % of offroad zeros but is shallow (median 0.005, d418 G2). Adding it on PAI human rows raised
  offroad from 4 to 7 (d420).
- PAI has no onc / soft lane rasters (only road_pai), so the drift class would be carried by road_pai + side alone.
- The DAgger map teacher already targets the same lateral classes with labels instead of hinges (d434, +0.072 on s1).

Expected size if built: **+0 to +5 points (+0.00 to +0.02), centre about +2, with a downside of about -3** (the d420 pattern). That is below
the two-seed detection limit of about 10 points. The CPU-first path is cheap: R6DAG's own forward on on1_PAID states with road_pai,
`smix_value.py fwd` style. Then apply decision 418's three gates plus one new gate: learnable-active PAI rows above 13 m/s >= 30 % of S
draws. If the gates pass, it should ride on the next DAgger iteration's training run as one joint arm (research-method rule), not as its
own GPU line.

Two things would change this verdict. One is an offline sign that some draw rule moves target hinge clearly past the ~5 % ceiling
measured here. The other is a closed-loop read where the S-addressable classes move by more than seed noise. Neither exists today.

## Cost

CPU only, 0 GPU: six stratified forwards of 22 062 rows each, 17-18 min each on a third of the box's cores (two waves of three, about
40 min wall). Plus the batch / cover / pai / learn analyses, a few minutes. About 70 core-hours in total. The forward needs no GPU, but it
loads the trainer's full build (navtrain + pool) in host memory, about 13 GB each (box memory 26 -> 66 GB with three running).
