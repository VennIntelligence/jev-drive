# nc-slow (lane NC1): is "where to slow down" decidable from what the driver has at inference time?

Written 2026-10-09. Follows decision 196 (the same-path longitudinal scaling oracle: +1.13 no-EC EPDMS on navtest at an EP cost of 0.05, no blanket
slowdown is net positive). Pre-registration [plans/2026-10-09-nc-slow-prereg.md](../plans/2026-10-09-nc-slow-prereg.md), committed and pushed before any
score of this lane was read. Code `scripts/nc_slow.py`, `scripts/nc_slow_chain.sh`. All tables: [nc_slow/tables.md](nc_slow/tables.md) (registered
read-outs), [nc_slow/posthoc.md](nc_slow/posthoc.md) (two descriptive read-outs added after the stage-1 read, marked as such below).
Ground-truth boxes and the logged future enter only the arms G0 / G1 (privileged probes). No WA-JEPA weights or features. No driver was trained.

## Answer

**Stage 1 fails the registered line; the lane stops; stage 2 was not run.** Every non-privileged arm that reaches the required gain does it with a broad
slowdown (23-39% of the tokens moved, EP -1.3 to -1.8 points; the line allows 0.3), and the gain is a drivable-area effect, not a collision one.

1. **Registered verdict.** Line: no-EC EPDMS gain >= +0.339 (30% of +1.13) with a 95% CI lower bound > 0 **and** EP loss < 0.3. Five of the six
   non-privileged arms pass the gain part (+0.49 to +0.63, 43-56% of +1.13, Bonferroni lower bounds > 0), none passes the EP part. The privileged arms fail
   the EP part too (G0 -1.45, G1 -0.81): the registered threshold rule maximises the net score and so buys net-positive slowdowns freely.
2. **The gain is not collisions.** At the registered operating point the tokens whose DAC changes contribute +0.51 to +0.66, i.e. more than the whole net
   gain of every non-privileged arm; tokens whose NC changes contribute +0.19 to +0.31; the EP cost of needless moves is -0.22 to -0.43 (post hoc split).
   A slower same-path plan leaves the road later, or after the 4 s window: decision 178's "slowing pushes the exit out of the horizon" caveat, not a fix.
3. **With the EP loss held near 0.3** (post hoc; threshold picked on navtrain under that cap), the non-privileged arms move 2.6-4.7% of the tokens and
   keep +0.23 to +0.39 (21-34% of +1.13), again all of it from DAC-changing tokens (+0.26 to +0.37); they fix 8-22 of the 176.5 NC failures and create
   2-3.5. The ground-truth lead gap fixes 47.5, the logged-future oracle 96.
4. **AUC for "this token fails NC / TTC and a slower scale cleanly recovers it" (`y_fail`)**: ego + plan alone 0.660; adding the policy hidden state 0.662,
   frozen Cinque vision tokens 0.657, both 0.652; openpilot's native lead / meta / action outputs **0.706** [0.663, 0.749]; ground-truth lead gap at t0
   0.758; plus the logged future 0.859. The frozen representation adds nothing measurable to ego + plan for locating collisions; openpilot's own lead
   outputs add a little (moving-lead failures recalled 57% against 24% for ego + plan); even the true t0 gap is far from the ceiling.
5. **Which input came closest**: by the registered gain, OPH (everything non-privileged, +0.628 [+0.403, +0.864], EP -1.32); by collisions, **OP**
   (openpilot's native outputs: NC failures 176.5 -> 127.5 at the registered point, NC or TTC failing tokens -0.51 pp [-0.69, -0.34]). The ground-truth
   ceilings: G0 (t0 lead gap) +0.749 [+0.481, +1.011], 66% of +1.13, NC failures -> 104; G1 (logged future) +1.015 [+0.753, +1.271], 90%, NC failures -> 81.5.

## Setup (as registered)

- **Family**: `turn_ceiling.speed` (same path, arc length x a), a in {0.7, 0.8, 0.9, 1.0, 1.1} scored; the selection set is {0.7, 0.8, 0.9, 1.0}.
  Scoring: `python -m jevdrive.bench score-poses --traffic non_reactive` (navtrain: `--mcache v2_navtrain`), no-EC EPDMS.
- **Train (navtrain)**: 57 378 tokens = the 28 323 turn tokens + 29 055 other tokens in the v2 navtrain metric cache (16 rest shards were built for this
  lane, `nt_cache.py run --stage rest --limit 17`; the cache now holds 19 of the 50 rest shards). Plans and features come from decision 190's fold model
  that held the token's log out. Rest tokens carry weight 2.58 so that the weighted turn share is navtrain's. Metadata from the shard `tab.npz`.
- **Test (navtest)**: 12 146 tokens x SH30-F-s0 / s1, scores from decision 196's `nc_tax/score_all.csv` (not re-scored). No navtest score enters any fit,
  model choice or threshold.
- **Labels**: g_a = score(a) - score(1.0) for a in {0.7, 0.8, 0.9}; `y_slow` = some slower scale strictly beats 1.0; `y_fail` = NC or TTC fails at 1.0
  and a slower scale passes it with a higher score.
- **Rule**: pick = argmax of the predicted gains if its maximum exceeds tau, else 1.0; tau maximises the weighted realised gain of the navtrain OOF
  predictions (5 log folds = decision 190's). Learners: T (gradient-boosted regression of the three gains), C (boosted classifier of `y_slow` + a fixed
  scale), M (decision 191's head, 5 initialisations; arms with the hidden state or vision tokens). Per arm, the learner with the highest navtrain OOF gain
  is the one read on navtest.

| arm | inputs | privileged |
|:--|:--|:--|
| E | ego vector (20) + the plan (8 poses, segment speeds, 4 s arc, end heading) | no |
| H | E + the policy's hidden state (`select_4`, `mean`) | no |
| V | E + frozen Cinque vision tokens (newest slot, 32 x 512) | no |
| N | E + H + V | no |
| OP | E + openpilot's native outputs through the parity path: `lead` (mean and std), `lead_prob`, `meta`, `action`, the plan's velocity / acceleration | no |
| OPH | E + H + V + OP | no |
| G0 | E + ground-truth lead gap at t0 along the plan's path (gap, lead speed, closing speed, time gap, TTC, second object, counts) | yes |
| G1 | G0 + per scale, the scaled plan's minimum distance to the logged future boxes | yes |

Gates, all passed: **G-id** (a = 1.0 rows on the 28 323 turn tokens equal decision 190's held-out score rows, max |diff| 0 on all 8 sub-scores and the
score); **G-leak / G-plan** (every row planned by the fold model whose training split excludes its log; the forward of this lane against the stored
plans: max 0.031 m over the first 15 points, 3 of 117 463 rows above the 0.03 m tolerance, one fp16 step, as in decision 192); **G-oracle** (decision
196's rule on `score_all.csv`: +1.125, s0 +1.099 / s1 +1.151, 178.5 tokens); navtest and navtrain share no log.

## Labels

| set | n | NC fail % | NC or TTC fail % | `y_fail` % | `y_slow` % | slower-only oracle on failing tokens (points) | best slower scale on every token, O_any (points) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| navtrain, weighted | 57 378 | 1.22 | 1.92 | 1.44 | 3.71 | 1.08 | 2.48 |
| - turn tokens | 28 323 | 1.57 | 3.17 | 2.42 | 7.39 | 1.60 | 5.03 |
| - rest tokens | 29 055 | 1.08 | 1.45 | 1.07 | 2.33 | 0.88 | 1.51 |
| navtest (seed mean) | 12 146 | 1.45 | 2.17 | 1.40 | 4.57 | 1.09 | 2.96 |

The held-out navtrain labels match navtest on the collision side (`y_fail` 1.44% against 1.40%, oracle 1.08 against 1.09). Note the gap between the last
two columns: on navtest a slower scale beats 1.0 on 4.6% of the tokens and is worth +2.96 when picked perfectly, while the collision part (decision 196's
oracle restricted to slower scales) is +1.09. The other +1.9 is mostly DAC: the same path travelled more slowly leaves the drivable area later.

## Stage 1 (navtest, 12 146 tokens x 2 seeds; change against a = 1.0 in points of the board mean; 95% CI = log-cluster bootstrap, B 10 000)

| arm | learner | AUC `y_slow` | AUC `y_fail` [95% CI] | moved % | precision % | no-EC EPDMS gain [95% CI] | share of +1.13 | NC | TTC | DAC | **EP** | NC failures (fixed / new) | NC or TTC failing tokens, pp | line |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| E | T | 0.687 | 0.660 [0.610, 0.705] | 25.0 | 6.8 | +0.300 [+0.091, +0.510] | 27% | +0.30 | +0.33 | +0.61 | **-1.83** | 176.5 -> 139.0 (44.0 / 6.5) | -0.37 [-0.53, -0.22] | no (gain, EP) |
| H | M | 0.706 | 0.662 [0.611, 0.714] | 37.3 | 6.0 | +0.532 [+0.293, +0.769] | 47% | +0.35 | +0.38 | +0.78 | **-1.84** | -> 133.0 (50.5 / 7.0) | -0.41 [-0.57, -0.25] | no (EP) |
| V | M | 0.703 | 0.657 [0.612, 0.704] | 27.4 | 7.1 | +0.578 [+0.337, +0.817] | 51% | +0.32 | +0.37 | +0.80 | **-1.64** | -> 137.0 (48.0 / 8.5) | -0.42 [-0.59, -0.27] | no (EP) |
| N | M | 0.698 | 0.652 [0.602, 0.707] | 39.3 | 5.5 | +0.581 [+0.363, +0.801] | 51% | +0.37 | +0.37 | +0.79 | **-1.77** | -> 131.5 (52.0 / 7.0) | -0.41 [-0.58, -0.25] | no (EP) |
| OP | T | 0.714 | **0.706** [0.663, 0.749] | 23.1 | 8.0 | +0.487 [+0.240, +0.735] | 43% | +0.38 | +0.47 | +0.69 | **-1.80** | -> 127.5 (55.0 / 6.0) | -0.51 [-0.69, -0.34] | no (EP) |
| OPH | M | 0.704 | 0.647 [0.604, 0.696] | 25.9 | 7.5 | **+0.628** [+0.403, +0.864] | 56% | +0.28 | +0.29 | +0.79 | **-1.32** | -> 141.0 (41.5 / 6.0) | -0.32 [-0.48, -0.17] | no (EP) |
| G0 (privileged) | T | 0.708 | 0.758 [0.705, 0.804] | 20.4 | 9.0 | +0.749 [+0.481, +1.011] | 66% | +0.59 | +0.69 | +0.64 | -1.45 | -> 104.0 (78.0 / 5.5) | -0.74 [-0.94, -0.53] | - |
| G1 (privileged) | T | 0.754 | 0.859 [0.805, 0.911] | 11.1 | 16.2 | +1.015 [+0.753, +1.271] | 90% | +0.77 | +0.86 | +0.53 | -0.81 | -> 81.5 (99.0 / 4.0) | -0.97 [-1.24, -0.70] | - |

How to read it:

- Every arm slows a fifth to two fifths of the board to collect the gain, and 92-94% of the moved tokens gain nothing (precision 5.5-8%). That is a
  learned partial blanket, not a decision about where: decision 196's oracle moves 1.5% of the tokens for EP -0.05.
- The gains are real against chance: the same picks handed to random other tokens score -0.30 to -0.42 (97.5th percentile at most -0.23).
- EP is the registered second condition and no arm is near it. The Bonferroni (m = 6) lower bounds of the gains are > 0 for all six non-privileged arms
  (+0.02 for E, +0.15 to +0.32 for the others), so the gain part is not a multiplicity artefact; the failure is the EP part alone.
- Navtrain OOF agreed with navtest (OOF gain 0.38 / 0.59 / 0.49 / 0.56 / 0.50 / 0.60 / 0.56 / 0.84 in table order): no train-to-test collapse.
- All 20 (arm, learner) rows are in `nc_slow/tables.md`; the classifier learner has the higher AUC for `y_slow` everywhere (0.77-0.83 OOF) and a
  lower realised gain, because it cannot choose the scale.

Learning curve (share of navtrain logs in the final fit, threshold from the full OOF): the non-privileged arms are near flat between 50% and 100%
(H +0.46 -> +0.53, N +0.39 -> +0.58, OP +0.45 -> +0.49, OPH +0.49 -> +0.63, E +0.32 -> +0.30); G0 still rises (+0.45 / +0.60 / +0.75).

Recall of decision 196's NC failures by class (moved and scoring higher, %): stopped vehicle ahead (58 tokens) 25-29% for every non-privileged arm,
47% for G0, 46% for G1; moving lead (35) E 24, H 37, V 34, N 40, **OP 57**, OPH 20, G0 60, G1 74; cut-in (23) 13-24% non-privileged, G0 52, G1 78;
t0 speed < 2 m/s (34) 4-17% non-privileged, G0 38, G1 58. The class decision 196 named as the target (a stopped vehicle ahead at low speed) is the one
the inputs locate worst, and the one where even the logged-future oracle recovers under half.

## Post hoc (not registered, written after the stage-1 read)

**Where the registered operating point's gain comes from** (tokens partitioned by which gate changes under the pick):

| arm | gain | tokens whose DAC changes | tokens whose NC changes (DAC same) | tokens whose only TTC changes | rest: EP cost of the other moved tokens |
|:--|--:|--:|--:|--:|--:|
| E | +0.300 | +0.506 [+0.321, +0.707] | +0.215 [+0.129, +0.307] | +0.011 | -0.432 |
| H | +0.532 | +0.638 [+0.428, +0.870] | +0.254 [+0.151, +0.357] | +0.005 | -0.365 |
| V | +0.578 | +0.664 [+0.431, +0.918] | +0.211 [+0.113, +0.310] | +0.023 | -0.320 |
| N | +0.581 | +0.663 [+0.456, +0.887] | +0.275 [+0.158, +0.392] | +0.016 | -0.374 |
| OP | +0.487 | +0.572 [+0.358, +0.806] | +0.307 [+0.193, +0.428] | +0.023 | -0.414 |
| OPH | +0.628 | +0.658 [+0.444, +0.890] | +0.187 [+0.093, +0.289] | +0.000 | -0.218 |
| G0 | +0.749 | +0.537 [+0.338, +0.754] | +0.482 [+0.325, +0.632] | +0.030 | -0.299 |
| G1 | +1.015 | +0.434 [+0.279, +0.606] | +0.644 [+0.451, +0.834] | +0.045 | -0.109 |

**Operating point with the EP loss capped at 0.3** (threshold = the navtrain OOF threshold of highest OOF gain whose OOF EP loss is <= 0.3; the last
column instead picks the threshold on navtest itself under the same cap, an optimistic bound for the arm):

| arm | moved % | gain [95% CI] | share of +1.13 | EP | gain from DAC-changing tokens | NC failures fixed / new | NC or TTC failing tokens, pp | navtest-tuned bound |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| E | 4.5 | +0.233 [+0.120, +0.356] | 21% | -0.45 | +0.256 | 15.5 / 3.0 | -0.12 [-0.20, -0.04] | +0.181 |
| H | 3.9 | +0.386 [+0.210, +0.575] | 34% | -0.41 | +0.372 | 15.5 / 3.0 | -0.10 [-0.20, -0.01] | +0.340 |
| V | 3.1 | +0.306 [+0.160, +0.457] | 27% | -0.37 | +0.338 | 12.0 / 3.0 | -0.12 [-0.21, -0.04] | +0.282 |
| N | 3.0 | +0.303 [+0.161, +0.457] | 27% | -0.31 | +0.311 | 12.5 / 2.0 | -0.12 [-0.22, -0.04] | +0.301 |
| OP | 4.7 | +0.274 [+0.138, +0.415] | 24% | -0.43 | +0.258 | **22.0** / 3.5 | **-0.18** [-0.28, -0.09] | +0.233 |
| OPH | 2.6 | +0.328 [+0.180, +0.488] | 29% | -0.27 | +0.320 | 8.0 / 3.5 | -0.05 [-0.13, +0.01] | +0.351 |
| G0 | 4.9 | +0.486 [+0.332, +0.635] | 43% | -0.42 | +0.269 | 47.5 / 4.0 | -0.45 [-0.60, -0.31] | +0.393 |
| G1 | 4.7 | +0.909 [+0.675, +1.144] | 80% | -0.37 | +0.273 | 96.0 / 3.0 | -0.89 [-1.13, -0.64] | +0.859 |

- The navtrain cap transfers loosely (EP -0.27 to -0.45 on navtest), so these rows sit at or slightly past the line's EP condition. Even so, and even
  with the threshold tuned on navtest, the best non-privileged arms land at the +0.339 line (H +0.340, OPH +0.351), not past it.
- At this point the non-privileged gain is the DAC term alone (gain +0.23 to +0.39, of which DAC-changing tokens +0.26 to +0.37). The collision gates
  barely move: 8-22 NC failures fixed of 176.5, NC or TTC failing tokens -0.05 to -0.18 pp. The stage-2 line (NC + TTC failures down with a CI excluding
  0, EPDMS +0.3 with EC) would be met, if at all, by a DAC effect that the 4 s horizon partly manufactures.
- The privileged rows show the size of the missing information at equal EP cost: the true t0 gap triples the fixed collisions (47.5), the logged future
  takes them to 96 for the same 4.7% of tokens moved.

## Secondary read-outs (registered)

- **Navtest-internal cross-fit by log** (labels from navtest itself, 5 folds, nested threshold, tree learners): E +0.284 [+0.072, +0.497] (EP -0.87), H +0.478 [+0.262, +0.700] (-0.90), V +0.434 [+0.216, +0.658] (-1.02), N +0.486 [+0.241, +0.736] (-1.20); G0 +0.617 [+0.340, +0.889] (-1.17), G1 +0.746 [+0.522, +0.964] (-0.50). Same order and no higher than the navtrain-trained heads, so the navtrain-to-navtest shift is not what limits the non-privileged arms. The cross-fits of OP and OPH were cancelled after 52 min on the loaded box (329+ features x 75 boosted fits) and are not reported.
- **Learning curve**: above.

## What it means

- **For the NC fix on the paper driver.** "Where to slow down for a collision" is not in the frozen Cinque tokens or the SH30 hidden state at this label
  scale (57 k tokens, about 1 000 `y_fail` positives): their AUC for `y_fail` equals ego + plan's (0.66) and the capped operating point fixes about 15
  of 176 NC failures. This is the third negative on the frozen representation for collisions, after the agent hinge on the plan head (decision 158)
  and ground-truth occupancy through the memory channel (decisions 197 / 200), and it is the cleanest one, because here the head is free of the plan
  loss and of the channel. What does carry information is an explicit object-level range read: openpilot's own lead outputs (0.706), then the true t0
  gap (0.758), then the others' future motion (0.859). So a longitudinal NC fix needs a better near-range object signal as an input or a supervised
  target of the trained branch, not a selection head on the present features. A scale-selection head as a method component is closed.
- **The scaling family's score gain is mostly not NC.** Of the +2.96 available to a perfect slower-scale pick on navtest, only +1.09 is the collision
  oracle; what a learned head harvests is the DAC part, and that part is partly the 4 s window (decision 178). Any future use of speed candidates
  in a selector (F19's speed axis, decision 190) should report the DAC-changing tokens separately.
- **For the competition driver.** Nothing to ship from this lane. A learned slowdown gate would trade progress for a drivable-area effect that AlpaSim
  does not score the same way, and it does not remove the at-fault collisions it was meant for. The one usable lead: openpilot's native lead / meta
  outputs recall 57% of the moving-lead failures at the registered point, so a lead-aware longitudinal term should start from those outputs (with
  decision 157's caveat that a fixed margin on `lead_x` failed) rather than from the vision tokens; it has to be read in AlpaSim, where traffic is the log.

## Limits

- Non-reactive log-replay traffic, 4 s, no EC: slowing is charged no rear-end and no comfort; picks are per frame with no temporal consistency.
- One fold-model seed for the navtrain plans; features of the test side come from SH30 itself (as in decision 191).
- About 1 000 `y_fail` positives on navtrain (1.44% weighted). The non-privileged curves are flat from 50% to 100% of the logs, which argues against
  "more of the same labels"; it does not rule out a much larger label set or a representation trained on them.
- Learners are generic (boosted trees on PCA-32 of the high-dimensional streams, decision 191's small head); no arm-specific tuning.
- G0 uses the 0 -> 0.5 s logged displacement as the object speed; G1 uses the 4 s logged future. Both are ceilings of this head, not of the information.
- The registered threshold rule maximises the net score, which is why every arm, privileged ones included, overshoots the EP condition; the EP-capped
  rows are post hoc and their cap is set on navtrain.
- The extraction's VRAM was first declared 10 GB against a measured 15.5 GB; one job hit an out-of-memory error on a shared card before the
  declaration was corrected to 17 GB.

## Cost

About 0.6 card-hour of GPU compute (14 extraction passes, the M fits), about 2 card-hours of booked 8-17 GB slots on shared cards; CPU about 16
core-hours for the 16 rest shards of the metric cache (11 GB, the cache is now 24 GB), 9 for scoring 57 378 tokens x 5 scales, about 25 for the fits
and cross-fits. Stage 2 was not started.

## Files

- Tables: `nc_slow/tables.md`, `nc_slow/posthoc.md`, `nc_slow/{stage1,all_learners,curve,recall_by_class,xfit,posthoc}.csv`, `nc_slow/summary.json`.
- Box: `$DATA_DIR/runs/op_parity/nc_slow/` (`poses_nt.npz`, `score_nt.csv` = the navtrain labels of the family, `meta_nt.npz`, `feat/full/`,
  `tab_{train,test}.npz`, `fit/`, `xfit/`, `report/`, chain log).
