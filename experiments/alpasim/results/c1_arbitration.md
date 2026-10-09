# C1 / C2: can a decision-time signal pick the better of SH30 and AP2 (2026-10-09, decision 202)

Follows [c0_public400.md](c0_public400.md): per-scene best-of-two is +0.0292 over the best single driver (AP2 0.9335) on 400 scenes, with hindsight.
Question: how much of that does a rule recover that sees only what a driver is sent. No simulator run was needed; nothing was submitted to AlpaSim; no WA-JEPA.
Code `scripts/c1_arb.py`; numbers `c1/arb.json`, `c1/arb_heldout.json`, per-scene signals `c1/arb_scenes.csv`.
Pre-registration of the held-out read: [plans/2026-10-09-c1-arbitration-prereg.md](../plans/2026-10-09-c1-arbitration-prereg.md), pushed (d7f390b6) before C0b's new-scene scores were read.

**Answer.** No. Plan disagreement does not identify the better driver (every disagreement gate is negative under cross-validation). The one signal with a positive estimate,
a standstill start, gives +0.008 to +0.012 on the 400 scenes depending on how the threshold is chosen, and **+0.0024 [-0.0066, +0.0131] on 300 held-out scenes**: the registered
line (>= +0.010, CI lower bound > 0) is not met. No gated driver was built.

## Why this can be read offline, exactly

The first 0.5 s of a rollout is log replay, so at decisions 0 and 1 both drivers receive identical inputs (anchor pose difference 0 in all 400 scenes; from decision 2 the
states differ, median 7 mm, max 45 mm at decision 2). A selector that runs both models at decisions 0 and 1 and then commits to one driver for the rest of the scene therefore
produces that driver's own rollout, and its score is the logged one. Assumption: the trajectory returned at decision 0 does not affect later states (consistent with the
identical states at decision 1; not tested separately). Signals: both drivers' plans at decisions 0 and 1 (length, lateral offset and heading at 4 s, largest point gap, change
between the two decisions), ego speed and acceleration, the command, the route waypoint. No map, no objects, no logged future.

## On the 400 scenes

SH30 is better in 26 scenes (sum +11.68 scene scores), AP2 in 49 (sum -12.84), 325 ties. One-threshold rules "commit to SH30 when the signal is beyond t, else AP2"; t is fit on
the training folds ("never switch" allowed) and scored on held-out folds, 5 folds split by log (46 logs), 20 repeats; CI = cluster bootstrap over logs of the out-of-fold gain.

| signal (commit to SH30 when ...) | scenes switched at the in-sample t | SH30 better / AP2 better among them | in-sample gain | cross-validated gain [95% CI] | share of +0.0292 |
|:--|--:|--:|--:|--:|--:|
| **ego speed below t** (in-sample t 4.4 m/s) | 147 | 12 / 18 | +0.0133 | **+0.0079 [-0.0037, +0.0188]** | 27% |
| AP2's plan longer than SH30's by more than t m | 20 | 1 / 0 | +0.0025 | -0.0097 [-0.0169, -0.0032] | -33% |
| AP2's plan shorter than SH30's by more than t m | 136 | 16 / 12 | +0.0031 | -0.0090 [-0.0214, +0.0034] | -31% |
| relative length gap above t | 20 | 2 / 2 | +0.0043 | -0.0071 [-0.0149, +0.0014] | -24% |
| lateral gap at 4 s above t m | 0 | 0 / 0 | 0 | -0.0102 [-0.0200, -0.0016] | -35% |
| largest point gap above t m | 20 | 1 / 0 | +0.0025 | -0.0087 [-0.0162, -0.0025] | -30% |
| heading gap at 4 s above t deg | 97 | 12 / 10 | +0.0067 | -0.0058 [-0.0158, +0.0064] | -20% |
| AP2's own lateral offset at 4 s above t m | 0 | 0 / 0 | 0 | -0.0131 [-0.0240, -0.0008] | -45% |
| AP2's plan change between decisions above t m | 28 | 4 / 3 | +0.0002 | -0.0124 [-0.0216, -0.0046] | -42% |
| route waypoint offset above t m | 12 | 3 / 2 | +0.0049 | -0.0066 [-0.0163, +0.0046] | -23% |
| ridge regression on all 17 signals | | | | +0.0032 [-0.0079, +0.0160] | 11% |
| gradient boosting on all 17 signals | | | | -0.0043 [-0.0131, +0.0066] | -15% |

Variants of the speed rule: leave-one-log-out +0.0085 [-0.0036, +0.0198]; threshold restricted to {never, 0.5, 1, 2, 3} m/s +0.0098 [+0.0006, +0.0189] (1 m/s chosen in 78 of 100 fits);
threshold fixed at 1 m/s after seeing the table below, not cross-validated, +0.0117 [+0.0016, +0.0223] (76 scenes; 6 AP2 zeros recovered, 8 SH30 slow scenes paid);
the same with decision-0 speed +0.0092 [-0.0004, +0.0189]. None reaches +0.010 with a lower bound above 0 under cross-validation.

Where the hindsight gain sits (speed at decision 1):

| start speed m/s | scenes | SH30 | AP2 | best of two | zeros SH30 / AP2 | slow SH30 / AP2 |
|:--|--:|--:|--:|--:|--:|--:|
| < 1 | 76 | **0.982** | 0.921 | 1.000 | 0 / 6 | 8 / 1 |
| 1-3 | 28 | 0.925 | 0.963 | 0.964 | 2 / 1 | 3 / 2 |
| 3-6 | 104 | 0.902 | 0.903 | 0.947 | 8 / 8 | 20 / 14 |
| 6-10 | 119 | 0.899 | 0.937 | 0.946 | 11 / 7 | 19 / 9 |
| >= 10 | 73 | 0.973 | 0.972 | 0.973 | 2 / 2 | 0 / 1 |

Half of the hindsight gain (6.0 of 11.7 scene scores) is in the standstill starts; the other half (5.6) is in the 3-10 m/s scenes, where the two drivers fail on different scenes for the same reason (the pre-turn shift,
[c1_zero_review.md](c1_zero_review.md)) and their first plans do not differ in a way that says which one will fail.

## Held-out read (pre-registered)

Frozen rule: ego speed at decision 1 below 1.0 m/s -> SH30 for the whole scene, else AP2. Data: C0b's runs of both drivers on the 300 scenes outside the 400 (30 logs).

| | value |
|:--|--:|
| SH30 / AP2 mean scene score | 0.8919 / 0.9074 |
| best of two minus the best single driver | +0.0259 |
| **registered rule minus the best single driver (AP2)** | **+0.0024 [-0.0066, +0.0131]**, 40 scenes switched |
| threshold 0.5 / 2.0 m/s | +0.0024 [-0.0066, +0.0131] / +0.0057 [-0.0036, +0.0163] |
| decision-0 speed below 1.0 m/s | +0.0024 [-0.0066, +0.0131] |
| standstill scenes: zeros SH30 / AP2; slow SH30 / AP2 | 5 / 7; 6 / 1 |

Line (>= +0.010 and CI lower bound > 0): **not met**. On the new scenes SH30 also takes zeros from standstill, so the split seen on the 400 does not hold up.

## Closed-loop caveat and the cheapest valid test

- Per-scene hindsight is an upper bound for any per-decision switching: once the drivers' states differ (from decision 2), switching changes the future state, and what the other
  driver would have done from there is not in the logs. Only commit-at-decision-1 rules are exact offline, and that is what was measured.
- The cheapest valid closed-loop test of such a rule costs no simulation: apply the frozen rule to runs of both drivers on scenes not used to choose it. That is the read above.
  A real gated driver (both models at decisions 0-1, one afterwards) would only be needed to confirm the equivalence assumption, about 0.1 card-hours on 100 scenes; it was
  not built because the line was not met.

## Limits

One training seed per driver; 400 + 300 scenes from 46 + 30 logs; the positive class is small (26 scenes, 11 of them AP2-only zeros), so the power to find a weak
selector is low: the CIs are about +/- 0.011 wide. Signals from decisions 2 onward and per-decision switching were not evaluated (not exact offline).
