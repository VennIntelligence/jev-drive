# hinge-m10: lambda 30 / margin 1.0 at full scale (2026-10-08, pre-registered)

Plan: [plans/2026-10-08-hinge-m10-prereg.md](../plans/2026-10-08-hinge-m10-prereg.md) (committed before scoring). Chain: `scripts/hinge_m10_chain.sh`. Tables: `results/hinge_m10/`.

## Answer

**SH30 stays.** SH30M10 (lambda 30 / margin 1.0, SH30 recipe otherwise, full scale, 2 seeds) fails two of the three rule conditions:

| rule condition | result | pass |
|:--|:--|:--|
| 1. navtest EPDMS vs SH30, CI lower bound > 0 | +0.02 [-0.36, +0.41] | no |
| 2a. EP vs SH30, CI upper bound >= 0 | -0.00 [-0.05, +0.04] | yes |
| 2b. straight (S5) EPDMS vs SH30, CI upper bound >= 0 | -0.54 [-0.96, -0.18] | **no** |
| 3. HUGSIM 64 HD vs SH30, CI upper bound >= 0 | -0.039 [-0.103, +0.021] | yes |

The pilot picture carries to full scale in sign: more margin removes more out-of-bounds (DAC) and turn failures but pays on straight roads (lane keeping), so the
total does not move. The full-scale navtest total (+0.02) is lower than the pilot's +0.32; the straight cost is larger (-0.54 vs -0.31). The one point decision 172
left open is closed: hinge strength is one trade-off axis and SH30 sits at its saturation point.

## navtest (12 146 tokens, 136 logs, seed means, log-cluster paired bootstrap)

| metric | stratum | P2H10 | SH30 | SH30M10 | M10 - SH30 | M10 - P2H10 |
|:--|:--|--:|--:|--:|:--|:--|
| EPDMS | all | 88.67 | 89.55 | 89.57 | +0.02 [-0.36, +0.41] | +0.90 [+0.41, +1.42] |
| EPDMS | straight (S5) | 92.94 | 93.41 | 92.87 | -0.54 [-0.96, -0.18] | -0.07 [-0.52, +0.38] |
| EPDMS | turn > 20 deg | 80.28 | 81.49 | 82.81 | +1.32 [+0.63, +2.04] | +2.53 [+1.53, +3.51] |
| EPDMS | turn > 45 deg | 78.61 | 79.49 | 80.33 | +0.84 [+0.06, +1.56] | +1.72 [+0.69, +2.69] |
| EP | all | 87.16 | 87.16 | 87.16 | -0.00 [-0.05, +0.04] | -0.00 [-0.06, +0.05] |
| EP | straight | 88.66 | 88.65 | 88.61 | -0.04 [-0.11, +0.02] | -0.05 [-0.11, +0.02] |
| DAC (score) | all | 96.17 | 97.13 | 97.81 | +0.67 | +1.63 |
| LK (score) | all | 97.39 | 97.37 | 95.88 | -1.49 | -1.51 |
| NC+TTC fail % | all | 2.19 | 2.17 | 2.40 | +0.22 [-0.06, +0.51] | +0.21 [-0.10, +0.52] |
| DAC fail % | turn > 20 / > 45 | 8.80 / 10.18 | 7.31 / 9.03 | 5.63 / 7.65 | -1.68 [-2.40, -1.01] / -1.38 [-2.11, -0.69] | |
| inside-cut % | turn > 20 / > 45 | 4.80 / 4.78 | 3.96 / 4.28 | 2.79 / 3.59 | -1.17 [-1.75, -0.64] / -0.69 [-1.27, -0.23] | |
| cannot-make-turn % | turn > 20 / > 45 | 1.90 / 2.60 | 1.54 / 2.47 | 1.35 / 2.11 | -0.19 [-0.41, +0.00] / -0.36 [-0.68, -0.11] | |

Per seed EPDMS: SH30M10 89.60 / 89.55, SH30 89.47 / 89.63. Full tables: `hinge_m10/turn/tables.md`, `hinge_m10/navtest_*.md`.

## Out-of-bounds split (all tokens, raw plan vs devkit replay, M10 - SH30, paired per seed)

| set | seed | replay out (DAC fail) | raw plan out | replay only | raw only |
|:--|:--|:--|:--|:--|:--|
| all | 0 | -0.79 [-1.09, -0.51] | -0.77 [-1.08, -0.47] | -0.21 [-0.40, -0.02] | -0.19 [-0.35, -0.04] |
| all | 1 | -0.55 [-0.83, -0.30] | -0.75 [-1.05, -0.48] | -0.12 [-0.30, +0.07] | -0.31 [-0.50, -0.14] |
| turn > 20 | 0 / 1 | -1.81 / -1.55 | -1.65 [-2.41, -0.95] / -1.93 [-2.75, -1.20] | -0.41 / -0.29 (CI contains 0) | -0.25 / -0.67 |
| turn > 45 | 0 / 1 | -1.65 / -1.12 | -2.18 [-3.25, -1.14] / -2.37 [-3.64, -1.27] | +0.07 / +0.07 (CI contains 0) | -0.46 / -1.19 |

The extra margin is a real raw-plan gain (out-of-bounds 2.45 -> 1.68 pp, 2.40 -> 1.65 pp; vs P2H10 -1.9 / -2.0 pp), not replay-specific. It does not convert into EPDMS
because lane keeping drops 1.5 points and NC+TTC failures rise 0.2 pp (not significant).

## navhard (GIMM frames, combined EPDMS, 225 scenes, 76 units)

| arm | combined | stage 1 | stage 2 | M10 - arm |
|:--|--:|--:|--:|:--|
| SH30M10 | 34.41 (34.84 / 33.98) | 77.58 | 44.10 | |
| SH30 | 33.67 | 75.90 | 44.68 | +0.74 [-0.78, +2.43] |
| P2H10 | 31.84 | 74.73 | 43.04 | +2.57 [+0.51, +5.06] |

Not gating; no difference from SH30 (stage 1 +1.68 [-0.11, +3.64], stage 2 -0.59 [-2.50, +1.32]).

## HUGSIM 64 (`spec_plan_smooth`, HD, one run per scenario per seed)

SH30M10 0.399 (0.415 / 0.383), SH30 0.439 (0.443 / 0.434), P2H10 0.433. M10 - SH30 -0.039 [-0.103, +0.021]; M10 - P2H10 -0.034 [-0.085, +0.018]. The point estimate is lower
than both; the seed spread of SH30M10 (0.032) is as large as the gap. Counts (seed means): stuck 0 / 0, spin 1 vs 0, launch stall 1 / 1, off-route 2 vs 1. See `hinge_m10/hugsim_*`.

## Deviations

- The chain's first attempt of the report stage failed on a typo of mine (a nonexistent replay set `sw`); fixed and resumed from the finished training and scoring, no result was recomputed
  with different settings.
- Scoring ordering/knowledge: training and benchmark runs finished before the report stage; the rule above was committed before any SH30M10 score existed, and the numbers above are read against it unchanged.
- No rc 137 / SIGKILL occurred in any job of this chain; RSS trends were not collected.

## Unverified

- Two seeds, one HUGSIM run per scenario per seed; HUGSIM CI for the HD contrast includes 0 but its upper bound is only +0.021.
- Only margin 1.0 at lambda 30 was run at full scale; intermediate margins (0.5 to 1.0) and lambda 10 / 1.0 (pilot +0.05) were not.
- "Real" path gain is measured against the devkit drivable polygons only.
- The straight-road cost is read from the S5 stratum (logical heading change < 5 deg over 4 s); its mechanism (lane keeping score 97.37 -> 95.88) was not dissected.
