# op_parity self-consist: the model's own road-edge / lead outputs do not carry the map constraint's information (calibrated AUC 0.641 on > 20 deg, line 0.65)

Written 2026-10-08. Read-only diagnosis, no training. Pre-registration: [plans/2026-10-08-self-consist-prereg.md](../plans/2026-10-08-self-consist-prereg.md) (committed before any AUC was read; only the navtrain calibration fit had been printed). Code: `scripts/sc_infer.py` (inference pass, plans bit-identical to the stored ones: max |dplan| 0 m for SH30 and the headline P2H10), `scripts/sc_analyze.py calib | eval | figs`. Tables: [self_consist/](self_consist/) (`metrics.csv`, `coverage.csv`, `calibration.json`); figure: [../figs/self_consist/self_consist_auc.png](../figs/self_consist/self_consist_auc.png).
navtest 12 146 tokens, 136 logs, log-cluster bootstrap B = 1 000. "Family P2H10+SH30" = mean over four (arm, seed) plans.

## Answer

**Stop rule hit: calibrated own-edge margin AUC for DAC failure on > 20 deg tokens is 0.641 [0.590, 0.695] (P2H10+SH30), below 0.65. The idea ends.** The CI straddles the line; the verdict follows the point estimate as pre-registered. Cinque (shipped) is the same: 0.664 [0.621, 0.706].

| AUC for DAC failure, P2H10+SH30 | > 20 | 20-45 | > 45 | > 20 left | > 20 right |
|:--|:--|:--|:--|:--|:--|
| map SDF margin (reference upper bound) | 0.921 [0.901, 0.939] | 0.909 | 0.928 | 0.907 | 0.938 |
| own edge, raw | 0.643 [0.594, 0.695] | 0.659 | 0.630 | 0.617 | 0.656 |
| **own edge, navtrain-calibrated** | **0.641 [0.590, 0.695]** | 0.631 [0.541, 0.707] | 0.644 [0.576, 0.710] | 0.596 | 0.666 |
| own edge, shuffled across tokens (control) | 0.588 [0.552, 0.620] | 0.586 | 0.565 | 0.590 | 0.565 |
| plan-only: end heading change | 0.650 [0.600, 0.699] | 0.660 | 0.655 | 0.653 | 0.642 |

Recall at 10 % FPR on > 20 deg: map 0.733, own edge calibrated 0.243 (raw 0.235), plan-only heading 0.232, shuffle 0.104.

1. **The map margin is the clean upper bound** (0.92 on > 20 deg, 0.87 for Cinque): the footprint-SDF margin of the stored plan almost reproduces the devkit DAC, so the gap to the own-edge margin is the information in the edge output, not the margin definition.
2. **Own edges beat the shuffle control by +0.05 but not the plan-only baseline**: the edge read is token-specific (shuffle 0.588), but a plan-only scalar (how far it turns) predicts DAC failure as well (0.650). Raw and calibrated are the same (0.643 / 0.641): the scale is not what limits the signal.
3. **Calibration**: one global (s, b) fitted on navtrain: s 1.287, b 0.216 m for the fine-tunes (8 608 tokens, 42 474 section pairs), s 1.087, b 0.986 m for Cinque (3 000 tokens). Residual robust sigma still 1.8 m (fine-tunes) / 2.1 m (Cinque): the edge head disagrees with the map boundary by about 2 m per section after calibration, which is the order of the quantity the margin must resolve (decision 88: first exit is 4.7 cm outside). Scatter: the right panel of the figure. The fine-tunes' edge reads are nearly identical (b 0.214-0.217; encoder frozen), and differ from Cinque's (gimm frames, TRT).
4. **Coverage**: 100 % usable at every bucket (raw and calibrated) under the pre-registered criterion (finite, ordered, width >= 2.5 m over the plan's span). The criterion is loose: coverage is not the failure, accuracy is. The "unusable counts as most risky" sensitivity equals the main numbers.
5. **Lead vs collisions (descriptive, no pre-registered line)**: plan-vs-own-lead clearance does not predict NC / TTC failure: AUC 0.517 [0.467, 0.563] (P2H10+SH30), 0.374 [0.336, 0.418] for Cinque (below 0.5: failures have a larger computed clearance), recall at 10 % FPR 0.165 / 0.038; with decision 157's near-range bias correction the same (0.518). On tokens where the lead head fires (48-50 %): 0.564 [0.496, 0.626]. Ego speed alone: 0.46-0.53. Consistent with decision 157: the head's range and its target do not line up with what the devkit penalises.
6. **WOD (item 4)**: not done. The stored WOD val predictions (`processed/wod_zeroshot/preds/op_cinque*`, WP2 / WLG) keep plan, lead and lead_prob only, no road_edges. A re-run is inference only through the parity path on about 1.4-1.6 k frames per (model, seed): well under 0.05 card-hour. Not started: the navtest read already ends the line, and WOD has no map to give the upper bound.

## Not verified

- Per-token margin uses the plan; devkit DAC uses the LQR-tracked trajectory (the map margin at 0.92 shows this matters little).
- One global (s, b), no distance dependence; a longitudinal rescale of the edge points was not tried; a better-fitted calibration (per distance, per curvature) was not tried and is outside the pre-registered rule.
- Fine-tunes were calibrated on navtrain shard 0 of 12, in-sample for training (edge head effectively frozen, so little risk); Cinque calibrated on 3 000 tokens only.
- Lead: only the head's first selection at t = 0, longitudinal clearance only (no lateral term); the NC / TTC target includes pedestrians and side vehicles the lead head does not report.
- Single inference per arm (deterministic); seeds enter only through the plan.
