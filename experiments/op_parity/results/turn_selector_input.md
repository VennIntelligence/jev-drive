# op_parity turn selector inputs: map margin recovers about 40% of the de-watered ceiling, model-side inputs recover none

Written 2026-10-08. Follow-up of [turn_ceiling_dewater.md](turn_ceiling_dewater.md) (decision 186). Pre-registration, committed before any selector number was read:
[plans/2026-10-08-turn-selector-input-prereg.md](../plans/2026-10-08-turn-selector-input-prereg.md). Code: `scripts/turn_selinput.py` (feats / select / nn / report),
`scripts/tsi_extract.py` (SH30 hidden state, pool GPU job), `scripts/turn_selinput_chain.sh`. Tables: [turn_selector_input/](turn_selector_input/); figure:
[../figs/turn_selector_input/selector_inputs.png](../figs/turn_selector_input/selector_inputs.png) (left: out-of-fold gain per arm, F19 x pc, > 20 deg; right: learning curves of E,
the best privileged arm P2 and the best-point non-privileged arm N7 on both families). Same 3 154 navtest turn tokens x 2 seeds, same folds by log, same families and
conventions as decision 186; no new simulator scoring. **Every row marked PRIV uses map geometry and is an upper bound, not a method.**

## Answer

Verdict by the pre-registered rule: **case (i)**, marginally. Exact map margin recovers 34-44% of the F19 x pc ceiling (+9.55); no model-side input recovers anything.

| arm (head) | input | F19 x pc, > 20 deg | recovery | L9 x epcap |
|:--|:--|:--|:--|:--|
| E (ridge), control = decision 186 | ego + plan | -0.27 [-0.52, -0.06] | -2.8% | -0.23 [-0.51, +0.01] |
| P1 (ridge) PRIV | E + map margin of every candidate | +3.77 [+2.58, +4.87] | 39.4% [30.3, 47.3] | +4.15 [+3.05, +5.24] |
| P2 (trees) PRIV | E + candidate params + own / identity / best map margin | +4.15 [+3.08, +5.29] | 43.5% [34.8, 52.3] | +4.42 [+3.25, +5.64] |
| P3 (1-parameter rule) PRIV | map margin only | +3.21 [+2.08, +4.39] | 33.6% [23.1, 44.0] | +3.46 [+2.41, +4.60] |
| N1 (ridge) | E + SH30 own-edge margin per candidate | +0.16 [-0.73, +0.95] | 1.7% | +0.04 [-0.84, +0.83] |
| N2 (trees) | same, candidate rows | -0.16 [-0.59, +0.30] | -1.6% | +0.13 [-0.30, +0.58] |
| N3 (rule) | own-edge margin only | -4.45 [-5.92, -2.96] | -46.6% | -4.45 [-5.92, -2.96] |
| N4 (ridge) | E + SH30 hidden state (select_4 + mean, PCA) | +0.30 [-0.50, +1.02] | 3.1% | +0.51 [-0.32, +1.28] |
| N5 (attention) | E + unpooled vision tokens | +0.11 [-0.71, +0.91] | 1.1% | +0.36 [-0.45, +1.18] |
| N6 (MLP) | E + SH30 hidden state | +0.13 [-0.95, +1.21] | 1.4% | +0.94 [-0.01, +1.88] |
| N7 (attention) | tokens + hidden + own-edge margins + E | +0.32 [-0.65, +1.26] | 3.4% | +0.68 [-0.38, +1.68] |

95% CIs, log-cluster bootstrap (B 10 000). Bonferroni CIs (98.33% for P, 99.29% for N) are in [turn_selector_input/selector.md](turn_selector_input/selector.md): P1 [+2.30, +5.10],
P2 [+2.87, +5.56], P3 [+1.84, +4.65]; every N arm's contains 0 (N3 is significantly negative).

Pre-registered reading: privileged "large" needs a Bonferroni lower bound > 0 and recovery >= 40%; only P2 meets it (43.5%; P1 39.4%, P3 33.6%, all three recovery CIs
straddle 40%). No N arm is positive after correction (best point estimates N7 +0.32 and N4 +0.30 on F19 x pc; none has an unadjusted lower bound > 0). Case (iii) does not apply.

## What the readings say

- **Geometry is what is missing, and it is worth about 4 EPDMS, not the whole 9.55.** Adding the map margin moves the selector from -0.27 to +3.8 / +4.2 / +3.2
  (P - E +3.5 to +4.4, CIs exclude 0). Even with exact margin about 56-66% of the ceiling stays out of reach: the margin covers boundary repairs, the ceiling also contains NC / TTC / LK repairs
  and tracking effects that a footprint margin does not see (descriptive: identity map margin < 0 on 5.0% of token-seeds, a repair exists on 15.3%; 99.3% of the repairs have some candidate with a positive map margin).
- **Learning curves of the privileged arms are flat above 50% of the logs** (P1 +3.44 -> +3.77, P2 +3.61 -> +4.15, P3 +3.30 -> +3.21 for 50% -> 100%): the label count is not what limits them.
- **The model's own boundary is useless per candidate.** SH30's calibrated own-edge margin has AUC 0.546 [0.506, 0.578] for "a repair exists" against 0.706 for the map margin, and it says the identity
  is unsafe on 55% of token-seeds (map: 5%); the one-parameter rule on it moves 28.6% of rows and costs -4.45. Learned heads on it (N1 / N2) do not beat E. Decision 179's calibrated residual of 1.8-2.1 m is the reason.
- **Richer model-side inputs do not help at 3 k tokens.** Unpooled tokens, SH30's hidden state and everything together are +0.1 to +0.3 on F19 x pc, CIs about +-1. On L9 x epcap N6 is +0.94 [-0.01, +1.88]
  and N6 - E is +1.17 [+0.33, +2.04] (unadjusted contrast, exploratory, not a verdict arm). Their learning curves rise from -1.7 / -3.0 at 25% to about 0 / +0.3 at 100%
  (75% -> 100% step >= +0.3 for N4-N7); with 3-repeat points below 100% and 5-10 repeats at 100% this step is partly noise. Extrapolating to navtrain scale (about 9x the tokens) is speculation.
- **AUC for "a repair exists"** (base rate 15.3%, [auc_repair_exists.md](turn_selector_input/auc_repair_exists.md)): ridge on E 0.692, + map margins (PRIV) 0.758, + own-edge margins 0.693, + hidden state 0.686;
  head scores of the N arms 0.45-0.55. Map margins add detection, but modestly.

## Gates and controls

G1 map-margin AUC for DAC failure of the SH30 identity 0.916 (>= 0.85); G2 own-edge margin 0.629 (in [0.58, 0.72]); G3 SH30 re-run plans identical to the stored ones (max diff 0 m, both seeds);
G4 the E arm reproduces decision 186's out-of-fold gains bit for bit (same code path, max |diff| 0). One neural smoke unit (N5, rep 0, F19 x pc) and the progress log of the full neural run were seen before the report;
the protocol and thresholds were fixed before.

## Limits

- The margin is the plan's footprint against the SDF raster, not the LQR-tracked trajectory the devkit scores; slow-down candidates share the path of the full-speed one, and the rule head P3 / N3 only considers speed >= 1.
  P2 is a single tree configuration; ridge / tree / rule heads were not tuned beyond the pre-registered grids.
- Neural heads: one architecture, epoch picked on a 20% inner split of the training logs, 5 repeats (3 for curves). A different head or auxiliary supervision (e.g. predicting the SDF margin, then selecting) was not tried; it is the natural way to test whether
  the frozen features contain the boundary at all. Decision 179 / the turn probe suggest they do poorly, but that is not measured here.
- Lane-line outputs per candidate were not evaluated (no pre-registered correspondence to DAC). Only the SH30 road-edge head was used as the model's boundary.
- Two privileged ingredients are mixed in P arms (margin plus the candidate geometry); P3 isolates the margin alone (33.6%).
- Selection uses the privileged ceiling's definition on navtest turn tokens by logged heading; no straight control, no reactive, no closed loop. Multiplicity across the two families is not corrected (verdict uses F19 x pc; L9 x epcap agrees in sign for every arm).
