# op_parity margin critic: a map-supervised head on the frozen representation reads the candidate margin to about 0.5 m and recovers a quarter to a third of the privileged selection gain

Written 2026-10-08. Follow-up of [turn_selector_input.md](turn_selector_input.md) (decision 187). Pre-registration, committed before any number of this line was read, with amendment 1 (leak gate) committed
before any full-scale reading: [plans/2026-10-08-margin-critic-prereg.md](../plans/2026-10-08-margin-critic-prereg.md). Code: `scripts/margin_critic.py` (extract / bank / train / select / report),
`scripts/margin_critic_chain.sh`. Tables: [margin_critic/](margin_critic/); figure: [../figs/margin_critic/margin_critic.png](../figs/margin_critic/margin_critic.png)
(left: predicted against map margin at 4 s, F19 candidates, main head; second: out-of-fold selection gain per margin source, bars = ridge / trees / rule; right: AUC of the identity margin for DAC failure and for
"a repair exists" against the number of navtrain training tokens, with the map, own-road-edge and ego-only references).

Question: decision 187 showed that the exact **map** margin of each candidate is worth +3.2 to +4.2 EPDMS on navtest turn tokens and that SH30's own road-edge output is worth nothing. Is the boundary absent from the
frozen representation, or had nobody trained a head for it? Here a head is trained on the map margin itself (map geometry as a training label only, all 101 499 navtrain training tokens, features from the fold model
that held the token's log out) and read on navtest > 20 deg (3 154 tokens x SH30 s0 / s1), which is never used to fit or select the margin head. No simulator scoring, no navtrain simulator scores.

## Answer

Verdict by the pre-registered rule: **(b1), partially**. One head (trees) has a Bonferroni lower bound above 0, none recovers half of the privileged arm.

| margin given to decision 187's heads | ridge (Q1 / P1) | trees (Q2 / P2) | 1-parameter rule (Q3 / P3) | AUC, DAC failure of the identity | MAE vs map, F19 4 s |
|:--|:--|:--|:--|:--|:--|
| map (PRIV, decision 187) | +3.77 [+2.58, +4.87] | +4.15 [+3.08, +5.29] | +3.21 [+2.08, +4.39] | 0.916 | 0 |
| **MC: frozen vision (8 frames) + SH30 hidden + ego, 101 k tokens** | **+1.34 [+0.17, +2.42]** | **+1.03 [+0.37, +1.70]** | **+0.19 [-0.58, +0.92]** | **0.739 [0.705, 0.774]** | **0.475 m [0.438, 0.517]** |
| ... recovery of the privileged arm | 35.7% [5.9, 57.4] | 24.8% [10.3, 37.4] | 6.0% [-21.8, 24.5] | | |
| ... recovery of the +9.55 ceiling | 14.1% | 10.8% | 2.0% | | |
| MC-V: no hidden state | +1.32 [+0.23, +2.31] | +0.70 [+0.10, +1.34] | +0.06 [-0.26, +0.41] | 0.744 | 0.465 m |
| MC-E: ego only (blind floor) | -0.26 [-0.50, -0.05] | +0.22 [-0.18, +0.64] | -0.25 [-0.53, -0.02] | 0.658 | 0.893 m |
| SH30 own road edge (decision 187, N1-N3) | +0.16 | -0.16 | -4.45 | 0.629 | 2.05 m |

F19 x pc, > 20 deg, EPDMS x 100 without EC, out-of-fold, 95% log-cluster bootstrap CIs. Bonferroni (m = 3, 98.33%) CIs of MC: ridge [-0.07, +2.68], trees [+0.22, +1.84], rule [-0.74, +1.08].
L9 x epcap agrees: +1.39 [+0.24, +2.47] / +1.46 [+0.63, +2.29] / +0.34 [-0.40, +1.05]. The heads are re-fitted inside navtest by 5 folds of logs exactly as decision 187 (same code path, folds and repeats), so the
only difference to the P arms is the margin's source; the selection heads therefore see the simulator scores of the other navtest folds, as every arm of decision 187 does. Full tables: [selection.md](margin_critic/selection.md),
[margin_quality.md](margin_critic/margin_quality.md).

## Readings

1. **Margin quality** (navtest > 20 deg). MC against the map margin: MAE 0.431 m on the identity plan and 0.475 m over the 19 F19 candidates at 4 s (0.34 / 0.36 / 0.40 m at 1 / 2 / 3 s), Pearson r 0.851,
   within-token Spearman over the 19 candidates 0.714 [0.688, 0.739]. SH30's own road edge on the same rows: 2.05 m, r 0.19, Spearman 0.198. AUC for DAC failure of the identity 0.739 [0.705, 0.774]
   (map 0.916, own road edge 0.629, ego-only head 0.658; MC minus ego-only +0.081 [+0.031, +0.131]); AUC for "a repair exists" 0.641 [0.609, 0.678] (map 0.706, own road edge 0.546, ego-only 0.619).
   The head is about 4x more accurate than the model's own boundary output, but it under-calls violations: it puts the identity below 0 on 2.2% of token-seeds against 5.0% for the map.
2. **Selection.** See the table: +1.0 to +1.3 with the learned heads (25-36% of the privileged arm, 11-14% of the ceiling), nothing with the one-parameter rule. Every Q arm is below its P arm by 2.4 to 3.1 (CIs exclude 0).
   A 0.45 m error is too coarse for a threshold rule on margins whose decisive range is a few tenths of a metre; heads that also see ego and the candidate geometry tolerate it better.
3. **Learning curve** ([learning_curve.md](margin_critic/learning_curve.md); nested log subsets, one seed per point):

   | navtrain tokens (logs) | dev MAE 4 s | navtest MAE F19 4 s | AUC DAC | ridge | trees | rule |
   |:--|:--|:--|:--|:--|:--|:--|
   | 2 470 (35) | 0.742 | 0.932 | 0.671 | -0.25 | +0.10 | -0.66 |
   | 9 099 (116) | 0.515 | 0.764 | 0.676 | -0.45 | -0.05 | -0.49 |
   | 31 497 (348) | 0.376 | 0.603 | 0.714 | +0.58 | +0.34 | +0.01 |
   | 101 499 (1 160) | 0.303 | 0.475 | 0.739 | +1.34 | +1.03 | +0.19 |

   By the registered rule the curve is **still rising** at 100% (30% -> 100%: AUC +0.025, MAE -21%). At the 3 k-token scale of decision 187's navtest labels the head is at the ego-only floor, which is why nothing was
   learnable there. Error falls by roughly 20% per tripling of the data with no sign of a plateau; where it ends is not measured.
4. **Reference arm** ([reference.md](margin_critic/reference.md)): the same head on WA-Cf tokens (NAVSIM-trained encoder, one frame of 32 x 512, 25 415 tokens of shards 2-4) against frozen Cinque tokens of the same frame
   and tokens (R-C1). R-WA: MAE 0.392 m, Spearman 0.778, AUC 0.800 [0.771, 0.827], selection +2.37 [+1.12, +3.46] / +1.95 [+1.15, +2.71] / +1.46 [+0.64, +2.29] (63% / 47% / 45% of the privileged arm, all Bonferroni lower bounds > 0).
   R-C1: MAE 0.611 m, AUC 0.708, selection +0.22 / +0.32 / -0.33. R-WA minus R-C1: MAE -0.219 m [-0.239, -0.199], AUC +0.092 [+0.058, +0.123], selection +2.16 / +1.63 / +1.78 (CIs exclude 0).
   By the registered reading the information is **lost in the frozen encoder**: with a quarter of the tokens and one frame instead of eight, the NAVSIM-trained encoder gives a margin that is more accurate than the full MC
   head and would have met threshold (a) on the ridge head. Diagnostic, not a method: WA was trained on navtrain (its training features are in-sample; navtest is held out for it as far as the driving objective goes).
5. **Inputs.** SH30's hidden state adds nothing to the vision tokens (MC-V 0.465 m / 0.744 against MC 0.475 m / 0.739). Eight frames against one frame at matched data: MC-30 (31 k tokens, 8 frames, + hidden) 0.603 m / 0.714,
   R-C1 (25 k tokens, 1 frame) 0.611 m / 0.708, so the extra frames add little (not a clean contrast: token sets and hidden state differ).

## What it means

The frozen representation does carry drivable-boundary geometry that the shipped road-edge head does not expose: a head trained for the footprint margin gets 0.45 m instead of 2 m. Decision 187's null for model-side inputs
was a label-count result, not an absence. But at 100 k tokens that margin buys +1.0 to +1.3, a third of what the map gives, and the two diagnostics point the same way about why: more labels still help (the map label is free,
but navtrain is all there is), and an encoder trained on NAVSIM is clearly better with a quarter of the data. For comparison, the parallel line (decision 191) trains a selector directly on simulator scores of 28 k navtrain turn
tokens with non-privileged inputs and gets +2.75 [+1.55, +3.82] on the same test: supervising the pick directly is worth about twice the margin route here, and the two have not been combined.

## Gates, deviations, cost

- G-label: this line's operator (torch footprint + `grid_sample`) against decision 187's `margins.npz` on all 2 x 33 x 3 154 candidates: max |diff| 6.2e-6 m (footprint 7e-15 m). G-P: the re-run P1 / P3 equal `select2.pkl` bit for bit (max diff 0).
  G-hidden: per-dimension correlation of the fold models' hidden state with SH30-F-s0's on navtest turn tokens, median 0.9965 (read from the turn-selnt lane's `feat/full/navtest.npz`).
- **Deviation (amendment 1).** The registered leak gate (max plan difference < 0.03 m against the stored fold-model plan) failed on the first full extraction: one row each in shards 1, 6 and 9 at 0.03125 m, one fp16 step
  over the tolerance. Before any full-scale training or reading the gate was amended to: at most 0.1% of rows at or over 0.03 m, max <= 0.0625 m, and an own wrong-fold control (next fold's model, 500 rows per shard) with
  >= 50% of rows over 0.03 m. Result over the 12 shards: 3 of 94 867 rows over tolerance, max 0.03125 m; control 84.6-89.2% of rows over, median 0.059-0.070 m. Fold hash and split-membership checks unchanged and passed.
- The first smoke (shard 0) failed on the reference arm because WA-Cf features exist only for shards 2-4; the smoke was moved to shard 2. Smoke readings (482 training tokens, 300 steps) were seen before the full run.
- A reporting bug was fixed after the first report: the within-token Spearman ranked over the seed axis instead of the candidate axis. It is a descriptive reading and not part of the verdict; no other number changed.
- Cost (measured): 8 training jobs 87 min of card time in total (8-14 it/s, peak VRAM 34 GB for the 8-frame arms), 24 extraction jobs about 25 min, smokes about 3 min: about 1.9 card-h and about 10 core-h (budget 6 / 80);
  wall clock about 35 min for the full chain, about 1 h 40 min for the line.

## Limits

- One head configuration, one training seed per arm, no tuning; a different head could move the 0.475 m. Each learning-curve point is one run.
- The label is the plan footprint against the 0.5 m SDF raster, not the LQR-tracked trajectory the devkit scores (decision 187's caveat); points outside the 64 x 48 m raster are clamped.
- Selection heads are re-fitted inside navtest (cross-fit by log) as in decision 187; a selector that never sees navtest scores (rule threshold or head fitted on navtrain) was not run. The rule's threshold grid is decision 187's
  and cannot abstain.
- Reference arm: one frame, 25 k tokens, WA in-sample on its training tokens; whether WA's pre-training saw navtest logs was not checked (decision 165's caveat). The encoder's last block was not unfrozen.
- navtest dev-to-test gap: navtrain dev MAE 0.303 m against 0.475 m on navtest turn candidates (dev is all tokens and F33, test is turns only; not decomposed).
- No straight control, no reactive scoring, no closed loop, no lane-line outputs; multiplicity across the two families is not corrected (verdict on F19 x pc).
