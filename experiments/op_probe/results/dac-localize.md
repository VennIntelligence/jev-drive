# DAC localization: P2's navtest drivable-area failures start in the frozen comma vision features; the temporal path does not lose them, the plan head's objective recovers about a third

Written 2026-10-06. Pre-registration [../plans/2026-10-06-dac-localize-prereg.md](../plans/2026-10-06-dac-localize-prereg.md) (addendum 1 written after
the small read, before the full read). Code `scripts/opb_labels.py` (labels), `opb_feats.py` (P2 / P0 taps, input ablations), `opb_wajepa.py`
(WA-JEPA taps), `opb_score.py` (per-token devkit `pdm_score`), `opb_probe.py` (sets, probes, decoders), `opb_report.py` (tables). Tables:
[probe.md](probe.md), [decode.md](decode.md), [decode_h10.md](decode_h10.md), [decode_paired.md](decode_paired.md), [ablate.md](ablate.md),
[fsplit.md](fsplit.md); small read [probe_small.md](probe_small.md), [decode_small.md](decode_small.md). Run data `$DATA_DIR/runs/op_probe/`.

## Answer

**Verdict: (a), in graded form, plus a smaller (c) component; (b) is not supported.** Evidence: medium for the decomposition (paired,
selection-free decoder contrasts whose CIs exclude 0; readout declared in addendum 1 before the full read). By the pre-registered probe rule
(section 5) no class is met, so formally the verdict is "undetermined, nearest (a)", weak: the comma vision tokens do carry coarse road geometry
(rule (a) needed the vision probe skill below half of WA-JEPA's; it is 0.42 vs 0.59), and the pre-registered AUC readout is confounded (below).

DAC failure rate on all of navtest (stratified: F / R / FF in full, 1 500 random both-pass tokens reweighted; % of tokens):

| what | DAC fail % [95% CI] |
|:--|:--|
| P2 itself (P2-F-s0, W frames) | 4.42 [3.58, 5.36] |
| WA-JEPA itself (devkit) | 1.80 |
| thin MLP head on Cinque vision tokens + ego (P2-V, drivable hinge lambda 10) | 5.12 [3.98, 6.41] |
| same head on WA-JEPA front encoder tokens + ego (WA-Cf, lambda 10) | 3.41 [2.64, 4.23] |
| same head on P2's plan-head hidden + ego (P2-H, lambda 10) | 3.49 [2.66, 4.43] |
| same head on WA-JEPA's trajectory-head hidden + ego (WA-H, lambda 10) | 1.56 [1.17, 2.00] |

| paired contrast (same tokens) | DAC fail pp | per-token score x100 |
|:--|:--|:--|
| encoder: P2-V - WA-Cf (lambda 10) | **+1.71 [+0.69, +2.78]** | -2.68 [-3.86, -1.60] |
| encoder: P2-V - WA-Cf (lambda 1 / imitation only) | +1.63 [+0.37, +3.07] / +1.96 [+0.29, +3.66] | -2.85 / -2.82 |
| head objective: P2-H (lambda 10) - P2 | **-0.93 [-1.69, -0.11]** | +0.62 [-0.22, +1.41] |
| head capacity only: P2-H (imitation) - P2 | +1.13 [+0.25, +2.12] | -1.20 [-2.15, -0.36] |
| what is left at the head: P2-H - WA-H (lambda 10) | **+1.93 [+1.13, +2.84]** | -2.98 [-3.96, -2.08] |
| a thin head on WA's frozen front encoder vs all of P2: WA-Cf (lambda 10) - P2 | -1.01 [-1.88, -0.18] | +0.91 [-0.10, +1.84] |

Reading: the gap P2 - WA-JEPA in DAC failures (2.6 pp; 2.9 pp against WA-H's head) splits into about 0.9 pp that a drivable-aware objective on P2's
existing head features recovers (c) and about 1.9 pp that no head on P2's features recovers. That remainder is the same size as the encoder gap
measured with identical thin heads (1.7 pp), and each model's policy path adds a similar amount over its own encoder (P2 V -> H: 5.12 -> 3.49;
WA Cf -> H: 3.41 -> 1.56): the deficit enters at the vision features and is carried through, not created downstream.

Supporting readouts:
- **Probes** (ridge on [stage, ego], 25.3 k navtrain tokens, navtest): drivable SDF raster test R^2 Cinque vision 0.66 vs WA front encoder 0.78
  (MLP probes 0.80 vs 0.89); corridor-distance skill over the ego-only probe on F 0.42 vs 0.59, on R (WA's failures) 0.48 vs 0.58: WA's encoder
  is better everywhere, also where WA fails, so this is not selection.
- **Where P2 fails, its own representations think the road continues.** Sampling each stage's predicted SDF along P2's own plan footprint
  (post hoc, addendum 1), the predicted margin on F is too optimistic by +0.75 m [0.56, 0.94] (vision) and +0.75 m (plan head), vs +0.38 m (WA
  front encoder) and +0.11 m [-0.06, 0.29] (WA all-view encoder); on turn-matched passing tokens all stages are within -0.2..+0.1 m.
- **Failures are inherited from the encoder, token by token.** Fresh decoders on Cinque vision + ego pass only 50% of F (ego-only decoders: 44%)
  but 74% of R; decoders on WA's encoder pass 77% of F but only 44% of R (ego-only: 43%). Each model's failure tokens are where its own encoder
  features add nothing over ego state.

## Setup

- **Model and stages** (P2 = op_parity P2-F-s0, frozen vision, W frames; taps reproduce its stored navtest plans bit-exactly): V `view_39`
  current-frame vision tokens (32 x 512), M `add_40` (after the adapter bias and two token-MLP blocks), T `select_4` (summary token after the
  4-layer temporal transformer), H `add_54` (plan-head hidden; one Gemm before the plan). P0 (shipped) T / H as reference. WA-JEPA (released
  checkpoint, bf16 batched; its plans within 0.12 m ADE of its stored navtest export): Cf = front view, newest tubelet of `context_scene`,
  4 x 4 pooled to 32 x 512 (like-for-like with V), Ca = all four views, 8 x 8 pooled to 32 tokens, T = input of `traj_out`, H = input of its last
  Linear (8 x 512 each). E = the 20 ego / pose / command numbers P2 receives; every probe and decoder sees [stage, E].
- **Labels**: drivable signed-distance raster in the t0 rear-axle frame from the nuPlan map API with the scorer's own DAC polygon types
  (ROADBLOCK, INTERSECTION, CARPARK_AREA), 0.5 m, x -8..56 m, y +-24 m; agrees with the v2 metric cache's DAC polygons on 99.97% of cells
  (300 tokens); every logged navtest future lies inside. Probe target: 1 m raster (3 072 cells, clipped +-10 m) and 6 corridor distances
  (free distance left / right of the logged path at 5 / 10 / 20 m arc length).
- **Probes**: ridge (z-scored, lambda on navsim/op-parity-full-dev logs) and a 2-layer MLP (report-only); train navtrain shards s2-s4 (25.3 k
  after dev logs), test navtest (log-disjoint). **Decoders**: 2-layer MLP [stage, E] -> 8 poses on the same rows, imitation of the logged poses,
  plus a hinge on the true SDF at the interpolated footprint corners (lambda 1 pre-registered; lambda 10 added after seeing lambda 1), scored with
  the devkit's `pdm_score` (`opb_score.py`, reproduces the devkit DAC on 400 / 400 checked tokens).
- **Sets** (navtest, P2-s0 vs WA-JEPA DAC from op_parity's gap tables): F = P2 fails, WA passes (433); R = WA fails, P2 passes (115); FF both fail
  (104); PP both pass (11 494; 1 500 random scored); PP-turn = PP with a logged heading change > 20 deg in 4 s. F is mostly turning (60% > 20 deg,
  median 29 deg; PP 24%, median 4 deg).

## F is a set of shallow, recipe-stable failures

| set | n | raw plan footprint out | LQR-only | max depth < 0.3 m | median depth |
|:--|--:|--:|--:|--:|--:|
| F | 433 | 72% | 28% | 50% | 0.30 m |
| F-plan (raw plan out) | 310 | 100% | 0 | 37% | 0.42 m |

The other seed of the same recipe (P2-F-s1) fails 89% of F too (seed control in [ablate.md](ablate.md)): the failures belong to the recipe /
representation, not to seed noise. Half are grazes under 0.3 m (decision 112's scorer-side share on navhard is the same order), so the verdict
was also checked on F-plan: every pattern above holds there (decoders on vision pass 47-52% of F-plan, on WA's encoder 76-83%).

## Probes per stage (ridge; corridor MAE in m, skill = 1 - MAE / MAE(ego probe))

| stage | raster R^2 | MAE F [CI] | MAE PP-turn | MAE R | skill F | skill PP-turn | skill R | F - PP-turn excess [CI] | margin bias F [CI] | AUC F vs PP |
|:--|--:|:--|--:|--:|--:|--:|--:|:--|:--|--:|
| E (ego only) | 0.15 | 4.56 [4.25, 4.82] | 4.29 | 4.70 | 0 | 0 | 0 | 0.27 [0.03, 0.52] | +0.81 [0.42, 1.17] | 0.75 |
| P2-V vision | 0.66 | 2.65 [2.40, 2.89] | 2.26 | 2.46 | 0.42 | 0.47 | 0.48 | 0.39 [0.18, 0.62] | +0.75 [0.56, 0.94] | 0.71 |
| P2-M + bias, token MLP | 0.68 | 2.53 [2.30, 2.74] | 2.13 | 2.30 | 0.44 | 0.50 | 0.51 | 0.40 [0.21, 0.60] | +1.01 [0.83, 1.18] | 0.71 |
| P2-T temporal summary | 0.49 | 3.12 [2.87, 3.36] | 2.70 | 3.18 | 0.31 | 0.37 | 0.32 | 0.42 [0.20, 0.67] | +0.71 [0.45, 0.96] | 0.76 |
| P2-H plan-head hidden | 0.48 | 3.23 [2.99, 3.47] | 2.72 | 3.25 | 0.29 | 0.37 | 0.31 | 0.51 [0.29, 0.77] | +0.75 [0.50, 0.99] | 0.76 |
| P0-H (shipped) | 0.47 | 3.23 [2.98, 3.48] | 2.79 | 3.33 | 0.29 | 0.35 | 0.29 | 0.44 [0.18, 0.74] | +0.91 [0.64, 1.18] | 0.74 |
| WA-Cf front encoder | 0.78 | 1.89 [1.67, 2.10] | 1.61 | 1.96 | 0.59 | 0.63 | 0.58 | 0.28 [0.11, 0.48] | +0.38 [0.20, 0.58] | 0.76 |
| WA-Ca all-view encoder | 0.81 | 1.54 [1.35, 1.75] | 1.32 | 1.71 | 0.66 | 0.69 | 0.64 | 0.22 [0.06, 0.41] | +0.11 [-0.06, 0.29] | 0.80 |
| WA-T | 0.67 | 2.21 [2.00, 2.44] | 1.97 | 2.49 | 0.51 | 0.54 | 0.47 | 0.25 [0.05, 0.48] | +0.33 [0.09, 0.59] | 0.77 |
| WA-H | 0.63 | 2.37 [2.17, 2.58] | 2.18 | 2.66 | 0.48 | 0.49 | 0.43 | 0.19 [-0.02, 0.42] | +0.39 [0.18, 0.63] | 0.76 |

MLP probes (report-only): R^2 P2-V 0.80, P2-H 0.68, WA-Cf 0.89, WA-Ca 0.91, WA-H 0.80; F skill 0.55 / 0.50 / 0.70 / 0.75 / 0.61. True-map AUC
0.95; 1 m resolution floor 0.06 m. Full columns (direct corridor ridge, F-plan, FF, CIs) in [probe.md](probe.md).

- Road geometry decays from vision to the plan head in both models (P2 R^2 0.66 -> 0.48, WA 0.78 -> 0.63); the decay is not failure-specific in
  P2 (F - PP-turn excess 0.39 -> 0.51 m, CIs overlap) and fine-tuning did not change the head's road content (P2-H = P0-H).
- The pre-registered M2 (AUC of the predicted footprint margin, F vs PP) is dominated by the plan's shape: the ego-only probe already gets 0.75,
  every stage 0.71-0.80. It does not separate the classes; the margin bias column (post hoc) does.

## Decoders per stage (stratified navtest DAC fail %, per-token score x100; F / R pass %)

| decoder [stage, E] | imitation: fail % | imit F / R pass | hinge 1: fail % | hinge 10: fail % | hinge 10 F / R pass |
|:--|:--|:--|:--|:--|:--|
| E | 16.88 | 44 / 43 | 15.95 | 14.21 | 54 / 53 |
| P2-V | 7.28 | 50 / 74 | 6.20 | 5.12 | 54 / 79 |
| P2-M | 6.12 | 47 / 73 | 5.60 | 4.70 | 52 / 81 |
| P2-T | 5.73 | 41 / 72 | 4.51 | 4.01 | 55 / 89 |
| P2-H | 5.56 | 41 / 79 | 4.65 | 3.49 | 52 / 86 |
| WA-Cf | 5.33 | 77 / 44 | 4.57 | 3.41 | 82 / 50 |
| WA-Ca | 5.40 | 82 / 35 | 4.04 | 3.97 | 84 / 51 |
| WA-H | 2.46 | 91 / 32 | 1.87 | 1.56 | 92 / 50 |

CIs and PP / FF columns in [decode.md](decode.md) / [decode_h10.md](decode_h10.md); paired contrasts in [decode_paired.md](decode_paired.md).
F and R pass rates are selection-biased in opposite directions (F is chosen by WA passing, R by P2 passing); the stratified fail % is not.

## Input ablations of P2 (stratified DAC fail %, F / F-plan / R pass %, PP score change x100)

| variant | DAC fail % | F | F-plan | R | PP score delta |
|:--|:--|--:|--:|--:|:--|
| full | 4.42 [3.58, 5.36] | 0 | 0 | 100 | 0 |
| P2-F-s1 (seed control, devkit) | 4.33 [3.49, 5.25] | 11 | 7 | 96 | |
| velocity / acceleration zeroed | 3.21 [2.53, 3.97] | 49 | 41 | 90 | -3.36 [-4.31, -2.46] |
| pose history zeroed | 4.49 | 6 | 4 | 97 | -0.51 |
| command zeroed | 4.86 | 23 | 21 | 89 | -1.42 |
| command forced straight | 4.02 | 27 | 26 | 95 | -0.48 [-0.92, -0.04] |
| whole adapter bias off (present = 0) | 6.38 | 39 | 38 | 77 | -6.45 |
| current frame only (8 older slots zeroed) | 15.16 | 47 | 48 | 51 | -15.15 |
| front frames GIMM instead of warp | 5.54 | 45 | 40 | 80 | -3.66 |
| front frames native 2 Hz keys | 17.43 | 27 | 22 | 55 | -32.07 |

Every perturbation that changes the plan flips 23-49% of F to pass (shallow failures, regression to the mean) and breaks some R / PP tokens; none
lowers navtest-wide failures without losing score (zeroing velocity is fewer failures through slower plans, PP score -3.4). The command matters
for about a quarter of F (forcing "straight" passes 27%), the pose history for almost none. No single input is misused in a way an input change
fixes.

## Pre-registered rule, checked (section 5)

| class | condition | value | met |
|:--|:--|:--|:--|
| (a) | sk(V, F) < 0.5 sk(WA-C, F) and AUC(V) < 0.65 | 0.42 vs 0.29 (Cf) / 0.33 (Ca); AUC 0.71 | no |
| (b) | V has it: sk(V, F) >= 0.75 sk(WA-C, F) or AUC(V) >= 0.75 | 0.42 vs 0.44 (Cf) / 0.50 (Ca); 0.71 | no (direct-corridor ridge: 0.39 vs 0.38, yes vs Cf) |
| (b) | drop V -> H >= 0.15 skill or 0.10 AUC, larger on F than on PP | skill drop F 0.13, PP-turn 0.11; AUC rises 0.71 -> 0.76 | no |
| (c) | AUC(H) >= 0.75, sk(H, F) >= sk(V, F) - 0.10 | 0.76; 0.29 vs 0.32 | AUC yes, skill no |
| (c) | D-hinge(H) passes >= 40 pp more of F than P2, PP score drop <= 1 | 46% (ego-only decoder 45%); PP 90.7 vs P2 92.9 | F yes but uninformative; PP no |

None met -> "undetermined, nearest class, weak" as written. The rule was built on two readouts the small read showed to be confounded (M2's
plan-shape prior; F's selection on WA passing), so addendum 1 moved the main readout to the stratified decoders before the full read. Those give
the decomposition in the Answer.

## What it implies for the next training step

1. **Not a lateral on-policy engine for this gap.** navtest failures are single-step on logged states; the deficit is in what the features say
   about the road, not in compounding state drift. (On-policy remains the lever for navhard stage 2, decision 145.)
2. **Not head capacity; partly the head's objective.** A fresh MLP head on P2-H with imitation only is worse than P2 (+1.1 pp); with a drivable
   hinge it is 0.9 pp better at no score cost. Cheapest next step: add the footprint SDF hinge (labels from `opb_labels.py`, all 103 k navtrain
   tokens in about a minute of CPU) to P2's plan-pathway fine-tune; expected ceiling about -0.9 pp DAC failures (per-token score +0.62 [-0.22, +1.41] in the decoder test,
   about a quarter of the DAC part of the EPDMS gap).
3. **The larger part (about 1.9 pp, two thirds) needs a different vision representation.** Decision 145's light unfreezing on the imitation loss
   moved nothing; the vision tokens are not short of capacity to read roads (MLP probe R^2 0.80) but of precise road-edge geometry in turns,
   which imitation does not supervise. Candidates, in cost order: (i) unfreeze the encoder (stage 4 or LoRA, higher lr than U2) with a dense
   drivable-SDF auxiliary head on `view_39` and the hinge; gate on navtest DAC failures <= 3.5%; (ii) give the P2 adapter WA-JEPA's / V-JEPA 2.1
   front encoder tokens as extra memory (a thin head on WA's frozen front tokens already beats all of P2 by 1.0 pp): a "trade" test, it imports
   the other pretraining rather than adapting comma's.
4. Side cameras: WA's all-view tokens map the road better (R^2 0.81 vs 0.78, margin bias 0.11 vs 0.38 m) but decode no better than front-only
   (3.97 vs 3.41%), consistent with P3 = P2 (decision 144).

## Deviations

- WA-JEPA features from batched bf16 forwards (flow noise per batch, not per sample); plans stay within 0.12 m ADE of its stored export.
- Probes trained on 3 navtrain shards (25.3 k tokens) instead of the pre-registered 5 (WA-JEPA extraction cost, ~4.3 tokens/s per card); labels
  exist for 5.
- "WA-C" split into Cf (front, the like-for-like comparison used in the rule) and Ca (all views).
- Post hoc / added after the small read: margin bias, PP-turn set, R / FF and stratified readouts (addendum 1, before the full read); hinge
  lambda 10 (after seeing lambda 1).
- HUGSIM launch-stall localization (secondary) not done: it needs per-step taps of X and C inside closed-loop runs, not cheap.
- Three early pool submissions were cancelled or failed (an argparse tag, two npz re-read slowdowns, a scheduler edge case); none produced results
  used here.
