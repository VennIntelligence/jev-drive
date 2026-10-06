# View vs turning: the view contrasts stratified by logged heading change. Added cameras and a taller virtual camera do not help sharp turns; the representation deficit does grow with turn sharpness

Written 2026-10-06. Script `scripts/opj_view_by_turn.py` (CPU, stored per-token outputs only, no model runs; box tables in `$DATA_DIR/runs/op_probe/view_by_turn/`).
Bins and clustering as the joint diagnosis ([joint/](joint/index.html), `opj_figs.py` `TURN_BINS`): logged |heading change| over the 4 s future in
0-5 / 5-20 / 20-45 / > 45 deg, plus left / right (sign of the heading change) within the three turning bins; 95 % bootstrap over the 136 navtest logs
(B 4 000). EPDMS x 100 per token (v2 devkit, W frames, seed means over s0 / s1), DAC fail = DAC sub-score < 1 (seed mean of the indicator), in % of tokens.

## Answer

The averages did not hide a sharp-turn effect for any view change that was *trained*. P3 - P2 (side / rear cameras) is null in every bin, including > 45 deg
(EPDMS -0.52 [-1.72, +0.62], DAC fail +1.05 pp [-0.23, +2.34]); masking the side cameras at test time, if anything, lowers DAC failures on sharp turns
(-1.42 pp [-2.73, -0.26]), the wrong direction for "side views help". The 1.40 m virtual camera after fine-tuning (V - F, pilot scale) is +1.07 [+0.02, +2.15] on
> 45 deg but -0.72 [-2.01, +0.54] on 20-45 deg, with DAC fail unchanged: not a coherent turn effect. The one view change with a clear sharp-turn signature is the
**zero-shot** height change of the shipped model (decision 145): +0.78 on 0-5 deg, +4.38 [+2.07, +6.85] on > 45 deg (left > 45: +6.80), DAC fail -3.30 pp [-5.51, -1.26];
but that is the shipped model getting inputs in the geometry it was trained on, and fine-tuning on the same W frames removes the need for it (P2 at 88.2 vs P0 at 1.40 m 81.6).
On the representation side the gap does concentrate in sharp turns **at equal view**: WA front-only encoder vs Cinque vision (Cf - V) lowers decoder DAC fail by
0.68 pp (0-5 deg, n.s.), 0.36 (5-20, n.s.), 4.48 [-8.15, -0.85] (20-45) and 5.28 [-9.05, -1.55] (> 45); adding WA's other views (Ca - Cf) changes nothing under the
lambda 10 objective (+0.59 [-1.48, +2.84] on > 45 deg) and a modest benefit appears only under the imitation-only objective (-2.59 [-4.32, -1.01] on > 45 deg).
So the turning gap is an encoder-quality deficit that grows with turn sharpness, not a field-of-view deficit; the "view / FOV" reading of the joint diagnosis is not supported by these contrasts.

Evidence: medium. Per bin n is 1 500-1 600 tokens for > 45 deg (about 600-900 per side), 4 contrasts x 11 strata, no multiplicity correction: a single CI that excludes 0 is descriptive.
The decoder contrasts rest on 371 scored tokens in the > 45 deg bin (weights back to 1 584 navtest-equivalent; F / R / FF tokens in full, the 1 500 random both-pass tokens reweighted).

## Per-bin summary (CI in the full tables below)

EPDMS difference [95 % CI] and DAC fail difference in pp, navtest, first-named minus second-named arm:

| bin (n) | P3 - P2: EPDMS / DAC pp | P3 masked - P3: EPDMS / DAC pp | V - F (pilot): EPDMS / DAC pp | P0 1.40 m - P0 W (zero-shot): EPDMS / DAC pp |
|:--|:--|:--|:--|:--|
| 0-5 deg (6 400) | -0.09 [-0.28, +0.08] / +0.02 | +0.09 [-0.07, +0.24] / -0.09 | +0.21 [-0.12, +0.57] / +0.01 | +0.78 [+0.24, +1.35] / -0.25 |
| 5-20 deg (2 592) | +0.01 [-0.44, +0.50] / +0.19 | +0.04 [-0.52, +0.57] / -0.25 | +0.09 [-0.63, +0.84] / -0.06 | +0.86 [-0.96, +2.83] / -1.81 [-3.56, -0.15] |
| 20-45 deg (1 637) | +0.49 [-0.24, +1.32] / -0.34 | -0.06 [-1.09, +0.99] / -0.61 | -0.72 [-2.01, +0.54] / +0.92 | -0.04 [-3.33, +3.31] / +0.06 |
| > 45 deg (1 517) | -0.52 [-1.72, +0.62] / +1.05 [-0.23, +2.34] | +0.59 [-0.46, +1.79] / -1.42 [-2.73, -0.26] | +1.07 [+0.02, +2.15] / -0.40 | +4.38 [+2.07, +6.85] / -3.30 [-5.51, -1.26] |
| all (12 146) | -0.05 [-0.31, +0.21] / +0.14 | +0.12 [-0.14, +0.38] / -0.36 [-0.64, -0.09] | +0.17 [-0.16, +0.50] / +0.07 | +1.13 [+0.22, +2.05] / -0.92 [-1.77, -0.06] |

Decoder DAC fail % on the navtest-weighted decoder sample (hinge lambda 10; each decoder sees [stage, ego]); contrasts in pp [95 % CI]:

| bin | ego-only | V (Cinque vision) | Cf (WA front) | Ca (WA all views) | Cf - V (representation, equal view) | Ca - Cf (extra views) |
|:--|--:|--:|--:|--:|:--|:--|
| 0-5 deg | 7.29 | 2.46 | 1.78 | 1.85 | -0.68 [-1.63, +0.18] | +0.07 [-0.68, +0.87] |
| 5-20 deg | 22.16 | 4.60 | 4.25 | 6.06 | -0.36 [-3.07, +2.10] | +1.82 [-0.81, +4.55] |
| 20-45 deg | 20.68 | 9.45 | 4.97 | 5.43 | -4.48 [-8.15, -0.85] | +0.46 [-1.47, +2.75] |
| > 45 deg | 22.53 | 12.31 | 7.03 | 7.62 | -5.28 [-9.05, -1.55] | +0.59 [-1.48, +2.84] |
| all | 14.21 | 5.12 | 3.41 | 3.97 | -1.71 [-2.85, -0.65] | +0.57 [-0.27, +1.42] |

Left / right: the sharp-turn representation gap is on both sides (Cf - V, > 45 deg: left -6.57 [-12.89, -1.35], right -3.36 [-8.51, +1.67]; 20-45 right -8.40 [-16.10, -1.52]);
the view contrasts show no consistent side (V - F right > 45: +2.32 [+0.13, +4.45], left +0.26). The imitation-only objective repeats the pattern (Cf - V > 45 deg -3.68 [-7.12, -0.35]; Ca - Cf -2.59).

## Does the plan leave the front / wide FOV? (proxy from the logged path, not the road)

Logged 4 s future poses (8 poses at 0.5 s, those > 3 m from the camera at x = 1.5 m ahead of the rear axle), bearing |atan2(y, x - 1.5)|. Cinque's wide camera is taken as about +-60 deg (decision 136: 120 deg front-wide); this is an assumption, the exact FOV of the warped W frames was not re-measured here.

| bin | poses outside +-30 deg | outside +-45 deg | outside +-60 deg | end pose outside +-30 / +-45 / +-60 deg |
|:--|--:|--:|--:|:--|
| 0-5 deg | 0.0 % | 0.0 % | 0.0 % | 0 / 0 / 0 % |
| 5-20 deg | 0.0 % | 0.0 % | 0.0 % | 0.0 / 0.0 / 0.0 % |
| 20-45 deg | 2.2 % | 0.0 % | 0.0 % | 5.8 / 0.0 / 0.0 % |
| > 45 deg | 29.0 % | 4.0 % | 0.2 % | 69.5 / 15.3 / 1.2 % |

For > 45 deg tokens the path itself stays inside a +-60 deg wide view (0.2 % of poses outside; 1.2 % of end poses), but 70 % of end poses sit beyond +-30 deg, i.e. outside a narrow / front-only view,
and 15 % beyond +-45 deg. The road the turn needs (exit lane, kerbs around the corner) lies wider than the path and was not computed from the map, so this bounds only how much of the *path* the front / wide view can see.
Limitation: the full-map computation (drivable polygon vs camera frustum) was skipped; the equal-view representation contrast (Cf - V) is the stronger evidence.

## Caveats

- V - F is pilot scale (3 000 x 64 steps, 17 k tokens) and V / F also differ in the warp road height (decision 145 note); P0 1.40 m - P0 W is a calibration change of an untrained-on-it model, not a view-content change.
- The decoder contrasts are on 2 152 tokens chosen by P2 / WA failure status plus 1 500 random both-pass tokens; the weights restore the navtest mix, the per-bin n is small (137-213 for left / right bins).
- Ca vs Cf differ in pooling (8 x 8 over four views vs 4 x 4 over one) as well as content; both are pooled to 32 x 512.
- The 1 -> 4 contrasts are on the same 12 146 tokens and logs; strata overlap with the manoeuvre strata of the joint page.

## Full tables

### 1 P3 - P2 (side + rear cameras added)

| contrast | group | stratum | n | logs | EPDMS A | EPDMS B | dEPDMS A-B | DACfail% A | DACfail% B | dDACfail pp A-B |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 P3 - P2 (side + rear cameras added) | all | all tokens | 12146 | 136 | 88.16 | 88.21 | -0.05 [-0.31, +0.21] | 4.47 | 4.33 | +0.14 [-0.11, +0.39] |
| 1 P3 - P2 (side + rear cameras added) | |heading change| | 0-5 deg | 6400 | 135 | 92.62 | 92.72 | -0.09 [-0.28, +0.08] | 1.67 | 1.65 | +0.02 [-0.09, +0.13] |
| 1 P3 - P2 (side + rear cameras added) | |heading change| | 5-20 deg | 2592 | 118 | 88.00 | 87.99 | +0.01 [-0.44, +0.50] | 4.42 | 4.22 | +0.19 [-0.20, +0.57] |
| 1 P3 - P2 (side + rear cameras added) | |heading change| | 20-45 deg | 1637 | 105 | 81.32 | 80.83 | +0.49 [-0.24, +1.32] | 8.19 | 8.52 | -0.34 [-1.22, +0.46] |
| 1 P3 - P2 (side + rear cameras added) | |heading change| | > 45 deg | 1517 | 101 | 77.00 | 77.52 | -0.52 [-1.72, +0.62] | 12.36 | 11.31 | +1.05 [-0.23, +2.34] |
| 1 P3 - P2 (side + rear cameras added) | left | 5-20 deg | 1446 | 108 | 87.74 | 87.48 | +0.26 [-0.36, +0.96] | 4.98 | 4.94 | +0.03 [-0.40, +0.47] |
| 1 P3 - P2 (side + rear cameras added) | right | 5-20 deg | 1146 | 99 | 88.33 | 88.64 | -0.31 [-0.96, +0.30] | 3.71 | 3.32 | +0.39 [-0.19, +1.05] |
| 1 P3 - P2 (side + rear cameras added) | left | 20-45 deg | 1057 | 85 | 83.20 | 82.69 | +0.52 [-0.52, +1.64] | 7.19 | 7.66 | -0.47 [-1.76, +0.73] |
| 1 P3 - P2 (side + rear cameras added) | right | 20-45 deg | 580 | 76 | 77.89 | 77.44 | +0.45 [-0.71, +1.50] | 10.00 | 10.09 | -0.09 [-1.21, +1.12] |
| 1 P3 - P2 (side + rear cameras added) | left | > 45 deg | 918 | 77 | 79.77 | 80.32 | -0.55 [-2.03, +0.86] | 10.02 | 8.88 | +1.14 [-0.28, +2.72] |
| 1 P3 - P2 (side + rear cameras added) | right | > 45 deg | 599 | 73 | 72.76 | 73.24 | -0.48 [-2.15, +1.42] | 15.94 | 15.03 | +0.92 [-1.43, +2.96] |

### 2 V - F (1.40 m virtual camera, pilot)

| contrast | group | stratum | n | logs | EPDMS A | EPDMS B | dEPDMS A-B | DACfail% A | DACfail% B | dDACfail pp A-B |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 V - F (1.40 m virtual camera, pilot) | all | all tokens | 12146 | 136 | 87.56 | 87.39 | +0.17 [-0.16, +0.50] | 4.80 | 4.74 | +0.07 [-0.24, +0.35] |
| 2 V - F (1.40 m virtual camera, pilot) | |heading change| | 0-5 deg | 6400 | 135 | 92.52 | 92.31 | +0.21 [-0.12, +0.57] | 1.66 | 1.66 | +0.01 [-0.21, +0.23] |
| 2 V - F (1.40 m virtual camera, pilot) | |heading change| | 5-20 deg | 2592 | 118 | 86.90 | 86.80 | +0.09 [-0.63, +0.84] | 5.17 | 5.23 | -0.06 [-0.79, +0.67] |
| 2 V - F (1.40 m virtual camera, pilot) | |heading change| | 20-45 deg | 1637 | 105 | 78.49 | 79.21 | -0.72 [-2.01, +0.54] | 10.75 | 9.84 | +0.92 [-0.37, +2.23] |
| 2 V - F (1.40 m virtual camera, pilot) | |heading change| | > 45 deg | 1517 | 101 | 77.58 | 76.50 | +1.07 [+0.02, +2.15] | 11.01 | 11.40 | -0.40 [-1.59, +0.78] |
| 2 V - F (1.40 m virtual camera, pilot) | left | 5-20 deg | 1446 | 108 | 86.71 | 86.37 | +0.33 [-0.69, +1.46] | 5.39 | 5.84 | -0.45 [-1.57, +0.56] |
| 2 V - F (1.40 m virtual camera, pilot) | right | 5-20 deg | 1146 | 99 | 87.13 | 87.34 | -0.21 [-1.37, +0.84] | 4.89 | 4.45 | +0.44 [-0.61, +1.65] |
| 2 V - F (1.40 m virtual camera, pilot) | left | 20-45 deg | 1057 | 85 | 80.73 | 81.26 | -0.54 [-2.09, +0.94] | 9.56 | 8.99 | +0.57 [-1.14, +2.21] |
| 2 V - F (1.40 m virtual camera, pilot) | right | 20-45 deg | 580 | 76 | 74.40 | 75.47 | -1.07 [-3.61, +1.18] | 12.93 | 11.38 | +1.55 [-0.66, +4.14] |
| 2 V - F (1.40 m virtual camera, pilot) | left | > 45 deg | 918 | 77 | 79.51 | 79.26 | +0.26 [-1.38, +1.86] | 9.53 | 9.15 | +0.38 [-1.22, +1.99] |
| 2 V - F (1.40 m virtual camera, pilot) | right | > 45 deg | 599 | 73 | 74.60 | 72.28 | +2.32 [+0.13, +4.45] | 13.27 | 14.86 | -1.59 [-4.07, +1.02] |

### 3 P0 at 1.40 m - P0 at W frames (zero-shot height)

| contrast | group | stratum | n | logs | EPDMS A | EPDMS B | dEPDMS A-B | DACfail% A | DACfail% B | dDACfail pp A-B |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | all | all tokens | 12146 | 136 | 81.64 | 80.51 | +1.13 [+0.22, +2.05] | 5.91 | 6.83 | -0.92 [-1.77, -0.06] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | |heading change| | 0-5 deg | 6400 | 135 | 87.76 | 86.98 | +0.78 [+0.24, +1.35] | 1.75 | 2.00 | -0.25 [-0.61, +0.05] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | |heading change| | 5-20 deg | 2592 | 118 | 81.60 | 80.74 | +0.86 [-0.96, +2.83] | 6.02 | 7.83 | -1.81 [-3.56, -0.15] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | |heading change| | 20-45 deg | 1637 | 105 | 70.25 | 70.29 | -0.04 [-3.33, +3.31] | 13.99 | 13.93 | +0.06 [-3.18, +3.22] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | |heading change| | > 45 deg | 1517 | 101 | 68.21 | 63.83 | +4.38 [+2.07, +6.85] | 14.57 | 17.86 | -3.30 [-5.51, -1.26] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | left | 5-20 deg | 1446 | 108 | 80.98 | 80.43 | +0.55 [-1.92, +3.46] | 6.36 | 7.40 | -1.04 [-3.12, +1.05] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | right | 5-20 deg | 1146 | 99 | 82.39 | 81.15 | +1.25 [-1.46, +4.51] | 5.58 | 8.38 | -2.79 [-6.02, -0.21] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | left | 20-45 deg | 1057 | 85 | 72.58 | 73.17 | -0.59 [-5.08, +4.29] | 12.20 | 11.83 | +0.38 [-4.38, +4.81] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | right | 20-45 deg | 580 | 76 | 66.00 | 65.04 | +0.96 [-2.16, +3.94] | 17.24 | 17.76 | -0.52 [-2.77, +1.96] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | left | > 45 deg | 918 | 77 | 70.43 | 63.63 | +6.80 [+4.22, +9.63] | 12.53 | 16.34 | -3.81 [-6.47, -1.49] |
| 3 P0 at 1.40 m - P0 at W frames (zero-shot height) | right | > 45 deg | 599 | 73 | 64.81 | 64.15 | +0.66 [-3.49, +4.69] | 17.70 | 20.20 | -2.50 [-6.71, +1.78] |

### 4 P3 side masked - P3 (test-time ablation)

| contrast | group | stratum | n | logs | EPDMS A | EPDMS B | dEPDMS A-B | DACfail% A | DACfail% B | dDACfail pp A-B |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 P3 side masked - P3 (test-time ablation) | all | all tokens | 12146 | 136 | 88.28 | 88.16 | +0.12 [-0.14, +0.38] | 4.11 | 4.47 | -0.36 [-0.64, -0.09] |
| 4 P3 side masked - P3 (test-time ablation) | |heading change| | 0-5 deg | 6400 | 135 | 92.71 | 92.62 | +0.09 [-0.07, +0.24] | 1.58 | 1.67 | -0.09 [-0.22, +0.03] |
| 4 P3 side masked - P3 (test-time ablation) | |heading change| | 5-20 deg | 2592 | 118 | 88.04 | 88.00 | +0.04 [-0.52, +0.57] | 4.17 | 4.42 | -0.25 [-0.69, +0.15] |
| 4 P3 side masked - P3 (test-time ablation) | |heading change| | 20-45 deg | 1637 | 105 | 81.26 | 81.32 | -0.06 [-1.09, +0.99] | 7.57 | 8.19 | -0.61 [-1.68, +0.45] |
| 4 P3 side masked - P3 (test-time ablation) | |heading change| | > 45 deg | 1517 | 101 | 77.59 | 77.00 | +0.59 [-0.46, +1.79] | 10.94 | 12.36 | -1.42 [-2.73, -0.26] |
| 4 P3 side masked - P3 (test-time ablation) | left | 5-20 deg | 1446 | 108 | 87.53 | 87.74 | -0.22 [-0.95, +0.53] | 4.81 | 4.98 | -0.17 [-0.76, +0.33] |
| 4 P3 side masked - P3 (test-time ablation) | right | 5-20 deg | 1146 | 99 | 88.70 | 88.33 | +0.37 [-0.37, +1.14] | 3.36 | 3.71 | -0.35 [-1.14, +0.32] |
| 4 P3 side masked - P3 (test-time ablation) | left | 20-45 deg | 1057 | 85 | 83.42 | 83.20 | +0.22 [-0.93, +1.53] | 6.48 | 7.19 | -0.71 [-2.15, +0.51] |
| 4 P3 side masked - P3 (test-time ablation) | right | 20-45 deg | 580 | 76 | 77.32 | 77.89 | -0.57 [-1.94, +0.95] | 9.57 | 10.00 | -0.43 [-1.88, +0.79] |
| 4 P3 side masked - P3 (test-time ablation) | left | > 45 deg | 918 | 77 | 80.45 | 79.77 | +0.68 [-0.85, +2.28] | 9.04 | 10.02 | -0.98 [-2.90, +0.77] |
| 4 P3 side masked - P3 (test-time ablation) | right | > 45 deg | 599 | 73 | 73.22 | 72.76 | +0.46 [-1.38, +2.39] | 13.86 | 15.94 | -2.09 [-4.09, -0.09] |

### decoders

| decoder objective | group | stratum | n scored | n navtest-equivalent | DACfail% ego-only | DACfail% V | DACfail% Cf | DACfail% Ca | dDACfail Cf-V pp | dDACfail Ca-Cf pp | score V | score Cf | score Ca | dscore Cf-V | dscore Ca-Cf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hinge10 | all | all tokens | 2152 | 12146 | 14.21 | 5.12 | 3.41 | 3.97 | -1.71 [-2.85, -0.65] | +0.57 [-0.27, +1.42] | 87.04 | 89.72 | 88.74 | +2.68 [+1.57, +3.87] | -0.98 [-2.01, +0.01] |
| hinge10 | |heading change| | 0-5 deg | 964 | 6374 | 7.29 | 2.46 | 1.78 | 1.85 | -0.68 [-1.63, +0.18] | +0.07 [-0.68, +0.87] | 89.89 | 91.51 | 91.05 | +1.61 [+0.10, +3.21] | -0.46 [-1.51, +0.61] |
| hinge10 | |heading change| | 5-20 deg | 467 | 2606 | 22.16 | 4.60 | 4.25 | 6.06 | -0.36 [-3.07, +2.10] | +1.82 [-0.81, +4.55] | 88.40 | 89.95 | 87.09 | +1.55 [-0.86, +4.06] | -2.86 [-5.58, -0.29] |
| hinge10 | |heading change| | 20-45 deg | 350 | 1583 | 20.68 | 9.45 | 4.97 | 5.43 | -4.48 [-8.15, -0.85] | +0.46 [-1.47, +2.75] | 81.56 | 87.02 | 86.63 | +5.46 [+1.47, +9.56] | -0.39 [-2.91, +1.93] |
| hinge10 | |heading change| | > 45 deg | 371 | 1584 | 22.53 | 12.31 | 7.03 | 7.62 | -5.28 [-9.05, -1.55] | +0.59 [-1.48, +2.84] | 78.77 | 84.85 | 84.30 | +6.07 [+1.91, +10.18] | -0.55 [-3.40, +1.95] |
| hinge10 | left | 5-20 deg | 285 | 1584 | 19.43 | 4.50 | 4.10 | 6.12 | -0.40 [-3.92, +2.78] | +2.02 [-1.32, +5.43] | 88.34 | 90.18 | 86.81 | +1.84 [-1.58, +5.41] | -3.37 [-6.82, -0.01] |
| hinge10 | right | 5-20 deg | 182 | 1021 | 26.39 | 4.76 | 4.47 | 5.97 | -0.29 [-4.06, +3.99] | +1.50 [-2.40, +6.04] | 88.48 | 89.58 | 87.52 | +1.11 [-2.70, +4.46] | -2.06 [-6.34, +1.34] |
| hinge10 | left | 20-45 deg | 213 | 1019 | 20.69 | 7.78 | 5.46 | 6.38 | -2.32 [-6.19, +1.35] | +0.91 [-2.08, +4.28] | 83.22 | 87.44 | 86.19 | +4.22 [-0.06, +8.77] | -1.24 [-4.93, +2.04] |
| hinge10 | right | 20-45 deg | 137 | 563 | 20.64 | 12.48 | 4.08 | 3.73 | -8.40 [-16.10, -1.52] | -0.35 [-1.63, +0.70] | 78.55 | 86.27 | 87.44 | +7.72 [+0.42, +15.64] | +1.16 [-0.15, +2.92] |
| hinge10 | left | > 45 deg | 196 | 949 | 25.00 | 11.84 | 5.27 | 5.86 | -6.57 [-12.89, -1.35] | +0.60 [-2.47, +3.69] | 80.96 | 87.45 | 86.38 | +6.50 [+0.73, +12.45] | -1.07 [-4.18, +2.05] |
| hinge10 | right | > 45 deg | 175 | 635 | 18.85 | 13.02 | 9.66 | 10.24 | -3.36 [-8.51, +1.67] | +0.58 [-1.56, +3.59] | 75.51 | 80.95 | 81.18 | +5.44 [-0.15, +10.68] | +0.23 [-4.63, +4.24] |
| imit | all | all tokens | 2152 | 12146 | 16.88 | 7.28 | 5.33 | 5.40 | -1.96 [-3.57, -0.38] | +0.07 [-0.99, +1.12] | 84.90 | 87.72 | 87.56 | +2.82 [+1.22, +4.51] | -0.16 [-1.21, +0.94] |
| imit | |heading change| | 0-5 deg | 964 | 6374 | 8.33 | 4.64 | 3.99 | 4.23 | -0.64 [-2.43, +1.16] | +0.24 [-1.23, +1.66] | 87.91 | 89.22 | 89.01 | +1.31 [-0.77, +3.41] | -0.21 [-1.76, +1.32] |
| imit | |heading change| | 5-20 deg | 467 | 2606 | 25.97 | 10.54 | 7.17 | 9.28 | -3.36 [-8.00, +0.89] | +2.11 [-0.46, +4.91] | 82.17 | 86.91 | 84.06 | +4.74 [+1.06, +8.81] | -2.84 [-5.76, -0.15] |
| imit | |heading change| | 20-45 deg | 350 | 1583 | 28.11 | 9.10 | 5.90 | 4.59 | -3.20 [-5.95, -0.19] | -1.31 [-3.77, +0.80] | 80.85 | 86.08 | 88.28 | +5.23 [+1.94, +8.41] | +2.19 [-0.61, +5.43] |
| imit | |heading change| | > 45 deg | 371 | 1584 | 25.14 | 10.77 | 7.09 | 4.50 | -3.68 [-7.12, -0.35] | -2.59 [-4.32, -1.01] | 81.37 | 84.69 | 86.79 | +3.32 [-0.63, +7.19] | +2.10 [-0.07, +4.10] |
| imit | left | 5-20 deg | 285 | 1584 | 23.13 | 9.17 | 7.15 | 10.62 | -2.02 [-7.37, +3.28] | +3.47 [-0.11, +7.55] | 83.02 | 86.88 | 82.94 | +3.86 [-1.12, +8.72] | -3.93 [-8.01, -0.23] |
| imit | right | 5-20 deg | 182 | 1021 | 30.37 | 12.66 | 7.21 | 7.21 | -5.45 [-12.89, +1.54] | +0.00 [-3.85, +3.35] | 80.85 | 86.95 | 85.80 | +6.10 [+0.14, +12.83] | -1.15 [-4.71, +2.71] |
| imit | left | 20-45 deg | 213 | 1019 | 25.14 | 8.44 | 5.66 | 4.68 | -2.78 [-7.23, +1.29] | -0.98 [-4.40, +2.14] | 81.97 | 87.15 | 89.48 | +5.19 [+1.10, +10.26] | +2.33 [-1.29, +6.47] |
| imit | right | 20-45 deg | 137 | 563 | 33.47 | 10.29 | 6.33 | 4.44 | -3.96 [-8.85, +1.65] | -1.89 [-5.33, +0.17] | 78.84 | 84.15 | 86.10 | +5.31 [-1.33, +11.21] | +1.95 [-2.30, +6.79] |
| imit | left | > 45 deg | 196 | 949 | 23.63 | 11.62 | 5.27 | 2.81 | -6.36 [-11.71, -1.53] | -2.46 [-5.08, -0.13] | 82.65 | 87.16 | 88.91 | +4.51 [-1.11, +10.27] | +1.74 [-0.80, +4.50] |
| imit | right | > 45 deg | 175 | 635 | 27.40 | 9.50 | 9.82 | 7.04 | +0.32 [-3.72, +4.88] | -2.78 [-4.76, -0.99] | 79.45 | 80.98 | 83.62 | +1.53 [-3.61, +6.27] | +2.64 [-1.10, +5.56] |

### fov

| stratum | n | tokens with any pose > 3 m | poses outside +-30 deg | end pose outside +-30 | poses outside +-45 deg | end pose outside +-45 | poses outside +-60 deg | end pose outside +-60 | poses outside +-75 deg | end pose outside +-75 |
|---|---|---|---|---|---|---|---|---|---|---|
| all tokens | 12146 | 92% | 4.1% | 10.3% | 0.5% | 2.1% | 0.0% | 0.2% | 0.0% | 0.0% |
| 0-5 deg | 6400 | 86% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 5-20 deg | 2592 | 99% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 20-45 deg | 1637 | 100% | 2.2% | 5.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| > 45 deg | 1517 | 100% | 29.0% | 69.5% | 4.0% | 15.3% | 0.2% | 1.2% | 0.0% | 0.0% |
