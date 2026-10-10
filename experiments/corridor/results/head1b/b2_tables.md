# HEAD1b step B2: heading-profile auxiliary loss on the pilot policy (tables)

**EXPLORATORY (HEAD1b: second attempt after the missed registered gate G3, user-authorised; not a registered read).** Plan: `plans/2026-10-10-head1-prereg.md`, amendment 2026-10-10 (B2) and its two B2 notes. Numbers: `b2_reads.json`. Code: `lib/heading_aux.py`, `experiments/op_parity/scripts/pp_train.py` (`--aux-lam`, `--holdout`), `experiments/corridor/scripts/head1_aux.py`, `head1_pilot_report.py` (the common reader).

Arm HA = the SH30 pilot recipe (P2, hinge 30 / 0.5 m, `navsim/op-parity-s234`, 3 000 steps x 64, warp) + a head (LN, Linear 512, GELU, Linear 22) on `select_4` (the state the plan head decodes from, downstream of the adapter) regressing the 22-point arc-length heading profile of the logged path (label L), Huber delta 0.1 rad on labelled points, imitation rows only. The head is dropped at inference. Tags: HA = `HAUX-F-s0/s1` (the tag `HA-F-s0` belongs to decision 158's agent-hinge pilot), baseline H0 = `GH0-F-s0/s1` (reused).

## 1. Identity smoke

`--aux-lam 0` (head built, trained on the detached state) against the run without any new flag, seed 0, 150 steps, dev eval every 50: **155 checkpoint tensors, 0 differ, max |diff| 0.0** (bit-identical: True); dev ADE 1.124589 = 1.124589. A third smoke (lambda 3 + `--holdout head1val`) exercised the hold-out, the gradient read and the dump.

## 2. Lambda selection (seed 0, trained without the validation logs, read on their rows)

| lambda | tag | rows (> 20 deg) | logs | plan 4 s heading RMS, > 20 deg (deg) | vs lambda 0 (paired) | > 45 deg | > 45 deg vs lambda 0 (paired) | ADE (m) | ADE / lambda 0 | head RMS at the plan's arc, > 45 / all (deg) | chosen |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 0.0 | HAsel-l0-s0 | 2566 (683) | 117 | 9.12 [7.89, 10.40] | +0.00 [+0.00, +0.00] | 10.95 [8.33, 13.44] | +0.00 [+0.00, +0.00] | 0.6008 [0.5749, 0.6271] | 1.0000 | 9.99 / 5.30 |  |
| 1.0 | HAsel-l1-s0 | 2566 (683) | 117 | 8.88 [7.77, 10.03] | -0.24 [-0.41, -0.08] | 10.50 [8.21, 12.72] | -0.45 [-0.75, -0.09] | 0.6000 [0.5746, 0.6256] | 0.9987 | 9.53 / 5.18 |  |
| 3.0 | HAsel-l3-s0 | 2566 (683) | 117 | 8.86 [7.74, 10.00] | -0.26 [-0.42, -0.10] | 10.56 [8.27, 12.78] | -0.39 [-0.70, -0.02] | 0.6007 [0.5750, 0.6264] | 0.9998 | 9.52 / 5.16 |  |
| 10.0 | HAsel-l10-s0 | 2566 (683) | 117 | 8.57 [7.62, 9.56] | -0.55 [-0.94, -0.17] | 10.11 [8.20, 11.96] | -0.84 [-1.59, +0.03] | 0.6012 [0.5754, 0.6273] | 1.0006 | 9.12 / 5.03 | yes |

chosen lambda 10.0; ADE within 2 %: [1.0, 3.0, 10.0]; below the lambda 0 reference: True

lambda 0 = the plain recipe on the same rows with a probe head on the detached state. The chosen lambda is the upper edge of the registered grid and the reads are monotone in lambda. Paired differences: log-cluster bootstrap on the same rows (117 logs).

**Size of the auxiliary gradient** (norm on the policy's parameters of lambda x auxiliary loss / of the rest of the loss, one batch at steps 1 000, 2 000, 3 000; before the joint clip at 1.0):

| run | lambda | step 1 000 | step 2 000 | step 3 000 |
|:--|:--|:--|:--|:--|
| HAsel-l1-s0 | 1 | 0.038 / 44.5 = 0.09 % | 0.041 / 13.9 = 0.29 % | 0.049 / 12.1 = 0.40 % |
| HAsel-l3-s0 | 3 | 0.113 / 44.3 = 0.25 % | 0.124 / 13.2 = 0.94 % | 0.152 / 11.3 = 1.34 % |
| HAsel-l10-s0 | 10 | 0.230 / 31.7 = 0.72 % | 0.373 / 13.6 = 2.74 % | 0.384 / 12.2 = 3.14 % |
| HAUX-F-s0 | 10 | 0.433 / 33.2 = 1.30 % | 0.243 / 20.2 = 1.20 % | 0.327 / 28.6 = 1.14 % |
| HAUX-F-s1 | 10 | 0.257 / 20.3 = 1.26 % | 0.356 / 12.2 = 2.92 % | 0.385 / 7.2 = 5.32 % |

Train loss at step 3 000 (mean of the last 25 steps): auxiliary Huber 0.0090 (lambda 0, probe) -> 0.0084 (lambda 10); imitation 0.768 -> 0.754.

## 3. Final arm HA (lambda 10, full `s234-train`, seeds 0 / 1) against H0 on navtest: the common reader

`b2`: navtest 12146 tokens, per-token seed means x 100; differences to H0 = GH0-F-s0, GH0-F-s1 (88.48; per seed 88.46 / 88.51) with 95% CI (log-cluster paired bootstrap, B 4000). Lines: direct.

| arm | tags | EPDMS - base | per seed | - shuffled | memory masked - on (seeds) | dev ADE on / masked / mismatched, per seed (m) | channel read | verdict |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| HA | HAUX-F-s0, HAUX-F-s1 | +0.04 [-0.04, +0.12] | +0.12 / -0.03 | - | - | 0.635 / - / - ; 0.631 / - / - | - | negative |

Baseline dev ADE (m): 0.632 ; 0.629

**EPDMS by logged heading change over 4 s (difference to the baseline).**

| arm / contrast | < 5 deg (straight) | 5-20 deg | 20-45 deg | > 45 deg | > 20 deg |
|:--|:--|:--|:--|:--|:--|
| HA - H0 | +0.01 [-0.06, +0.09] | +0.06 [-0.15, +0.28] | -0.06 [-0.35, +0.23] | +0.26 [-0.10, +0.70] | +0.09 [-0.15, +0.37] |

**Turn failures at > 45 deg (1 517 tokens; %, difference in pp), DAC failures, sub-scores (difference to the baseline).**

| arm | cut inside: arm / base, diff | cannot make the turn: arm / base, diff | DAC fail %, > 45 | DAC fail %, > 20 | NC | DAC | EP | TTC | NC + TTC fail % |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| HA | 3.99 / 4.12, -0.13 [-0.53, +0.18] | 4.19 / 4.19, +0.00 [-0.26, +0.25] | -0.13 [-0.66, +0.32] | -0.02 [-0.32, +0.25] | +0.03 [-0.02, +0.07] | +0.01 [-0.08, +0.10] | +0.02 [+0.01, +0.04] | +0.01 [-0.03, +0.05] | -0.02 [-0.06, +0.03] |

**4 s arc length of the plan and the plan's own 4 s heading error (RMS, deg; token x seed pooled; before = baseline, after = arm).**

| arm | arc ratio to base | arm / log, base / log | straight / > 45 deg ratio | heading error all: before, after, diff | > 20 deg | > 45 deg |
|:--|:--|:--|:--|:--|:--|:--|
| HA | 0.9998 [0.9997, 1.0000] | 0.9983, 0.9984 | 0.9997 / 0.9998 | 7.38 [6.24, 8.50], 6.94 [5.93, 7.94], -0.44 [-0.65, -0.24] | 12.90 [11.03, 14.74], 11.94 [10.32, 13.52], -0.96 [-1.36, -0.59] | 15.60 [13.31, 17.57], 14.39 [12.37, 16.15], -1.21 [-1.71, -0.73] |

**Guard lines (step B2).**

| arm | straight bucket CI not wholly below 0 | abs(arc ratio - 1) <= 0.01 |
|:--|:--|:--|
| HA | True | True |


Replays: `geo_s0` (GH0-F-s0), `geo_e2e` (GH0-F-s1), `h1b2` (HAUX-F-s0/s1: 886 (model, token) rows on 482 tokens, replay DAC = bench DAC on all).

## 4. The auxiliary head's own accuracy

Heading error RMS (deg) against the logged 4 s heading; head = its profile interpolated at an arc length. navtest, 12 146 tokens, log-cluster bootstrap. The HEAD1 stage-1 head (`L s0`: its own 3.4 M network on the frozen vision tokens, navtrain fold 0) is read at GH0-F's plan arc (results/head1/reads.json).

| bucket | n | HA plan (s0 ; s1) | HA head at its plan's 4 s arc (s0 ; s1) | HA head at the logged 4 s arc (s0 ; s1) | H0 plan (GH0-F, 2 seeds pooled) | stage-1 head L at GH0-F's arc | stage-1 head L at the logged arc | blind head |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| all | 12146 | 7.16 [6.06, 8.25] ; 6.71 [5.78, 7.61] | 7.11 [6.08, 8.16] ; 6.65 [5.81, 7.49] | 6.54 [5.56, 7.52] ; 6.05 [5.28, 6.80] | 7.38 [6.24, 8.50] | 5.27 [4.68, 5.82] | 4.11 [3.57, 4.60] | 10.55 [9.65, 11.39] |
| >20 | 3154 | 12.40 [10.62, 14.14] ; 11.47 [9.99, 12.89] | 12.10 [10.40, 13.77] ; 11.02 [9.70, 12.31] | 11.19 [9.54, 12.81] ; 10.19 [8.95, 11.36] | 12.90 [11.03, 14.74] | 8.41 [7.58, 9.20] | 6.68 [5.76, 7.48] | 15.58 [14.47, 16.63] |
| >45 | 1517 | 14.94 [12.73, 16.88] ; 13.81 [11.99, 15.41] | 14.48 [12.31, 16.36] ; 13.26 [11.56, 14.73] | 13.63 [11.51, 15.48] ; 12.51 [10.83, 13.94] | 15.60 [13.31, 17.57] | 9.02 [8.01, 9.98] | 7.49 [6.37, 8.47] | 15.72 [14.35, 17.11] |

Held-out validation logs of the selection runs (2 566 rows, 117 logs; lambda 0 = probe on the plain policy's detached state):

| lambda | bucket | n | plan | head at the plan's 4 s arc | head at the logged 4 s arc | head per-point RMS 0-40 m |
|:--|:--|:--|:--|:--|:--|:--|
| 0.0 | all | 2566 | 5.30 [4.64, 6.02] | 5.30 [4.65, 5.99] | 4.91 [4.20, 5.71] | 10.38 |
| 0.0 | >20 | 683 | 9.12 [7.89, 10.40] | 8.53 [7.33, 9.80] | 8.23 [6.88, 9.67] | 13.91 |
| 0.0 | >45 | 242 | 10.95 [8.33, 13.44] | 9.99 [7.53, 12.33] | 9.86 [7.09, 12.58] | 13.45 |
| 1.0 | all | 2566 | 5.17 [4.55, 5.83] | 5.18 [4.59, 5.82] | 4.73 [4.11, 5.41] | 10.02 |
| 1.0 | >20 | 683 | 8.88 [7.77, 10.03] | 8.29 [7.19, 9.42] | 7.87 [6.75, 9.04] | 13.32 |
| 1.0 | >45 | 242 | 10.50 [8.21, 12.72] | 9.53 [7.43, 11.55] | 9.10 [6.98, 11.20] | 12.77 |
| 3.0 | all | 2566 | 5.16 [4.55, 5.82] | 5.16 [4.57, 5.79] | 4.71 [4.11, 5.37] | 9.86 |
| 3.0 | >20 | 683 | 8.86 [7.74, 10.00] | 8.28 [7.18, 9.40] | 7.84 [6.75, 8.98] | 13.09 |
| 3.0 | >45 | 242 | 10.56 [8.27, 12.78] | 9.52 [7.45, 11.52] | 9.01 [6.96, 11.01] | 12.66 |
| 10.0 | all | 2566 | 5.03 [4.48, 5.62] | 5.03 [4.50, 5.60] | 4.53 [4.01, 5.08] | 9.26 |
| 10.0 | >20 | 683 | 8.57 [7.62, 9.56] | 8.03 [7.10, 8.99] | 7.46 [6.62, 8.34] | 12.22 |
| 10.0 | >45 | 242 | 10.11 [8.20, 11.96] | 9.12 [7.51, 10.70] | 8.22 [6.91, 9.59] | 11.99 |

The navtest pass that produced the head's profiles reproduces bench's plans (max |dxy| 0.025 m, both seeds).

## 5. Cost

About 0.83 card-hours of summed GPU job wall time: 3 smokes 3 min, 4 selection runs 26 min, 2 final runs 15 min, 2 cancelled starts 2 min, bench plans 2 min, the head's navtest pass 2 min. Peak VRAM 8.1 GB per training.
