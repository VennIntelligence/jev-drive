# Vision unfreeze pilot: no variant passes the gate. Letting the front encoder adapt (stage 4, LoRA, or all of it) adds 0.00 to +0.11 EPDMS, and the 1.40 m virtual camera +0.17 (n.s.)

Written 2026-10-06. Pre-registration: [../plans/2026-10-06-unfreeze-prereg.md](../plans/2026-10-06-unfreeze-prereg.md), written before any
scored run. Code: `scripts/pp_unfreeze.py` (caches, trainer, pixel-path plans), chain `scripts/pp_unfreeze_chain.sh`, report
`scripts/pp_unfreeze_report.py` -> [unfreeze_pilot_arms.md](unfreeze_pilot_arms.md), [unfreeze_pilot_paired.md](unfreeze_pilot_paired.md),
[unfreeze_pilot_gate.json](unfreeze_pilot_gate.json).

Setup: the full run's P2 recipe (protocol W, ego / pose / command adapter, 0.25 anchor rows distilled to shipped on the same frames) on
navtrain shards `navtrain_full.s0of12` + `s1of12` (17 216 tokens), 3 000 steps x batch 64, seeds 0 / 1. Only the vision part differs: F frozen;
U1 stage 4 + head trainable (102 M, lr 1e-5) on cached `conv2d_36` features; U1L LoRA r16 on the six stage-4 MLP matmuls (lr 3e-4);
U2 the whole encoder (349 M, lr 5e-6) on pixels rendered online, gradient through the t0 pair; V frozen on a 1.40 m virtual camera.
Readout: all 12 146 navtest tokens through one pixel path (render -> each model's own encoder -> policy), v2 EPDMS on the devkit that
reproduces WA-JEPA's 91.71; per-token paired bootstrap over 136 logs, seed means.

## Numbers

| arm | EPDMS | s0 / s1 | NC | DAC | EP | TTC | dev ADE (m) | dev drift_off (m) |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| F (frozen, pilot scale) | 87.39 | 87.40 / 87.39 | 98.39 | 95.26 | 87.02 | 97.64 | 0.60 | 0.05 |
| U1 stage 4 + head | 87.40 | 87.44 / 87.37 | 98.39 | 95.25 | 87.00 | 97.65 | 0.60 | 0.05 |
| U1L LoRA r16, stage 4 | 87.40 | 87.40 / 87.39 | 98.37 | 95.26 | 87.03 | 97.65 | 0.60 | 0.05 |
| U2 whole encoder | 87.51 | 87.53 / 87.49 | 98.42 | 95.35 | 86.99 | 97.70 | 0.60 | 0.05 |
| V 1.40 m virtual camera (frozen) | 87.56 | 87.57 / 87.56 | 98.49 | 95.20 | 87.05 | 97.88 | 0.59 | 0.05 |
| P2 full run (frozen, 103 k tokens, 10 k x 128) | 88.21 | 88.12 / 88.30 | 98.55 | 95.67 | 87.15 | 97.92 | | |
| WA-JEPA | 91.71 | | 99.40 | 98.20 | 87.87 | 98.89 | | |
| P0 shipped, W frames / 1.40 m frames | 80.51 / 81.64 | | | 93.17 / 94.09 | 82.72 / 80.63 | | | |

| contrast | EPDMS [95% CI] | dEP | dDAC | dNC | dTTC |
|:--|:--|--:|--:|--:|--:|
| U1 - F | +0.01 [-0.06, +0.08] | -0.02 | -0.01 | 0.00 | +0.01 |
| U1L - F | +0.00 [-0.10, +0.11] | +0.01 | 0.00 | -0.02 | +0.01 |
| U2 - F | +0.11 [+0.01, +0.22] | -0.03 | +0.09 | +0.03 | +0.06 |
| V - F | +0.17 [-0.14, +0.48] | +0.02 | -0.07 | +0.10 | +0.24 |
| F - WA-JEPA | -4.32 [-5.14, -3.49] | -0.85 | -2.94 | -1.01 | -1.25 |
| P2 full - F (data and steps) | +0.81 [+0.50, +1.13] | +0.13 | +0.41 | +0.15 | +0.28 |

Every plan speed ratio is 1.00 (guard 1.00 +- 5 %), no EP / DAC trade, dev drift_off 0.047-0.054 m (guard <= 0.10 m).

How far the weights moved (seed 0, relative Frobenius change of the trained vision tensors): U1 stage 4 0.3 %; U1L merged stage-4 weights
8.5 % (max 9.9 %); U2 stage 1-3 0.07 %, stage 4 + head 0.11 %.

Equivalence checks (before scoring): frozen stage 4 on the `conv2d_36` cache reproduces the W token cache (mean |d| 4e-4, max 0.03 on RMS
1.7); the pixel path reproduces the cached-token plans of P0 and the full-run P2-F-s0 (mean 0.0014 / 0.0054 m).

## Pre-registered verdicts

| arm | U - F >= +0.5 | speed | no trade | drift_off | gate |
|:--|:--|:--|:--|:--|:--|
| U1 | +0.01, no | pass | pass | pass | **fails** |
| U1L | +0.00, no | pass | pass | pass | **fails** |
| U2 | +0.11, no | pass | pass | pass | **fails** |
| V | +0.17, no | pass | pass | pass | **fails** |

No U arm passes, so by the pre-registration the line stops here: no full run, no navhard / HUGSIM readout of U arms. V fails too, so no
V + U combination.

## What this says

- **The frozen front encoder is not where the remaining 3.5-4.3 points sit, at least not reachable by fine-tuning on navtrain with the
  anchor in place.** Stage-4 capacity is not the limit: U1L moved stage 4 by ~9 % and changed nothing (+0.00), U1 by 0.3 % (+0.01). The
  whole encoder at lr 5e-6 gives +0.11 (CI just above 0), a twentieth of the 3.55 gap.
- **The camera-height fix helps the shipped model, not the trained one.** Zero-shot, shipped Cinque gains 1.13 EPDMS on the 1.40 m camera
  (80.51 -> 81.64, DAC +0.9, EP -2.1). After P2 training on the same frames the gain is +0.17 (n.s.): the trained plan pathway and ego
  inputs already absorb the scale error of decision 104. The residual gap to WA-JEPA keeps the same profile in every arm (DAC -2.9,
  TTC -1.0 to -1.25, NC -1.0, EP -0.85).
- **Data and steps matter more than any vision variant:** the full run (6x tokens, 6.7x samples) is +0.81 over the pilot-scale F. The
  pilot-scale gap to WA-JEPA is 4.32, the full run's 3.50.

## Caveats

- Pilot scale (17 k tokens, 3 000 steps). A vision effect that only shows with full data cannot be excluded, but the pre-registered
  gate was set for this scale and no arm is close (best U +0.11 against a +0.5 line).
- U2 moved the weights little (0.1 %) at lr 5e-6 with the anchor; a higher-lr full fine-tune is the one untested corner. U1L shows that
  large changes to stage 4 alone do not help.
- The anchor rows (inputs off, distilled to shipped on navtrain frames) constrain the encoder in every U arm by design (decision 137);
  dropping them was not tested. drift_off is a navtrain-frame proxy; native-camera (comma) forgetting was not measured.
- V's warp uses the correct road-plane height (1.40 m), W the op_lb convention (CAM_F0 height above the rear-axle origin, 1.53 m), so
  V - F includes that difference.

## Cost

Caches: `conv2d_36` 8.5 GB per shard (56 tokens/s, 3 min, 22 cores), V 2.2 GB per shard. Training at batch 64 on one RTX 6000D:
F 8 it/s, U1 / U1L 5.5 it/s, U2 0.7 it/s (CPU rendering bound, 26 cores, 35 GB). Pilot total about 4 GPU·h including navtest plans.
For a full run (if ever wanted): U1 needs the 103 k-token `conv2d_36` cache (105 GB) and about 1 h per seed; U2 about 7 h per seed and
26 cores, over the 6 GPU·h line.
