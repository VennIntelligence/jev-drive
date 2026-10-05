# Full run: with WA-JEPA's inputs and navtrain fine-tuning, Cinque reaches navtest EPDMS 88.2 (WA-JEPA 91.7) and HUGSIM HD 0.396 (WA-JEPA 0.451)

Written 2026-10-06. Plan and protocol choice: [../plans/2026-10-06-parity-prereg.md](../plans/2026-10-06-parity-prereg.md), [stageB.md](stageB.md)
(protocol W, approved with the W-frame anchor teacher). Chain: `scripts/pp_full_chain.sh` (staged: 1 cache shard -> 11 -> 1 training -> 5 ->
readouts, sanity gates `scripts/pp_full_check.py`). NAVSIM tables: [full_navtest_arms.md](full_navtest_arms.md), [full_navtest_paired.md](full_navtest_paired.md)
(`scripts/pp_full_report.py`); HUGSIM: [hugsim_full.md](hugsim_full.md) and [hugsim_full/](hugsim_full/).

## Setup

- Data: all 103 288 navtrain tokens with a logged future (`navsim/op-parity-full-{train,dev}`, dev = 2 % of logs, log-disjoint), 12 shards
  rendered on the fly (4 CAM_F0 keys + 6 CPU-warped lattice frames, side / rear cameras), Cinque's frozen encoder -> cached tokens
  (front 27 GB, side 30 GB). Anchor teacher = shipped Cinque on the same W frames. Shard gate: token RMS within 1 % of the Stage B cache,
  teacher ADE vs log 1.59 m (Stage A W8: 1.57), speed ratio 0.99 on every shard.
- Training: plan pathway + adapter, batch 128, 10 000 steps (12 epochs), warmup 300, otherwise the pilot recipe; arms P1 / P2 / P3 x seeds
  0 / 1. Token rows gathered per batch from the page-cached memmaps (bit-equivalent to the GPU-resident store: 50-step losses equal to 1e-4);
  21-62 min per run (2.8-6.7 it/s, CPU-shared). Dev ADE vs log (8 poses): P1 1.44, P2 0.54, P3 0.55 / 0.56 m; drift to shipped with inputs
  off 0.04 (P2 / P3), 0.16 (P1).
- NAVSIM: navtest 12 146 tokens, W frames, devkit navsim main @ 0a380a9 v2 EPDMS, WA-JEPA's stored per-token CSV from the same devkit / cache.
- HUGSIM: 64 scenarios (`derot_all64.txt`), one run each, presets `exam` / tree `fixed` (the wajepa_ref harness) and `spec` / tree opctrl.

## NAVSIM navtest (seed means; per-token paired bootstrap over 136 logs)

| arm | EPDMS | s0 / s1 | NC | DAC | EP | TTC | EC | ADE vs log (m) | speed ratio |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| P0 shipped (W frames) | 80.51 | | 96.89 | 93.17 | 82.72 | 96.86 | 69.97 | 1.57 | 0.98 |
| P1 fine-tune, inputs zeroed | 81.98 | 81.97 / 81.99 | 97.22 | 93.95 | 83.03 | 97.10 | 73.48 | 1.44 | 0.98 |
| P2 + ego / pose history / command | **88.21** | 88.12 / 88.30 | 98.55 | 95.67 | 87.15 | 97.92 | 88.54 | 0.56 | 1.00 |
| P3 + side / rear cameras | 88.16 | 88.23 / 88.09 | 98.60 | 95.53 | 87.26 | 97.95 | 88.57 | 0.56 | 1.00 |
| P3, side cameras masked at test | 88.28 | 88.34 / 88.22 | 98.59 | 95.89 | 86.85 | 97.85 | 88.70 | 0.57 | 1.00 |
| WA-JEPA (released ckpt) | 91.71 | | 99.40 | 98.20 | 87.87 | 98.89 | 88.05 | | |

| contrast | EPDMS [95% CI] | dEP | dDAC | dNC | dTTC |
|:--|:--|--:|--:|--:|--:|
| P1 - P0 (fine-tuning alone) | +1.47 [+1.24, +1.72] | +0.31 | +0.78 | +0.33 | +0.23 |
| **P3 - P1 (full parity, the pre-registered main contrast)** | **+6.18 [+5.40, +7.00]** | +4.23 | +1.58 | +1.38 | +0.85 |
| P2 - P1 | +6.23 [+5.43, +7.10] | +4.12 | +1.72 | +1.33 | +0.82 |
| P3 - P2 | -0.05 [-0.31, +0.20] | | | | |
| P3 - P3 side masked | -0.12 [-0.39, +0.14] | | | | |
| **P3 - WA-JEPA** | **-3.55 [-4.27, -2.84]** | -0.60 | -2.67 | -0.80 | -0.94 |
| P2 - WA-JEPA | -3.50 [-4.24, -2.76] | -0.72 | -2.53 | -0.86 | -0.97 |
| P0 - WA-JEPA | -11.20 [-12.47, -10.02] | | | | |

## HUGSIM 64 (HD; seed means for P1-P3; details and the collision breakdown in hugsim_full.md)

| arm | exam HD | spec HD | exam spin / stuck / launch stall | spec spin / stuck / launch stall |
|:--|:--|:--|:--|:--|
| WA-JEPA | 0.451 [0.354, 0.548] | (same run) | 4 / 1 / 2 | |
| P0 | 0.263 | 0.294 | 9 / 16 / 9 | 0 / 24 / 18 |
| P1 | 0.287 | 0.325 | 17 / 4 / 6.5 | 0 / 15.5 / 16 |
| P2 | **0.396** | **0.393** | 9.5 / 0.5 / 1 | 1.5 / 0 / 1 |
| P3 | 0.362 | 0.386 | 9.5 / 0 / 1 | 1.5 / 0.5 / 1.5 |

Paired HD (unit = scenario): P2 - P1 exam +0.108 [+0.040, +0.180], spec +0.069 [-0.004, +0.147]; P3 - P1 +0.075 [+0.008, +0.146] /
+0.061 [-0.007, +0.134]; P2 - WA-JEPA -0.055 [-0.135, +0.025] / -0.057 [-0.134, +0.019]; P3 - WA-JEPA -0.088 [-0.176, +0.000] / -0.065
[-0.139, +0.006]; P1 - WA-JEPA -0.163 / -0.126 (CIs exclude 0).

Foreground collisions (the requested breakdown): every arm and WA-JEPA have 24-30 fg collisions; P2 28 (exam) / 29 (spec) vs P0 26 / 24.
In exam, 17-18 of P2's are scenarios where P0 also collides (steps 12-22, hard / extreme), only 9-10 where P0 is stopped or stuck; in spec
21 of 29 are in "P0 stopped" scenarios, but in 13 of those P0 itself ended in an fg collision at near-zero speed, and only 7-8 are scenarios
P0 sat out to the step cap. On P0's stuck scenarios (19 exam / 31 spec) P2 leaves none stuck, completes 5-10 and collides in the rest; HD there
0.24 -> 0.45-0.57 (exam), 0.22 -> 0.41-0.44 (spec). So the extra collisions are the cost of moving where P0 did not move, but they are a
minority of P2's collisions, and P2's total is close to P0's and WA-JEPA's.

## Answer to the question

- **Inputs, not pretraining, were most of the gap.** With equal inputs and the same fine-tuning data, Cinque's navtest gap to WA-JEPA shrinks
  from 11.2 to 3.5 EPDMS (69 % closed); fine-tuning alone closes 1.5 of it, the ego / pose / command inputs 6.2. In HUGSIM the gap shrinks
  from 0.19 to 0.06 HD (exam; 0.16 -> 0.06 spec), and P2 / P3 are no longer separable from WA-JEPA at 64 scenarios (CIs include 0).
- **What remains on NAVSIM is drivable-area and safety, not progress:** P3 - WA-JEPA DAC -2.7, TTC -0.9, NC -0.8, EP -0.6, extended comfort
  +0.5. [I] Cinque reads the world through a frozen front encoder trained on comma's rig (decision 104: the NAVSIM camera height shrinks its
  world to ~0.7); WA-JEPA's encoder was fine-tuned on navtrain. That is the remaining "trade" candidate, not tested here.
- **Side and rear cameras add nothing**, on navtest (P3 - P2 -0.05, masking them -0.12 both with CIs around 0) and in HUGSIM (P3 <= P2),
  consistent with decisions 135 / 136. Parity on cameras is satisfied but carries no information Cinque uses through its own encoder.
- **Closed loop:** the ego inputs end the stuck and launch-stall failures under both presets (P0 16 / 24 stuck -> P2 0.5 / 0) and keep spins at
  P0's level (exam 9.5 vs 9; spec 1.5 vs 0); the launch spin of decisions 92 / 111 is not removed under `exam` (pilot: onset unchanged).
  NC and TTC drop (spec NC 0.81 -> 0.68) because the car now moves.

## Caveats

- One fine-tuning recipe (plan pathway + adapter, vision frozen, 10 000 steps); WA-JEPA trains end to end. Two seeds; seed spread is small on
  NAVSIM (<= 0.18) and large in HUGSIM (P3 exam 0.397 vs 0.327), so P2 vs P3 in HUGSIM is not resolved.
- Protocol W (CPU-warped lattice frames, no learned interpolator); Stage B put the G (GIMM) protocol 0.90 above W on the pilot, so the
  W-protocol number may understate Cinque by ~1 EPDMS.
- HUGSIM: one run per scenario (rerun spread ~0.2 HD per scenario), WA-JEPA is a single run used for both presets, and it had GT route and
  ego state like these arms. The fg "P0 stopped" class depends on its definition (hugsim_full.md gives both).
- HUGSIM ego speed is fed on the dilated model clock (x1.25, results/hugsim_harness.md); training zeroes the oldest context slot, HUGSIM fills it.
