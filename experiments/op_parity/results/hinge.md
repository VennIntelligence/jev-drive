# Hinge lane: a footprint drivable-area SDF hinge in the P2 fine-tune cuts navtest DAC failures by 0.50 pp and adds +0.46 EPDMS; 55% of the predicted DAC effect, most of the gap to WA-JEPA stays

Written 2026-10-06. Pre-registration [../plans/2026-10-06-hinge-prereg.md](../plans/2026-10-06-hinge-prereg.md). Question from decision 147: does adding the
footprint SDF hinge to P2's own fine-tune lower navtest DAC failures by about 0.9 pp without losing EPDMS. Chain `scripts/pp_hinge_chain.sh`, reports
`scripts/pp_hinge_report.py` (navtest, navhard, gate), `scripts/pp_hinge_strata.py` (strata), HUGSIM via `scripts/pp_hugsim_report.py full` (FULL_OUT
`results/hugsim_hinge/`). Hinge code `lib/drivable_hinge.py` (geometry checked identical to op_probe's `corners_torch` / `sdf_at`), option `--hinge-lam` in
`pp_train.py`; labels `opb_labels.py` over all 103 288 navtrain tokens (32 s on 60 cores, all ok).

## Setup

P2 = decision 144 recipe unchanged (navtrain 103 288 tokens, W protocol, frozen vision, plan pathway + adapter, batch 128, 10 000 steps, anchor rows 0.25, same
row stream per seed), plus `10 * mean(relu(0.3 - sdf))` over the 4 footprint corners of the 8 plan poses interpolated to 0.1 s, on imitation rows only. 2 seeds
(P2H10-F-s0 / s1), 24-35 min each. Only lambda = 10 was run: the small read passed, so the lambda = 3 fallback was not needed. The hinge loss is small at the end
of training (0.003 x 10 against imitation 0.6). References: P2-F-s0 / s1 and WA-JEPA, same devkit (navsim main @ 0a380a9 v2), metric cache and tokens.

## Small read (gate), one seed

P2H10-s0 vs P2-F-s0 on all navtest: EPDMS +0.46 [+0.27, +0.66], DAC failures 3.86% vs 4.42% (-0.56 pp [-0.75, -0.38]). Gate (stop if EPDMS < -0.3 or DAC drop
< 0.3 pp) passed; the lane continued.

## navtest (12 146 tokens, W frames, seed means; paired cluster bootstrap over 136 logs)

| arm | EPDMS | s0 / s1 | NC | DAC | EP | TTC | EC | DAC fail % | ADE vs log (m) | speed ratio |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| P2 | 88.21 | 88.12 / 88.30 | 98.55 | 95.67 | 87.15 | 97.92 | 88.54 | 4.33 | 0.56 | 1.00 |
| P2 + hinge (lambda 10) | **88.67** | 88.58 / 88.77 | 98.58 | 96.17 | 87.16 | 97.96 | 88.62 | **3.83** | 0.56 | 1.00 |
| WA-JEPA | 91.71 | | 99.40 | 98.20 | 87.87 | 98.89 | 88.05 | 1.80 | | |

| contrast | EPDMS [95% CI] | DAC fail pp [95% CI] | dNC | dTTC | dEP |
|:--|:--|:--|--:|--:|--:|
| **hinge - P2** | **+0.46 [+0.33, +0.61]** | **-0.50 [-0.64, -0.36]** | +0.03 | +0.05 | +0.01 |
| hinge - P2, seed 0 / seed 1 | +0.46 [+0.27, +0.66] / +0.47 [+0.34, +0.60] | -0.56 [-0.75, -0.38] / -0.44 [-0.57, -0.32] | | | |
| hinge - WA-JEPA | -3.04 [-3.74, -2.32] | +2.03 [+1.37, +2.76] | -0.83 | -0.93 | -0.70 |
| P2 - WA-JEPA | -3.50 [-4.24, -2.76] | +2.53 [+1.84, +3.29] | -0.86 | -0.97 | -0.72 |

(The paired bootstrap CIs differ slightly between `hinge_navtest_paired.md`, B 10 000, and the strata table, B 2 000; both are cluster bootstraps.)
Tables: [hinge_navtest_arms.md](hinge_navtest_arms.md), [hinge_navtest_paired.md](hinge_navtest_paired.md).

The hinge buys 0.50 of the 2.53 pp DAC gap to WA-JEPA (20%) and 0.46 of the 3.50 EPDMS gap (13%); nothing else moves (EP, TTC, NC, ADE, speed unchanged).
Against the decision-147 prediction (-0.9 pp, thin head on frozen P2 hidden) the realized effect is 55% of it, CI [-0.64, -0.36] excludes -0.9.

## Stratified (same binning as `op_probe/results/joint`; [hinge_strata.md](hinge_strata.md))

navtest by logged manoeuvre (EPDMS x 100; DAC failure %; contrasts with 95% cluster CI):

| stratum | n | EPDMS P2 / hinge / WA | hinge - P2 | hinge - WA | DAC fail % P2 / hinge / WA | dDACfail hinge - P2 (pp) | hinge - WA (pp) |
|:--|--:|:--|:--|:--|:--|:--|:--|
| straight | 5267 | 92.30 / 92.64 / 93.67 | +0.35 [+0.18, +0.54] | -1.03 [-1.88, -0.24] | 1.97 / 1.62 / 1.14 | -0.35 [-0.54, -0.19] | +0.48 [-0.24, +1.26] |
| curve 8-20 deg | 1522 | 87.55 / 87.72 / 89.92 | +0.17 [-0.27, +0.52] | -2.20 [-3.81, -0.78] | 4.24 / 3.94 / 2.37 | -0.30 [-0.60, +0.03] | +1.58 [+0.06, +3.26] |
| left turn > 20 deg | 1737 | 80.47 / 81.30 / 88.68 | +0.82 [+0.43, +1.25] | -7.38 [-9.62, -5.07] | 8.89 / 7.97 / 2.01 | -0.92 [-1.37, -0.53] | +5.96 [+3.90, +7.93] |
| right turn > 20 deg | 1010 | 73.40 / 74.99 / 84.17 | +1.59 [+0.91, +2.39] | -9.18 [-11.60, -6.68] | 14.06 / 12.48 / 5.74 | -1.58 [-2.30, -0.94] | +6.73 [+3.79, +9.43] |

by |heading change| over the logged 4 s:

| bin | n | EPDMS P2 / hinge / WA | hinge - P2 | hinge - WA | DAC fail % P2 / hinge / WA | hinge - P2 (pp) | hinge - WA (pp) |
|:--|--:|:--|:--|:--|:--|:--|:--|
| 0-5 deg | 6400 | 92.72 / 92.94 / 94.05 | +0.22 [+0.08, +0.39] | -1.11 [-1.82, -0.47] | 1.65 / 1.41 / 1.05 | -0.24 [-0.39, -0.12] | +0.36 [-0.21, +0.96] |
| 5-20 deg | 2592 | 87.99 / 88.34 / 90.65 | +0.35 [+0.06, +0.68] | -2.31 [-3.61, -1.16] | 4.22 / 3.76 / 2.16 | -0.46 [-0.78, -0.20] | +1.60 [+0.38, +3.07] |
| 20-45 deg | 1637 | 80.83 / 81.83 / 88.23 | +1.00 [+0.53, +1.50] | -6.40 [-8.25, -4.62] | 8.52 / 7.51 / 2.87 | -1.01 [-1.43, -0.60] | +4.64 [+2.75, +6.66] |
| > 45 deg | 1517 | 77.52 / 78.61 / 87.41 | +1.09 [+0.56, +1.60] | -8.80 [-10.72, -6.66] | 11.31 / 10.18 / 3.23 | -1.12 [-1.66, -0.49] | +6.95 [+5.07, +8.86] |

The effect scales with turning: DAC failures fall 0.24 pp at < 5 deg, 1.0-1.1 pp above 20 deg, and 1.6 pp on right turns, so the gain is where the gap is.
But the gap there is large and mostly stays: on turns > 20 deg the hinge arm still fails DAC 7.5-10% against WA-JEPA's 2.9-3.2%, and trails by 6.4-8.8 EPDMS.
On straights the hinge arm is within noise of WA-JEPA on DAC (+0.48 pp [-0.24, +1.26]). The turning deficit identified in the joint diagnosis (P2 - WA: straight
-1.4, left -8.2, right -10.8) becomes straight -1.0, left -7.4, right -9.2 with the hinge.

## navhard two-stage, GIMM frames (protocol G; W-trained checkpoints fed GIMM frames, like P2-G in navhard.md)

| arm | combined | stage 1 | stage 2 | s0 / s1 |
|:--|--:|--:|--:|:--|
| P2 (G) | 30.34 | 73.13 | 42.44 | 30.26 / 30.43 |
| P2 + hinge (G) | **31.84** | 74.73 | 43.04 | 31.40 / 32.28 |
| P0 shipped (G) | 33.54 | 71.79 | 47.08 | |
| WA-JEPA | 35.41 | 81.90 | 43.53 | |

hinge - P2: combined +1.49 [+0.65, +2.39], stage 1 +1.60 [+0.48, +2.82], stage 2 +0.61 [-0.28, +1.56]. hinge - WA-JEPA: -3.57 [-7.57, +0.21] (P2 - WA -5.07
[-9.03, -1.22]); stage 1 -7.17 [-12.27, -2.54]. The hinge arm's combined score is not separable from WA-JEPA's any more, and stage 2 is no longer behind it; it is still
below the shipped model under G (-1.71 [-5.21, +1.87]) and 4.0 below it in stage 2. Tables: [hinge_navhard_arms.md](hinge_navhard_arms.md), [hinge_navhard_paired.md](hinge_navhard_paired.md).

## HUGSIM 64 (HD, one run per scenario, seed means; [hugsim_hinge/tables.md](hugsim_hinge/tables.md))

| preset | P2 | hinge | WA-JEPA | hinge - P2 | hinge - WA-JEPA |
|:--|--:|--:|--:|:--|:--|
| exam | 0.396 | 0.400 | 0.451 | +0.004 [-0.026, +0.034] | -0.051 [-0.130, +0.027] |
| spec | 0.393 | 0.395 | 0.451 | +0.002 [-0.006, +0.012] | -0.055 [-0.132, +0.022] |

Straight vs turning routes (41 / 23 scenarios; turning = unwrapped route yaw range >= 30 deg, as in the joint diagnosis; hinge - P2, then hinge - WA-JEPA):
exam straight -0.01 [-0.04, +0.02] / -0.01 [-0.10, +0.07], turning +0.02 [-0.03, +0.08] / -0.12 [-0.27, +0.03]; spec straight +0.00 [-0.01, +0.02] / +0.01 [-0.09, +0.10],
turning -0.00 [-0.01, +0.00] / -0.16 [-0.30, -0.04] ([hinge_strata.md](hinge_strata.md), 3-decimal values in `hinge_strata_hugsim.csv` are rounded to 2 there). HUGSIM
outcome counts (exam / spec): spins 9 / 2, stuck 1 / 0, completes 18.5 / 21 (P2: 9.5 / 1.5, 0.5 / 0, 17.5 / 19.5); fg collisions 28.5 / 27 (P2 28 / 29). No change in closed loop.

## Reading

- DAC lever confirmed in sign and in location (turns), smaller than predicted: -0.50 pp, not -0.9. Score does not drop, it rises +0.46 (DAC term only).
- It does not carry to HUGSIM (HD, DAC and collision counts unchanged within noise); the P2 - WA-JEPA closed-loop deficit on turning routes (-0.14 exam, -0.16 spec) is untouched.
  A navtest DAC failure is mostly a single-step graze (decision 147), the closed loop's turning failures are different events.
- The remaining 2.0 pp DAC and 3.0 EPDMS to WA-JEPA are concentrated on turns > 20 deg, as decision 147 located them in the vision features; a plan-side hinge does not reach them.

## Deviations and caveats

- lambda = 10 only (small read passed). The hinge weight is not the probe's relative weight: pp_train's imitation loss is sigma-normalised Huber, the probe's was metre Huber.
- 2 seeds; the seed spread of the effect is small (-0.56 / -0.44 pp, +0.46 / +0.47). HUGSIM is one run per scenario (rerun spread ~0.2 HD per scenario).
- The hinge sees the 8 raw plan poses, not the LQR replay (28% of P2's failures leave the road only in the replay, decision 147).
- Incident, not a result issue: on the first HUGSIM pass six closed-loop scenarios per job (pandaset scenes started together at the tail) failed at setup with
  `ConnectionRefusedError` on the parity bias server's unix socket (server process alive, `listen(64)`, socket refusing connections), and each such attempt hung until the 5 400 s
  timeout before the runner's retry. I cancelled both pool jobs, which restarted the servers; the chain resumed and ran only the 8 missing scenarios, all completed (HD from the retries
  is in the tables). Cause of the refusal not identified; it did not occur on the restarted servers. The P2 / WA numbers are untouched.
