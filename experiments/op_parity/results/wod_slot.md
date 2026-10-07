# The 8 real + 1 zero slot training protocol is not why P2H10 loses on WOD val: serving it that way costs P2H10 a further 0.21 RFS

Written 2026-10-08. Pre-registration: [plans/2026-10-08-wod-slot-prereg.md](../plans/2026-10-08-wod-slot-prereg.md) (committed before any `_s8` prediction was scored). Code:
`scripts/wod_slot.py` (`onnx` / `check` / `report`). Tables: [wod_slot/](wod_slot/) (`arms`, `contrasts`, `ade`, `ade_contrasts`, `plan_shift`, `check_*.json`).

## Answer

No. The navtrain arms were trained with the oldest of the 9 policy slots zeroed (after the intent bias is added); the WOD harness serves 9 real slots. Serving WOD val with
the oldest slot zeroed, to match the training protocol, makes P2H10 worse, not better: RFS 7.708 -> 7.499 (**-0.208 [-0.328, -0.085]**, both seeds -0.20 / -0.22), ADE@3s +0.21 m.
SH30 behaves the same (-0.182 [-0.309, -0.057]). Label by the pre-registered rule: **not** (carries needs a positive C1 of at least +0.15 with CI lower bound > 0). The
P2H10 gap to shipped grows from -0.296 (9 real) to -0.505 [-0.747, -0.278] against 9-real shipped, and to -0.419 [-0.652, -0.201] against shipped served the same way. The
protocol mismatch explains none of the WOD loss; the loss stays where decisions 162 / 163 put it (standstill, -0.70 [-1.16, -0.25] at 9 real).

Zeroing the oldest slot hurts every arm a little at serving time (shipped -0.086 [-0.170, -0.002], ADE@3s +0.24 m), so 8 + zero is simply a less informed input on WOD; the
navtrain arms lose about 2x as much as shipped (point estimates; the difference of differences was not tested). The pre-registered expectation that WP2 (trained with 9 real)
would lose from the zero slot is half met: RFS +0.028 [-0.060, +0.116] (no loss, n.s.), ADE@3s +0.118 m [+0.091, +0.148] (worse), and the standstill stratum even gains
+0.133 [+0.037, +0.235].

## Table (WOD val, 479 rater frames cluster-mean RFS, paired bootstrap over sequences B 4000; ADE over 1 437 frames; arm = per-frame mean of 2 seeds)

| arm | 9 real RFS | 8 + zero RFS | d RFS (8+zero - 9 real) [95% CI] | d ADE@3s (m) [CI] | d ADE@5s (m) [CI] | plan shift (m, mean / at 5 s) |
|:--|--:|--:|:--|:--|:--|:--|
| shipped (ONNX, zero bias) | 8.005 | 7.919 | -0.086 [-0.170, -0.002] | +0.237 [+0.209, +0.266] | +0.343 [+0.290, +0.398] | 0.61 / 1.30 |
| **P2H10** (seeds 7.718 / 7.699 -> 7.517 / 7.482) | 7.708 | 7.499 | **-0.208 [-0.328, -0.085]** | +0.205 [+0.177, +0.237] | +0.345 [+0.297, +0.397] | 0.53 / 1.15 |
| **SH30** (7.741 / 7.727 -> 7.558 / 7.544) | 7.734 | 7.551 | **-0.182 [-0.309, -0.057]** | +0.173 [+0.145, +0.204] | +0.292 [+0.243, +0.342] | 0.52 / 1.16 |
| WP2 (trained with 9 real; 8.114 / 8.109 -> 8.133 / 8.146) | 8.111 | 8.139 | +0.028 [-0.060, +0.116] | +0.118 [+0.091, +0.148] | +0.154 [+0.110, +0.202] | 0.54 / 1.19 |

| contrast (RFS) | all | standstill (v0 < 0.5, n 120) | moving (n 359) | turn intent (n 52) |
|:--|:--|:--|:--|:--|
| P2H10 8+zero - P2H10 9 real (C1) | -0.208 [-0.328, -0.085] | -0.369 [-0.579, -0.161] | -0.153 [-0.313, +0.000] | -0.024 [-0.261, +0.184] |
| SH30 8+zero - SH30 9 real (C4) | -0.182 [-0.309, -0.057] | -0.350 [-0.573, -0.149] | -0.119 [-0.290, +0.041] | +0.009 [-0.251, +0.247] |
| shipped 8+zero - shipped 9 real (C5) | -0.086 [-0.170, -0.002] | -0.014 [-0.145, +0.123] | -0.129 [-0.238, -0.022] | -0.177 [-0.470, +0.346] |
| WP2 8+zero - WP2 9 real (C6) | +0.028 [-0.060, +0.116] | +0.133 [+0.037, +0.235] | -0.007 [-0.139, +0.120] | +0.050 [-0.222, +0.312] |
| P2H10 8+zero - shipped 9 real (C2) | -0.505 [-0.747, -0.278] | -1.065 [-1.526, -0.611] | -0.344 [-0.621, -0.080] | -0.569 [-1.197, +0.229] |
| P2H10 8+zero - shipped 8+zero (C3) | -0.419 [-0.652, -0.201] | -1.051 [-1.495, -0.620] | -0.215 [-0.462, +0.024] | -0.392 [-1.095, +0.304] |
| SH30 8+zero - shipped 9 real | -0.453 [-0.691, -0.227] | -1.035 [-1.490, -0.587] | -0.288 [-0.561, -0.028] | -0.336 [-0.935, +0.435] |
| SH30 8+zero - shipped 8+zero | -0.367 [-0.598, -0.145] | -1.021 [-1.457, -0.588] | -0.159 [-0.415, +0.083] | -0.159 [-0.877, +0.533] |
| reference, stored: P2H10 9 real - shipped 9 real | -0.296 [-0.501, -0.093] | -0.696 [-1.155, -0.245] | -0.190 [-0.391, +0.020] | -0.545 [-1.123, +0.229] |

ADE contrasts of P2H10 8+zero: vs shipped 9 real +0.489 [+0.414, +0.569] (3 s) / +0.897 [+0.761, +1.039] (5 s); vs shipped 8+zero +0.252 [+0.187, +0.320] / +0.554
[+0.436, +0.674]. Absolute ADE@3s / @5s (m): shipped 1.041 / 2.117 (8+zero 1.277 / 2.460), P2H10 1.324 / 2.668 (1.530 / 3.013), SH30 1.360 / 2.715 (1.532 / 3.007), WP2 0.574 / 1.457
(0.692 / 1.610). All 8+zero strata (night, day, SS, SL, launch) are in `wod_slot/contrasts.csv`.

Where the loss of the matched protocol sits: the standstill stratum (P2H10 -0.369) and moving (-0.153, CI upper bound +0.000); the turn-intent stratum is unchanged (-0.024, n 52,
CI wide). The standstill drop is the navtrain arms' standstill bias getting worse, not better, when a slot is removed; shipped's own standstill is unaffected (-0.014).

## Verified / not verified

Verified:
- G0 scoring path: the stored 9-real predictions rescored by this report give shipped 8.005, P2H10 7.708 (7.718 / 7.699), SH30 7.734, WP2 8.111, as stored (tolerance 0.002).
- The mask is applied where the training protocol applies it: inside the served ONNX, on the policy's 9-slot gather (`cat_3` queue index 24 = oldest of the last 9), after the intent bias
  was added to the past slots, recurrent queue untouched. Same harness, 10 s warm-up, same bias files (`bias-<tag>.npz`), same 479 + extra frames, 1 437 targets per arm.
- G2 the mask moves the plan: shipped 8+zero vs 9 real differs by 0.61 m mean over the 20 waypoints on the 479 rater frames (1.01 m on the 8-target check, `check_shipped.json`),
  the same order as mixed-domain's 1.09 m (33-point plan on cache rows, a different frame set and horizon). The other arms shift 0.52 - 0.54 m.

Not verified:
- Whether the sign would flip for a model trained on 8 + zero slots of WOD frames themselves (the mixed-domain MX pilot is the only such arm, trained, not slot-ablated at serving).
- The difference of differences (navtrain arms lose more than shipped) has no CI; stated from point estimates only.
- The 9-real references are the stored runs; no 9-real arm was re-served (the ones-mask build was checked instead, below).

## Deviations from the pre-registration

- G1 is not bit-identical. The ones-mask ONNX vs the unmasked ONNX differs by 0.021 m mean (max 0.18 m, 8 targets, `check_keep.json`) because the two TensorRT engines are built
  separately; this is the engine noise already measured for ONNX-vs-stock shipped (0.022 m, `wod/check.json`), 50x below the effect of the mask (1.01 m). The path is judged
  correct on that level, not bitwise.
- The shipped 8+zero arm is the zero-bias ONNX (`pp-shipped`), the 9-real shipped reference is the stock model (`op_cinque`), so C5 carries that 0.02 m engine noise as well.
- Serving jobs were SIGKILLed (rc 137) several times mid-run while 5 - 6 harness jobs shared the box (no OOM recorded in the cgroup; cause not identified); each was resubmitted
  and resumed from its written predictions (per-target independent), so every arm has all 1 437 targets from the same ONNX. No target was served by a different model.
- Prereg labelled the carries / part thresholds on a positive C1; C1 came out negative so the label is "not" without further rule application. The pre-registered WP2 expectation
  (C6 < 0) did not hold on RFS (see Answer).
