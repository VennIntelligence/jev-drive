# Pilot: Cinque + WA-JEPA's inputs, 600 steps on 4 874 navtrain tokens: navtest EPDMS +4.8 over the fine-tune control, gap to WA-JEPA -5.1

Written 2026-10-06. Pre-registration: [../plans/2026-10-06-parity-prereg.md](../plans/2026-10-06-parity-prereg.md) (pilot gate; addendum 1 was
written before these scores were read). Tables: [navtest_pilot_arms.md](navtest_pilot_arms.md), [navtest_pilot_paired.md](navtest_pilot_paired.md)
(`scripts/pp_eval.py report`); HUGSIM: [hugsim_spin10.md](hugsim_spin10.md), harness and its equivalence tests [hugsim_harness.md](hugsim_harness.md);
frame-protocol Stage A: [stageA.md](stageA.md).

## Numbers

navtest, all 12 146 tokens, devkit navsim main @ 0a380a9 v2 EPDMS (metric cache `v2_navtest`, the harness that reproduced WA-JEPA's 91.71),
GIMM frame protocol (op_lb), single seed. Paired diffs: per-token EPDMS, bootstrap over 136 navtest logs.

| arm | EPDMS | diff vs P1 [95% CI] | diff vs WA-JEPA [95% CI] | EP | DAC | NC | TTC | EC | ADE vs log (m, 8 poses) | speed ratio (plan / log, 4 s) |
|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| shipped ONNX (op_lb `gimm@cinque`) | 81.11 | | | | | | | | | |
| P0 shipped (port) | 81.11 | -0.62 [-0.76, -0.48] | -10.60 [-11.57, -9.62] | 78.5 | 95.6 | 98.3 | 97.7 | 46.9 | 2.28 | 0.817 |
| P1 fine-tune, inputs zeroed | 81.73 | ref | -9.98 [-10.95, -8.99] | 79.1 | 95.8 | 98.3 | 97.8 | 49.8 | 2.12 | 0.835 |
| P2 + ego / pose history / command | **86.57** | **+4.84 [+4.11, +5.55]** | **-5.14 [-6.02, -4.27]** | 86.9 | 94.9 | 98.4 | 97.6 | 86.1 | 0.85 | 1.006 |
| P3 + side / rear cameras | 86.10 | +4.37 [+3.69, +5.04] | -5.61 [-6.45, -4.80] | 86.6 | 95.0 | 98.2 | 97.3 | 82.5 | 0.99 | 1.001 |
| P3, side cameras masked at test | 86.16 | +4.43 [+3.73, +5.11] | -5.55 [-6.40, -4.70] | 86.7 | 95.0 | 98.1 | 97.3 | 83.4 | 0.98 | 1.006 |
| WA-JEPA (released ckpt, same devkit / cache) | 91.71 | | | 87.9 | 98.2 | 99.4 | 98.9 | 88.0 | | |

HUGSIM, the 10 spinner scenarios (preset `exam`, tree `fixed`, as the stored Cinque row of wajepa_ref.md): HD P0 0.352, P1 0.330, P2 0.265,
P3 0.321 (WA-JEPA 0.593); spins P0 8, P1 8, P2 7, P3 8 of 10; launch stalls 0 in every arm. The launch-spin onset (step 11-18) is unchanged;
P2 / P3 launch harder and never stand (hugsim_spin10.md).

## Pre-registered pilot gate

| # | condition | result | verdict |
|:--|:--|:--|:--|
| 1 | P1/P2/P3-init plans bit-identical to P0 on navtest; P0 port vs shipped ONNX: 4 s position L2 mean < 0.05 m, EPDMS within 0.2 | max abs diff 0.0 for all three init arms; port vs ONNX 0.030 m, EPDMS 81.112 vs 81.106 | **holds** |
| 2 | P2 - P1 and P3 - P1 EPDMS >= 0 | +4.84 [+4.11, +5.55], +4.37 [+3.69, +5.04] | **holds** |
| 3 | HUGSIM spin10: P2, P3 spins <= P1 | 7, 8 vs 8 | **holds formally**; the spin itself is not changed |

## What this says

- The ego inputs carry the NAVSIM gain, and it is plan quality, not a speed artifact: P2's ADE to the log falls 2.12 -> 0.85 m and its plan
  speed matches the log (ratio 1.006 vs 0.835 for P1); EP +7.8 and two-frame extended comfort +36 (frame-to-frame consistent plans) carry
  most of the EPDMS gain, DAC -0.9 costs some of it.
- Side / rear cameras add nothing on navtest: P3 <= P2 (-0.47), and masking them at test changes P3 by +0.06 (decisions 135 / 136 agree).
- Fine-tuning alone on navtrain (P1 - P0) is +0.62: the inputs, not the fine-tune, close half of the 10.6-point gap; 5.1 remains at pilot scale.
- In closed loop the same inputs change launch speed, not the launch spin (decisions 92 / 111 / 118's ego-motion suspicion is not supported at
  this scale; see hugsim_spin10.md for the clock caveat).

## Caveats

- Pilot: 4 874 navtrain tokens (WA-JEPA trains on 103 k), 600 steps, one seed. Front frames use the GIMM protocol, which Stage A shows reads
  ego motion ~6 % slow for the shipped model.
- The P0 - P1 bootstrap row uses the P1 run's token set (identical). The HUGSIM per-scenario differences below ~0.2 HD are within rerun spread.
- Training always zeroes the oldest of the 9 context frames (the op_lb protocol); HUGSIM feeds all 9 with the bias. HUGSIM ego speed is fed on
  the dilated model clock (x1.25), see hugsim_harness.md.
