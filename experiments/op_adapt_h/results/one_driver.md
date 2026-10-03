# One driver across boards: shipped Cinque vs it_dw3-s0, with and without the selector

2026-10-04. Reuses decisions 94 / 96 / 98 numbers; new cells are marked NEW. Cinque, single seed, HUGSIM one run per cell.

## Table

| driver | navtest PDMS (n 12 146) | navhard two-stage EPDMS (n 5 912) | HUGSIM 64: HD mean | spins (of 64) | non-spin HD paired delta (n 54) |
|:--|:--|:--|:--|:--|:--|
| 1 shipped Cinque | 84.18 | 33.33 (s1 71.70 / s2 46.90) | 0.278 | 10 | - |
| 2 shipped + selector | 84.09, -0.09 [-0.18, -0.01] | 34.31 (s1 71.76 / s2 48.24), +0.98 [+0.03, +1.97] | 0.294 | 3 | -0.005 [-0.015, +0.003] |
| 3 it_dw3-s0 | 84.68 (NEW op_lb ONNX run), +0.50 [+0.30, +0.71]; decision 98 port readout +0.49 [+0.25, +0.73] | 35.40 (decision 98 port readout, no CI); NEW op_lb ONNX run 35.17, +1.84 [-0.35, +3.98] | NEW 0.299 | NEW 13 | NEW +0.036 [-0.013, +0.086] |
| 4 it_dw3-s0 + selector | NEW 84.70, +0.52 [+0.31, +0.73] vs shipped (+0.02 [-0.06, +0.09] vs it_dw3) | NEW 35.76 (s1 71.72 / s2 49.41), +2.43 [+0.17, +4.58] vs shipped (+0.58 [+0.13, +1.12] vs it_dw3) | NEW 0.345 | NEW 4 | NEW +0.056 [+0.001, +0.115] |

Deltas are against the shipped row on the same board, 95% CI: navtest per-token paired bootstrap; navhard paired bootstrap over the 225
scene-mapping groups; HUGSIM paired bootstrap over scenarios against the exam run `cinque-fixed` (the only full baseline). HUGSIM
non-spin = the 54 scenarios that do not spin in the exam run. Row 3 navhard CI comes only from the op_lb ONNX run (35.17); the
port readout (35.40) is the same weights through the training port, the 0.23 gap is the serving-path difference.

## Tricks per board (all labelled; the rows differ only in these)

| row | navtest | navhard | HUGSIM 64 |
|:--|:--|:--|:--|
| 2, 4 selector | `sel-rot0-r0.6`: second rollout with the 9 history frames rotated in place to the current heading (`--align rot0`), kept only if its 0-4 s lateral plan std < 0.6 x the native one; swaps 1.7% of tokens for it_dw3 (shipped: 1.9%) | same rule; swaps 5.1% of tokens for it_dw3 (shipped: 9.2%; 0.6 was chosen on navtrain, ~3% there) | `sel3`: speed < 3 m/s and yaw in the last 25 steps -> replay the 25 frames at the current heading, rule's plan kept if std < 0.6 x normal; taken on ~1% of steps |
| cost | 2x model compute | 2x | 2x only below 3 m/s |
| provenance | post hoc; ratio 0.6 fitted on navtrain with the shipped model and NOT refitted for it_dw3 | same | same ratio; HUGSIM arm added after the spin diagnosis (decision 96) |
| 3, 4 adaptation | it_dw3-s0 = layer-3 pairs (fake yaw / dropped history / offset recovery), distillation weight 3; navtest gain is mostly fine-tuning on navtrain (control `ctl` +1.43 > pair arms), not the pairs | same | same checkpoint, PR #57 controller, no rule |
| nothing else | no calibration, no N4 head, no x1.06 longitudinal scale in any row | | |

## Reading

- navtest: the adapted model carries its +0.5; the selector adds nothing on top (+0.02 [-0.06, +0.09]), and on the shipped model it costs -0.09.
- navhard: it_dw3 alone is not significant over shipped (CI crosses 0); adding the selector gets row 4 to +2.43 [+0.17, +4.58], and the
  selector's own increment over it_dw3 is +0.58 [+0.13, +1.12]. Larger than the shipped-only selector gain (+0.98) but a different base.
  The selector form was chosen after seeing navhard stage splits (decision 94), so this is not a clean confirmation.
- HUGSIM: it_dw3 alone does not reduce spins (13 vs 10; 5 new, 8 of the 10 original still spin). With sel3 spins drop to 4 (3 of the 10
  original; 1 new: scene-164701907483-hard-00), HD mean 0.345 against 0.294 for shipped + sel3, non-spin delta +0.056 [+0.001, +0.115]
  (shipped + sel3: -0.005). The spin fix is the selector; the HD gain on non-spin scenes is the adapted model, as in decision 98 for pilot.
- Row 4 is the best row on all three boards, and no board has it worse than shipped. It needs the selector on both open-loop and
  closed-loop and doubles compute where it fires.

## Caveats

Single seed; HUGSIM same-day reruns reproduce 8 / 10 spin scenarios exactly, so spin counts of 3 vs 4 are within run noise. Row 4
navhard CI lower bound is +0.17 with the selector chosen post hoc on the same split. The ratio 0.6 was not refitted for the adapted
model (the adapted plan std distribution may differ; pick rate on navhard fell from 9.2% to 5.1%).

## Files

- Rollouts: `scripts/op_lb.py run --onnx <it_dw3-s0.onnx> --tag Oit_dw3-s0 [--align rot0]` (new flags), selector
  `experiments/skill_pack/scripts/hist_align_select.py --tag`, report `hist_align_report.py --base-stem`.
- Chains: `scripts/h_onedriver_nav.sh` (navtest + navhard), `scripts/h_onedriver_hugsim.sh` (HUGSIM 64, no rule and sel3).
- Raw: `results/one_driver/{navtest,navhard}_{vs_shipped,sel_vs_native}.json`, `one_driver/tables/`, `one_driver/hugsim64/it_dw3-s0_{all64,sel3}/{summary.json,spins.csv}`.
- Shipped numbers: experiments/skill_pack/results/history-align/report.md, experiments/hugsim/results/derot/summary.md, results/it2/summary.md.
