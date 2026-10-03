# One driver across boards: shipped Cinque vs it_dw3-s0, with and without the selector

2026-10-04. Reuses decisions 94 / 96 / 98 numbers; new cells are marked NEW. Cinque, single seed, HUGSIM one run per cell.

## Table

| driver | navtest PDMS (n 12 146) | navhard two-stage EPDMS (n 5 912) | HUGSIM 64: HD mean | spins (of 64) | non-spin HD paired delta (n 54) | WOD val RFS (n 479, OD2) | B2D drive arm DS (19 routes x 2 seeds, OD2) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 shipped Cinque | 84.18 | 33.33 (s1 71.70 / s2 46.90) | 0.278 | 10 | - | 8.004 | pending (lane od2-b2d; shipped = vmerge2's drive s2 / s3) |
| 2 shipped + selector | 84.09, -0.09 [-0.18, -0.01] | 34.31 (s1 71.76 / s2 48.24), +0.98 [+0.03, +1.97] | 0.294 | 3 | -0.005 [-0.015, +0.003] | 8.004, +0.000 (fires on 1 / 479) | not run |
| 3 it_dw3-s0 | 84.68 (NEW op_lb ONNX run), +0.50 [+0.30, +0.71]; decision 98 port readout +0.49 [+0.25, +0.73] | 35.40 (decision 98 port readout, no CI); NEW op_lb ONNX run 35.17, +1.84 [-0.35, +3.98] | NEW 0.299 | NEW 13 | NEW +0.036 [-0.013, +0.086] | 7.979, -0.026 [-0.118, +0.062] | pending (od2it) |
| 4 it_dw3-s0 + selector | NEW 84.70, +0.52 [+0.31, +0.73] vs shipped (+0.02 [-0.06, +0.09] vs it_dw3) | NEW 35.76 (s1 71.72 / s2 49.41), +2.43 [+0.17, +4.58] vs shipped (+0.58 [+0.13, +1.12] vs it_dw3) | NEW 0.345 | NEW 4 | NEW +0.056 [+0.001, +0.115] | 7.979, -0.026 [-0.118, +0.062] (fires on 0 / 479) | pending (od2sel) |

WOD: official RFS port, paired bootstrap over segments, training-port path (serving ONNX agrees to 0.001); B2D: vlm_arb `drive` arm, paired
over (route, seed), route-cluster bootstrap; details and the per-board trick rows in the OD2 section below. Deltas are against the shipped row on the same board, 95% CI: navtest per-token paired bootstrap; navhard paired bootstrap over the 225
scene-mapping groups; HUGSIM paired bootstrap over scenarios against the exam run `cinque-fixed` (the only full baseline). HUGSIM
non-spin = the 54 scenarios that do not spin in the exam run. Row 3 navhard CI comes only from the op_lb ONNX run (35.17); the
port readout (35.40) is the same weights through the training port, the 0.23 gap is the serving-path difference.

## Tricks per board (all labelled; the rows differ only in these)

| row | navtest | navhard | HUGSIM 64 |
|:--|:--|:--|:--|
| 2, 4 selector | `sel-rot0-r0.6`: second rollout with the 9 history frames rotated in place to the current heading (`--align rot0`), kept only if its 0-4 s lateral plan std < 0.6 x the native one; swaps 1.7% of tokens for it_dw3 (shipped: 1.9%) | same rule; swaps 5.1% of tokens for it_dw3 (shipped: 9.2%; 0.6 was chosen on navtrain, ~3% there) | `sel3`: speed < 3 m/s and yaw in the last 25 steps -> replay the 25 frames at the current heading, rule's plan kept if std < 0.6 x normal; taken on ~1% of steps |
| cost | 2x model compute | 2x | 2x only below 3 m/s |
| provenance | post hoc; ratio 0.6 fitted on navtrain with the shipped model; refitted for it_dw3 on navtrain by the same rule (OD2): 0.6 again | same | same ratio; HUGSIM arm added after the spin diagnosis (decision 96) |
| 3, 4 adaptation | it_dw3-s0 = layer-3 pairs (fake yaw / dropped history / offset recovery), distillation weight 3; navtest gain is mostly fine-tuning on navtrain (control `ctl` +1.43 > pair arms), not the pairs | same | same checkpoint, PR #57 controller, no rule |
| nothing else | no calibration, no N4 head, no x1.06 longitudinal scale in any row (x1.06 appears only as the labelled WOD trick rows of the OD2 section) | | |

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

## OD2: ratio refit, WOD, B2D (2026-10-04)

Pre-registration: [plans/2026-10-04-od2-prereg.md](../plans/2026-10-04-od2-prereg.md) (sections 1-3, each written before its run).

**Selector ratio refitted for it_dw3 on navtrain only** (decision 94's procedure: lb_navtrain 3 000 real tokens, official v1 subset
PDMS, x = the largest ratio whose navtrain delta >= -0.15): x = **0.6 again**, so rows 2 / 4 stand and no rescoring was needed.
it_dw3 native on navtrain 82.36 (shipped 82.12); rot0 everywhere 72.15.

| ratio | 1.0 | 0.9 | 0.8 | 0.7 | **0.6** | 0.5 |
|:--|:--|:--|:--|:--|:--|:--|
| navtrain delta vs it_dw3 native (pick rate) | -0.78 (22%) | -0.47 (10%) | -0.35 (6.5%) | -0.21 (4.0%) | **-0.07 [-0.27, +0.12] (2.3%)** | -0.00 (1.3%) |
| navtest delta, exact decomposition (pick rate), read after the choice | -0.33 (24%) | -0.16 (8.2%) | -0.10 (4.2%) | -0.05 (2.7%) | +0.02 [-0.06, +0.09] (1.7%) | +0.05 (1.1%) |

Raw: `one_driver/select_navtrain_it_dw3-s0.json`, `select_navtest_it_dw3-s0.json`; chain `scripts/h_od2_ratio.sh`.

**WOD** ([one_driver/wod.md](one_driver/wod.md)): rows 3 / 4 7.979 vs 8.004, -0.026 [-0.118, +0.062]; ADE +0.023 m. The selector does
not fire on WOD's real 10 Hz history (rot0 / native std ratio median 1.07), and rot0 everywhere costs -0.22 RFS, the decision-94
reading again. Per-board trick, labelled: longitudinal x1.06 gives shipped 8.120 (+0.116 [+0.023, +0.209]), it_dw3 (+ selector)
8.080; with x1.06 on both sides it_dw3 is -0.040 [-0.124, +0.042]. x1.06 was cross-fitted for shipped on these frames, not refitted for it_dw3.

**B2D** (vlm_arb `drive` arm, agent and controller unchanged; only the openpilot server differs): `od2it` serves the it_dw3 ONNX,
`od2sel` adds the selector inside `op_arb_server.py` (`OP_SEL=0.6`): below 3 m/s with >= 0.05 deg of yaw in the buffered history, a
second session sees the last 132 camera frames (6.6 s, Cinque's state queues) rotated in place to a reference heading (rebuilt when
the heading moved >= 0.5 deg and >= 4 steps passed, else stepped with the new frame), and its plan replaces the native one iff its
0-4 s lateral std < 0.6 x native. The unrotated control reproduces the native plan exactly over a CARLA route (0.00 m). Three bugs
were found in the smokes before the batch (0.1 deg rotation steps, black out-of-image rows, a spurious desire pulse at the replay
start, a 100-frame window shorter than the model's 132-step queues); prereg section 3 lists them. Results: pending, see
[one_driver/b2d/b2d.md](one_driver/b2d/b2d.md) when the lane finishes.

## Caveats

Single seed; HUGSIM same-day reruns reproduce 8 / 10 spin scenarios exactly, so spin counts of 3 vs 4 are within run noise. Row 4
navhard CI lower bound is +0.17 with the selector chosen post hoc on the same split. The ratio 0.6, first fitted with the shipped model,
was refitted for the adapted model on navtrain (OD2) and came out 0.6 again (pick rate on navhard fell from 9.2% to 5.1%).

## Files

- Rollouts: `scripts/op_lb.py run --onnx <it_dw3-s0.onnx> --tag Oit_dw3-s0 [--align rot0]` (new flags), selector
  `experiments/skill_pack/scripts/hist_align_select.py --tag`, report `hist_align_report.py --base-stem`.
- Chains: `scripts/h_onedriver_nav.sh` (navtest + navhard), `scripts/h_onedriver_hugsim.sh` (HUGSIM 64, no rule and sel3).
- Raw: `results/one_driver/{navtest,navhard}_{vs_shipped,sel_vs_native}.json`, `one_driver/tables/`, `one_driver/hugsim64/it_dw3-s0_{all64,sel3}/{summary.json,spins.csv}`.
- Shipped numbers: experiments/skill_pack/results/history-align/report.md, experiments/hugsim/results/derot/summary.md, results/it2/summary.md.
