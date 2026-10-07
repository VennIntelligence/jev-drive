# The stop gate does not carry over to P2H10: it costs navtest 1.50 EPDMS and 8.4 HD points on HUGSIM, because the navtrain-trained adapter bias is what launches the car from rest

Written 2026-10-08. Pre-registration (before any arm was scored): [../plans/2026-10-08-stop-gate-xboard-prereg.md](../plans/2026-10-08-stop-gate-xboard-prereg.md).
Code: `scripts/sg_xboard.py` (read-outs), switches `jevdrive/bench` `P2H10-F-s*@warp:sg|dn` (navtest) and `lib/parity_hugsim.py` opts `stop_gate` / `sub_bias` (HUGSIM),
`pp_train.PModel.gate / bias_sub`. All off by default; the default P2H10 path is unchanged. Tables: [stop_gate_xboard/](stop_gate_xboard/).

## Answer

Serving-side, on the stored P2H10-F-s0 / s1 weights (seed means, no training, threshold 0.5 m/s fixed):

| board | BASE | SG (stop gate) | DN (`biasdenav`) | label by the pre-registered rule |
|:--|--:|:--|:--|:--|
| navtest EPDMS (12 146 tokens, paired cluster bootstrap over 136 logs) | 88.67 | 87.17, **-1.50 [-1.85, -1.19]** (s0 -1.51, s1 -1.49) | 81.96, **-6.71 [-7.59, -5.90]** | SG hurts, DN hurts |
| HUGSIM 64, `spec_plan_smooth`, HD (paired over scenarios) | 0.431 | 0.347, **-0.084 [-0.141, -0.035]** (s0 -0.075, s1 -0.093) | 0.262, **-0.169 [-0.253, -0.091]** | SG hurts, DN hurts |

- **SG hurts on both boards, for the same reason**: P2H10's adapter bias is what makes it launch from rest, so removing it at standstill removes the launch. On WOD the same
  bias creeps a stationary car (decision 162 / 164); on navtrain-trained P2H10's own boards it is trained, and used, to move. One driver with one gate does not hold across
  boards; the WOD fix has to be trained in (`pp_train --stop-gate`, the wod-launch WLG arm), not served onto P2H10.
- **No oscillation at the threshold; the failure is a latch.** In every scenario-run where the gate fired (29 of 128: 14 + 15) it switched on once and never off again
  (max 1 flip, 0 oscillating scenarios by the pre-registered definition). The car decelerates, falls under 0.5 m/s (fed speed, model clock), loses the bias, and the plan
  at zero speed does not launch: 23 of the 29 runs end `stuck` at max_steps with v_end ~ 0.01 m/s, against 0 of the same scenario-runs in BASE (which end at 2-3 m/s).
- **DN is not a candidate**: subtracting the navtest-mean bias at all speeds costs -6.7 EPDMS (standstill tokens -21, moving -5.6) and breaks HUGSIM launches (launch stall
  34 per arm vs 1). Decision 162 asked what the WOD serving fix costs on navtest: this much, so it is not a deployable fix for P2H10.

## navtest (W frames, BASE = stored devkit CSVs of the same harness)

BASE reuse is exact: SG plans equal the stored plans on every token the gate does not touch (0 of 11 294 differ, both seeds; `stop_gate_xboard/check.md`) and BASE reproduces
decision 148's 88.67.

Tokens touched: the gate fires on 852 of 12 146 tokens (7.0%; fed vx < 0.5 m/s), 751 of them launch tokens (v0 < 2 m/s and logged 4 s distance > 5 m; the launch stratum has 1 902),
101 are stay tokens. DN changes the plan of every token.

| stratum | n | BASE | SG | SG - BASE [95% CI] | DN | DN - BASE [95% CI] |
|:--|--:|--:|--:|:--|--:|:--|
| all | 12 146 | 88.67 | 87.17 | -1.50 [-1.85, -1.19] | 81.96 | -6.71 [-7.59, -5.90] |
| v0 < 0.5 m/s | 848 | 92.80 | 73.37 | -19.42 [-21.74, -17.01] | 71.65 | -21.14 [-23.70, -18.55] |
| launch (v0 < 2, logged 4 s > 5 m) | 1 902 | 93.20 | 84.31 | -8.88 [-10.51, -7.42] | 78.42 | -14.78 [-17.01, -12.67] |
| moving (v0 >= 0.5) | 11 298 | 88.36 | 88.21 | -0.16 [-0.20, -0.12] | 82.73 | -5.63 [-6.53, -4.83] |
| turn > 20 deg | 3 154 | 80.28 | 79.08 | -1.20 [-1.59, -0.82] | 70.84 | -9.44 [-11.65, -7.40] |
| gate touched (fed vx < 0.5) | 852 | 92.55 | 73.31 | -19.23 [-21.53, -16.87] | 71.66 | -20.89 [-23.42, -18.37] |

Sub-scores, all tokens (BASE / SG / DN; SG - BASE): NC 98.58 / 98.12 / 97.62 (-0.45 [-0.66, -0.28]); DAC 96.17 / 96.30 / 93.91 (+0.13 [+0.06, +0.21]); EP 87.16 / 84.13 / 82.36
(-3.03 [-3.59, -2.51]); TTC 97.96 / 98.00 / 97.77 (+0.04 [0.00, +0.10]); EC 88.62 / 86.25 / 71.07 (-2.37 [-2.88, -1.92]); DDC, TLC, LK, HC within 0.1. SG loses progress (EP) and
extended comfort (EC), and a little NC; DAC does not suffer. Full table: `stop_gate_xboard/navtest_subscores.md`.

The `moving` stratum moves by -0.16 although its plans are bit-identical: EC compares a token with its neighbour in the same log, and a launch token's changed plan changes the
neighbour's score. It is a spillover, not a plan change.

## HUGSIM 64 closed loop (`spec_plan_smooth`, 2 seeds x 64 scenarios, one run per scenario)

BASE = the stored `P2H10-F-s*_spec_plan_smooth-rr1 / rr2` runs (mean of the two repeats; they differ by 0.0004-0.0010 HD, the sim is near deterministic). Check: the scenario-runs
where the gate never fired reproduce BASE exactly (HD change 0.000, none beyond +-0.1), so BASE reuse is valid and the CI below reflects scenario sampling only.

| arm | HD | HD - BASE [95% CI] | launch stall | stuck | spin | fg coll. | bg coll. | off route | complete |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| BASE | 0.431 | | 1 | 0 | 0 | 30 | 7 | 2 | 25 |
| SG | 0.347 | -0.084 [-0.141, -0.035] (3 wins / 10 losses / 51 ties, |d| < 0.02) | 1 | 11.5 | 0 | 29 | 7 | 2 | 14.5 |
| DN | 0.262 | -0.169 [-0.253, -0.091] (12 / 28 / 24) | 34 | 21.5 | 0 | 26 | 6.5 | 0 | 10 |

Counts per 64 scenarios, mean over the two seeds (per-seed values in `hugsim_classes.md`). Launch stall = peak speed over the first 40 steps < 1.6 m/s, stuck = ends at max_steps,
spin = heading error >= 60 deg (conventions of `hugsim_spin10.md`). The BASE stall / stuck floor is 1 / 0, so SG had nothing to repair; collisions do not move.

Gate behaviour (`hugsim_gate_summary.md`, `hugsim_gate_latch.md`, speed traces `stop_gate_xboard/figs/gate_speed_traces.png`): gate fired in 14 (s0) and 15 (s1) scenarios, median first gated step 30,
4 021 / 4 467 gated steps, **max 1 flip per scenario, 0 oscillating scenarios**. Mean HD change in the 29 gated scenario-runs -0.373. The traces show the pattern: the car pulls away
from the start at 1 m/s, slows to a stop (peak 3-7 m/s, then braking), crosses 0.5 m/s once, and never moves again, where BASE (same scenario, same first 25-30 steps) rolls on at 2-3 m/s.
Two of the six picked traces are scenarios where SG scores higher than BASE (+0.04 to +0.08, ending stuck instead of colliding): a stuck car also avoids collisions, so the HD gain there is not a launch gain.

## Verified vs not

Verified: BASE navtest = 88.67 (decision 148); SG plans identical to stored on untouched tokens; HUGSIM runs without a gated step identical to stored BASE; gate counts, flips and speeds from the per-step logs
(`zs_steps.jsonl` `parity.gated`, fed ego vx).
Not tested: the gate trained in (the WLG route: training with `--stop-gate` on navtrain, so the plan pathway learns to launch without the adapter at rest); other thresholds (fixed at 0.5 by rule);
another HUGSIM preset (`spec`, `exam`); navhard. A gate that does not latch (for example gating on a speed band instead of a step, or only when the plan itself is stationary) was not tried; it would be tuning.

## Caveats

- Serving-side on weights trained without the gate: the model never saw "bias off at rest", so this is a statement about P2H10 weights, not about the idea. A navtrain-trained gated arm may behave differently (untested).
- HUGSIM threshold acts on the speed fed to the adapter (model clock, x1.25), the same input the WOD / training rule reads; in m/s of the simulator the gate fires below 0.4.
- One preset (`spec_plan_smooth`); HUGSIM is one run per scenario; 64 scenarios from a fixed set. Bootstrap CIs are over scenarios (HUGSIM) and logs (navtest); both contrasts exclude 0 by wide margins.
- DN uses the navtest-mean bias computed in sample on the same tokens it is scored on (label-free; mean-bias rms 1.06 / 1.00).
- Cost: 4 navtest plan / scoring runs (plans: seconds on one card) and 4 HUGSIM 64 runs of about 0.6 card-hours each, about 2.4 card-hours in total on shared cards (slightly above the 2 card-hour budget because the HUGSIM runs queued behind other lanes and ran with 4 slots).

## Reused

Stored P2H10 weights, stored navtest devkit CSVs and HUGSIM `rr1 / rr2` runs, `jevdrive.bench` (navtest plans, devkit scoring, HUGSIM harness), `op_probe/joint/navtest_tokens.parquet` (v0, logged distance, heading change).
