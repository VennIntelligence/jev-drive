# History alignment as a navhard input rule (2026-10-03 night)

Plan and pre-registration (three supplements, each written before its numbers): [../../plans/2026-10-04-history-align-plan.md](../../plans/2026-10-04-history-align-plan.md).
Gate fixed in advance: navhard combined EPDMS delta > 0 with CI lower bound > 0, and navtest delta >= -0.30.
Model: shipped Cinque (op_lb). navhard = official two-stage EPDMS (5,912 tokens), paired bootstrap over 225 scene-mapping groups;
navtest = official v1 PDMS (12,146 tokens), per-token paired bootstrap; navtrain = lb_navtrain, 3,000 real tokens (calibration only).

## Rules

All rules change only the 9 history frames and use only the four 2 Hz ego poses the leaderboard provides.
- `rot0`: each history frame stays in place, rotated to the current heading.
- `straight`: the t0 frame warped back along a straight constant-arc-length path (the decision 88 intervention).
- `straight_keys`: the real history frames moved onto the same straight path.
- `sel`: run shipped and rule rollouts, keep per token the one with the smaller 0-4 s lateral plan std.
- `sel-rot0-r0.6`: switch to the rule rollout only if its std < 0.6 x the shipped std (0.6 chosen on navtrain; switches ~3% of tokens).

## Scores

| arm | source | navhard (s1 / s2) | delta [95% CI] | navtest | delta [95% CI] |
|:--|:--|:--|:--|:--|:--|
| shipped | - | 33.33 (71.70 / 46.90) | | 84.18 | |
| rot0 | pre-registered | 32.67 (61.69 / 52.69) | -0.66 [-3.80, +2.37] | 76.14 | -8.04 [-8.59, -7.48] |
| straight | pre-registered | 31.99 (61.70 / 50.75) | -1.34 [-5.22, +2.40] | 74.10 | -10.08 [-10.71, -9.45] |
| straight_keys | supplement 1 | 33.00 (63.34 / 51.90) | -0.33 [-3.27, +2.53] | not run | |
| sel-rot0 | supplement 2, post hoc | 35.59 (71.59 / 50.08) | +2.26 [+0.33, +4.13] | 83.53 | -0.65 [-0.85, -0.46] |
| sel-straight | supplement 2, post hoc | 37.08 (71.13 / 52.37) | +3.75 [+1.48, +6.02] | 80.80 | -3.38 [-3.71, -3.05] |
| **sel-rot0-r0.6** | supplement 3, post hoc, ratio from navtrain | **34.31** (71.76 / 48.24) | **+0.98 [+0.03, +1.97]** | **84.09** | **-0.09 [-0.18, -0.01]** |

navtrain calibration: shipped 82.12, rot0 71.23, straight 72.49. sel-rot0 delta for ratio 1.0 ... 0.5: -2.52 / -1.82 / -1.05 / -0.42 / -0.06 / -0.05; straight never passes (best -0.72).

## Wrong-direction class

| arm | stage-2 reversal | stage-1 reversal | of the 312 shipped reversals still reversed | of those, DAC pass |
|:--|--:|--:|--:|--:|
| shipped | 5.7% | 1.3% | 100% | 46.8% |
| rot0 | 6.1% | 15.3% | 22.8% | 76.0% |
| straight | 2.7% | 6.2% | 15.4% | 70.5% |
| sel-rot0 | 3.1% | - | 46.2% | 68.6% |
| sel-straight | 2.7% | - | 35.6% | 72.1% |
| sel-rot0-r0.6 | 4.2% | - | 71.5% | 59.6% |

Of the 4,196 tokens the shipped model passes, plain rules keep 91-94%, selectors 98.5-99.9%.

## Reading

- History rotation is a useful signal in real scenes: removing it everywhere gains 4-6 points on stage 2 and loses ~10 on stage 1, 8-10 on navtest and ~10 on navtrain.
- No consistency gate exists: stage-2 synthetic history images agree with their own pose yaw (phase-correlation measured / predicted slope 0.95; stage 1 0.97, navtest 0.99; 300 tokens each).
- The model's own plan std separates the right rollout (AUC 0.86 on rot0 reversal flips); this is the selector's basis. The selector doubles model cost.
- Checks: straight reproduces decision 88 (15.4% vs 15%); rot0 rotation removal measured by phase correlation (residual 0-6 px); navtest decomposition matches the official result.
- Limits: the selector form was chosen after seeing the stage split; 7 arms scored on navhard, no multiplicity correction (CI lower bound +0.03); N4 not covered (needs a refit).
