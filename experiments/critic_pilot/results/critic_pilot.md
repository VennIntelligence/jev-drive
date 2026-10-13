# CRITIC-PILOT read (2026-10-13)

Prereg: [2026-10-13-critic-pilot-prereg.md](../../route_follow/plans/2026-10-13-critic-pilot-prereg.md) (`51c8cb57`, pushed before training).
Numbers: [critic_pilot.json](critic_pilot.json). Box run: `$DATA_DIR/runs/critic_pilot/train/20261013-122*` (ckpt/, held preds), labels
`$DATA_DIR/runs/critic_pilot/labels/` (cp_labels.py; G0 pass on every pool, median |controller - anchor| = 0.00 m).
Test = RSG-L18-s0 held rollouts (`on1_L18RSGallheld`): 7 678 decision rows, 39 logs, any-zero within 2 s 88 positive rows (4 s: 170), lat2 on
6 500 rows. Decision level, log-clustered bootstrap (B = 10 000 for the lines, 2 000 otherwise). Spin rule never fired in any pool (0
positives), so spin is read only as wzpk R^2.

## Lines (C2 = trained on RB0 + RBSG, RSG unseen policy and unseen scenes)

| line | C2 | needed | verdict |
|:--|:--|:--|:--|
| L1 any-zero 2 s AUC | 0.846 [0.738, 0.935] | >= 0.90 | fail |
| L2 lat2 R^2 | 0.445 [0.365, 0.513] | >= 0.60 | fail |
| L3 CF flip (75 junction rows, 14 logs) | 0.41 [0.32, 0.51] | >= 0.70 | fail |

## All arms on RSG held

| arm | any2 | any4 | coll4 (32) | off2 (52) | off4 (98) | corr2 (24) | corr4 (48) | lat2 R^2 | prog4 R^2 | wzpk R^2 | flip |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C2 | 0.846 | 0.861 | 0.821 | 0.916 | 0.905 | 0.928 | 0.931 | 0.445 | 0.968 | 0.869 | 0.41 |
| C3 (RSG in training) | 0.833 | 0.846 | 0.782 | 0.918 | 0.887 | 0.933 | 0.928 | 0.523 | 0.967 | 0.872 | 0.40 |
| C2-blind (no tokens) | 0.842 | 0.832 | 0.779 | 0.864 | 0.821 | 0.953 | 0.934 | 0.531 | 0.969 | 0.860 | - |
| C2-noroute | 0.844 | 0.832 | 0.700 | 0.917 | 0.892 | 0.904 | 0.855 | 0.096 | 0.968 | 0.818 | 0.05 |
| C2-aug (route shift) | 0.863 | 0.851 | 0.731 | 0.918 | 0.893 | 0.911 | 0.849 | 0.425 | 0.966 | 0.860 | 0.49 |
| hand rule b_plan (served plan offset from route at 10 m) | 0.825 | 0.802 | 0.606 | 0.824 | 0.796 | **0.996** | 0.952 | - | - | - | - |
| hand rule b_log (ego offset from log path now) | 0.807 | 0.747 | 0.676 | 0.776 | 0.724 | 0.958 | 0.830 | - | - | - | - |
| hand rule b_now (ego offset from route now) | 0.648 | 0.588 | 0.437 | 0.635 | 0.610 | 0.780 | 0.652 | - | - | - | - |

collision within 2 s has 16 positive rows (not read). Blind arm on RBSG2 held reads (unseen policy, inputs from drive.jsonl, no lat2):
any2 0.807 [0.708, 0.897] (s0, 100 pos) / 0.835 [0.757, 0.916] (s1); corridor4 0.876 / 0.899, offroad4 0.811 / 0.864, collision4 0.66 / 0.70 (wide CIs).

## Counterfactual flip detail

279 CFdev junction rows; only 75 have the two routes >= 1 m apart at the 2 s arc (median gap 0.16 m: most junction splits lie beyond 2 s).
C2 prefers the plan on the given route in 61 % (real memory) / 73 % (cf memory) of rows, both at once 41 %; C2-aug 61 % / 84 %, 49 %;
C2-noroute 52 % / 51 %, 5 %. The route input is read (noroute -> lat2 R^2 0.10, flip 0.05), but not strongly enough to pass.

## Gradient sanity (g = -d risk / d plan xy at the 1.0-2.0 s poses; risk = sum of 4 s event logits + |lat2_hat|)

| arm | route rows (80 read): share toward route, median cos | collision rows (32): share away from actor, median cos | median |g| critic / S (S-active rows) | cos(critic, S) median, share > 0 |
|:--|:--|:--|:--|:--|
| C2 | 0.61, 0.44 | 0.81, 0.36 | 0.040 / 0.31 (route rows), 0.024 / 0.70 (collision) | 0.18, 0.67 / 0.39, 0.73 |
| C3 | 0.55, 0.31 | 0.53, 0.12 | 0.038 / 0.31, 0.029 / 0.70 | 0.01, 0.51 / 0.28, 0.68 |
| C2-aug | 0.61, 0.51 | 0.50, 0.01 | 0.012 / 0.31, 0.010 / 0.70 | 0.01, 0.51 / 0.47, 0.73 |

S = 10 x agent hinge (margin 0.5) + 10 x road hinge (margin 0.3), shape-only, at its training weight; critic gradient per unit of risk.
The critic gradient leans the right way on most rows (soft line 0.70 met only by C2 on collision rows, 26 of 32) and is 8-30x smaller than S's.

## Reading (executor, not a verdict change)

- The critic does not beat the hand rule it would replace: any-zero 2 s 0.846 vs 0.825 for |served plan offset from the route at 10 m|,
  and that rule is near perfect on corridor zeros (0.996 vs 0.928). The critic's real gain is offroad (0.92 vs 0.82) and collision at 4 s
  (0.82 vs 0.61, 32 positives, CI 0.59-0.96).
- Vision tokens add nothing to any2 or lat2 (blind 0.842 / 0.531 vs 0.846 / 0.445); they help offroad only (0.916 vs 0.864). Plan shape +
  ego + route memory carry most of what is predictable. prog4 (R^2 0.97) and wzpk (0.87) are easy and also need no tokens.
- Training on the test policy's own train-scene rollouts (C3) does not help: the limit is scenes and events (88 positive rows, 39 logs), not
  policy shift. Unseen-policy transfer itself looks fine (RBSG2 reads with the blind arm 0.81-0.84, same as RSG).
- Data, not architecture, is the binding constraint for a plan-conditioned critic: every row has exactly one executed plan, so the critic
  never sees two plans from one state; the flip test is the only place it is asked to rank plans and it does so weakly. A critic meant to
  give gradients would need branched rollouts (several plans per state, cf. research/carla-rewind-branching.md) or exact-relabel
  augmentation (C2-aug shows relabelling helps the ranking: 84 % on cf memory, flip 0.49) before it can replace a hinge.
