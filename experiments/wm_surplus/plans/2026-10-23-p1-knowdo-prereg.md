# P1 knows-vs-does probe on WA-JEPA: does the planner input encode what the plan should do? (2026-10-23, before any probe job)

Decision row: PD5 (with P1 levels (a) / (b)). Same kind of probe as the Cinque readability reads (总表第 223 / 383 条): a small
linear probe on frozen features against the plan's own behaviour. Level (c) (causal attribution) stays out.

## Data

- Tokens: all navtest tokens (12 146). Folds: 5 folds of whole logs (fixed by sorted log name, `i % 5`); a probe is never fit
  on the logs it is scored on. The planner's numbers use the same tokens.
- Ground truth: the logged 4 s future (`runs/navsim_zs/index/navtest_future.npz`, t0 rear-axle frame) and the speed at t0 (bench
  strata cache), classed with `jevdrive.bench.tables.motion` (the bench's manoeuvre strata): stationary, launch, stop, left turn,
  right turn (|dyaw| > 20 deg), lane change (|dyaw| < 8 deg, |y(4 s)| > 2 m, path > 10 m), curve, straight.
- Plan: WA-JEPA's unprojected navtest plans (level (a) arm `off`, `PROBE-wajepa-ctx-off`), classed by the same function with
  the same t0 speed.

## Features (WA-JEPA)

The planner-input tensor of level (a) (`context_scene`, [4096, 512]), pooled without training: for each camera, the later
tubelet's 16 x 32 grid average-pooled to 2 x 4 cells, each cell projected on the top-64 navtrain PCA directions (level (a)'s
fit) -> 4 x 8 x 64 = 2048 values, plus the global token mean (512). Blind inputs = what the planner reads besides images:
command one-hot, ego velocity / acceleration, 4-pose history (20 values).

## Probes

Multinomial logistic regression (standardised inputs, L2, C chosen per outer fold by an inner 3-fold log split from
{0.01, 0.1, 1}), three input sets: `blind` (20), `feat` (2560), `feat+blind`. Regression side (ridge, same folds): GT dyaw and
GT lateral offset at 4 s.

## Read-out (fixed now)

Per GT manoeuvre class c (classes with >= 50 tokens): knows(c) = recall of the `feat+blind` probe, blind(c) = recall of the
`blind` probe, does(c) = share of tokens whose plan has class c. Paired differences with 95 % CIs, bootstrap over logs
(B 10 000, seed 0, `jevdrive.stats.paired`, groups = log).

- **sees but does not do**: knows - does >= 0.10 with CI lower bound > 0.
- **does not see (features add nothing)**: knows - blind < 0.05 with CI upper bound < 0.10, and knows < 0.6.
- **does what it sees**: |knows - does| < 0.10.
- otherwise unclear. Also reported: overall balanced accuracy of each probe and of the plan; R^2 of the dyaw / lateral offset
  regressions vs the plan's own dyaw / offset error (|plan - GT|).

## Caveats fixed in advance

- A linear probe on pooled features is a lower bound on what the features encode; "does not see" means "not linearly
  readable after this pooling", not "absent".
- The probe is fit on navtest labels (log folds); the planner was trained on navtrain. Probe recall is in-domain for navtest
  logs' statistics, so knows - does slightly favours the probe; the 0.10 margin is meant to absorb that.
- Manoeuvre labels come from the logged future, which is one of several acceptable behaviours (a "stop" token may also be
  fine to slow down only); the EPDMS sub-scores are the behavioural read-out, this table is a diagnostic.
